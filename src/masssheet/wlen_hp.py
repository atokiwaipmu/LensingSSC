
import os
import numpy as np
import bigfile
from nbodykit.lab import BigFileCatalog
from nbodykit.cosmology import Planck15
from dask_jobqueue import PBSCluster
from dask.distributed import Client
import dask.array as da
import gc
import json
import argparse

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def setup_dask_cluster():
    # Setup Dask cluster using PBSCluster from Dask JobQueue
    cluster = PBSCluster(
        cores=52,  # adjust to your cluster's configuration
        memory='120GB',  # adjust to your cluster's configuration
        queue='small',  # the name of the queue
        walltime='02:00:00',
        local_directory='$TMPDIR'
    )
    cluster.scale(4)  # Adjust the scaling as needed
    client = Client(cluster)
    return client, cluster

def make_massmap(cat, npix, client=None, vel_flag=True):
    pid = cat['ID']    
    ipix = pid % npix    
    ipix = ipix.compute()
    #ipix = client.compute(ipix)
    #ipix = ipix.result()
    ipix = ipix.astype('i4')
    counts_map_for_slice = np.bincount(ipix, minlength=npix)  # the number of particles in each pixel

    mass = cat['Mass'].compute()
    #mass = client.compute(cat['Mass'])
    #mass = mass.result()
    mass_map_for_slice = np.bincount(ipix, weights=mass, minlength=npix)   # the total mass
    
    if vel_flag:
        Rmom = cat['Rmom'].compute()
        #Rmom = client.compute(cat['Rmom'])
        #Rmom = Rmom.result()
        rmom_map_for_slice = np.bincount(ipix, weights=Rmom, minlength=npix)  # the total momentum

        velocity_map_for_slice = rmom_map_for_slice / (mass_map_for_slice + np.ones_like(mass_map_for_slice) * (mass_map_for_slice == 0))  # replace zeros with ones for dividing velocity map
        
        return mass_map_for_slice, velocity_map_for_slice, counts_map_for_slice
    else:
        return mass_map_for_slice, counts_map_for_slice

def read_slice(cat, islice):
    aemitIndex_offset = cat.attrs['aemitIndex.offset']
    start = aemitIndex_offset[1 + islice]
    end = aemitIndex_offset[2 + islice]
    if cat.comm.rank == 0:
        cat.logger.info("Range of index is %d to %d" %(( start + 1, end + 1)))
    cat =  cat.query_range(start, end)
    return cat

def inv_sigma(ds, dl, zl):
    ddls = 1 - np.multiply.outer(1 / ds, dl)
    ddls = ddls.clip(0)
    w = (100. / 3e5) ** 2 * (1 + zl)* dl
    inv_sigma_c = (ddls * w)
    return inv_sigma_c
    
def wlen(Om, dl, zl, ds, Nzs=1):
    """
        Parameters
        ----------
        dl, zl: distance and redshift of lensing objects
        
        ds: distance source plane bins. if a single scalar, do a delta function bin.
        
        Nzs : number of objects in each ds bin. len(ds) - 1 items
        
    """
    ds = np.atleast_1d(ds) # promote to 1d, sum will get rid of it
    integrand = 1.5 * Om * Nzs * inv_sigma(ds, dl, zl)
    Ntot = np.sum(Nzs)
    w_lensing = np.sum(integrand, axis=0) / Ntot
    
    return w_lensing
     
def make_kappabar(Om, ds_list, zmin, zmax):
    dmin = Planck15.comoving_distance(zmin)
    dmax = Planck15.comoving_distance(zmax)
    kappabar_list = []
    for ds in ds_list:
        LensKernel_min = wlen(Om, dmin, zmin, ds)
        LensKernel_max = wlen(Om, dmax, zmax, ds)
        kappabar = (LensKernel_max + LensKernel_min) / (dmax - dmin) / 2
        kappabar_list.append(kappabar)
    return np.array(kappabar_list)

def main(config_file):
    #client, cluster = setup_dask_cluster()  # Set up the Dask client and cluster
    with open(config_file, 'r') as file:
        config = json.load(file)
    zs_list = config['zs']
    zlmin = config['zlmin']
    zlmax = config['zlmax']

    # Compute comoving distance using Planck15 cosmology
    ds_list = Planck15.comoving_distance(zs_list) 

    datadir = config['datadir']
    path = os.path.join(datadir, config['source'])

    # Load the catalog
    cat = BigFileCatalog(path, dataset=config['dataset'])

    npix = cat.attrs['healpix.npix'][0]
    nside = cat.attrs['healpix.nside'][0]
    nbar = (cat.attrs['NC'] ** 3  / cat.attrs['BoxSize'] ** 3 * cat.attrs['ParticleFraction'])[0]
    Om = cat.attrs['OmegaM'][0]

    localsize = npix * (cat.comm.rank + 1) // cat.comm.size - npix * (cat.comm.rank) // cat.comm.size

    kappa = np.zeros([len(zs_list), npix], dtype="f8")
    Nm = np.zeros([len(zs_list), npix], dtype="i4")
    kappabar = np.zeros([len(zs_list)], dtype="f8")
    if cat.comm.rank == 0:
        cat.logger.info("pre-allocation done")
        cat.logger.info("memory usage by kappa, Nm: %f GB" % ((kappa.nbytes + Nm.nbytes) / 1024 ** 3))

    a = cat.attrs['aemitIndex.edges']
    for islice, (amin, amax) in enumerate(zip(a[:-1], a[1:])):
        if amin == 0 or amax == 0: continue
        zmin, zmax = 1 / amax - 1, 1 / amin - 1
        if (zmin > zlmax) or (zmax < zlmin): continue
        if cat.comm.rank == 0:
            cat.logger.info("nbar = %g, z = %2.2g - %2.2g" % (nbar, zmin, zmax))
        sliced = read_slice(cat, islice)
        if sliced.csize == 0: continue
        if cat.comm.rank == 0:
            cat.logger.info("read %d particles" % sliced.csize)

        zl = (zmin + zmax) / 2
        dl = Planck15.comoving_distance(zl)
        area = (4 * np.pi / npix) * dl**2
        Lenskernel_list = []
        for ds in ds_list:  
            Lenskernel_list.append(wlen(Om, dl, zl, ds))
        Lenskernel_list = np.array(Lenskernel_list)
        weights_list = (Lenskernel_list / (area * nbar))
        
        Mmap, Nmap = make_massmap(sliced, npix, client=None, vel_flag=False)
        if cat.comm.rank == 0:
            cat.logger.info("mass map computed")

        kappa1bar = make_kappabar(Om, ds_list, zmin, zmax)

        #if cat.comm.rank == 0:
        #print("length of weights_list: ", len(weights_list))
        #print("shape of Mmap: ", Mmap.shape)
        kappa += weights_list.reshape(-1, 1) *  Mmap
        Nm += Nmap
        kappabar += kappa1bar

    if cat.comm.rank == 0:
        cat.logger.info("kappa computation done")
        # use bigfile because it allows concurrent write to different datasets.
        cat.logger.info("writing to %s", config['destination'])

    save_path = os.path.join(datadir, config['destination'])
    for i, (zs, ds) in enumerate(zip(zs_list, ds_list)):
        std = np.std(cat.comm.allgather(len(kappa[i])))
        mean = np.mean(cat.comm.allgather(len(kappa[i])))
        if cat.comm.rank == 0:
            cat.logger.info("started gathering source plane %s, size-var = %g, size-bar = %g" % (zs, std, mean))

        if cat.comm.rank == 0:
            cat.logger.info("done gathering source plane %s" % zs)

        if cat.comm.rank == 0:
            fname = save_path + "/WL-%02.2f-N%04d" % (zs, nside)
            cat.logger.info("started writing source plane %s" % zs)

            with bigfile.File(fname, create=True) as ff:

                ds1 = ff.create_from_array("kappa", kappa[i], Nfile=1)
                ds2 = ff.create_from_array("Nm", Nm[i], Nfile=1)

                for d in ds1, ds2:
                    d.attrs['kappabar'] = kappabar[i]
                    d.attrs['nside'] = nside
                    d.attrs['zlmin'] = zlmin
                    d.attrs['zlmax'] = zlmax
                    d.attrs['zs'] = zs
                    d.attrs['ds'] = ds
                    d.attrs['nbar'] = nbar

        cat.comm.barrier()
        if cat.comm.rank == 0:
            # use bigfile because it allows concurrent write to different datasets.
            cat.logger.info("source plane at %g written. " % zs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    # choices=["tiled", "bigbox"] #['config_tiled_hp.json', 'config_bigbox_hp.json']
    parser.add_argument('config', type=str, choices=['tiled', 'bigbox'], help='Configuration file')
    args = parser.parse_args()
    config_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_%s_hp.json' % args.config)
    main(config_file)