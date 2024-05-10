
import os
import numpy as np
import bigfile
from nbodykit.lab import BigFileCatalog
from nbodykit.cosmology import Planck15
from astropy.cosmology import z_at_value
from dask_jobqueue import PBSCluster
from dask.distributed import Client
import dask.array as da
import gc
import json
import argparse

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def make_massmap(cat, npix):
    pid = cat['ID']    
    ipix = pid % npix    
    ipix = ipix.compute()
    ipix = ipix.astype('i4')

    mass = cat['Mass'].compute()
    mass_map_for_slice = np.bincount(ipix, weights=mass, minlength=npix)   # the total mass
    
    return mass_map_for_slice

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
    
def wlen_integrand(Om, dl, ds):
    zl = z_at_value(Planck15.comoving_distance, dl)
    ds = np.atleast_1d(ds) # promote to 1d, sum will get rid of it
    return 1.5 * Om * inv_sigma(ds, dl, zl)

def wlen_int(Om, dlmin, dlmax, ds, dbin=100):
    dl = np.linspace(dlmin, dlmax, dbin)
    ddl = dl[1] - dl[0]
    w_int = wlen_integrand(Om, dl, ds)
    w_lensing = np.sum(w_int, axis=0) * ddl
    return w_lensing

def wlen(Om, dlmin, dlmax, ds, dbin=100):
    dl = (dlmin + dlmax) / 2
    ddl = dlmax - dlmin
    w_int = wlen_integrand(Om, dl, ds)
    w_lensing = np.sum(w_int, axis=0) * ddl
    return w_lensing

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
    rhobar = cat.attrs['MassTable'][1] * nbar
    Om = cat.attrs['OmegaM'][0]

    kappa = np.zeros([len(zs_list), npix], dtype="f8")
    kappa_int = np.zeros([len(zs_list), npix], dtype="f8")
    if cat.comm.rank == 0:
        cat.logger.info("pre-allocation done")
        cat.logger.info("memory usage by kappa: %f GB" % ((kappa.nbytes+kappa_int.nbytes) / 1024 ** 3))

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

        dmin = Planck15.comoving_distance(zmin)
        dmax = Planck15.comoving_distance(zmax)
        volume_diff = (dmax ** 3 - dmin ** 3) * 4 * np.pi / (3 * npix)
        weights_list = np.array([wlen(Om, dmin, dmax, ds) for ds in ds_list])
        weights_int_list = np.array([wlen_int(Om, dmin, dmax, ds) for ds in ds_list])
        
        Mmap = make_massmap(sliced, npix, client=None, vel_flag=False)
        if cat.comm.rank == 0:
            cat.logger.info("mass map computed")

        kappa += weights_list.reshape(-1, 1) *  (Mmap / rhobar / volume_diff - 1)
        kappa_int += weights_int_list.reshape(-1, 1) *  (Mmap / rhobar / volume_diff - 1)

    if cat.comm.rank == 0:
        cat.logger.info("kappa computation done")
        # use bigfile because it allows concurrent write to different datasets.
        cat.logger.info("writing to %s", config['destination'])

    save_path = os.path.join(datadir, config['destination'])
    os.makedirs(save_path, exist_ok=True)
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
                ds2 = ff.create_from_array("kappa_int", kappa_int[i], Nfile=1)

                for d in ds1, ds2:
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