
import argparse
import os
import warnings

import bigfile
import numpy as np
from astropy.cosmology import z_at_value
from nbodykit.cosmology import Planck15
from nbodykit.lab import BigFileCatalog

from .ConfigData import ConfigData

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
    w_lensing = np.sum(w_int, axis=1) * ddl
    return w_lensing

def wlen(Om, dlmin, dlmax, ds):
    dl = (dlmin + dlmax) / 2
    ddl = dlmax - dlmin
    w_int = wlen_integrand(Om, dl, ds)
    w_lensing = w_int * ddl
    return w_lensing

def compute_weights_and_delta(config, sliced, zmin, zmax, ds_list):
    dmin = Planck15.comoving_distance(zmin)
    dmax = Planck15.comoving_distance(zmax)
    volume_diff = (dmax ** 3 - dmin ** 3) * 4 * np.pi / (3 * config.npix)
    
    weights_list = np.array([wlen(config.Om, dmin, dmax, ds) for ds in ds_list])
    weights_int_list = np.array([wlen_int(config.Om, dmin, dmax, ds) for ds in ds_list])
    
    delta = make_massmap(sliced, config.npix) / config.rhobar / volume_diff - 1
    return weights_list, weights_int_list, delta

def process_slices(config, ds_list, logger):
    kappa, kappa_int = 0, 0  # Initialize kappa values
    for islice, (amin, amax) in enumerate(zip(config.aemitIndex[:-1], config.aemitIndex[1:])):
        if amin == 0 or amax == 0:
            continue
        zmin, zmax = 1 / amax - 1, 1 / amin - 1
        if zmin > config.zlmax or zmax < config.zlmin:
            continue
        logger.info(f"nbar = {config.nbar:g}, z = {zmin:.2g} - {zmax:.2g}")
        
        sliced = read_slice(config.cat, islice)
        if sliced.csize == 0:
            continue
        logger.info(f"read {sliced.csize} particles")
        
        try:
            weights_list, weights_int_list, delta = compute_weights_and_delta(config, sliced, zmin, zmax, ds_list)
            kappa += weights_list.reshape(-1, 1) * delta
            kappa_int += weights_int_list.reshape(-1, 1) * delta
        except Exception as e:
            logger.error(f"Error computing weights or mass map: {e}")
            continue
        
    logger.info("kappa map computed")

    return kappa, kappa_int

def main(config_file):
    config = ConfigData.from_json(config_file)
    logger = config.cat.logger

    # Compute comoving distances using Planck15 cosmology
    ds_list = Planck15.comoving_distance(config.zs_list)

    kappa = np.zeros([len(config.zs_list), config.npix], dtype="f8")
    kappa_int = np.zeros([len(config.zs_list), config.npix], dtype="f8")
    logger.info("pre-allocation done")
    logger.info("memory usage by kappa: %f GB" % ((kappa.nbytes + kappa_int.nbytes) / 1024 ** 3))

    logger.info("starting kappa computation")   
    kappa, kappa_int = process_slices(config, ds_list, logger)
    logger.info("kappa computation done")
    logger.info("writing to %s", config['destination'])

    save_path = os.path.join(config.datadir, config['destination'])
    os.makedirs(save_path, exist_ok=True)
    for i, (zs, ds) in enumerate(zip(config.zs_list, ds_list)):
        fname = save_path + "/WL-%02.2f-N%04d" % (zs, config.nside)
        logger.info("started writing source plane %s" % zs)

        with bigfile.File(fname, create=True) as ff:

            ds1 = ff.create_from_array("kappa", kappa[i], Nfile=1)
            ds2 = ff.create_from_array("kappa_int", kappa_int[i], Nfile=1)

            for d in ds1, ds2:
                d.attrs['nside'] = config.nside
                d.attrs['zlmin'] = config.zlmin
                d.attrs['zlmax'] = config.zlmax
                d.attrs['zs'] = zs
                d.attrs['ds'] = ds
                d.attrs['nbar'] = config.nbar

        # use bigfile because it allows concurrent write to different datasets.
        logger.info("source plane at %g written. " % zs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    # choices=["tiled", "bigbox"] #['config_tiled_hp.json', 'config_bigbox_hp.json']
    parser.add_argument('config', type=str, choices=['tiled', 'bigbox'], help='Configuration file')
    args = parser.parse_args()
    config_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_%s_hp.json' % args.config)
    main(config_file)