import argparse
import gc
import logging
import os

import healpy as hp
import numpy as np
from astropy.io import fits
from astropy import cosmology
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from src.utils.ConfigData import ConfigData, CatalogHandler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_cosmology(h=0.6774, om=0.309):
    """
    Set the cosmological parameters for the simulation
    if not using the default Planck15 cosmology.
    """
    global HUBBLE_CONSTANT, OMEGA_MATTER, H0, cosmo
    HUBBLE_CONSTANT = h
    OMEGA_MATTER = om
    H0 = 100 * HUBBLE_CONSTANT / (const.c.cgs.value / 1e5)
    cosmo = FlatLambdaCDM(H0=HUBBLE_CONSTANT * 100, Om0=OMEGA_MATTER)

def get_delta(msheets, i, save_path, alms=False):
    """
    Extracts and processes mass sheet data for given index 'i' from 'msheets'.
    Returns delta mass sheet contrast, and chi1, chi2 distances.
    """
    npix = msheets.attrs['healpix.npix'][0]
    start=msheets.attrs['aemitIndex.offset'][i+1]
    end=msheets.attrs['aemitIndex.offset'][i+2]
    logging.info(f"Processing sheet {i}: reading ID and Mass")

    pid = msheets['ID'][start:end].compute() 
    mass = msheets['Mass'][start:end].compute()
    ipix = pid % npix
    massshell = np.bincount(ipix, weights=mass, minlength=npix) # [Msun/(Vshell/npix)], RING ordering
    del pid, mass, ipix
    gc.collect()

    # Calculate mean density contrast
    boxsize, M_cdm, nc = msheets.attrs['BoxSize'][0], msheets.attrs['MassTable'][1], msheets.attrs['NC'][0]
    volbox = boxsize**3 # [(Mpc/h)^3]
    Npart = nc**3 # Number of particles
    rhobar = M_cdm * Npart / volbox # Mean density [Msun/(Mpc/h)^3]

    # Calculate delta mass sheet contrast
    a1, a2 = msheets.attrs['aemitIndex.edges'][i:i+2]
    z1, z2 = 1. / a1 - 1., 1. / a2 - 1.
    chi1, chi2 = cosmo.comoving_distance([z1, z2]).value * HUBBLE_CONSTANT # [Mpc/h]

    chiplane = 3 / 4 * (chi1**4 - chi2**4) / (chi1**3 - chi2**3) # [Mpc/h]
    zplane = cosmology.z_at_value(cosmo.comoving_distance, chiplane * u.Mpc / HUBBLE_CONSTANT).value # Redshift at plane
    volshell = 4 * np.pi * (chi1**3 - chi2**3) / 3 # [(Mpc/h)^3]
    volpix = volshell / npix # Volume per pixel
    pixpart = volpix / volbox * Npart # Number of particles per pixel

    factor = 3 / 2 * OMEGA_MATTER * H0 ** 2 / HUBBLE_CONSTANT**2 #[(Mpc/h)^-2]

    if not os.path.exists(f"{save_path}/delta/delta-rho-{i}.fits"):
        # save delta rho
        delta_rho = massshell / volpix / rhobar - 1 # Density contrast
        hdu = fits.PrimaryHDU(np.array(delta_rho, dtype=np.float32))
        hdu.header['CHI'] = (chiplane, 'Comoving distance to plane [Mpc/h]')
        hdu.header['Z'] = (zplane, 'Redshift at plane')
        hdu.header['FACTOR'] = (factor, 'Factor for further computation')
        hdu.header['ORDERING'] = ('RING', 'Healpix ordering')
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(f"{save_path}/delta/delta-rho-{i}.fits", overwrite=True)

        del delta_rho, hdu, hdulist
        gc.collect()

    if alms:
        # Calculate delta mass sheet contrast
        kappa = factor * volbox * npix / Npart / (4*np.pi) * (massshell / M_cdm - pixpart) # Convergence
        del massshell
        gc.collect()

        kappa_alm = hp.map2alm(kappa, use_pixel_weights=True)
        del kappa
        gc.collect()

        Phi_alm = np.zeros_like(kappa_alm)
        lmax = hp.Alm.getlmax(len(kappa_alm))
        ls, ms = hp.Alm.getlm(lmax)
        for l, m in zip(ls, ms):
            idx = hp.Alm.getidx(lmax, l, m)
            if l == 0:
                kappa_alm[idx] = 0
            else:
                Phi_alm[idx] = 2 / l / (l + 1) * kappa_alm[idx]

        Phi_alm = np.array(Phi_alm, dtype=np.complex64)
        real_hdu = fits.ImageHDU(np.real(Phi_alm), name='REAL')
        imaginary_hdu = fits.ImageHDU(np.imag(Phi_alm), name='IMAG')
        hdu = fits.PrimaryHDU()
        hdu.header['CHI'] = (chiplane, 'Comoving distance to plane [Mpc/h]')
        hdu.header['Z'] = (zplane, 'Redshift at plane')
        hdu.header['FACTOR'] = (factor, 'Factor for further computation')
        hdulist = fits.HDUList([hdu, real_hdu, imaginary_hdu])
        hdulist.writeto(f"{save_path}/potential/Phi-alm-{i}.fits", overwrite=True)
    
    return 

def main(msheets, datadir):
    for i in range(20, 100):
        if msheets.attrs['aemitIndex.offset'][i+1] == msheets.attrs['aemitIndex.offset'][i+2]:
            logging.info(f"Sheet {i} is empty. Skipping...")
            continue
        os.makedirs(os.path.join(datadir, "delta"), exist_ok=True)
        os.makedirs(os.path.join(datadir, "potential"), exist_ok=True)
        get_delta(msheets, i, datadir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('datadir', type=str, help='Data directory')
    args = parser.parse_args()
    cath = CatalogHandler(args.datadir, "usmesh/", "HEALPIX/")

    set_cosmology()
    main(cath.cat, args.datadir)
