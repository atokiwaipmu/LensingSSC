import argparse
import logging
import os

import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from .ConfigData import ConfigData, CatalogHandler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_cosmology(h=0.6774, om=0.309):
    """
    Set the cosmological parameters for the simulation
    if not using the default Planck15 cosmology.
    """
    global HUBBLE_CONSTANT, OMEGA_MATTER, cosmo
    HUBBLE_CONSTANT = h
    OMEGA_MATTER = om
    cosmo = FlatLambdaCDM(H0=HUBBLE_CONSTANT * 100, Om0=OMEGA_MATTER)

def get_mass_sheet(msheets, i):
    """
    Extracts and processes mass sheet data for given index 'i' from 'msheets'.
    Returns delta mass sheet contrast, and chi1, chi2 distances.
    """
    try:
        npix = msheets.attrs['healpix.npix'][0]
        start=msheets.attrs['aemitIndex.offset'][i+1]
        end=msheets.attrs['aemitIndex.offset'][i+2]
        logging.info(f"Processing sheet {i}: reading ID and Mass")

        pid = msheets['ID'][start:end].compute() 
        mass = msheets['Mass'][start:end].compute()
        ipix = pid % npix
        map_for_slice = np.bincount(ipix, weights=mass, minlength=npix)    

        # Calculate mean density contrast
        boxsize, M_cdm, nc = msheets.attrs['BoxSize'][0], msheets.attrs['MassTable'][1], msheets.attrs['NC'][0]
        rhobar = M_cdm * (nc / boxsize)**3

        a1, a2 = msheets.attrs['aemitIndex.edges'][i:i+2]
        z1, z2 = 1. / a1 - 1., 1. / a2 - 1.
        chi1, chi2 = cosmo.comoving_distance([z1, z2]).value * HUBBLE_CONSTANT
        volume_diff = 4 * np.pi * (chi1**3 - chi2**3) / (3 * npix)
        delta = map_for_slice / volume_diff / rhobar - 1
        return delta, chi1 / HUBBLE_CONSTANT, chi2 / HUBBLE_CONSTANT
    except Exception as e:
        logging.error(f"Error processing mass sheet {i}: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('config', type=str, choices=['tiled', 'bigbox'], help='Configuration file')
    parser.add_argument('index', type=int, help='Index of mass sheet')
    args = parser.parse_args()
    config_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_%s_hp.json' % args.config)
    config = ConfigData.from_json(config_file)
    cath = CatalogHandler(config.datadir, config.source, config.dataset)
    idx = args.index

    set_cosmology()
    delta, chi1, chi2 = get_mass_sheet(cath.cat, idx)
    delta = np.array(delta, dtype="float32")
    delta = hp.reorder(delta, n2r=True)
    delta_4096 = hp.ud_grade(delta, 4096)

    save_path = os.path.join(config.datadir, "mass_sheets")
    os.makedirs(save_path, exist_ok=True)
    np.savez(f"{save_path}/delta-sheet-{idx}.npz", delta=delta, chi1=np.asarray([chi1]), chi2=np.asarray([chi2]))
    np.savez(f"{save_path}/delta-sheet-4096-{idx}.npz", delta=delta_4096, chi1=np.asarray([chi1]), chi2=np.asarray([chi2]))
