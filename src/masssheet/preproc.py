import argparse
import logging
import os
from tracemalloc import start

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from src.utils.ConfigData import CatalogHandler

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

def get_indices(msheets, i, extra_index=100, start=None):
    # find where to start
    # start can be inherited from previous sheet
    if start is not None:
        start=start
        end=msheets.attrs['aemitIndex.offset'][i+2]
    else:
        start=msheets.attrs['aemitIndex.offset'][i+1]
        end=msheets.attrs['aemitIndex.offset'][i+2]
        if extra_index is not None:
            search_start_last = np.min([start + extra_index, end])
            aemit_start = msheets['Aemit'][start:search_start_last].compute()
            change_index_start = np.where(np.diff(aemit_start) == 0.01)[0]
            if len(change_index_start) != 0:
                logging.info(f"Aemit {np.round(aemit_start[change_index_start[0]], 2):.2f} start changed from {start} to {start + change_index_start[0]}")
                start = start + change_index_start[0]

    if start == end:
        return start, end
    
    logging.info(f"start: {start}, end: {end}")

    # take (start - extra_index) to (start + extra_index) to avoid bugs
    if extra_index is not None:
        search_end_first = np.max([end - extra_index, start])
        aemit_end = msheets['Aemit'][search_end_first:end].compute()
        change_index_end = np.where(np.round(np.diff(aemit_end), 2) == 0.01)[0]      
        if len(change_index_end) != 0:
            logging.info(f"Aemit {np.round(aemit_end[change_index_end[0]], 2):.2f} end changed from {end} to {end - change_index_end[0]}")
            end = end - change_index_end[0]
        return  start, end
    
    return start, end

def get_mass_sheet(msheets, i, start, end):
    """
    Extracts and processes mass sheet data for given index 'i' from 'msheets'.
    Returns delta mass sheet contrast, and chi1, chi2 distances.
    """
    npix = msheets.attrs['healpix.npix'][0]
    logging.info(f"Processing sheet {i}: reading ID and Mass")

    pid = msheets['ID'][start:end].compute() 
    mass = msheets['Mass'][start:end].compute()
    ipix = pid % npix
    map_for_slice = np.bincount(ipix, weights=mass, minlength=npix)    

    # Calculate mean density contrast
    boxsize, M_cdm, nc = msheets.attrs['BoxSize'][0], msheets.attrs['MassTable'][1], msheets.attrs['NC'][0]
    rhobar = M_cdm * (nc / boxsize)**3 # Mean density [Msun/(Mpc/h)^3]

    a1, a2 = msheets.attrs['aemitIndex.edges'][i:i+2]
    z1, z2 = 1. / a1 - 1., 1. / a2 - 1.
    chi1, chi2 = cosmo.comoving_distance([z1, z2]).value * HUBBLE_CONSTANT # [Mpc/h]
    volume_diff = 4 * np.pi * (chi1**3 - chi2**3) / (3 * npix) # [(Mpc/h)^3]
    delta = map_for_slice / volume_diff / rhobar - 1 # Density contrast
    return delta, chi1 / HUBBLE_CONSTANT, chi2 / HUBBLE_CONSTANT # [], [Mpc], [Mpc]

def main(msheets, datadir, overwrite=False):
    os.makedirs(os.path.join(datadir, "mass_sheets"), exist_ok=True)
    prev_end = None
    for i in range(20, 100):
        if msheets.attrs['aemitIndex.offset'][i+1] == msheets.attrs['aemitIndex.offset'][i+2]:
            logging.info(f"Sheet {i} is empty. Skipping...")
            continue
        save_path = f"{datadir}/mass_sheets/delta-sheet-{i}.npz"
        if os.path.exists(save_path) and not overwrite:
            logging.info(f"Sheet {i} already exists. Skipping...")
            prev_end = None
            continue
        start, end = get_indices(msheets, i, extra_index=50, start=prev_end)
        prev_end = end
        delta, chi1, chi2 = get_mass_sheet(msheets, i, start, end)
        delta = np.array(delta, dtype="float32")
        np.savez(f"{datadir}/mass_sheets/delta-sheet-{i}.npz", delta=delta, chi1=np.asarray([chi1]), chi2=np.asarray([chi2]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('datadir', type=str, help='Data directory')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing mass sheets')
    args = parser.parse_args()
    cath = CatalogHandler(args.datadir, "usmesh/", "HEALPIX/")

    set_cosmology()
    main(cath.cat, args.datadir, args.overwrite)
