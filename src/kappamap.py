import argparse
import logging
import os
import yaml
import warnings
from functools import partial

import numpy as np
import healpy as hp
from astropy import constants as const
from astropy import cosmology
from astropy import units as u
import multiprocessing as mp

from src.utils import CosmologySettings, extract_seed_from_path

# Suppress future warnings and set up logging
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SheetMapper:
    """Handles operations related to sheet mapping for cosmological data visualization."""

    def __init__(self, nside=8192):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.maps = {}

    def new_map(self, map_name, dtype='float32'):
        """Create a new map with the given name and data type."""
        self.maps[map_name] = np.zeros(self.npix, dtype=dtype)

    def add_sheet_to_map(self, map_name, sheet, wlen, chi1, chi2, cosmology, zs):
        """Add a sheet to the map using a weak lensing integral."""
        chi =  3/4 * (chi1**4 - chi2**4) / (chi1**3 - chi2**3)  # [Mpc]
        dchi = chi1 - chi2 # [Mpc]
        wlen_integral = wlen(chi, cosmology, zs) * dchi  # [1]
        self.maps[map_name] += wlen_integral * sheet  

def load_delta_sheet(path, index):
    """Load a delta sheet from a specified path and index."""
    filename = f"delta-sheet-{index}.fits"
    delta = hp.read_map(os.path.join(path, filename))
    return delta

def wlen_chi_kappa(chi, cosmo, zs):
    """Compute the weight function for weak lensing convergence."""
    chis = cosmo.comoving_distance(zs).value # Mpc
    H0 = 100 * cosmo.h / (const.c.cgs.value / 1e5)  # 1/Mpc
    z = cosmology.z_at_value(cosmo.comoving_distance, chi * u.Mpc).value
    dchi = (1 - chi / chis).clip(0)
    return 3 / 2 * cosmo.Om0 * H0 ** 2 * (1 + z) * chi * dchi # 1/Mpc

def index_to_chi(index, cosmo):
    """Convert an index to a comoving distance."""
    a1, a2 = 0.01 * index, 0.01 * (index + 1)
    z1, z2 = 1. / a1 - 1., 1. / a2 - 1.
    chi1, chi2 = cosmo.comoving_distance([z1, z2]).value * cosmo.h
    return chi1, chi2

def process_delta_sheet(i, data_path, cosmo, mapper, wlen_chi_kappa, zs):
    logging.info(f"Processing delta sheet index {i}")
    delta = load_delta_sheet(data_path, i)
    chi1, chi2 = index_to_chi(i, cosmo)
    mapper.add_sheet_to_map("kappa", delta.astype('float32'), wlen_chi_kappa, chi1, chi2, cosmo, zs)
    return mapper.maps["kappa"]

def reduce_maps(results):
    global_kappa = np.zeros_like(results[0])
    for local_kappa in results:
        global_kappa += local_kappa
    return global_kappa

def compute_weak_lensing_maps(data_path, save_path, zs, i_start=28, i_end=99):
    """
    Compute weak lensing convergence maps for a given redshift.

    Parameters:
    - data_path (str): Path to the input data (delta sheets).
    - save_path (str): Path to save the output convergence maps.
    - zs (float): Source redshift for the weak lensing map.
    - i_start (int): Starting index for processing delta sheets.
    - i_end (int): Ending index for processing delta sheets.
    """
    logging.info("Starting the computation of weak lensing convergence maps.")

    # Initialize cosmology model
    cosmo = CosmologySettings().get_cosmology()

    # Initialize SheetMapper and create maps
    mapper = SheetMapper()
    mapper.new_map("kappa")

    # Create a partial function with fixed arguments for use with multiprocessing
    process_partial = partial(process_delta_sheet, data_path=data_path, cosmo=cosmo,
                              mapper=mapper, wlen_chi_kappa=wlen_chi_kappa, zs=zs)
    
    # Use multiprocessing Pool to parallelize the work
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_partial, range(i_start, i_end))

    # Reduce the results across all processes
    global_kappa = reduce_maps(results)

    # Save the results
    hp.write_map(save_path, global_kappa, dtype=np.float32)
    logging.info(f"Output maps saved to {save_path}")
    logging.info("Computation of weak lensing convergence maps completed.")

def main(datadir, output=None, zs_list = [0.5, 1.0, 2.0, 3.0], overwrite=False):
    """
    Main execution function to configure the environment and run the computation.
    """
    # Define paths for data and saving results
    if output is not None:
        save_path = output
    else:
        save_path = os.path.join(datadir, "..", "kappa")
    os.makedirs(save_path, exist_ok=True)

    # Extract seed from directory name
    seed = extract_seed_from_path(datadir)

    # Loop through the redshift list and compute maps
    for zs in zs_list:
        logging.info(f"Computing weak lensing convergence maps for zs={zs}")
        save_path_zs = os.path.join(save_path, f"kappa_zs{zs:.1f}_{seed}.fits")
        if os.path.exists(save_path_zs) and not overwrite:
            logging.info(f"Output file {save_path_zs} already exists. Skipping.")
            continue
        compute_weak_lensing_maps(datadir, save_path_zs, zs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('datadir', type=str, help='Data directory')
    parser.add_argument("--output", type=str, help="Output directory to save convergence maps")
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--config', type=str, help='Configuration file path')
    args = parser.parse_args()

   # Initialize empty config
    config = {}

    # Load configuration from YAML if provided and exists
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as file:
            try:
                config = yaml.safe_load(file)  # Load the configuration from YAML
            except yaml.YAMLError as exc:
                print("Warning: The config file is empty or invalid. Proceeding with default parameters.")
                print(exc)

    # Override YAML configuration with command-line arguments if provided
    config.update({
        'datadir': args.datadir,
        'output': args.output if args.output else config.get('output', None),
        'overwrite': args.overwrite if args.overwrite else config.get('overwrite', False),
    })

    allowed_keys = {'datadir', 'output', 'zs_list', 'overwrite'}
    config = {k: v for k, v in config.items() if k in allowed_keys}

    main(**config)