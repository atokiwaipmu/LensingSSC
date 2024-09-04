import argparse
import logging
import os
import yaml
import warnings

import numpy as np
import healpy as hp
from astropy import constants as const
from astropy import cosmology
from astropy import units as u
from mpi4py import MPI

from src.utils import CosmologySettings, extract_seed_from_path
from src.kappamap import compute_weak_lensing_maps

# Suppress future warnings and set up logging
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(datadir, output=None, overwrite=False):
    """
    Main execution function to configure the environment and run the computation.
    """
    # Define paths for data and saving results
    if output is not None:
        save_path = output
    else:
        save_path = os.path.join(datadir, "..", "kappa_cmb")
    os.makedirs(save_path, exist_ok=True)

    # Extract seed from directory name
    seed = extract_seed_from_path(datadir)

    # Loop through the redshift list and compute maps
    zs = 1100
    logging.info(f"Computing weak lensing convergence maps for zs={zs}")
    save_path_zs = os.path.join(save_path, f"kappa_zs{zs:.1f}_{seed}.fits")
    if os.path.exists(save_path_zs) and not overwrite:
        logging.info(f"Output file {save_path_zs} already exists. Skipping.")
        return
    compute_weak_lensing_maps(datadir, save_path_zs, zs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('datadir', type=str, help='Data directory')
    parser.add_argument("--output", type=str, help="Output directory to save convergence maps")
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()

   # Initialize empty config
    config = {}

    # Override YAML configuration with command-line arguments if provided
    config.update({
        'datadir': args.datadir,
        'output': args.output if args.output else config.get('output', None),
        'overwrite': args.overwrite if args.overwrite else config.get('overwrite', False),
    })

    allowed_keys = {'datadir', 'output', 'overwrite'}
    config = {k: v for k, v in config.items() if k in allowed_keys}

    main(**config)