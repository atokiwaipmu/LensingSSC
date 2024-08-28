import argparse
import logging
import os
import warnings

import numpy as np
import healpy as hp
from astropy import constants as const
from astropy import cosmology
from astropy import units as u
from mpi4py import MPI

from src.utils.ConfigData import ConfigData
from src.utilities import CosmologySettings, extract_seed_from_directory

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

    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Process delta sheets across MPI ranks
    for i in range(i_start, i_end):
        if (i - i_start) % size == rank:
            logging.info(f"Rank {rank} processing delta sheet index {i}")
            delta = load_delta_sheet(data_path, i)
            chi1, chi2 = index_to_chi(i, cosmo)
            mapper.add_sheet_to_map("kappa", delta.astype('float32'), wlen_chi_kappa, chi1, chi2, cosmo, zs)

    # Gather and reduce local maps across all processes
    local_kappa = mapper.maps["kappa"]
    global_kappa = np.zeros_like(local_kappa) if rank == 0 else None
    comm.Reduce([local_kappa, MPI.FLOAT], [global_kappa, MPI.FLOAT], op=MPI.SUM, root=0)

    # Save the results at the root process
    if rank == 0:
        hp.write_map(save_path, global_kappa, dtype=np.float32)
        logging.info(f"Output maps saved to {save_path}")

    logging.info("Computation of weak lensing convergence maps completed.")

def main(args, config):
    """
    Main execution function to configure the environment and run the computation.
    """
    # Define paths for data and saving results
    data_path = os.path.join(args.datadir, "mass_sheets")
    if args.output is not None:
        save_path = args.output
    else:
        save_path = os.path.join(args.datadir, "kappa")
    os.makedirs(save_path, exist_ok=True)

    # Extract seed from directory name
    seed = extract_seed_from_directory(args.datadir)

    # Loop through the redshift list and compute maps
    for zs in config.zs_list:
        logging.info(f"Computing weak lensing convergence maps for zs={zs}")
        save_path_zs = os.path.join(save_path, f"kappa_zs{zs:.1f}_{seed}.fits")
        if os.path.exists(save_path_zs) and not args.overwrite:
            logging.info(f"Output file {save_path_zs} already exists. Skipping.")
            continue
        compute_weak_lensing_maps(data_path, save_path_zs, zs)

if __name__ == "__main__":
     # Load configuration file
    config_file = os.path.join(
        "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_data.json'
    )
    config = ConfigData.from_json(config_file)

    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('datadir', type=str, help='Data directory')
    parser.add_argument("--output", type=str, help="Output directory to save convergence maps")
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()

    main(args, config)