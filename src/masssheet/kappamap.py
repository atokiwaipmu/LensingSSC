import argparse
import logging
import os
import warnings

import numpy as np
import healpy as hp
from astropy import constants as const
from astropy import cosmology
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from mpi4py import MPI

from src.utils.ConfigData import ConfigData, ConfigAnalysis

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

def load_delta_sheet(path, index, randomize=False):
    """Load a delta sheet from a specified path and index."""
    filename = f"/delta-sheet-{index}.npz"
    data = np.load(path + filename)
    delta = data['delta']
    if randomize:
        lon, lat = np.random.rand() * 2 * np.pi, np.random.rand() * np.pi
        rot = hp.Rotator(rot = [lon, lat], deg=False)
        delta = rot.rotate_map_alms(delta)
        logging.info(f"Randomized delta sheet {index} with lon={lon}, lat={lat}")
    chi1, chi2 = data['chi1'][0], data['chi2'][0]
    return delta, chi1, chi2

def wlen_chi_kappa(chi, cosmo, zs):
    """Compute the weight function for weak lensing convergence."""
    chis = cosmo.comoving_distance(zs).value # Mpc
    H0 = 100 * cosmo.h / (const.c.cgs.value / 1e5)  # 1/Mpc
    z = cosmology.z_at_value(cosmo.comoving_distance, chi * u.Mpc).value
    dchi = (1 - chi / chis).clip(0)
    return 3 / 2 * cosmo.Om0 * H0 ** 2 * (1 + z) * chi * dchi # 1/Mpc

def main(data_path, save_path, zs, i_start=28, i_end=99, randomize=False):
    """
    Main function to compute weak lensing convergence maps.
    
    Parameters:
    - data_path (str): Path to the input data.
    - save_path (str): Path to save the output maps.
    """
    logging.info("Starting the computation of weak lensing convergence maps.")

    # Initialize cosmology model
    cosmo = FlatLambdaCDM(H0=67.74, Om0=0.309)
    logging.info(f"Using cosmology: H0={cosmo.H0}, Om0={cosmo.Om0}")

    # Initialize SheetMapper and create maps
    mapper = SheetMapper()
    mapper.new_map("kappa")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Load and process delta sheets
    for i in range(i_start, i_end):
        if (i - i_start) % size == rank:
            logging.info(f"Rank {rank} processing delta sheet index {i}")
            delta, chi1, chi2 = load_delta_sheet(data_path, i, randomize)
            mapper.add_sheet_to_map("kappa", delta.astype('float32'), wlen_chi_kappa, chi1, chi2, cosmo, zs)

    # Gather local maps
    local_kappa = mapper.maps["kappa"]

    # Initialize global maps at root
    global_kappa = np.zeros_like(local_kappa) if rank == 0 else None

    # Reduce maps across all processes
    comm.Reduce([local_kappa, MPI.FLOAT], [global_kappa, MPI.FLOAT], op=MPI.SUM, root=0)

    # Save the results at root
    if rank == 0:
        hp.write_map(save_path, global_kappa, dtype=np.float32)
        logging.info("Output maps saved to %s", save_path)

    logging.info("Computation of weak lensing convergence maps completed.")

if __name__ == "__main__":
    # Usage: python kappamap.py /path/to/data --randomize

    config_file = os.path.join(
        "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_data.json'
    )
    config = ConfigData.from_json(config_file)

    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('datadir', type=str, help='Data directory')
    parser.add_argument('--randomize', action='store_true', help='Randomize delta sheets')
    args = parser.parse_args()

    data_path = os.path.join(args.datadir, "mass_sheets")
    save_path = os.path.join(args.datadir, "kappa") if not args.randomize else os.path.join(args.datadir, "kappa_randomized")
    os.makedirs(save_path, exist_ok=True)

    # extract seed from datadir
    seed = os.path.basename(args.datadir).split("_")[-2]

    for zs in config.zs_list:
        logging.info(f"Computing weak lensing convergence maps for zs={zs}")
        save_path_zs = os.path.join(save_path, f"kappa_zs{zs:.1f}_{seed}.fits")
        if os.path.exists(save_path_zs):
            logging.info(f"Output file {save_path_zs} already exists. Skipping.")
            continue
        main(data_path, save_path_zs, zs, randomize=args.randomize)