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

from .ConfigData import ConfigData

warnings.simplefilter(action='ignore', category=FutureWarning)

class SheetMapper:
    """Handles operations related to sheet mapping for cosmological data visualization."""

    def __init__(self, nside=8192):
        self.nside = nside
        self.maps = {}

    def new_map(self, map_name, dtype='float32'):
        """Create a new map with the given name and data type."""
        self.maps[map_name] = np.zeros(12 * self.nside ** 2, dtype=dtype)

    def add_sheet_to_map(self, map_name, sheet, wlen, chi1, chi2, cosmology, zs):
        """Add a sheet to the map using a weak lensing integral."""
        chi = (chi1 + chi2) / 2
        dchi = chi2 - chi1
        wlen_integral = wlen(chi, cosmology, zs) * dchi
        self.maps[map_name] += wlen_integral * sheet

    def add_sheet_to_map_int(self, map_name, sheet, wlen, chi1, chi2, cosmology, zs):
        """Add a sheet to the map using an integral approach over 100 points."""
        nchi = 100
        chi = np.linspace(chi1, chi2, nchi)
        dchi = chi[1] - chi[0]
        wlen_integral = wlen(chi, cosmology, zs).sum() * dchi
        self.maps[map_name] += wlen_integral * sheet


def load_delta_sheet(path, index, r4096=False):
    """Load a delta sheet from a specified path and index."""
    filename = f"/delta-sheet-4096-{index}.npz" if r4096 else f"/delta-sheet-{index}.npz"
    data = np.load(path + filename)
    delta = data['delta']
    chi1, chi2 = data['chi1'][0], data['chi2'][0]
    return delta, chi1, chi2


def wlen_chi_kappa(chi, cosmo, zs):
    """Compute the weight function for weak lensing convergence."""
    chis = cosmo.comoving_distance(zs).value
    H0 = 100 * cosmo.h / (const.c.cgs.value / 1e5)  # 1/Mpc
    z = cosmology.z_at_value(cosmo.comoving_distance, chi * u.Mpc).value
    dchi = (1 - chi / chis).clip(0)
    return 3 / 2 * cosmo.Om0 * H0 ** 2 * (1 + z) * chi * dchi


def main(data_path, save_path, zs, i_start=20, i_end=99, r4096=False):
    """
    Main function to compute weak lensing convergence maps.
    
    Parameters:
    - data_path (str): Path to the input data.
    - save_path (str): Path to save the output maps.
    - r4096 (bool): Flag to use 4096 resolution.
    """
    logging.info("Starting the computation of weak lensing convergence maps.")

    # Initialize cosmology model
    cosmo = FlatLambdaCDM(H0=67.74, Om0=0.309)
    logging.info(f"Using cosmology: H0={cosmo.H0}, Om0={cosmo.Om0}")

    # Initialize SheetMapper and create maps
    mapper = SheetMapper() if not r4096 else SheetMapper(nside=4096)
    mapper.new_map("kappa")
    mapper.new_map("kappa_int")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Load and process delta sheets
    for i in range(i_start, i_end):
        if (i - i_start) % size == rank:
            logging.info(f"Rank {rank} processing delta sheet index {i}")
            delta, chi1, chi2 = load_delta_sheet(data_path, i, r4096)
            mapper.add_sheet_to_map("kappa", delta.astype('float32'), wlen_chi_kappa, chi1, chi2, cosmo, zs)
            mapper.add_sheet_to_map_int("kappa_int", delta.astype('float32'), wlen_chi_kappa, chi1, chi2, cosmo, zs)

    # Gather local maps
    local_kappa = mapper.maps["kappa"]
    local_kappa_int = mapper.maps["kappa_int"]

    # Initialize global maps at root
    global_kappa = np.zeros_like(local_kappa) if rank == 0 else None
    global_kappa_int = np.zeros_like(local_kappa_int) if rank == 0 else None

    # Reduce maps across all processes
    comm.Reduce([local_kappa, MPI.FLOAT], [global_kappa, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce([local_kappa_int, MPI.FLOAT], [global_kappa_int, MPI.FLOAT], op=MPI.SUM, root=0)

    # Save the results at root
    if rank == 0:
        global_kappa = hp.reorder(global_kappa, r2n=True)
        global_kappa_int = hp.reorder(global_kappa_int, r2n=True)
        hp.write_map(os.path.join(save_path, "kappa.fits"), global_kappa, dtype=np.float32)
        hp.write_map(os.path.join(save_path, "kappa_int.fits"), global_kappa_int, dtype=np.float32)
        logging.info("Output maps saved to %s", save_path)
        logging.info("Kappa map min/max: %f/%f", global_kappa.min(), global_kappa.max())
        logging.info("Kappa_int map min/max: %f/%f", global_kappa_int.min(), global_kappa_int.max())

    logging.info("Computation of weak lensing convergence maps completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('config', type=str, choices=['tiled', 'bigbox'], help='Configuration file')
    parser.add_argument('--r4096', type=bool, default=False, help='Use 4096 resolution')
    args = parser.parse_args()

    config_file = os.path.join(
        "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs",
        f'config_{args.config}_hp.json'
    )
    config = ConfigData.from_json(config_file)

    data_path = os.path.join(config.datadir, "mass_sheets")
    save_path = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/results", args.config)
    if args.r4096:
        save_path += "-4096"
    os.makedirs(save_path, exist_ok=True)

    for zs in config.zs_list:
        logging.info(f"Computing weak lensing convergence maps for zs={zs}")
        save_path_zs = os.path.join(save_path, f"zs-{zs}")
        os.makedirs(save_path_zs, exist_ok=True)
        main(data_path, save_path_zs, zs, r4096=args.r4096)
