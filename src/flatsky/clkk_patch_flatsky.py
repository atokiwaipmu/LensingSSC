
import os
import logging
from glob import glob
import argparse

import numpy as np
from astropy import units as u
from lenstools import ConvergenceMap

from src.utils.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_power_spectrum(patch, angle, lmax=5000, lmin=100):
    convergence_map = ConvergenceMap(patch, angle=angle * u.deg)
    multipole_edges = np.arange(lmin, lmax + 1, 100)
    ell, cl = convergence_map.powerSpectrum(multipole_edges)
    return ell, cl

def main(kappa_map_files, save_directory, patch_size_deg=10, lmin=100, lmax=5000):
    logging.info("Starting the kappa maps processing.")
    for kappa_map_file in kappa_map_files:
        patch = np.load(kappa_map_file)
        ell, cl = compute_power_spectrum(patch, angle=patch_size_deg, lmax=lmax, lmin=lmin)
        save_filename = os.path.join(save_directory, os.path.basename(kappa_map_file).replace('.npy', 
                                f'_Clkk_ell_{lmin}_{lmax}.npz'))
        np.savez(save_filename, ell=ell, clkk=cl, lmin=lmin, lmax=lmax)
        logging.info(f"Saved the results to {save_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id', type=str, help='Configuration identifier')
    parser.add_argument('source_redshift', type=float, help='Source redshift')
    parser.add_argument('--patch_size_deg', type=int, default=10, help='Size of each patch in degrees')
    args = parser.parse_args()

    config_path = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json"
    config_analysis = ConfigAnalysis.from_json(config_path)

    results_directory = os.path.join(config_analysis.resultsdir, args.config_id)
    kappa_map_files = glob(os.path.join(results_directory, "patch_flat", f"zs{args.source_redshift:.1f}", f"kappa_zs{args.source_redshift:.1f}*.npy"))

    save_directory = os.path.join(results_directory, "Clkk", "patch_flat", f"zs{args.source_redshift:.1f}")
    os.makedirs(save_directory, exist_ok=True)

    main(kappa_map_files, save_directory, patch_size_deg=args.patch_size_deg)
    logging.info("Processing complete.")