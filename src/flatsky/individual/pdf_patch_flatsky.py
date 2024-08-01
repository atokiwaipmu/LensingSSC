
import os
import logging
from glob import glob
import argparse

import healpy as hp
import numpy as np
from astropy import units as u
from lenstools import ConvergenceMap

from src.utils.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_pdf(patch, angle, bins):
    conv_map = ConvergenceMap(patch, angle=angle * u.deg)
    nu,p = conv_map.pdf(bins)
    return nu, p

def main(kappa_map_files, save_directory, bins, patch_size_deg=10):
    logging.info("Starting the kappa maps processing.")
    for kappa_map_file in kappa_map_files:
        patch = np.load(kappa_map_file)
        nu, p = calculate_pdf(patch, patch_size_deg, bins)
        save_filename = os.path.join(save_directory, os.path.basename(kappa_map_file).replace('.npy', 
                                f'_pdf.npz'))
        np.savez(save_filename, nu=nu, p=p)
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

    save_directory = os.path.join(results_directory, "PDF","patch_flat", f"zs{args.source_redshift:.1f}")
    os.makedirs(save_directory, exist_ok=True)

    fullsky_file_bigbox = os.path.join(config_analysis.resultsdir, "bigbox", "data", f"kappa_zs{args.source_redshift:.1f}.fits")
    fullsky_map_bigbox = hp.reorder(hp.read_map(fullsky_file_bigbox), n2r=True)
    sigma_bigbox = np.std(fullsky_map_bigbox)

    bins = np.linspace(-4*sigma_bigbox, 4*sigma_bigbox, 15, endpoint=True)
    logging.info(f"Using bins from {bins[0]} to {bins[-1]}, with {len(bins)-1} bins.")

    main(kappa_map_files, save_directory, bins, patch_size_deg=args.patch_size_deg)
    logging.info("Processing complete.")