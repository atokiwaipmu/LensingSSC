
import os
import logging
import argparse

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from src.utils.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def project_gnomonic(kappa_map, center_ra, center_dec, xsize=800, resolution=1.5):
    return hp.gnomview(kappa_map, rot=[center_ra, center_dec], xsize=xsize, reso=resolution, return_projected_map=True)

def process_kappa_map(save_directory, kappa_map_path, patch_size_deg=10):
    logging.info("Starting the kappa maps processing.")
    kappa_map = hp.read_map(kappa_map_path)
    kappa_map = hp.reorder(kappa_map, n2r=True)

    for ra in range(0, 360, patch_size_deg):
        for dec in range(-30, 40, patch_size_deg):
            patch = project_gnomonic(kappa_map, center_ra=ra, center_dec=dec, xsize=800, resolution=patch_size_deg*60/800)

            save_filename = os.path.join(save_directory, os.path.basename(kappa_map_path).replace('.fits', 
                                f'_{patch_size_deg}x{patch_size_deg}_center{ra}_{dec}.npy'))
            np.save(save_filename, patch.data)

            plt.close()
            logging.info(f"Saved the results to {save_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id', type=str, help='Configuration identifier')
    parser.add_argument('source_redshift', type=float, help='Source redshift')
    parser.add_argument('smoothing_length', type=int, help='Smoothing length in arcmin')
    parser.add_argument('--patch_size_deg', type=int, default=10, help='Size of each patch in degrees')
    args = parser.parse_args()

    config_path = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json"
    config_analysis = ConfigAnalysis.from_json(config_path)

    results_directory = os.path.join(config_analysis.resultsdir, args.config_id)
    save_directory = os.path.join(results_directory, "smoothed_patch_flat", f"zs{args.source_redshift:.1f}", f"sl{args.smoothing_length}")
    os.makedirs(save_directory, exist_ok=True)

    kappa_map_file = os.path.join(results_directory, "smoothed", f"kappa_zs{args.source_redshift:.1f}_smoothed_s{args.smoothing_length}.fits")
    logging.info(f"Using file: {kappa_map_file}")

    process_kappa_map(save_directory, kappa_map_file, patch_size_deg=args.patch_size_deg)
    logging.info("Processing complete.")