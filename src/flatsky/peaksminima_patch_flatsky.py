
import os
import logging
from glob import glob
import argparse

import numpy as np
from astropy import units as u
from lenstools import ConvergenceMap

from src.utils.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_peaks_minima(patch, angle):
    conv_map = ConvergenceMap(patch, angle=angle * u.deg)
    l_edges = np.arange(-0.01, 0.06, 0.002)
    peak_height,peak_positions = conv_map.locatePeaks(l_edges)

    conv_map_minus = ConvergenceMap(-patch, angle=angle * u.deg)
    minima_height,minima_positions = conv_map_minus.locatePeaks(l_edges)

    return peak_height, peak_positions, minima_height, minima_positions

def main(kappa_map_files, save_directory, patch_size_deg=10):
    logging.info("Starting the kappa maps processing.")
    for kappa_map_file in kappa_map_files:
        patch = np.load(kappa_map_file)
        peak_height, peak_positions, minima_height, minima_positions = calculate_peaks_minima(patch, angle=patch_size_deg)
        logging.info(f"Found {len(peak_positions)} peaks and {len(minima_positions)} minima.")
        save_filename = os.path.join(save_directory, os.path.basename(kappa_map_file).replace('.npy', 
                                f'_peaksminima.npz'))
        np.savez(save_filename,peak_height=peak_height, peak_positions=peak_positions, minima_height=minima_height, minima_positions=minima_positions)
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
    kappa_map_files = glob(os.path.join(results_directory, "smoothed_patch_flat", f"zs{args.source_redshift:.1f}", f"sl{args.smoothing_length}", "*.npy"))

    save_directory = os.path.join(results_directory, "peakminima","smoothed_patch_flat", 
                                  f"zs{args.source_redshift:.1f}", f"sl{args.smoothing_length}")
    os.makedirs(save_directory, exist_ok=True)

    main(kappa_map_files, save_directory, patch_size_deg=args.patch_size_deg)
    logging.info("Processing complete.")