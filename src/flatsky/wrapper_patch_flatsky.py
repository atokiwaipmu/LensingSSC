import os
import logging
import argparse
from glob import glob
import multiprocessing as mp

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from src.utils.compute_sigma import parse_file_name
from src.utils.ConfigData import ConfigAnalysis
from src.flatsky.analysis_patch_flatsky import WeakLensingAnalysis, WeakLensingCovariance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def project_gnomonic(kappa_map, center_ra, center_dec, xsize, resolution):
    return hp.gnomview(kappa_map, rot=[center_ra, center_dec], xsize=xsize, reso=resolution, return_projected_map=True)

def process_kappa_map(dir_results, kappa_map_path, patch_size_deg=10):
    logging.info(f"Processing {kappa_map_path}")
    source_redshift, smoothing_length, survey = parse_file_name(kappa_map_path)
    if smoothing_length is None:
        save_directory = os.path.join(dir_results, f"zs{source_redshift:.1f}", "patch_flat")
    elif survey is None:
        save_directory = os.path.join(dir_results, f"zs{source_redshift:.1f}", f"sl{smoothing_length}", "patch_flat")
    else:
        save_directory = os.path.join(dir_results, f"zs{source_redshift:.1f}", f"sl{smoothing_length}", f"{survey}", "patch_flat")
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(os.path.join(save_directory, "data"), exist_ok=True)

    kappa_map = hp.read_map(kappa_map_path)
    stddev = np.std(kappa_map)
    bins = np.linspace(-4*stddev, 4*stddev, 15)
    
    for ra in range(0, 360, patch_size_deg):
        for dec in range(-30, 40, patch_size_deg):
            patch = project_gnomonic(kappa_map, center_ra=ra, center_dec=dec, xsize=1024, resolution=patch_size_deg*60/1024)
            save_filename = os.path.basename(kappa_map_path).replace('.fits', f'_{patch_size_deg}x{patch_size_deg}_center{ra}_{dec}.npy')
            np.save(os.path.join(save_directory, "data", save_filename), patch.data)

            wl = WeakLensingAnalysis(save_directory, save_filename, patch, angle=patch_size_deg, lmax=3000, lmin=300, nbin=15, xsize=1024, save=True)
            ell, equilateral, halfed, squeezed = wl.compute_bispectrum()
            ell, cl = wl.compute_power_spectrum()
            nu, p = wl.calculate_pdf(bins=bins)
            peak_bins, peak_height, peak_positions = wl.calculate_peaks(peak_bins=bins)
            minima_bins, minima_height, minima_positions = wl.calculate_minima(minima_bins=bins)

            plt.close()
            logging.info(f"Saved the results to {save_filename}")

    wlcov = WeakLensingCovariance(save_directory, save=True)
    wlcov.cov_full()
    logging.info(f"Saved the results to {save_directory}")

def process_all_kappa_maps(kappa_map_dir, dir_results, patch_size_deg=10):
    kappa_map_paths = glob(os.path.join(kappa_map_dir, "*.fits"))
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(process_kappa_map, [(kappa_map_path, dir_results, patch_size_deg) for kappa_map_path in kappa_map_paths])

if __name__ == '__main__':
    config_analysis_path = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json"
    config_analysis = ConfigAnalysis.from_json(config_analysis_path)

    data_paths_bigbox = glob(f"{config_analysis.resultsdir}/bigbox/data/kappa_zs*.fits")
    data_paths_tiled = glob(f"{config_analysis.resultsdir}/tiled/data/kappa_zs*.fits")

    data_paths_bigbox_smoothed = glob(f"{config_analysis.resultsdir}/bigbox/smoothed/sl=*/kappa_zs*.fits")
    data_paths_tiled_smoothed = glob(f"{config_analysis.resultsdir}/tiled/smoothed/sl=*/kappa_zs*.fits")

    dir_results_bigbox = os.path.join(config_analysis.resultsdir, "bigbox")
    dir_results_tiled = os.path.join(config_analysis.resultsdir, "tiled")

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(process_kappa_map, [(dir_results_bigbox, data_path) for data_path in data_paths_bigbox + data_paths_bigbox_smoothed])

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(process_kappa_map, [(dir_results_tiled, data_path) for data_path in data_paths_tiled + data_paths_tiled_smoothed])