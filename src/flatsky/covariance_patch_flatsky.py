import os
import logging
from glob import glob
import multiprocessing as mp

from src.utils.compute_sigma import parse_file_name
from src.utils.ConfigData import ConfigAnalysis
from src.flatsky.analysis_patch_flatsky import WeakLensingCovariance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_covariance(dir_results, kappa_map_path):
    logging.info(f"Processing {kappa_map_path}")
    source_redshift, smoothing_length, survey = parse_file_name(kappa_map_path)
    if smoothing_length is None:
        save_directory = os.path.join(dir_results, f"zs{source_redshift:.1f}", "patch_flat")
    elif survey is None:
        save_directory = os.path.join(dir_results, f"zs{source_redshift:.1f}", f"sl{smoothing_length}", "patch_flat")
    else:
        save_directory = os.path.join(dir_results, f"zs{source_redshift:.1f}", f"sl{smoothing_length}", f"{survey}", "patch_flat")

    if not os.path.exists(save_directory):
        logging.info(f"direcotry {save_directory} does not exist")
        return
    
    wl = WeakLensingCovariance(save_directory, save=True)
    wl.cov_full()
    logging.info(f"Saved the results to {save_directory}")

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
        pool.starmap(process_covariance, [(dir_results_bigbox, data_path) for data_path in data_paths_bigbox + data_paths_bigbox_smoothed])

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(process_covariance, [(dir_results_tiled, data_path) for data_path in data_paths_tiled + data_paths_tiled_smoothed])

    # close the pool
    pool.close()
    pool.join()

    logging.info("Finished")