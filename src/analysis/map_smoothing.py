
import numpy as np
import healpy as hp
import os
from glob import glob
import logging
import multiprocessing as mp
from typing import List, Tuple
import itertools

from ..masssheet.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_map(mapfile: str, sl_arcmin: float) -> None:
    smoothed_map_file = mapfile.replace(".fits", f"_smoothed_s{sl_arcmin}.fits").replace("data", "smoothed")
    if os.path.exists(smoothed_map_file):
        logging.info(f"File already exists: {smoothed_map_file}")
        return
    else:
        logging.info(f"Processing map: {mapfile}")
        try:
            sl_rad = sl_arcmin / 60 / 180 * np.pi
            kappa_masked = hp.ma(hp.reorder(hp.read_map(mapfile), n2r=True))
            smoothed_map = hp.smoothing(kappa_masked , sigma=sl_rad)
            hp.write_map(smoothed_map_file, smoothed_map, dtype=np.float32)
            logging.info(f"Processed and saved: {smoothed_map_file}")
        except Exception as e:
            logging.error(f"Error processing {mapfile} with smoothing length {sl_arcmin}: {e}")

def worker(params: Tuple[str, np.ndarray, float]) -> None:
    filename, sl_arcmin = params
    process_map(filename, sl_arcmin)
    logging.info(f"Processed map for {filename} with smoothing length {sl_arcmin}.")


def main() -> None:
    config_analysis_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_analysis.json')
    config_analysis = ConfigAnalysis.from_json(config_analysis_file)

    tasks = []
    for config_id in ['tiled', 'bigbox']:
        logging.info(f"Config ID: {config_id}")
        dir_data = os.path.join(config_analysis.resultsdir, config_id, "data")
        filenames = sorted(glob(os.path.join(dir_data, "kappa_zs*.fits")))
        logging.info(f"Found {len(filenames)} files.")
        
        # Create tasks for each combination of config_id, mapbin, and smoothing length
        tasks.extend(itertools.product(filenames, config_analysis.sl_arcmin))

        # Process all combinations in parallel
        with mp.Pool() as pool:
            pool.map(worker, tasks)

if __name__ == '__main__':
    main()
