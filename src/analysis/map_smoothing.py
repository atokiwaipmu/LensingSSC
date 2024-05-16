
import json
import numpy as np
import healpy as hp
import os
import sys
from .kappamap import KappaMaps
from glob import glob
import logging
import argparse

from ..masssheet.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def main(filenames, sl, nside=8192, dataformat='fits'):
    logging.info("Starting the kappa maps smoothing.")
    kappa_maps = KappaMaps(filenames=filenames, nside=nside)

    try:
        if dataformat == 'fits':
            kappa_maps.readmaps_healpy()
        elif dataformat == 'npy':
            kappa_maps.readmaps_npy()

        for i, map_i in enumerate(kappa_maps.mapbins):
            logging.info(f'Processing file: {filenames[i]}')
            smoothed_map = kappa_maps.smoothing(mapbin=map_i, sl=sl)
            smoothed_map_file = filenames[i].replace(f".{dataformat}", f"_smoothed.{dataformat}")
            if dataformat == 'fits':
                hp.write_map(smoothed_map_file, smoothed_map, dtype=np.float32)
            elif dataformat == 'npy':
                np.save(smoothed_map_file, smoothed_map)
            logging.info('Smoothing done.')

    except Exception as e:
        logging.error(f"Error during map processing: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id', type=str, help='Configuration identifier')
    parser.add_argument('dataformat', choices=['npy', 'fits'], help='Data format of the map files')
    args = parser.parse_args()

    config_analysis_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_analysis.json')
    config_analysis = ConfigAnalysis.from_json(config_analysis_file)

    # Using the base directory from JSON and appending the config identifier
    dir_results = os.path.join(config_analysis.resultsdir, args.config_id)
    logging.info(f"Using directory: {dir_results}")

    # Using glob to find files of the specified data format
    filenames = sorted(glob(os.path.join(dir_results, "data", f"kappa_zs*.{args.dataformat}")))
    logging.info(f"Found {len(filenames)} files.")
    for f in filenames:
        logging.info(f)

    for sl in config_analysis.sl_arcmin:
        main(filenames, sl, nside=config_analysis.nside, dataformat=args.dataformat)
    logging.info("All done.")