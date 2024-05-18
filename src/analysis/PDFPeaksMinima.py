import json
import numpy as np
import os
import sys
from .kappamap import KappaCodes
from ..masssheet.ConfigData import ConfigData, ConfigAnalysis
from glob import glob
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
def main(dir_results, filenames, nside=8192, lmax=5000, dataformat='npy'):
    logging.info("Starting the kappa maps processing.")
    kappa_maps = KappaCodes(dir_results=dir_results, filenames=filenames, nside=nside, lmax=lmax)

    try:
        if dataformat == 'fits':
            kappa_maps.readmaps_healpy()
        elif dataformat == 'npy':
            kappa_maps.readmaps_npy()

        for i, map_i in enumerate(kappa_maps.mapbins):
            logging.info(f'Processing tomo bin: {i}')
            kappa_maps.run_PDFPeaksMinima(map_i, i)
            logging.info('PDF, peaks, minima done.')

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
    filenames = sorted(glob(os.path.join(dir_results, "data", f"kappa_zs0.5_smoothed_s*.{args.dataformat}")))
    logging.info(f"Found {len(filenames)} files.")
    for f in filenames:
        logging.info(f)

    main(dir_results, filenames, nside=config_analysis.nside, lmax=config_analysis.lmax, dataformat=args.dataformat)
    logging.info("All done.")