import json
import numpy as np
import os
import sys
from glob import glob
import logging
import argparse
from .kappamap import KappaCodes
from ..masssheet.ConfigData import ConfigData, ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(dir_results: str, filenames: list, nside: int = 8192, lmax: int = 5000):
    logging.info("Starting the kappa maps processing.")
    kappa_maps = KappaCodes(dir_results=dir_results, filenames=filenames, nside=nside, lmax=lmax)

    try:
        kappa_maps.readmaps_healpy()
        
        for i, map_i in enumerate(kappa_maps.mapbins):
            logging.info(f'Processing tomo bin: {i}')
            kappa_maps.run_PeaksMinima(map_i, i)
            logging.info('PDF, peaks, minima done.')

    except Exception as e:
        logging.error(f"Error during map processing: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id', type=str, help='Configuration identifier')
    parser.add_argument('zs', type=float, help='Source redshift')
    args = parser.parse_args()

    config_analysis_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_analysis.json')
    config_analysis = ConfigAnalysis.from_json(config_analysis_file)

    # Using the base directory from JSON and appending the config identifier
    dir_results = os.path.join(config_analysis.resultsdir, args.config_id)
    logging.info(f"Using directory: {dir_results}")
    
    # Gather filenames for each redshift
    filenames = sorted(glob(os.path.join(dir_results, "smoothed", f"kappa_zs{args.zs}_smoothed_s*.fits")))
    logging.info(f"Found {len(filenames)} files for redshift {args.zs}.")
    
    if not filenames:
        logging.error("No files found. Please check the directory and file naming conventions.")
        sys.exit(1)
    
    for f in filenames:
        logging.info(f"Processing file: {f}")

    main(dir_results, filenames, nside=config_analysis.nside, lmax=config_analysis.lmax)
    logging.info("All done.")
