
import json
import numpy as np
import os
import sys
from .kappamap import KappaCodes
from glob import glob
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def main(dir_results, filenames, nside=8192, thetas=[4, 8, 16, 32], dataformat='npy'):
    logging.info("Starting the kappa maps processing.")
    kappa_maps = KappaCodes(dir_results=dir_results, filenames=filenames, nside=nside)

    try:
        if dataformat == 'fits':
            kappa_maps.readmaps_healpy()
        elif dataformat == 'npy':
            kappa_maps.readmaps_npy()

        for i, map_i in enumerate(kappa_maps.mapbins):
            logging.info(f'Tomo bin {i}')
            kappa_maps.run_map2alm(i)
            logging.info('Map2 transformation done.')
            kappa_maps.run_map3(i, thetas=thetas)
            logging.info('Map3 processing done.')

            logging.info('Starting cross correlation...')
            for j in range(i):
                kappa_maps.run_map2alm(Nmap1=i, Nmap2=j, is_cross=True)
                logging.info(f'Map2 cross {i} {j} done.')

    except Exception as e:
        logging.error(f"Error during map processing: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id', type=str, help='Configuration identifier')
    parser.add_argument('dataformat', choices=['npy', 'fits'], help='Data format of the map files')
    parser.add_argument('--config_file', type=str, default='/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json', help='Path to configuration JSON file')
    args = parser.parse_args()

    config = load_config(args.config_file)

    # Using the base directory from JSON and appending the config identifier
    dir_results = os.path.join(config['base_directory'], args.config_id)
    # Using glob to find files of the specified data format
    filenames = glob(os.path.join(dir_results, "wlen", "*", f"*.{args.dataformat}"))
    logging.info(f"Found {len(filenames)} files.")
    for f in filenames:
        logging.info(f)

    main(dir_results, filenames)
    logging.info("All done.")