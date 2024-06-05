
import os
import sys
from glob import glob
import logging
import argparse

from src.utils.kappamap import KappaCodes
from src.utils.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id', type=str, help='Configuration identifier')
    parser.add_argument('zs', type=float, help='Source redshift')
    args = parser.parse_args()

    config_analysis_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_analysis.json')
    config_analysis = ConfigAnalysis.from_json(config_analysis_file)
    dir_results = os.path.join(config_analysis.resultsdir, args.config_id)

    filenames = sorted(glob(os.path.join(dir_results, "data", f"kappa_zs{args.zs:.1f}.fits")))

    if not filenames:
        logging.error("No files found. Please check the directory and file naming conventions.")
        sys.exit(1)

    logging.info(f"Found {len(filenames)} files.")
    for f in filenames:
        logging.info(f)

    kappa_maps = KappaCodes(dir_results=dir_results, filenames=filenames, nside=config_analysis.nside, lmax=config_analysis.lmax)
    kappa_maps.readmaps_healpy(n2r=True)

    for i, map_i in enumerate(kappa_maps.mapbins):
        logging.info(f'Processing file: {os.path.basename(filenames[i])}')
        kappa_maps.run_PeaksMinima(map_i, i)
    
    logging.info("All done.")