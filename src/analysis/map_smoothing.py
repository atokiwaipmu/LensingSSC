
import glob
import numpy as np
import healpy as hp
import os
import logging
import argparse
from pathlib import Path

from src.utils.ConfigData import ConfigAnalysis
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_map(mapfile: str, savefile: str, sl_arcmin: float, noise_map: str = None) -> None:
    try:
        sl_rad = sl_arcmin / 60 / 180 * np.pi
        kappa_masked = hp.read_map(mapfile)

        if noise_map is not None:
            noise = hp.read_map(noise_map)
            kappa_masked += noise

        kappa_masked = hp.ma(hp.reorder(kappa_masked, n2r=True))

        smoothed_map = hp.smoothing(kappa_masked, sigma=sl_rad)
        hp.write_map(savefile, smoothed_map, dtype=np.float32)
        logging.info(f"Processed and saved: {savefile}")
    except FileNotFoundError:
        logging.error(f"File not found: {mapfile}")
    except Exception as e:
        logging.error(f"Error processing {mapfile} with smoothing length {sl_arcmin}: {e}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id', type=str, help='Configuration identifier')
    parser.add_argument('zs', type=float, help='Source redshift')
    parser.add_argument('sl_arcmin', type=int, help='Smoothing length in arcmin')
    parser.add_argument('--noise_file', type=str, default=None, help='Noise file to add to the map')
    args = parser.parse_args()

    config_analysis_file = Path("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json")
    if not config_analysis_file.exists():
        logging.error(f"Configuration file not found: {config_analysis_file}")
        return

    config_analysis = ConfigAnalysis.from_json(config_analysis_file)
    dir_data = Path(config_analysis.resultsdir) / args.config_id / "data"
    filename = dir_data / f"kappa_zs{args.zs:.1f}.fits"
    if not filename.exists():
        logging.error(f"No files found for redshift {args.zs}.")
        return
    
    dir_smoothed = Path(config_analysis.resultsdir) / args.config_id / "smoothed" / f"sl={args.sl_arcmin}"
    dir_smoothed.mkdir(parents=True, exist_ok=True)

    if args.noise_file is not None:
        # Add the file name to the end of the smoothed_sl{args.sl_arcmin} filename
        filename_smoothed = dir_smoothed / filename.name.replace('.fits', f'_smoothed_sl{args.sl_arcmin}_' + Path(args.noise_file).stem + '.fits')
        if filename_smoothed.exists():
            logging.info(f"File already exists: {filename_smoothed}")
            return
    else:
        filename_smoothed = dir_smoothed / filename.name.replace('.fits', f'_smoothed_sl{args.sl_arcmin}.fits')
        if filename_smoothed.exists():
            logging.info(f"File already exists: {filename_smoothed}")
            return

    logging.info(f"Processing map: {filename}")
    process_map(str(filename), str(filename_smoothed), args.sl_arcmin, args.noise_file)
    logging.info(f"Processed and saved: {filename_smoothed}")
    
if __name__ == '__main__':
    main()
