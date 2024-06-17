
import numpy as np
import healpy as hp
import os
import logging
import argparse
from pathlib import Path

from src.utils.ConfigData import ConfigAnalysis
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_noise_map(nside: int, ngal: int, sigma_ell: float = 0.4, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    npix = hp.nside2npix(nside)
    area = hp.nside2pixarea(nside, degrees=True)
    # convert area to arcmin^2
    area = area * 3600
    std_per_pixel = np.sqrt(sigma_ell ** 2 / (2 * ngal * area))
    noise_map = np.random.normal(0, std_per_pixel, npix)
    return noise_map

def main() -> None:
    config_analysis_file = Path("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json")
    config_analysis = ConfigAnalysis.from_json(config_analysis_file)

    seed = 100
    ngals = [7, 15, 30, 50]
    survey_names = ["DES-KiDS", "HSC", "Euclid-LSST", "Roman"]
    for ngal, survey_name in zip(ngals, survey_names):
        dir_noise = Path(config_analysis.resultsdir) / "noise" 
        dir_noise.mkdir(parents=True, exist_ok=True)

        filename_noise = dir_noise / f"noise_{survey_name}_seed{seed}.fits"
        if filename_noise.exists():
            logging.info(f"File already exists: {filename_noise}")
            continue

        logging.info(f"Generating noise map: {filename_noise}")
        noise_map = generate_noise_map(config_analysis.nside, ngal, 0.4, seed)
        hp.write_map(str(filename_noise), noise_map, overwrite=True, dtype=np.float32)
        logging.info(f"Generated and saved: {filename_noise}")

if __name__ == '__main__':
    main()