
import os
from glob import glob
import logging

from src.analysis.kappamap import KappaCodes
from src.masssheet.ConfigData import ConfigData, ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config_analysis_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_analysis.json')
config_analysis = ConfigAnalysis.from_json(config_analysis_file)
dir_results = os.path.join(config_analysis.resultsdir, "tiled")
filenames = sorted(glob(os.path.join(dir_results, "smoothed", f"kappa_zs0.5_smoothed_s*.fits")))

kappa_maps = KappaCodes(dir_results, filenames, nside=config_analysis.nside, lmax=config_analysis.lmax)
kappa_maps.readmaps_healpy()

for i, map_i in enumerate(kappa_maps.mapbins):
    print(f'Processing {filenames[i]}')
    kappa_maps.run_PeaksMinima(map_i, i)