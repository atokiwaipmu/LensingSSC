import os
import numpy as np
import healpy as hp
import multiprocessing as mp
from glob import glob
import pickle

from src.utils.ConfigData import ConfigAnalysis

def parse_file_name(file_name):
    file_name = os.path.basename(file_name)
    zs = float(file_name.split("_")[1][2:5])
    if "sl" in file_name:
        sl = file_name.split("_")[3][2:]
        if ".fits" in sl:
            sl = int(sl[:-5])
        else:
            sl = int(sl)
    else:
        sl = None
    if "noise" in file_name:
        survey = file_name.split("_")[5]
    else:
        survey = None
    return zs, sl, survey

def compute_stddev(data_path):
    kappa = hp.read_map(data_path)
    zs, sl, survey = parse_file_name(data_path)
    stddev = np.std(kappa)
    return (zs, sl, survey), stddev

def get_stddev(data_paths, save_path):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(compute_stddev, data_paths)
    stddev_dict = {result[0]: result[1] for result in results}
    
    # Save the stddev_dict to a file
    with open(save_path, "wb") as f:
        pickle.dump(stddev_dict, f)
    
    return 

if __name__ == "__main__":
    config_analysis_path = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json"
    config_analysis = ConfigAnalysis.from_json(config_analysis_path)

    data_paths_bigbox = glob(f"{config_analysis.resultsdir}/bigbox/data/kappa_zs*.fits")
    data_paths_tiled = glob(f"{config_analysis.resultsdir}/tiled/data/kappa_zs*.fits")

    data_paths_bigbox_smoothed = glob(f"{config_analysis.resultsdir}/bigbox/smoothed/sl=*/kappa_zs*.fits")
    data_paths_tiled_smoothed = glob(f"{config_analysis.resultsdir}/tiled/smoothed/sl=*/kappa_zs*.fits")

    save_path_bigbox = f"{config_analysis.resultsdir}/bigbox/stddev_dict.pkl"
    save_path_tiled = f"{config_analysis.resultsdir}/tiled/stddev_dict.pkl"

    get_stddev(data_paths_bigbox + data_paths_bigbox_smoothed, save_path_bigbox)
    get_stddev(data_paths_tiled + data_paths_tiled_smoothed, save_path_tiled)

