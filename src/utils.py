
import os
import argparse
import inspect
import yaml
from glob import glob
import numpy as np
import multiprocessing as mp

def compute_histogram_shared(data_chunk, bins, shared_hist, lock):
    hist, _ = np.histogram(data_chunk, bins=bins)
    with lock:
        for i in range(len(hist)):
            shared_hist[i] += hist[i]

def parallel_histogram(data, bins, num_processes=mp.cpu_count(), chunk_size=100000):
    data_chunks = np.array_split(data, len(data) // chunk_size)
    manager = mp.Manager()
    shared_hist = manager.list([0] * (len(bins) - 1))
    lock = manager.Lock()
    
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(compute_histogram_shared, [(chunk, bins, shared_hist, lock) for chunk in data_chunks])
    
    return np.array(shared_hist)

def find_data_dirs(workdir="/lustre/work/akira.tokiwa/Projects/LensingSSC/"):
    raw_dirs = sorted(glob(os.path.join(workdir, "data", "*", "*", "usmesh")))
    data_dirs = [os.path.dirname(d) for d in raw_dirs]
    return data_dirs
    
def filter_config(config, cls):
    params = inspect.signature(cls.__init__).parameters
    filtered_config = {k: v for k, v in config.items() if k in params}
    return filtered_config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Smooth kappa maps")
    parser.add_argument("datadir", type=str, help="Directory containing kappa maps")
    parser.add_argument("config_file", type=str, help="Configuration file")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing smoothed maps")
    return parser.parse_args()

def load_config(config_file):
    try:
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
        return config
    except yaml.YAMLError as exc:
        print(exc)
        exit(1)