
import argparse
import inspect
import yaml
import numpy as np
import multiprocessing as mp
import logging
from pathlib import Path
from typing import List, Tuple 

from src.info_extractor import InfoExtractor

def compute_histogram_shared(data_chunk, bins, shared_hist, lock):
    hist, _ = np.histogram(data_chunk, bins=bins)
    with lock:
        for i in range(len(hist)):
            shared_hist[i] += hist[i]

def parallel_histogram(data, bins, num_processes=mp.cpu_count(), chunk_size=10000):
    if chunk_size <= 0:
        chunk_size = 1  # デフォルトの値を設定

    data_chunks = np.array_split(data, max(1, len(data) // chunk_size))
    manager = mp.Manager()
    shared_hist = manager.list([0] * (len(bins) - 1))
    lock = manager.Lock()
    
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(compute_histogram_shared, [(chunk, bins, shared_hist, lock) for chunk in data_chunks])
    
    return np.array(shared_hist)


def find_data_dirs(
    workdir: Path = Path("/lustre/work/akira.tokiwa/Projects/LensingSSC/"),
    data_subpath: str = "data/*/*/usmesh"
) -> List[Path]:
    """
    Searches for data directories within the specified working directory.

    The function looks for directories matching the pattern:
    {workdir}/data/*/*/usmesh, sorts them, and returns the parent directories.

    Args:
        workdir (Path, optional): The base working directory to search within.
            Defaults to Path("/lustre/work/akira.tokiwa/Projects/LensingSSC/").
        data_subpath (str, optional): The glob pattern to locate 'usmesh' directories.
            Defaults to "data/*/*/usmesh".

    Returns:
        List[Path]: A sorted list of data directories containing 'usmesh'.
    """
    logging.debug(f"Searching for data directories in: {workdir / data_subpath}")
    raw_dirs = sorted(workdir.glob(data_subpath))
    data_dirs = [raw_dir.parent for raw_dir in raw_dirs if raw_dir.is_dir()]

    logging.info(f"Found {len(data_dirs)} data directories.")
    return data_dirs

def find_kappa_files(datadir: Path) -> List[Path]:
    """
    Find all .fits kappa files within the 'kappa' subdirectory of the given data directory.

    Args:
        datadir (Path): The data directory to search within.

    Returns:
        List[Path]: A list of Paths to kappa .fits files.
    """
    kappa_dir = datadir / "kappa"
    if not kappa_dir.exists():
        logging.warning(f"'kappa' directory not found in {datadir}. Skipping this directory.")
        return []
    kappa_files = list(kappa_dir.glob("*.fits"))
    logging.debug(f"Found {len(kappa_files)} kappa files in {kappa_dir}.")
    return kappa_files

def separate_dirs(data_dirs: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Separates data directories into 'tiled' and 'bigbox' based on their box type.

    Utilizes the InfoExtractor to determine the box type of each directory.

    Args:
        data_dirs (List[Path]): A list of data directories to separate.

    Returns:
        Tuple[List[Path], List[Path]]:
            - List of directories classified as 'tiled'.
            - List of directories classified as 'bigbox'.
    """
    tiled_dirs = []
    bigbox_dirs = []

    logging.debug(f"Separating {len(data_dirs)} data directories into 'tiled' and 'bigbox'.")

    for data_dir in data_dirs:
        info = InfoExtractor.extract_info_from_path(data_dir)
        box_type = info.get('box_type')

        if box_type == 'tiled':
            tiled_dirs.append(data_dir)
            logging.debug(f"Directory {data_dir} classified as 'tiled'.")
        elif box_type == 'bigbox':
            bigbox_dirs.append(data_dir)
            logging.debug(f"Directory {data_dir} classified as 'bigbox'.")
        else:
            logging.warning(f"Directory {data_dir} has unrecognized box type: {box_type}.")

    logging.info(f"Separated into {len(tiled_dirs)} 'tiled' and {len(bigbox_dirs)} 'bigbox' directories.")
    return tiled_dirs, bigbox_dirs

def filter_paths(paths: List[Path], input_path: Path):
    """
    Filters paths based on noise, redshift, and seed information.

    Args:
        paths (List[Path]): The list of paths to filter.
        info (dict): Extracted information containing redshift and seed.
        noise (str): The noise identifier to filter paths.

    Returns:
        List[Path]: The filtered list of paths.
    """
    info = InfoExtractor.extract_info_from_path(input_path)
    redshift_str = f"zs{info.get('redshift')}"
    seed_str = f"s{info.get('seed')}"
    noise = info.get('noise')
    noise = "noiseless" if noise==0 else f"ngal{noise}"
    filtered = [
        path for path in paths
        if noise in path.name and redshift_str in path.name and seed_str in path.name
    ]

    logging.debug(f"Filtered {len(filtered)} paths with noise='{noise}', redshift='{redshift_str}', seed='{seed_str}'.")
    return filtered

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

def setup_logging():
    """Configure the logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
