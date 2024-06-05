
import numpy as np
import healpy as hp
import os
import logging
import argparse

from src.utils.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_clkk(patch, start, end, nside, lmax=5000):
    """
    Calculate the power spectrum (Cl) for a patch of the kappa map.

    Args:
        patch (numpy.ndarray): The patch of the kappa map to process.
        start (int): The starting index of the patch.
        end (int): The ending index of the patch.
        nside (int): The nside parameter for HEALPix.
        lmax (int): The maximum multipole to compute.

    Returns:
        numpy.ndarray: The power spectrum Cl for the patch.
    """
    logging.info(f"Processing patch from {start} to {end}, {end - start} elements")

    if len(patch) != end - start:
        raise ValueError(f"Invalid patch size: {len(patch)} != {end - start}")
    if start < 0 or end > hp.nside2npix(nside):
        raise ValueError(f"Invalid indices: {start}, {end}")
    if start >= end:
        raise ValueError(f"Invalid indices: {start}, {end}")

    full_patch = np.zeros(hp.nside2npix(nside))
    full_patch[start:end] = patch
    full_patch = hp.ma(full_patch)
    full_patch.mask = np.zeros(full_patch.size, dtype=bool)
    full_patch.mask[:start] = True
    full_patch.mask[end:] = True

    full_patch = hp.reorder(full_patch, n2r=True)
    cl = hp.anafast(full_patch.filled(), lmax=lmax)
    return cl


def process_chunk(chunk_info):
    """
    Worker function for multiprocessing. Processes a chunk of the kappa map.

    Args:
        chunk_info (tuple): A tuple containing the chunk, start index, end index, nside, and lmax.

    Returns:
        numpy.ndarray: The power spectrum Cl for the chunk.
    """
    chunk, start, end, nside, lmax = chunk_info
    return calculate_clkk(chunk, start, end, nside, lmax)


def process_kappa_map(save_dir, file_path, nside=8192, lmax=5000, base_pix=1048576):
    """
    Main function to process the kappa map and calculate Cl for each patch.

    Args:
        save_dir (str): The directory to save the results.
        file_path (str): The path to the kappa map file.
        nside (int): The nside parameter for HEALPix.
        lmax (int): The maximum multipole to compute.
        base_pix (int): The base pixel size for processing.

    Returns:
        None
    """
    logging.info("Starting the kappa maps processing.")
    kappa = hp.read_map(file_path)

    ell = np.arange(lmax + 1)
    for i in range(0, len(kappa), base_pix):
        save_filename = os.path.join(save_dir, os.path.basename(file_path).replace('.fits', 
            f'_Clkk_ell_0_{lmax}_{nside}_{int(np.sqrt(base_pix))}x{int(np.sqrt(base_pix))}_chunk_{i}.npz'))
        if os.path.exists(save_filename):
            logging.info(f"Skipping chunk {i}, already processed.")
            continue

        start = i
        end = i + base_pix
        chunk = kappa[start:end]
        cl = calculate_clkk(chunk, start, end, nside, lmax)
        np.savez(save_filename, ell=ell, clkk=cl, nside=nside, lmax=lmax)
        logging.info(f"Saved Clkk to {save_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id', type=str, help='Configuration identifier')
    parser.add_argument('zs', type=float, help='Source redshift')
    parser.add_argument('--base_pix', type=int, default=1024, help='Base pixel size for processing')
    args = parser.parse_args()

    config_path = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json"
    config_analysis = ConfigAnalysis.from_json(config_path)

    results_dir = os.path.join(config_analysis.resultsdir, args.config_id)
    save_dir = os.path.join(results_dir, "Clkk", "patch", f"zs{args.zs:.1f}")
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Using directory: {save_dir}")

    kappa_file = os.path.join(results_dir, "data", f"kappa_zs{args.zs:.1f}.fits")
    logging.info(f"Using file: {kappa_file}")


    process_kappa_map(results_dir, kappa_file, nside=config_analysis.nside, lmax=config_analysis.lmax, base_pix=args.base_pix**2)
    logging.info("Processing complete.")
