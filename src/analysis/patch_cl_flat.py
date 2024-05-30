import json
import numpy as np
import healpy as hp
import os
import sys
import logging
import argparse
from glob import glob
import matplotlib.pyplot as plt
from lenstools import ConvergenceMap

from .kappamap import KappaCodes
from ..masssheet.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def gnomonic_projection(kappa_map, center_ra, center_dec, xsize=800, reso=1.5):
    """
    Project a patch of the kappa map using gnomonic projection.

    Args:
        kappa_map (numpy.ndarray): The full-sky kappa map.
        center_ra (float): The right ascension of the center of the patch (degrees).
        center_dec (float): The declination of the center of the patch (degrees).
        xsize (int): The size of the patch in pixels.
        reso (float): The resolution of the patch in arcminutes.

    Returns:
        numpy.ndarray: The projected patch.
    """
    return hp.gnomview(kappa_map, rot=[center_ra, center_dec], xsize=xsize, reso=reso, return_projected_map=True)


def calculate_cl_flatsky(patch):
    """
    Calculate the power spectrum on a flat sky patch.

    Args:
        patch (numpy.ndarray): The flat sky patch.

    Returns:
        numpy.ndarray: The power spectrum Cl for the patch.
    """
    conv_map = ConvergenceMap(patch, angle=10.0)
    ell, cl = conv_map.powerSpectrum()
    return ell, cl

def save_kappa_patch_figure(patch, save_path, ra, dec):
    """
    Save a figure of the kappa patch.

    Args:
        patch (numpy.ndarray): The kappa patch.
        save_path (str): The path to save the figure.
        ra (float): The right ascension of the center of the patch.
        dec (float): The declination of the center of the patch.

    Returns:
        None
    """
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(patch, origin='lower')
    plt.colorbar(label='Kappa')
    plt.title(f'Kappa Patch at RA={ra}°, DEC={dec}°')
    

    plt.figure()
    plt.imshow(patch, origin='lower')
    plt.colorbar(label='Kappa')
    plt.title(f'Kappa Patch at RA={ra}°, DEC={dec}°')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved kappa patch figure to {save_path}")


def process_kappa_map(save_dir, file_path, nside=8192, lmax=5000, base_pix=1048576, patch_size_deg=10):
    """
    Main function to process the kappa map and calculate Cl for each patch.

    Args:
        save_dir (str): The directory to save the results.
        file_path (str): The path to the kappa map file.
        nside (int): The nside parameter for HEALPix.
        lmax (int): The maximum multipole to compute.
        base_pix (int): The base pixel size for processing.
        patch_size_deg (int): The size of each patch in degrees.

    Returns:
        None
    """
    logging.info("Starting the kappa maps processing.")
    kappa = hp.read_map(file_path)

    patches = []
    clkk = []

    for ra in range(0, 360, patch_size_deg):
        for dec in range(-40, 40, patch_size_deg):
            patch = gnomonic_projection(kappa, center_ra=ra, center_dec=dec, xsize=800, reso=patch_size_deg*60/800)
            patches.append((ra, dec, patch))

    for ra, dec, patch in patches:
        ell, cl = calculate_cl_flatsky(patch)
        clkk.append((ra, dec, ell, cl))

        # Save kappa patch figure
        fig_save_path = os.path.join(save_dir, f'kappa_patch_RA{ra}_DEC{dec}.png')
        save_kappa_patch_figure(patch, fig_save_path, ra, dec)

    save_filename = os.path.join(save_dir, os.path.basename(file_path).replace('.fits', f'_Clkk_ell_{nside}_{patch_size_deg}deg.npz'))
    np.savez(save_filename, clkk=clkk, nside=nside, lmax=lmax)
    logging.info(f"Saved the results to {save_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id', type=str, help='Configuration identifier')
    parser.add_argument('zs', type=float, help='Source redshift')
    parser.add_argument('--base_pix', type=int, default=1024, help='Base pixel size for processing')
    parser.add_argument('--patch_size_deg', type=int, default=10, help='Size of each patch in degrees')
    args = parser.parse_args()

    config_path = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json"
    config_analysis = ConfigAnalysis.from_json(config_path)

    results_dir = os.path.join(config_analysis.resultsdir, args.config_id)
    logging.info(f"Using directory: {results_dir}")

    kappa_file = os.path.join(results_dir, "data", f"kappa_zs{args.zs:.1f}.fits")
    logging.info(f"Using file: {kappa_file}")

    process_kappa_map(results_dir, kappa_file, nside=config_analysis.nside, lmax=config_analysis.lmax, base_pix=args.base_pix**2, patch_size_deg=args.patch_size_deg)
    logging.info("Processing complete.")
