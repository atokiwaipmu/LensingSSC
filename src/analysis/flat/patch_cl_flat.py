import argparse
import logging
import os
from glob import glob

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from lenstools import ConvergenceMap

from ...masssheet.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def project_gnomonic(kappa_map, center_ra, center_dec, xsize=800, resolution=1.5):
    """
    Project a patch of the kappa map using gnomonic projection.

    Args:
        kappa_map (numpy.ndarray): The full-sky kappa map.
        center_ra (float): The right ascension of the center of the patch (degrees).
        center_dec (float): The declination of the center of the patch (degrees).
        xsize (int): The size of the patch in pixels.
        resolution (float): The resolution of the patch in arcminutes.

    Returns:
        numpy.ndarray: The projected patch.
    """
    return hp.gnomview(kappa_map, rot=[center_ra, center_dec], xsize=xsize, reso=resolution, return_projected_map=True)

def compute_power_spectrum(patch, angle, lmax=5000, lmin=100):
    """
    Calculate the power spectrum on a flat sky patch.

    Args:
        patch (numpy.ndarray): The flat sky patch.
        angle (float): The angular size of the patch in degrees.
        lmax (int): The maximum multipole to compute.
        lmin (int): The minimum multipole to compute.

    Returns:
        numpy.ndarray: The power spectrum Cl for the patch.
    """
    convergence_map = ConvergenceMap(patch, angle=angle * u.deg)
    multipole_edges = np.arange(lmin, lmax + 1, 100)
    ell, cl = convergence_map.powerSpectrum(multipole_edges)
    return ell, cl

def save_kappa_patch_image(patch, save_path, ra, dec, patch_size_deg):
    """
    Save a figure of the kappa patch.

    Args:
        patch (numpy.ndarray): The kappa patch.
        save_path (str): The path to save the figure.
        ra (float): The right ascension of the center of the patch.
        dec (float): The declination of the center of the patch.
        patch_size_deg (int): The size of each patch in degrees.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    x = np.linspace(-patch_size_deg/2, patch_size_deg/2, patch.shape[1])
    y = np.linspace(-patch_size_deg/2, patch_size_deg/2, patch.shape[0])
    X, Y = np.meshgrid(x, y)
    cax = ax.pcolormesh(X, Y, patch, shading='auto')
    cbar = fig.colorbar(cax, ax=ax, label='Kappa')
    ax.set_title(f'Kappa Patch at RA={ra}°, DEC={dec}°')
    ax.set_xlabel('Degrees from center RA')
    ax.set_ylabel('Degrees from center DEC')
    fig.savefig(save_path)
    plt.close(fig)
    logging.info(f"Saved kappa patch figure to {save_path}")

def process_kappa_map(save_directory, image_directory, kappa_map_path, nside=8192, lmax=5000, patch_size_deg=10):
    """
    Main function to process the kappa map and calculate Cl for each patch.

    Args:
        save_directory (str): The directory to save the results.
        image_directory (str): The directory to save the kappa patch images.
        kappa_map_path (str): The path to the kappa map file.
        nside (int): The nside parameter for HEALPix.
        lmax (int): The maximum multipole to compute.
        patch_size_deg (int): The size of each patch in degrees.

    Returns:
        None
    """
    logging.info("Starting the kappa maps processing.")
    kappa_map = hp.read_map(kappa_map_path)
    kappa_map = hp.reorder(kappa_map, n2r=True)

    for ra in range(0, 360, patch_size_deg):
        for dec in range(-30, 40, patch_size_deg):
            patch = project_gnomonic(kappa_map, center_ra=ra, center_dec=dec, xsize=800, resolution=patch_size_deg*60/800)
            image_save_path = f"{image_directory}/kappa_patch_ra{ra}_dec{dec}.png"

            if ra == 0 and dec == 0:
                save_kappa_patch_image(patch, image_save_path, ra, dec, patch_size_deg)

            ell, cl = compute_power_spectrum(patch, angle=patch_size_deg, lmax=lmax)
            save_filename = os.path.join(save_directory, os.path.basename(kappa_map_path).replace('.fits', 
                                f'_Clkk_ell_{nside}_{lmax}_ra{ra}_dec{dec}_patch{patch_size_deg}.npz'))
            np.savez(save_filename, ell=ell, cl=cl)
            logging.info(f"Saved the results to {save_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id', type=str, help='Configuration identifier')
    parser.add_argument('source_redshift', type=float, help='Source redshift')
    parser.add_argument('--patch_size_deg', type=int, default=10, help='Size of each patch in degrees')
    args = parser.parse_args()

    config_path = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json"
    config_analysis = ConfigAnalysis.from_json(config_path)

    results_directory = os.path.join(config_analysis.resultsdir, args.config_id)
    save_directory = os.path.join(results_directory, "Clkk", "patch_flat", f"zs{args.source_redshift:.1f}")
    os.makedirs(save_directory, exist_ok=True)
    logging.info(f"Using directory: {save_directory}")

    image_directory = os.path.join(config_analysis.imgdir, args.config_id, "kappa_patches_flat")
    os.makedirs(image_directory, exist_ok=True)
    logging.info(f"Using directory: {image_directory}")

    kappa_map_file = os.path.join(results_directory, "data", f"kappa_zs{args.source_redshift:.1f}.fits")
    logging.info(f"Using file: {kappa_map_file}")

    process_kappa_map(save_directory, image_directory, kappa_map_file, nside=config_analysis.nside, lmax=config_analysis.lmax, patch_size_deg=args.patch_size_deg)
    logging.info("Processing complete.")
