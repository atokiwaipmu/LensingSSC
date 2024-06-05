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
from astropy import units as u

from ..kappamap import KappaCodes
from ...masssheet.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def gnomonic_projection(kappa_map, center_ra, center_dec, xsize=800, reso=1.5):
    return hp.gnomview(kappa_map, rot=[center_ra, center_dec], xsize=xsize, reso=reso, return_projected_map=True)


def calculate_peaks_minima(patch, angle):
    conv_map = ConvergenceMap(patch, angle=angle * u.deg)
    l_edges = np.arange(-0.01, 0.06, 0.002)
    peak_height,peak_positions = conv_map.locatePeaks(l_edges)

    conv_map_minus = ConvergenceMap(-patch, angle=angle * u.deg)
    minima_height,minima_positions = conv_map_minus.locatePeaks(l_edges)

    return peak_height, peak_positions, minima_height, minima_positions

def save_kappa_patch_figure(patch, save_path, ra, dec, patch_size_deg, peak_positions, minima_positions):
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
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create meshgrid for the x and y axis
    x = np.linspace(-patch_size_deg/2, patch_size_deg/2, patch.shape[1])
    y = np.linspace(-patch_size_deg/2, patch_size_deg/2, patch.shape[0])
    X, Y = np.meshgrid(x, y)

    # Display the image using pcolormesh
    cax = ax.pcolormesh(X, Y, patch, shading='auto')

    # circle the peaks and minima
    for peak in peak_positions.value:
        ax.plot(peak[1]-patch_size_deg/2, peak[0]-patch_size_deg/2, 'ro', markersize=1)
    #for minima in minima_positions.value:
    #    ax.plot(minima[1]-patch_size_deg/2, minima[0]-patch_size_deg/2, 'bo', markersize=1)

    # Add a colorbar with a label
    cbar = fig.colorbar(cax, ax=ax, label='Kappa')

    # Set the title of the plot
    ax.set_title(f'Kappa Patch at RA={ra}°, DEC={dec}°')

    # Label the axes
    ax.set_xlabel('Degrees from center RA')
    ax.set_ylabel('Degrees from center DEC')

    # Save the figure
    fig.savefig(save_path)

    # Close the figure to free up memory
    plt.close(fig)

    # Log the save action
    logging.info(f"Saved kappa patch figure to {save_path}")


def process_kappa_map(save_dir, img_dir, file_path, patch_size_deg=10):
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
    kappa = hp.reorder(kappa, n2r=True)

    for ra in range(0, 360, patch_size_deg):
        for dec in range(-30, 40, patch_size_deg):
            patch = gnomonic_projection(kappa, center_ra=ra, center_dec=dec, xsize=800, reso=patch_size_deg*60/800)
            save_path = f"{img_dir}/kappa_patch_ra{ra}_dec{dec}.png"

            peak_height, peak_positions, minima_height, minima_positions = calculate_peaks_minima(patch, angle=patch_size_deg)


            if ra == 0 and dec == 0:
                save_kappa_patch_figure(patch, save_path, ra, dec, patch_size_deg, peak_positions, minima_positions)

            save_filename = os.path.join(save_dir, os.path.basename(file_path).replace('.fits', 
                                f'_peaksminima_{patch_size_deg}deg_ra{ra}_dec{dec}.npz'))
            np.savez(save_filename, peak_height=peak_height, peak_positions=peak_positions, minima_height=minima_height, minima_positions=minima_positions)
            logging.info(f"Saved the results to {save_filename}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id', type=str, help='Configuration identifier')
    parser.add_argument('zs', type=float, help='Source redshift')
    parser.add_argument('sl', type=int, help='Smoothing length')
    parser.add_argument('--patch_size_deg', type=int, default=10, help='Size of each patch in degrees')
    args = parser.parse_args()

    config_path = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json"
    config_analysis = ConfigAnalysis.from_json(config_path)

    results_dir = os.path.join(config_analysis.resultsdir, args.config_id)
    save_dir = os.path.join(results_dir, "peakminima", "patch_flat", f"zs{args.zs:.1f}")
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Using directory: {save_dir}")

    img_dir = os.path.join(config_analysis.imgdir, args.config_id, "kappa_patches_flat")
    os.makedirs(img_dir, exist_ok=True)
    logging.info(f"Using directory: {img_dir}")

    kappa_file = os.path.join(results_dir, "smoothed", f"kappa_zs{args.zs:.1f}_smoothed_s{args.sl}.fits")
    logging.info(f"Using file: {kappa_file}")

    process_kappa_map(save_dir, img_dir, kappa_file, patch_size_deg=args.patch_size_deg)
    logging.info("Processing complete.")
