
import os
import logging
import argparse
from glob import glob
import multiprocessing as mp

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from src.utils.compute_sigma import parse_file_name
from src.utils.ConfigData import ConfigAnalysis

def plot_full_covariance():
    # Define the labels for each statistics block
    labels = [r'$C^{\kappa\kappa}_{\ell}$', r'$B(\ell, \ell, \ell)$', r'$B(\ell, \ell, \ell/2)$', r'$B(\ell, \ell, \sim0)$', 'PDF']

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # Add labels and title
    plt.title(r"$r_{bigbox}$"+f", z={zs_list[0]}", fontsize=12)

    # Define tick positions and labels
    tick_positions = [7 + 14 * i for i in range(5)]  # Center positions of each 14x14 block

    plt.xticks(tick_positions, labels, fontsize=12)
    plt.yticks(tick_positions, labels, fontsize=12, rotation=90, ha='right')

    # Add grid lines to separate statistics
    num_blocks = 70 // 14  # There are 5 blocks
    for i in range(1, num_blocks):
        plt.axhline(y=i * 14 - 0.5, color='black', linewidth=2)
        plt.axvline(x=i * 14 - 0.5, color='black', linewidth=2)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    config_analysis_path = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json"
    config_analysis = ConfigAnalysis.from_json(config_analysis_path)

    data_paths_bigbox = glob(f"{config_analysis.resultsdir}/bigbox/data/kappa_zs*.fits")
    data_paths_tiled = glob(f"{config_analysis.resultsdir}/tiled/data/kappa_zs*.fits")

    data_paths_bigbox_smoothed = glob(f"{config_analysis.resultsdir}/bigbox/smoothed/sl=*/kappa_zs*.fits")
    data_paths_tiled_smoothed = glob(f"{config_analysis.resultsdir}/tiled/smoothed/sl=*/kappa_zs*.fits")

    dir_results_bigbox = os.path.join(config_analysis.resultsdir, "bigbox")
    dir_results_tiled = os.path.join(config_analysis.resultsdir, "tiled")