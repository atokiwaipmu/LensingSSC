
import os

import numpy as np
from matplotlib import pyplot as plt

from ..masssheet.ConfigData import ConfigAnalysis, ConfigData

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_power_spectra(results_dir, img_dir, zs):
    """
    Generate and save a plot of power spectra.

    Parameters:
        file_path (str): Path to the data directory.
        zs (float): Source redshift.

    Returns:
        None
    """
    bigbox_dir = os.path.join(results_dir, 'bigbox', 'Clkk')
    ell_bigbox, cl_bigbox = np.load(f'{bigbox_dir}/kappa_zs{zs:.1f}_Clkk_ell_0_5000.npz').values()

    tiled_dir = os.path.join(results_dir, 'tiled', 'Clkk')
    ell_tiled, cl_tiled = np.load(f'{tiled_dir}/kappa_zs{zs:.1f}_Clkk_ell_0_5000.npz').values()

    halofit_dir = os.path.join(results_dir, 'halofit')
    ell, clkk, clkk_lin = np.load(f'{halofit_dir}/kappa_zs{zs:.1f}_Clkk_ell_0_5000.npz').values()

    # Create a figure and a set of subplots with custom height ratios
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05}, sharex=True)

    # Main plot
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$[\ell(\ell+1)/2\pi] C_\ell^\mathrm{kk}$', fontsize=14)

    ax.plot(ell, clkk_lin * ell * (ell + 1) / 2. / np.pi, 'c--', lw=2, alpha=0.7, label='Linear')
    ax.plot(ell, clkk * ell * (ell + 1) / 2. / np.pi, 'g-', lw=2, alpha=0.7, label='Halofit')
    ax.plot(ell_bigbox, cl_bigbox * ell_bigbox * (ell_bigbox + 1) / 2. / np.pi, 'k-', lw=2, alpha=0.8, label='BigBox')
    ax.plot(ell_tiled, cl_tiled * ell_tiled * (ell_tiled + 1) / 2. / np.pi, 'r-', lw=2, alpha=0.8, label='Tiled')

    ax.legend(frameon=True, loc='upper left', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_title(f'Power Spectrum Comparison on source redshift={zs:.1f}', fontsize=16)

    # Ratio plot
    #ratio = cl_tiled / cl_bigbox
    ratio1 = cl_tiled[1:-1] / clkk
    ratio2 = cl_bigbox[1:-1] / clkk
    ax_ratio.set_xscale('log')
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_xlabel(r'$\ell$', fontsize=14)
    #ax_ratio.set_ylabel('Tiled/BigBox', fontsize=12)
    ax_ratio.plot(ell, ratio1, 'r-', lw=2, alpha=0.8, label='Tiled/Halofit')
    ax_ratio.plot(ell, ratio2, 'b-', lw=2, alpha=0.8, label='BigBox/Halofit')
    ax_ratio.grid(True, which='both', linestyle='--', linewidth=0.5)

    logging.info(f"Saving plot to {img_dir}/kappa_cl_z{zs:.1f}.png")
    fig.savefig(f'{img_dir}/kappa_cl_comparison_z{zs:.1f}_.png', bbox_inches='tight')

def main():
    config_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_data.json')
    config = ConfigData.from_json(config_file)

    config_analysis_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_analysis.json')
    config_analysis = ConfigAnalysis.from_json(config_analysis_file)

    img_dir = os.path.join(config_analysis.imgdir, 'comparison')
    os.makedirs(img_dir, exist_ok=True)

    for zs in config.zs_list:
        logging.info(f"Plotting power spectra for zs={zs}")
        plot_power_spectra(config_analysis.resultsdir, img_dir, zs)

if __name__ == "__main__":
    main()