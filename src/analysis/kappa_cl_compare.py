
import os
import json
import logging
import argparse
from datetime import date

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from bigfile import File
from .calc_cl import calc_cl
from ..masssheet.ConfigData import ConfigData

logging.basicConfig(level=logging.INFO)

# Accuracy parameters
def set_lmax(nside):
    global LMAX
    LMAX = min([1000, nside])

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def compute_and_save_cl(data_dir, zs, recalculate=False, intg=False):
    """
    Compute and save power spectrum data if not already done.

    Parameters:
        data_path (str): Path to the simulation file.
        zs (float): Source redshift.
        recalculate (bool): Whether to force recalculation.

    Returns:
        tuple: Ell and Cl data from simulation.
    """
    cl_path = f'{data_dir}/kappa_cl_z{zs:.1f}.npz'
    if not os.path.isfile(cl_path) or recalculate:
        logging.info(f"Computing Cl for zs={zs:.1f}")
        data_path = f'{data_dir}/kappa.fits' if not intg else f'{data_dir}/kappa_int.fits'
        logging.info(f"Reading data from {data_path}")
        kappa = hp.read_map(data_path)
        kappa = hp.reorder(kappa, n2r=True)
        cl = hp.anafast(kappa, lmax=LMAX)
        logging.info(f"Saving Cl to {cl_path}")
        np.savez(cl_path, ell=np.arange(LMAX + 1), cl=cl)
    logging.info(f"Loading Cl from {cl_path}")
    data = np.load(cl_path)
    return data['ell'], data['cl']

def compute_and_save_cambcl(halofit_dir, zs, recalculate=False):
    """
    Compute and save CAMB power spectrum data if not already done.

    Parameters:
        halofit_path (str): Path to the CAMB power spectrum file.
        zs (float): Source redshift.
        recalculate (bool): Whether to force recalculation.

    Returns:
        tuple: Ell and Cl data from simulation.
    """
    cl_path = f'{halofit_dir}/kappa_cl_camb_z{zs:.1f}.npz'
    if not os.path.isfile(cl_path) or recalculate:
        logging.info(f"Computing CAMB Cl for zs={zs:.1f}")
        ell, clkk, clkk_lin = calc_cl(zs)
        os.makedirs(halofit_dir, exist_ok=True)
        logging.info(f"Saving CAMB Cl to {cl_path}")
        np.savez(cl_path, ell=ell, clkk=clkk, clkk_lin=clkk_lin)
    logging.info(f"Loading CAMB Cl from {cl_path}")
    data = np.load(cl_path)
    return data['ell'], data['clkk'], data['clkk_lin']

def plot_power_spectra(tiled_dir, bigbox_dir, halofit_dir, img_dir, zs):
    """
    Generate and save a plot of power spectra.

    Parameters:
        tiled_dir (str): Path to the tiled data directory.
        bigbox_dir (str): Path to the bigbox data directory.
        halofit_dir (str): Path to the halofit data directory.
        img_dir (str): Path to the image directory.
        zs (float): Source redshift.

    Returns:
        None
    """
    ell_tiled, cl_tiled = compute_and_save_cl(tiled_dir, zs)
    ell_bigbox, cl_bigbox = compute_and_save_cl(bigbox_dir, zs)
    ell, clkk, clkk_lin = compute_and_save_cambcl(halofit_dir, zs)

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
    ratio = cl_tiled / cl_bigbox
    ax_ratio.set_xscale('log')
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_xlabel(r'$\ell$', fontsize=14)
    ax_ratio.set_ylabel('Tiled/BigBox', fontsize=12)
    ax_ratio.plot(ell_bigbox, ratio, 'b-', lw=2)
    ax_ratio.grid(True, which='both', linestyle='--', linewidth=0.5)

    logging.info(f"Saving plot to {img_dir}/kappa_cl_z{zs:.1f}.png")
    fig.savefig(f'{img_dir}/kappa_cl_comparison_z{zs:.1f}.png', bbox_inches='tight')

def main():
    """
    Main function to process command-line arguments and initiate computation.
    """
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('--r4096', type=bool, default=False, help='Use 4096 resolution')
    args = parser.parse_args()

    config_dir = '/lustre/work/akira.tokiwa/Projects/LensingSSC/configs'
    data_config_file = os.path.join(config_dir,f'config_tiled_hp.json')
    logging.info(f"Reading data config from {data_config_file}")
    data_config = ConfigData.from_json(data_config_file)

    result_config_file = os.path.join(config_dir, 'config_analysis.json')
    logging.info(f"Reading result config from {result_config_file}")
    result_config = load_config(result_config_file)

    set_lmax(result_config.nside if args.r4096 else 4096)

    halofit_dir = f'{result_config["result_dir"]}/halofit'
    os.makedirs(halofit_dir, exist_ok=True)
    img_dir = f'/lustre/work/akira.tokiwa/Projects/LensingSSC/img'
    os.makedirs(img_dir, exist_ok=True)
    for zs in data_config.zs_list:
        tiled_dir = f'{result_config["result_dir"]}/tiled/zs-{zs:.1f}'
        bigbox_dir = f'{result_config["result_dir"]}/bigbox/zs-{zs:.1f}'
        plot_power_spectra(tiled_dir, bigbox_dir, halofit_dir, img_dir, zs)

if __name__ == '__main__':
    main()