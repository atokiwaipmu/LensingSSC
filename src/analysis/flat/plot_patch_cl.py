import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse

from ...masssheet.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_power_spectrum_data(directory):
    """
    Load ell and cl arrays from .npz files in the specified directory.

    Parameters:
    directory (str): Directory containing the .npz files.

    Returns:
    tuple: Two numpy arrays containing ell and cl data respectively.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    ell_values = []
    cl_values = []
    
    for file in files:
        try:
            data = np.load(os.path.join(directory, file))
            ell_values.append(data['ell'])
            cl_values.append(data['cl'])
        except IOError:
            logging.error(f"Error loading file {file}")
            continue
    
    return np.array(ell_values), np.array(cl_values)

def calculate_mean_and_covariance(cl_values):
    """
    Calculate the mean and covariance matrix of cl_values.

    Parameters:
    cl_values (array-like): List of cl arrays.

    Returns:
    tuple: Mean and covariance matrix of cl_values.
    """
    cl_mean = np.mean(cl_values, axis=0)
    cl_covariance = np.cov(cl_values, rowvar=False)
    return cl_mean, cl_covariance

def plot_power_spectrum_covariance(results_directory, image_directory, redshift, config_id, ell_values, 
                                   cl_mean, cl_covariance, comparison_config_id=None, comparison_cl_mean=None, comparison_cl_covariance=None):
    """
    Plot the power spectrum with error bars and save the figure.

    Parameters:
    results_directory (str): Directory containing the results.
    image_directory (str): Directory to save the plot images.
    redshift (float): Source redshift.
    config_id (str): Configuration identifier.
    ell_values (array-like): Array of ell values.
    cl_mean (array-like): Mean power spectrum values.
    cl_covariance (array-like): Covariance matrix of power spectrum values.
    comparison_config_id (str, optional): Comparison configuration identifier.
    comparison_cl_covariance (array-like, optional): Covariance matrix of the comparison power spectrum values.
    """
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05}, sharex=True)

    # Main plot
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$[\ell(\ell+1)/2\pi] C_\ell^\mathrm{kk}$', fontsize=14)

    ax.plot(ell_values, np.diag(cl_covariance), 'o', label=f"{config_id}"+ r"$C_\ell^{\kappa \kappa}$")
    if comparison_config_id and comparison_cl_covariance is not None:
        ax.plot(ell_values, np.diag(comparison_cl_covariance), 'o', label=f"{comparison_config_id}"+ r"$C_\ell^{\kappa \kappa}$")

    ax.set_xlabel('$\ell$', fontsize=14)
    #ax.set_ylim(3*1e-6, 3*1e-3)
    ax.set_xlim(100, 5000)

    ax.legend(frameon=True, loc='upper left', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_title('Power Spectrum, Redshift: {:.1f}'.format(redshift), fontsize=14)

    # Ratio plot
    if comparison_config_id and comparison_cl_covariance is not None:
        cov_ratio = np.diag(cl_covariance) / np.diag(comparison_cl_covariance)
        ax_ratio.plot(ell_values, cov_ratio, 'o', markersize=2, label='Diagonal term of Covariance')
        ax_ratio.set_ylabel(f'{config_id} / {comparison_config_id}', fontsize=14)

    ax_ratio.set_xscale('log')
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_xlabel(r'$\ell$', fontsize=14)
    ax_ratio.legend(frameon=True, loc='upper left', fontsize=10)
    ax_ratio.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the figure
    fig.savefig(os.path.join(image_directory, f'clkk_{config_id}_zs{redshift:.1f}.png'))
    logging.info(f"Plot saved to {os.path.join(image_directory, f'clkk_{config_id}_zs{redshift:.1f}.png')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id', type=str, help='Configuration identifier (tiled or bigbox)')
    parser.add_argument('redshift', type=float, help='Source redshift')
    parser.add_argument('--patch_size_deg', type=int, default=10, help='Size of each patch in degrees')
    parser.add_argument('--compare_with', type=str, help='Comparison configuration identifier (tiled or bigbox)', required=False)
    args = parser.parse_args()

    config_path = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json"
    config_analysis = ConfigAnalysis.from_json(config_path)

    results_directory = os.path.join(config_analysis.resultsdir, args.config_id, 'Clkk', "patch_flat")
    save_directory = os.path.join(results_directory, f"zs{args.redshift:.1f}")
    os.makedirs(save_directory, exist_ok=True)
    logging.info(f"Using directory: {save_directory}")

    image_directory = os.path.join(config_analysis.imgdir, args.config_id, "kappa_patches_flat")
    os.makedirs(image_directory, exist_ok=True)
    logging.info(f"Using directory: {image_directory}")

    ell_values, cl_values = load_power_spectrum_data(save_directory)
    cl_mean, cl_covariance = calculate_mean_and_covariance(cl_values)
    logging.info("Data shape: ell_values: {}, cl_values: {}".format(ell_values.shape, cl_values.shape))

    comparison_cl_covariance = None
    if args.compare_with:
        comparison_directory = os.path.join(config_analysis.resultsdir, args.compare_with, 'Clkk', "patch_flat")
        comparison_save_directory = os.path.join(comparison_directory, f"zs{args.redshift:.1f}")
        ell_comparison_values, cl_comparison_values = load_power_spectrum_data(comparison_save_directory)
        comparison_cl_mean, comparison_cl_covariance = calculate_mean_and_covariance(cl_comparison_values)
        logging.info("Comparison data shape: ell_values: {}, cl_values: {}".format(ell_comparison_values.shape, cl_comparison_values.shape))

    plot_power_spectrum_covariance(results_directory, image_directory, args.redshift, args.config_id, ell_values[0], 
                                   cl_mean, cl_covariance, args.compare_with, comparison_cl_mean, comparison_cl_covariance)
    logging.info("Plotting complete.")
