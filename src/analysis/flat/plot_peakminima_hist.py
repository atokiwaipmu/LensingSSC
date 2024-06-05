import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

from ...masssheet.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to read data from the .npz file
def read_npz_data(file_path):
    data = np.load(file_path)
    peak_height = data['peak_height']
    minima_height = data['minima_height']
    return peak_height, minima_height

# Function to calculate histograms for each file and aggregate them
def aggregate_histograms(npz_files, num_bins=30):
    peak_histograms = []
    minima_histograms = []
    bin_edges = None

    for file_path in npz_files:
        peak_height, minima_height = read_npz_data(file_path)
        
        peak_hist, bin_edges = np.histogram(peak_height, bins=num_bins, range=(0, 1))
        minima_hist, _ = np.histogram(minima_height, bins=num_bins, range=(0, 1))
        
        peak_histograms.append(peak_hist)
        minima_histograms.append(minima_hist)
    
    peak_histograms = np.array(peak_histograms)
    minima_histograms = np.array(minima_histograms)
    
    peak_mean = np.mean(peak_histograms, axis=0)
    peak_std = np.std(peak_histograms, axis=0)
    minima_mean = np.mean(minima_histograms, axis=0)
    minima_std = np.std(minima_histograms, axis=0)
    
    return bin_edges, peak_mean, peak_std, minima_mean, minima_std


# Function to plot and save aggregated histograms with error bars
def plot_and_save_aggregated_histograms(bin_edges, peak_mean, peak_std, minima_mean, minima_std, save_dir, img_dir):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(12, 6))

    # Plot aggregated histogram for peak heights
    plt.subplot(1, 2, 1)
    plt.bar(bin_centers, peak_mean, width=bin_edges[1] - bin_edges[0], color='blue', alpha=0.75, label='Peak Heights')
    plt.errorbar(bin_centers, peak_mean, yerr=peak_std, fmt='o', color='black')
    plt.title('Aggregated Histogram of Peak Heights with Error Bars')
    plt.xlabel('Peak Height')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot aggregated histogram for minima heights
    plt.subplot(1, 2, 2)
    plt.bar(bin_centers, minima_mean, width=bin_edges[1] - bin_edges[0], color='red', alpha=0.75, label='Minima Heights')
    plt.errorbar(bin_centers, minima_mean, yerr=minima_std, fmt='o', color='black')
    plt.title('Aggregated Histogram of Minima Heights with Error Bars')
    plt.xlabel('Minima Height')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    
    # Construct the filename for the saved histogram
    save_filename = os.path.join(img_dir, f'histograms_peaksminima.png')
    plt.savefig(save_filename)
    plt.close()

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

    img_dir = os.path.join(config_analysis.imgdir, args.config_id, "kappa_patches_flat")

    # Gather all .npz files in the directory
    npz_files = glob.glob(os.path.join(save_dir, f'*s{args.sl}*.npz'))
    logging.info(f"Found {len(npz_files)} .npz files in the directory.")

    # Calculate and aggregate histograms
    bin_edges, peak_mean, peak_std, minima_mean, minima_std = aggregate_histograms(npz_files)

    # Plot and save aggregated histograms with error bars
    plot_and_save_aggregated_histograms(bin_edges, peak_mean, peak_std, minima_mean, minima_std, save_dir, img_dir)
