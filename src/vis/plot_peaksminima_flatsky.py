import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from glob import glob
from src.utils.ConfigData import ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(files):
    data = []
    for filename in files:
        data.append(np.load(filename))
    return data

def calculate_bin_counts(data, edges):
    peak_bin_counts = np.zeros((len(data), len(edges) - 1))
    minima_bin_counts = np.zeros((len(data), len(edges) - 1))
    
    for i, d in enumerate(data):
        peak_bin_counts[i], _ = np.histogram(d['peak_height'], bins=edges)
        minima_bin_counts[i], _ = np.histogram(d['minima_height'], bins=edges)
        
    return peak_bin_counts, minima_bin_counts

def compute_mean_and_std(bin_counts):
    mean_counts = np.mean(bin_counts, axis=0)
    std_counts = np.std(bin_counts, axis=0)
    return mean_counts, std_counts

def compute_covariance(bin_counts):
    covariance_matrix = np.cov(bin_counts, rowvar=False)
    diagonal_terms = np.diag(covariance_matrix)
    return diagonal_terms

def plot_and_save(data1, data2, config_id1, config_id2, save_path):
    edges = np.arange(-0.04, 0.04, 0.002)
    peak_bin_counts1, minima_bin_counts1 = calculate_bin_counts(data1, edges)
    peak_bin_counts2, minima_bin_counts2 = calculate_bin_counts(data2, edges)

    # normalize by the sum
    peak_bin_counts1 = peak_bin_counts1 / np.sum(peak_bin_counts1, axis=1)[:, None]
    minima_bin_counts1 = minima_bin_counts1 / np.sum(minima_bin_counts1, axis=1)[:, None]

    peak_bin_counts2 = peak_bin_counts2 / np.sum(peak_bin_counts2, axis=1)[:, None]
    minima_bin_counts2 = minima_bin_counts2 / np.sum(minima_bin_counts2, axis=1)[:, None]
    
    mean_peak_counts1, std_peak_counts1 = compute_mean_and_std(peak_bin_counts1)
    mean_minima_counts1, std_minima_counts1 = compute_mean_and_std(minima_bin_counts1)
    
    mean_peak_counts2, std_peak_counts2 = compute_mean_and_std(peak_bin_counts2)
    mean_minima_counts2, std_minima_counts2 = compute_mean_and_std(minima_bin_counts2)
    
    peak_diagonal_terms1 = compute_covariance(peak_bin_counts1)
    peak_diagonal_terms2 = compute_covariance(peak_bin_counts2)

    minima_diagonal_terms1 = compute_covariance(minima_bin_counts1)
    minima_diagonal_terms2 = compute_covariance(minima_bin_counts2)
    
    # Compute the ratio of the diagonal terms
    ratio_peak_diagonal_terms = peak_diagonal_terms1 / peak_diagonal_terms2
    ratio_minima_diagonal_terms = minima_diagonal_terms1 / minima_diagonal_terms2
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 20))
    
    # Peaks histogram
    ax1.hist(edges[:-1], edges, weights=mean_peak_counts1, alpha=0.5)
    ax1.hist(edges[:-1], edges, weights=mean_peak_counts2, alpha=0.5)
    ax1.errorbar(edges[:-1]+0.001, mean_peak_counts1, yerr=std_peak_counts1, fmt='o', label=f'Peak Heights {config_id1}')
    ax1.errorbar(edges[:-1]+0.001, mean_peak_counts2, yerr=std_peak_counts2, fmt='o', label=f'Peak Heights {config_id2}')
    ax1.set_xlim(-0.04, 0.04)
    ax1.set_title('Mean and Std of Bin Counts for Peak Heights')
    ax1.set_xlabel('Height')
    ax1.set_ylabel('Mean Bin Count')
    ax1.legend()
    
    # Minima histogram
    ax2.hist(edges[:-1], edges, weights=mean_minima_counts1, alpha=0.5)
    ax2.hist(edges[:-1], edges, weights=mean_minima_counts2, alpha=0.5)
    ax2.errorbar(edges[:-1]+0.001, mean_minima_counts1, yerr=std_minima_counts1, fmt='o', label=f'Minima Heights {config_id1}')
    ax2.errorbar(edges[:-1]+0.001, mean_minima_counts2, yerr=std_minima_counts2, fmt='o', label=f'Minima Heights {config_id2}')
    ax2.set_xlim(-0.04, 0.04)
    ax2.set_title('Mean and Std of Bin Counts for Minima Heights')
    ax2.set_xlabel('Height')
    ax2.set_ylabel('Mean Bin Count')
    ax2.legend()

    # Covariance plot
    ax3.plot(edges[:-1], peak_diagonal_terms1, label=f'Peak Covariance Diagonal Terms ({config_id1})')
    ax3.plot(edges[:-1], peak_diagonal_terms2, label=f'Peak Covariance Diagonal Terms ({config_id2})')
    ax3.plot(edges[:-1], minima_diagonal_terms1, label=f'Minima Covariance Diagonal Terms ({config_id1})')
    ax3.plot(edges[:-1], minima_diagonal_terms2, label=f'Minima Covariance Diagonal Terms ({config_id2})')
    ax3.set_xlim(-0.04, 0.04)
    ax3.set_title('Diagonal Terms of Covariance Matrix')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Covariance')
    ax3.legend()
    
    # Covariance ratio plot
    ax4.plot(edges[:-1], ratio_peak_diagonal_terms, label='Peak Covariance Diagonal Terms Ratio')
    ax4.plot(edges[:-1], ratio_minima_diagonal_terms, label='Minima Covariance Diagonal Terms Ratio')
    ax4.set_xlim(-0.04, 0.04)
    ax4.set_title(f'Ratio of Diagonal Terms of Covariance Matrix: ({config_id1} / {config_id2})')
    ax4.set_xlabel('Index')
    ax4.set_ylabel('Covariance Ratio')
    ax4.legend()
    
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')

def main(config_id1, config_id2, source_redshift, smoothing_length, patch_size_deg):
    config_path = "/lustre/work/akira.tokiwa/Projects/LensingSSC/configs/config_analysis.json"
    config_analysis = ConfigAnalysis.from_json(config_path)
    
    def get_files(config_id):
        results_directory = os.path.join(config_analysis.resultsdir, config_id)
        return glob(os.path.join(results_directory, "peakminima", "smoothed_patch_flat", 
                                  f"zs{source_redshift:.1f}", f"sl{str(smoothing_length)}", "*.npz"))
    
    files1 = get_files(config_id1)
    files2 = get_files(config_id2)
    
    data1 = load_data(files1)
    data2 = load_data(files2)
    
    save_directory = os.path.join(config_analysis.imgdir, f"{config_id1}_vs_{config_id2}", "smoothed_patch_flat")
    os.makedirs(save_directory, exist_ok=True)
    
    save_path = os.path.join(save_directory, f"peaksminima_zs{source_redshift:.1f}_sl{str(smoothing_length)}.png")
    
    plot_and_save(data1, data2, config_id1, config_id2, save_path)
    logging.info(f"Saved the plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process kappa maps based on provided configuration.")
    parser.add_argument('config_id1', type=str, help='First configuration identifier')
    parser.add_argument('config_id2', type=str, help='Second configuration identifier')
    parser.add_argument('source_redshift', type=float, help='Source redshift')
    parser.add_argument('smoothing_length', type=int, help='Smoothing length in arcmin')
    parser.add_argument('--patch_size_deg', type=int, default=10, help='Size of each patch in degrees')
    args = parser.parse_args()
    
    main(args.config_id1, args.config_id2, args.source_redshift, args.smoothing_length, args.patch_size_deg)
    logging.info("Processing complete.")
