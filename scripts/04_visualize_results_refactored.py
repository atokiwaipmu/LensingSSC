#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Import from the new modular plotting packages
from lensing_ssc.io.file_handlers import load_results_from_hdf5
from lensing_ssc.plotting import (
    set_plotting_defaults,
    plot_binned_statistics_comparison,
    plot_covariance_matrix,
    plot_matrix_ratio,
    plot_correlation_ratio,
    plot_rmp_rip_ratios,
    plot_bad_patch_comparison,
    process_statistics_set
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global definitions for plotting
ZS_LIST_DEFAULT = [0.5, 1.0, 1.5, 2.0, 2.5]
NGAL_LIST_DEFAULT = [0, 7, 15, 30, 50]  # represent ngal values
SL_LIST_DEFAULT = [2, 5, 8, 10]  # represent smoothing lengths in arcmin

# Define statistic labels and titles
LABELS_ELL = ["power_spectrum", "bispectrum_equilateral", "bispectrum_isosceles", "bispectrum_squeezed"]
TITLES_ELL = ["$C_{\\ell}$", "$B_{\\ell}^{(eq)}$", "$B_{\\ell}^{(iso)}$", "$B_{\\ell}^{(sq)}$"]
LABELS_NU = ["pdf", "peak_counts", "minima_counts"]
TITLES_NU = ["PDF", "Peaks", "Minima"]

# Default plotting ranges for mean and variance ratios
MRANGES_ELL = [(0.99, 1.005), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1)]
VRANGES_ELL = [(0.95, 1.25), (0.9, 1.2), (0.9, 1.2), (0.9, 1.2)]
MRANGES_NU = [(0.99, 1.015), (0.99, 1.015), (0.97, 1.05)]
VRANGES_NU = [(0.95, 1.3), (0.95, 1.3), (0.95, 1.3)]

# Survey info for plot labels
SURVEY_INFO = {
    '0': 'Noiseless',
    '7': 'DES/KiDS',
    '15': 'HSC',
    '30': 'LSST/Euclid',
    '50': 'Roman'
}

def create_output_dirs(base_dir: Path) -> Dict[str, Path]:
    """
    Create output directories for different plot types.
    
    Args:
        base_dir (Path): Base output directory.
        
    Returns:
        Dict[str, Path]: Dictionary of created directories.
    """
    dirs = {
        "ell_stats": base_dir / "ell_statistics",
        "nu_stats": base_dir / "nu_statistics",
        "correlation": base_dir / "correlation_matrices",
        "rmp_rip": base_dir / "rmp_rip_analysis",
        "bad_patch": base_dir / "bad_patch_analysis",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs

def load_theory_data(project_root: Path, zs_list: List[float]) -> Dict[str, Any]:
    """
    Load theoretical prediction data for comparison with simulation results.
    
    Args:
        project_root (Path): Project root directory.
        zs_list (List[float]): List of redshift values.
        
    Returns:
        Dict[str, Any]: Dictionary of theoretical data.
    """
    theory = {
        "cl": {},
        "cl_cov": {},
        "pdf": {},
        "pdf_cov": {}
    }
    
    # Load power spectrum theory data
    for zs in zs_list:
        theory_path = project_root / f"lensing_ssc/theory/halofit/kappa_zs{zs}_Clkk_ell_0_3000.npz"
        if theory_path.exists():
            theory_data = np.load(theory_path)
            ell_th = theory_data["ell"]
            clkk_th = theory_data["clkk"] * theory_data["ell"] * (theory_data["ell"] + 1) / 2 / np.pi
            theory["cl"][str(zs)] = {"ell": ell_th, "clkk": clkk_th}
            logging.info(f"Loaded power spectrum theory data for z={zs}")
        else:
            logging.warning(f"Theory file not found: {theory_path}")
    
    # Load PDF theory data
    for zs in zs_list:
        pdf_path = project_root / f"lensing_ssc/theory/hmpdf/fid_z{zs}_z{zs}_bin8_pdf_noisy.txt"
        if pdf_path.exists():
            theory["pdf"][str(zs)] = np.loadtxt(pdf_path)
            logging.info(f"Loaded PDF theory data for z={zs}")
        else:
            logging.warning(f"Theory PDF file not found: {pdf_path}")
        
        pdf_cov_path = project_root / f"lensing_ssc/theory/hmpdf/fid_z{zs}_z{zs}_bin8_cov_noisy.txt"
        if pdf_cov_path.exists():
            theory["pdf_cov"][str(zs)] = np.diag(np.loadtxt(pdf_cov_path).reshape(8, 8))
            logging.info(f"Loaded PDF covariance theory data for z={zs}")
        else:
            logging.warning(f"Theory PDF covariance file not found: {pdf_cov_path}")
    
    return theory

def extract_file_info(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata information from file path and contents.
    
    Args:
        file_path (str): Path to the HDF5 file.
        
    Returns:
        Dict[str, Any]: Dictionary with extracted information.
    """
    info = {"box_type": "unknown", "redshift": None}
    
    # Try to extract info from path
    path_str = str(file_path)
    if "bigbox" in path_str.lower():
        info["box_type"] = "bigbox"
    elif "tiled" in path_str.lower():
        info["box_type"] = "tiled"
    
    # Try to extract redshift from filename
    zs_match = re.search(r"zs(\d+\.\d+)", path_str)
    if zs_match:
        info["redshift"] = float(zs_match.group(1))
    
    return info

def load_rmp_rip_indices(rmp_idx_file: str) -> Dict[str, np.ndarray]:
    """
    Load RMP (Regular Meaningful Patches) and RIP (Repeated Impact Patches) indices.
    
    Args:
        rmp_idx_file (str): Path to the file containing indices.
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with keys 'valid_idx' (RMP), 'excluded_in_scenario2' (RIP), etc.
    """
    indices = {}
    
    try:
        data = np.load(rmp_idx_file, allow_pickle=True)
        # Try to extract common index arrays
        for key in ['valid_idx', 'excluded_in_scenario2', 'valid_idx_scenario1', 'valid_idx_scenario2', 'black_list']:
            if key in data:
                indices[key] = data[key]
    except Exception as e:
        # If loading fails, use default indices based on legacy notebook logic
        N_opt = 260  # Default from legacy notebooks
        indices['valid_idx_scenario1'] = np.arange(N_opt)
        indices['valid_idx_scenario2'] = np.arange(N_opt - 50)
        indices['excluded_in_scenario2'] = np.setdiff1d(indices['valid_idx_scenario1'], indices['valid_idx_scenario2'])
        if not indices['excluded_in_scenario2'].size:
            indices['excluded_in_scenario2'] = np.arange(50)
        indices['black_list'] = np.array([132])  # Default from legacy notebooks
        
        # Calculate valid_idx (used for RMP)
        tmp_valid_idx = np.arange(len(indices['valid_idx_scenario1']))
        indices['valid_idx'] = np.delete(tmp_valid_idx, indices['excluded_in_scenario2'])
        
        logging.warning(f"Error loading RMP/RIP indices from {rmp_idx_file}: {e}. Using default indices.")
    
    return indices

def process_rmp_rip_data(aggregated_stats_raw: Dict, indices: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    """
    Process the aggregated statistics to separate RMP and RIP data.
    
    Args:
        aggregated_stats_raw (Dict): Raw aggregated statistics.
        indices (Dict[str, np.ndarray]): Dictionary with RMP and RIP indices.
        
    Returns:
        Tuple[Dict, Dict, Dict, Dict, Dict, Dict]: 
            (means_rmp, stds_rmp, means_rip, stds_rip, means_bl, stds_bl)
    """
    # Initialize dictionaries
    combined_data_rmp = {}
    combined_data_rip = {}
    combined_data_bl = {}
    
    # Get indices
    valid_idx = indices.get('valid_idx', None)
    excluded_idx = indices.get('excluded_in_scenario2', None)
    black_list = indices.get('black_list', None)
    
    if valid_idx is None or excluded_idx is None:
        logging.error("Missing required indices for RMP/RIP separation.")
        return {}, {}, {}, {}, {}, {}
    
    # Process aggregated data to separate RMP, RIP, and bad patches
    for (box_type, zs), data_by_config in aggregated_stats_raw.items():
        combined_data_rmp.setdefault((box_type, zs), {})
        combined_data_rip.setdefault((box_type, zs), {})
        combined_data_bl.setdefault((box_type, zs), {})
        
        for config_key, stats_by_name in data_by_config.items():
            combined_data_rmp[(box_type, zs)].setdefault(config_key, {})
            combined_data_rip[(box_type, zs)].setdefault(config_key, {})
            combined_data_bl[(box_type, zs)].setdefault(config_key, {})
            
            for stat_name, arrays in stats_by_name.items():
                # Each array should be a collection of patches
                # We need to extract the appropriate patches for RMP, RIP, and black list
                rmp_arrays = []
                rip_arrays = []
                bl_arrays = []
                
                for arr in arrays:
                    # Skip empty or malformed arrays
                    if not isinstance(arr, np.ndarray) or arr.size == 0:
                        continue
                    
                    # Ensure array has enough patches
                    max_idx = max(np.max(valid_idx) if valid_idx.size > 0 else 0,
                                 np.max(excluded_idx) if excluded_idx.size > 0 else 0,
                                 np.max(black_list) if black_list is not None and black_list.size > 0 else 0)
                    
                    if arr.shape[0] <= max_idx:
                        continue
                    
                    # Extract the arrays for each type
                    rmp_arrays.append(arr[valid_idx])
                    rip_arrays.append(arr[excluded_idx])
                    if black_list is not None and black_list.size > 0:
                        bl_arrays.append(arr[black_list])
                
                combined_data_rmp[(box_type, zs)][config_key][stat_name] = rmp_arrays
                combined_data_rip[(box_type, zs)][config_key][stat_name] = rip_arrays
                combined_data_bl[(box_type, zs)][config_key][stat_name] = bl_arrays
    
    # Calculate statistics for each set
    means_rmp, stds_rmp, _, _ = {}, {}, {}, {}
    means_rip, stds_rip, _, _ = {}, {}, {}, {}
    means_bl, stds_bl, _, _ = {}, {}, {}, {}
    
    process_statistics_set(combined_data_rmp, means_rmp, stds_rmp, {}, {}, "RMP")
    process_statistics_set(combined_data_rip, means_rip, stds_rip, {}, {}, "RIP")
    process_statistics_set(combined_data_bl, means_bl, stds_bl, {}, {}, "BL")
    
    return means_rmp, stds_rmp, means_rip, stds_rip, means_bl, stds_bl

def main(args):
    # Set plotting defaults
    set_plotting_defaults(fontsize=14)
    
    logging.info(f"Starting visualization script.")
    input_files = [Path(f) for f in args.input_files]
    output_dir = Path(args.output_dir)
    output_dirs = create_output_dirs(output_dir)
    
    logging.info(f"Processing {len(input_files)} input files.")
    logging.info(f"Visualizations will be saved to: {output_dir}")
    
    # Aggregate raw statistics data
    aggregated_stats_raw = {}
    
    # These will store the processed statistics
    all_means = {}
    all_stds = {}
    all_covariances = {}
    all_correlations = {}
    
    # RMP/RIP analysis dictionaries
    rmp_means, rmp_stds = {}, {}
    rip_means, rip_stds = {}, {}
    bl_means, bl_stds = {}  # Bad patch statistics
    
    # Metadata for plotting
    plot_metadata = {
        "ell": None,
        "nu_bins": None,
        "l_edges": None
    }
    metadata_loaded = False
    
    # Determine project root for theory data loading
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent.parent
    
    # Load RMP/RIP indices if provided
    rmp_rip_indices = {}
    if args.rmp_idx_file:
        rmp_rip_indices = load_rmp_rip_indices(args.rmp_idx_file)
    
    # Data loading and aggregation
    for hdf5_file in input_files:
        logging.info(f"Loading and processing: {hdf5_file}")
        try:
            data = load_results_from_hdf5(str(hdf5_file))
            metadata = data.get('metadata', {})
            run_params = metadata.get('parameters', {})
            
            # Extract file info
            file_info = extract_file_info(str(hdf5_file))
            box_type = file_info["box_type"]
            source_redshift = file_info["redshift"]
            
            # Try to get redshift from run parameters if not found in filename
            if source_redshift is None:
                source_redshift = run_params.get('zs')
                if source_redshift is None:
                    logging.warning(f"Could not determine source redshift for {hdf5_file}, skipping file.")
                    continue
            
            logging.debug(f"File {hdf5_file}: box_type={box_type}, zs={source_redshift}")
            
            # Load metadata from the first file
            if not metadata_loaded and run_params:
                plot_metadata["ell"] = run_params.get('ell_values')
                plot_metadata["nu_bins"] = run_params.get('nu_bins')
                plot_metadata["l_edges"] = run_params.get('l_edges')
                if plot_metadata["ell"] is not None and plot_metadata["nu_bins"] is not None:
                    metadata_loaded = True
                    logging.info(f"Loaded plot metadata (ell, nu_bins) from {hdf5_file}")
                else:
                    logging.warning(f"Could not load all plot metadata from {hdf5_file}")
            
            # Initialize dict for this box_type and zs if not present
            agg_key = (box_type, source_redshift)
            if agg_key not in aggregated_stats_raw:
                aggregated_stats_raw[agg_key] = {}
            
            # Process statistics from each patch
            stats_data_per_patch = data.get('statistics', {})
            for patch_id_str, patch_data in stats_data_per_patch.items():
                for noise_key, noise_data in patch_data.items():
                    ngal_val = int(noise_key.split('_')[-1])
                    ngal_config_key = f"ngal_{ngal_val}"
                    
                    # Handle non-smoothed stats (power_spectrum, bispectrum)
                    for stat_name in LABELS_ELL:
                        if stat_name in noise_data:
                            stat_array = noise_data[stat_name]
                            sl_config_key = "sl_none"
                            if (ngal_config_key, sl_config_key) not in aggregated_stats_raw[agg_key]:
                                aggregated_stats_raw[agg_key][(ngal_config_key, sl_config_key)] = {}
                            if stat_name not in aggregated_stats_raw[agg_key][(ngal_config_key, sl_config_key)]:
                                aggregated_stats_raw[agg_key][(ngal_config_key, sl_config_key)][stat_name] = []
                            aggregated_stats_raw[agg_key][(ngal_config_key, sl_config_key)][stat_name].append(stat_array)
                    
                    # Handle smoothed stats (pdf, peaks, minima)
                    for sl_key, sl_data_content in noise_data.items():
                        if not sl_key.startswith("sl_"): 
                            continue
                        
                        sl_val = float(sl_key.split('_')[-1])
                        sl_config_key = f"sl_{sl_val}"
                        
                        for stat_name in LABELS_NU:
                            if stat_name in sl_data_content:
                                stat_array = sl_data_content[stat_name]
                                if (ngal_config_key, sl_config_key) not in aggregated_stats_raw[agg_key]:
                                    aggregated_stats_raw[agg_key][(ngal_config_key, sl_config_key)] = {}
                                if stat_name not in aggregated_stats_raw[agg_key][(ngal_config_key, sl_config_key)]:
                                    aggregated_stats_raw[agg_key][(ngal_config_key, sl_config_key)][stat_name] = []
                                aggregated_stats_raw[agg_key][(ngal_config_key, sl_config_key)][stat_name].append(stat_array)
        except Exception as e:
            logging.error(f"Error processing file {hdf5_file}: {e}", exc_info=True)
            continue
    
    # Process statistics from aggregated raw data
    logging.info("Calculating statistics from aggregated data...")
    for agg_key, data_for_agg_key in aggregated_stats_raw.items():
        box_type, source_redshift = agg_key
        
        # Initialize dictionaries if needed
        all_means.setdefault(agg_key, {})
        all_stds.setdefault(agg_key, {})
        all_covariances.setdefault(agg_key, {})
        all_correlations.setdefault(agg_key, {})
        
        # Process each configuration
        for (ngal_sl_key), stats_dict_raw in data_for_agg_key.items():
            # Ensure we have empty dictionaries for this configuration
            all_means[agg_key].setdefault(ngal_sl_key, {})
            all_stds[agg_key].setdefault(ngal_sl_key, {})
            all_covariances[agg_key].setdefault(ngal_sl_key, {})
            all_correlations[agg_key].setdefault(ngal_sl_key, {})
            
            # Process each statistic
            for stat_name, list_of_arrays in stats_dict_raw.items():
                if not list_of_arrays:
                    logging.warning(f"No data found for {stat_name} under {agg_key} and {ngal_sl_key}")
                    continue
                try:
                    # Stack arrays: each array is from a patch
                    stacked_arrays = np.array(list_of_arrays)
                    if stacked_arrays.ndim == 1:  # If only one patch and stat is 1D array
                        stacked_arrays = stacked_arrays[:, np.newaxis]
                    elif stacked_arrays.ndim == 0:  # Skip if somehow it's a scalar
                        logging.warning(f"Skipping scalar data for {stat_name} at {agg_key}, {ngal_sl_key}")
                        continue
                    
                    # Calculate statistics
                    all_means[agg_key][ngal_sl_key][stat_name] = np.mean(stacked_arrays, axis=0)
                    all_stds[agg_key][ngal_sl_key][stat_name] = np.std(stacked_arrays, axis=0)
                    
                    # Calculate covariance and correlation if enough samples
                    if stacked_arrays.shape[0] >= 2 and stacked_arrays.shape[1] > 1:
                        all_covariances[agg_key][ngal_sl_key][stat_name] = np.cov(stacked_arrays, rowvar=False)
                        all_correlations[agg_key][ngal_sl_key][stat_name] = np.corrcoef(stacked_arrays, rowvar=False)
                        
                        # Handle cases where variance is zero leading to NaNs in corrcoef
                        if np.any(np.isnan(all_correlations[agg_key][ngal_sl_key][stat_name])):
                            np.fill_diagonal(all_correlations[agg_key][ngal_sl_key][stat_name], 1)
                    else:
                        # For single sample or 1D arrays, use placeholder matrices
                        dim = stacked_arrays.shape[1] if stacked_arrays.ndim > 1 else 1
                        all_covariances[agg_key][ngal_sl_key][stat_name] = np.full((dim, dim), np.nan)
                        all_correlations[agg_key][ngal_sl_key][stat_name] = np.full((dim, dim), np.nan)
                        np.fill_diagonal(all_correlations[agg_key][ngal_sl_key][stat_name], 1)
                    
                except Exception as e:
                    logging.error(f"Error during statistics calculation for {stat_name} ({agg_key}, {ngal_sl_key}): {e}", exc_info=True)
                    continue
    
    # Process RMP/RIP data if indices are available
    if rmp_rip_indices:
        logging.info("Processing RMP/RIP data...")
        rmp_means, rmp_stds, rip_means, rip_stds, bl_means, bl_stds = process_rmp_rip_data(
            aggregated_stats_raw, rmp_rip_indices
        )
    
    # Load theory data if requested
    theory_data = {}
    if args.include_theory:
        theory_data = load_theory_data(project_root, ZS_LIST_DEFAULT)
    
    # Set default ell and nu if not loaded from metadata
    if not metadata_loaded:
        logging.warning("Metadata not loaded. Using default ell and nu values.")
        plot_metadata["ell"] = np.logspace(np.log10(300), np.log10(3000), 8)
        plot_metadata["nu_bins"] = np.linspace(-4, 4, 8)
        plot_metadata["l_edges"] = np.logspace(np.log10(300), np.log10(3000), 9)
    
    # Create plots based on command line arguments
    if args.plot_ell_stats or args.plot_all:
        logging.info("Generating ell-space statistics plots...")
        for ngal_val in args.ngal_values:
            for sl_val in args.sl_values:
                ngal_key = f"ngal_{ngal_val}"
                sl_key = f"sl_{sl_val}"
                config_key = (ngal_key, sl_key if sl_val != "none" else "sl_none")
                
                # Use the imported function from statistics_plots.py
                fig = plot_binned_statistics_comparison(
                    bin_values=plot_metadata["ell"],
                    means_data1=all_means,
                    means_data2=all_means,
                    stds_data1=all_stds,
                    stds_data2=all_stds,
                    zs_list=args.zs_values,
                    box_types=("bigbox", "tiled"),
                    config_key=config_key,
                    stat_labels=LABELS_ELL,
                    stat_titles=TITLES_ELL,
                    colors=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
                    bin_scale='log',
                    x_lims=(300, 3000),
                    y_lims_mean=MRANGES_ELL,
                    suptitle=f"Ell Statistics: {SURVEY_INFO.get(str(ngal_val), f'ngal={ngal_val}')}",
                    output_path=output_dirs["ell_stats"] / f"ell_stats_ngal{ngal_val}_sl{sl_val}.pdf"
                )
                plt.close(fig)
    
    if args.plot_nu_stats or args.plot_all:
        logging.info("Generating nu-space statistics plots...")
        for ngal_val in args.ngal_values:
            for sl_val in args.sl_values:
                if sl_val == "none":
                    continue  # Skip sl_none for nu-space statistics
                
                ngal_key = f"ngal_{ngal_val}"
                sl_key = f"sl_{sl_val}"
                config_key = (ngal_key, sl_key)
                
                # Plot bigbox vs tiled comparison
                fig = plot_binned_statistics_comparison(
                    bin_values=plot_metadata["nu_bins"],
                    means_data1=all_means,
                    means_data2=all_means,
                    stds_data1=all_stds,
                    stds_data2=all_stds,
                    zs_list=args.zs_values,
                    box_types=("bigbox", "tiled"),
                    config_key=config_key,
                    stat_labels=LABELS_NU,
                    stat_titles=TITLES_NU,
                    colors=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
                    bin_scale='linear',
                    x_lims=(-4, 4),
                    y_lims_mean=MRANGES_NU,
                    suptitle=f"Nu Statistics: {SURVEY_INFO.get(str(ngal_val), f'ngal={ngal_val}')}, sl={sl_val}",
                    output_path=output_dirs["nu_stats"] / f"nu_stats_ngal{ngal_val}_sl{sl_val}.pdf"
                )
                plt.close(fig)
    
    if args.plot_correlation or args.plot_all:
        logging.info("Generating correlation matrix plots...")
        for box_type in ["bigbox", "tiled"]:
            for zs_val in args.zs_values:
                for ngal_val in args.ngal_values:
                    for sl_val in args.sl_values:
                        if sl_val == "none":
                            stat_list = LABELS_ELL
                            stat_titles = TITLES_ELL
                        else:
                            stat_list = LABELS_NU
                            stat_titles = TITLES_NU
                        
                        ngal_key = f"ngal_{ngal_val}"
                        sl_key = f"sl_{sl_val}"
                        config_key = (ngal_key, sl_key)
                        
                        for i, stat in enumerate(stat_list):
                            # Check if data exists
                            if ((box_type, zs_val) in all_correlations and 
                                config_key in all_correlations[(box_type, zs_val)] and
                                stat in all_correlations[(box_type, zs_val)][config_key]):
                                
                                corr_matrix = all_correlations[(box_type, zs_val)][config_key][stat]
                                
                                # Plot correlation matrix
                                fig, ax = plt.subplots(figsize=(8, 6))
                                plot_covariance_matrix(
                                    corr_matrix,
                                    title=f"{stat_titles[i]}: {box_type}, z={zs_val}, {ngal_key}, {sl_key}",
                                    ax=ax,
                                    cmap='coolwarm',
                                    vmin=-1,
                                    vmax=1
                                )
                                fig.tight_layout()
                                fig.savefig(
                                    output_dirs["correlation"] / f"corr_{box_type}_z{zs_val}_{ngal_key}_{sl_key}_{stat}.pdf",
                                    bbox_inches="tight"
                                )
                                plt.close(fig)
    
    if args.plot_covariance_ratio or args.plot_all:
        logging.info("Generating covariance matrix ratio plots...")
        for zs_val in args.zs_values:
            for ngal_val in args.ngal_values:
                for sl_val in args.sl_values:
                    if sl_val == "none":
                        stat_list = LABELS_ELL
                        stat_titles = TITLES_ELL
                    else:
                        stat_list = LABELS_NU
                        stat_titles = TITLES_NU
                    
                    ngal_key = f"ngal_{ngal_val}"
                    sl_key = f"sl_{sl_val}"
                    config_key = (ngal_key, sl_key)
                    
                    for i, stat in enumerate(stat_list):
                        # Check if data exists for both box types
                        if (('bigbox', zs_val) in all_covariances and 
                            config_key in all_covariances[('bigbox', zs_val)] and 
                            stat in all_covariances[('bigbox', zs_val)][config_key] and
                            ('tiled', zs_val) in all_covariances and
                            config_key in all_covariances[('tiled', zs_val)] and
                            stat in all_covariances[('tiled', zs_val)][config_key]):
                            
                            bigbox_cov = all_covariances[('bigbox', zs_val)][config_key][stat]
                            tiled_cov = all_covariances[('tiled', zs_val)][config_key][stat]
                            
                            # Plot ratio matrix
                            fig, ax = plt.subplots(figsize=(8, 6))
                            plot_matrix_ratio(
                                bigbox_cov, 
                                tiled_cov,
                                title=f"Cov Ratio {stat_titles[i]}: z={zs_val}, {ngal_key}, {sl_key}",
                                ax=ax,
                                cmap='RdBu_r',
                                vmin=0.6, 
                                vmax=1.4
                            )
                            fig.tight_layout()
                            fig.savefig(
                                output_dirs["correlation"] / f"cov_ratio_z{zs_val}_{ngal_key}_{sl_key}_{stat}.pdf",
                                bbox_inches="tight"
                            )
                            plt.close(fig)
    
    # RMP vs RIP analysis
    if args.plot_rmp_rip or args.plot_all:
        if rmp_means and rip_means:
            logging.info("Generating RMP vs RIP comparison plots...")
            for ngal_val in args.ngal_values:
                for sl_val in args.sl_values:
                    if sl_val == "none":
                        continue  # Skip sl_none for RMP/RIP analysis
                    
                    ngal_key = f"ngal_{ngal_val}"
                    sl_key = f"sl_{sl_val}"
                    
                    # Generate combined ratio plot
                    fig = plot_rmp_rip_ratios(
                        means_rmp=rmp_means,
                        means_rip=rip_means,
                        stds_rmp=rmp_stds,
                        stds_rip=rip_stds,
                        zs_list=args.zs_values,
                        box_type=["bigbox", "tiled"],
                        ngal_comp_plot=ngal_key,
                        sl_comp_plot=sl_key,
                        ell=plot_metadata["ell"],
                        nu=plot_metadata["nu_bins"],
                        labels_ell=LABELS_ELL,
                        labels_nu=LABELS_NU,
                        titles_ell=TITLES_ELL,
                        titles_nu=TITLES_NU,
                        colors=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
                        output_path=output_dirs["rmp_rip"] / f"rmp_rip_ratios_{ngal_key}_{sl_key}.pdf"
                    )
                    plt.close(fig)
                    
                    # For selected redshifts, plot bad patch comparison
                    for zs_val in [1.0, 2.0]:  # Selected redshifts for bad patch comparison
                        if zs_val in args.zs_values and bl_means:
                            fig = plot_bad_patch_comparison(
                                means_rmp=rmp_means,
                                means_rip=rip_means,
                                means_bl=bl_means,
                                zs_value=zs_val,
                                box_type="bigbox",  # Usually focus on bigbox for this analysis
                                ngal_comp_plot=ngal_key,
                                sl_comp_plot=sl_key,
                                ell=plot_metadata["ell"],
                                labels_ell=LABELS_ELL,
                                titles_ell=TITLES_ELL,
                                output_path=output_dirs["bad_patch"] / f"bad_patch_z{zs_val}_{ngal_key}_{sl_key}.pdf"
                            )
                            plt.close(fig)
        else:
            logging.warning("RMP/RIP analysis requested but data not available. Check if --rmp-idx-file is provided.")
    
    logging.info("Visualization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations from statistical analysis results.")
    parser.add_argument("input_files", type=str, nargs='+',
                        help="Paths to HDF5 files containing statistical results from 03_run_analysis.py.")
    parser.add_argument("-o", "--output-dir", type=str, default="results/figures/visualizations",
                        help="Directory to save the generated plots.")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Project root directory for finding theory data.")
    
    # Plot selection arguments
    parser.add_argument("--plot-all", action="store_true", 
                        help="Generate all available plots.")
    parser.add_argument("--plot-ell-stats", action="store_true", 
                        help="Generate ell-space statistics plots.")
    parser.add_argument("--plot-nu-stats", action="store_true", 
                        help="Generate nu-space statistics plots.")
    parser.add_argument("--plot-correlation", action="store_true", 
                        help="Generate correlation matrix plots.")
    parser.add_argument("--plot-covariance-ratio", action="store_true", 
                        help="Generate covariance ratio matrix plots.")
    parser.add_argument("--plot-rmp-rip", action="store_true", 
                        help="Generate RMP vs RIP analysis plots.")
    parser.add_argument("--plot-bad-patch", action="store_true", 
                        help="Generate bad patch analysis plots.")
    
    # Data filtering arguments
    parser.add_argument("--zs-values", type=float, nargs='+', default=ZS_LIST_DEFAULT,
                        help="Redshift values to include in plots.")
    parser.add_argument("--ngal-values", type=str, nargs='+', default=['0', '7', '30'],
                        help="Galaxy density values to include in plots.")
    parser.add_argument("--sl-values", type=str, nargs='+', default=['2', '5', '10', 'none'],
                        help="Smoothing length values to include in plots. Use 'none' for ell statistics.")
    
    # Optional arguments for theory comparison
    parser.add_argument("--include-theory", action="store_true",
                        help="Include theoretical predictions in plots where available.")
    
    # Optional arguments for RMP/RIP analysis
    parser.add_argument("--rmp-idx-file", type=str, default=None,
                        help="Path to file containing RMP/RIP patch indices. Required for RMP/RIP analysis.")
    
    args = parser.parse_args()
    main(args) 