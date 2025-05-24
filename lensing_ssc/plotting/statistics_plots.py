"""
Statistics plots for lensing-ssc.

This module contains functions for plotting statistical distributions,
including ell-space and nu-space statistics.
"""

from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lensing_ssc.plotting.plot_utils import (
    create_comparison_grid,
    configure_comparison_axes,
    add_comparison_legend
)


def plot_binned_statistics_comparison(bin_values: np.ndarray,
                                    means_data1: Dict[Any, Dict[Tuple[str, str], Dict[str, np.ndarray]]],
                                    means_data2: Dict[Any, Dict[Tuple[str, str], Dict[str, np.ndarray]]],
                                    stds_data1: Dict[Any, Dict[Tuple[str, str], Dict[str, np.ndarray]]],
                                    stds_data2: Dict[Any, Dict[Tuple[str, str], Dict[str, np.ndarray]]],
                                    zs_list: List[float],
                                    box_types: Tuple[str, str],
                                    config_key: Tuple[str, str],
                                    stat_labels: List[str],
                                    stat_titles: List[str],
                                    colors: List[str] = None,
                                    bin_scale: str = 'log',
                                    x_lims: Tuple[float, float] = None,
                                    y_lims_mean: List[Tuple[float, float]] = None,
                                    y_lims_var: List[Tuple[float, float]] = None,
                                    suptitle: str = None,
                                    output_path: str = None) -> plt.Figure:
    """
    Create a comprehensive comparison plot of binned statistics (e.g., ell-binned or nu-binned)
    for two different simulation types.
    
    Args:
        bin_values (np.ndarray): Array of bin center values (x-axis).
        means_data1 (Dict): Dictionary containing mean values for first simulation type.
        means_data2 (Dict): Dictionary containing mean values for second simulation type.
        stds_data1 (Dict): Dictionary containing standard deviations for first simulation type.
        stds_data2 (Dict): Dictionary containing standard deviations for second simulation type.
        zs_list (List[float]): List of redshift values to plot.
        box_types (Tuple[str, str]): Names of the two simulation types (e.g., ("bigbox", "tiled")).
        config_key (Tuple[str, str]): Key to access the specific configuration in the data dictionaries.
        stat_labels (List[str]): Labels of statistics to plot (e.g., ["power_spectra", "pdf"]).
        stat_titles (List[str]): Display titles for the statistics (e.g., ["$C_{\ell}$", "PDF"]).
        colors (List[str], optional): Colors to use for different redshifts.
        bin_scale (str, optional): Scale for bin axis ('linear' or 'log'). Default is 'log'.
        x_lims (Tuple[float, float], optional): Limits for the bin axis.
        y_lims_mean (List[Tuple[float, float]], optional): Y-limits for mean ratio plots.
        y_lims_var (List[Tuple[float, float]], optional): Y-limits for variance ratio plots.
        suptitle (str, optional): Super title for the figure.
        output_path (str, optional): Path to save the figure. If None, figure is not saved.
        
    Returns:
        plt.Figure: The created figure.
    """
    # Use default colors if not provided
    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    
    # Create layout with mean and variance panels
    fig, ax_mean, ax_var, ax_mean_ratio, ax_var_ratio = create_comparison_grid(
        n_stats=len(stat_labels),
        figsize=(14, 3 * len(stat_labels)),
        suptitle=suptitle
    )
    
    # Plot data for each redshift and statistic
    for i, zs in enumerate(zs_list):
        for j, stat_label in enumerate(stat_labels):
            # Check if data exists for both simulation types
            data_exists = all(
                (box_type, zs) in data_dict and 
                config_key in data_dict[(box_type, zs)] and
                stat_label in data_dict[(box_type, zs)][config_key]
                for data_dict, box_type in zip(
                    [means_data1, means_data2, stds_data1, stds_data2], 
                    [box_types[0], box_types[1], box_types[0], box_types[1]]
                )
            )
            
            if data_exists:
                # Extract data
                mean1 = means_data1[(box_types[0], zs)][config_key][stat_label]
                mean2 = means_data2[(box_types[1], zs)][config_key][stat_label]
                std1 = stds_data1[(box_types[0], zs)][config_key][stat_label]
                std2 = stds_data2[(box_types[1], zs)][config_key][stat_label]
                
                # Handle 2D data by taking mean along axis 0 if needed
                mean1_plot = mean1.mean(axis=0) if mean1.ndim > 1 else mean1
                mean2_plot = mean2.mean(axis=0) if mean2.ndim > 1 else mean2
                std1_plot = std1.mean(axis=0) if std1.ndim > 1 else std1
                std2_plot = std2.mean(axis=0) if std2.ndim > 1 else std2
                
                # Ensure data aligns with bin_values
                if len(mean1_plot) > len(bin_values):
                    mean1_plot = mean1_plot[:len(bin_values)]
                    std1_plot = std1_plot[:len(bin_values)]
                if len(mean2_plot) > len(bin_values):
                    mean2_plot = mean2_plot[:len(bin_values)]
                    std2_plot = std2_plot[:len(bin_values)]
                
                # Plot means with error bars
                ax_mean[j].errorbar(bin_values, mean1_plot, yerr=std1_plot, 
                                  fmt='o', color=colors[i], alpha=0.5, linestyle='-')
                ax_mean[j].errorbar(bin_values, mean2_plot, yerr=std2_plot, 
                                  fmt='^', color=colors[i], alpha=0.5, linestyle='--')
                
                # Plot variances
                ax_var[j].plot(bin_values, std1_plot**2, color=colors[i], marker='o', linestyle='-')
                ax_var[j].plot(bin_values, std2_plot**2, color=colors[i], marker='^', linestyle='--')
                
                # Plot ratios
                ax_mean_ratio[j].plot(bin_values, mean1_plot/mean2_plot, color=colors[i])
                ax_var_ratio[j].plot(bin_values, (std1_plot**2)/(std2_plot**2), color=colors[i])
    
    # Configure axes
    if bin_scale == 'log':
        x_ticks = [300, 500, 1000, 2000, 3000] if x_lims and x_lims[0] >= 300 else None
        x_ticklabels = [str(x) for x in x_ticks] if x_ticks else None
    else:
        x_ticks = None
        x_ticklabels = None
    
    configure_comparison_axes(
        [ax_mean, ax_var, ax_mean_ratio, ax_var_ratio],
        x_scale=bin_scale,
        y_scale='log',
        x_limits=x_lims,
        y_limits_lower=y_lims_mean,
        x_ticks=x_ticks,
        x_ticklabels=x_ticklabels,
        labels=stat_labels,
        titles=stat_titles,
        fontsize=16
    )
    
    # Add legend
    add_comparison_legend(
        fig, 
        redshift_values=zs_list,
        sim_types=[box_types[0].capitalize(), box_types[1].capitalize()],
        colors=colors,
        linestyles=["-", "--"]
    )
    
    # Save figure if output path is provided
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    
    return fig 