"""
Correlation matrix plots for lensing-ssc.

This module contains functions for visualizing correlation and covariance matrices,
including matrix comparisons and ratio plots.
"""

from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

from lensing_ssc.plotting.plot_utils import (
    calculate_correlation_average,
    merge_correlation_matrices,
)


def plot_covariance_matrix(covariance_matrix: np.ndarray, title: str = None, 
                           ax: Optional[plt.Axes] = None, cmap: str = 'viridis',
                           colorbar: bool = True, vmin: float = None, vmax: float = None) -> plt.Axes:
    """
    Plot a covariance matrix as a heatmap.
    
    Args:
        covariance_matrix (np.ndarray): Covariance matrix to plot.
        title (str, optional): Title for the plot.
        ax (plt.Axes, optional): Axes to plot on. If None, a new figure is created.
        cmap (str, optional): Colormap to use. Default is 'viridis'.
        colorbar (bool, optional): Whether to display a colorbar. Default is True.
        vmin (float, optional): Minimum value for colormap scaling.
        vmax (float, optional): Maximum value for colormap scaling.
        
    Returns:
        plt.Axes: The axes on which the plot was drawn.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a custom colormap that handles NaN values
    cmap_obj = plt.get_cmap(cmap)
    cmap_obj.set_bad(color="gray")
    
    im = ax.imshow(covariance_matrix, cmap=cmap_obj, origin="lower", vmin=vmin, vmax=vmax)
    
    if title:
        ax.set_title(title, fontsize=14)
    
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Clean up the axes
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return ax


def plot_matrix_ratio(matrix1: np.ndarray, matrix2: np.ndarray, title: str = None, 
                     ax: Optional[plt.Axes] = None, cmap: str = 'RdBu_r',
                     colorbar: bool = True, vmin: float = 0.6, vmax: float = 1.4,
                     min_threshold: float = 1e-10) -> plt.Axes:
    """
    Plot the ratio of two matrices as a heatmap.
    
    Args:
        matrix1 (np.ndarray): Numerator matrix.
        matrix2 (np.ndarray): Denominator matrix.
        title (str, optional): Title for the plot.
        ax (plt.Axes, optional): Axes to plot on. If None, a new figure is created.
        cmap (str, optional): Colormap to use. Default is 'RdBu_r'.
        colorbar (bool, optional): Whether to display a colorbar. Default is True.
        vmin (float, optional): Minimum value for colormap scaling. Default is 0.6.
        vmax (float, optional): Maximum value for colormap scaling. Default is 1.4.
        min_threshold (float, optional): Threshold below which denominator values are 
                                        considered zero. Default is 1e-10.
        
    Returns:
        plt.Axes: The axes on which the plot was drawn.
    """
    ratio = np.where(np.abs(matrix2) > min_threshold, matrix1 / matrix2, np.nan)
    return plot_covariance_matrix(ratio, title, ax, cmap, colorbar, vmin, vmax)


def plot_correlation_ratio(correlations_data1: Dict, correlations_data2: Dict,
                         zs_list: List[float], box_type: str,
                         ngal_comp_plot: str, sl_comp_plot: str,
                         stat_list: List[str], stat_titles: List[str],
                         output_path: str = None, fontsize: int = 16):
    """
    Plot the ratio of correlation matrices between two datasets.
    
    Args:
        correlations_data1 (Dict): First correlation data dictionary (e.g., RMP)
        correlations_data2 (Dict): Second correlation data dictionary (e.g., RIP)
        zs_list (List[float]): List of redshift values to analyze
        box_type (str): Box type to analyze (e.g., "bigbox")
        ngal_comp_plot (str): Galaxy density configuration key (e.g., "ngal_0")
        sl_comp_plot (str): Smoothing length configuration key (e.g., "sl_2")
        stat_list (List[str]): List of statistics to plot
        stat_titles (List[str]): Formatted titles for the statistics
        output_path (str, optional): Path to save the output figure
        fontsize (int, optional): Font size for text elements
        
    Returns:
        plt.Figure: The created figure
    """
    # Create a figure with subplots for each statistic and redshift
    n_stats = len(stat_list)
    n_zs = len(zs_list)
    
    fig = plt.figure(figsize=(14, 3 * n_stats))
    gs = GridSpec(n_stats, n_zs, figure=fig, wspace=0.1, hspace=0.3)
    
    fig.suptitle(f"Correlation Matrix Ratio: Data1/Data2 ({box_type}, {ngal_comp_plot}, {sl_comp_plot})",
                fontsize=fontsize+2)
    
    # Create a custom colormap for correlation ratios
    cmap = plt.cm.RdBu_r
    
    for i, stat in enumerate(stat_list):
        for j, zs in enumerate(zs_list):
            ax = fig.add_subplot(gs[i, j])
            
            # Check if data is available
            data_available = (
                (box_type, zs) in correlations_data1 and
                (ngal_comp_plot, sl_comp_plot) in correlations_data1[(box_type, zs)] and
                stat in correlations_data1[(box_type, zs)][(ngal_comp_plot, sl_comp_plot)] and
                (box_type, zs) in correlations_data2 and
                (ngal_comp_plot, sl_comp_plot) in correlations_data2[(box_type, zs)] and
                stat in correlations_data2[(box_type, zs)][(ngal_comp_plot, sl_comp_plot)]
            )
            
            if data_available:
                # Get correlation matrices
                corr1 = correlations_data1[(box_type, zs)][(ngal_comp_plot, sl_comp_plot)][stat]
                corr2 = correlations_data2[(box_type, zs)][(ngal_comp_plot, sl_comp_plot)][stat]
                
                # Calculate average off-diagonal correlation
                avg1 = calculate_correlation_average(corr1)
                avg2 = calculate_correlation_average(corr2)
                
                # Calculate ratio (handling division by zero/NaN)
                if np.isfinite(avg1) and np.isfinite(avg2) and avg2 != 0:
                    ratio = avg1 / avg2
                else:
                    ratio = np.nan
                
                # Plot ratio of matrices
                ratio_matrix = np.where(
                    (np.abs(corr2) > 1e-10) & np.isfinite(corr1) & np.isfinite(corr2),
                    corr1 / corr2,
                    np.nan
                )
                
                im = ax.imshow(ratio_matrix, cmap=cmap, vmin=0.5, vmax=1.5)
                
                # Add text showing the average ratio
                if np.isfinite(ratio):
                    ax.text(0.5, 0.9, f"Avg Ratio: {ratio:.2f}",
                          transform=ax.transAxes, ha='center',
                          color='white', fontsize=fontsize-4)
            else:
                ax.text(0.5, 0.5, "Data not available",
                      ha='center', va='center', transform=ax.transAxes)
            
            # Set titles and labels
            if j == 0:
                ax.set_ylabel(stat_titles[i], fontsize=fontsize)
            
            if i == 0:
                ax.set_title(f"$z_s = {zs}$", fontsize=fontsize)
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add colorbar
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    plt.colorbar(im, cax=cax, label="Correlation Ratio (Data1/Data2)")
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    
    # Save if path is provided
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    
    return fig 