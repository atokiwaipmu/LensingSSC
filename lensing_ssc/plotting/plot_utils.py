# lensing_ssc/plotting/plot_utils.py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import os

from lensing_ssc.core.fibonacci_utils import PatchOptimizer, FibonacciGrid


def plot_coverage_map(coverage_map: np.ndarray, title: str, subplot: tuple, rot: tuple = (0, 0, 0)):
    """
    Creates a Healpy orthview plot of the coverage map.

    Args:
        coverage_map (np.ndarray): The coverage map to display.
        title (str): The title of the plot.
        subplot (tuple): A tuple specifying the subplot layout (nrows, ncols, index).
        rot (tuple, optional): The rotation angles of the view, default is (0, 0, 0).
    """
    hp.orthview(coverage_map, nest=True, half_sky=True, title=title,
                sub=subplot, rot=rot, cmap='viridis', cbar=False)
    hp.graticule()


def plot_fibonacci_grid(optimizer: PatchOptimizer, n: int = None):
    """
    Displays the number of patch pixels along with the Fibonacci grid on a Healpix map.

    Args:
        optimizer (PatchOptimizer): The optimizer to use.
        n (int, optional): The number of patches to plot. If None, N_opt is used.

    Returns:
        tuple: (fig, pixels) where 'pixels' is a list of the number of patch pixels.
    """
    if n is None:
        n = optimizer.N_opt
    if n is None:
        raise ValueError("The optimal number of patches has not been set. Please run optimize() first.")

    npix = hp.nside2npix(optimizer.nside)
    tmp = np.zeros(npix)

    points = FibonacciGrid.fibonacci_grid_on_sphere(n)
    valid_points = points[(points[:, 0] < np.pi - optimizer.radius) & (points[:, 0] > optimizer.radius)]
    invalid_points = points[(points[:, 0] >= np.pi - optimizer.radius) | (points[:, 0] <= optimizer.radius)]

    pixels = []
    for center in valid_points:
        vertices = optimizer.rotated_vertices(center)
        vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
        ipix = hp.query_polygon(nside=optimizer.nside, vertices=vecs, nest=True)
        tmp[ipix] += 1
        pixels.append(len(ipix))

    for center in invalid_points:
        vertices = optimizer.rotated_vertices(center)
        vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
        ipix = hp.query_polygon(nside=optimizer.nside, vertices=vecs, nest=True)
        tmp[ipix] -= 1
        pixels.append(len(ipix))

    fig = plt.figure(figsize=(10, 5))

    hp.orthview(tmp, title=f'Fibonacci Grid ({optimizer.patch_size}° Patches), {n} Patches',
                nest=True, half_sky=True, cbar=False, cmap='viridis', fig=fig, sub=(1, 2, 1))
    hp.orthview(tmp, title=f'Top View: {n} Patches', nest=True,
                rot=(0, 90, 0), half_sky=True, cbar=False, cmap='viridis', fig=fig, sub=(1, 2, 2))

    return fig, pixels


# Helper functions for covariance/correlation analysis
def calculate_correlation_average(corr_matrix: np.ndarray) -> float:
    """
    Calculate the average of off-diagonal elements of a correlation matrix.
    
    Args:
        corr_matrix (np.ndarray): Correlation matrix to analyze.
        
    Returns:
        float: Mean of off-diagonal correlation values, or NaN if all values are invalid.
    """
    # Use mask to exclude diagonal elements and NaN values
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool) & ~np.isnan(corr_matrix)
    if np.sum(mask) == 0:
        return np.nan  # Avoid mean of empty slice warning if all are NaN or diagonal
    return np.nanmean(corr_matrix[mask])


def safe_mean(arr: np.ndarray) -> float:
    """
    Calculate the mean of an array, handling NaN and infinite values.
    
    Args:
        arr (np.ndarray): Array of values.
        
    Returns:
        float: Mean of non-infinite values, or NaN if all are invalid.
    """
    arr_no_inf = np.where(np.isinf(arr), np.nan, arr)
    return np.nanmean(arr_no_inf)


def merge_correlation_matrices(corr1: np.ndarray, corr2: np.ndarray) -> np.ndarray:
    """
    Merge two correlation matrices, using the upper triangle from corr1 
    and the lower triangle from corr2. Useful for visual comparison.
    
    Args:
        corr1 (np.ndarray): First correlation matrix (upper triangle used).
        corr2 (np.ndarray): Second correlation matrix (lower triangle used).
        
    Returns:
        np.ndarray: Merged correlation matrix.
    """
    if corr1.shape != corr2.shape or corr1.ndim != 2 or corr1.shape[0] != corr1.shape[1]:
        raise ValueError("Both correlation matrices must be square and have the same shape.")
    
    merged_corr = np.zeros_like(corr1)
    upper_indices = np.triu_indices_from(corr1, k=1)
    lower_indices = np.tril_indices_from(corr1, k=-1)
    
    merged_corr[upper_indices] = corr1[upper_indices]
    merged_corr[lower_indices] = corr2[lower_indices]
    np.fill_diagonal(merged_corr, 1.0)
    return merged_corr


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


def plot_correlation_comparison(corr1: np.ndarray, corr2: np.ndarray, title: str = None, 
                              ax: Optional[plt.Axes] = None, cmap: str = 'coolwarm',
                              colorbar: bool = True) -> plt.Axes:
    """
    Plot a merged visualization of two correlation matrices for comparison.
    Upper triangle from corr1, lower triangle from corr2.
    
    Args:
        corr1 (np.ndarray): First correlation matrix (shown in upper triangle).
        corr2 (np.ndarray): Second correlation matrix (shown in lower triangle).
        title (str, optional): Title for the plot.
        ax (plt.Axes, optional): Axes to plot on. If None, a new figure is created.
        cmap (str, optional): Colormap to use. Default is 'coolwarm'.
        colorbar (bool, optional): Whether to display a colorbar. Default is True.
        
    Returns:
        plt.Axes: The axes on which the plot was drawn.
    """
    merged_corr = merge_correlation_matrices(corr1, corr2)
    return plot_covariance_matrix(merged_corr, title, ax, cmap, colorbar, vmin=-1, vmax=1)


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


def create_statistics_grid(n_rows: int, n_cols: int, figsize: tuple = (12, 8),
                          suptitle: str = None, hspace: float = 0.3, wspace: float = 0.3) -> Tuple:
    """
    Create a grid layout for displaying multiple statistical plots.
    
    Args:
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.
        figsize (tuple, optional): Figure size. Default is (12, 8).
        suptitle (str, optional): Super title for the entire figure.
        hspace (float, optional): Horizontal spacing. Default is 0.3.
        wspace (float, optional): Vertical spacing. Default is 0.3.
        
    Returns:
        Tuple: (fig, axes) where axes is a 2D array of matplotlib Axes objects.
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)
    
    # Ensure axes is always 2D for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    return fig, axes


def create_comparison_grid(n_stats: int, figsize: tuple = (14, 0), 
                          height_per_stat: float = 3.0,
                          suptitle: str = None) -> Tuple:
    """
    Create a specialized grid layout for comparing statistics between two datasets,
    with paired panels (mean and variance) and ratio subpanels.
    
    Args:
        n_stats (int): Number of statistics to display (rows in the grid).
        figsize (tuple, optional): Figure size (width, height). If height is 0,
                                  it will be calculated based on n_stats and height_per_stat.
        height_per_stat (float, optional): Height allocated per statistic. Default is 3.0.
        suptitle (str, optional): Super title for the entire figure.
        
    Returns:
        Tuple: (fig, ax_upper_left, ax_upper_right, ax_lower_left, ax_lower_right) where each ax is a list
               of subplot axes for the respective panel positions.
    """
    # Calculate figure height if not specified
    if figsize[1] == 0:
        figsize = (figsize[0], n_stats * height_per_stat)
    
    fig = plt.figure(figsize=figsize)
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=0.93)
    
    # Create master grid for all statistics
    gs_master = GridSpec(nrows=n_stats, ncols=2, hspace=0.2, wspace=0.2)
    
    # Create sub-grids for each statistic with upper (main) and lower (ratio) panels
    gs_left = [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[i, 0], 
                                     height_ratios=[3, 1], hspace=0.1) 
             for i in range(n_stats)]
    
    gs_right = [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[i, 1], 
                                      height_ratios=[3, 1], hspace=0.1) 
              for i in range(n_stats)]
    
    # Create all subplot axes
    ax_upper_left = [fig.add_subplot(gs_left[i][0]) for i in range(n_stats)]
    ax_upper_right = [fig.add_subplot(gs_right[i][0]) for i in range(n_stats)]
    ax_lower_left = [fig.add_subplot(gs_left[i][1]) for i in range(n_stats)]
    ax_lower_right = [fig.add_subplot(gs_right[i][1]) for i in range(n_stats)]
    
    return fig, ax_upper_left, ax_upper_right, ax_lower_left, ax_lower_right


def configure_comparison_axes(ax_sets: List[List[plt.Axes]], 
                             x_scale: str = 'log', y_scale: str = 'log',
                             x_limits: Tuple[float, float] = None,
                             y_limits_upper: List[Tuple[float, float]] = None,
                             y_limits_lower: List[Tuple[float, float]] = None,
                             x_ticks: List[float] = None,
                             x_ticklabels: List[str] = None,
                             labels: List[str] = None,
                             titles: List[str] = None,
                             fontsize: int = 16) -> None:
    """
    Configure axes for a comparison plot grid created by create_comparison_grid.
    
    Args:
        ax_sets (List[List[plt.Axes]]): List of axis sets [ax_upper_left, ax_upper_right, ax_lower_left, ax_lower_right]
        x_scale (str, optional): Scale for x-axis ('linear' or 'log'). Default is 'log'.
        y_scale (str, optional): Scale for y-axis of upper panels. Default is 'log'.
        x_limits (Tuple[float, float], optional): Limits for x-axis.
        y_limits_upper (List[Tuple[float, float]], optional): List of y-limits for upper panels.
        y_limits_lower (List[Tuple[float, float]], optional): List of y-limits for lower panels.
        x_ticks (List[float], optional): Locations for x-ticks.
        x_ticklabels (List[str], optional): Labels for x-ticks.
        labels (List[str], optional): Names of statistics for ylabel generation.
        titles (List[str], optional): Formatted titles for statistics.
        fontsize (int, optional): Font size for labels. Default is 16.
    """
    ax_upper_left, ax_upper_right, ax_lower_left, ax_lower_right = ax_sets
    n_stats = len(ax_upper_left)
    
    for ax_set in [ax_upper_left, ax_upper_right, ax_lower_left, ax_lower_right]:
        for k_ax, ax in enumerate(ax_set):
            # Set scales
            ax.set_xscale(x_scale)
            
            # Set x limits if provided
            if x_limits:
                ax.set_xlim(x_limits)
            
            # Set x ticks if provided
            if x_ticks:
                ax.set_xticks(x_ticks)
            
            # Configure upper vs lower panels differently
            if ax_set in [ax_upper_left, ax_upper_right]:
                # Upper panels (means/variances)
                ax.set_yscale(y_scale)
                ax.set_xticklabels([])  # Hide x tick labels for upper panels
            else:
                # Lower panels (ratios)
                ax.hlines(1, x_limits[0] if x_limits else ax.get_xlim()[0], 
                        x_limits[1] if x_limits else ax.get_xlim()[1], 
                        color="black", linestyle="--")
                
                # Set y limits for ratio panels if provided
                if y_limits_lower and k_ax < len(y_limits_lower):
                    ax.set_ylim(y_limits_lower[k_ax])
                
                # Add x labels to bottom panels
                if k_ax == n_stats - 1:  # Last row
                    if x_scale == 'log':
                        ax.set_xlabel(r"$\ell$", fontsize=fontsize)
                    else:
                        ax.set_xlabel(r"$\nu$", fontsize=fontsize)
                    
                    if x_ticklabels:
                        ax.set_xticklabels(x_ticklabels)
                else:
                    ax.tick_params(bottom=False, labelbottom=False)
            
            # Set y limits for upper panels if provided
            if ax_set in [ax_upper_left, ax_upper_right] and y_limits_upper and k_ax < len(y_limits_upper):
                ax.set_ylim(y_limits_upper[k_ax])
            
            # Set y labels
            if titles and labels and k_ax < len(titles) and k_ax < len(labels):
                if ax_set is ax_upper_left:
                    ax.set_ylabel(f"$\mu$: {titles[k_ax]}", fontsize=fontsize)
                elif ax_set is ax_lower_left:
                    ax.set_ylabel(f"$R_\mu$: {titles[k_ax]}", fontsize=fontsize)
                elif ax_set is ax_upper_right:
                    ax.set_ylabel(f"$\sigma^2$: {titles[k_ax]}", fontsize=fontsize)
                elif ax_set is ax_lower_right:
                    ax.set_ylabel(f"$R_{{\sigma^2}}$: {titles[k_ax]}", fontsize=fontsize)


def add_comparison_legend(fig: plt.Figure, 
                         redshift_values: List[float] = None,
                         sim_types: List[str] = None,
                         colors: List[str] = None,
                         linestyles: List[str] = None,
                         y_position: float = 0.07,
                         fontsize: int = 12) -> None:
    """
    Add a legend to a comparison figure showing redshift values and simulation types.
    
    Args:
        fig (plt.Figure): The figure to add the legend to.
        redshift_values (List[float], optional): List of redshift values.
        sim_types (List[str], optional): Names of simulation types (e.g., ["Bigbox", "Tiled"]).
        colors (List[str], optional): List of colors for redshift values.
        linestyles (List[str], optional): List of linestyles for simulation types.
        y_position (float, optional): Vertical position of the legend. Default is 0.07.
        fontsize (int, optional): Font size for legend. Default is 12.
    """
    legend_elements = []
    
    # Add redshift custom lines if provided
    if redshift_values and colors:
        # Default to standard matplotlib colors if not enough provided
        if len(colors) < len(redshift_values):
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        redshift_lines = [Line2D([0], [0], color=colors[i], lw=2) 
                        for i in range(len(redshift_values))]
        legend_elements.extend(redshift_lines)
    
    # Add simulation type custom lines if provided
    if sim_types and linestyles:
        sim_lines = [Line2D([0], [0], linestyle=ls, lw=2, color="black") 
                   for ls in linestyles[:len(sim_types)]]
        legend_elements.extend(sim_lines)
    
    # Create labels
    labels = []
    if redshift_values:
        labels.extend([f"$z_s = {z}$" for z in redshift_values])
    if sim_types:
        labels.extend(sim_types)
    
    # Add the legend
    if legend_elements and labels:
        fig.legend(legend_elements, labels, bbox_to_anchor=(0.5, y_position),
                loc='upper center', ncol=len(labels), fontsize=fontsize)


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


# General configuration for plots
def set_plotting_defaults(fontsize: int = 14):
    """
    Set default plotting parameters for consistent visualization.
    
    Args:
        fontsize (int, optional): Base fontsize for plots. Default is 14.
    """
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['axes.titlesize'] = fontsize + 2
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize - 2
    plt.rcParams['ytick.labelsize'] = fontsize - 2
    plt.rcParams['legend.fontsize'] = fontsize - 2


def create_custom_legend(fig: plt.Figure, lines: List[tuple], labels: List[str], 
                         bbox_to_anchor: Tuple[float, float] = (0.5, 0.95),
                         ncol: int = None, fontsize: int = 12, loc: str = 'upper center'):
    """
    Create a custom legend with specified lines and labels.
    
    Args:
        fig (plt.Figure): Figure to add the legend to
        lines (List[tuple]): List of (linestyle, color, marker) tuples
        labels (List[str]): Labels for each line
        bbox_to_anchor (Tuple[float, float], optional): Position of the legend
        ncol (int, optional): Number of columns. If None, uses len(labels)
        fontsize (int, optional): Font size for legend text
        loc (str, optional): Legend location. Default is 'upper center'
    """
    if ncol is None:
        ncol = len(labels)
    
    custom_lines = [
        Line2D([0], [0], linestyle=ls, color=color, marker=marker, lw=2) 
        for ls, color, marker in lines
    ]
    
    fig.legend(custom_lines, labels, bbox_to_anchor=bbox_to_anchor,
              loc=loc, ncol=ncol, fontsize=fontsize) 


def plot_rmp_rip_ratios(means_rmp: Dict, means_rip: Dict,
                      stds_rmp: Dict, stds_rip: Dict,
                      zs_list: List[float], box_type: List[str],
                      ngal_comp_plot: str, sl_comp_plot: str,
                      ell: np.ndarray, nu: np.ndarray,
                      labels_ell: List[str], labels_nu: List[str],
                      titles_ell: List[str], titles_nu: List[str],
                      colors: List[str], output_path: str = None,
                      mranges_ell: List[Tuple] = None, vranges_ell: List[Tuple] = None,
                      mranges_nu: List[Tuple] = None, vranges_nu: List[Tuple] = None,
                      fontsize: int = 16):
    """
    Plot ratios of statistics between BIGBOX and TILED simulations for RMP and RIP patches.
    
    Args:
        means_rmp (Dict): Dictionary of means for Regular Meaningful Patches (RMP)
        means_rip (Dict): Dictionary of means for Random Invalid Patches (RIP)
        stds_rmp (Dict): Dictionary of standard deviations for RMP
        stds_rip (Dict): Dictionary of standard deviations for RIP
        zs_list (List[float]): List of redshift values to plot
        box_type (List[str]): Box types (e.g., ["bigbox", "tiled"])
        ngal_comp_plot (str): Galaxy density configuration key (e.g., "ngal_0")
        sl_comp_plot (str): Smoothing length configuration key (e.g., "sl_2")
        ell (np.ndarray): Array of ell values for power spectrum plots
        nu (np.ndarray): Array of nu values for PDF/peaks/etc plots
        labels_ell (List[str]): Labels for ell-based statistics
        labels_nu (List[str]): Labels for nu-based statistics
        titles_ell (List[str]): Formatted titles for ell-based statistics
        titles_nu (List[str]): Formatted titles for nu-based statistics
        colors (List[str]): Colors to use for different redshifts
        output_path (str, optional): Path to save the output figure
        mranges_ell (List[Tuple], optional): Y-axis ranges for means of ell statistics
        vranges_ell (List[Tuple], optional): Y-axis ranges for variances of ell statistics
        mranges_nu (List[Tuple], optional): Y-axis ranges for means of nu statistics
        vranges_nu (List[Tuple], optional): Y-axis ranges for variances of nu statistics
        fontsize (int, optional): Font size for text elements
        
    Returns:
        plt.Figure: The created figure
    """
    if ell is None or nu is None or len(ell) == 0 or len(nu) == 0:
        raise ValueError("ell and nu arrays must be provided and non-empty")
    
    # Create figure and grid layout
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(r"Ratio of Statistics between BIGBOX and TILED: RMP and RIP", 
                fontsize=18, y=0.98)
    
    gs_master = GridSpec(2, 2, height_ratios=[len(labels_ell), len(labels_nu)], 
                       width_ratios=[1, 1], wspace=0.2, hspace=0.2)
    
    gs_ell_m = GridSpecFromSubplotSpec(len(labels_ell), 1, 
                                    subplot_spec=gs_master[0, 0], hspace=0.2)
    gs_ell_v = GridSpecFromSubplotSpec(len(labels_ell), 1, 
                                    subplot_spec=gs_master[0, 1], hspace=0.2)
    gs_nu_m = GridSpecFromSubplotSpec(len(labels_nu), 1, 
                                   subplot_spec=gs_master[1, 0], hspace=0.2)
    gs_nu_v = GridSpecFromSubplotSpec(len(labels_nu), 1, 
                                   subplot_spec=gs_master[1, 1], hspace=0.2)
    
    # Create axes
    ax_ell_m = [fig.add_subplot(gs_ell_m[i, 0]) for i in range(len(labels_ell))]
    ax_ell_v = [fig.add_subplot(gs_ell_v[i, 0]) for i in range(len(labels_ell))]
    ax_nu_m = [fig.add_subplot(gs_nu_m[i, 0]) for i in range(len(labels_nu))]
    ax_nu_v = [fig.add_subplot(gs_nu_v[i, 0]) for i in range(len(labels_nu))]
    
    # Plot data for RMP and RIP with different line styles
    for means_set, stds_set, line_style in [(means_rmp, stds_rmp, "-"), 
                                          (means_rip, stds_rip, "--")]:
        for i, zs_i in enumerate(zs_list):
            # Plot nu-based statistics (PDF, peaks, etc)
            for j, label_nu in enumerate(labels_nu):
                if (('bigbox', zs_i) in means_set and (ngal_comp_plot, sl_comp_plot) in means_set[('bigbox', zs_i)] and
                    ('tiled', zs_i) in means_set and (ngal_comp_plot, sl_comp_plot) in means_set[('tiled', zs_i)]):
                    
                    if (label_nu in means_set[('bigbox', zs_i)][(ngal_comp_plot, sl_comp_plot)] and 
                        label_nu in means_set[('tiled', zs_i)][(ngal_comp_plot, sl_comp_plot)] and
                        label_nu in stds_set[('bigbox', zs_i)][(ngal_comp_plot, sl_comp_plot)] and 
                        label_nu in stds_set[('tiled', zs_i)][(ngal_comp_plot, sl_comp_plot)]):
                        
                        # Extract and process data
                        bb_m = means_set[('bigbox', zs_i)][(ngal_comp_plot, sl_comp_plot)][label_nu]
                        ti_m = means_set[('tiled', zs_i)][(ngal_comp_plot, sl_comp_plot)][label_nu]
                        bb_s = stds_set[('bigbox', zs_i)][(ngal_comp_plot, sl_comp_plot)][label_nu]
                        ti_s = stds_set[('tiled', zs_i)][(ngal_comp_plot, sl_comp_plot)][label_nu]
                        
                        # Convert to 1D if needed
                        bb_m_1d = bb_m.mean(axis=0) if bb_m.ndim > 1 else bb_m
                        ti_m_1d = ti_m.mean(axis=0) if ti_m.ndim > 1 else ti_m
                        bb_s_1d = bb_s.mean(axis=0) if bb_s.ndim > 1 else bb_s
                        ti_s_1d = ti_s.mean(axis=0) if ti_s.ndim > 1 else ti_s
                        
                        # Plot the ratios
                        ax_nu_m[j].plot(nu, bb_m_1d/ti_m_1d, color=colors[i], linestyle=line_style)
                        ax_nu_v[j].plot(nu, (bb_s_1d**2)/(ti_s_1d**2), color=colors[i], linestyle=line_style)
            
            # Plot ell-based statistics (power spectrum, bispectra)
            for j, label_ell in enumerate(labels_ell):
                if (('bigbox', zs_i) in means_set and (ngal_comp_plot, sl_comp_plot) in means_set[('bigbox', zs_i)] and
                    ('tiled', zs_i) in means_set and (ngal_comp_plot, sl_comp_plot) in means_set[('tiled', zs_i)]):
                    
                    if (label_ell in means_set[('bigbox', zs_i)][(ngal_comp_plot, sl_comp_plot)] and 
                        label_ell in means_set[('tiled', zs_i)][(ngal_comp_plot, sl_comp_plot)] and
                        label_ell in stds_set[('bigbox', zs_i)][(ngal_comp_plot, sl_comp_plot)] and 
                        label_ell in stds_set[('tiled', zs_i)][(ngal_comp_plot, sl_comp_plot)]):
                        
                        # Extract and process data
                        bb_m = means_set[('bigbox', zs_i)][(ngal_comp_plot, sl_comp_plot)][label_ell]
                        ti_m = means_set[('tiled', zs_i)][(ngal_comp_plot, sl_comp_plot)][label_ell]
                        bb_s = stds_set[('bigbox', zs_i)][(ngal_comp_plot, sl_comp_plot)][label_ell]
                        ti_s = stds_set[('tiled', zs_i)][(ngal_comp_plot, sl_comp_plot)][label_ell]
                        
                        # Convert to 1D if needed
                        bb_m_1d = bb_m.mean(axis=0) if bb_m.ndim > 1 else bb_m
                        ti_m_1d = ti_m.mean(axis=0) if ti_m.ndim > 1 else ti_m
                        bb_s_1d = bb_s.mean(axis=0) if bb_s.ndim > 1 else bb_s
                        ti_s_1d = ti_s.mean(axis=0) if ti_s.ndim > 1 else ti_s
                        
                        # Plot the ratios
                        ax_ell_m[j].plot(ell, bb_m_1d/ti_m_1d, color=colors[i], linestyle=line_style)
                        ax_ell_v[j].plot(ell, (bb_s_1d**2)/(ti_s_1d**2), color=colors[i], linestyle=line_style)
    
    # Format axes
    for j, (ax, title) in enumerate(zip(ax_ell_m, titles_ell)):
        ax.set_xscale('log')
        ax.set_title(f"Mean: {title}", fontsize=fontsize)
        ax.set_ylabel(f"Ratio", fontsize=fontsize)
        ax.axhline(y=1.0, linestyle=':', color='black', alpha=0.5)
        if mranges_ell and j < len(mranges_ell):
            ax.set_ylim(mranges_ell[j])
        if j < len(ax_ell_m) - 1:
            ax.set_xticklabels([])
    
    for j, (ax, title) in enumerate(zip(ax_ell_v, titles_ell)):
        ax.set_xscale('log')
        ax.set_title(f"Variance: {title}", fontsize=fontsize)
        ax.set_ylabel(f"Ratio", fontsize=fontsize)
        ax.axhline(y=1.0, linestyle=':', color='black', alpha=0.5)
        if vranges_ell and j < len(vranges_ell):
            ax.set_ylim(vranges_ell[j])
        if j < len(ax_ell_v) - 1:
            ax.set_xticklabels([])
    
    for j, (ax, title) in enumerate(zip(ax_nu_m, titles_nu)):
        ax.set_title(f"Mean: {title}", fontsize=fontsize)
        ax.set_ylabel(f"Ratio", fontsize=fontsize)
        ax.axhline(y=1.0, linestyle=':', color='black', alpha=0.5)
        if mranges_nu and j < len(mranges_nu):
            ax.set_ylim(mranges_nu[j])
        if j < len(ax_nu_m) - 1:
            ax.set_xticklabels([])
    
    for j, (ax, title) in enumerate(zip(ax_nu_v, titles_nu)):
        ax.set_title(f"Variance: {title}", fontsize=fontsize)
        ax.set_ylabel(f"Ratio", fontsize=fontsize)
        ax.axhline(y=1.0, linestyle=':', color='black', alpha=0.5)
        if vranges_nu and j < len(vranges_nu):
            ax.set_ylim(vranges_nu[j])
        if j < len(ax_nu_v) - 1:
            ax.set_xticklabels([])
    
    # Add x-labels to bottom plots
    ax_ell_m[-1].set_xlabel(r"$\ell$", fontsize=fontsize)
    ax_ell_v[-1].set_xlabel(r"$\ell$", fontsize=fontsize)
    ax_nu_m[-1].set_xlabel(r"$\nu$", fontsize=fontsize)
    ax_nu_v[-1].set_xlabel(r"$\nu$", fontsize=fontsize)
    
    # Add legend
    custom_lines = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(zs_list))] + \
                   [Line2D([0], [0], ls=ls, lw=2, color="black") for ls in ["-", "--"]]
    
    fig.legend(custom_lines, 
             [f"$z_s={z}$" for z in zs_list] + ["RMPs", "RIPs"],
             bbox_to_anchor=(0.5, 0.95), loc='upper center', 
             ncol=len(zs_list) + 2, fontsize=fontsize-4)
    
    # Save if path is provided
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    
    return fig 


def plot_bad_patch_comparison(means_rmp: Dict, means_rip: Dict, means_bl: Dict,
                            zs_value: float, box_type: str, 
                            ngal_comp_plot: str, sl_comp_plot: str,
                            ell: np.ndarray, labels_ell: List[str], 
                            titles_ell: List[str], output_path: str = None,
                            fontsize: int = 16):
    """
    Compare statistics between Regular Meaningful Patches (RMP), 
    Random Invalid Patches (RIP), and known bad patches.
    
    Args:
        means_rmp (Dict): Dictionary of means for Regular Meaningful Patches
        means_rip (Dict): Dictionary of means for Random Invalid Patches
        means_bl (Dict): Dictionary of means for bad list patches
        zs_value (float): Redshift value to analyze
        box_type (str): Box type to analyze (e.g., "bigbox")
        ngal_comp_plot (str): Galaxy density configuration key (e.g., "ngal_0")
        sl_comp_plot (str): Smoothing length configuration key (e.g., "sl_2")
        ell (np.ndarray): Array of ell values for power spectrum plots
        labels_ell (List[str]): Labels for ell-based statistics
        titles_ell (List[str]): Formatted titles for ell-based statistics
        output_path (str, optional): Path to save the output figure
        fontsize (int, optional): Font size for text elements
        
    Returns:
        plt.Figure: The created figure
    """
    if ell is None or len(ell) == 0:
        raise ValueError("ell array must be provided and non-empty")
    
    # Create figure with a row of subplots
    fig, axes = plt.subplots(1, len(labels_ell), figsize=(20, 4), sharey=True)
    fig.suptitle(f"Comparison of Bad Patches vs RMP/RIP (z={zs_value}, {box_type})", 
                fontsize=fontsize)
    
    # Define line styles and colors for the different patch types
    styles = {
        "RMP": dict(color="blue", linestyle="-", label="RMP"),
        "RIP": dict(color="red", linestyle="--", label="RIP"),
        "Bad": dict(color="black", linestyle=":", label="Bad Patches")
    }
    
    # Plot each statistic in its own subplot
    for j, (label_ell, title_ell) in enumerate(zip(labels_ell, titles_ell)):
        ax = axes[j] if len(labels_ell) > 1 else axes
        
        # Check if data is available for all patch types
        data_available = all(
            data_dict.get((box_type, zs_value), {}).get((ngal_comp_plot, sl_comp_plot), {}).get(label_ell) is not None
            for data_dict in [means_rmp, means_rip, means_bl]
        )
        
        if data_available:
            # Extract and plot data for each patch type
            for data_dict, patch_type in zip([means_rmp, means_rip, means_bl], styles.keys()):
                data = data_dict[(box_type, zs_value)][(ngal_comp_plot, sl_comp_plot)][label_ell]
                data_1d = data.mean(axis=0) if data.ndim > 1 else data
                
                # Ensure data length matches ell length
                plot_data = data_1d[:len(ell)] if len(data_1d) > len(ell) else data_1d
                plot_ell = ell[:len(plot_data)]
                
                # Plot the data
                ax.plot(plot_ell, plot_data, **styles[patch_type])
            
            # Set subplot title and format
            ax.set_title(title_ell, fontsize=fontsize)
            ax.set_xscale('log')
            
            # Only add y-label to leftmost plot
            if j == 0:
                ax.set_ylabel("Value", fontsize=fontsize)
            
            # Always add x-label
            ax.set_xlabel(r"$\ell$", fontsize=fontsize)
        else:
            ax.text(0.5, 0.5, "Data not available", 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes)
    
    # Add a single legend for the entire figure
    handles = [plt.Line2D([0], [0], **style) for style in styles.values()]
    fig.legend(handles, [style["label"] for style in styles.values()],
              loc='upper center', bbox_to_anchor=(0.5, 0.95), 
              ncol=len(styles), fontsize=fontsize-2)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    
    # Save if path is provided
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    
    return fig 


def process_statistics_set(combined_data: Dict, 
                         output_means: Dict, output_stds: Dict, 
                         output_correlations: Dict, output_covariances: Dict,
                         set_label: str = ""):
    """
    Process a set of statistical data to calculate means, standard deviations,
    correlations, and covariances.
    
    Args:
        combined_data (Dict): Dictionary containing the raw statistics
        output_means (Dict): Dictionary to store calculated means
        output_stds (Dict): Dictionary to store calculated standard deviations
        output_correlations (Dict): Dictionary to store calculated correlation matrices
        output_covariances (Dict): Dictionary to store calculated covariance matrices
        set_label (str, optional): Label for logging purposes
        
    Returns:
        None: Results are stored in the provided output dictionaries
    """
    for box_set, zs_set in combined_data.keys():
        for (ngal_cfg, sl_cfg), stats_dict in combined_data[(box_set, zs_set)].items():
            for stat_type, stat_arrays in stats_dict.items():
                # Skip empty or invalid data
                if not stat_arrays or not isinstance(stat_arrays[0], np.ndarray) or stat_arrays[0].size == 0:
                    continue
                
                try:
                    # Stack arrays for statistical calculations
                    stacked = np.vstack(stat_arrays)
                except ValueError as e:
                    print(f"VStack {set_label} Error: {e}")
                    continue
                
                if stacked.size == 0:
                    continue
                
                # Initialize dictionaries if needed
                for d_init in [output_means, output_stds, output_covariances, output_correlations]:
                    d_init.setdefault((box_set, zs_set), {}).setdefault((ngal_cfg, sl_cfg), {})
                
                # Calculate statistics
                output_means[(box_set, zs_set)][(ngal_cfg, sl_cfg)][stat_type] = np.mean(stacked, axis=0)
                output_stds[(box_set, zs_set)][(ngal_cfg, sl_cfg)][stat_type] = np.std(stacked, axis=0)
                
                # Calculate covariance and correlation if enough samples
                if stacked.shape[0] > 1 and stacked.shape[1] > 0:
                    output_covariances[(box_set, zs_set)][(ngal_cfg, sl_cfg)][stat_type] = np.cov(stacked, rowvar=False)
                    output_correlations[(box_set, zs_set)][(ngal_cfg, sl_cfg)][stat_type] = np.corrcoef(stacked, rowvar=False)
                else:
                    # Use NaN matrices if not enough samples
                    nan_matrix = np.full((stacked.shape[1], stacked.shape[1]), np.nan)
                    output_covariances[(box_set, zs_set)][(ngal_cfg, sl_cfg)][stat_type] = nan_matrix
                    output_correlations[(box_set, zs_set)][(ngal_cfg, sl_cfg)][stat_type] = nan_matrix


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