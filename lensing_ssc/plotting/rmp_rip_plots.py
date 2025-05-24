"""
RMP/RIP analysis plots for lensing-ssc.

This module contains functions for visualizing Regular Meaningful Patches (RMP) and
Repeated Impact Patches (RIP) analysis results.
"""

from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


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