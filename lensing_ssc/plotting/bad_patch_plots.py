"""
Bad patch analysis plots for lensing-ssc.

This module contains functions for visualizing the analysis of problematic patches
and comparing them with other patch types.
"""

from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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