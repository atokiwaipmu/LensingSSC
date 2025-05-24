# lensing_ssc/stats/peak_counts.py
import numpy as np
from lenstools import ConvergenceMap
from astropy import units as u
from typing import Tuple

def _exclude_edges(heights: np.ndarray, positions: np.ndarray, 
                  patch_size_deg: float, x_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exclude positions too close to the patch edges.
    
    Parameters
    ----------
    heights : np.ndarray
        Peak/minima heights.
    positions : np.ndarray
        Peak/minima positions (in astropy.units.deg).
    patch_size_deg : float
        Patch size in degrees.
    x_size : int
        Patch size in pixels along one dimension.
        
    Returns
    -------
    tuple
        (filtered_heights, filtered_positions_pixels)
        Filtered positions are in pixel coordinates.
    """
    # Convert positions from degrees to pixel coordinates
    # The `positions` from locatePeaks are in degrees relative to the patch center, so scaling is direct.
    # However, locatePeaks returns angular positions, not pixel indices directly.
    # For now, assuming positions are scaled appropriately or this function adapts if positions are pixel indices.
    # If positions are angular offsets from patch center, conversion to pixel coordinates depends on projection.
    # Given the original code `positions.value * xsize / patch_size`, it implies positions were angular.
    # Let's assume positions are angular extent or similar that can be scaled.
    # If positions output by locatePeaks are already pixel indices, this scaling might be incorrect.
    # For safety, we assume positions are in degrees as per astropy units typically used with ConvergenceMap.
    
    # To correctly map angular positions to pixel coordinates in a patch:
    # One needs to know the pixel scale (e.g., arcmin/pixel or deg/pixel).
    # pixel_scale_deg = patch_size_deg / x_size
    # For a position (dx, dy) in degrees from the center of the patch:
    # px = (dx / pixel_scale_deg) + x_size / 2
    # py = (dy / pixel_scale_deg) + y_size / 2 (assuming y_size = x_size)
    
    # The original `tmp_positions = positions.value * xsize / patch_size` seems to assume
    # that `positions.value` is a fraction of `patch_size` that directly maps to `xsize`.
    # This is only true if `positions` were normalized coordinates [0,1]*patch_size.
    # `lenstools.ConvergenceMap.locatePeaks` returns positions in angular units of the map.
    # Let's stick to the original logic for now, assuming it was working in its context.
    
    tmp_positions_scaled = positions.to(u.deg).value * x_size / patch_size_deg
    
    # Create a mask to exclude edge pixels.
    # Boundary is 1 pixel from the edge: >0 and < x_size-1
    mask = (tmp_positions_scaled[:, 0] > 0) & (tmp_positions_scaled[:, 0] < x_size - 1) & \
           (tmp_positions_scaled[:, 1] > 0) & (tmp_positions_scaled[:, 1] < x_size - 1)
           
    return heights[mask], tmp_positions_scaled[mask].astype(int)


def calculate_peak_counts(conv_map: ConvergenceMap, bins: np.ndarray, 
                            is_minima: bool = False) -> np.ndarray:
    """
    Compute peak or minima counts (normalized histogram) for a convergence map.
    
    Parameters
    ----------
    conv_map : ConvergenceMap
        Input convergence map. Should be appropriately smoothed and/or normalized 
        (e.g., to an SNR map) before calling this function, as peak statistics
        are sensitive to map preprocessing.
    bins : np.ndarray
        The bin edges for the peak height histogram.
    is_minima : bool, optional
        If True, compute minima counts by inverting the map. Defaults to False.
        
    Returns
    -------
    np.ndarray
        Normalized peak/minima counts (PDF-like histogram).
    """
    map_to_analyze = conv_map
    if is_minima:
        # Invert the map for minima computation by creating a new ConvergenceMap
        map_to_analyze = ConvergenceMap(-conv_map.data, angle=conv_map.angle)

    # locatePeaks returns: (heights, positions)
    # heights: peak heights
    # positions: astropy Quantity array of peak positions (e.g., in degrees)
    peak_heights, peak_positions = map_to_analyze.locatePeaks(bins)
    
    # Ensure map angle and side_pixels are available for _exclude_edges
    patch_size_deg = conv_map.angle.to(u.deg).value
    # Assuming square map, xsize is one dimension of the data array
    x_size = conv_map.data.shape[0] 
    
    # Exclude peaks near the edges
    # Note: _exclude_edges expects patch_size_deg and x_size from the original map context.
    filtered_heights, _ = _exclude_edges(peak_heights, peak_positions, 
                                           patch_size_deg, x_size)
    
    # Compute the histogram of peak heights
    counts, _ = np.histogram(filtered_heights, bins=bins)
    
    # Normalize the counts to represent a PDF-like distribution
    # Sum of counts gives total number of peaks in the filtered set.
    # Bin width is assumed constant for this normalization.
    total_peaks = np.sum(counts)
    bin_width = bins[1] - bins[0]
    
    if total_peaks > 0:
        normalized_counts = counts / total_peaks / bin_width
    else:
        normalized_counts = np.zeros_like(counts, dtype=float)

    if is_minima:
        # For minima, the heights were from the inverted map.
        # If bins are, e.g., [-4, -3, ..., 3, 4], and we inverted the map,
        # a peak at nu=3 in inverted map corresponds to a minimum at nu=-3 in original.
        # So, we reverse the resulting histogram to match the original bin definition for minima.
        normalized_counts = normalized_counts[::-1]
        
    return normalized_counts 