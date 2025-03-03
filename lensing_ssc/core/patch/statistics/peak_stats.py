import numpy as np
from lenstools import ConvergenceMap
from astropy import units as u


def exclude_edges(heights: np.ndarray, positions: np.ndarray, 
                 patch_size: float, xsize: int) -> tuple:
    """
    Exclude positions too close to the patch edges.
    
    Parameters
    ----------
    heights : np.ndarray
        Peak/minima heights
    positions : np.ndarray
        Peak/minima positions
    patch_size : float
        Patch size in degrees
    xsize : int
        Patch size in pixels
        
    Returns
    -------
    tuple
        (filtered_heights, filtered_positions)
    """
    # Scale positions to the patch size and apply boundary mask
    tmp_positions = positions.value * xsize / patch_size
    mask = (tmp_positions[:, 0] > 0) & (tmp_positions[:, 0] < xsize - 1) & \
           (tmp_positions[:, 1] > 0) & (tmp_positions[:, 1] < xsize - 1)
    return heights[mask], tmp_positions[mask].astype(int)


def compute_peak_statistics(snr_map: ConvergenceMap, bins: np.ndarray, 
                          is_minima: bool = False, 
                          patch_size: float = 10.0,
                          xsize: int = 2048) -> np.ndarray:
    """
    Compute peak or minima statistics.
    
    Parameters
    ----------
    snr_map : ConvergenceMap
        Input SNR map
    bins : np.ndarray
        Height bins
    is_minima : bool, optional
        Whether to compute minima instead of peaks
    patch_size : float, optional
        Patch size in degrees
    xsize : int, optional
        Patch size in pixels
        
    Returns
    -------
    np.ndarray
        Peak/minima PDF
    """
    if is_minima:
        # Invert the map for minima computation
        snr_map = ConvergenceMap(-snr_map.data, angle=patch_size * u.deg)

    height, positions = snr_map.locatePeaks(bins)
    height, positions = exclude_edges(height, positions, patch_size, xsize)
    peaks = np.histogram(height, bins=bins)[0]
    binwidth = bins[1] - bins[0]
    peaks = peaks / np.sum(peaks) / binwidth
    if is_minima:
        peaks = peaks[::-1]
    return peaks