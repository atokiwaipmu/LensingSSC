# lensing_ssc/stats/power_spectrum.py
import numpy as np
from astropy import units as u
from lenstools import ConvergenceMap
from typing import Tuple

def calculate_power_spectrum(conv_map: ConvergenceMap, l_edges: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """
    Calculates the power spectrum of a convergence map.

    Parameters
    ----------
    conv_map : ConvergenceMap
        The input convergence map.
    l_edges : np.ndarray
        The edges of the multipole bins.
    ell : np.ndarray
        The central values of the multipole bins.

    Returns
    -------
    np.ndarray
        The calculated power spectrum (Cl * l * (l+1) / (2*pi)).
    """
    # The powerSpectrum method returns a tuple: (ell_centers, ps)
    # We take the power spectrum part (ps) which is at index 1.
    ps = conv_map.powerSpectrum(l_edges)[1]
    # Apply the conventional scaling for Cl
    cl_scaled = ps * ell * (ell + 1) / (2 * np.pi)
    return cl_scaled 