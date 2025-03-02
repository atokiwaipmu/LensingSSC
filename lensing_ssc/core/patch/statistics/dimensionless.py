import numpy as np
from typing import Tuple, Any


def dimensionless_cl(cl: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """
    Converts a power spectrum to its dimensionless form.
    
    Parameters
    ----------
    cl : np.ndarray
        Raw power spectrum.
    ell : np.ndarray
        Multipole moments corresponding to the power spectrum.
        
    Returns
    -------
    np.ndarray
        Dimensionless power spectrum.
    """
    return ell * (ell + 1) * cl / (2 * np.pi)


def dimensionless_bispectrum(bispec: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """
    Converts a bispectrum to its dimensionless form.
    
    Parameters
    ----------
    bispec : np.ndarray
        Raw bispectrum.
    ell : np.ndarray
        Multipole moments corresponding to the bispectrum.
        
    Returns
    -------
    np.ndarray
        Dimensionless bispectrum.
    """
    return bispec * ell**4 / (2 * np.pi)**2


def dimensionless_moments(moments: Tuple[Any, ...], global_std: float) -> Tuple[np.ndarray, ...]:
    """
    Converts raw moments to dimensionless form.
    
    Parameters
    ----------
    moments : tuple
        Raw moments from the ConvergenceMap.
    global_std : float
        Global standard deviation used for normalization.
        
    Returns
    -------
    Tuple[np.ndarray, ...]
        Dimensionless skewness and kurtosis terms.
    """
    _, sigma1, S0, S1, S2, K0, K1, K2, K3 = moments
    sigma0 = global_std

    S0 = S0 / sigma0**3
    S1 = S1 / (sigma0 * sigma1**2)
    S2 = S2 * (sigma0 / sigma1**4)

    K0 = K0 / sigma0**4
    K1 = K1 / (sigma0**2 * sigma1**2)
    K2 = K2 / sigma1**4
    K3 = K3 / sigma1**4

    return S0, S1, S2, K0, K1, K2, K3