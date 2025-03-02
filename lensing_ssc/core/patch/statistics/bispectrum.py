import numpy as np
from lenstools import ConvergenceMap


def dimensionless_bispectrum(bispec: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """
    Convert bispectrum to dimensionless form.
    
    Parameters
    ----------
    bispec : np.ndarray
        Input bispectrum values
    ell : np.ndarray
        Multipole values
        
    Returns
    -------
    np.ndarray
        Dimensionless bispectrum
    """
    return bispec * ell**4 / (2*np.pi)**2


def compute_bispectrum(conv_map: ConvergenceMap, l_edges: np.ndarray, ell: np.ndarray) -> tuple:
    """
    Compute bispectrum in different configurations.
    
    Parameters
    ----------
    conv_map : ConvergenceMap
        Input convergence map
    l_edges : np.ndarray
        Edges of multipole bins
    ell : np.ndarray
        Center values of multipole bins
        
    Returns
    -------
    tuple
        (equilateral, isosceles, squeezed) bispectra
    """
    equilateral = conv_map.bispectrum(l_edges, configuration='equilateral')[1]
    isosceles = conv_map.bispectrum(l_edges, ratio=0.5, configuration='folded')[1]
    squeezed = conv_map.bispectrum(l_edges, ratio=0.1, configuration='folded')[1]

    equilateral = np.abs(dimensionless_bispectrum(equilateral, ell))
    isosceles = np.abs(dimensionless_bispectrum(isosceles, ell))
    squeezed = np.abs(dimensionless_bispectrum(squeezed, ell))

    return equilateral, isosceles, squeezed