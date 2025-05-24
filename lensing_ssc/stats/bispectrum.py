# lensing_ssc/stats/bispectrum.py
import numpy as np
from lenstools import ConvergenceMap
from typing import Tuple

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


def calculate_bispectrum(conv_map: ConvergenceMap, l_edges: np.ndarray, ell: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bispectrum in different configurations (equilateral, isosceles, squeezed).
    
    Parameters
    ----------
    conv_map : ConvergenceMap
        Input convergence map.
    l_edges : np.ndarray
        Edges of multipole bins.
    ell : np.ndarray
        Center values of multipole bins.
        
    Returns
    -------
    tuple
        A tuple containing the (equilateral, isosceles, squeezed) dimensionless bispectra.
    """
    # Equilateral configuration
    # bispectrum returns a tuple: (ell_centers, B_ell)
    equilateral_b = conv_map.bispectrum(l_edges, configuration='equilateral')[1]
    
    # Isosceles configuration (using folded with a specific ratio)
    # For isosceles, lenstools uses 'folded' configuration with a ratio. Common default is ratio=0.5
    isosceles_b = conv_map.bispectrum(l_edges, ratio=0.5, configuration='folded')[1]
    
    # Squeezed configuration (using folded with a small ratio)
    # For squeezed, lenstools uses 'folded' configuration with a small ratio, e.g., 0.1
    squeezed_b = conv_map.bispectrum(l_edges, ratio=0.1, configuration='folded')[1]

    # Convert to dimensionless and take absolute value
    equilateral_dim = np.abs(dimensionless_bispectrum(equilateral_b, ell))
    isosceles_dim = np.abs(dimensionless_bispectrum(isosceles_b, ell))
    squeezed_dim = np.abs(dimensionless_bispectrum(squeezed_b, ell))

    return equilateral_dim, isosceles_dim, squeezed_dim 