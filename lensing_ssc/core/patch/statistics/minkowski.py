import numpy as np
from scipy.special import erf
from lenstools import ConvergenceMap


def compute_gradient(data: np.ndarray) -> list:
    """
    Compute the gradient of a 2D array.
    
    Parameters
    ----------
    data : np.ndarray
        Input 2D array
        
    Returns
    -------
    list
        [gradient_y, gradient_x]
    """
    grad_y, grad_x = np.gradient(data)
    return [grad_y, grad_x]


def compute_hessian(data: np.ndarray) -> list:
    """
    Compute the Hessian matrix components of a 2D array.
    
    Parameters
    ----------
    data : np.ndarray
        Input 2D array
        
    Returns
    -------
    list
        [hessian_xx, hessian_yy, hessian_xy]
    """
    grad_y, grad_x = np.gradient(data)
    hess_yy, hess_yx = np.gradient(grad_y)
    hess_xy, hess_xx = np.gradient(grad_x)
    return [hess_xx, hess_yy, hess_xy]


def gaussian_minkowski_functionals(nu: np.ndarray, mu: float, sigma0: float, sigma1: float) -> tuple:
    """
    Compute Gaussian Minkowski functionals.
    
    Parameters
    ----------
    nu : np.ndarray
        Threshold values
    mu : float
        Mean value
    sigma0 : float
        Standard deviation
    sigma1 : float
        First-derivative measure
        
    Returns
    -------
    tuple
        (V0, V1, V2) Minkowski functionals
    """
    V0 = 0.5 * (1.0 - erf((nu - mu) / (np.sqrt(2) * sigma0)))
    V1 = (sigma1 / (8.0 * sigma0 * np.sqrt(2))) * np.exp(-((nu - mu) ** 2) / (2.0 * sigma0 ** 2))
    V2 = ((nu - mu) * (sigma1 ** 2) / (sigma0 ** 3) /
          (2.0 * (2.0 * np.pi) ** 1.5)) * np.exp(-((nu - mu) ** 2) / (2.0 * sigma0 ** 2))
    return V0, V1, V2


def compute_minkowski_functionals_from_arrays(data: np.ndarray, nu: np.ndarray, 
                                             grad: list, hess: list) -> tuple:
    """
    Compute Minkowski functionals from array data and precomputed gradients/Hessians.
    
    Parameters
    ----------
    data : np.ndarray
        Input 2D array
    nu : np.ndarray
        Threshold values
    grad : list
        [gradient_y, gradient_x]
    hess : list
        [hessian_xx, hessian_yy, hessian_xy]
        
    Returns
    -------
    tuple
        (V0, V1, V2) Minkowski functionals
    """
    gradient_y, gradient_x = grad
    hessian_xx, hessian_yy, hessian_xy = hess

    denominator = gradient_x**2 + gradient_y**2
    s1 = np.sqrt(denominator)
    
    # Avoid division by zero
    denominator = np.where(denominator == 0, np.nan, denominator)
    frac = (2.0 * gradient_x * gradient_y * hessian_xy - 
            gradient_x**2 * hessian_yy - 
            gradient_y**2 * hessian_xx) / denominator
    frac = np.nan_to_num(frac)  # Replace NaNs with zero

    delta = np.diff(nu)

    # Initialize Minkowski functionals
    V0 = np.zeros(len(delta))
    V1 = np.zeros(len(delta))
    V2 = np.zeros(len(delta))

    # Normalize by total number of pixels
    total_pixels = data.size

    # Precompute all masks at once
    masks = [(data > lower) & (data < upper) for lower, upper in zip(nu[:-1], nu[1:])]
    masks = np.array(masks)

    V0 = np.array([np.sum(data > threshold) for threshold in nu[:-1]]) / total_pixels
    
    # Vectorized computation for V1 and V2
    for i in range(len(delta)):
        mask = masks[i]
        V1[i] = np.sum(s1[mask]) / (4.0 * delta[i] * total_pixels)
        V2[i] = np.sum(frac[mask]) / (2.0 * np.pi * delta[i] * total_pixels)

    return V0, V1, V2


def compute_minkowski_functionals(conv_map: ConvergenceMap, bins: np.ndarray) -> tuple:
    """
    Compute Minkowski functionals for a convergence map.
    
    Parameters
    ----------
    conv_map : ConvergenceMap
        Convergence map
    bins : np.ndarray
        Threshold bins
        
    Returns
    -------
    tuple
        (v0, v1, v2, sigma1) Minkowski functionals and sigma1 parameter
    """
    grad = compute_gradient(conv_map.data)
    hess = compute_hessian(conv_map.data)
    sigma1 = np.sqrt(np.mean(grad[0]**2 + grad[1]**2))

    v0, v1, v2 = compute_minkowski_functionals_from_arrays(conv_map.data, bins, grad, hess)

    return v0, v1, v2, sigma1