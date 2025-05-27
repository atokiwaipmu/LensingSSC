"""
Abstract interfaces for computational providers.

This module defines interfaces for mathematical and statistical computations,
allowing for different backend implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np

from ..base.data_structures import MapData, PatchData, StatisticsData
from .data_interface import DataProvider


class ComputeProvider(DataProvider):
    """Abstract base class for computational providers."""
    
    @abstractmethod
    def set_backend(self, backend: str) -> None:
        """Set the computational backend (e.g., 'numpy', 'cupy', 'jax')."""
        pass
    
    @abstractmethod
    def get_backend(self) -> str:
        """Get the current computational backend."""
        pass


class StatisticsProvider(ComputeProvider):
    """Abstract interface for statistical computation providers."""
    
    @abstractmethod
    def power_spectrum(self, data: np.ndarray, l_edges: np.ndarray, 
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power spectrum.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (map or patches)
        l_edges : np.ndarray
            Multipole bin edges
        **kwargs
            Additional arguments
            
        Returns
        -------
        tuple
            (ell_centers, power_spectrum)
        """
        pass
    
    @abstractmethod
    def bispectrum(self, data: np.ndarray, l_edges: np.ndarray,
                  configuration: str = "equilateral", **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bispectrum.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        l_edges : np.ndarray
            Multipole bin edges
        configuration : str
            Bispectrum configuration
        **kwargs
            Additional arguments
            
        Returns
        -------
        tuple
            (ell_centers, bispectrum)
        """
        pass
    
    @abstractmethod
    def probability_density_function(self, data: np.ndarray, bins: np.ndarray,
                                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate probability density function.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        bins : np.ndarray
            Bin edges
        **kwargs
            Additional arguments
            
        Returns
        -------
        tuple
            (bin_centers, pdf_values)
        """
        pass
    
    @abstractmethod
    def peak_counts(self, data: np.ndarray, threshold_bins: np.ndarray,
                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate peak counts.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        threshold_bins : np.ndarray
            Threshold bin edges
        **kwargs
            Additional arguments
            
        Returns
        -------
        tuple
            (bin_centers, peak_counts)
        """
        pass
    
    @abstractmethod
    def minkowski_functionals(self, data: np.ndarray, threshold_bins: np.ndarray,
                            **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Minkowski functionals.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        threshold_bins : np.ndarray
            Threshold bin edges
        **kwargs
            Additional arguments
            
        Returns
        -------
        tuple
            (thresholds, V0, V1, V2) - Minkowski functionals
        """
        pass
    
    @abstractmethod
    def correlation_function(self, data1: np.ndarray, data2: Optional[np.ndarray] = None,
                           bins: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate correlation function.
        
        Parameters
        ----------
        data1, data2 : np.ndarray
            Input data arrays
        bins : np.ndarray, optional
            Separation bins
        **kwargs
            Additional arguments
            
        Returns
        -------
        tuple
            (separations, correlation_function)
        """
        pass
    
    @abstractmethod
    def covariance_matrix(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate covariance matrix.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (samples x features)
        **kwargs
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Covariance matrix
        """
        pass


class GeometryProvider(ComputeProvider):
    """Abstract interface for geometric computation providers."""
    
    @abstractmethod
    def fibonacci_grid(self, n_points: int, **kwargs) -> np.ndarray:
        """Generate Fibonacci grid on sphere.
        
        Parameters
        ----------
        n_points : int
            Number of points
        **kwargs
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Grid points in spherical coordinates
        """
        pass
    
    @abstractmethod
    def patch_extraction(self, map_data: MapData, centers: np.ndarray,
                        patch_size_deg: float, xsize: int, **kwargs) -> PatchData:
        """Extract patches from a map.
        
        Parameters
        ----------
        map_data : MapData
            Input map data
        centers : np.ndarray
            Patch center coordinates
        patch_size_deg : float
            Patch size in degrees
        xsize : int
            Patch size in pixels
        **kwargs
            Additional arguments
            
        Returns
        -------
        PatchData
            Extracted patches
        """
        pass
    
    @abstractmethod
    def spherical_distance(self, coords1: np.ndarray, coords2: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate spherical distances.
        
        Parameters
        ----------
        coords1, coords2 : np.ndarray
            Spherical coordinates
        **kwargs
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Angular distances
        """
        pass
    
    @abstractmethod
    def rotation_matrix(self, angles: np.ndarray, convention: str = "euler", **kwargs) -> np.ndarray:
        """Generate rotation matrix.
        
        Parameters
        ----------
        angles : np.ndarray
            Rotation angles
        convention : str
            Rotation convention
        **kwargs
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Rotation matrix
        """
        pass
    
    @abstractmethod
    def coordinate_transform(self, coords: np.ndarray, 
                           input_system: str, output_system: str, **kwargs) -> np.ndarray:
        """Transform between coordinate systems.
        
        Parameters
        ----------
        coords : np.ndarray
            Input coordinates
        input_system : str
            Input coordinate system
        output_system : str
            Output coordinate system
        **kwargs
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Transformed coordinates
        """
        pass


class OptimizationProvider(ComputeProvider):
    """Abstract interface for optimization providers."""
    
    @abstractmethod
    def minimize(self, func: Callable, x0: np.ndarray, method: str = "BFGS", 
                **kwargs) -> Dict[str, Any]:
        """Minimize a function.
        
        Parameters
        ----------
        func : Callable
            Function to minimize
        x0 : np.ndarray
            Initial guess
        method : str
            Optimization method
        **kwargs
            Additional arguments
            
        Returns
        -------
        dict
            Optimization result
        """
        pass
    
    @abstractmethod
    def curve_fit(self, func: Callable, xdata: np.ndarray, ydata: np.ndarray,
                 p0: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a curve to data.
        
        Parameters
        ----------
        func : Callable
            Function to fit
        xdata, ydata : np.ndarray
            Data to fit
        p0 : np.ndarray, optional
            Initial parameter guess
        **kwargs
            Additional arguments
            
        Returns
        -------
        tuple
            (fitted_parameters, covariance_matrix)
        """
        pass


class InterpolationProvider(ComputeProvider):
    """Abstract interface for interpolation providers."""
    
    @abstractmethod
    def interpolate_1d(self, x: np.ndarray, y: np.ndarray, x_new: np.ndarray,
                      kind: str = "linear", **kwargs) -> np.ndarray:
        """1D interpolation.
        
        Parameters
        ----------
        x, y : np.ndarray
            Data points
        x_new : np.ndarray
            Points to interpolate at
        kind : str
            Interpolation method
        **kwargs
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Interpolated values
        """
        pass
    
    @abstractmethod
    def interpolate_2d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                      x_new: np.ndarray, y_new: np.ndarray, 
                      method: str = "linear", **kwargs) -> np.ndarray:
        """2D interpolation.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Data points and values
        x_new, y_new : np.ndarray
            Points to interpolate at
        method : str
            Interpolation method
        **kwargs
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Interpolated values
        """
        pass
    
    @abstractmethod
    def spherical_interpolation(self, coords: np.ndarray, values: np.ndarray,
                              coords_new: np.ndarray, **kwargs) -> np.ndarray:
        """Spherical interpolation.
        
        Parameters
        ----------
        coords : np.ndarray
            Spherical coordinates of data points
        values : np.ndarray
            Values at data points
        coords_new : np.ndarray
            Spherical coordinates to interpolate at
        **kwargs
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Interpolated values
        """
        pass


class FilteringProvider(ComputeProvider):
    """Abstract interface for filtering and smoothing providers."""
    
    @abstractmethod
    def gaussian_filter(self, data: np.ndarray, sigma: float, **kwargs) -> np.ndarray:
        """Apply Gaussian filter.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        sigma : float
            Standard deviation for Gaussian kernel
        **kwargs
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Filtered data
        """
        pass
    
    @abstractmethod
    def median_filter(self, data: np.ndarray, size: int, **kwargs) -> np.ndarray:
        """Apply median filter.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        size : int
            Filter size
        **kwargs
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Filtered data
        """
        pass
    
    @abstractmethod
    def fourier_filter(self, data: np.ndarray, filter_func: Callable, **kwargs) -> np.ndarray:
        """Apply filter in Fourier space.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        filter_func : Callable
            Filter function
        **kwargs
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Filtered data
        """
        pass