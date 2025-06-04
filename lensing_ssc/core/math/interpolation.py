"""
Interpolation utilities with minimal dependencies.

This module provides comprehensive interpolation methods for 1D, 2D, and spherical
data using only numpy and scipy.
"""

import numpy as np
from typing import Optional, Tuple, Union, Callable, List
from scipy import interpolate
import logging

from ..base.exceptions import StatisticsError
from ..base.coordinates import SphericalCoordinates, CartesianCoordinates


class Interpolator1D:
    """1D interpolation utilities with various methods."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, kind: str = 'linear',
                 bounds_error: bool = False, fill_value: Union[float, str] = 'extrapolate',
                 assume_sorted: bool = False):
        """Initialize 1D interpolator.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates
        y : np.ndarray
            Y values
        kind : str
            Interpolation method ('linear', 'cubic', 'quintic', etc.)
        bounds_error : bool
            Whether to raise error for out-of-bounds values
        fill_value : float or str
            Value for out-of-bounds points
        assume_sorted : bool
            Whether x is already sorted
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.kind = kind
        
        if len(self.x) != len(self.y):
            raise StatisticsError("x and y arrays must have same length")
        
        # Sort by x values if needed
        if not assume_sorted:
            sort_idx = np.argsort(self.x)
            self.x = self.x[sort_idx]
            self.y = self.y[sort_idx]
        
        # Check for duplicates
        if len(np.unique(self.x)) != len(self.x):
            logging.warning("Duplicate x values found, removing duplicates")
            unique_mask = np.concatenate(([True], np.diff(self.x) != 0))
            self.x = self.x[unique_mask]
            self.y = self.y[unique_mask]
        
        # Create interpolator
        try:
            self.interpolator = interpolate.interp1d(
                self.x, self.y, kind=kind, bounds_error=bounds_error,
                fill_value=fill_value, assume_sorted=True
            )
        except ValueError as e:
            raise StatisticsError(f"Failed to create interpolator: {e}")
    
    def __call__(self, x_new: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate interpolator at new points.
        
        Parameters
        ----------
        x_new : float or np.ndarray
            Points to evaluate at
            
        Returns
        -------
        float or np.ndarray
            Interpolated values
        """
        return self.interpolator(x_new)
    
    def derivative(self, x_new: Union[float, np.ndarray], order: int = 1) -> Union[float, np.ndarray]:
        """Calculate derivative of interpolated function.
        
        Parameters
        ----------
        x_new : float or np.ndarray
            Points to evaluate derivative at
        order : int
            Derivative order
            
        Returns
        -------
        float or np.ndarray
            Derivative values
        """
        if self.kind in ['linear', 'nearest', 'zero', 'slinear']:
            logging.warning(f"Derivative not well-defined for {self.kind} interpolation")
        
        # Create spline interpolator for derivatives
        if not hasattr(self, '_spline'):
            try:
                self._spline = interpolate.UnivariateSpline(self.x, self.y, s=0)
            except Exception as e:
                raise StatisticsError(f"Failed to create spline for derivatives: {e}")
        
        return self._spline.derivative(order)(x_new)
    
    def integral(self, a: float, b: float) -> float:
        """Calculate definite integral.
        
        Parameters
        ----------
        a, b : float
            Integration bounds
            
        Returns
        -------
        float
            Integral value
        """
        if not hasattr(self, '_spline'):
            self._spline = interpolate.UnivariateSpline(self.x, self.y, s=0)
        
        return self._spline.integral(a, b)
    
    def roots(self) -> np.ndarray:
        """Find roots of interpolated function.
        
        Returns
        -------
        np.ndarray
            Root locations
        """
        if not hasattr(self, '_spline'):
            self._spline = interpolate.UnivariateSpline(self.x, self.y, s=0)
        
        return self._spline.roots()
    
    def get_knots(self) -> np.ndarray:
        """Get interpolation knots.
        
        Returns
        -------
        np.ndarray
            Knot locations
        """
        return self.x.copy()
    
    def get_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get interpolation coefficients (for splines).
        
        Returns
        -------
        tuple
            (knots, coefficients)
        """
        if not hasattr(self, '_spline'):
            self._spline = interpolate.UnivariateSpline(self.x, self.y, s=0)
        
        return self._spline.get_knots(), self._spline.get_coeffs()


class Interpolator2D:
    """2D interpolation utilities for regular and scattered data."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 kind: str = 'linear', bounds_error: bool = False,
                 fill_value: float = np.nan):
        """Initialize 2D interpolator.
        
        Parameters
        ----------
        x, y : np.ndarray
            Coordinate arrays
        z : np.ndarray
            Values array
        kind : str
            Interpolation method
        bounds_error : bool
            Whether to raise error for out-of-bounds
        fill_value : float
            Fill value for out-of-bounds
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.z = np.asarray(z)
        self.kind = kind
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        
        if self.z.ndim == 1:
            # Scattered data interpolation
            if len(self.x) != len(self.y) or len(self.x) != len(self.z):
                raise StatisticsError("x, y, z arrays must have same length for scattered data")
            
            self.points = np.column_stack([self.x, self.y])
            self.values = self.z
            self._scattered = True
            
        elif self.z.ndim == 2:
            # Regular grid interpolation
            if len(self.x) != self.z.shape[1] or len(self.y) != self.z.shape[0]:
                raise StatisticsError("Grid dimensions must match z array shape")
            
            # Check if coordinates are sorted
            if not (np.all(np.diff(self.x) > 0) and np.all(np.diff(self.y) > 0)):
                logging.warning("Coordinates should be sorted for optimal performance")
            
            try:
                self.interpolator = interpolate.RegularGridInterpolator(
                    (self.y, self.x), self.z, method=kind, bounds_error=bounds_error,
                    fill_value=fill_value
                )
            except ValueError as e:
                raise StatisticsError(f"Failed to create regular grid interpolator: {e}")
            
            self._scattered = False
        else:
            raise StatisticsError("z array must be 1D (scattered) or 2D (regular grid)")
    
    def __call__(self, x_new: np.ndarray, y_new: np.ndarray) -> np.ndarray:
        """Evaluate interpolator at new points.
        
        Parameters
        ----------
        x_new, y_new : np.ndarray
            Coordinates to evaluate at
            
        Returns
        -------
        np.ndarray
            Interpolated values
        """
        x_new = np.asarray(x_new)
        y_new = np.asarray(y_new)
        
        if x_new.shape != y_new.shape:
            raise StatisticsError("x_new and y_new must have same shape")
        
        if self._scattered:
            # Scattered data interpolation
            points_new = np.column_stack([x_new.ravel(), y_new.ravel()])
            
            try:
                result = interpolate.griddata(
                    self.points, self.values, points_new, method=self.kind,
                    fill_value=self.fill_value
                )
                return result.reshape(x_new.shape)
            except Exception as e:
                raise StatisticsError(f"Scattered interpolation failed: {e}")
        else:
            # Regular grid interpolation
            points_new = np.column_stack([y_new.ravel(), x_new.ravel()])
            try:
                result = self.interpolator(points_new)
                return result.reshape(x_new.shape)
            except Exception as e:
                raise StatisticsError(f"Regular grid interpolation failed: {e}")
    
    def gradient(self, x_new: np.ndarray, y_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate gradient of interpolated function.
        
        Parameters
        ----------
        x_new, y_new : np.ndarray
            Coordinates to evaluate gradient at
            
        Returns
        -------
        tuple
            (grad_x, grad_y)
        """
        if self._scattered:
            raise StatisticsError("Gradient not implemented for scattered data")
        
        # Calculate gradient using finite differences
        h = 1e-8
        
        # Gradient in x direction
        try:
            grad_x = (self(x_new + h, y_new) - self(x_new - h, y_new)) / (2 * h)
        except:
            # Fallback to forward/backward differences at boundaries
            grad_x = (self(x_new + h, y_new) - self(x_new, y_new)) / h
        
        # Gradient in y direction  
        try:
            grad_y = (self(x_new, y_new + h) - self(x_new, y_new - h)) / (2 * h)
        except:
            grad_y = (self(x_new, y_new + h) - self(x_new, y_new)) / h
        
        return grad_x, grad_y
    
    def laplacian(self, x_new: np.ndarray, y_new: np.ndarray) -> np.ndarray:
        """Calculate Laplacian of interpolated function.
        
        Parameters
        ----------
        x_new, y_new : np.ndarray
            Coordinates to evaluate Laplacian at
            
        Returns
        -------
        np.ndarray
            Laplacian values
        """
        if self._scattered:
            raise StatisticsError("Laplacian not implemented for scattered data")
        
        h = 1e-6
        
        # Second derivatives using finite differences
        f_xx = (self(x_new + h, y_new) - 2*self(x_new, y_new) + self(x_new - h, y_new)) / h**2
        f_yy = (self(x_new, y_new + h) - 2*self(x_new, y_new) + self(x_new, y_new - h)) / h**2
        
        return f_xx + f_yy


class SphericalInterpolator:
    """Interpolation on the sphere using various methods."""
    
    def __init__(self, theta: np.ndarray, phi: np.ndarray, values: np.ndarray,
                 method: str = 'linear', normalize_coords: bool = True):
        """Initialize spherical interpolator.
        
        Parameters
        ----------
        theta : np.ndarray
            Polar angles (0 to π)
        phi : np.ndarray
            Azimuthal angles (0 to 2π)
        values : np.ndarray
            Values at coordinates
        method : str
            Interpolation method ('linear', 'nearest', 'cubic')
        normalize_coords : bool
            Whether to normalize coordinates to valid ranges
        """
        self.theta = np.asarray(theta)
        self.phi = np.asarray(phi)
        self.values = np.asarray(values)
        self.method = method
        
        if len(self.theta) != len(self.phi) or len(self.theta) != len(self.values):
            raise StatisticsError("theta, phi, values arrays must have same length")
        
        if normalize_coords:
            # Normalize coordinates to valid ranges
            self.theta = np.clip(self.theta, 0, np.pi)
            self.phi = self.phi % (2 * np.pi)
        
        # Convert to Cartesian coordinates for interpolation
        self.cart_coords = self._spherical_to_cartesian(self.theta, self.phi)
        
        # Create interpolator based on method
        if method == 'nearest':
            from scipy.spatial import cKDTree
            self.tree = cKDTree(self.cart_coords)
        elif method == 'rbf':
            from scipy.interpolate import Rbf
            self.rbf = Rbf(self.cart_coords[:, 0], self.cart_coords[:, 1], 
                          self.cart_coords[:, 2], self.values, function='multiquadric')
        else:
            # Use scattered data interpolation
            self.interpolator = None
    
    def _spherical_to_cartesian(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Convert spherical to Cartesian coordinates on unit sphere.
        
        Parameters
        ----------
        theta : np.ndarray
            Polar angles
        phi : np.ndarray
            Azimuthal angles
            
        Returns
        -------
        np.ndarray
            Cartesian coordinates (N, 3)
        """
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    
    def __call__(self, theta_new: np.ndarray, phi_new: np.ndarray) -> np.ndarray:
        """Evaluate interpolator at new spherical coordinates.
        
        Parameters
        ----------
        theta_new : np.ndarray
            New polar angles
        phi_new : np.ndarray
            New azimuthal angles
            
        Returns
        -------
        np.ndarray
            Interpolated values
        """
        theta_new = np.asarray(theta_new)
        phi_new = np.asarray(phi_new)
        
        # Normalize coordinates
        theta_new = np.clip(theta_new, 0, np.pi)
        phi_new = phi_new % (2 * np.pi)
        
        cart_new = self._spherical_to_cartesian(theta_new, phi_new)
        
        if self.method == 'nearest':
            # Nearest neighbor interpolation
            distances, indices = self.tree.query(cart_new)
            return self.values[indices]
        
        elif self.method == 'rbf':
            # Radial basis function interpolation
            return self.rbf(cart_new[:, 0], cart_new[:, 1], cart_new[:, 2])
        
        else:
            # Barycentric interpolation using spherical triangulation
            return self._barycentric_interpolation(cart_new)
    
    def _barycentric_interpolation(self, cart_new: np.ndarray) -> np.ndarray:
        """Barycentric interpolation on sphere.
        
        Parameters
        ----------
        cart_new : np.ndarray
            New Cartesian coordinates
            
        Returns
        -------
        np.ndarray
            Interpolated values
        """
        result = np.zeros(len(cart_new))
        
        for i, point in enumerate(cart_new):
            # Normalize point to unit sphere
            point = point / np.linalg.norm(point)
            
            # Find nearest vertices for triangulation
            distances = np.sum((self.cart_coords - point)**2, axis=1)
            nearest_idx = np.argsort(distances)[:4]  # Use 4 nearest for robustness
            
            # Calculate weights using inverse distance weighting
            # For spherical surface, use great circle distances
            weights = []
            for idx in nearest_idx:
                vertex = self.cart_coords[idx]
                # Great circle distance (more accurate on sphere)
                dot_product = np.clip(np.dot(point, vertex), -1, 1)
                angular_dist = np.arccos(dot_product)
                weight = 1.0 / (angular_dist + 1e-10)  # Add small epsilon
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
            
            # Interpolate
            result[i] = np.sum(weights * self.values[nearest_idx])
        
        return result
    
    def gradient_spherical(self, theta_new: np.ndarray, phi_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate gradient in spherical coordinates.
        
        Parameters
        ----------
        theta_new : np.ndarray
            Polar angles
        phi_new : np.ndarray
            Azimuthal angles
            
        Returns
        -------
        tuple
            (grad_theta, grad_phi)
        """
        h_theta = 1e-6
        h_phi = 1e-6
        
        # Gradient in theta direction
        theta_plus = np.clip(theta_new + h_theta, 0, np.pi)
        theta_minus = np.clip(theta_new - h_theta, 0, np.pi)
        grad_theta = (self(theta_plus, phi_new) - self(theta_minus, phi_new)) / (2 * h_theta)
        
        # Gradient in phi direction (accounting for spherical geometry)
        phi_plus = (phi_new + h_phi) % (2 * np.pi)
        phi_minus = (phi_new - h_phi) % (2 * np.pi)
        grad_phi = (self(theta_new, phi_plus) - self(theta_new, phi_minus)) / (2 * h_phi)
        
        # Scale phi gradient by sin(theta)
        grad_phi = grad_phi / (np.sin(theta_new) + 1e-10)
        
        return grad_theta, grad_phi


class AdaptiveInterpolator:
    """Adaptive interpolation with error estimation and refinement."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, tolerance: float = 1e-6,
                 max_points: int = 1000):
        """Initialize adaptive interpolator.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates
        y : np.ndarray
            Y values
        tolerance : float
            Error tolerance for adaptive refinement
        max_points : int
            Maximum number of points in adaptive grid
        """
        self.x_data = np.asarray(x)
        self.y_data = np.asarray(y)
        self.tolerance = tolerance
        self.max_points = max_points
        
        # Sort data
        sort_idx = np.argsort(self.x_data)
        self.x_data = self.x_data[sort_idx]
        self.y_data = self.y_data[sort_idx]
        
        # Build adaptive grid
        self.adaptive_grid = self._build_adaptive_grid()
        
        # Create final interpolator
        x_grid = self.adaptive_grid[:, 0]
        y_grid = self.adaptive_grid[:, 1]
        self.interpolator = Interpolator1D(x_grid, y_grid, kind='cubic')
    
    def _build_adaptive_grid(self) -> np.ndarray:
        """Build adaptive grid based on function curvature and error estimates.
        
        Returns
        -------
        np.ndarray
            Adaptive grid points (N, 2)
        """
        # Start with original points
        grid_points = list(zip(self.x_data, self.y_data))
        
        # Iteratively add points where error is high
        iteration = 0
        while len(grid_points) < self.max_points and iteration < 100:
            new_points = []
            current_x = np.array([p[0] for p in grid_points])
            current_y = np.array([p[1] for p in grid_points])
            
            # Create temporary interpolator
            temp_interp = Interpolator1D(current_x, current_y, kind='linear')
            
            # Check error at midpoints
            for i in range(len(grid_points) - 1):
                x1, y1 = grid_points[i]
                x2, y2 = grid_points[i + 1]
                
                if x2 - x1 < 1e-10:  # Skip very small intervals
                    continue
                
                # Midpoint
                x_mid = (x1 + x2) / 2
                
                # True value (interpolated from original data)
                y_true = np.interp(x_mid, self.x_data, self.y_data)
                
                # Estimated value from current grid
                y_est = temp_interp(x_mid)
                
                # Error estimate
                error = abs(y_true - y_est)
                
                if error > self.tolerance:
                    new_points.append((x_mid, y_true))
            
            if not new_points:
                break
            
            # Add new points
            grid_points.extend(new_points)
            grid_points.sort(key=lambda p: p[0])
            
            iteration += 1
        
        return np.array(grid_points)
    
    def __call__(self, x_new: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate adaptive interpolator.
        
        Parameters
        ----------
        x_new : float or np.ndarray
            Points to evaluate at
            
        Returns
        -------
        float or np.ndarray
            Interpolated values
        """
        return self.interpolator(x_new)
    
    def error_estimate(self, x_new: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Estimate interpolation error.
        
        Parameters
        ----------
        x_new : float or np.ndarray
            Points to estimate error at
            
        Returns
        -------
        float or np.ndarray
            Estimated errors
        """
        # Compare cubic and linear interpolation
        x_grid = self.adaptive_grid[:, 0]
        y_grid = self.adaptive_grid[:, 1]
        
        linear_interp = Interpolator1D(x_grid, y_grid, kind='linear')
        
        cubic_result = self.interpolator(x_new)
        linear_result = linear_interp(x_new)
        
        return np.abs(cubic_result - linear_result)
    
    def get_grid_density(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get grid density information.
        
        Returns
        -------
        tuple
            (grid_points, local_density)
        """
        x_grid = self.adaptive_grid[:, 0]
        
        # Calculate local grid spacing
        spacings = np.diff(x_grid)
        density = 1.0 / np.concatenate(([spacings[0]], spacings))
        
        return x_grid, density


class MultiScaleInterpolator:
    """Multi-scale interpolation for hierarchical data analysis."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, n_levels: int = 3,
                 base_method: str = 'linear'):
        """Initialize multi-scale interpolator.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates
        y : np.ndarray
            Y values
        n_levels : int
            Number of scale levels
        base_method : str
            Base interpolation method
        """
        self.x_data = np.asarray(x)
        self.y_data = np.asarray(y)
        self.n_levels = n_levels
        self.base_method = base_method
        
        # Build hierarchy of interpolators
        self.interpolators = self._build_hierarchy()
        self.level_weights = self._compute_level_weights()
    
    def _build_hierarchy(self) -> List[Interpolator1D]:
        """Build hierarchy of interpolators at different scales.
        
        Returns
        -------
        list
            List of interpolators at different scales
        """
        interpolators = []
        
        # Sort data
        sort_idx = np.argsort(self.x_data)
        x_sorted = self.x_data[sort_idx]
        y_sorted = self.y_data[sort_idx]
        
        for level in range(self.n_levels):
            # Subsample data for this level
            step = 2**level
            x_level = x_sorted[::step]
            y_level = y_sorted[::step]
            
            if len(x_level) >= 2:
                try:
                    interp = Interpolator1D(x_level, y_level, kind=self.base_method)
                    interpolators.append(interp)
                except StatisticsError:
                    # If interpolation fails, use previous level
                    if interpolators:
                        interpolators.append(interpolators[-1])
                    break
        
        return interpolators
    
    def _compute_level_weights(self) -> np.ndarray:
        """Compute weights for combining different scale levels.
        
        Returns
        -------
        np.ndarray
            Weights for each level
        """
        # Exponential decay weights (finer scales get higher weight)
        weights = np.array([1.0 / (2**i) for i in range(len(self.interpolators))])
        return weights / np.sum(weights)
    
    def __call__(self, x_new: Union[float, np.ndarray], level: Optional[int] = None, 
                 adaptive_weights: bool = False) -> Union[float, np.ndarray]:
        """Evaluate at specified level or combine all levels.
        
        Parameters
        ----------
        x_new : float or np.ndarray
            Points to evaluate at
        level : int, optional
            Specific level to use (if None, combine all levels)
        adaptive_weights : bool
            Whether to use adaptive weights based on local smoothness
            
        Returns
        -------
        float or np.ndarray
            Interpolated values
        """
        x_new = np.asarray(x_new)
        
        if level is not None:
            if 0 <= level < len(self.interpolators):
                return self.interpolators[level](x_new)
            else:
                raise StatisticsError(f"Level {level} not available")
        
        # Combine all levels
        if len(self.interpolators) == 0:
            raise StatisticsError("No interpolators available")
        
        if adaptive_weights:
            weights = self._compute_adaptive_weights(x_new)
        else:
            weights = self.level_weights
        
        result = np.zeros_like(x_new, dtype=float)
        
        for i, interp in enumerate(self.interpolators):
            if i < len(weights):
                result += weights[i] * interp(x_new)
        
        return result
    
    def _compute_adaptive_weights(self, x_new: np.ndarray) -> np.ndarray:
        """Compute adaptive weights based on local data density and smoothness.
        
        Parameters
        ----------
        x_new : np.ndarray
            Evaluation points
            
        Returns
        -------
        np.ndarray
            Adaptive weights for each level
        """
        weights = np.zeros((len(x_new), len(self.interpolators)))
        
        for i, x_point in enumerate(x_new.flat):
            # Estimate local smoothness using second differences
            local_curvature = self._estimate_local_curvature(x_point)
            
            # Higher curvature -> prefer finer scales
            scale_preference = np.exp(-local_curvature * np.arange(len(self.interpolators)))
            scale_preference = scale_preference / np.sum(scale_preference)
            
            weights[i, :len(scale_preference)] = scale_preference
        
        return weights.mean(axis=0)  # Average over all points
    
    def _estimate_local_curvature(self, x_point: float) -> float:
        """Estimate local curvature of the function.
        
        Parameters
        ----------
        x_point : float
            Point to estimate curvature at
            
        Returns
        -------
        float
            Estimated curvature
        """
        # Use finest available interpolator
        if not self.interpolators:
            return 0.0
        
        finest_interp = self.interpolators[0]
        
        # Estimate second derivative
        h = (self.x_data.max() - self.x_data.min()) / 1000
        
        try:
            f_center = finest_interp(x_point)
            f_left = finest_interp(x_point - h)
            f_right = finest_interp(x_point + h)
            
            # Second derivative approximation
            second_deriv = (f_right - 2*f_center + f_left) / h**2
            return abs(second_deriv)
        except:
            return 0.0
    
    def decompose(self, x_new: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose signal into different scale components.
        
        Parameters
        ----------
        x_new : np.ndarray
            Evaluation points
            
        Returns
        -------
        dict
            Dictionary with components at each scale level
        """
        components = {}
        
        for i, interp in enumerate(self.interpolators):
            components[f'level_{i}'] = interp(x_new)
        
        # Compute scale differences (detail coefficients)
        for i in range(len(self.interpolators) - 1):
            detail = components[f'level_{i}'] - components[f'level_{i+1}']
            components[f'detail_{i}'] = detail
        
        return components