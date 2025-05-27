"""
Interpolation utilities with minimal dependencies.
"""

import numpy as np
from typing import Optional, Tuple, Union, Callable
from scipy import interpolate
import logging

from ..base.exceptions import StatisticsError
from ..base.coordinates import SphericalCoordinates, CartesianCoordinates


class Interpolator1D:
    """1D interpolation utilities."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, kind: str = 'linear',
                 bounds_error: bool = False, fill_value: Union[float, str] = 'extrapolate'):
        """Initialize 1D interpolator."""
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.kind = kind
        
        if len(self.x) != len(self.y):
            raise StatisticsError("x and y arrays must have same length")
        
        # Sort by x values
        sort_idx = np.argsort(self.x)
        self.x = self.x[sort_idx]
        self.y = self.y[sort_idx]
        
        # Create interpolator
        self.interpolator = interpolate.interp1d(
            self.x, self.y, kind=kind, bounds_error=bounds_error,
            fill_value=fill_value, assume_sorted=True
        )
    
    def __call__(self, x_new: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate interpolator at new points."""
        return self.interpolator(x_new)
    
    def derivative(self, x_new: Union[float, np.ndarray], order: int = 1) -> Union[float, np.ndarray]:
        """Calculate derivative of interpolated function."""
        if self.kind in ['linear', 'nearest', 'zero', 'slinear']:
            logging.warning(f"Derivative not well-defined for {self.kind} interpolation")
        
        # Create spline interpolator for derivatives
        if not hasattr(self, '_spline'):
            self._spline = interpolate.UnivariateSpline(self.x, self.y, s=0)
        
        return self._spline.derivative(order)(x_new)
    
    def integral(self, a: float, b: float) -> float:
        """Calculate definite integral."""
        if not hasattr(self, '_spline'):
            self._spline = interpolate.UnivariateSpline(self.x, self.y, s=0)
        
        return self._spline.integral(a, b)
    
    def roots(self) -> np.ndarray:
        """Find roots of interpolated function."""
        if not hasattr(self, '_spline'):
            self._spline = interpolate.UnivariateSpline(self.x, self.y, s=0)
        
        return self._spline.roots()


class Interpolator2D:
    """2D interpolation utilities."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 kind: str = 'linear', bounds_error: bool = False,
                 fill_value: float = np.nan):
        """Initialize 2D interpolator."""
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.z = np.asarray(z)
        self.kind = kind
        
        if self.z.ndim == 1:
            # Scattered data interpolation
            if len(self.x) != len(self.y) or len(self.x) != len(self.z):
                raise StatisticsError("x, y, z arrays must have same length for scattered data")
            
            # Use griddata for scattered interpolation
            self.points = np.column_stack([self.x, self.y])
            self.values = self.z
            self._scattered = True
            
        elif self.z.ndim == 2:
            # Regular grid interpolation
            if len(self.x) != self.z.shape[1] or len(self.y) != self.z.shape[0]:
                raise StatisticsError("Grid dimensions must match z array shape")
            
            self.interpolator = interpolate.RegularGridInterpolator(
                (self.y, self.x), self.z, method=kind, bounds_error=bounds_error,
                fill_value=fill_value
            )
            self._scattered = False
        else:
            raise StatisticsError("z array must be 1D (scattered) or 2D (regular grid)")
    
    def __call__(self, x_new: np.ndarray, y_new: np.ndarray) -> np.ndarray:
        """Evaluate interpolator at new points."""
        if self._scattered:
            # Scattered data interpolation
            points_new = np.column_stack([x_new.ravel(), y_new.ravel()])
            result = interpolate.griddata(
                self.points, self.values, points_new, method=self.kind,
                fill_value=np.nan
            )
            return result.reshape(x_new.shape)
        else:
            # Regular grid interpolation
            points_new = np.column_stack([y_new.ravel(), x_new.ravel()])
            result = self.interpolator(points_new)
            return result.reshape(x_new.shape)
    
    def gradient(self, x_new: np.ndarray, y_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate gradient of interpolated function."""
        if self._scattered:
            raise StatisticsError("Gradient not implemented for scattered data")
        
        # Calculate gradient using finite differences
        h = 1e-8
        
        # Gradient in x direction
        grad_x = (self(x_new + h, y_new) - self(x_new - h, y_new)) / (2 * h)
        
        # Gradient in y direction  
        grad_y = (self(x_new, y_new + h) - self(x_new, y_new - h)) / (2 * h)
        
        return grad_x, grad_y


class SphericalInterpolator:
    """Interpolation on the sphere."""
    
    def __init__(self, theta: np.ndarray, phi: np.ndarray, values: np.ndarray,
                 method: str = 'linear'):
        """Initialize spherical interpolator."""
        self.theta = np.asarray(theta)
        self.phi = np.asarray(phi)
        self.values = np.asarray(values)
        self.method = method
        
        if len(self.theta) != len(self.phi) or len(self.theta) != len(self.values):
            raise StatisticsError("theta, phi, values arrays must have same length")
        
        # Convert to Cartesian coordinates for interpolation
        self.cart_coords = self._spherical_to_cartesian(self.theta, self.phi)
        
        # Create interpolator in 3D Cartesian space
        if method == 'nearest':
            from scipy.spatial import cKDTree
            self.tree = cKDTree(self.cart_coords)
        else:
            # Use scattered data interpolation
            self.interpolator = None
    
    def _spherical_to_cartesian(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Convert spherical to Cartesian coordinates."""
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    
    def __call__(self, theta_new: np.ndarray, phi_new: np.ndarray) -> np.ndarray:
        """Evaluate interpolator at new spherical coordinates."""
        cart_new = self._spherical_to_cartesian(theta_new, phi_new)
        
        if self.method == 'nearest':
            # Nearest neighbor interpolation
            distances, indices = self.tree.query(cart_new)
            return self.values[indices]
        
        else:
            # Barycentric interpolation using spherical triangulation
            return self._barycentric_interpolation(cart_new)
    
    def _barycentric_interpolation(self, cart_new: np.ndarray) -> np.ndarray:
        """Barycentric interpolation on sphere."""
        from scipy.spatial import SphericalVoronoi
        
        # Create spherical Voronoi diagram
        sv = SphericalVoronoi(self.cart_coords, radius=1)
        
        # For each query point, find containing triangle and interpolate
        result = np.zeros(len(cart_new))
        
        for i, point in enumerate(cart_new):
            # Find nearest vertices (simplified approach)
            distances = np.sum((self.cart_coords - point)**2, axis=1)
            nearest_idx = np.argsort(distances)[:3]
            
            # Calculate barycentric coordinates
            vertices = self.cart_coords[nearest_idx]
            weights = self._calculate_barycentric_weights(point, vertices)
            
            # Interpolate
            result[i] = np.sum(weights * self.values[nearest_idx])
        
        return result
    
    def _calculate_barycentric_weights(self, point: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """Calculate barycentric weights for spherical triangle."""
        # Simplified implementation using area coordinates
        # Normalize point to unit sphere
        point = point / np.linalg.norm(point)
        
        # Calculate areas of sub-triangles
        def triangle_area(p1, p2, p3):
            # Spherical triangle area using cross product
            cross = np.cross(p2 - p1, p3 - p1)
            return np.linalg.norm(cross) / 2
        
        total_area = triangle_area(vertices[0], vertices[1], vertices[2])
        
        if total_area < 1e-10:
            # Degenerate triangle, use distance-based weights
            distances = np.linalg.norm(vertices - point, axis=1)
            weights = 1 / (distances + 1e-10)
            return weights / np.sum(weights)
        
        # Calculate sub-triangle areas
        areas = np.array([
            triangle_area(point, vertices[1], vertices[2]),
            triangle_area(vertices[0], point, vertices[2]),
            triangle_area(vertices[0], vertices[1], point)
        ])
        
        weights = areas / total_area
        return weights / np.sum(weights)  # Normalize


class AdaptiveInterpolator:
    """Adaptive interpolation with error estimation."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, tolerance: float = 1e-6):
        """Initialize adaptive interpolator."""
        self.x_data = np.asarray(x)
        self.y_data = np.asarray(y)
        self.tolerance = tolerance
        
        # Sort data
        sort_idx = np.argsort(self.x_data)
        self.x_data = self.x_data[sort_idx]
        self.y_data = self.y_data[sort_idx]
        
        # Build adaptive grid
        self.adaptive_grid = self._build_adaptive_grid()
    
    def _build_adaptive_grid(self) -> np.ndarray:
        """Build adaptive grid based on function curvature."""
        # Start with original points
        grid_points = list(zip(self.x_data, self.y_data))
        
        # Add points where curvature is high
        for i in range(len(self.x_data) - 2):
            x1, y1 = self.x_data[i], self.y_data[i]
            x2, y2 = self.x_data[i + 1], self.y_data[i + 1] 
            x3, y3 = self.x_data[i + 2], self.y_data[i + 2]
            
            # Estimate curvature using second difference
            if x3 - x1 > 0:
                curvature = abs((y3 - y2) / (x3 - x2) - (y2 - y1) / (x2 - x1)) / (x3 - x1)
                
                if curvature > self.tolerance:
                    # Add midpoint
                    x_mid = (x1 + x3) / 2
                    # Simple linear interpolation for midpoint value
                    y_mid = y1 + (y3 - y1) * (x_mid - x1) / (x3 - x1)
                    grid_points.append((x_mid, y_mid))
        
        # Sort and return
        grid_points.sort(key=lambda p: p[0])
        return np.array(grid_points)
    
    def __call__(self, x_new: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate adaptive interpolator."""
        x_grid = self.adaptive_grid[:, 0]
        y_grid = self.adaptive_grid[:, 1]
        
        interpolator = interpolate.interp1d(x_grid, y_grid, kind='cubic',
                                          bounds_error=False, fill_value='extrapolate')
        
        return interpolator(x_new)
    
    def error_estimate(self, x_new: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Estimate interpolation error."""
        # Compare cubic and linear interpolation
        x_grid = self.adaptive_grid[:, 0]
        y_grid = self.adaptive_grid[:, 1]
        
        cubic_interp = interpolate.interp1d(x_grid, y_grid, kind='cubic',
                                          bounds_error=False, fill_value='extrapolate')
        linear_interp = interpolate.interp1d(x_grid, y_grid, kind='linear',
                                           bounds_error=False, fill_value='extrapolate')
        
        cubic_result = cubic_interp(x_new)
        linear_result = linear_interp(x_new)
        
        return np.abs(cubic_result - linear_result)


class MultiScaleInterpolator:
    """Multi-scale interpolation for hierarchical data."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, n_levels: int = 3):
        """Initialize multi-scale interpolator."""
        self.x_data = np.asarray(x)
        self.y_data = np.asarray(y)
        self.n_levels = n_levels
        
        # Build hierarchy of interpolators
        self.interpolators = self._build_hierarchy()
    
    def _build_hierarchy(self) -> list:
        """Build hierarchy of interpolators at different scales."""
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
                interp = interpolate.interp1d(x_level, y_level, kind='linear',
                                            bounds_error=False, fill_value='extrapolate')
                interpolators.append(interp)
        
        return interpolators
    
    def __call__(self, x_new: Union[float, np.ndarray], level: Optional[int] = None) -> Union[float, np.ndarray]:
        """Evaluate at specified level or combine all levels."""
        if level is not None:
            if 0 <= level < len(self.interpolators):
                return self.interpolators[level](x_new)
            else:
                raise StatisticsError(f"Level {level} not available")
        
        # Combine all levels with weights
        x_new = np.asarray(x_new)
        result = np.zeros_like(x_new, dtype=float)
        weights_sum = 0
        
        for i, interp in enumerate(self.interpolators):
            weight = 1 / (2**i)  # Coarser levels get less weight
            result += weight * interp(x_new)
            weights_sum += weight
        
        return result / weights_sum if weights_sum > 0 else result