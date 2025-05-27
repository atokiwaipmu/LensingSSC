"""
Coordinate system utilities with minimal dependencies.

This module provides coordinate transformation utilities for spherical and
Cartesian coordinate systems, independent of external astronomy libraries.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union, Optional
import numpy as np
import math

from .exceptions import GeometryError


class Coordinates(ABC):
    """Abstract base class for coordinate systems."""
    
    @abstractmethod
    def to_cartesian(self) -> "CartesianCoordinates":
        """Convert to Cartesian coordinates."""
        pass
    
    @abstractmethod
    def to_spherical(self) -> "SphericalCoordinates":
        """Convert to spherical coordinates."""
        pass


@dataclass
class SphericalCoordinates(Coordinates):
    """Spherical coordinate system (r, theta, phi).
    
    Parameters
    ----------
    r : float
        Radial distance (default: 1.0 for unit sphere)
    theta : float
        Polar angle in radians [0, π]
    phi : float
        Azimuthal angle in radians [0, 2π]
    """
    
    r: float = 1.0
    theta: float = 0.0
    phi: float = 0.0
    
    def __post_init__(self):
        """Validate coordinates after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate spherical coordinates."""
        if self.r < 0:
            raise GeometryError("Radial coordinate must be non-negative")
        
        if not (0 <= self.theta <= np.pi):
            raise GeometryError("Theta must be in range [0, π]")
        
        # Normalize phi to [0, 2π]
        self.phi = self.phi % (2 * np.pi)
    
    def to_cartesian(self) -> "CartesianCoordinates":
        """Convert to Cartesian coordinates."""
        sin_theta = math.sin(self.theta)
        x = self.r * sin_theta * math.cos(self.phi)
        y = self.r * sin_theta * math.sin(self.phi)
        z = self.r * math.cos(self.theta)
        return CartesianCoordinates(x, y, z)
    
    def to_spherical(self) -> "SphericalCoordinates":
        """Return self (already spherical)."""
        return SphericalCoordinates(self.r, self.theta, self.phi)
    
    @classmethod
    def from_degrees(cls, r: float = 1.0, theta_deg: float = 0.0, 
                    phi_deg: float = 0.0) -> "SphericalCoordinates":
        """Create from degrees instead of radians."""
        return cls(r, math.radians(theta_deg), math.radians(phi_deg))
    
    def to_degrees(self) -> Tuple[float, float, float]:
        """Get coordinates in degrees."""
        return self.r, math.degrees(self.theta), math.degrees(self.phi)
    
    def angular_distance(self, other: "SphericalCoordinates") -> float:
        """Calculate angular distance to another point (haversine formula)."""
        if not isinstance(other, SphericalCoordinates):
            raise TypeError("Other coordinate must be SphericalCoordinates")
        
        # Convert to colatitude for haversine formula
        lat1 = np.pi/2 - self.theta
        lat2 = np.pi/2 - other.theta
        
        dlat = lat2 - lat1
        dlon = other.phi - self.phi
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        
        return 2 * math.asin(math.sqrt(a))


@dataclass
class CartesianCoordinates(Coordinates):
    """Cartesian coordinate system (x, y, z).
    
    Parameters
    ----------
    x : float
        X coordinate
    y : float
        Y coordinate  
    z : float
        Z coordinate
    """
    
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_cartesian(self) -> "CartesianCoordinates":
        """Return self (already Cartesian)."""
        return CartesianCoordinates(self.x, self.y, self.z)
    
    def to_spherical(self) -> "SphericalCoordinates":
        """Convert to spherical coordinates."""
        r = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        
        if r == 0:
            return SphericalCoordinates(0.0, 0.0, 0.0)
        
        theta = math.acos(self.z / r) if r > 0 else 0.0
        phi = math.atan2(self.y, self.x)
        
        # Ensure phi is in [0, 2π]
        if phi < 0:
            phi += 2 * np.pi
            
        return SphericalCoordinates(r, theta, phi)
    
    @property
    def magnitude(self) -> float:
        """Get the magnitude of the vector."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> "CartesianCoordinates":
        """Return normalized vector."""
        mag = self.magnitude
        if mag == 0:
            raise GeometryError("Cannot normalize zero vector")
        return CartesianCoordinates(self.x/mag, self.y/mag, self.z/mag)
    
    def dot(self, other: "CartesianCoordinates") -> float:
        """Dot product with another vector."""
        if not isinstance(other, CartesianCoordinates):
            raise TypeError("Other coordinate must be CartesianCoordinates")
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: "CartesianCoordinates") -> "CartesianCoordinates":
        """Cross product with another vector."""
        if not isinstance(other, CartesianCoordinates):
            raise TypeError("Other coordinate must be CartesianCoordinates")
        
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        
        return CartesianCoordinates(x, y, z)


class CoordinateTransformer:
    """Utility class for batch coordinate transformations."""
    
    @staticmethod
    def spherical_to_cartesian_batch(coords: np.ndarray) -> np.ndarray:
        """Convert array of spherical coordinates to Cartesian.
        
        Parameters
        ----------
        coords : np.ndarray
            Array of shape (N, 3) with columns [r, theta, phi]
            
        Returns
        -------
        np.ndarray
            Array of shape (N, 3) with columns [x, y, z]
        """
        if coords.shape[1] != 3:
            raise GeometryError("Input must have shape (N, 3)")
        
        r, theta, phi = coords[:, 0], coords[:, 1], coords[:, 2]
        
        sin_theta = np.sin(theta)
        x = r * sin_theta * np.cos(phi)
        y = r * sin_theta * np.sin(phi)
        z = r * np.cos(theta)
        
        return np.column_stack([x, y, z])
    
    @staticmethod
    def cartesian_to_spherical_batch(coords: np.ndarray) -> np.ndarray:
        """Convert array of Cartesian coordinates to spherical.
        
        Parameters
        ----------
        coords : np.ndarray
            Array of shape (N, 3) with columns [x, y, z]
            
        Returns
        -------
        np.ndarray
            Array of shape (N, 3) with columns [r, theta, phi]
        """
        if coords.shape[1] != 3:
            raise GeometryError("Input must have shape (N, 3)")
        
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Handle zero radius case
        theta = np.where(r > 0, np.arccos(np.clip(z / r, -1, 1)), 0.0)
        phi = np.arctan2(y, x)
        
        # Ensure phi is in [0, 2π]
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)
        
        return np.column_stack([r, theta, phi])
    
    @staticmethod
    def angular_distance_batch(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Calculate angular distances between arrays of spherical coordinates.
        
        Parameters
        ----------
        coords1, coords2 : np.ndarray
            Arrays of shape (N, 2) or (N, 3) with columns [theta, phi] or [r, theta, phi]
            
        Returns
        -------
        np.ndarray
            Array of angular distances
        """
        # Extract theta, phi (ignore r if present)
        if coords1.shape[1] >= 2:
            theta1, phi1 = coords1[:, -2], coords1[:, -1]
        else:
            raise GeometryError("Coordinates must have at least 2 columns")
            
        if coords2.shape[1] >= 2:
            theta2, phi2 = coords2[:, -2], coords2[:, -1]
        else:
            raise GeometryError("Coordinates must have at least 2 columns")
        
        # Convert to colatitude for haversine formula
        lat1 = np.pi/2 - theta1
        lat2 = np.pi/2 - theta2
        
        dlat = lat2 - lat1
        dlon = phi2 - phi1
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
        
        return 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class RotationMatrix:
    """Utilities for rotation matrices and rotations on the sphere."""
    
    @staticmethod
    def rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Create rotation matrix for rotation about an axis.
        
        Parameters
        ----------
        axis : np.ndarray
            Rotation axis (will be normalized)
        angle : float
            Rotation angle in radians
            
        Returns
        -------
        np.ndarray
            3x3 rotation matrix
        """
        # Normalize axis
        axis = axis / np.linalg.norm(axis)
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle
        
        x, y, z = axis
        
        return np.array([
            [cos_angle + x*x*one_minus_cos, x*y*one_minus_cos - z*sin_angle, x*z*one_minus_cos + y*sin_angle],
            [y*x*one_minus_cos + z*sin_angle, cos_angle + y*y*one_minus_cos, y*z*one_minus_cos - x*sin_angle],
            [z*x*one_minus_cos - y*sin_angle, z*y*one_minus_cos + x*sin_angle, cos_angle + z*z*one_minus_cos]
        ])
    
    @staticmethod
    def rotation_matrix_euler(alpha: float, beta: float, gamma: float) -> np.ndarray:
        """Create rotation matrix from Euler angles (ZYZ convention).
        
        Parameters
        ----------
        alpha, beta, gamma : float
            Euler angles in radians
            
        Returns
        -------
        np.ndarray
            3x3 rotation matrix
        """
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        
        return np.array([
            [ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb],
            [sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb],
            [-sb*cg, sb*sg, cb]
        ])
    
    @staticmethod
    def rotate_spherical_coordinates(coords: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """Rotate spherical coordinates using a rotation matrix.
        
        Parameters
        ----------
        coords : np.ndarray
            Array of spherical coordinates [r, theta, phi] of shape (N, 3)
        rotation_matrix : np.ndarray
            3x3 rotation matrix
            
        Returns
        -------
        np.ndarray
            Rotated spherical coordinates
        """
        # Convert to Cartesian
        cartesian = CoordinateTransformer.spherical_to_cartesian_batch(coords)
        
        # Apply rotation
        rotated_cartesian = cartesian @ rotation_matrix.T
        
        # Convert back to spherical
        return CoordinateTransformer.cartesian_to_spherical_batch(rotated_cartesian)