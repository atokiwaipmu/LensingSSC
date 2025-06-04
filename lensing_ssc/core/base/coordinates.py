"""
Coordinate system utilities with minimal dependencies.

This module provides coordinate transformation utilities for spherical and
Cartesian coordinate systems, independent of external astronomy libraries.
All operations use only numpy and standard library functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union, Optional, List
import numpy as np
import math

from .exceptions import GeometryError, ValidationError


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
    
    @abstractmethod
    def distance_to(self, other: "Coordinates") -> float:
        """Calculate distance to another coordinate."""
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
            raise GeometryError(f"Theta must be in range [0, π], got {self.theta}")
        
        # Normalize phi to [0, 2π]
        self.phi = self.phi % (2 * np.pi)
    
    def to_cartesian(self) -> "CartesianCoordinates":
        """Convert to Cartesian coordinates.
        
        Returns
        -------
        CartesianCoordinates
            Equivalent Cartesian coordinates
        """
        sin_theta = math.sin(self.theta)
        x = self.r * sin_theta * math.cos(self.phi)
        y = self.r * sin_theta * math.sin(self.phi)
        z = self.r * math.cos(self.theta)
        return CartesianCoordinates(x, y, z)
    
    def to_spherical(self) -> "SphericalCoordinates":
        """Return self (already spherical).
        
        Returns
        -------
        SphericalCoordinates
            Copy of self
        """
        return SphericalCoordinates(self.r, self.theta, self.phi)
    
    def distance_to(self, other: "Coordinates") -> float:
        """Calculate distance to another coordinate.
        
        Parameters
        ----------
        other : Coordinates
            Other coordinate point
            
        Returns
        -------
        float
            Distance between coordinates
        """
        if isinstance(other, SphericalCoordinates):
            return self.angular_distance(other)
        else:
            # Convert to Cartesian and calculate Euclidean distance
            cart_self = self.to_cartesian()
            cart_other = other.to_cartesian()
            return cart_self.distance_to(cart_other)
    
    @classmethod
    def from_degrees(cls, r: float = 1.0, theta_deg: float = 0.0, 
                    phi_deg: float = 0.0) -> "SphericalCoordinates":
        """Create from degrees instead of radians.
        
        Parameters
        ----------
        r : float
            Radial distance
        theta_deg : float
            Polar angle in degrees
        phi_deg : float
            Azimuthal angle in degrees
            
        Returns
        -------
        SphericalCoordinates
            New spherical coordinates
        """
        return cls(r, math.radians(theta_deg), math.radians(phi_deg))
    
    def to_degrees(self) -> Tuple[float, float, float]:
        """Get coordinates in degrees.
        
        Returns
        -------
        tuple
            (r, theta_deg, phi_deg)
        """
        return self.r, math.degrees(self.theta), math.degrees(self.phi)
    
    def angular_distance(self, other: "SphericalCoordinates") -> float:
        """Calculate angular distance to another point (haversine formula).
        
        Parameters
        ----------
        other : SphericalCoordinates
            Other spherical coordinate point
            
        Returns
        -------
        float
            Angular distance in radians
        """
        if not isinstance(other, SphericalCoordinates):
            raise TypeError("Other coordinate must be SphericalCoordinates")
        
        # Convert to colatitude for haversine formula
        lat1 = np.pi/2 - self.theta
        lat2 = np.pi/2 - other.theta
        
        dlat = lat2 - lat1
        dlon = other.phi - self.phi
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        
        return 2 * math.asin(math.sqrt(min(1.0, a)))  # Clamp to avoid numerical issues
    
    def great_circle_bearing(self, other: "SphericalCoordinates") -> float:
        """Calculate initial bearing for great circle path.
        
        Parameters
        ----------
        other : SphericalCoordinates
            Destination point
            
        Returns
        -------
        float
            Initial bearing in radians
        """
        if not isinstance(other, SphericalCoordinates):
            raise TypeError("Other coordinate must be SphericalCoordinates")
        
        lat1 = np.pi/2 - self.theta
        lat2 = np.pi/2 - other.theta
        dlon = other.phi - self.phi
        
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(y, x)
        return (bearing + 2 * np.pi) % (2 * np.pi)  # Normalize to [0, 2π]
    
    def __str__(self) -> str:
        """String representation."""
        return f"SphericalCoordinates(r={self.r:.3f}, θ={self.theta:.3f}, φ={self.phi:.3f})"


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
        """Return self (already Cartesian).
        
        Returns
        -------
        CartesianCoordinates
            Copy of self
        """
        return CartesianCoordinates(self.x, self.y, self.z)
    
    def to_spherical(self) -> "SphericalCoordinates":
        """Convert to spherical coordinates.
        
        Returns
        -------
        SphericalCoordinates
            Equivalent spherical coordinates
        """
        r = self.magnitude
        
        if r == 0:
            return SphericalCoordinates(0.0, 0.0, 0.0)
        
        theta = math.acos(np.clip(self.z / r, -1, 1))  # Clamp to avoid numerical issues
        phi = math.atan2(self.y, self.x)
        
        # Ensure phi is in [0, 2π]
        if phi < 0:
            phi += 2 * np.pi
            
        return SphericalCoordinates(r, theta, phi)
    
    def distance_to(self, other: "Coordinates") -> float:
        """Calculate Euclidean distance to another coordinate.
        
        Parameters
        ----------
        other : Coordinates
            Other coordinate point
            
        Returns
        -------
        float
            Euclidean distance
        """
        if isinstance(other, CartesianCoordinates):
            dx = self.x - other.x
            dy = self.y - other.y
            dz = self.z - other.z
            return math.sqrt(dx*dx + dy*dy + dz*dz)
        else:
            # Convert to Cartesian first
            cart_other = other.to_cartesian()
            return self.distance_to(cart_other)
    
    @property
    def magnitude(self) -> float:
        """Get the magnitude of the vector.
        
        Returns
        -------
        float
            Vector magnitude
        """
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    @property
    def magnitude_squared(self) -> float:
        """Get the squared magnitude (faster than magnitude).
        
        Returns
        -------
        float
            Squared vector magnitude
        """
        return self.x*self.x + self.y*self.y + self.z*self.z
    
    def normalize(self) -> "CartesianCoordinates":
        """Return normalized vector.
        
        Returns
        -------
        CartesianCoordinates
            Unit vector in same direction
            
        Raises
        ------
        GeometryError
            If trying to normalize zero vector
        """
        mag = self.magnitude
        if mag == 0:
            raise GeometryError("Cannot normalize zero vector")
        return CartesianCoordinates(self.x/mag, self.y/mag, self.z/mag)
    
    def dot(self, other: "CartesianCoordinates") -> float:
        """Dot product with another vector.
        
        Parameters
        ----------
        other : CartesianCoordinates
            Other vector
            
        Returns
        -------
        float
            Dot product
        """
        if not isinstance(other, CartesianCoordinates):
            raise TypeError("Other coordinate must be CartesianCoordinates")
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: "CartesianCoordinates") -> "CartesianCoordinates":
        """Cross product with another vector.
        
        Parameters
        ----------
        other : CartesianCoordinates
            Other vector
            
        Returns
        -------
        CartesianCoordinates
            Cross product vector
        """
        if not isinstance(other, CartesianCoordinates):
            raise TypeError("Other coordinate must be CartesianCoordinates")
        
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        
        return CartesianCoordinates(x, y, z)
    
    def angle_between(self, other: "CartesianCoordinates") -> float:
        """Calculate angle between two vectors.
        
        Parameters
        ----------
        other : CartesianCoordinates
            Other vector
            
        Returns
        -------
        float
            Angle in radians [0, π]
        """
        if not isinstance(other, CartesianCoordinates):
            raise TypeError("Other coordinate must be CartesianCoordinates")
        
        mag_product = self.magnitude * other.magnitude
        if mag_product == 0:
            raise GeometryError("Cannot compute angle with zero vector")
        
        cos_angle = np.clip(self.dot(other) / mag_product, -1, 1)
        return math.acos(cos_angle)
    
    def project_onto(self, other: "CartesianCoordinates") -> "CartesianCoordinates":
        """Project this vector onto another vector.
        
        Parameters
        ----------
        other : CartesianCoordinates
            Vector to project onto
            
        Returns
        -------
        CartesianCoordinates
            Projected vector
        """
        if not isinstance(other, CartesianCoordinates):
            raise TypeError("Other coordinate must be CartesianCoordinates")
        
        other_mag_sq = other.magnitude_squared
        if other_mag_sq == 0:
            raise GeometryError("Cannot project onto zero vector")
        
        scalar_proj = self.dot(other) / other_mag_sq
        return CartesianCoordinates(
            scalar_proj * other.x,
            scalar_proj * other.y, 
            scalar_proj * other.z
        )
    
    def __add__(self, other: "CartesianCoordinates") -> "CartesianCoordinates":
        """Vector addition."""
        if not isinstance(other, CartesianCoordinates):
            return NotImplemented
        return CartesianCoordinates(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: "CartesianCoordinates") -> "CartesianCoordinates":
        """Vector subtraction."""
        if not isinstance(other, CartesianCoordinates):
            return NotImplemented
        return CartesianCoordinates(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> "CartesianCoordinates":
        """Scalar multiplication."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return CartesianCoordinates(scalar * self.x, scalar * self.y, scalar * self.z)
    
    def __rmul__(self, scalar: float) -> "CartesianCoordinates":
        """Reverse scalar multiplication."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> "CartesianCoordinates":
        """Scalar division."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if scalar == 0:
            raise GeometryError("Cannot divide by zero")
        return CartesianCoordinates(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __str__(self) -> str:
        """String representation."""
        return f"CartesianCoordinates(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


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
        if coords.ndim != 2 or coords.shape[1] != 3:
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
        if coords.ndim != 2 or coords.shape[1] != 3:
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
        axis = np.asarray(axis)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            raise GeometryError("Rotation axis cannot be zero vector")
        axis = axis / axis_norm
        
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
    def rotation_matrix_euler(alpha: float, beta: float, gamma: float, 
                             convention: str = "ZYZ") -> np.ndarray:
        """Create rotation matrix from Euler angles.
        
        Parameters
        ----------
        alpha, beta, gamma : float
            Euler angles in radians
        convention : str
            Euler angle convention ("ZYZ", "XYZ", "ZXZ")
            
        Returns
        -------
        np.ndarray
            3x3 rotation matrix
        """
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        
        if convention == "ZYZ":
            return np.array([
                [ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb],
                [sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb],
                [-sb*cg, sb*sg, cb]
            ])
        elif convention == "XYZ":
            return np.array([
                [cb*cg, -cb*sg, sb],
                [ca*sg + sa*sb*cg, ca*cg - sa*sb*sg, -sa*cb],
                [sa*sg - ca*sb*cg, sa*cg + ca*sb*sg, ca*cb]
            ])
        elif convention == "ZXZ":
            return np.array([
                [ca*cg - sa*cb*sg, -ca*sg - sa*cb*cg, sa*sb],
                [sa*cg + ca*cb*sg, -sa*sg + ca*cb*cg, -ca*sb],
                [sb*sg, sb*cg, cb]
            ])
        else:
            raise GeometryError(f"Unknown Euler angle convention: {convention}")
    
    @staticmethod
    def rotation_matrix_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Create rotation matrix that rotates v1 to v2.
        
        Parameters
        ----------
        v1, v2 : np.ndarray
            Vectors to align (will be normalized)
            
        Returns
        -------
        np.ndarray
            3x3 rotation matrix
        """
        v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Check if vectors are already aligned
        dot_product = np.dot(v1_norm, v2_norm)
        if np.abs(dot_product - 1.0) < 1e-10:
            return np.eye(3)  # Identity matrix
        
        # Check if vectors are opposite
        if np.abs(dot_product + 1.0) < 1e-10:
            # Find any perpendicular vector
            if np.abs(v1_norm[0]) < 0.9:
                perp = np.array([1, 0, 0])
            else:
                perp = np.array([0, 1, 0])
            
            axis = np.cross(v1_norm, perp)
            axis = axis / np.linalg.norm(axis)
            return RotationMatrix.rotation_matrix_axis_angle(axis, np.pi)
        
        # General case: use Rodrigues' rotation formula
        cross_product = np.cross(v1_norm, v2_norm)
        axis = cross_product / np.linalg.norm(cross_product)
        angle = np.arccos(np.clip(dot_product, -1, 1))
        
        return RotationMatrix.rotation_matrix_axis_angle(axis, angle)
    
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
    
    @staticmethod
    def compose_rotations(*rotation_matrices: np.ndarray) -> np.ndarray:
        """Compose multiple rotation matrices.
        
        Parameters
        ----------
        *rotation_matrices : np.ndarray
            Sequence of 3x3 rotation matrices
            
        Returns
        -------
        np.ndarray
            Composed rotation matrix
        """
        if not rotation_matrices:
            return np.eye(3)
        
        result = rotation_matrices[0]
        for matrix in rotation_matrices[1:]:
            result = result @ matrix
        
        return result