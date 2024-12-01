# src/fibonacci_helper.py

import math
from typing import Tuple

import numpy as np
from scipy.ndimage import rotate

class Rotation:
    @staticmethod
    def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
        """
        Generate a rotation matrix for a counterclockwise rotation around a given axis by theta radians.

        Args:
            axis (np.ndarray): The axis to rotate around (3-dimensional).
            theta (float): The rotation angle in radians.

        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        axis = axis / np.linalg.norm(axis)
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        return np.array([
            [a * a + b * b - c * c - d * d, 2 * (b * c + a * d),     2 * (b * d - a * c)],
            [2 * (b * c - a * d),             a * a + c * c - b * b - d * d, 2 * (c * d + a * b)],
            [2 * (b * d + a * c),             2 * (c * d - a * b),     a * a + d * d - b * b - c * c]
        ])
    
class SphericalConverter:
    @staticmethod
    def spherical_to_cartesian(theta: float, phi: float, r: float = 1.0) -> Tuple[float, float, float]:
        """
        Convert spherical coordinates to Cartesian coordinates.

        Args:
            theta (float): Polar angle in radians.
            phi (float): Azimuthal angle in radians.
            r (float, optional): Radius. Defaults to 1.0.

        Returns:
            Tuple[float, float, float]: Cartesian coordinates (x, y, z).
        """
        sin_theta = np.sin(theta)
        x = r * sin_theta * np.cos(phi)
        y = r * sin_theta * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert Cartesian coordinates to spherical coordinates.

        Args:
            x (float): X-coordinate.
            y (float): Y-coordinate.
            z (float): Z-coordinate.

        Returns:
            Tuple[float, float, float]: Spherical coordinates (r, theta, phi).
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r) if r != 0 else 0.0
        phi = np.arctan2(y, x)
        return r, theta, phi
    
class SphereRotator:
    @staticmethod
    def rotation_matrix(theta: float, phi: float) -> np.ndarray:
        """
        Generate a composite rotation matrix for rotating a point on the sphere by theta and phi.

        Args:
            theta (float): Polar rotation angle in radians.
            phi (float): Azimuthal rotation angle in radians.

        Returns:
            np.ndarray: A 3x3 composite rotation matrix.
        """
        R_phi = Rotation.rotation_matrix(np.array([0, 0, 1]), phi)
        # Rotate around the axis perpendicular to the new phi direction
        axis_theta = np.array([np.sin(phi), -np.cos(phi), 0])
        R_theta = Rotation.rotation_matrix(axis_theta, math.pi / 2 - theta)
        return R_theta @ R_phi


    @staticmethod
    def rotate_point(theta: float, phi: float, point: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Rotate a point on the sphere by theta and phi angles.

        Args:
            theta (float): Polar rotation angle in radians.
            phi (float): Azimuthal rotation angle in radians.
            point (Tuple[float, float, float]): Original point coordinates (theta, phi).

        Returns:
            Tuple[float, float]: Rotated point coordinates (theta, phi).
        """
        x, y, z = SphericalConverter.spherical_to_cartesian(*point)
        rotation_mat = SphereRotator.rotation_matrix(theta, phi)
        x_rot, y_rot, z_rot = rotation_mat @ np.array([x, y, z])
        _, theta_rot, phi_rot = SphericalConverter.cartesian_to_spherical(x_rot, y_rot, z_rot)
        return theta_rot, phi_rot

class FibonacciHelper:
    @staticmethod
    def fibonacci_grid_on_sphere(n: int) -> np.ndarray:
        """
        Generate a Fibonacci lattice grid of N points distributed over the surface of a sphere.

        Args:
            N (int): Number of points.

        Returns:
            np.ndarray: An Nx2 array of spherical coordinates (theta, phi).
        """
        if n//2 == 0:
            raise ValueError("Number of points must be an odd integer.")
        else:
            N = (n-1) // 2

        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        indices = np.arange(-N, N+1, 1, dtype=int)
        theta_i = np.arcsin(2 * indices / (2 * N + 1))
        phi_i = 2 * np.pi * indices / phi

        # Ensure theta_i is within [0, π]
        theta_i += np.pi / 2

        # Ensure phi_i is within [0, 2π)
        phi_i = np.mod(phi_i, 2 * np.pi)
        
        return np.column_stack((theta_i, phi_i))

    @staticmethod
    def get_patch_pixels(image: np.ndarray, side_length: int) -> np.ndarray:
        """
        Extract a square patch of pixels from the center of an image after rotating it by 45 degrees.

        Args:
            image (np.ndarray): Input image array.
            side_length (int): Length of the square patch.

        Returns:
            np.ndarray: Extracted patch of pixels.
        """
        # Rotate the image by 45 degrees without changing the shape
        rotated_image = rotate(image, 45, reshape=False)

        center_y, center_x = np.array(rotated_image.shape) // 2
        half_side = side_length // 2

        y_start = max(center_y - half_side, 0)
        y_end = y_start + side_length
        x_start = max(center_x - half_side, 0)
        x_end = x_start + side_length

        # Ensure the patch does not exceed image boundaries
        y_end = min(y_end, rotated_image.shape[0])
        x_end = min(x_end, rotated_image.shape[1])

        patch = rotated_image[y_start:y_end, x_start:x_end]

        # If the patch is smaller than desired due to image boundaries, pad it
        if patch.shape[0] != side_length or patch.shape[1] != side_length:
            patch = np.pad(
                patch,
                (
                    (0, max(side_length - patch.shape[0], 0)),
                    (0, max(side_length - patch.shape[1], 0))
                ),
                mode='constant',
                constant_values=0
            )

        return patch