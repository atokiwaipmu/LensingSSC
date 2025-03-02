import math
import numpy as np
from typing import Tuple

from lensing_ssc.core.fibonacci.spherical import SphericalConverter

class Rotation:
    @staticmethod
    def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
        """
        Generate a rotation matrix for a counterclockwise rotation about a given axis by theta radians.

        Args:
            axis (np.ndarray): A 3-dimensional axis.
            theta (float): Rotation angle in radians.

        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        axis = axis / np.linalg.norm(axis)
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        return np.array([
            [a * a + b * b - c * c - d * d, 2 * (b * c + a * d),       2 * (b * d - a * c)],
            [2 * (b * c - a * d),             a * a + c * c - b * b - d * d, 2 * (c * d + a * b)],
            [2 * (b * d + a * c),             2 * (c * d - a * b),       a * a + d * d - b * b - c * c]
        ])

class SphereRotator:
    @staticmethod
    def rotation_matrix(theta: float, phi: float) -> np.ndarray:
        """
        Generate a composite rotation matrix for rotating a point on the sphere by the given angles.

        Args:
            theta (float): Polar rotation angle in radians.
            phi (float): Azimuthal rotation angle in radians.

        Returns:
            np.ndarray: A 3x3 composite rotation matrix.
        """
        R_phi = Rotation.rotation_matrix(np.array([0, 0, 1]), phi)
        # Rotate about an axis perpendicular to the new phi direction
        axis_theta = np.array([np.sin(phi), -np.cos(phi), 0])
        R_theta = Rotation.rotation_matrix(axis_theta, math.pi / 2 - theta)
        return R_theta @ R_phi

    @staticmethod
    def rotate_point(theta: float, phi: float, point: Tuple[float, ...]) -> Tuple[float, float]:
        """
        Rotate a point on the sphere by the specified theta and phi angles.

        Args:
            theta (float): Polar rotation angle in radians.
            phi (float): Azimuthal rotation angle in radians.
            point (Tuple[float, ...]): Original point in spherical coordinates.
                Should be provided as (theta, phi) or (theta, phi, r); if r is omitted, r defaults to 1.0.

        Returns:
            Tuple[float, float]: Rotated spherical coordinates (theta, phi).
        """
        if len(point) == 2:
            point = (*point, 1.0)
        x, y, z = SphericalConverter.spherical_to_cartesian(point[0], point[1], point[2])
        rotation_mat = SphereRotator.rotation_matrix(theta, phi)
        x_rot, y_rot, z_rot = rotation_mat @ np.array([x, y, z])
        _, theta_rot, phi_rot = SphericalConverter.cartesian_to_spherical(x_rot, y_rot, z_rot)
        return theta_rot, phi_rot