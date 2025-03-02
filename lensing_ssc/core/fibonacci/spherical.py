import numpy as np
from typing import Tuple

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
            Tuple[float, float, float]: Spherical coordinates (r, theta, phi),
            where theta is the polar angle and phi the azimuthal angle.
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r) if r != 0 else 0.0
        phi = np.arctan2(y, x)
        return r, theta, phi