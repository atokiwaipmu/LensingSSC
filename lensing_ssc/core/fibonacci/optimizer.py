import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from typing import Tuple

from lensing_ssc.core.fibonacci.fibonacci import FibonacciGrid
from lensing_ssc.core.fibonacci.rotation import SphereRotator

class PatchOptimizer:
    """
    Optimizes the number of non-overlapping patches for a given Healpix map.

    Attributes:
        nside (int): Healpix resolution parameter.
        patch_size (float): Patch size in degrees.
        Ninit (int): Initial upper bound on the number of patches.
        radius (float): Patch radius in radians (derived from patch_size).
        N_opt (Optional[int]): The optimal (maximum feasible) number of patches.
    """

    def __init__(self, nside=1024, patch_size=10, Ninit=280):
        """
        Initialize the FibonacciOptimizer.

        Args:
            nside (int): Healpix resolution parameter.
            patch_size (float): Patch size in degrees.
            Ninit (int): Initial upper bound on the number of patches.
        """
        self.nside = nside
        self.patch_size = patch_size
        # Calculate the effective patch radius (in radians)
        self.radius = np.radians(patch_size) * np.sqrt(2)
        self.Ninit = Ninit
        self.N_opt = None

    def optimize(self, verbose=False) -> int:
        """
        Optimize the patch count using a binary search approach.

        Args:
            verbose (bool, optional): If True, prints progress messages.

        Returns:
            int: The optimal (maximum feasible) number of patches.
        """
        low = 1
        high = self.Ninit
        best_feasible = 0

        while low <= high:
            mid = (low + high) // 2
            # Ensure mid is odd
            if mid % 2 == 0:
                if mid + 1 <= high:
                    mid += 1
                else:
                    mid -= 1
            if mid < low or mid > high:
                break
            if verbose:
                print(f"Testing feasibility for N = {mid} ...")

            if self.is_feasible(mid):
                best_feasible = mid
                low = mid + 2  # Only test odd numbers
                if verbose:
                    print(f"N = {mid} is feasible; trying larger values.")
            else:
                high = mid - 2  # Only test odd numbers
                if verbose:
                    print(f"N = {mid} is not feasible; trying smaller values.")

        self.N_opt = best_feasible
        return best_feasible

    def is_feasible(self, N: int) -> bool:
        """
        Check if N patches can be placed without overlap.

        Args:
            N (int): Number of patches to test.

        Returns:
            bool: True if patches do not overlap; False otherwise.
        """
        # Generate N points using Fibonacci grid
        points = FibonacciGrid.fibonacci_grid_on_sphere(N)
        # Select only those points that are not too close to the poles
        valid_points = points[(points[:, 0] < np.pi - self.radius) & (points[:, 0] > self.radius)]
        
        # Initialize coverage count for each pixel
        npix = hp.nside2npix(self.nside)
        counts = np.zeros(npix, dtype=int)

        for center in valid_points:
            vertices = self.rotated_vertices(center)
            # Convert patch vertices (theta, phi) to unit vectors
            vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
            # Query pixels covered by the patch polygon
            ipix = hp.query_polygon(nside=self.nside, vertices=vecs, nest=True)
            counts[ipix] += 1
            # Early exit if any pixel is covered more than once
            if np.any(counts[ipix] > 1):
                return False

        return True

    def rotated_vertices(self, center: Tuple[float, float]) -> np.ndarray:
        """
        Compute rotated vertices for a patch centered at a given point.

        Args:
            center (Tuple[float, float]): The center of the patch in (theta, phi).

        Returns:
            np.ndarray: An array of shape (4, 2) containing the patch vertices in (theta, phi).
        """
        # Generate base vertices for a patch centered at [pi/2, 0]
        base_vertices = self.vertices_from_center((np.pi / 2, 0))
        # Rotate each vertex so that the patch centers at the given point
        rotated = np.array([SphereRotator.rotate_point(center[0], center[1], vertex)
                            for vertex in base_vertices])
        return rotated

    def vertices_from_center(self, center: Tuple[float, float]) -> np.ndarray:
        """
        Compute the vertices of a square patch based on a given center in spherical coordinates.

        Note: This method approximates a square patch by displacing the polar angle
        and azimuthal angle by half the side length.

        Args:
            center (Tuple[float, float]): Center of the patch (theta, phi).

        Returns:
            np.ndarray: An array of shape (4, 2) with vertices in (theta, phi).
        """
        theta, phi = center
        half_size = np.radians(self.patch_size) / np.sqrt(2)
        vertices = np.array([
            [theta, phi + half_size * np.sin(theta)],
            [theta + half_size, phi],
            [theta, phi - half_size * np.sin(theta)],
            [theta - half_size, phi]
        ])
        return vertices