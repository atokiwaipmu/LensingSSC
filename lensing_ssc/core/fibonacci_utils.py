# lensing_ssc/core/fibonacci_utils.py
import numpy as np
import healpy as hp
from pathlib import Path
import logging
from typing import Optional, Tuple
import math # Added for Rotation/SphereRotator

class FibonacciGrid:
    @staticmethod
    def fibonacci_grid_on_sphere(n: int) -> np.ndarray:
        """
        Generate a Fibonacci lattice grid of n points distributed over the surface of a sphere.

        Note: n must be an odd integer greater than or equal to 3.

        Args:
            n (int): Number of points (odd integer).

        Returns:
            np.ndarray: An (n x 2) array of spherical coordinates (theta, phi).
        """
        if n < 3 or n % 2 == 0:
            raise ValueError("Number of points must be an odd integer and at least 3.")
        N = (n - 1) // 2
        golden_ratio = (1 + np.sqrt(5)) / 2  # Golden ratio
        indices = np.arange(-N, N + 1, 1, dtype=int)
        theta_i = np.arcsin(2 * indices / (2 * N + 1))
        phi_i = 2 * np.pi * indices / golden_ratio

        # Shift theta into [0, π]
        theta_i += np.pi / 2
        # Wrap phi into [0, 2π)
        phi_i = np.mod(phi_i, 2 * np.pi)
        return np.column_stack((theta_i, phi_i))

    @staticmethod
    def get_patch_pixels(image: np.ndarray, side_length: int) -> np.ndarray:
        """
        Extract a square patch from the center of an image after rotating it by 45 degrees.
        If the extracted patch is smaller than the desired side length due to image boundaries,
        it is padded with zeros.

        Args:
            image (np.ndarray): Input image array.
            side_length (int): Desired side length of the square patch.

        Returns:
            np.ndarray: Extracted and, if necessary, padded square patch.
        """
        from scipy.ndimage import rotate
        # Rotate the image by 45 degrees without changing its overall shape.
        rotated_image = rotate(image, 45, reshape=False)

        center_y, center_x = np.array(rotated_image.shape) // 2
        half_side = side_length // 2

        y_start = max(center_y - half_side, 0)
        y_end = y_start + side_length
        x_start = max(center_x - half_side, 0)
        x_end = x_start + side_length

        y_end = min(y_end, rotated_image.shape[0])
        x_end = min(x_end, rotated_image.shape[1])

        patch = rotated_image[y_start:y_end, x_start:x_end]

        # Pad the patch if it's smaller than the desired side length.
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
    
    @staticmethod
    def load_or_generate_points(patch_size: float, n_opt: int, 
                                data_dir: str = "lensing_ssc/core/fibonacci/center_points/") -> np.ndarray: # Updated default path
        """
        Load or generate Fibonacci grid points.

        Args:
            patch_size (float): Patch size (degrees).
            n_opt (int): Optimal number of patches.
            data_dir (str, optional): Directory to save/load Fibonacci points. 
                                    Defaults to "lensing_ssc/core/fibonacci/center_points/".

        Returns:
            np.ndarray: Array containing the spherical coordinates (theta, phi) of each point.
        """
        data_path = Path(data_dir) / f"fibonacci_points_{patch_size}.txt"
        try:
            points = np.loadtxt(data_path)
            logging.info(f"Loaded Fibonacci points from {data_path}")
            return points
        except FileNotFoundError:
            logging.warning(f"File not found: {data_path}. Generating new points...")
        
        points = FibonacciGrid.fibonacci_grid_on_sphere(n_opt)
        
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        np.savetxt(data_path, points)
        logging.info(f"Saved new Fibonacci points to {data_path}")
        return points 

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
        Initialize the PatchOptimizer.

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
        self.N_opt = None # Will be set by optimize() or can be pre-loaded

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
            if verbose:
                logging.info(f"Testing feasibility for N = {mid} ...") # Use logging

            if self.is_feasible(mid):
                best_feasible = mid
                low = mid + 1
                if verbose:
                    logging.info(f"N = {mid} is feasible; trying larger values.")
            else:
                high = mid - 1
                if verbose:
                    logging.info(f"N = {mid} is not feasible; trying smaller values.")

        self.N_opt = best_feasible
        if verbose:
            logging.info(f"Optimal number of patches found: {self.N_opt}")
        return best_feasible

    def is_feasible(self, N: int) -> bool:
        """
        Check if N patches can be placed without overlap.

        Args:
            N (int): Number of patches to test.

        Returns:
            bool: True if patches do not overlap; False otherwise.
        """
        if N <= 0: return True # No patches or non-positive number is feasible (no overlaps)
        if N % 2 == 0: N +=1 # Ensure N is odd for FibonacciGrid, or handle appropriately
        
        points = FibonacciGrid.fibonacci_grid_on_sphere(N)
        valid_points = points[(points[:, 0] < np.pi - self.radius) & (points[:, 0] > self.radius)]
        
        if not valid_points.any(): # No valid points to place patches
            return True # Feasible if no patches can be placed (e.g., N too small, radius too large)

        npix = hp.nside2npix(self.nside)
        counts = np.zeros(npix, dtype=int)

        for center in valid_points:
            vertices = self.rotated_vertices(center)
            vecs = hp.ang2vec(vertices[:, 0], vertices[:, 1])
            ipix = hp.query_polygon(nside=self.nside, vertices=vecs, nest=True)
            counts[ipix] += 1
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
        base_vertices = self.vertices_from_center((np.pi / 2, 0))
        rotated_list = []
        for vertex in base_vertices:
             # Ensure vertex is (theta, phi) before passing to rotate_point
            rot_theta, rot_phi = SphereRotator.rotate_point(center[0], center[1], (vertex[0], vertex[1]))
            rotated_list.append([rot_theta, rot_phi])
        return np.array(rotated_list)

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
        half_size_rad = np.radians(self.patch_size) / np.sqrt(2) # This is half diagonal for a square of side patch_size
        # For a square patch on a sphere, defining vertices is non-trivial.
        # This method is an approximation.
        # V1: (theta, phi + d_phi) where d_phi depends on theta
        # V2: (theta + d_theta, phi)
        # V3: (theta, phi - d_phi)
        # V4: (theta - d_theta, phi)
        # d_phi should be half_size_rad / sin(theta) to maintain angular width.
        # This approximation can lead to distortions, especially near poles.
        
        # Using a simpler fixed angular displacement for theta and phi as in original, but can be improved.
        # The displacement in phi should ideally scale with 1/sin(theta).
        # However, the original code used `half_size * np.sin(theta)` which is peculiar.
        # Let's assume `half_size` refers to angular displacement along cardinal directions from center.
        # Consider a small square patch. d_theta = half_angular_side, d_phi = half_angular_side / sin(theta_center)
        
        d_theta = half_size_rad 
        d_phi = half_size_rad # Simplified; for true squareness, d_phi should be d_theta / np.sin(theta)
                              # if theta is small, this can be large. For now, using equal angular displacement.
                              # The original `half_size * np.sin(theta)` is dimensionally incorrect if half_size is angle.
                              # Let's assume `half_size_rad` is the angular extent.

        vertices = np.array([
            [theta, phi + d_phi],  # East
            [theta + d_theta, phi],  # South
            [theta, phi - d_phi],  # West
            [theta - d_theta, phi]   # North
        ])
        # Ensure theta is within [0, pi] and phi within [0, 2pi]
        vertices[:,0] = np.clip(vertices[:,0], 0, np.pi)
        vertices[:,1] = vertices[:,1] % (2 * np.pi)
        return vertices 