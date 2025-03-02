import numpy as np
import healpy as hp
from pathlib import Path
import logging
from typing import Optional, Tuple

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
                                data_dir: str = "/lustre/work/akira.tokiwa/Projects/LensingSSC/src/core/fibonacci/center_points/") -> np.ndarray:
        """
        Load or generate Fibonacci grid points.

        Args:
            patch_size (float): Patch size (degrees).
            n_opt (int): Optimal number of patches.
            data_dir (str, optional): Directory to save/load Fibonacci points.

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