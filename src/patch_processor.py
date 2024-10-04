import logging
import multiprocessing
from multiprocessing import shared_memory
from pathlib import Path
from typing import List, Tuple

import healpy as hp
import numpy as np

from src.fibonacci_helper import FibonacciHelper


class PatchProcessor:
    """
    Processes a full-sky map by extracting patches at specified points.

    Attributes:
        npatch (int): Number of patches to extract.
        patch_size_deg (float): Size of each patch in degrees.
        xsize (int): Size of the projected patch image in pixels.
        padding (float): Padding factor for the projected patch.
        resolution_arcmin (float): Resolution of the projected patch in arcminutes.
    """

    # Conversion constants
    DEG_TO_RAD: float = np.pi / 180.0
    ARCMIN_TO_RAD: float = DEG_TO_RAD / 60.0

    def __init__(self, npatch: int = 273, patch_size_deg: float = 10.0, xsize: int = 2048) -> None:
        """
        Initializes the PatchProcessor with the desired configuration.

        Args:
            npatch (int, optional): Number of patches to extract. Defaults to 273.
            patch_size_deg (float, optional): Size of each patch in degrees. Defaults to 10.0.
            xsize (int, optional): Size of the projected patch image in pixels. Defaults to 2048.
        """
        self.npatch = npatch
        self.patch_size_deg = patch_size_deg
        self.xsize = xsize
        self.padding = 0.1 + np.sqrt(2)
        self.resolution_arcmin = (self.patch_size_deg * 60.0) / self.xsize

        logging.info(
            f"Initialized PatchProcessor with npatch={self.npatch}, "
            f"patch_size_deg={self.patch_size_deg}, xsize={self.xsize}"
        )

    def make_patches(
        self, input_map: np.ndarray, num_processes: int = multiprocessing.cpu_count()
    ) -> np.ndarray:
        """
        Extracts patches from the input map at valid points using multiprocessing.

        Args:
            input_map (np.ndarray): The input full-sky map.
            num_processes (int, optional): Number of parallel processes to use. Defaults to number of CPU cores.

        Returns:
            np.ndarray: A 3D array of extracted patches with shape (npatch, xsize, xsize).
        """
        logging.info("Starting patch extraction process.")

        valid_points = self._get_valid_points()
        points_lonlat_deg = [
            hp.rotator.vec2dir(hp.ang2vec(*point), lonlat=True) for point in valid_points
        ]

        logging.debug(f"Total valid points for patch extraction: {len(points_lonlat_deg)}")

        # Create shared memory for the input map
        shm = shared_memory.SharedMemory(create=True, size=input_map.nbytes)
        shared_input_map = np.ndarray(input_map.shape, dtype=input_map.dtype, buffer=shm.buf)
        np.copyto(shared_input_map, input_map)
        logging.debug("Input map copied to shared memory.")

        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                patches = pool.starmap(
                    self._extract_patch_worker,
                    [(shm.name, input_map.shape, input_map.dtype, point) for point in points_lonlat_deg]
                )
            logging.info("Patch extraction completed successfully.")

        except Exception as e:
            logging.error(f"An error occurred during patch extraction: {e}")
            raise e

        finally:
            # Clean up shared memory
            shm.close()
            shm.unlink()
            logging.debug("Shared memory cleaned up.")

        return np.array(patches, dtype=np.float32)

    def _extract_patch_worker(
        self,
        shm_name: str,
        shape: Tuple[int, int],
        dtype: np.dtype,
        point_lonlat_deg: Tuple[float, float],
    ) -> np.ndarray:
        """
        Worker function to extract a single patch from the input map.

        Args:
            shm_name (str): Name of the shared memory block.
            shape (Tuple[int, int]): Shape of the input map.
            dtype_str (str): Data type of the input map.
            point_lonlat_deg (Tuple[float, float]): Longitude and latitude in degrees of the patch center.

        Returns:
            np.ndarray: The extracted and processed patch with shape (xsize, xsize).
        """
        try:
            # Access the shared memory
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            input_map = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

            # Extract the patch using gnomonic projection
            patch_projected = hp.gnomview(
                input_map,
                rot=point_lonlat_deg,
                xsize=self.xsize * self.padding,
                reso=self.resolution_arcmin,
                return_projected_map=True,
                nest=False,
                no_plot=True,
            )

            # Process the patch to the desired xsize
            patch_processed = FibonacciHelper.get_patch_pixels(patch_projected, self.xsize).astype(np.float32)

            # Clean up shared memory in the worker
            existing_shm.close()

            logging.debug(f"Patch extracted at {point_lonlat_deg}.")

            return patch_processed

        except Exception as e:
            logging.error(f"Error in patch extraction worker for point {point_lonlat_deg}: {e}")
            return np.zeros((2048, 2048), dtype=np.float32)  # Return an empty patch on failure

    def _get_valid_points(self) -> np.ndarray:
        """
        Retrieves valid points on the sphere for patch extraction using a Fibonacci grid.

        Returns:
            np.ndarray: A 2D array of valid points in theta-phi coordinates (radians).
        """
        logging.debug("Generating Fibonacci grid points on the sphere.")
        points = FibonacciHelper.fibonacci_grid_on_sphere(self.npatch)

        # Calculate the angular radius in radians
        angular_radius_rad = self.DEG_TO_RAD * self.patch_size_deg * np.sqrt(2)

        # Filter points to ensure patches do not overlap the poles excessively
        valid_mask = (points[:, 0] < (np.pi - angular_radius_rad)) & (points[:, 0] > angular_radius_rad)
        valid_points = points[valid_mask]

        logging.debug(f"Valid points after applying angular radius filter: {len(valid_points)}")

        return valid_points