import logging
import multiprocessing
from multiprocessing import shared_memory
from pathlib import Path
from typing import Tuple

import healpy as hp
import numpy as np

from lensing_ssc.core.fibonacci.fibonacci import FibonacciGrid


class PatchProcessor:
    """Processes a full-sky map by extracting patches at specified center points.

    Attributes:
        npatch (int): Total number of center points (patches) loaded from file.
        patch_size_deg (float): Size of each patch in degrees.
        xsize (int): Size (in pixels) of the projected patch image.
        padding (float): Padding factor applied during projection.
        resolution_arcmin (float): Resolution of the projected patch in arcminutes.
        center_points (np.ndarray): Array of center points (theta, phi in radians) loaded from file.
    """
    # Class constants
    DEG_TO_RAD: float = np.pi / 180.0
    ARCMIN_TO_RAD: float = DEG_TO_RAD / 60.0

    def __init__(
        self,
        patch_size_deg: float = 10.0,
        xsize: int = 2048,
        center_points_path: str = "/lustre/work/akira.tokiwa/Projects/LensingSSC/src/core/fibonacci/center_points/"
    ) -> None:
        """Initializes the PatchProcessor with configuration parameters.

        Args:
            patch_size_deg (float, optional): Size of each patch in degrees. Defaults to 10.0.
            xsize (int, optional): Size (in pixels) of the projected patch image. Defaults to 2048.
            center_points_path (str, optional): Directory where the center points file is stored.
                The file is expected to be named as 'fibonacci_points_{patch_size_deg}.txt'.

        Raises:
            FileNotFoundError: If the center points file is not found at the specified path.
        """
        self.patch_size_deg = patch_size_deg
        self.xsize = xsize
        self.padding = 0.1 + np.sqrt(2)
        self.resolution_arcmin = (self.patch_size_deg * 60.0) / self.xsize

        file_path = Path(center_points_path) / f"fibonacci_points_{patch_size_deg}.txt"
        if file_path.exists():
            self.center_points = np.loadtxt(file_path)
            self.npatch = self.center_points.shape[0]
        else:
            error_msg = f"Center points file not found at {file_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        logging.info(
            f"Initialized PatchProcessor with npatch={self.npatch}, "
            f"patch_size_deg={self.patch_size_deg}, xsize={self.xsize}"
        )

    def make_patches(
        self,
        input_map: np.ndarray,
        num_processes: int = multiprocessing.cpu_count()
    ) -> np.ndarray:
        """Extracts patches from the input full-sky map at valid center points using multiprocessing.

        Args:
            input_map (np.ndarray): The full-sky map from which patches are extracted.
            num_processes (int, optional): Number of parallel processes to use. Defaults to CPU count.

        Returns:
            np.ndarray: A 3D array of extracted patches with shape (npatch, xsize, xsize).

        Raises:
            Exception: Propagates any exception encountered during patch extraction.
        """
        logging.info("Starting patch extraction process.")

        valid_points: np.ndarray = self._get_valid_points()
        points_lonlat_deg = [
            hp.rotator.vec2dir(hp.ang2vec(*point), lonlat=True) for point in valid_points
        ]
        logging.debug(f"Total valid points for patch extraction: {len(points_lonlat_deg)}")

        shm = shared_memory.SharedMemory(create=True, size=input_map.nbytes)
        shared_input_map = np.ndarray(input_map.shape, dtype=input_map.dtype, buffer=shm.buf)
        np.copyto(shared_input_map, input_map)
        logging.debug("Input map copied to shared memory.")

        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                args_list = [
                    (shm.name, input_map.shape, input_map.dtype, point)
                    for point in points_lonlat_deg
                ]
                patches = pool.starmap(self._extract_patch_worker, args_list)
            logging.info("Patch extraction completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during patch extraction: {e}")
            raise e
        finally:
            shm.close()
            shm.unlink()
            logging.debug("Shared memory cleaned up.")

        return np.array(patches, dtype=np.float32)

    def _extract_patch_worker(
        self,
        shm_name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        point_lonlat_deg: Tuple[float, float]
    ) -> np.ndarray:
        """Worker function to extract a single patch using a gnomonic projection.

        Args:
            shm_name (str): Name of the shared memory block.
            shape (Tuple[int, ...]): Shape of the input map.
            dtype (np.dtype): Data type of the input map.
            point_lonlat_deg (Tuple[float, float]): (Longitude, Latitude) in degrees for the patch center.

        Returns:
            np.ndarray: The extracted patch with shape (xsize, xsize). If extraction fails, returns an empty patch.
        """
        try:
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            input_map = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

            patch_projected = hp.gnomview(
                input_map,
                rot=point_lonlat_deg,
                xsize=int(self.xsize * self.padding),
                reso=self.resolution_arcmin,
                return_projected_map=True,
                nest=False,
                no_plot=True,
            )

            patch_processed = FibonacciGrid.get_patch_pixels(patch_projected, self.xsize).astype(np.float32)
            existing_shm.close()
            logging.debug(f"Patch extracted at {point_lonlat_deg}.")
            return patch_processed

        except Exception as e:
            logging.error(f"Error in patch extraction worker for point {point_lonlat_deg}: {e}")
            return np.zeros((self.xsize, self.xsize), dtype=np.float32)

    def _get_valid_points(self) -> np.ndarray:
        """Filters the loaded center points to exclude those too close to the poles.

        Returns:
            np.ndarray: A 2D array of valid center points in (theta, phi) radians.
        """
        angular_radius_rad = self.DEG_TO_RAD * self.patch_size_deg * np.sqrt(2)
        valid_mask = (
            (self.center_points[:, 0] < (np.pi - angular_radius_rad)) &
            (self.center_points[:, 0] > angular_radius_rad)
        )
        valid_points = self.center_points[valid_mask]
        logging.debug(f"Valid points after angular radius filter: {len(valid_points)}")
        return valid_points