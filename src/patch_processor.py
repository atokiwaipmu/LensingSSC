
import numpy as np
import healpy as hp
import logging
import multiprocessing
from multiprocessing import shared_memory

from src.fibonacci_helper import FibonacciHelper

class PatchProcessor:
    """Processes a full-sky map by extracting patches at given points.

    Attributes:
        npatch (int): Number of patches to extract.
        patch_size (float): Size of each patch in degrees.
        xsize (int): Size of the projected patch image in pixels.
        padding (float, optional): Padding factor for the projected patch.
        reso (float, optional): Resolution of the projected patch in arcminutes.
    """

    def __init__(self, npatch=273, patch_size=10, xsize=2048):
        self.npatch = npatch
        self.patch_size = patch_size
        self.xsize = xsize
        self.padding = 0.1 + np.sqrt(2)
        self.reso = self.patch_size * 60 / self.xsize
        logging.info(f"Initializing PatchProcessor with npatch={npatch}, patch_size={patch_size}")

    def make_patches(self, input_map, num_processes=multiprocessing.cpu_count()):
        """Extracts patches from the input map at valid points.

        Args:
            input_map (np.ndarray): The input full-sky map.

        Returns:
            np.ndarray: A 3D array of extracted patches (npatch, xsize, xsize).
        """

        valid_points = self._get_valid_points()
        points_lonlatdeg = [hp.rotator.vec2dir(hp.ang2vec(*point), lonlat=True) for point in valid_points]

        # Create shared memory for the input map
        shm = shared_memory.SharedMemory(create=True, size=input_map.nbytes)
        shared_input_map = np.ndarray(input_map.shape, dtype=input_map.dtype, buffer=shm.buf)
        np.copyto(shared_input_map, input_map)

        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                patches = pool.starmap(
                    self._extract_patch_worker,
                    [(shm.name, input_map.shape, input_map.dtype, point) for point in points_lonlatdeg]
                )
        finally:
            # Clean up shared memory
            shm.close()
            shm.unlink()

        return np.array(patches).reshape((len(patches), self.xsize, self.xsize)).astype(np.float32)

    def _extract_patch_worker(self, shm_name, shape, dtype, point_lonlatdeg):
        """Extracts a single patch from the input map at a given point.

        Args:
            shm_name (str): The name of the shared memory block.
            shape (tuple): The shape of the input map.
            dtype (np.dtype): The data type of the input map.
            point_lonlatdeg (tuple): The longitude and latitude (in degrees) of the patch center.

        Returns:
            np.ndarray: The extracted patch (xsize, xsize).
        """

        logging.info(f"Extracting patch at {point_lonlatdeg}")

        # Access the shared memory
        shm = shared_memory.SharedMemory(name=shm_name)
        input_map = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # Extract the patch
        patch = hp.gnomview(
            input_map,
            nest=False,
            rot=point_lonlatdeg,
            xsize=int(self.xsize * self.padding),
            reso=self.reso,
            return_projected_map=True,
            no_plot=True
        )

        # Clean up shared memory in the worker
        shm.close()

        return FibonacciHelper.get_patch_pixels(patch, self.xsize).astype(np.float32)
    
    def _get_valid_points(self):
        """Retrieves valid points for patch extraction on the sphere.

        Returns:
            np.ndarray: A 2D array of valid points in theta-phi coordinates.
        """

        points = FibonacciHelper.fibonacci_grid_on_sphere(self.npatch)
        radius = np.radians(self.patch_size) * np.sqrt(2)
        valid_points = points[(points[:, 0] < np.pi - radius) & (points[:, 0] > radius)]
        return valid_points
    