# lensing_ssc/core/patching_utils.py
import logging
import multiprocessing
from multiprocessing import shared_memory
from pathlib import Path
from typing import Tuple, List

import healpy as hp
import numpy as np

from lensing_ssc.core.fibonacci_utils import FibonacciGrid # Updated import
from lensing_ssc.utils.extractors import InfoExtractor # Will be moved later if necessary


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
        center_points_path: str = "lensing_ssc/core/fibonacci/center_points/" # Updated default path
    ) -> None:
        """Initializes the PatchProcessor with configuration parameters.

        Args:
            patch_size_deg (float, optional): Size of each patch in degrees. Defaults to 10.0.
            xsize (int, optional): Size (in pixels) of the projected patch image. Defaults to 2048.
            center_points_path (str, optional): Directory where the center points file is stored.
                The file is expected to be named as 'fibonacci_points_{patch_size_deg}.txt'.
                Defaults to "lensing_ssc/core/fibonacci/center_points/".

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
            # Try to load or generate points using FibonacciGrid
            logging.warning(f"Center points file not found at {file_path}. Attempting to generate using FibonacciGrid.")
            # A default n_opt is needed here. Let's assume a reasonable default or require it.
            # For now, let's set a placeholder n_opt. This should be configured appropriately.
            n_opt_placeholder = 2 * int((4 * np.pi * (180/np.pi)**2) / (patch_size_deg**2)) + 1 # Estimate based on area
            if n_opt_placeholder % 2 == 0:
                n_opt_placeholder +=1
            self.center_points = FibonacciGrid.load_or_generate_points(patch_size_deg, n_opt_placeholder, center_points_path)
            self.npatch = self.center_points.shape[0]
            if self.npatch == 0:
                 error_msg = f"Failed to load or generate center points using FibonacciGrid from {center_points_path}"
                 logging.error(error_msg)
                 raise FileNotFoundError(error_msg)
            logging.info(f"Generated and loaded {self.npatch} points from {center_points_path}")

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


class PatchGenerator:
    """
    Generates patches from full-sky kappa maps using a provided PatchProcessor.

    Parameters
    ----------
    pp : PatchProcessor
        An instance of PatchProcessor.
    zs_list : List[float]
        List of source redshifts to process.
    data_dir : str, optional
        Directory containing full-sky maps. Defaults to the specified path.
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    """
    def __init__(
        self,
        pp: PatchProcessor,
        zs_list: List[float],
        data_dir: str = "/lustre/work/akira.tokiwa/Projects/LensingSSC/data",
        overwrite: bool = False
    ) -> None:
        self.data_dir: Path = Path(data_dir)
        self.zs_list: List[float] = zs_list
        self.pp: PatchProcessor = pp
        self.overwrite: bool = overwrite

        # Expect files organized under fullsky/<seed>/zs*/ with filenames matching kappa_zs*_s*.fits
        self.kappa_paths: List[Path] = sorted(self.data_dir.glob("fullsky/*/zs*/*.fits"))
        self.output_dir: Path = self.data_dir / f"patch_oa{int(self.pp.patch_size_deg)}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dir_initialize()

    def dir_initialize(self) -> None:
        """
        Create necessary output subdirectories for each redshift and box type.
        """
        for zs in self.zs_list:
            for box_type in ["bigbox", "tiled"]:
                output_subdir: Path = self.output_dir / box_type / f"zs{zs} / noiseless"
                output_subdir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory {output_subdir}")

    def run(self) -> None:
        """
        Process each full-sky kappa map to generate patches.
        """
        for kappa_path in self.kappa_paths:
            info = InfoExtractor.extract_info_from_path(kappa_path)
            box_type: str = kappa_path.parts[-3]  # Assumes structure: .../fullsky/<seed>/zs*/...
            # Handle cases where info extraction might fail or box_type is not as expected.
            if not info or not box_type or not info.get('redshift') or not info.get('seed'):
                logging.warning(f"Could not extract complete info or valid box_type from {kappa_path}. Skipping.")
                continue
            
            input_map: np.ndarray = self.load_map(kappa_path)
            output_dir_str_template = f"{{box_type}}/zs{{redshift}}/noiseless" # Using f-string like syntax for clarity
            output_dir: Path = self.output_dir / output_dir_str_template.format(box_type=box_type, redshift=info['redshift'])
            output_dir.mkdir(parents=True, exist_ok=True) # Ensure output_dir exists

            logging.info(f"Output directory: {output_dir}")
            output_path: Path = self.generate_fname(kappa_path, output_dir, info)
            if output_path.exists() and not self.overwrite:
                logging.info(f"Skipping existing file: {output_path}")
                continue
            logging.info(f"Generating patches for seed = {info['seed']} zs = {info['redshift']}")
            self.make_save_patches(input_map, output_path)

    def generate_fname(self, data_path: Path, output_dir: Path, info: dict, ngal: int = 0) -> Path:
        """
        Generate the output filename based on the input file and noise level.

        Parameters
        ----------
        data_path : Path
            Path to the input kappa map file.
        output_dir : Path
            Output directory to save patches.
        info : dict
            Dictionary containing extracted information like seed and redshift.
        ngal : int, optional
            Galaxy noise parameter. Defaults to 0.

        Returns
        -------
        Path
            Generated output file path.
        """
        # Use info dict for seed and redshift to ensure consistency
        seed = info.get('seed', 'unknown')
        redshift = info.get('redshift', 'unknown')
        
        # Construct filename from parts to avoid issues with data_path.stem if it contains unexpected characters
        base_fname = f"patches_zs{redshift}_s{seed}"
        
        if ngal == 0:
            fname = f"{base_fname}_oa{int(self.pp.patch_size_deg)}_noiseless.npy"
        else:
            fname = f"{base_fname}_oa{int(self.pp.patch_size_deg)}_ngal{ngal}.npy"
        return output_dir / fname

    def load_map(self, data_path: Path) -> np.ndarray:
        """
        Load and reorder a Healpy map from the given file.

        Parameters
        ----------
        data_path : Path
            Path to the .fits file.

        Returns
        -------
        np.ndarray
            Reordered kappa map.
        """
        logging.debug(f"Loading map from {data_path}")
        map_data: np.ndarray = hp.read_map(str(data_path))
        return hp.reorder(map_data, n2r=True)

    def make_save_patches(self, input_map: np.ndarray, output_path: Path) -> None:
        """
        Extract patches using the PatchProcessor and save them to disk.

        Parameters
        ----------
        input_map : np.ndarray
            The full-sky kappa map.
        output_path : Path
            File path where the patches are saved.
        """
        patches: np.ndarray = self.pp.make_patches(input_map)
        np.save(output_path, patches)
        logging.info(f"Saved patches to {output_path}") 