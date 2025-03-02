import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import healpy as hp

from lensing_ssc.utils.extractors import InfoExtractor

class KappaSmoother:
    """
    Smooths kappa maps using a Gaussian kernel at specified smoothing scales.
    
    The module processes both "kappa" and "noisy" maps (if available) by reading
    the corresponding FITS files, smoothing them, and saving the output to a designated
    directory. Metadata (such as seed, redshift, and galaxy number) is extracted from
    the file path using InfoExtractor.

    Attributes:
        datadir (Path): Directory containing the data.
        nside (int): Healpix NSIDE parameter.
        sl_list (List[float]): List of smoothing scales in arcminutes.
        overwrite (bool): Flag to overwrite existing smoothed maps.
    """

    ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)  # Conversion factor from arcminutes to radians

    def __init__(
        self,
        datadir: str,
        nside: int,
        sl_list: List[float],
        overwrite: bool = False,
    ):
        """
        Initializes the KappaSmoother with directory paths and smoothing parameters.

        Args:
            datadir (str): Path to the data directory.
            nside (int): Healpix NSIDE parameter.
            sl_list (List[float]): List of smoothing scales in arcminutes.
            overwrite (bool, optional): Whether to overwrite existing smoothed maps.
        """
        self.datadir = Path(datadir)
        self.nside = nside
        self.sl_list = sl_list
        self.overwrite = overwrite

        self.smoothed_dir = self.datadir / "smoothed_maps"
        self.smoothed_dir.mkdir(parents=True, exist_ok=True)

        self.noisy_dir = self.datadir / "noisy_maps"
        self.noisy_map_paths = self._get_noisy_map_paths()

        self.kappa_dir = self.datadir / "kappa"
        self.kappa_map_paths = self._get_kappa_map_paths()

    def _get_kappa_map_paths(self) -> List[Path]:
        """
        Retrieves and validates the kappa map file paths.

        Returns:
            List[Path]: Sorted list of kappa map file paths.

        Raises:
            FileNotFoundError: If the kappa directory or maps are not found.
        """
        if not self.kappa_dir.exists():
            logging.error(f"Kappa maps directory not found: {self.kappa_dir}")
            raise FileNotFoundError(f"Kappa maps directory not found: {self.kappa_dir}")

        kappa_map_paths = sorted(self.kappa_dir.glob("*.fits"))
        if not kappa_map_paths:
            logging.error(f"No kappa maps found in {self.kappa_dir}")
            raise FileNotFoundError(f"No kappa maps found in {self.kappa_dir}")

        logging.info(f"Found {len(kappa_map_paths)} kappa maps in {self.kappa_dir}")
        return kappa_map_paths

    def _get_noisy_map_paths(self) -> List[Path]:
        """
        Retrieves and validates the noisy map file paths.

        Returns:
            List[Path]: Sorted list of noisy map file paths.

        Raises:
            FileNotFoundError: If the noisy directory or maps are not found.
        """
        if not self.noisy_dir.exists():
            logging.error(f"Noisy maps directory not found: {self.noisy_dir}")
            raise FileNotFoundError(f"Noisy maps directory not found: {self.noisy_dir}")

        noisy_map_paths = sorted(self.noisy_dir.glob("*.fits"))
        if not noisy_map_paths:
            logging.error(f"No noisy maps found in {self.noisy_dir}")
            raise FileNotFoundError(f"No noisy maps found in {self.noisy_dir}")

        logging.info(f"Found {len(noisy_map_paths)} noisy maps in {self.noisy_dir}")
        return noisy_map_paths

    def smooth_kappa(self) -> None:
        """
        Smooths all available kappa and noisy maps with each smoothing scale in sl_list.
        """
        all_maps = self.kappa_map_paths + self.noisy_map_paths
        total_maps = len(all_maps)
        for idx, map_path in enumerate(all_maps, start=1):
            logging.info(f"Processing map {idx}/{total_maps}: {map_path.name}")
            info: Dict[str, Any] = InfoExtractor.extract_info_from_path(map_path)
            for sl in self.sl_list:
                self._process_single_map(map_path, info, sl)

    def _process_single_map(
        self,
        map_path: Path,
        info: Dict[str, Any],
        sl: float,
    ) -> None:
        """
        Smooths a single kappa map at a given smoothing scale and saves the output.

        Args:
            map_path (Path): Path to the input kappa map.
            info (Dict[str, Any]): Metadata extracted from the file (e.g., seed, redshift, ngal).
            sl (float): Smoothing scale in arcminutes.
        """
        ngal = info.get("ngal", 0)
        suffix = f"sl{sl}_noiseless" if (ngal == 0) else f"sl{sl}_ngal{ngal}"
        output_filename = f"kappa_smoothed_s{info.get('seed')}_zs{info.get('redshift')}_{suffix}.fits"
        output_file = self.smoothed_dir / output_filename

        if output_file.exists() and not self.overwrite:
            logging.debug(f"Output file exists and overwrite is False: {output_file}")
            return

        try:
            kappa_map = self._load_kappa_map(map_path)
            smoothed_map = self._smooth_map(kappa_map, sl)
            hp.write_map(str(output_file), smoothed_map, dtype=np.float32, overwrite=True)
            logging.info(f"Saved smoothed map to {output_file}")
        except Exception as e:
            logging.error(f"Failed to process {map_path.name} (ngal={ngal}, sl={sl}): {e}")

    def _load_kappa_map(self, map_path: Path) -> np.ndarray:
        """
        Loads and reorders a kappa map from a FITS file.

        Args:
            map_path (Path): Path to the kappa map file.

        Returns:
            np.ndarray: Reordered kappa map.
        """
        logging.debug(f"Loading kappa map from {map_path.name}")
        kappa = hp.read_map(str(map_path))
        kappa_reordered = hp.reorder(kappa, n2r=True)
        return kappa_reordered

    def _load_noisy_map(self, noisy_path: Path) -> np.ndarray:
        """
        Loads and reorders a noisy map from a FITS file.

        Args:
            noisy_path (Path): Path to the noisy map file.

        Returns:
            np.ndarray: Reordered noisy map.
        """
        logging.debug(f"Loading noisy map from {noisy_path.name}")
        noisy = hp.read_map(str(noisy_path))
        noisy_reordered = hp.reorder(noisy, n2r=True)
        return noisy_reordered

    def _smooth_map(self, kappa: np.ndarray, sl_arcmin: float) -> np.ndarray:
        """
        Smooths a kappa map with a Gaussian kernel.

        Args:
            kappa (np.ndarray): The input kappa map.
            sl_arcmin (float): Smoothing scale in arcminutes.

        Returns:
            np.ndarray: The smoothed kappa map.
        """
        sigma_rad = sl_arcmin * self.ARCMIN_TO_RAD
        logging.debug(f"Smoothing map with sigma={sigma_rad:.4f} radians.")
        smoothed = hp.smoothing(kappa, sigma=sigma_rad)
        return smoothed