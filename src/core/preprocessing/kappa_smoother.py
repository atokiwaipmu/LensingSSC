import logging
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import healpy as hp

from utils.info_extractor import InfoExtractor

class KappaSmoother:
    """
    A class to smooth kappa maps with specified smoothing scales and noise levels.

    Attributes:
        datadir (Path): The directory containing kappa maps.
        nside (int): The Healpix nside parameter.
        ngal_list (List[int]): List of galaxy numbers for noise generation.
        sl_list (List[float]): List of smoothing scales in arcminutes.
        overwrite (bool): Flag to overwrite existing smoothed maps.
    """

    # Constants for unit conversions
    ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)

    def __init__(
        self,
        datadir: str,
        nside: int,
        sl_list: List[float],
        overwrite: bool = False,
    ):
        """
        Initializes the KappaSmoother with directory paths and parameters.

        Args:
            datadir (str): Path to the data directory.
            nside (int): Healpix nside parameter.
            ngal_list (List[int]): List of galaxy numbers.
            sl_list (List[float]): List of smoothing scales in arcminutes.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
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
        Retrieves and validates the kappa map paths.

        Returns:
            List[Path]: Sorted list of kappa map file paths.

        Raises:
            FileNotFoundError: If kappa directory or maps are not found.
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
        Retrieves and validates the noisy map paths.

        Returns:
            List[Path]: Sorted list of noisy map file paths.

        Raises:
            FileNotFoundError: If noisy directory or maps are not found.
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
        Smooths all kappa maps with specified smoothing scales and noise levels.
        """
        total_maps = len(self.kappa_map_paths + self.noisy_map_paths)
        for idx, kappa_path in enumerate(self.kappa_map_paths + self.noisy_map_paths, start=1):
            logging.info(f"Processing kappa map {idx}/{total_maps}: {kappa_path.name}")
            info = InfoExtractor.extract_info_from_path(kappa_path)
            for sl in self.sl_list:
                self._process_single_map(kappa_path, info, sl)

    def _process_single_map(
        self,
        kappa_path: Path,
        info: Dict[str, any],
        sl: float,
    ) -> None:
        """
        Processes a single kappa map with given parameters.

        Args:
            kappa_path (Path): Path to the kappa map file.
            info (Dict[str, any]): Extracted information from the kappa path.
            ngal (int): Number of galaxies for noise.
            sl (float): Smoothing scale in arcminutes.
        """
        ngal = info["ngal"]
        suffix = f"sl{sl}_noiseless" if (ngal == 0) else f"sl{sl}_ngal{ngal}"
        output_filename = f"kappa_smoothed_s{info['seed']}_zs{info['redshift']}_{suffix}.fits"
        output_file = self.smoothed_dir / output_filename

        if output_file.exists() and not self.overwrite:
            logging.debug(f"Output file exists and overwrite is False: {output_file}")
            return

        try:
            kappa_map = self._load_kappa_map(kappa_path)
            smoothed_map = self._smooth_map(kappa_map, sl)
            hp.write_map(str(output_file), smoothed_map, dtype=np.float32, overwrite=True)
            logging.info(f"Saved smoothed map to {output_file}")
        except Exception as e:
            logging.error(f"Failed to process {kappa_path.name} with ngal={ngal}, sl={sl}: {e}")

    def _load_kappa_map(self, kappa_path: Path) -> np.ndarray:
        """
        Loads and reorders a kappa map.

        Args:
            kappa_path (Path): Path to the kappa map file.

        Returns:
            np.ndarray: Reordered kappa map.
        """
        logging.debug(f"Loading kappa map from {kappa_path.name}")
        kappa = hp.read_map(str(kappa_path))
        kappa_reordered = hp.reorder(kappa, n2r=True)
        return kappa_reordered
    
    def _load_noisy_map(self, noisy_path: Path) -> np.ndarray:
        """
        Loads and reorders a noisy map.

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
        Smooths the kappa map with a Gaussian kernel.

        Args:
            kappa (np.ndarray): Kappa map to smooth.
            sl_arcmin (float): Smoothing scale in arcminutes.

        Returns:
            np.ndarray: Smoothed kappa map.
        """
        sigma_rad = sl_arcmin * self.ARCMIN_TO_RAD
        logging.debug(f"Smoothing map with sigma={sigma_rad} radians.")
        smoothed = hp.smoothing(kappa, sigma=sigma_rad)
        return smoothed
    
if __name__ == "__main__":
    from utils.utils import parse_arguments, load_config, filter_config, find_data_dirs
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    config = load_config(args.config_file)

    filtered_config_pp = filter_config(config, KappaSmoother)
    ks = KappaSmoother(args.datadir, **filtered_config_pp, overwrite=True)
    ks.smooth_kappa()

    #data_dirs = find_data_dirs()

    #for datadir in data_dirs:
    #    ks = KappaSmoother(datadir, **filtered_config_pp, overwrite=args.overwrite)
    #    ks.smooth_kappa()