import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional

from lensing_ssc.utils.constants import PathConfig
from lensing_ssc.utils.extractors import InfoExtractor

class PathHandler:
    """
    Handles file system operations, directory management, and filtering logic
    for paths related to lensing or simulation data.
    """

    def __init__(self, config: PathConfig = PathConfig()):
        """
        Initialize the PathHandler with a given configuration.
        """
        self.config = config
        self._validate_base_paths()

    def _validate_base_paths(self) -> None:
        """
        Validate that the base working directory exists.
        """
        if not self.config.DEFAULT_WORKDIR.exists():
            raise FileNotFoundError(
                f"Work directory not found: {self.config.DEFAULT_WORKDIR}"
            )

    @lru_cache(maxsize=32)
    def find_data_dirs(self, workdir: Optional[Path] = None) -> List[Path]:
        """
        Find data directories by globbing for the configured DATA_SUBPATH.
        """
        base_dir = workdir or self.config.DEFAULT_WORKDIR
        search_path = base_dir / self.config.DATA_SUBPATH
        logging.debug(f"Searching for data directories in: {search_path}")

        matched_paths = sorted(search_path.parent.glob(self.config.DATA_SUBPATH))
        data_dirs = [p.parent for p in matched_paths if p.is_dir()]

        logging.info(f"Found {len(data_dirs)} data directories.")
        return data_dirs

    def find_kappa_files(self, datadir: Path) -> List[Path]:
        """
        Find kappa FITS files within a specified directory, using parallel processing.
        """
        kappa_dir = datadir / self.config.KAPPA_DIR
        if not kappa_dir.exists():
            logging.warning(f"Kappa directory not found in {datadir}.")
            return []

        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            results = list(executor.map(
                lambda d: list(d.glob(self.config.FITS_PATTERN)),
                [kappa_dir]
            ))
        
        kappa_files = results[0] if results else []
        logging.debug(f"Found {len(kappa_files)} kappa files in {kappa_dir}.")
        return kappa_files

    def categorize_data_dirs_by_box_type(self, data_dirs: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Categorize the provided data directories into 'tiled' and 'bigbox' based on their box size.
        """
        tiled_dirs, bigbox_dirs = [], []

        for dir_path in data_dirs:
            box_type = InfoExtractor.extract_info_from_path(dir_path).get("box_type")
            if box_type == "tiled":
                tiled_dirs.append(dir_path)
            elif box_type == "bigbox":
                bigbox_dirs.append(dir_path)
            else:
                logging.warning(f"Unknown box type for: {dir_path}")

        return tiled_dirs, bigbox_dirs

    def filter_paths_by_metadata(self, paths: List[Path], reference_path: Path) -> List[Path]:
        """
        Filter a list of paths to only those matching metadata from a 'reference' path.
        """
        info = InfoExtractor.extract_info_from_path(reference_path)
        # Build filter tokens from the extracted metadata
        tokens = {
            "redshift": f"zs{info.get('redshift')}",
            "seed": f"s{info.get('seed')}",
            "noise": "noiseless" if info.get("ngal") == 0 else f"ngal{info.get('ngal')}"
        }

        filtered_list = [
            path for path in paths
            if all(token in path.name for token in tokens.values())
        ]
        logging.debug(
            f"Filtered {len(filtered_list)} out of {len(paths)} paths based on reference metadata: {tokens}."
        )

        return filtered_list