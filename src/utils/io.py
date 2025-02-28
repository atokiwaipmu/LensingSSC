import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (
    ClassVar, Dict, Optional, Pattern, Union,
    List, Tuple, Set, Any
)


# =============================================================================
# Path handling configurations
# =============================================================================
@dataclass(frozen=True)
class PathConfig:
    """
    Holds default configuration values for path and file handling.

    Attributes
    ----------
    DEFAULT_WORKDIR : Path
        Base working directory for projects.
    DATA_SUBPATH : str
        Subpath pattern within the work directory to locate data directories.
    KAPPA_DIR : str
        Name of the sub-directory containing kappa files.
    FITS_PATTERN : str
        File pattern to match FITS files.
    MAX_WORKERS : int
        Maximum number of workers for parallel file search.
    """
    DEFAULT_WORKDIR: Path = Path("/lustre/work/akira.tokiwa/Projects/LensingSSC/")
    DATA_SUBPATH: str = "data/*/*/usmesh"
    KAPPA_DIR: str = "kappa"
    FITS_PATTERN: str = "*.fits"
    MAX_WORKERS: int = 4


@dataclass
class PathPatterns:
    """
    Regular expression patterns for extracting information from paths.

    Attributes
    ----------
    SEED : str
        Pattern to capture the seed (e.g., '_s1234').
    REDSHIFT : str
        Pattern to capture the redshift (e.g., '_zs1.5').
    BOX_SIZE : str
        Pattern to capture the box size (e.g., '_size5000').
    OA : str
        Pattern to capture the opening angle (e.g., '_oa45').
    SL : str
        Pattern to capture the smoothing length (e.g., '_sl10').
    NOISE : str
        Pattern to capture the galaxy count or noise settings (e.g., '_ngal30').
    """
    SEED: str = r"_s(\d+)"
    REDSHIFT: str = r"_zs(\d+\.\d+)"
    BOX_SIZE: str = r"_size(\d+)"
    OA: str = r"_oa(\d+)"
    SL: str = r"_sl(\d+)"
    NOISE: str = r"_ngal(\d+)"


@dataclass
class BoxSizes:
    """
    Valid sets of box sizes for distinguishing 'bigbox' from 'tiled'.

    Attributes
    ----------
    BIG_BOX : set of int
        Set of valid sizes considered as large box simulations.
    SMALL_BOX : set of int
        Set of valid sizes considered as tiled or smaller box simulations.
    """
    BIG_BOX: Set[int] = frozenset({3750, 5000})
    SMALL_BOX: Set[int] = frozenset({625})


class InfoExtractor:
    """
    Provides methods to extract simulation parameters from file paths.

    Attributes
    ----------
    PATTERNS : Dict[str, Pattern]
        Precompiled regex patterns derived from `PathPatterns`.
    """
    PATTERNS: ClassVar[Dict[str, Pattern]] = {
        name: re.compile(pattern)
        for name, pattern in PathPatterns.__dict__.items()
        if not name.startswith("_")
    }

    @staticmethod
    def extract_info_from_path(path: Union[str, Path]) -> Dict[str, Optional[Union[int, float, str]]]:
        """
        Extract multiple pieces of metadata from a path string or Path object.

        Parameters
        ----------
        path : Union[str, Path]
            The file or directory path to analyze.

        Returns
        -------
        Dict[str, Optional[Union[int, float, str]]]
            Dictionary containing metadata fields such as 'seed', 'redshift', 
            'box_type', 'oa', 'sl', and 'ngal'.
        """
        path_str = str(path)
        return {
            "seed": InfoExtractor.extract_seed_from_path(path_str),
            "redshift": InfoExtractor.extract_redshift_from_path(path_str),
            "box_type": InfoExtractor.extract_box_type_from_path(path_str),
            "oa": InfoExtractor.extract_oa_from_path(path_str),
            "sl": InfoExtractor.extract_sl_from_path(path_str),
            "ngal": InfoExtractor.extract_ngal_from_path(path_str),
        }

    @classmethod
    def _extract_value(
        cls,
        path: str,
        pattern_key: str,
        value_type: type,
        default: Optional[Union[int, float]] = None
    ) -> Optional[Union[int, float]]:
        """
        Generic extraction of numeric values from a path using regex.

        Parameters
        ----------
        path : str
            The path string to search.
        pattern_key : str
            Key for the regex pattern in cls.PATTERNS.
        value_type : type
            Desired Python type for the captured value.
        default : Optional[Union[int, float]]
            Default value if extraction fails or the pattern is not found.

        Returns
        -------
        Optional[Union[int, float]]
            Extracted value in the desired type, or the default if not found.
        """
        match = cls.PATTERNS[pattern_key].search(path)
        if not match:
            return default

        try:
            value = value_type(match.group(1))
            logging.debug(f"Extracted '{pattern_key}' -> {value} from path: {path}")
            return value
        except ValueError as err:
            logging.error(f"Failed to convert '{pattern_key}' value: {err}")
            return default

    @classmethod
    def extract_seed_from_path(cls, path: str) -> Optional[int]:
        """
        Extract the random seed from the path.

        Parameters
        ----------
        path : str
            The path to parse.

        Returns
        -------
        Optional[int]
            Seed value if present, otherwise None.
        """
        return cls._extract_value(path, "SEED", int)

    @classmethod
    def extract_redshift_from_path(cls, path: str) -> Optional[float]:
        """
        Extract the redshift value from the path.

        Parameters
        ----------
        path : str
            The path to parse.

        Returns
        -------
        Optional[float]
            Redshift value if present, otherwise None.
        """
        return cls._extract_value(path, "REDSHIFT", float)

    @classmethod
    def extract_box_type_from_path(cls, path: str) -> Optional[str]:
        """
        Determine whether the path corresponds to a 'bigbox' or 'tiled' simulation,
        based on the box size.

        Parameters
        ----------
        path : str
            The path to parse.

        Returns
        -------
        Optional[str]
            'bigbox' if box size is in BoxSizes.BIG_BOX,
            'tiled' if box size is in BoxSizes.SMALL_BOX,
            otherwise None.
        """
        size = cls._extract_value(path, "BOX_SIZE", int)
        if not size:
            return None
        if size in BoxSizes.BIG_BOX:
            return "bigbox"
        if size in BoxSizes.SMALL_BOX:
            return "tiled"
        return None

    @classmethod
    def extract_oa_from_path(cls, path: str) -> Optional[int]:
        """
        Extract the opening angle (oa) from the path.

        Parameters
        ----------
        path : str
            The path to parse.

        Returns
        -------
        Optional[int]
            Opening angle value if found, otherwise None.
        """
        return cls._extract_value(path, "OA", int)

    @classmethod
    def extract_sl_from_path(cls, path: str) -> Optional[int]:
        """
        Extract the smoothing length (sl) from the path.

        Parameters
        ----------
        path : str
            The path to parse.

        Returns
        -------
        Optional[int]
            Smoothing length value if found, otherwise None.
        """
        return cls._extract_value(path, "SL", int)

    @classmethod
    def extract_ngal_from_path(cls, path: str) -> int:
        """
        Extract the galaxy count (ngal) from the path or return 0 for noiseless paths.

        Parameters
        ----------
        path : str
            The path to parse.

        Returns
        -------
        int
            The extracted galaxy count, or 0 if 'noiseless' is found in the path.
        """
        if "noiseless" in path.lower():
            return 0
        return cls._extract_value(path, "NOISE", int, default=0) or 0


class PathHandler:
    """
    Handles file system operations, directory management, and filtering logic
    for paths related to lensing or simulation data.

    Attributes
    ----------
    config : PathConfig
        Configuration object containing path-related defaults.
    """

    def __init__(self, config: PathConfig = PathConfig()):
        """
        Initialize the PathHandler with a given configuration.

        Parameters
        ----------
        config : PathConfig, optional
            A PathConfig instance with desired path defaults, by default PathConfig().
        """
        self.config = config
        self._validate_base_paths()

    def _validate_base_paths(self) -> None:
        """
        Validate that the base working directory exists.

        Raises
        ------
        FileNotFoundError
            If the default working directory does not exist on the filesystem.
        """
        if not self.config.DEFAULT_WORKDIR.exists():
            raise FileNotFoundError(
                f"Work directory not found: {self.config.DEFAULT_WORKDIR}"
            )

    @lru_cache(maxsize=32)
    def find_data_dirs(self, workdir: Optional[Path] = None) -> List[Path]:
        """
        Find data directories by globbing for the configured DATA_SUBPATH.

        This method is memoized for performance (caches results for up to 32 unique arguments).

        Parameters
        ----------
        workdir : Optional[Path]
            The base directory in which to search for data.
            If None, uses self.config.DEFAULT_WORKDIR.

        Returns
        -------
        List[Path]
            A sorted list of paths containing data directories.
        """
        base_dir = workdir or self.config.DEFAULT_WORKDIR
        search_path = base_dir / self.config.DATA_SUBPATH
        logging.debug(f"Searching for data directories in: {search_path}")

        # For each matched path, return its parent directory
        matched_paths = sorted(search_path.parent.glob(self.config.DATA_SUBPATH))
        data_dirs = [p.parent for p in matched_paths if p.is_dir()]

        logging.info(f"Found {len(data_dirs)} data directories.")
        return data_dirs

    def find_kappa_files(self, datadir: Path) -> List[Path]:
        """
        Find kappa FITS files within a specified directory, using parallel processing.

        Parameters
        ----------
        datadir : Path
            Directory in which to look for kappa files.

        Returns
        -------
        List[Path]
            A list of paths to kappa FITS files in the directory.
            Returns an empty list if the directory does not exist.
        """
        kappa_dir = datadir / self.config.KAPPA_DIR
        if not kappa_dir.exists():
            logging.warning(f"Kappa directory not found in {datadir}.")
            return []

        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            # We only have one directory here, so we wrap it in a list to map
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

        Parameters
        ----------
        data_dirs : List[Path]
            List of directory paths containing simulation data.

        Returns
        -------
        Tuple[List[Path], List[Path]]
            A tuple of (tiled_dirs, bigbox_dirs).
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

        Specifically, matches on seed, redshift, and noise (ngal or noiseless).

        Parameters
        ----------
        paths : List[Path]
            The list of potential paths to filter.
        reference_path : Path
            A path whose metadata is used as the filter criteria.

        Returns
        -------
        List[Path]
            Subset of 'paths' that match the metadata derived from 'reference_path'.
        """
        info = InfoExtractor.extract_info_from_path(reference_path)
        # Build filter tokens from the extracted metadata
        tokens = {
            "redshift": f"zs{info.get('redshift')}",
            "seed": f"s{info.get('seed')}",
            "noise": "noiseless" if info.get("ngal") == 0 else f"ngal{info.get('ngal')}"
        }

        # Keep only paths whose filenames contain all tokens
        filtered_list = [
            path for path in paths
            if all(token in path.name for token in tokens.values())
        ]
        logging.debug(
            f"Filtered {len(filtered_list)} out of {len(paths)} paths based on reference metadata: {tokens}."
        )

        return filtered_list