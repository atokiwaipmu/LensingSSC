import logging
import re
from pathlib import Path
from typing import Optional, Dict, Union, Callable, Any

from lensing_ssc.utils.constants import PathPatterns, BoxSizes

class InfoExtractor:
    """
    A utility class to extract information from file paths based on predefined patterns.
    """

    @staticmethod
    def extract_info_from_path(path: Union[str, Path]) -> Dict[str, Optional[Union[int, float, str]]]:
        """
        Extracts seed, redshift, box type, OA, SL, and noise galaxy count information from a path.

        Args:
            path (Union[str, Path]): The path to extract information from.

        Returns:
            Dict[str, Optional[Union[int, float, str]]]: A dictionary containing extracted information.
        """
        path_str = str(path)
        seed = InfoExtractor.extract_seed_from_path(path_str)
        redshift = InfoExtractor.extract_redshift_from_path(path_str)
        box_type = InfoExtractor.extract_box_type_from_path(path_str)
        oa = InfoExtractor.extract_oa_from_path(path_str)
        sl = InfoExtractor.extract_sl_from_path(path_str)
        ngal = InfoExtractor.extract_ngal_from_path(path_str)

        return {
            "seed": seed,
            "redshift": redshift,
            "box_type": box_type,
            "oa": oa,
            "sl": sl,
            "ngal": ngal
        }

    @classmethod
    def _extract_value(cls, pattern: re.Pattern, path: str, cast: Callable[[str], Any], 
                       value_name: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Helper method to extract a value from the path using a regex pattern.
        """
        match = pattern.search(path)
        if match:
            value = cast(match.group(1))
            logging.debug(f"Extracted {value_name}: {value} from path: {path}")
            return value
        else:
            return default

    @classmethod
    def extract_seed_from_path(cls, path: str) -> Optional[int]:
        return cls._extract_value(PathPatterns.SEED, path, int, "seed")

    @classmethod
    def extract_redshift_from_path(cls, path: str) -> Optional[float]:
        return cls._extract_value(PathPatterns.REDSHIFT, path, float, "redshift")

    @classmethod
    def extract_box_type_from_path(cls, path: str) -> Optional[str]:
        box_size = cls._extract_value(PathPatterns.BOX_SIZE, path, int, "box size")
        if box_size is not None:
            if box_size in BoxSizes.BIG_BOX:
                logging.debug(f"Box size {box_size} classified as 'bigbox' for path: {path}")
                return 'bigbox'
            elif box_size in BoxSizes.SMALL_BOX:
                logging.debug(f"Box size {box_size} classified as 'tiled' for path: {path}")
                return 'tiled'
            else:
                logging.warning(f"Box size {box_size} is not recognized in the path: {path}")
                return None
        return None

    @classmethod
    def extract_oa_from_path(cls, path: str) -> Optional[int]:
        return cls._extract_value(PathPatterns.OA, path, int, "OA")

    @classmethod
    def extract_sl_from_path(cls, path: str) -> Optional[int]:
        return cls._extract_value(PathPatterns.SL, path, int, "SL")

    @classmethod
    def extract_ngal_from_path(cls, path: str) -> int:
        """
        Extracts the number of galaxies (ngal) from the path.
        """
        if 'noiseless' in path.lower():
            logging.debug(f"'noiseless' found in path: {path}. Setting ngal to 0.")
            return 0
        return cls._extract_value(PathPatterns.NOISE, path, int, "ngal", default=0)