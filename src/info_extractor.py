
import re
import logging
from pathlib import Path
from typing import Optional, Dict, Union, ClassVar

class InfoExtractor:
    """
    A utility class to extract information such as seed, redshift, and box type from file paths.
    """

    # Precompiled regular expressions for performance
    SEED_PATTERN: ClassVar[re.Pattern] = re.compile(r'_s(\d+)')
    REDSHIFT_PATTERN: ClassVar[re.Pattern] = re.compile(r'_zs(\d+\.\d+)')
    BOX_SIZE_PATTERN: ClassVar[re.Pattern] = re.compile(r'_size(\d+)')
    OA_PATTERN: ClassVar[re.Pattern] = re.compile(r'_oa(\d+)')
    SL_PATTERN: ClassVar[re.Pattern] = re.compile(r'_sl(\d+)')
    NOISE_PATTERN: ClassVar[re.Pattern] = re.compile(r'_ngal(\d+)')

    # Define box size categories
    BBOX_SIZES: ClassVar[set] = {3750, 5000}
    SBOX_SIZES: ClassVar[set] = {625}

    @staticmethod
    def extract_info_from_path(path: Union[str, Path]) -> Dict[str, Optional[Union[int, float, str]]]:
        """
        Extracts seed, redshift, box type, OA, SL, and noise galaxy count information from a path.

        Args:
            path (Union[str, Path]): The path to extract information from.

        Returns:
            Dict[str, Optional[Union[int, float, str]]]: A dictionary containing extracted information.
                - seed (Optional[int]): Seed number or None if not found.
                - redshift (Optional[float]): Redshift value or None if not found.
                - box_type (Optional[str]): Box type ('bigbox' or 'tiled') or None if not found.
                - oa (Optional[int]): OA number or None if not found.
                - sl (Optional[int]): SL number or None if not found.
                - ngal (int): Number of galaxies (0 if 'noiseless' is found, else extracted value or None).
        """
        path_str = str(path)  # Ensure the path is a string for regex operations
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
    def extract_seed_from_path(cls, path: str) -> Optional[int]:
        """
        Extracts the seed number from the path.

        Args:
            path (str): The path to extract the seed from.

        Returns:
            Optional[int]: The seed number if found, else None.
        """
        match = cls.SEED_PATTERN.search(path)
        if match:
            seed = int(match.group(1))
            logging.debug(f"Extracted seed: {seed} from path: {path}")
            return seed
        else:
            #logging.warning(f"Seed number not found in the path: {path}")
            return None

    @classmethod
    def extract_redshift_from_path(cls, path: str) -> Optional[float]:
        """
        Extracts the redshift value from the path.

        Args:
            path (str): The path to extract the redshift from.

        Returns:
            Optional[float]: The redshift value if found, else None.
        """
        match = cls.REDSHIFT_PATTERN.search(path)
        if match:
            redshift = float(match.group(1))
            logging.debug(f"Extracted redshift: {redshift} from path: {path}")
            return redshift
        else:
            #logging.warning(f"Redshift value not found in the path: {path}")
            return None

    @classmethod
    def extract_box_type_from_path(cls, path: str) -> Optional[str]:
        """
        Determines the box type based on the box size in the path.

        Args:
            path (str): The path to extract the box type from.

        Returns:
            Optional[str]: 'bigbox', 'tiled', or None if not determinable.
        """
        match = cls.BOX_SIZE_PATTERN.search(path)
        if match:
            box_size = int(match.group(1))
            if box_size in cls.BBOX_SIZES:
                logging.debug(f"Box size {box_size} classified as 'bigbox' for path: {path}")
                return 'bigbox'
            elif box_size in cls.SBOX_SIZES:
                logging.debug(f"Box size {box_size} classified as 'tiled' for path: {path}")
                return 'tiled'
            else:
                logging.warning(f"Box size {box_size} is not recognized in the path: {path}")
                return None
        else:
            #logging.warning(f"Box size not found in the path: {path}")
            return None
        
    @classmethod
    def extract_oa_from_path(cls, path: str) -> Optional[int]:
        """
        Extracts the opening angle (OA) from the path.

        Args:
            path (str): The path to extract the OA from.

        Returns:
            Optional[int]: The OA if found, else None.
        """
        match = cls.OA_PATTERN.search(path)
        if match:
            oa = int(match.group(1))
            logging.debug(f"Extracted OA: {oa} from path: {path}")
            return oa
        else:
            #logging.warning(f"OA number not found in the path: {path}")
            return None

    @classmethod
    def extract_sl_from_path(cls, path: str) -> Optional[int]:
        """
        Extracts the smoothing length (SL) from the path.

        Args:
            path (str): The path to extract the SL number from.

        Returns:
            Optional[int]: The SL number if found, else None.
        """
        match = cls.SL_PATTERN.search(path)
        if match:
            sl = int(match.group(1))
            logging.debug(f"Extracted SL: {sl} from path: {path}")
            return sl
        else:
            #logging.warning(f"SL number not found in the path: {path}")
            return None

    @classmethod
    def extract_ngal_from_path(cls, path: str) -> int:
        """
        Extracts the number of galaxies (ngal) from the path.
        If 'noiseless' is found in the path, ngal is set to 0.

        Args:
            path (str): The path to extract ngal from.

        Returns:
            int: The number of galaxies. 0 if 'noiseless' is found, else the extracted value or 0 if not found.
        """
        if 'noiseless' in path.lower():
            logging.debug(f"'noiseless' found in path: {path}. Setting ngal to 0.")
            return 0

        match = cls.NOISE_PATTERN.search(path)
        if match:
            ngal = int(match.group(1))
            logging.debug(f"Extracted ngal: {ngal} from path: {path}")
            return ngal
        else:
            #logging.warning(f"Number of galaxies (ngal) not found in the path: {path}. Setting ngal to 0.")
            return 0  # Default to 0 if not found
    