
import re
import logging

class InfoExtractor:
    @staticmethod
    def extract_info_from_path(path):
        """Extracts seed, redshift, and box size information from a path.

        Args:
            path (str): The path to extract information from.

        Returns:
            dict[str, Optional[str]]: A dictionary containing extracted information.
                - seed (Optional[int]): Seed number or None if not found.
                - redshift (Optional[float]): Redshift value or None if not found.
                - box_type (Optional[str]): Box type ('bigbox' or 'tiled') or None if not found.
        """

        seed = InfoExtractor.extract_seed_from_path(path)
        redshift = InfoExtractor.extract_redshift_from_path(path)
        box_type = InfoExtractor.extract_type_from_path(path)

        return {"seed": seed, "redshift": redshift, "box_type": box_type}

    @staticmethod
    def extract_seed_from_path(path):
        # Regular expression to find the pattern '_s{seed_number}_'
        match = re.search(r'_s(\d+)_', path)
        if match:
            return int(match.group(1))
        else:
            logging.error("Seed number not found in the given path.")
            return None
        
    @staticmethod
    def extract_redshift_from_path(path):
        # Regular expression to find the pattern '_zs{redshift}' e.g. '_zs2.0'
        match = re.search(r'_zs(\d+\.\d+)', path)
        if match:
            return float(match.group(1))
        else:
            #logging.error("Redshift value not found in the given path.")
            return None
        
    @staticmethod
    def extract_type_from_path(path, bbox_size=[3750, 5000], sbox_size=[625]):
        # Regular expression to find the pattern '_size{box_size}_'
        match = re.search(r'_size(\d+)_', path)
        if match:
            box_size = int(match.group(1))
            if box_size in bbox_size:
                return 'bigbox'
            elif box_size in sbox_size:
                return 'tiled'
            else:
                logging.error("Box size is not valid.")
                return None
        else:
            logging.error("Box size not found in the given path.")
            return None