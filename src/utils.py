
import re
from astropy.cosmology import FlatLambdaCDM

class CosmologySettings:
    def __init__(self, h=0.6774, om=0.309):
        """
        Initialize the cosmology with specified Hubble constant (h) and matter density (om).
        Default values correspond to the Planck15 cosmology.
        """
        self.h = h
        self.om = om
        self.cosmo = FlatLambdaCDM(H0=self.h * 100, Om0=self.om)

    def get_cosmology(self):
        """
        Return the cosmology object for further use in distance calculations.
        """
        return self.cosmo

def extract_seed_from_path(path):
    """
    Extract the seed number from a file path.

    Parameters:
    - path (str): The file path that includes the seed number in the format '_s{seed_number}_'.

    Returns:
    - seed_number (int): The seed number extracted from the path
    """
    # Regular expression to find the pattern '_s{seed_number}_'
    match = re.search(r'_s(\d+)_', path)
    
    if match:
        return int(match.group(1))
    else:
        print("Seed number not found in the given path.")
        return None
    
def extract_redshift_from_path(path):
    """
    Extract the redshift value from a file path.

    Parameters:
    - path (str): The file path that includes the redshift value in the format '_zs{redshift}'.

    Returns:
    - redshift (float): The redshift value extracted from the path
    """
    # Regular expression to find the pattern '_zs{redshift}' e.g. '_zs2.0'
    match = re.search(r'_zs(\d+\.\d+)', path)

    if match:
        return float(match.group(1))
    else:
        print("Redshift value not found in the given path.")
        return "UnknownRedshift"