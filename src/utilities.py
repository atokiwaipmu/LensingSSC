
import os
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
    

def extract_seed_from_directory(datadir):
    """
    Extracts the seed value from the directory name.

    Parameters:
    - datadir (str): The data directory path.

    Returns:
    - seed (str): Extracted seed value.
    """
    base_name = os.path.basename(datadir)
    seed_parts = base_name.split('_')
    if "rfof" in seed_parts:
        return seed_parts[-10][1:]
    return seed_parts[4][1:]