
import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Type, TypeVar
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
        return "UnknownSeed"

T = TypeVar('T', bound='BaseConfig')

@dataclass
class BaseConfig:
    @staticmethod
    def from_json(config_file: str, required_keys: List[str], cls: Type[T]) -> T:
        """Load configuration data from a JSON file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} not found.")
        
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in configuration file.")

        required_keys_set = set(required_keys)
        config_keys_set = set(config.keys())

        if config_keys_set == required_keys_set:
            pass
        elif config_keys_set < required_keys_set:
            missing_keys = required_keys_set - config_keys_set
            raise KeyError(f"Missing required keys in configuration file: {missing_keys}")
        elif config_keys_set > required_keys_set:
            extra_keys = config_keys_set - required_keys_set
            print(f"Extra keys: {extra_keys}")
            for key in extra_keys:
                config.pop(key)
            print(f"Removed extra keys: {config.keys()}")

        return cls(**config)

@dataclass
class ConfigData(BaseConfig):
    zs_list: List[float]
    datadir: str
    bigboxdir: str
    source: str
    destination: str
    dataset: str

    @staticmethod
    def from_json(config_file: str) -> 'ConfigData':
        required_keys = ['zs_list', 'zlmin', 'zlmax', 'zstep','tileddir', 'bigboxdir', 'source', 'destination', 'dataset']
        return BaseConfig.from_json(config_file, required_keys, ConfigData)

@dataclass
class ConfigAnalysis(BaseConfig):
    resultsdir: str
    imgdir: str
    sl_arcmin: List[int]
    n_gal: List[int]
    nside: int
    void_val: float
    lmin: int
    lmax: int

    @staticmethod
    def from_json(config_file: str) -> 'ConfigAnalysis':
        required_keys = ['resultsdir', 'imgdir', 'sl_arcmin', 'n_gal', 'nside', 'void_val', 'lmin', 'lmax']
        return BaseConfig.from_json(config_file, required_keys, ConfigAnalysis)

@dataclass
class ConfigCosmo(BaseConfig):
    ombh2: float
    omch2: float
    A_s: float
    h: float
    n_s: float
    tau: float
    OmegaB: float = field(init=False)
    OmegaM: float = field(init=False)

    def __post_init__(self):
        self.OmegaB = self.ombh2 / self.h ** 2
        self.OmegaM = self.omch2 / self.h ** 2

    @staticmethod
    def from_json(config_file: str) -> 'ConfigCosmo':
        required_keys = ['ombh2', 'omch2', 'A_s', 'h', 'n_s', 'tau']
        return BaseConfig.from_json(config_file, required_keys, ConfigCosmo)
    