
import os
import json
from dataclasses import dataclass, field
from nbodykit.lab import BigFileCatalog
from typing import List, Type, TypeVar, Union


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
        else:
            missing_keys = required_keys_set - config_keys_set
            extra_keys = config_keys_set - required_keys_set
            print(f"Missing keys: {missing_keys}")
            print(f"Extra keys: {extra_keys}")
            if missing_keys:
                raise KeyError(f"Missing required keys in configuration file: {missing_keys}")

        return cls(**config)

@dataclass
class ConfigData(BaseConfig):
    zs_list: List[float]
    zlmin: float
    zlmax: float
    zstep: float
    tileddir: str
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
    nside: int
    void_val: float
    lmax: int

    @staticmethod
    def from_json(config_file: str) -> 'ConfigAnalysis':
        required_keys = ['resultsdir', 'imgdir', 'sl_arcmin', 'nside', 'void_val', 'lmax']
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

@dataclass
class CatalogHandler:
    datadir: str
    source: str
    dataset: str
    cat: 'BigFileCatalog' = field(init=False, repr=False)
    npix: int = field(init=False)
    nside: int = field(init=False)
    nbar: float = field(init=False)
    rhobar: float = field(init=False)
    Om: float = field(init=False)

    def __post_init__(self):
        """Initialize the BigFileCatalog and extract its attributes."""
        path = os.path.join(self.datadir, self.source)
        self.cat = BigFileCatalog(path, dataset=self.dataset)
        self.npix = self.cat.attrs['healpix.npix'][0]
        self.nside = self.cat.attrs['healpix.nside'][0]
        self.nbar = (self.cat.attrs['NC'] ** 3 / self.cat.attrs['BoxSize'] ** 3 * self.cat.attrs['ParticleFraction'])[0]
        self.rhobar = self.cat.attrs['MassTable'][1] * self.nbar
        self.Om = self.cat.attrs['OmegaM'][0]
        self.aemitIndex = self.cat.attrs['aemitIndex.edges']