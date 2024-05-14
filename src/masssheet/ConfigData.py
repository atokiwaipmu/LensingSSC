
import json
import os
from nbodykit.lab import BigFileCatalog
from dataclasses import dataclass, field
from typing import List

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

@dataclass
class ConfigData:
    zs_list: List[float]
    zlmin: float
    zlmax: float
    datadir: str
    source: str
    destination: str
    dataset: str

    @staticmethod
    def from_json(config_file: str) -> 'ConfigData':
        """Load configuration data from a JSON file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} not found.")
        
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in configuration file.")

        required_keys = ['zs', 'zlmin', 'zlmax', 'datadir', 'source', 'destination', 'dataset']
        if not all(key in config for key in required_keys):
            missing_keys = [key for key in required_keys if key not in config]
            raise KeyError(f"Missing required keys in configuration file: {missing_keys}")

        return ConfigData(
            zs_list=config['zs'],
            zlmin=config['zlmin'],
            zlmax=config['zlmax'],
            datadir=config['datadir'],
            source=config['source'],
            destination=config['destination'],
            dataset=config['dataset']
        )