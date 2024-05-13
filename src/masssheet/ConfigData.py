
import json
import os
from nbodykit.lab import BigFileCatalog
from dataclasses import dataclass, field

@dataclass
class ConfigData:
    zs_list: list
    zlmin: float
    zlmax: float
    datadir: str
    source: str
    destination: str
    dataset: str
    cat: 'BigFileCatalog' = field(repr=False, init=False)
    npix: int = field(init=False)
    nside: int = field(init=False)
    nbar: float = field(init=False)
    rhobar: float = field(init=False)
    Om: float = field(init=False)

    def __post_init__(self):
        path = os.path.join(self.datadir, self.source)
        self.cat = BigFileCatalog(path, dataset=self.dataset)
        self.npix = self.cat.attrs['healpix.npix'][0]
        self.nside = self.cat.attrs['healpix.nside'][0]
        self.nbar = (self.cat.attrs['NC'] ** 3 / self.cat.attrs['BoxSize'] ** 3 * self.cat.attrs['ParticleFraction'])[0]
        self.rhobar = self.cat.attrs['MassTable'][1] * self.nbar
        self.Om = self.cat.attrs['OmegaM'][0]
        self.aemitIndex = self.cat.attrs['aemitIndex.edges']

    @staticmethod
    def from_json(config_file: str) -> 'ConfigData':
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)
        except FileNotFoundError:
            raise Exception(f"Configuration file {config_file} not found.")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON in configuration file.")

        return ConfigData(
            zs_list=config['zs'],
            zlmin=config['zlmin'],
            zlmax=config['zlmax'],
            datadir=config['datadir'],
            source=config['source'],
            destination=config['destination'],
            dataset=config['dataset']
        )