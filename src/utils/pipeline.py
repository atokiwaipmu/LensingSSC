
import os
import json
import numpy as np
from nbodykit.lab import BigFileCatalog
from nbodykit.cosmology import Planck15
import healpy as healpix

class CosmicShearPipeline:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.dst = self.config['destination']
        self.cat = None
        self.z = None
        self.zs_list = self.config['zs']
        self.ds_list = None
        self.nsources = len(self.zs_list)
        self.nside = self.config['nside']
        self.npix = None
        self.nbar = None
        self.inititalize()
        self.Om = self.cat.attrs['OmegaM'][0]

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def create_output_directory(self):
        if not os.path.exists(self.dst):
            os.makedirs(self.dst)

    def initialize_catalog(self):
        path = self.config['source']
        dataset = self.config['dataset']
        self.cat = BigFileCatalog(path, dataset=dataset)

    def calculate_redshift_range(self):
        zlmin = self.config['zlmin']
        zlmax = self.config.get('zlmax', None)
        zstep = self.config['zstep']
        if zlmax is None:
            zlmax = max(self.zs_list)
        Nsteps = int(np.round((zlmax - zlmin) / zstep))
        if Nsteps < 2:
            Nsteps = 2
        self.z = np.linspace(zlmax, zlmin, Nsteps, endpoint=True)

    def calculate_comoving_distance(self):
        self.ds_list = Planck15.comoving_distance(self.zs_list)

    def initialize_healpix(self):
        self.npix = healpix.nside2npix(self.nside)

    def calculate_number_density(self):
        self.nbar = (self.cat.attrs['NC'] ** 3 / self.cat.attrs['BoxSize'] ** 3 * self.cat.attrs['ParticleFraction'])[0]

    def inititalize(self):
        self.create_output_directory()
        self.initialize_catalog()
        self.calculate_redshift_range()
        self.calculate_comoving_distance()
        self.initialize_healpix()
        self.calculate_number_density()