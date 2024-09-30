
import os
import numpy as np
import healpy as hp
import glob
import logging

from src.info_extractor import InfoExtractor
from src.noise_generator import NoiseGenerator

class KappaSmoother:
    def __init__(self, datadir, nside, ngal_list, sl_list, overwrite=False):
        self.datadir = datadir
        self._check_ifkappa_dir_exists()

        self.nside = nside
        self.smoothed_dir = os.path.join(self.datadir, 'smoothed_maps')
        os.makedirs(self.smoothed_dir, exist_ok=True)

        self.ngal_list = ngal_list
        self.sl_list = sl_list
        self.overwrite = overwrite

        self.ng = NoiseGenerator(nside=self.nside)

    def _check_ifkappa_dir_exists(self):
        self.kappa_dir = os.path.join(self.datadir, 'kappa')
        if not os.path.exists(self.kappa_dir):
            logging.error(f"Kappa maps directory not found in {self.kappa_dir}")
            raise FileNotFoundError(f"Kappa maps directory not found in {self.kappa_dir}")
        
        self.kappa_map_paths = sorted(glob.glob(os.path.join(self.kappa_dir, '*.fits')))
        if not self.kappa_map_paths:
            logging.error(f"No kappa maps found in {self.kappa_dir}")
            raise FileNotFoundError(f"No kappa maps found in {self.kappa_dir}")
        logging.info(f"Found {len(self.kappa_map_paths)} kappa maps")

    def smooth_kappa(self):
        for idx in range(len(self.kappa_map_paths)):
            logging.info(f"Smoothing {idx+1}/{len(self.kappa_map_paths)} kappa maps")
            self._reset_variables()
            self.kappa_path = self.kappa_map_paths[idx]

            info = InfoExtractor.extract_info_from_path(self.kappa_path)
            output_path = os.path.join(self.smoothed_dir, f"kappa_smoothed_s{info['seed']}_zs{info['redshift']}_*.fits")

            for ngal in self.ngal_list:
                for sl in self.sl_list:
                    self._reset_params(sl=sl, ngal=ngal)
                    if self.noiseless:
                        output_file = output_path.replace('*', f"sl{sl}_noiseless")
                    else:
                        output_file = output_path.replace('*', f"sl{sl}_ngal{ngal}") 
                    
                    if not os.path.exists(output_file) or self.overwrite:
                        self._check_kappa_existance()
                        logging.info(f"Smoothing kappa map with {sl}"+r"$^\circ$")
                        smoothed_map = hp.smoothing(self.kappa_noisy, sigma=sl / 60 * np.pi / 180)
                        hp.write_map(output_file, smoothed_map.astype(np.float32))

    def _reset_variables(self):
        self.kappa = None
        self.kappa_noisy = None

    def _reset_params(self, sl=2, ngal=None):
        logging.info(f"Setting parameters: Smoothing Scale {sl}, Number of Galaxies {ngal}")
        if ngal==0:
            self.noiseless = True
        else:
            self.ng.ngal = ngal
            self.noiseless = False

    def _check_kappa_existance(self):
        if self.kappa is None: self._load_kappa()
        if self.kappa_noisy is None: 
            info = InfoExtractor.extract_info_from_path(self.kappa_path)
            self._add_noise(seed=info["seed"])

    def _load_kappa(self):
        logging.info(f"Loading kappa map from {os.path.basename(self.kappa_path)}")
        self.kappa = hp.read_map(self.kappa_path)
        self.kappa = hp.reorder(self.kappa, n2r=True)

    def _add_noise(self, seed=np.random.randint(0, 2**32)):
        if self.noiseless: 
            self.kappa_noisy = self.kappa.copy()
        else:
            self.kappa_noisy = self.ng.add_noise(input_map=self.kappa, seed=seed)