

import argparse
import yaml
import logging
import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from src.analysis import NoiseGenerator, SuffixGenerator, KappaProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PreKappaProcessor(KappaProcessor):
    def __init__(self, datadir, nside, ngal_list, sl_list, args):
        super().__init__(datadir)
        self.img_dir = os.path.join(self.datadir, "img")
        os.makedirs(self.img_dir, exist_ok=True)

        self.vmin = -0.06
        self.vmax = 0.06
        self.ng = NoiseGenerator(nside=nside)
        self.ngal_list = ngal_list
        self.sl_list = sl_list

        self.overwrite_map = args.overwrite_map
        self.overwrite_img = args.overwrite_img
        self.overwrite_cls = args.overwrite_cls

    def preprocess(self):
        for idx in range(len(self.kappa_map_paths)):
            logging.info(f"Smoothing {idx+1}/{len(self.kappa_map_paths)} kappa maps")
            self._reset_variables()
            self.kappa_path = self.kappa_map_paths[idx]

            for ngal in self.ngal_list:
                for sl in self.sl_list:
                    self._reset_params(sl=sl, ngal=ngal)
                    self._generate_outputs()

                    if not os.path.exists(self.outputs["smoothed"]) or self.overwrite_map:
                        self._check_kappa_existance()
                        logging.info(f"Smoothing kappa map with {self.ks.scale_angle}"+r"$^\circ$")
                        self.smoothed_map = hp.smoothing(self.kappa_noisy, sigma=self.ks.scale_angle / 60 * np.pi / 180)
                        hp.write_map(self.outputs["smoothed"], self.smoothed_map.astype(np.float32))
                        self._plot_maps()

                    if not os.path.exists(self.outputs["cls"]) or self.overwrite_cls:
                        self._check_kappa_existance()
                        logging.info(f"Calculating power spectrum")
                        cl = hp.anafast(self.kappa_noisy, lmax=self.fsa.lmax)
                        hp.write_cl(self.outputs["cls"], cl)

    def _reset_variables(self):
        self.kappa = None
        self.kappa_noisy = None
        self.smoothed_map = None
        self.outputs = None

    def _check_kappa_existance(self):
        if self.kappa is None: self._load_kappa()
        if self.kappa_noisy is None: self._add_noise() 

    def _load_kappa(self):
        logging.info(f"Loading kappa map from {os.path.basename(self.kappa_path)}")
        self.kappa = hp.read_map(self.kappa_path)
        self.kappa = hp.reorder(self.kappa, n2r=True)
        
    def _add_noise(self):
        if self.noiseless: 
            self.kappa_noisy = self.kappa.copy()
        else:
            noise_map = self.ng.generate_noise(seed=SuffixGenerator.extract_seed_from_path(self.kappa_path))
            self.kappa_noisy = self.kappa + noise_map
            logging.info(f"Noise map added to kappa map")

    def _reset_params(self, sl=2, ngal=None):
        logging.info(f"Setting parameters: Smoothing Scale {sl}, Number of Galaxies {ngal}")
        self.ks.scale_angle = sl
        if ngal==0:
            self.sg = SuffixGenerator(ks=self.ks)
            self.noiseless = True
        else:
            self.ng.ngal = ngal
            self.sg = SuffixGenerator(ng=self.ng, ks=self.ks)
            self.noiseless = False

    def _plot_maps(self):
        suffix_fullsky = self.sg.generate_fullsky_suffix(self.kappa_path)["fullsky"]
        suffix_cls = self.sg.generate_cls_suffix(self.kappa_path)
        img_kappa_path = os.path.join(self.img_dir, f"kappa_{suffix_cls}.png")
        img_smoothed_path = os.path.join(self.img_dir, f"smoothed_{suffix_fullsky}.png")

        seed = SuffixGenerator.extract_seed_from_path(self.kappa_path)
        zs = SuffixGenerator.extract_redshift_from_path(self.kappa_path)
        conf = SuffixGenerator.extract_type_from_path(self.kappa_path)

        if os.path.exists(img_kappa_path) or not self.overwrite_img:
            logging.info(f"Kappa Images already exist. Skipping...")
        else:
            logging.info(f"Plotting kappa map")
            fig_kappa = plt.figure(figsize=(7, 4))
            title = r"$\Kappa$"+f" ({conf}): Seed {seed}, Source Redshift {zs}"
            hp.orthview(self.kappa, title=title, min=self.vmin, max=self.vmax, cbar=True, fig=fig_kappa)
            fig_kappa.savefig(img_kappa_path, bbox_inches='tight')
            plt.close(fig_kappa)

        if os.path.exists(img_smoothed_path) or not self.overwrite_img:
            logging.info(f"Smoothed Images already exist. Skipping...")
        else:
            logging.info(f"Plotting smoothed map")
            fig_smoothed = plt.figure(figsize=(7, 4))
            title = r"Smoothed $\Kappa$"+f" ({conf}): Seed {seed}, Source Redshift {zs}, Smoothing Scale {self.ks.scale_angle}"+r"$^\circ$"
            hp.orthview(self.smoothed_map, title=title, min=self.vmin, max=self.vmax, cbar=True, fig=fig_smoothed)
            fig_smoothed.savefig(img_smoothed_path, bbox_inches='tight')
            plt.close(fig_smoothed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('datadir', type=str, help='Data directory of convergence maps')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    parser.add_argument('--overwrite_map', action='store_true', help='Overwrite existing files')
    parser.add_argument('--overwrite_img', action='store_true', help='Overwrite existing images')
    parser.add_argument('--overwrite_cls', action='store_true', help='Overwrite existing cls')

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    nside = config.get('NoiseGenerator', {}).get('nside', None)
    ngal_list = config.get('PreKappaProcessor', {}).get('ngal_list', [0, 15, 30, 50])
    sl_list = config.get('PreKappaProcessor', {}).get('sl_list', [2, 5, 8])

    pkp = PreKappaProcessor(args.datadir, nside, ngal_list, sl_list, args)
    pkp.preprocess()