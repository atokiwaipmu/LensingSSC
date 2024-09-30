

import logging
import os
import glob
import numpy as np
import healpy as hp

from src.patch_processor import PatchProcessor
from src.noise_generator import NoiseGenerator
from src.info_extractor import InfoExtractor

class PatchGenerator:
    def __init__(self, data_dir, pp: PatchProcessor, ngal_list=[0, 7, 15, 30, 50]):
        self.data_dir = data_dir
        self.kappa_paths = sorted(glob.glob(os.path.join(data_dir, "kappa", "*.fits")))
        self.smoothed_paths = sorted(glob.glob(os.path.join(data_dir, "smoothed_maps", "*.fits")))
        self.output_dir = os.path.join(data_dir, "patch")
        os.makedirs(self.output_dir, exist_ok=True)

        self.pp = pp
        self.ngal_list = ngal_list
        self.ng_list = {ngal: NoiseGenerator(ngal) for ngal in ngal_list}

    def run(self):
        for kappa_path in self.kappa_paths:
            input_map = None 
            info = InfoExtractor.extract_info_from_path(kappa_path)
            for ngal in self.ngal_list:
                noise = f"ngal{ngal}" if ngal > 0 else "noiseless"
                output_path = self.generate_fname(info, noise, is_snr=False)
                if not os.path.exists(output_path):
                    if input_map is None:
                        input_map = hp.read_map(kappa_path)
                        input_map = hp.reorder(input_map, n2r=True)
                    logging.info(f"Generating kappa patches for seed = {info['seed']} zs = {info['redshift']} {noise}")
                    self.make_save_patches(input_map, output_path, ngal=ngal, is_snr=False)
                tmp_smoothed_paths = self.filter_paths(self.smoothed_paths, info, noise)
                for smoothed_path in tmp_smoothed_paths:
                    output_path = self.generate_fname(info, noise, is_snr=True)
                    sl = smoothed_path.split("_")[-2]
                    output_path = output_path.replace(f"_{noise}.npy", f"_{sl}_{noise}.npy")
                    if os.path.exists(output_path):
                        logging.info(f"Skipping snr patches for seed = {info['seed']} zs = {info['redshift']} {sl} {noise}")
                        continue
                    else:
                        logging.info(f"Generating snr patches for seed = {info['seed']} zs = {info['redshift']} {sl} {noise}")
                        smoothed_map = hp.read_map(smoothed_path) 
                        global_std = np.std(smoothed_map)                   
                        self.make_save_patches(smoothed_map/global_std, output_path, seed=info["seed"], is_snr=True)

    def generate_fname(self, info, noise, is_snr=False):
        prefix = "snr_patches" if is_snr else "kappa_patches"
        return os.path.join(self.output_dir, f"{prefix}_s{info['seed']}_zs{info['redshift']}_oa{self.pp.patch_size}_{noise}.npy")
    
    def filter_paths(self, paths, info, noise):
        return [path for path in paths if (noise in path) and (f"zs{info['redshift']}" in path)]

    def make_save_patches(self, input_map, output_path, seed=0, ngal=0, is_snr=False):
        if is_snr:
            patches = self.pp.make_patches(input_map)
            np.save(output_path, patches)
        else:
            noisy_map = self.ng_list[ngal].add_noise(input_map, seed=seed) if ngal > 0 else input_map
            patches = self.pp.make_patches(noisy_map)
            np.save(output_path, patches)

if __name__ == "__main__":
    from src.utils import parse_arguments, load_config, filter_config
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    config = load_config(args.config_file)
    filtered_config = filter_config(config, PatchProcessor)
    pp = PatchProcessor(**filtered_config)
    pg = PatchGenerator(args.datadir, pp)
    pg.run()