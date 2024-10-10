

import os
import glob
import logging
import numpy as np
import healpy as hp

from src.info_extractor import InfoExtractor
from src.patch_analyser import PatchAnalyser
from src.fullsky_analyser import FullSkyAnalyser

class KappaAnalyser:
    def __init__(self, datadir,
                 pa: PatchAnalyser,
                 fsa: FullSkyAnalyser,
                 overwrite=False):
        self.datadir = datadir
        self.pa = pa
        self.fsa = fsa
        self.overwrite = overwrite
        
        self._prepare_output_path()
        self._check_preprocessed_data()

    def analyse(self):
        #self._analyse_fullsky()
        self._analyse_patch()

    def _analyse_fullsky(self):
        for cl_path in self.cls_paths:
            tmp_cl = None
            tmp_smoothed_paths = self._filter_paths_by_input(self.smoothed_map_paths, cl_path)

            for smoothed_path in tmp_smoothed_paths:
                fname = os.path.basename(smoothed_path).replace('.fits', '.npy').replace('kappa_smoothed', 'fullsky_clpdpm')
                output_path = os.path.join(self.full_dir, fname)

                if not os.path.exists(output_path):
                    if tmp_cl is None:
                        tmp_cl = hp.read_cl(cl_path)

                    logging.info(f"Analysing fullsky map for {fname}")
                    smoothed_map = hp.read_map(smoothed_path)
                    global_std = np.std(smoothed_map)
                    data = self.fsa.process_map(smoothed_map/global_std, tmp_cl)
                    np.save(output_path, data)
                else:
                    logging.info(f"Skipping fullsky map for {fname}")

    def _analyse_patch(self):
        for patch_kappa_path in self.patch_kappa_paths:
            tmp_kappa_patches = None
            tmp_snr_paths = self._filter_paths_by_input(self.patch_snr_paths, patch_kappa_path) # assuming oa is the same for kappa and snr

            for patch_snr_path in tmp_snr_paths:
                fname = os.path.basename(patch_snr_path).replace('patches', 'analysis')
                output_path = os.path.join(self.analysis_patch_dir, fname)

                if not os.path.exists(output_path) or self.overwrite:
                    if tmp_kappa_patches is None:
                        tmp_kappa_patches = np.load(patch_kappa_path)

                    logging.info(f"Analysing patch for {fname}")
                    snr_patches = np.load(patch_snr_path)
                    data = self.pa.process_patches(tmp_kappa_patches, snr_patches)
                    np.save(output_path, data)
                else:
                    logging.info(f"Skipping patch for {fname}")

    def _prepare_output_path(self):
        self.kappa_dir = os.path.join(self.datadir, 'kappa')
        self.smoothed_dir = os.path.join(self.datadir, 'smoothed_maps')
        self.cls_dir = os.path.join(self.datadir, 'cls')

        self.patch_kappa_dir = os.path.join(self.datadir, 'patch_kappa')
        self.patch_snr_dir = os.path.join(self.datadir, 'patch_snr')
        self.analysis_patch_dir = os.path.join(self.datadir, 'analysis_patch')
        self.analysis_fullsky_dir = os.path.join(self.datadir, 'analysis_fullsky')

        os.makedirs(self.analysis_patch_dir, exist_ok=True)
        os.makedirs(self.analysis_fullsky_dir, exist_ok=True)
        logging.info(f"Output directories created")

    def _check_preprocessed_data(self):
        self.kappa_map_paths = sorted(glob.glob(os.path.join(self.kappa_dir, '*.fits')))
        self.smoothed_map_paths = sorted(glob.glob(os.path.join(self.smoothed_dir, '*.fits')))
        self.cls_paths = sorted(glob.glob(os.path.join(self.cls_dir, '*.fits')))
        self.patch_snr_paths = sorted(glob.glob(os.path.join(self.patch_snr_dir, '*.npy')))
        self.patch_kappa_paths = sorted(glob.glob(os.path.join(self.patch_kappa_dir, '*.npy')))

    def _filter_paths_by_input(self, paths, input_path):
        info = InfoExtractor.extract_info_from_path(input_path)
        noise = 'noiseless' if info['ngal'] == 0 else f'ngal{info["ngal"]}'
        return [path for path in paths if (noise in path) and (f"zs{info['redshift']}" in path) and (f"s{info['seed']}" in path)]

if __name__ == "__main__":
    from src.utils import parse_arguments, load_config, filter_config
    from src.patch_processor import PatchProcessor
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    config = load_config(args.config_file)

    filtered_config_pp = filter_config(config, PatchProcessor)
    pp = PatchProcessor(**filtered_config_pp)

    filtered_config_pa = filter_config(config, PatchAnalyser)
    pa = PatchAnalyser(pp, **filtered_config_pa)

    filtered_config_fsa = filter_config(config, FullSkyAnalyser)
    fsa = FullSkyAnalyser(**filtered_config_fsa)

    ka = KappaAnalyser(args.datadir, pa, fsa, overwrite=args.overwrite)
    ka.analyse()