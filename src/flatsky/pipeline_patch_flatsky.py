import os
import logging
import argparse
from glob import glob
import multiprocessing as mp

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from lenstools import ConvergenceMap

from src.flatsky.fibonacci_patch import fibonacci_grid_on_sphere, get_patch_pixels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class kappa_processor():
    def __init__(self, data_dir, nside=8192, npatch=273, patch_size = 10, xsize=2048, nbin=15, lmin=300, lmax=3000, scale_angle=2, ngal=30, nest=True, noiseless=False, overwrite=False):
        # data initialization
        self.data_dir = data_dir
        self.nside = nside
        self.nest = nest

        # patch initialization
        self.npatch = npatch
        self.points = fibonacci_grid_on_sphere(self.npatch)
        self.radius = np.radians(patch_size) * np.sqrt(2)
        self.points = self.points[(self.points[:, 0] < np.pi - self.radius) & (self.points[:, 0] > self.radius)]
        self.points_lonlatdeg = np.array([hp.rotator.vec2dir(hp.ang2vec(center[0], center[1]), lonlat=True) for center in self.points])
        self.patch_size = patch_size
        self.xsize = xsize
        self.padding = 0.1 + np.sqrt(2)
        self.reso = patch_size*60/xsize

        # noise initialization
        self.noiseless = noiseless
        self.ngal = ngal
        self.pixarea = hp.nside2pixarea(self.nside, degrees=True) * 60**2 # arcmin^2

        # smoothing initialization
        self.scale_angle = scale_angle # arcmin

        # bin initialization
        self.nbin= nbin
        self.lmin, self.lmax = lmin, lmax
        self.bins = np.linspace(-4, 4, self.nbin+1, endpoint=True)
        self.l_edges = np.linspace(self.lmin, self.lmax, self.nbin+1, endpoint=True)

        self.overwrite = overwrite
        os.makedirs(os.path.join(self.data_dir, "flat"), exist_ok=True)
        self.kappa_map_paths = glob(os.path.join(self.data_dir, "kappa", "*.fits"))

    def run_analysis(self):
        for idx in range(len(self.kappa_map_paths)):
            logging.info(f"Processing {idx+1}/{len(self.kappa_map_paths)}")
            self.process_data(idx)

    def process_data(self, idx):
        kappa_path = self.kappa_map_paths[idx]
        suffix = os.path.basename(kappa_path).split("_", 1)[1].rsplit(".", 1)[0]
        if self.noiseless:
            output_path = os.path.join(self.data_dir, "flat", f"analysis_sqclpdpm_{suffix}_sl{self.scale_angle}_noiseless.npy")
        else:
            output_path = os.path.join(self.data_dir, "flat", f"analysis_sqclpdpm_{suffix}_sl{self.scale_angle}_ngal{self.ngal}.npy")

        if os.path.exists(output_path) and not self.overwrite:
            logging.info(f"Skipping {output_path}")
            return
        logging.info(f"Reading and processing kappa map from {kappa_path}")
        kappa_map, snr_map = self._read_addNoise_smoothing(kappa_path, seed=idx)

        patches_kappa = [get_patch_pixels(hp.gnomview(kappa_map, nest=self.nest, rot=point, xsize=self.xsize*self.padding, reso=self.reso, return_projected_map=True, no_plot=True), self.xsize) for point in self.points_lonlatdeg]
        patches_snr = [get_patch_pixels(hp.gnomview(snr_map, nest=self.nest, rot=point, xsize=self.xsize*self.padding, reso=self.reso, return_projected_map=True, no_plot=True), self.xsize) for point in self.points_lonlatdeg]
        args = [(i, patch_kappa, patch_snr) for i, patch_kappa, patch_snr in zip(range(self.npatch), patches_kappa, patches_snr)]
    
        with mp.Pool(processes=mp.cpu_count()) as pool:
            datas = pool.starmap(self._process_patch, args)

        data = np.array(datas)
        
        logging.info(f"Saving processed data to {output_path}")
        np.save(output_path, data)

    def _process_patch(self, i, patch_pixels, patch_snr_pixels):
        logging.info(f"Processing patch{i}")
        if i == self.npatch // 2:
            logging.info(f"saving demo patch{i} image")
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            cax = ax[0].imshow(patch_pixels, vmin=-0.024, vmax=0.024)
            fig.colorbar(cax, ax=ax[0])
            cax = ax[1].imshow(patch_snr_pixels, vmin=-2, vmax=2)
            fig.colorbar(cax, ax=ax[1])
            fig.savefig(os.path.join(self.data_dir, f"demo_patch{i}.png"))
            plt.close(fig)

        ell, squeezed, cl = self._perform_analysis(patch_pixels)
        nu, p, peaks, minima = self._perform_analysis_snr(patch_snr_pixels)

        data_tmp = np.hstack([squeezed, cl, p, peaks, minima])
        return data_tmp

    def _gen_noise(self, seed=0):
        logging.info(f"Generating noise with seed {seed}")
        np.random.seed(seed)
        sigma = 0.3 / np.sqrt(self.ngal * self.pixarea)
        noise_map = np.random.normal(loc=0, scale=sigma, size=(hp.nside2npix(self.nside),))
        return noise_map

    def _read_addNoise_smoothing(self, path, seed=0):
        logging.info(f"Reading kappa map from {path}")
        kappa_map = hp.read_map(path)
        if not self.noiseless:
            noise_map = self._gen_noise(seed)
            kappa_map += noise_map
        logging.info(f"Smoothing kappa map with sigma {self.scale_angle/60*np.pi/180}")
        kappa_map = hp.smoothing(kappa_map, sigma=self.scale_angle/60*np.pi/180, nest=self.nest)
        global_std = np.std(kappa_map)
        snr_map = kappa_map / global_std
        return kappa_map, snr_map
    
    def _perform_analysis(self, convergence):
        convergence_map = ConvergenceMap(convergence, angle=self.patch_size * u.deg)
        #ell, equilateral = convergence_map.bispectrum(self.l_edges, configuration='equilateral')
        ell, squeezed = convergence_map.bispectrum(self.l_edges, ratio=0.1, configuration='folded')
        _, cl = convergence_map.powerSpectrum(self.l_edges)

        return ell, squeezed, cl
    
    def _perform_analysis_snr(self, snr):
        snr_map = ConvergenceMap(snr, angle=self.patch_size * u.deg)
        nu, p = snr_map.pdf(self.bins)
        peak_height, peak_positions = snr_map.locatePeaks(self.bins)
        peak_height, peak_positions = self._exclude_edges(peak_height, peak_positions)
        peaks = np.histogram(peak_height, bins=self.bins)[0]

        snr_map_minus = ConvergenceMap(-snr_map.data, angle=self.patch_size * u.deg)
        minima_height, minima_positions = snr_map_minus.locatePeaks(self.bins)
        minima_height, minima_positions = self._exclude_edges(minima_height, minima_positions)
        minima = np.histogram(minima_height, bins=self.bins)[0]

        return nu, p, peaks, minima

    def _exclude_edges(self, heights, positions):
        tmp_positions = positions.value * self.xsize / self.patch_size
        mask = (tmp_positions[:, 0] > 0) & (tmp_positions[:, 0] < self.xsize-1) & (tmp_positions[:, 1] > 0) & (tmp_positions[:, 1] < self.xsize-1)
        return heights[mask], tmp_positions[mask].astype(int)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('datadir', type=str, help='Data directory')
    parser.add_argument('--noiseless', action='store_true', help='Noiseless simulation')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()
    kappa_proc = kappa_processor(args.datadir, noiseless=args.noiseless, overwrite=args.overwrite)
    kappa_proc.run_analysis()