import os
import yaml
import logging
import argparse
from glob import glob
import multiprocessing as mp

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from lenstools import ConvergenceMap

from src.utils import extract_seed_from_path, extract_redshift_from_path
from src.fibonacci_patch import fibonacci_grid_on_sphere, get_patch_pixels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class KappaProcessor:
    """
    A class to process weak lensing convergence maps.
    It handles noise addition, smoothing, and various analyses like bispectrum and power spectrum.
    """

    def __init__(self, datadir, output=None, nside=8192, npatch=273, patch_size=10, xsize=2048, nbin=15, lmin=300, lmax=3000, scale_angle=2, ngal=30, nest=True, localmean=False, noiseless=False, overwrite=False):
        """
        Initialize the KappaProcessor with parameters for data processing.
        
        Parameters:
        - datadir: Directory containing input data.
        - output: Directory to save output data.
        - nside: Healpix resolution parameter.
        - npatch: Number of patches for analysis.
        - patch_size: Size of each patch in degrees.
        - xsize: Size of the patches in pixels.
        - nbin: Number of bins for statistical analysis.
        - lmin, lmax: Min and max multipoles for analysis.
        - scale_angle: Smoothing scale angle in arcminutes.
        - ngal: Galaxy number density for noise simulation.
        - nest: Healpix nesting scheme.
        - localmean: If use local mean instead of global mean for SNR.
        - noiseless: Flag for noiseless simulation.
        - overwrite: Flag to overwrite existing results.
        """
        self.datadir = datadir
        self.nside = nside
        self.nest = nest

        self.npatch = npatch
        self.points = fibonacci_grid_on_sphere(self.npatch)
        self.radius = np.radians(patch_size) * np.sqrt(2)
        self.points = self.points[(self.points[:, 0] < np.pi - self.radius) & (self.points[:, 0] > self.radius)]
        self.points_lonlatdeg = np.array([hp.rotator.vec2dir(hp.ang2vec(center[0], center[1]), lonlat=True) for center in self.points])
        self.patch_size = patch_size
        self.xsize = xsize
        self.padding = 0.1 + np.sqrt(2)
        self.reso = patch_size * 60 / xsize

        self.noiseless = noiseless
        self.epsilon = 0.3
        self.ngal = ngal
        self.pixarea = hp.nside2pixarea(self.nside, degrees=True) * 60 ** 2  # arcmin^2

        self.plot_demo = False
        self.plot_idx = self.npatch // 2

        self.scale_angle = scale_angle  # arcmin
        self.localmean = localmean
        self.nbin = nbin
        self.lmin, self.lmax = lmin, lmax
        self.bins = np.linspace(-4, 4, self.nbin + 1, endpoint=True)
        self.l_edges = np.linspace(self.lmin, self.lmax, self.nbin + 1, endpoint=True)

        self.overwrite = overwrite
        if output is not None:
            self.savedir = output
        else:
            self.savedir = os.path.join(os.path.dirname(self.datadir), "flat")
        os.makedirs(self.savedir, exist_ok=True)
        self.kappa_map_paths = glob(os.path.join(self.datadir, "*.fits"))
        logging.info(f"Found {len(self.kappa_map_paths)} kappa maps in {self.datadir}")

    def run_analysis(self):
        """
        Runs the full analysis pipeline on all kappa maps found in the specified directory.
        """
        for idx, kappa_path in enumerate(self.kappa_map_paths):
            logging.info(f"Processing {idx + 1}/{len(self.kappa_map_paths)}")
            self.process_data(idx)

    def process_data(self, idx):
        """
        Processes a single kappa map by reading, adding noise, smoothing, and performing patch analysis.
        
        Parameters:
        - idx: Index of the kappa map to process.
        """
        kappa_path = self.kappa_map_paths[idx]
        seed = extract_seed_from_path(kappa_path)
        zs = extract_redshift_from_path(kappa_path)
        suffix = f"s{seed}_zs{zs:.1f}_oa{self.patch_size}_sl{self.scale_angle}"
        if self.noiseless:
            suffix += "_noiseless"
        else:
            suffix += f"_ngal{self.ngal}"
        output_filename = f"analysis_sqclpdpm_{suffix}.npy"
        output_path = os.path.join(self.savedir, output_filename)

        if os.path.exists(output_path) and not self.overwrite:
            logging.info(f"Skipping {output_path}")
            return

        logging.info(f"Reading and processing kappa map from {kappa_path}")
        kappa_map, global_std = self._read_add_noise_and_smooth(kappa_path, seed=idx)
        self.plot_kappa_map(kappa_map, seed, zs)

        patches_kappa = [
            get_patch_pixels(
                hp.gnomview(
                    kappa_map,
                    nest=self.nest,
                    rot=point,
                    xsize=self.xsize * self.padding,
                    reso=self.reso,
                    return_projected_map=True,
                    no_plot=True
                ),
                self.xsize
            )
            for point in self.points_lonlatdeg
        ]
        if self.localmean:
            patches_snr = [patch / np.std(patch) for patch in patches_kappa]
        else:
            patches_snr = [patch / global_std for patch in patches_kappa]

        args = list(zip(range(self.npatch), patches_kappa, patches_snr))
        with mp.Pool(processes=mp.cpu_count()) as pool:
            datas = pool.starmap(self._process_patch, args)

        data = np.array(datas)

        logging.info(f"Saving processed data to {output_path}")
        np.save(output_path, data)

    def plot_kappa_map(self, kappa_map, seed, zs):
        """
        Plots the kappa map for a given index.
        
        Parameters:
        - idx: Index of the kappa map to plot.
        """
        fig = plt.figure(figsize=(10, 5))
        hp.mollview(kappa_map, nest=self.nest, title='Kappa Map: Seed {}, source redshift {}, scale angle {}'.format(seed, zs, self.scale_angle), fig=fig.number, min=-0.024, max=0.024)
        fig.savefig(os.path.join(os.path.dirname(self.datadir), f"kappa_seed{seed}_zs{zs:.1f}_sl{self.scale_angle}.png"))
        plt.close(fig)

    def plot_demo_patch(self, idx, patch_pixels, patch_snr_pixels):
        """
        Plots a demo patch for a given kappa map index.
        
        Parameters:
        - idx: Index of the kappa map to plot.
        """
        logging.info(f"Saving demo patch {idx} image")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(patch_pixels, vmin=-0.024, vmax=0.024)
        ax[1].imshow(patch_snr_pixels, vmin=-2, vmax=2)
        fig.savefig(os.path.join(os.path.dirname(self.datadir),f"demo_patch{idx}.png"))
        plt.close(fig)

    def _process_patch(self, i, patch_pixels, patch_snr_pixels):
        """
        Processes a single patch, performing various analyses and optionally saving a demo image.
        
        Parameters:
        - i: Patch index.
        - patch_pixels: Pixel values of the kappa patch.
        - patch_snr_pixels: Pixel values of the signal-to-noise ratio patch.
        """
        logging.info(f"Processing patch {i}")
        if i == self.plot_idx and self.plot_demo:
            self.plot_demo_patch(i, patch_pixels, patch_snr_pixels)

        ell, squeezed, cl = self._perform_analysis(patch_pixels)
        nu, p, peaks, minima = self._perform_analysis_snr(patch_snr_pixels)

        data_tmp = np.hstack([squeezed, cl, p, peaks, minima])
        return data_tmp

    def _generate_noise(self, seed=0):
        """
        Generates a noise map using a specified seed.
        
        Parameters:
        - seed: Random seed for noise generation.
        
        Returns:
        - A noise map with the same dimensions as the kappa map.
        """
        logging.info(f"Generating noise with seed {seed}")
        np.random.seed(seed)
        sigma = self.epsilon / np.sqrt(self.ngal * self.pixarea)
        noise_map = np.random.normal(loc=0, scale=sigma, size=(hp.nside2npix(self.nside),))
        return noise_map

    def _read_add_noise_and_smooth(self, path, seed=0):
        """
        Reads a kappa map, adds noise if specified, and applies smoothing.
        
        Parameters:
        - path: Path to the kappa map file.
        - seed: Random seed for noise generation.
        
        Returns:
        - Tuple of the kappa map and the signal-to-noise ratio map.
        """
        logging.info(f"Reading kappa map from {path}")
        kappa_map = hp.read_map(path)
        if not self.noiseless:
            noise_map = self._generate_noise(seed)
            kappa_map += noise_map
        logging.info(f"Smoothing kappa map with sigma {self.scale_angle / 60 * np.pi / 180}")
        kappa_map = hp.smoothing(kappa_map, sigma=self.scale_angle / 60 * np.pi / 180, nest=self.nest)
        global_std = np.std(kappa_map)
        return kappa_map, global_std

    def _perform_analysis(self, convergence):
        """
        Performs bispectrum and power spectrum analysis on the given convergence map.
        
        Parameters:
        - convergence: Convergence map data.
        
        Returns:
        - ell: Multipole moments.
        - squeezed: Bispectrum for squeezed configurations.
        - cl: Power spectrum.
        """
        convergence_map = ConvergenceMap(convergence, angle=self.patch_size * u.deg)
        ell, squeezed = convergence_map.bispectrum(self.l_edges, ratio=0.1, configuration='folded')
        _, cl = convergence_map.powerSpectrum(self.l_edges)
        return ell, squeezed, cl

    def _perform_analysis_snr(self, snr):
        """
        Performs analysis on the signal-to-noise ratio (SNR) map, including peak detection and PDF calculation.
        
        Parameters:
        - snr: SNR map data.
        
        Returns:
        - nu: PDF bin centers.
        - p: PDF values.
        - peaks: Histogram of peak heights.
        - minima: Histogram of minima heights.
        """
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
        """
        Excludes peaks/minima near the edges of the patch.
        
        Parameters:
        - heights: Heights of the peaks or minima.
        - positions: Positions of the peaks or minima.
        
        Returns:
        - Filtered heights and positions excluding those near the edges.
        """
        tmp_positions = positions.value * self.xsize / self.patch_size
        mask = (tmp_positions[:, 0] > 0) & (tmp_positions[:, 0] < self.xsize - 1) & (tmp_positions[:, 1] > 0) & (tmp_positions[:, 1] < self.xsize - 1)
        return heights[mask], tmp_positions[mask].astype(int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute weak lensing convergence maps')
    parser.add_argument('datadir', type=str, help='Data directory of convergence maps')
    parser.add_argument("--output", type=str, help="Output directory to save results")
    parser.add_argument('--noiseless', action='store_true', help='Noiseless simulation')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--config', type=str, help='Path to YAML config file')

    args = parser.parse_args()

    # Initialize empty config
    config = {}

    # Load configuration from YAML if provided and exists
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as file:
            try:
                config = yaml.safe_load(file)  # Load the configuration from YAML
            except yaml.YAMLError as exc:
                print("Warning: The config file is empty or invalid. Proceeding with default parameters.")
                print(exc)

    # Override YAML configuration with command-line arguments if provided
    config.update({
        'datadir': args.datadir,
        'output': args.output if args.output else config.get('output', None),
        'noiseless': args.noiseless if args.noiseless else config.get('noiseless', False),
        'overwrite': args.overwrite if args.overwrite else config.get('overwrite', False),
    })

    # Define the allowed keys
    allowed_keys = {
        'datadir', 'output', 'nside', 'npatch', 'patch_size', 'xsize', 'nbin', 
        'lmin', 'lmax', 'scale_angle', 'ngal', 'nest', 'localmean', 'noiseless', 'overwrite'
    }
    config = {k: v for k, v in config.items() if k in allowed_keys}

    kappa_proc = KappaProcessor(**config)
    kappa_proc.run_analysis()
