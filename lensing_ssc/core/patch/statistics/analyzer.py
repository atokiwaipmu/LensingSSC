import logging
import multiprocessing as mp
import numpy as np
from astropy import units as u
from lenstools import ConvergenceMap
from pathlib import Path

from lensing_ssc.core.patch.statistics.minkowski import compute_minkowski_functionals
from lensing_ssc.core.patch.statistics.bispectrum import compute_bispectrum, dimensionless_bispectrum
from lensing_ssc.core.patch.statistics.peak_stats import compute_peak_statistics
from lensing_ssc.utils.io import save_results_to_hdf5

class PatchAnalyzer:
    """
    Analyzes convergence map patches using various statistical methods.
    
    Provides:
    - Power spectrum analysis
    - Bispectrum calculation
    - Minkowski functionals
    - Peak/minima statistics
    - PDF computation
    
    Parameters
    ----------
    ngal_list : list, optional
        Galaxy density values for noise simulation, by default [0, 7, 15, 30, 50]
    sl_list : list, optional
        Smoothing lengths in arcminutes, by default [2, 5, 8, 10]
    nbin : int, optional
        Number of bins for histograms, by default 8
    lmin, lmax : int, optional
        Minimum and maximum multipole values, by default 300, 3000
    epsilon : float, optional
        Galaxy shape noise parameter, by default 0.26
    patch_size_deg : float, optional
        Size of the patch in degrees, by default 10
    xsize : int, optional
        Resolution of the patch in pixels, by default 2048
    """
    def __init__(self, 
                 ngal_list=[0, 7, 15, 30, 50], 
                 sl_list = [2, 5, 8, 10], 
                 nbin=8, 
                 lmin=300, 
                 lmax=3000, 
                 epsilon=0.26, 
                 patch_size_deg=10, 
                 xsize=2048):
        # Initialize parameters
        self.nbin = nbin
        self.bins = np.linspace(-4, 4, self.nbin + 1, endpoint=True)
        self.nu = (self.bins[1:] + self.bins[:-1]) / 2

        self.lmin, self.lmax = lmin, lmax
        self.l_edges = np.logspace(np.log10(self.lmin), np.log10(self.lmax), self.nbin + 1, endpoint=True)
        self.ell = (self.l_edges[1:] + self.l_edges[:-1]) / 2
        self.binwidth = self.bins[1] - self.bins[0]

        self.ngal_list = ngal_list
        self.sl_list = sl_list
        self.epsilon = epsilon

        self.patch_size = patch_size_deg
        self.xsize = xsize
        self.pixarea_arcmin2 = (patch_size_deg * 60 / xsize)**2
        
        logging.info(f"PatchAnalyzer initialized with patch_size={patch_size_deg} deg, xsize={xsize}")

    def process_patches(self, patches_kappa, num_processes=mp.cpu_count()):
        """
        Process patches for all ngal and smoothing length values.
        
        Parameters
        ----------
        patches_kappa : numpy.ndarray
            Array of convergence map patches
        num_processes : int, optional
            Number of processes for parallel computation
            
        Returns
        -------
        dict
            Dictionary with computed statistics
        """
        results = {}
        
        for ngal in self.ngal_list:
            logging.info(f"Processing kappa for ngal={ngal}")
            
            results[ngal] = {}
            with mp.Pool(processes=num_processes) as pool:
                datas = pool.starmap(self._process_kappa, 
                                     zip(patches_kappa, [ngal] * len(patches_kappa)))
            
            datas = np.array(datas).astype(np.float32)
        
            results[ngal]["equilateral"] = datas[:, 0, :].astype(np.float32)
            results[ngal]["isosceles"] = datas[:, 1, :].astype(np.float32)
            results[ngal]["squeezed"] = datas[:, 2, :].astype(np.float32)
            results[ngal]["cl"] = datas[:, 3, :].astype(np.float32)

            logging.info(f"kappa processed for ngal={ngal}")

            for sl in self.sl_list:
                logging.info(f"Processing snr for ngal={ngal}, sl={sl}")

                results[ngal][sl] = {}
                with mp.Pool(processes=num_processes) as pool:
                    datas = pool.starmap(self._process_snr, 
                                         zip(patches_kappa, [ngal] * len(patches_kappa), 
                                             [sl] * len(patches_kappa)))

                pdfs, peaks, minimas, v0s, v1s, v2s, s0s, s1s = zip(*datas)
                
                results[ngal][sl]["pdf"] = np.array(pdfs).astype(np.float32)
                results[ngal][sl]["peaks"] = np.array(peaks).astype(np.float32)
                results[ngal][sl]["minima"] = np.array(minimas).astype(np.float32)
                results[ngal][sl]["v0"] = np.array(v0s).astype(np.float32)
                results[ngal][sl]["v1"] = np.array(v1s).astype(np.float32)
                results[ngal][sl]["v2"] = np.array(v2s).astype(np.float32)
                results[ngal][sl]["sigma0"] = np.array(s0s).astype(np.float32)
                results[ngal][sl]["sigma1"] = np.array(s1s).astype(np.float32)

                logging.info(f"snr processed for ngal={ngal}, sl={sl}")

        return results
    
    def _process_kappa(self, patch_pixels, ngal):
        """Process a single kappa patch with given noise level"""
        logging.info(f"Processing kappa patch with ngal={ngal}")
        if ngal == 0:
            pixels = patch_pixels
        else:
            noise_level = self.epsilon / np.sqrt(ngal * self.pixarea_arcmin2)
            noise_map = np.random.normal(0, noise_level, patch_pixels.shape)
            pixels = patch_pixels + noise_map

        conv_map = ConvergenceMap(pixels, angle=self.patch_size * u.deg)
        equilateral, isosceles, squeezed = compute_bispectrum(conv_map, self.l_edges, self.ell)
        cl = conv_map.powerSpectrum(self.l_edges)[1] * self.ell * (self.ell + 1) / (2 * np.pi)

        return [equilateral, isosceles, squeezed, cl]

    def _process_snr(self, patch_pixels, ngal, sl):
        """Process a single patch with given noise level and smoothing length"""
        logging.info(f"Processing snr patch with ngal={ngal}, sl={sl}")
        if ngal == 0:
            pixels = patch_pixels
        else:
            noise_level = self.epsilon / np.sqrt(ngal * self.pixarea_arcmin2)
            noise_map = np.random.normal(0, noise_level, patch_pixels.shape)
            pixels = patch_pixels + noise_map

        conv_map = ConvergenceMap(pixels, angle=self.patch_size * u.deg)
        smoothed_map = conv_map.smooth(sl*u.arcmin).data
        sigma0 = np.std(smoothed_map)
        smoothed_map = ConvergenceMap(smoothed_map/sigma0, angle=self.patch_size * u.deg)
        
        pdf = smoothed_map.pdf(self.bins)[1]
        peaks = compute_peak_statistics(smoothed_map, self.bins, is_minima=False, 
                                       patch_size=self.patch_size, xsize=self.xsize)
        minima = compute_peak_statistics(smoothed_map, self.bins, is_minima=True,
                                        patch_size=self.patch_size, xsize=self.xsize)
        
        v0, v1, v2, sigma1 = compute_minkowski_functionals(smoothed_map, self.bins)

        return pdf, peaks, minima, v0, v1, v2, sigma0, sigma1

    def analyze_batch(self, input_paths, output_path, overwrite=False):
        """
        Analyze multiple patch files and save results to an HDF5 file.
        
        Parameters
        ----------
        input_paths : list
            List of paths to patch files
        output_path : str or Path
            Path to the output HDF5 file
        overwrite : bool, optional
            Whether to overwrite existing output file
        """
        output_path = Path(output_path)
        
        if output_path.exists() and not overwrite:
            logging.info(f"Output file {output_path} already exists, skipping.")
            return
            
        # Load and process all patches
        all_results = {}
        for input_path in input_paths:
            logging.info(f"Processing {input_path}")
            patches = np.load(input_path, mmap_mode='r')
            results = self.process_patches(patches)
            file_key = Path(input_path).stem
            all_results[file_key] = results
            
        # Save results with metadata
        save_results_to_hdf5(all_results, output_path, analyzer=self)
        logging.info(f"Results saved to {output_path}")