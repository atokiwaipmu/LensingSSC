
import logging
import numpy as np
from astropy import units as u
from lenstools import ConvergenceMap
import multiprocessing as mp

from src.patch_processor import PatchProcessor

class PatchAnalyser:
    def __init__(self, pp: PatchProcessor, nbin=15, lmin=300, lmax=3000):
        logging.info(f"Initializing FlatPatchAnalyser with nbin={nbin}, lmin={lmin}, lmax={lmax}")
        self.nbin = nbin
        self.lmin, self.lmax = lmin, lmax
        self.bins = np.linspace(-4, 4, self.nbin + 1, endpoint=True)
        self.l_edges = np.logspace(np.log10(self.lmin), np.log10(self.lmax), self.nbin + 1, endpoint=True)
        self.binwidth = self.bins[1] - self.bins[0]

        self.patch_size = pp.patch_size
        self.xsize = pp.xsize

    def process_patches(self, patches_kappa, patches_snr, num_processes=mp.cpu_count()):
        with mp.Pool(processes=num_processes) as pool:
            datas = pool.starmap(self._process_patch, zip(patches_kappa, patches_snr))
        return np.array(datas).astype(np.float32)
    
    def _process_patch(self, patch_pixels, patch_snr_pixels):
        """
        Processes a single patch, computing various statistics (bispectrum, power spectrum, peak counts, etc.).
        """
        # Process kappa (convergence) map
        conv_map = ConvergenceMap(patch_pixels, angle=self.patch_size * u.deg)
        logging.info(f"Computing bispectrum...")
        squeezed_bispectrum = self._compute_bispectrum(conv_map)
        logging.info(f"Computing power spectrum...")
        cl_power_spectrum = self._compute_power_spectrum(conv_map)
        
        # Process SNR map
        snr_map = ConvergenceMap(patch_snr_pixels, angle=self.patch_size * u.deg)
        logging.info(f"Computing PDF...")
        pdf_vals = self._compute_pdf(snr_map)
        logging.info(f"Finding peaks and minima...")
        peaks = self._compute_peak_statistics(snr_map, is_minima=False)
        minima = self._compute_peak_statistics(snr_map, is_minima=True)
        
        # Concatenate all computed statistics
        data_tmp = np.hstack([squeezed_bispectrum, cl_power_spectrum, pdf_vals, peaks, minima])
        return data_tmp

    def _compute_bispectrum(self, conv_map: ConvergenceMap):
        _, squeezed = conv_map.bispectrum(self.l_edges, ratio=0.1, configuration='folded')
        return squeezed
    
    def _compute_power_spectrum(self, conv_map: ConvergenceMap):
        _, cl = conv_map.powerSpectrum(self.l_edges)
        return cl
    
    def _compute_pdf(self, snr_map: ConvergenceMap):
        _, pdf_vals = snr_map.pdf(self.bins)
        return pdf_vals
    
    def _compute_peak_statistics(self, snr_map: ConvergenceMap, is_minima=False):
        if is_minima:
            # Invert the map for minima computation
            snr_map = ConvergenceMap(-snr_map.data, angle=self.patch_size * u.deg)

        height, positions = snr_map.locatePeaks(self.bins)
        height, positions = self._exclude_edges(height, positions)
        peaks = np.histogram(height, bins=self.bins)[0]
        peaks = peaks / np.sum(peaks) / self.binwidth
        return peaks
    
    def _exclude_edges(self, heights, positions):
        """
        Excludes edge values from the peak or minima positions to avoid boundary issues.
        """
        # Scale positions to the patch size and apply boundary mask
        tmp_positions = positions.value * self.xsize / self.patch_size
        mask = (tmp_positions[:, 0] > 0) & (tmp_positions[:, 0] < self.xsize - 1) & \
               (tmp_positions[:, 1] > 0) & (tmp_positions[:, 1] < self.size - 1)
        return heights[mask], tmp_positions[mask].astype(int)