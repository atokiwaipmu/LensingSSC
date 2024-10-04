# src/fullsky_analyser.py

import logging
import numpy as np
from typing import Tuple

from src.utils import parallel_histogram
from src.extrema_finder import ExtremaFinder

class FullSkyAnalyser:
    """
    Analyzes full-sky maps by computing histograms and identifying extrema.

    Attributes:
        nside (int): Healpix resolution parameter.
        nbin (int): Number of histogram bins.
        lmin (int): Minimum multipole for analysis.
        lmax (int): Maximum multipole for analysis.
        bins (np.ndarray): Linearly spaced bin edges for histograms.
        l_edges (np.ndarray): Logarithmically spaced bin edges for discretized C_l.
        binwidth (float): Width of each histogram bin.
        ef (ExtremaFinder): Object for detecting peaks and minima.
    """

    def __init__(self, nside: int = 8192, nbin: int = 15, lmin: int = 300, lmax: int = 3000):
        self.nside = nside
        self.nbin = nbin
        self.lmin, self.lmax = lmin, lmax
        self.bins = np.linspace(-4, 4, nbin + 1)
        self.l_edges = np.logspace(np.log10(lmin), np.log10(lmax), nbin + 1)
        self.binwidth = self.bins[1] - self.bins[0]
        self.ef = ExtremaFinder(nside=nside)

        logging.info(
            f"FullSkyAnalyser initialized with nside={nside}, nbin={nbin}, "
            f"lmin={lmin}, lmax={lmax}"
        )

    def process_map(self, snr_map: np.ndarray, cl: np.ndarray) -> np.ndarray:
        """
        Processes a full-sky map to compute discretized C_l, histograms, and extrema amplitudes.

        Args:
            snr_map (np.ndarray): Signal-to-noise ratio map.
            cl (np.ndarray): Continuous C_l spectrum.

        Returns:
            np.ndarray: Combined array of discretized C_l, PDF, peaks, and minima histograms.
        """
        cl_disc = self._discretize_cl(cl)
        logging.info("Computing PDF of SNR map...")
        pdf_vals = self._compute_normalized_histogram(snr_map)

        logging.info("Identifying extrema in SNR map...")
        _, peak_amp, _, minima_amp = self.ef.find_extrema(snr_map)

        logging.info("Computing histogram for peaks...")
        peaks_hist = self._compute_normalized_histogram(peak_amp)

        logging.info("Computing histogram for minima...")
        minima_hist = self._compute_normalized_histogram(minima_amp)

        return np.concatenate([cl_disc, pdf_vals, peaks_hist, minima_hist])
    
    def _compute_normalized_histogram(self, data: np.ndarray) -> np.ndarray:
        """
        Computes a normalized histogram for the given data.

        Args:
            data (np.ndarray): Data to be histogrammed.

        Returns:
            np.ndarray: Normalized histogram.
        """
        hist = parallel_histogram(data=data, bins=self.bins)
        return hist / (hist.sum() * self.binwidth)
    
    def _discretize_cl(self, cl_cont: np.ndarray) -> np.ndarray:
        """
        Discretizes the continuous C_l spectrum into defined bins.

        Args:
            cl_cont (np.ndarray): Continuous C_l spectrum.

        Returns:
            np.ndarray: Discretized C_l spectrum.
        """
        ell = np.arange(2, self.lmax + 1)
        bin_indices = np.digitize(ell, self.l_edges, right=True)

        # Exclude out-of-range indices
        valid = (bin_indices > 0) & (bin_indices <= self.nbin)
        bin_indices = bin_indices[valid]
        cl_values = cl_cont[1 : self.lmax]  # Assuming cl_cont starts at ell=0

        cl_sum = np.bincount(bin_indices, weights=cl_values[valid], minlength=self.nbin + 1)
        counts = np.bincount(bin_indices, minlength=self.nbin + 1)

        with np.errstate(divide='ignore', invalid='ignore'):
            cl_disc = np.divide(cl_sum, counts, where=counts > 0)[1:self.nbin + 1]

        return cl_disc
