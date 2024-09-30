
import logging
import numpy as np

from src.utils import parallel_histogram
from src.extrema_finder import ExtremaFinder

class FullSkyAnalyser:
    """
    Analyzes full-sky maps, computing histograms and finding extrema.

    Attributes:
        nside (int): Healpix resolution parameter.
        nbin (int): Number of bins for histograms.
        lmin (int): Minimum multipole for analysis.
        lmax (int): Maximum multipole for analysis.
        bins (np.ndarray): Array of bin edges for histograms.
        l_edges (np.ndarray): Array of bin edges for discretized C_l.
        binwidth (float): Width of each histogram bin.
        ef (ExtremaFinder): Extrema finder object for peak/minima detection.

    """

    def __init__(self, nside=8192, nbin=15, lmin=300, lmax=3000):
        self.nside = nside
        self.nbin = nbin
        self.lmin, self.lmax = lmin, lmax
        self.bins = np.linspace(-4, 4, self.nbin + 1, endpoint=True)
        self.l_edges = np.logspace(np.log10(self.lmin), np.log10(self.lmax), self.nbin + 1, endpoint=True)
        self.binwidth = self.bins[1] - self.bins[0]

        self.ef = ExtremaFinder(nside=self.nside)

        logging.info(f"FullSkyAnalyser initialized: nside={nside}, nbin={nbin}, lmin={lmin}, lmax={lmax}")

    def process_map(self, snr_map, cl):
        """Processes a full-sky map and computes various statistics.

        Args:
            snr_map (np.ndarray): The signal-to-noise ratio map.
            cl (np.ndarray): The continuous C_l spectrum.

        Returns:
            np.ndarray: A combined array containing discretized C_l, histograms,
                         peak amplitudes, and minima amplitudes.
        """

        cl_disc = self._continuous_to_discrete(cl)
        logging.info(f"Computing PDF...")
        pdf_vals = self._compute_histogram(data=snr_map)

        logging.info(f"Finding extrema...")
        _, peak_amp, _, minima_amp = self.ef.find_extrema(snr_map)
        logging.info(f"Computing histograms for peaks and minima...")
        peaks = self._compute_histogram(data=peak_amp)
        minima = self._compute_histogram(data=minima_amp)

        data_tmp = np.hstack([cl_disc, pdf_vals, peaks, minima])
        return data_tmp
    
    def _compute_histogram(self, data):
        """Computes a normalized histogram using multiprocessing (optional).

        Args:
            data (np.ndarray): The data to be binned.

        Returns:
            np.ndarray: The normalized histogram.
        """
        hist = parallel_histogram(data=data, bins=self.bins)
        return hist / np.sum(hist) / self.binwidth
    
    def _continuous_to_discrete(self, cl_cont):
        """Discretizes a continuous C_l spectrum.

        Args:
            cl_cont (np.ndarray): The continuous C_l spectrum.

        Returns:
            np.ndarray: The discretized C_l spectrum.
        """

        ell_cont = np.arange(2, self.lmax + 1)
        ell_idx = np.digitize(ell_cont, self.l_edges, right=True)
        cl_count = np.bincount(ell_idx, weights=cl_cont[1:-1])
        ell_bincount = np.bincount(ell_idx)
        cl_disc = (cl_count / ell_bincount)[1:]
        return cl_disc
