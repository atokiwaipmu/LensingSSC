from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import logging
from dataclasses import dataclass
from src.analysis.fullsky.calculator import ExtremaFinder

@dataclass
class AnalysisConfig:
    """Analysis configuration parameters"""
    NSIDE_DEFAULT: int = 8192
    NBIN_DEFAULT: int = 15
    LMIN_DEFAULT: int = 300
    LMAX_DEFAULT: int = 3000
    SNR_MIN: float = -4.0
    SNR_MAX: float = 4.0

class FullSkyAnalyser:
    """Analyzes full-sky maps by computing histograms and identifying extrema."""

    def __init__(
        self, 
        nside: int = AnalysisConfig.NSIDE_DEFAULT,
        nbin: int = AnalysisConfig.NBIN_DEFAULT,
        lmin: int = AnalysisConfig.LMIN_DEFAULT,
        lmax: int = AnalysisConfig.LMAX_DEFAULT
    ) -> None:
        """Initialize analyzer with specified parameters."""
        self.nside = nside
        self.nbin = nbin
        self.lmin = lmin
        self.lmax = lmax
        
        # Initialize bin edges
        self.bins = np.linspace(AnalysisConfig.SNR_MIN, AnalysisConfig.SNR_MAX, nbin + 1)
        self.l_edges = np.logspace(np.log10(lmin), np.log10(lmax), nbin + 1)
        self.binwidth = self.bins[1] - self.bins[0]
        
        # Initialize extrema finder
        self.ef = ExtremaFinder(nside=nside)

        self._log_initialization()

    def process_map(
        self, 
        snr_map: NDArray[np.float64], 
        cl: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Process full-sky map and return combined statistics."""
        try:
            results = []
            
            # Calculate discretized C_l
            results.append(self._discretize_cl(cl))
            
            # Calculate SNR PDF
            logging.info("Computing PDF of SNR map...")
            results.append(self._compute_normalized_histogram(snr_map))
            
            # Find and process extrema
            logging.info("Identifying extrema in SNR map...")
            peak_amp, minima_amp = self._find_extrema(snr_map)
            
            # Calculate extrema histograms
            results.append(self._compute_normalized_histogram(peak_amp))
            results.append(self._compute_normalized_histogram(minima_amp))
            
            return np.concatenate(results)
            
        except Exception as e:
            logging.error(f"Error processing map: {str(e)}")
            raise

    def _find_extrema(
        self, 
        snr_map: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extract peak and minima amplitudes from SNR map."""
        _, peak_amp, _, minima_amp = self.ef.find_extrema(snr_map)
        return peak_amp, minima_amp

    def _log_initialization(self) -> None:
        """Log initialization parameters."""
        logging.info(
            f"FullSkyAnalyser initialized with "
            f"nside={self.nside}, nbin={self.nbin}, "
            f"lmin={self.lmin}, lmax={self.lmax}"
        )