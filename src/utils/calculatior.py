from typing import List, Optional
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from numpy.typing import NDArray


#===============================================================================
# Histogram computation
#===============================================================================
@dataclass
class HistogramConfig:
    """Histogram computation configuration."""
    MIN_CHUNK_SIZE: int = 1000
    DEFAULT_PROCESSES: int = mp.cpu_count()

class ParallelHistogramCalculator:
    """Computes histograms in parallel using multiprocessing."""
    
    def __init__(
        self,
        num_processes: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> None:
        self.num_processes = num_processes or HistogramConfig.DEFAULT_PROCESSES
        self.chunk_size = max(chunk_size or HistogramConfig.MIN_CHUNK_SIZE, 1)

    def compute(
        self,
        data: NDArray[np.float64],
        bins: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute histogram in parallel.

        Args:
            data: Input array to histogram
            bins: Bin edges

        Returns:
            NDArray: Computed histogram
        """
        if len(data) == 0:
            raise ValueError("Empty input data")

        data_chunks = self._split_data(data)
        shared_hist = self._initialize_shared_hist(len(bins) - 1)
        
        with mp.Pool(processes=self.num_processes) as pool:
            pool.starmap(
                self._compute_chunk,
                [(chunk, bins, shared_hist) for chunk in data_chunks]
            )

        return np.array(shared_hist)

    def _split_data(self, data: NDArray[np.float64]) -> List[NDArray[np.float64]]:
        """Split data into chunks for parallel processing."""
        num_chunks = max(1, len(data) // self.chunk_size)
        return np.array_split(data, num_chunks)

    @staticmethod
    def _compute_chunk(
        data_chunk: NDArray[np.float64],
        bins: NDArray[np.float64],
        shared_hist: mp.managers.ListProxy,
        lock: mp.synchronize.Lock
    ) -> None:
        """Compute histogram for a single chunk."""
        hist, _ = np.histogram(data_chunk, bins=bins)
        with lock:
            for i, count in enumerate(hist):
                shared_hist[i] += count

    @staticmethod
    def _initialize_shared_hist(size: int) -> mp.managers.ListProxy:
        """Initialize shared histogram array."""
        manager = mp.Manager()
        return manager.list([0] * size)

def normalize_histogram(
    hist: NDArray[np.float64],
    binwidth: float
) -> NDArray[np.float64]:
    """Normalize histogram by area."""
    total = hist.sum() * binwidth
    if total == 0:
        raise ValueError("Cannot normalize histogram with zero total")
    return hist / total

#===============================================================================
# Power spectrum binning
#===============================================================================
@dataclass
class BinningConfig:
    """Power spectrum binning configuration."""
    lmin: int
    lmax: int
    nbin: int

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.lmin < 2:
            raise ValueError(f"lmin must be >= 2, got {self.lmin}")
        if self.lmax <= self.lmin:
            raise ValueError(f"lmax ({self.lmax}) must be > lmin ({self.lmin})")
        if self.nbin < 1:
            raise ValueError(f"nbin must be positive, got {self.nbin}")

class PowerSpectrumBinner:
    """Handles power spectrum binning operations."""

    def __init__(self, config: BinningConfig):
        self.config = config
        self.l_edges = self._create_bin_edges()

    def _create_bin_edges(self) -> NDArray[np.float64]:
        """Create logarithmic bin edges."""
        return np.logspace(
            np.log10(self.config.lmin),
            np.log10(self.config.lmax),
            self.config.nbin + 1
        )

    def _validate_spectrum(self, cl_spectrum: NDArray[np.float64]) -> None:
        """Validate input spectrum."""
        if len(cl_spectrum) < self.config.lmax + 1:
            raise ValueError(
                f"C_l spectrum length ({len(cl_spectrum)}) "
                f"must be >= lmax+1 ({self.config.lmax+1})"
            )

    def discretize(
        self,
        cl_spectrum: NDArray[np.float64],
        multipoles: Optional[NDArray[np.int64]] = None
    ) -> NDArray[np.float64]:
        """Discretize power spectrum into logarithmic bins."""
        self._validate_spectrum(cl_spectrum)
        
        # Prepare multipoles
        ell = (multipoles if multipoles is not None 
               else np.arange(2, self.config.lmax + 1))
        
        # Bin assignment
        bin_indices = np.digitize(ell, self.l_edges, right=True)
        valid_mask = (bin_indices > 0) & (bin_indices <= self.config.nbin)
        
        # Extract valid data
        valid_indices = bin_indices[valid_mask]
        valid_cl = cl_spectrum[2:self.config.lmax + 1][valid_mask]
        
        # Compute statistics
        binned_sum = np.bincount(
            valid_indices,
            weights=valid_cl,
            minlength=self.config.nbin + 1
        )
        bin_counts = np.bincount(
            valid_indices,
            minlength=self.config.nbin + 1
        )
        
        # Safe averaging
        with np.errstate(divide='ignore', invalid='ignore'):
            binned_spectrum = np.divide(
                binned_sum,
                bin_counts,
                where=bin_counts > 0
            )[1:self.config.nbin + 1]
            
        if not np.isfinite(binned_spectrum).all():
            raise ValueError("Non-finite values detected in binned spectrum")
            
        return binned_spectrum
