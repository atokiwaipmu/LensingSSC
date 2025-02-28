## From https://github.com/LSSTDESC/HOS-Y1-prep
from typing import Tuple, Optional
import numpy as np
import healpy as hp
import multiprocessing as mp
from numpy.typing import NDArray

def find_extrema_worker(
    pixel_values: NDArray[np.float64], 
    neighbor_values: NDArray[np.float64]
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Compare each pixel with its neighbors to find local extrema.

    Args:
        pixel_values: Values of the pixels being checked
        neighbor_values: Values of the 8 neighboring pixels for each pixel

    Returns:
        Tuple containing:
            - peaks: Boolean mask indicating peaks
            - minima: Boolean mask indicating minima
    """
    N_NEIGHBORS = 8
    
    # Reshape pixel values for comparison
    pixel_matrix = np.tile(pixel_values, (N_NEIGHBORS, 1)).T
    
    # Find peaks and minima
    peaks = np.all(pixel_matrix > neighbor_values, axis=-1)
    minima = np.all(pixel_matrix < neighbor_values, axis=-1)
    
    return peaks, minima

class ExtremaFinder:
    """Finds local extrema (peaks and minima) in a HEALPix map.
    
    This class efficiently detects local maxima (peaks) and minima in a given
    HEALPix map using parallel processing capabilities.
    """

    def __init__(self, nside: int = 8192) -> None:
        """Initialize the ExtremaFinder.

        Args:
            nside: HEALPix resolution parameter. Must be a power of 2.

        Raises:
            ValueError: If nside is not a power of 2.
        """
        if not hp.isnsideok(nside):
            raise ValueError("nside must be a power of 2")
        
        self.nside: int = nside
        self.npix: int = hp.nside2npix(self.nside)
        self.neighbours: Optional[NDArray] = None
        self.ipix: Optional[NDArray] = None

    def _initialize_neighbours(self) -> None:
        """Lazily initialize neighbor indices.
        
        Computes and stores the indices of 8 neighboring pixels for each pixel.
        For large maps, this initialization is delayed until needed.
        """
        if self.neighbours is None:
            self.ipix = np.arange(self.npix)
            self.neighbours = hp.get_all_neighbours(self.nside, self.ipix)

    def find_extrema(
        self, 
        kappa_map: NDArray[np.float64], 
        num_processes: int = mp.cpu_count()
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Find extrema in the kappa map.

        Args:
            kappa_map: Input kappa map with shape (npix,)
            num_processes: Number of processes for parallel computation

        Returns:
            Tuple[NDArray, NDArray, NDArray, NDArray]: Tuple containing:
                - peak_pos: Positions of peaks (theta, phi)
                - peak_amp: Amplitudes of peaks
                - minima_pos: Positions of minima (theta, phi)
                - minima_amp: Amplitudes of minima

        Raises:
            ValueError: If kappa_map size doesn't match npix
        """
        if len(kappa_map) != self.npix:
            raise ValueError(f"kappa_map size ({len(kappa_map)}) doesn't match npix ({self.npix})")

        self._initialize_neighbours()

        # Split map for parallel processing
        kappa_map_chunks = np.array_split(kappa_map, num_processes)
        neighbours_chunks = np.array_split(kappa_map[self.neighbours.T], num_processes)

        # Find extrema in parallel
        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(
                find_extrema_worker,
                [(kappa_map_chunks[i], neighbours_chunks[i]) for i in range(num_processes)]
            )

        # Combine results and convert coordinates
        peaks_chunks, minima_chunks = zip(*results)
        peaks = np.concatenate(peaks_chunks)
        minima = np.concatenate(minima_chunks)

        peak_pos = np.asarray(hp.pix2ang(self.nside, self.ipix[peaks], lonlat=False)).T
        peak_amp = kappa_map[self.ipix[peaks]]

        minima_pos = np.asarray(hp.pix2ang(self.nside, self.ipix[minima], lonlat=False)).T
        minima_amp = kappa_map[self.ipix[minima]]

        return peak_pos, peak_amp, minima_pos, minima_amp
    
