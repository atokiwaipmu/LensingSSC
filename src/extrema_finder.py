## From https://github.com/LSSTDESC/HOS-Y1-prep
import logging
import numpy as np
import healpy as hp
import multiprocessing as mp

class ExtremaFinder:
    """
    Finds local extrema (peaks and minima) in a HEALPix map.
    """

    def __init__(self, nside: int = 8192):
        """
        Initializes the ExtremaFinder with the desired HEALPix nside.

        Args:
            nside: HEALPix nside parameter.
        """
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.neighbours = None  # Initialize as None for lazy initialization

    def _initialize_neighbours(self):
        """
        Lazily initializes the neighbor indices for each pixel.
        """
        if self.neighbours is None:
            self.ipix = np.arange(self.npix)
            self.neighbours = hp.get_all_neighbours(self.nside, self.ipix)

    def find_extrema(self, kappa_map: np.ndarray, num_processes: int = mp.cpu_count()) -> tuple:
        """
        Finds local extrema (peaks and minima) in the given kappa map.

        Args:
            kappa_map: The input kappa map.
            num_processes: Number of processes to use for parallelization.

        Returns:
            A tuple containing peak positions, peak amplitudes, minima positions, and minima amplitudes.
        """
        self._initialize_neighbours()

        kappa_map_chunks = np.array_split(kappa_map, num_processes)
        neighbours_chunks = np.array_split(kappa_map[self.neighbours.T], num_processes)

        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(
                find_extrema_worker,
                [(kappa_map_chunks[i], neighbours_chunks[i]) for i in range(num_processes)]
            )

        peaks_chunks, minima_chunks = zip(*results)
        peaks = np.concatenate(peaks_chunks)
        minima = np.concatenate(minima_chunks)

        peak_pos = np.asarray(hp.pix2ang(self.nside, self.ipix[peaks], lonlat=False)).T
        peak_amp = kappa_map[self.ipix[peaks]]

        minima_pos = np.asarray(hp.pix2ang(self.nside, self.ipix[minima], lonlat=False)).T
        minima_amp = kappa_map[self.ipix[minima]]

        return peak_pos, peak_amp, minima_pos, minima_amp
    
def find_extrema_worker(pixel_val, neighbour_vals): 
    peaks = np.all(np.tile(pixel_val, (8, 1)).T > neighbour_vals, axis=-1)
    minima = np.all(np.tile(pixel_val, (8, 1)).T < neighbour_vals, axis=-1)
    return peaks, minima