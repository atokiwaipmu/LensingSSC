## From https://github.com/LSSTDESC/HOS-Y1-prep
import numpy as np
import healpy as hp
import multiprocessing as mp
from multiprocessing import shared_memory

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

        def find_extrema_worker(shm_name, shape, dtype, chunk_indices):
            shm = shared_memory.SharedMemory(name=shm_name)
            kappa_map_shared = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            
            pixel_val = kappa_map_shared[self.ipix[chunk_indices]]
            neighbour_vals = kappa_map_shared[self.neighbours.T[:, chunk_indices]]
            
            peaks = np.all(np.tile(pixel_val, (8, 1)).T > neighbour_vals, axis=-1)
            minima = np.all(np.tile(pixel_val, (8, 1)).T < neighbour_vals, axis=-1)
            
            shm.close()
            return peaks, minima

        # Create shared memory for the kappa map
        shm = shared_memory.SharedMemory(create=True, size=kappa_map.nbytes)
        kappa_map_shared = np.ndarray(kappa_map.shape, dtype=kappa_map.dtype, buffer=shm.buf)
        np.copyto(kappa_map_shared, kappa_map)

        try:
            chunks = np.array_split(range(self.npix), self.npix // 10000)
            with mp.Pool(processes=num_processes) as pool:
                results = pool.starmap(
                    find_extrema_worker,
                    [(shm.name, kappa_map.shape, kappa_map.dtype, chunk) for chunk in chunks]
                )
        finally:
            # Clean up shared memory
            shm.close()
            shm.unlink()

        peaks_chunks, minima_chunks = zip(*results)
        peaks = np.concatenate(peaks_chunks)
        minima = np.concatenate(minima_chunks)

        peak_pos = np.asarray(hp.pix2ang(self.nside, self.ipix[peaks], lonlat=False)).T
        peak_amp = kappa_map[self.ipix[peaks]]

        minima_pos = np.asarray(hp.pix2ang(self.nside, self.ipix[minima], lonlat=False)).T
        minima_amp = kappa_map[self.ipix[minima]]

        return peak_pos, peak_amp, minima_pos, minima_amp