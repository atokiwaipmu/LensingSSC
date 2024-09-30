## From https://github.com/LSSTDESC/HOS-Y1-prep
import numpy as np
import healpy as hp
import multiprocessing as mp
from tqdm import tqdm

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

        def find_extrema_worker(pixel_val, neighbour_vals):
            peaks = np.all(np.tile(pixel_val, (8, 1)).T > neighbour_vals, axis=-1)
            minima = np.all(np.tile(pixel_val, (8, 1)).T < neighbour_vals, axis=-1)
            return peaks, minima

        neighbour_vals = kappa_map[self.neighbours.T]
        pixel_val = kappa_map[self.ipix]

        neighbours_chunks = np.array_split(neighbour_vals, self.npix // 10000)
        pixel_val_chunks = np.array_split(pixel_val, self.npix // 10000)

        extrema_chunks = [find_extrema_worker(pixel_val_chunks[i], neighbours_chunks[i]) for i in tqdm(range(len(neighbours_chunks)))]

        #chunks = np.array_split(range(self.npix), self.npix // 10000)
        #results = [find_extrema_worker(pixel_val[chunk], neighbour_vals[:, chunk]) for chunk in chunks]
        #with mp.Pool(processes=num_processes) as pool:
        #    results = pool.starmap(find_extrema_worker, [(pixel_val[chunk], neighbour_vals[:, chunk]) for chunk in chunks])

        peaks_chunks, minima_chunks = zip(*extrema_chunks)
        peaks = np.concatenate(peaks_chunks)
        minima = np.concatenate(minima_chunks)

        peak_pos = np.asarray(hp.pix2ang(self.nside, self.ipix[peaks], lonlat=False)).T
        peak_amp = pixel_val[peaks]

        minima_pos = np.asarray(hp.pix2ang(self.nside, self.ipix[minima], lonlat=False)).T
        minima_amp = pixel_val[minima]

        return peak_pos, peak_amp, minima_pos, minima_amp