## From https://github.com/LSSTDESC/HOS-Y1-prep
import numpy as np
import healpy as hp
import multiprocessing as mp
from typing import Tuple, List

def find_extrema_worker(i: int, pixel_val: np.ndarray, neighbour_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    peaks = np.all(np.tile(pixel_val, (8, 1)).T > neighbour_vals, axis=-1)
    minima = np.all(np.tile(pixel_val, (8, 1)).T < neighbour_vals, axis=-1)
    return peaks, minima

def find_extrema(kappa_map: np.ma.MaskedArray, lonlat: bool = False, nside: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find both peaks and minima in a smoothed masked healpix map.

    Parameters
    ----------
    kappa_map : np.ma.MaskedArray
        Smoothed masked healpix map for which extrema are to be identified.
    lonlat : bool, optional
        If True, return positions in longitude and latitude. Default is False.
    nside : int, optional
        Nside parameter. Default is None.

    Returns
    -------
    peak_pos : np.ndarray
        Peak positions on sphere, theta and phi, in radians.
    peak_amp : np.ndarray
        Peak amplitudes in kappa.
    minima_pos : np.ndarray
        Minima positions on sphere, theta and phi, in radians.
    minima_amp : np.ndarray
        Minima amplitudes in kappa.
    """
    
    nside = hp.get_nside(kappa_map) if nside is None else nside
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)[~kappa_map.mask][0]
    neighbours = hp.get_all_neighbours(nside, ipix)

    neighbour_vals = kappa_map.data[neighbours.T]
    pixel_val = kappa_map.data[ipix]

    neighbours_chunks = np.array_split(neighbour_vals, len(ipix) // 10000)
    pixel_val_chunks = np.array_split(pixel_val, len(ipix) // 10000)

    try:
        """
        with mp.Pool(processes=mp.cpu_count()) as pool:
            extrema_chunks = pool.starmap(
                find_extrema_worker, 
                [(i, pixel_val_chunks[i], neighbours_chunks[i]) for i in range(len(pixel_val_chunks))]
            )
        """
        extrema_chunks = [
            find_extrema_worker(i, pixel_val_chunks[i], neighbours_chunks[i]) 
            for i in range(len(pixel_val_chunks))
        ]
    except Exception as e:
        print(f"An error occurred during multiprocessing: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

    peaks_chunks, minima_chunks = zip(*extrema_chunks)
    peaks = np.concatenate(peaks_chunks)
    minima = np.concatenate(minima_chunks)

    print(f'Number of peaks identified: {np.sum(peaks)}')
    print(f'Number of minima identified: {np.sum(minima)}')

    peak_pos = np.asarray(hp.pix2ang(nside, ipix[peaks], lonlat=lonlat)).T
    peak_amp = kappa_map[ipix][peaks].data

    minima_pos = np.asarray(hp.pix2ang(nside, ipix[minima], lonlat=lonlat)).T
    minima_amp = kappa_map[ipix][minima].data
    
    return peak_pos, peak_amp, minima_pos, minima_amp