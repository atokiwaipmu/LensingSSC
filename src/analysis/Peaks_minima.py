## From https://github.com/LSSTDESC/HOS-Y1-prep

import numpy as np
import healpy as hp
import multiprocessing as mp
from tqdm import tqdm

def find_extrema_worker(pixel_val, neighbour_vals, minima=False):
    if minima:
        return np.all(np.tile(pixel_val, [8, 1]).T < neighbour_vals, axis=-1)
    else:
        return np.all(np.tile(pixel_val, [8, 1]).T > neighbour_vals, axis=-1)

def find_extrema(kappa_map, minima=False, lonlat=False):
    """find extrema in a smoothed masked healpix map
       default is to find peaks, finds minima with minima=True
    
       Parameters
       ----------
       kappa_masked_smooth: MaskedArray (healpy object)
           smoothed masked healpix map for which extrema are to be identified
       minima: bool
           if False, find peaks. if True, find minima
       
       Returns
       -------
       extrema_pos: np.ndarray
           extrema positions on sphere, theta and phi, in radians
       extrema_amp: np.ndarray
           extrema amplitudes in kappa
       
    """

    # First create an array of all neighbours for all valid healsparse pixels
    nside = hp.get_nside(kappa_map) # get nside
    ipix = np.arange(hp.nside2npix(nside))[kappa_map.mask == False] # list all pixels and remove masked ones
    neighbours = hp.get_all_neighbours(nside, ipix) # find neighbours for all pixels we care about

    # Get kappa values for each pixel in the neighbour array
    neighbour_vals = kappa_map.data[neighbours.T]
    # Get kappa values for all valid healsparse pixels
    pixel_val = kappa_map.data[ipix]

    # Initialize the progress bar
    with tqdm(total=len(pixel_val), desc="Processing Extrema") as pbar:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Create a callback to update the progress bar
            def update(*a):
                pbar.update()
            
            # Start the parallel computation
            extrema = pool.starmap_async(
                find_extrema_worker, 
                [(pixel_val[i], neighbour_vals[i], minima) for i in range(len(pixel_val))],
                callback=update
            ).get()

    extrema = np.array(extrema)
    # Print the number of extrema identified
    if minima:
        print(f'number of minima identified: {np.where(extrema)[0].shape[0]}')
    else:
        print(f'number of peaks identified: {np.where(extrema)[0].shape[0]}')
        
    extrema_pos = np.asarray(hp.pix2ang(nside, ipix[extrema], lonlat=lonlat)).T # find the extrema positions
    extrema_amp = kappa_map[ipix][extrema].data # find the extrema amplitudes
    
    return extrema_pos, extrema_amp