import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os

nside = 32
if not os.path.exists('/lustre/work/akira.tokiwa/Projects/LensingSSC/data/kappa_test.fits'):
    print('Creating kappa_test.fits')
    # Load your HEALPIX map (example with nside=8192)
    healpix_map = hp.read_map('/lustre/work/akira.tokiwa/Projects/LensingSSC/results/tiled/data/kappa_zs0.5_smoothed_s5.fits')

    # downsample the map
    healpix_map = hp.ud_grade(healpix_map, nside_out=nside)
    hp.write_map('/lustre/work/akira.tokiwa/Projects/LensingSSC/data/kappa_test.fits', healpix_map, overwrite=True)
else:
    healpix_map = hp.read_map('/lustre/work/akira.tokiwa/Projects/LensingSSC/data/kappa_test.fits')

healpix_map = hp.ma(healpix_map)

def find_extrema(kappa_map, minima=False, lonlat=False):
    """Find extrema in a smoothed masked healpix map."""
    nside = hp.get_nside(kappa_map)
    ipix = np.arange(hp.nside2npix(nside))
    neighbours = hp.get_all_neighbours(nside, ipix)
    
    neighbour_vals = kappa_map.data[neighbours.T]
    pixel_val = kappa_map.data[ipix]

    if minima:
        extrema = np.all(np.tile(pixel_val, [8, 1]).T < neighbour_vals, axis=-1)
    else:
        extrema = np.all(np.tile(pixel_val, [8, 1]).T > neighbour_vals, axis=-1)

    print(f'number of {"minima" if minima else "peaks"} identified: {np.where(extrema)[0].shape[0]}')
    
    extrema_pos = np.asarray(hp.pix2ang(nside, ipix[extrema], lonlat=lonlat)).T
    extrema_amp = kappa_map[ipix][extrema].data
    
    return extrema_pos, extrema_amp

if os.path.exists('peaks_minima.npz'): 
    saved_peaks_minima = np.load('peaks_minima.npz', allow_pickle=True) 
    peaks, minima = saved_peaks_minima['peaks'], saved_peaks_minima['minima']
    peak_amps, minima_amps = saved_peaks_minima['peak_amps'], saved_peaks_minima['minima_amps']
else:
    peaks, peak_amps = find_extrema(healpix_map, minima=False, lonlat=True)
    minima, minima_amps = find_extrema(healpix_map, minima=True, lonlat=True)
    np.savez('peaks_minima.npz', peaks=peaks, minima=minima, peak_amps=peak_amps, minima_amps=minima_amps)

fig = plt.figure(figsize=(12, 6))
hp.orthview(healpix_map.data, title='Kappa Map with Peaks and Minima', half_sky=True, fig=fig)
hp.projscatter(peaks.T,marker = '.',s=0.5,color='red',lonlat=True, label='Peaks')
hp.projscatter(minima.T,marker = '.',s=0.5,color='yellow',lonlat=True, label='Minima')
hp.graticule()

fig.savefig('kappa_map_peaks_minima.png', bbox_inches='tight')
