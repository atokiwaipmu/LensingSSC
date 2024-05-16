## From https://github.com/LSSTDESC/HOS-Y1-prep

import os
import numpy as np
import healpy as hp
from healpy import anafast
# Uncomment the following line if healsparse is needed
# import healsparse

from .aperture_mass_computer import measureMap3FromKappa
from .Peaks_minima import find_extrema

class KappaMaps:
    """
    A class to handle operations on kappa maps including reading and smoothing.
    
    Attributes:
        nside (int): The nside of the map, which defines the map's resolution.
        filenames (list): List of paths to map files.
        mapbins (list): Container for maps loaded from filenames.
    """
    
    def __init__(self, filenames, nside):
        if not isinstance(nside, int) or nside <= 0:
            raise ValueError("nside must be a positive integer.")
        self.nside = nside
        self.filenames = filenames
        self.mapbins = []

    def readmaps_npy(self):
        """Read maps from .npy files."""
        try:
            self.mapbins = [np.load(l) for l in self.filenames]
        except Exception as e:
            raise IOError(f"Failed to read .npy files: {e}")

    def readmaps_healpy(self, n2r=False):
        """Read maps using healpy's read_map function."""
        try:
            if n2r:
                self.mapbins = [hp.reorder(hp.read_map(l), n2r=True) for l in self.filenames]
            else:
                self.mapbins = [hp.read_map(l) for l in self.filenames]
        except Exception as e:
            raise IOError(f"Failed to read maps with healpy: {e}")
        
    def smoothing(self, mapbin, sl, is_map=True):
        """
        Apply Gaussian smoothing to a map.

        Parameters:
            mapbin (int or array): Index of the map in mapbins or the map array itself.
            sl (float): Smoothing length in arcminutes.
            is_map (bool): Flag to indicate if mapbin is a map array or an index.
        
        Returns:
            ndarray: The smoothed map.
        """
        if not is_map:
            mapbin = self.mapbins[mapbin]
        
        sl_rad = sl / 60 / 180 * np.pi
        kappa_masked = hp.ma(mapbin)
        try:
            smoothed_map = hp.smoothing(kappa_masked, sigma=sl_rad, verbose=False)
            return smoothed_map
        except Exception as e:
            raise RuntimeError(f"Error during smoothing: {e}")

class KappaCodes(KappaMaps):
    """
    A subclass of KappaMaps to handle additional operations like map analysis and statistics.
    """
    
    def __init__(self, dir_results, filenames, nside, lmax=5000):
        super().__init__(filenames, nside)
        self.Nmaps = len(self.filenames)
        self.dir_results = os.path.abspath(dir_results)
        os.makedirs(self.dir_results, exist_ok=True)
        self.lmax = lmax

    def run_Clkk(self, Nmap1):
        """
        Calculate angular power spectra of the maps.
        """
        map1 = self.mapbins[Nmap1]
        Cl = anafast(map1=map1, lmax=self.lmax)
        
        fn_header = os.path.basename(self.filenames[Nmap1]).split('.')[0]
        dir_Clkk = os.path.join(self.dir_results, 'Clkk')
        os.makedirs(dir_Clkk, exist_ok=True)
        
        fn_out = os.path.join(dir_Clkk, f'{fn_header}_Clkk_ell_0_{self.lmax}.npz')
        
        np.savez(fn_out, ell=np.arange(self.lmax + 1), Cl=Cl)
        return Cl
        
    def run_map2alm(self, Nmap1, Nmap2=None, is_cross=False):
        """
        Calculate angular power spectra of the maps.
        """
        map1 = self.mapbins[Nmap1]
        map2 = self.mapbins[Nmap2] if is_cross else None
        Cl = anafast(map1=map1, map2=map2, lmax=5000, iter=1, alm=False, pol=False, use_weights=False, gal_cut=0)
        Cl *= 8.0
        
        fn_header = os.path.basename(self.filenames[Nmap1]).split('.')[0]
        dir_map2 = os.path.join(self.dir_results, 'map2')
        os.makedirs(dir_map2, exist_ok=True)
        
        suffix = f'_Nmap{Nmap1 + 1}_{Nmap2 + 1}' if is_cross else f'_Nmap{Nmap1 + 1}'
        fn_out = os.path.join(dir_map2, f'{fn_header}{suffix}_map2_Cell_ell_0_5000.dat')
        
        np.savetxt(fn_out, Cl)
        return Cl

    def run_map3(self, Nmap1, thetas, is_cross=False):
        """
        Perform third-order map statistics.
        """
        fn_header = os.path.basename(self.filenames[Nmap1]).split('.')[0]
        dir_map3 = os.path.join(self.dir_results, 'map3')
        os.makedirs(dir_map3, exist_ok=True)
        
        fn_out = os.path.join(dir_map3, f'{fn_header}_map3_DV_thetas.dat')
        measureMap3FromKappa(self.mapbins[Nmap1], thetas=thetas, nside=self.nside, fn_out=fn_out, verbose=False, doPlots=False)
        
        results_map3 = np.loadtxt(fn_out)
        return results_map3

    def run_PDFPeaksMinima(self, map1_smooth, Nmap1, map2_smooth=None, Nmap2=None, is_cross=False):
        """
        Calculate the PDF, peaks, and minima for a smoothed map.
        """
        # Merge maps if cross-analysis is enabled
        if is_cross and map2_smooth is not None:
            map1_smooth = np.append(map1_smooth, map2_smooth, axis=0)
            
        # Calculate histogram
        bins = np.linspace(-0.1 - 0.001, 0.1 + 0.001, 201)
        counts_smooth, _ = np.histogram(map1_smooth, density=True, bins=bins)

        map1_smooth_ma = hp.ma(map1_smooth)
        peak_pos, peak_amp = find_extrema(map1_smooth_ma, lonlat=True)
        minima_pos, minima_amp = find_extrema(map1_smooth_ma, minima=True, lonlat=True)
        
        peaks = np.vstack([peak_pos.T, peak_amp]).T
        minima = np.vstack([minima_pos.T, minima_amp]).T
        
        fn_header = os.path.basename(self.filenames[Nmap1]).split('.')[0]
        dir_PDF = os.path.join(self.dir_results, 'PDF')
        dir_peaks = os.path.join(self.dir_results, 'peaks')
        dir_minima = os.path.join(self.dir_results, 'minima')
        
        os.makedirs(dir_PDF, exist_ok=True)
        os.makedirs(dir_peaks, exist_ok=True)
        os.makedirs(dir_minima, exist_ok=True)

        fn_out_counts = os.path.join(dir_PDF, f'{fn_header}_Nmap{Nmap1 + 1}_Counts_kappa_width0.1_200Kappabins.dat')
        fn_out_minima = os.path.join(dir_peaks, f'{fn_header}_Nmap{Nmap1 + 1}_minima_posRADEC_amp.dat')
        fn_out_peaks = os.path.join(dir_minima, f'{fn_header}_Nmap{Nmap1 + 1}_peaks_posRADEC_amp.dat')

        np.savetxt(fn_out_counts, counts_smooth)
        np.savetxt(fn_out_minima, minima)
        np.savetxt(fn_out_peaks, peaks)

        return counts_smooth, peaks, minima

