
from distutils import dir_util
import os
import logging

import numpy as np
from astropy import units as u
from lenstools import ConvergenceMap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeakLensingAnalysis:
    def __init__(self, dir_save, fname, patch, angle, lmax=3000, lmin=300, nbin=15, xsize=1024, save=True):
        self.angle = angle
        self.lmax = lmax
        self.lmin = lmin
        self.nbin = nbin
        self.xsize = xsize

        self.dir_save = dir_save
        self.fname = fname
        self.save = save

        self.l_edges = np.linspace(self.lmin, self.lmax, self.nbin, endpoint=True)
        self.convergence_map = ConvergenceMap(patch, angle=angle * u.deg)
        self.conv_map_minus = ConvergenceMap(-patch, angle=self.angle * u.deg)
    
    def exclude_edges(self, heights, positions, return_index=True):
        tmp_positions = positions.value * self.xsize / self.angle
        mask = (tmp_positions[:, 0] > 0) & (tmp_positions[:, 0] < self.xsize-1) & (tmp_positions[:, 1] > 0) & (tmp_positions[:, 1] < self.xsize-1)
        if return_index:
            return heights[mask], tmp_positions[mask].astype(int)
        else:
            return heights[mask], positions[mask]
    
    def compute_bispectrum(self):
        ell, equilateral = self.convergence_map.bispectrum(self.l_edges, configuration='equilateral')
        _, halfed = self.convergence_map.bispectrum(self.l_edges, ratio=0.5, configuration='folded')
        _, squeezed = self.convergence_map.bispectrum(self.l_edges, ratio=0.001, configuration='folded')
        if self.save:
            dir_save_bispectrum = os.path.join(self.dir_save, "bispectrum")
            os.makedirs(dir_save_bispectrum, exist_ok=True)
            save_filename = os.path.join(dir_save_bispectrum, self.fname.replace('.npy', f'_bispectrum_ell_{self.lmin}_{self.lmax}.npz'))
            np.savez(save_filename, ell=ell, equilateral=equilateral, halfed=halfed, squeezed=squeezed, lmin=self.lmin, lmax=self.lmax)
            logging.info(f"Saved the results to {save_filename}")
        return ell, equilateral, halfed, squeezed
    
    def compute_power_spectrum(self):
        ell, cl = self.convergence_map.powerSpectrum(self.l_edges)
        if self.save:
            dir_save_cl = os.path.join(self.dir_save, "Clkk")
            os.makedirs(dir_save_cl, exist_ok=True)
            save_filename = os.path.join(dir_save_cl, self.fname.replace('.npy', f'_clkk_ell_{self.lmin}_{self.lmax}.npz'))
            np.savez(save_filename, ell=ell, cl=cl, lmin=self.lmin, lmax=self.lmax)
            logging.info(f"Saved the results to {save_filename}")
        return ell, cl
    
    def calculate_pdf(self, bins):
        nu, p = self.convergence_map.pdf(bins)
        if self.save:
            dir_save_pdf = os.path.join(self.dir_save, "pdf")
            os.makedirs(dir_save_pdf, exist_ok=True)
            save_filename = os.path.join(dir_save_pdf, self.fname.replace('.npy', f'_pdf.npz'))
            np.savez(save_filename, nu=nu, p=p)
            logging.info(f"Saved the results to {save_filename}")
        return nu, p
    
    def calculate_peaks(self, peak_bins):
        peak_height, peak_positions = self.convergence_map.locatePeaks(peak_bins)
        peak_height, peak_positions = self.exclude_edges(peak_height, peak_positions)
        if self.save:
            dir_save_peaks = os.path.join(self.dir_save, "peaks")
            os.makedirs(dir_save_peaks, exist_ok=True)
            save_filename = os.path.join(dir_save_peaks, self.fname.replace('.npy', f'_peaks.npz'))
            np.savez(save_filename, peak_height=peak_height, peak_positions=peak_positions)
            logging.info(f"Saved the results to {save_filename}")
        return peak_height, peak_positions
    
    def calculate_minima(self, minima_bins):
        minima_height, minima_positions = self.conv_map_minus.locatePeaks(minima_bins)
        minima_height = -minima_height
        minima_height, minima_positions = self.exclude_edges(minima_height, minima_positions)
        if self.save:
            dir_save_minima = os.path.join(self.dir_save, "minima")
            os.makedirs(dir_save_minima, exist_ok=True)
            save_filename = os.path.join(dir_save_minima, self.fname.replace('.npy', f'_minima.npz'))
            np.savez(save_filename, minima_height=minima_height, minima_positions=minima_positions)
            logging.info(f"Saved the results to {save_filename}")
        return minima_height, minima_positions
    
