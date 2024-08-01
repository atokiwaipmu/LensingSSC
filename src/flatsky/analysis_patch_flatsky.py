
import os
import logging
from glob import glob

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
        _, squeezed = self.convergence_map.bispectrum(self.l_edges, ratio=0.1, configuration='folded')
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
            np.savez(save_filename, peak_height=peak_height, peak_positions=peak_positions, bins=peak_bins)
            logging.info(f"Saved the results to {save_filename}")
        return peak_bins, peak_height, peak_positions
    
    def calculate_minima(self, minima_bins):
        minima_height, minima_positions = self.conv_map_minus.locatePeaks(minima_bins)
        minima_height = -minima_height
        minima_height, minima_positions = self.exclude_edges(minima_height, minima_positions)
        if self.save:
            dir_save_minima = os.path.join(self.dir_save, "minima")
            os.makedirs(dir_save_minima, exist_ok=True)
            save_filename = os.path.join(dir_save_minima, self.fname.replace('.npy', f'_minima.npz'))
            np.savez(save_filename, minima_height=minima_height, minima_positions=minima_positions, bins=minima_bins)
            logging.info(f"Saved the results to {save_filename}")
        return minima_bins, minima_height, minima_positions
    
class WeakLensingCovariance:
    def __init__(self, dir_data, save=True):
        self.dir_data = dir_data
        self.save = save

        self.dir_cov = os.path.join(self.dir_data, "covariance")
        os.makedirs(self.dir_cov, exist_ok=True)

    def cov_clkk(self):
        clkk_files = glob(os.path.join(self.dir_data, "Clkk", "*.npz"))
        self.clkks = np.array([np.load(f)['cl'] for f in clkk_files])
        ell = np.load(clkk_files[0])['ell']

        cov_clkk = np.cov(self.clkks, rowvar=False)

        diagonal_terms = np.diag(cov_clkk)
        correlation = cov_clkk / np.sqrt(diagonal_terms[:, None] * diagonal_terms[None, :])

        if self.save:
            save_filename = os.path.join(self.dir_cov, "cov_clkk.npz")
            np.savez(save_filename, ell=ell, cov_clkk=cov_clkk, correlation=correlation, diagonal_terms=diagonal_terms)
            logging.info(f"Saved the results to {save_filename}")
            return

        return ell, self.clkks, cov_clkk, correlation, diagonal_terms
    
    def cov_bispectrum(self):
        bispectrum_files = glob(os.path.join(self.dir_data, "bispectrum", "*.npz"))
        equilateral, halfed, squeezed = [], [], []
        for f in bispectrum_files:
            data = np.load(f)
            equilateral.append(data['equilateral'])
            halfed.append(data['halfed'])
            squeezed.append(data['squeezed'])
            if 'ell' not in locals(): 
                ell = data['ell']
        
        self.equilateral = np.array(equilateral)
        self.halfed = np.array(halfed)
        self.squeezed = np.array(squeezed)

        cov_equilateral = np.cov(self.equilateral, rowvar=False)
        cov_halfed = np.cov(self.halfed, rowvar=False)
        cov_squeezed = np.cov(self.squeezed, rowvar=False)

        diagonal_equilateral = np.diag(cov_equilateral)
        diagonal_halfed = np.diag(cov_halfed)
        diagonal_squeezed = np.diag(cov_squeezed)

        correlation_equilateral = cov_equilateral / np.sqrt(diagonal_equilateral[:, None] * diagonal_equilateral[None, :])
        correlation_halfed = cov_halfed / np.sqrt(diagonal_halfed[:, None] * diagonal_halfed[None, :])
        correlation_squeezed = cov_squeezed / np.sqrt(diagonal_squeezed[:, None] * diagonal_squeezed[None, :])

        if self.save:
            save_filename_equilateral = os.path.join(self.dir_cov, "cov_bispectrum_equilateral.npz")
            save_filename_halfed = os.path.join(self.dir_cov, "cov_bispectrum_halfed.npz")
            save_filename_squeezed = os.path.join(self.dir_cov, "cov_bispectrum_squeezed.npz")

            np.savez(save_filename_equilateral, ell=ell, cov_equilateral=cov_equilateral, correlation_equilateral=correlation_equilateral, diagonal_terms=diagonal_equilateral)
            np.savez(save_filename_halfed, ell=ell, cov_halfed=cov_halfed, correlation_halfed=correlation_halfed, diagonal_terms=diagonal_halfed)
            np.savez(save_filename_squeezed, ell=ell, cov_squeezed=cov_squeezed, correlation_squeezed=correlation_squeezed, diagonal_terms=diagonal_squeezed)

            logging.info(f"Saved the results to {save_filename_equilateral}")
            return

        return ell, self.equilateral, self.halfed, self.squeezed, cov_equilateral, cov_halfed, cov_squeezed, correlation_equilateral, correlation_halfed, correlation_squeezed, diagonal_equilateral, diagonal_halfed, diagonal_squeezed
    
    def cov_pdf(self):
        pdf_files = glob(os.path.join(self.dir_data, "pdf", "*.npz"))
        pdfs = []
        for f in pdf_files:
            data = np.load(f)
            if 'nu' not in locals(): 
                nu = data['nu']
            pdfs.append(data['p'])
        
        self.pdfs = np.array(pdfs)

        cov_pdf = np.cov(self.pdfs, rowvar=False)

        diagonal_terms = np.diag(cov_pdf)

        correlation = cov_pdf / np.sqrt(diagonal_terms[:, None] * diagonal_terms[None, :])

        if self.save:
            save_filename = os.path.join(self.dir_cov, "cov_pdf.npz")
            np.savez(save_filename, cov_pdf=cov_pdf, correlation=correlation, diagonal_terms=diagonal_terms)
            logging.info(f"Saved the results to {save_filename}")
            return
        
        return nu, self.pdfs, cov_pdf, correlation, diagonal_terms


    def cov_peaks(self):
        peak_files = glob(os.path.join(self.dir_data, "peaks", "*.npz"))
        peak_counts = []
        for f in peak_files:
            data = np.load(f)
            if 'bins' not in locals(): 
                bins = data['bins']
            peak_counts.append(np.histogram(data['peak_height'], bins=bins)[0])
        
        self.peaks = np.array(peak_counts)

        cov_peaks = np.cov(self.peaks, rowvar=False)

        diagonal_terms = np.diag(cov_peaks)

        correlation = cov_peaks / np.sqrt(diagonal_terms[:, None] * diagonal_terms[None, :])

        if self.save:
            save_filename = os.path.join(self.dir_cov, "cov_peaks.npz")
            np.savez(save_filename, cov_peaks=cov_peaks, correlation=correlation, diagonal_terms=diagonal_terms)
            logging.info(f"Saved the results to {save_filename}")
            return
        
        return bins, self.peaks, cov_peaks, correlation, diagonal_terms
    
    def cov_minima(self):
        minima_files = glob(os.path.join(self.dir_data, "minima", "*.npz"))
        minima_counts = []
        for f in minima_files:
            data = np.load(f)
            if 'bins' not in locals(): 
                bins = data['bins']
            minima_counts.append(np.histogram(data['minima_height'], bins=bins)[0])
        
        self.minima = np.array(minima_counts)

        cov_minima = np.cov(self.minima, rowvar=False)

        diagonal_terms = np.diag(cov_minima)

        correlation = cov_minima / np.sqrt(diagonal_terms[:, None] * diagonal_terms[None, :])

        if self.save:
            save_filename = os.path.join(self.dir_cov, "cov_minima.npz")
            np.savez(save_filename, cov_minima=cov_minima, correlation=correlation, diagonal_terms=diagonal_terms)
            logging.info(f"Saved the results to {save_filename}")
            return
        
        return bins, self.minima, cov_minima, correlation, diagonal_terms
    
    def cov_full(self):
        # check if all the data is loaded
        if not hasattr(self, 'clkks'):
            self.cov_clkk()
        if not hasattr(self, 'equilateral'):
            self.cov_bispectrum()
        if not hasattr(self, 'pdfs'):
            self.cov_pdf()
        if not hasattr(self, 'peaks'):
            self.cov_peaks()
        if not hasattr(self, 'minima'):
            self.cov_minima()

        # check if the data has the same length
        assert len(self.clkks) == len(self.equilateral) == len(self.halfed) == len(self.squeezed) == len(self.peaks) == len(self.minima)

        data_full = np.hstack([self.clkks, self.peaks, self.minima, self.pdfs, self.equilateral, self.halfed, self.squeezed])
        cov_full = np.cov(data_full, rowvar=False)

        diagonal_terms = np.diag(cov_full)
        correlation = cov_full / np.sqrt(diagonal_terms[:, None] * diagonal_terms[None, :])

        if self.save:
            save_filename = os.path.join(self.dir_cov, "cov_full.npz")
            np.savez(save_filename, cov_full=cov_full, correlation=correlation, diagonal_terms=diagonal_terms)
            logging.info(f"Saved the results to {save_filename}")
            return
        
        return data_full, cov_full, correlation, diagonal_terms
