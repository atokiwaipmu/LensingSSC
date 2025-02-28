
import logging
import numpy as np
from astropy import units as u
from lenstools import ConvergenceMap
import multiprocessing as mp

from core.patch.patch_processor import PatchProcessor

class PatchAnalyser:
    def __init__(self, pp: PatchProcessor, nbin=15, lmin=300, lmax=3000):
        logging.info(f"Initializing FlatPatchAnalyser with nbin={nbin}, lmin={lmin}, lmax={lmax}")
        self.nbin = nbin
        self.lmin, self.lmax = lmin, lmax
        self.bins = np.linspace(-4, 4, self.nbin + 1, endpoint=True)
        self.l_edges = np.logspace(np.log10(self.lmin), np.log10(self.lmax), self.nbin + 1, endpoint=True)
        self.ell = (self.l_edges[1:] + self.l_edges[:-1]) / 2
        self.binwidth = self.bins[1] - self.bins[0]

        self.patch_size = pp.patch_size_deg
        self.xsize = pp.xsize

    def process_patches(self, patches_kappa, patches_snr, num_processes=mp.cpu_count()):
        global_std = np.std(patches_snr)
        with mp.Pool(processes=num_processes) as pool:
            datas = pool.starmap(self._process_patch, zip(patches_kappa, patches_snr, [global_std] * len(patches_kappa)))
        return np.array(datas).astype(np.float32)
    
    def _process_patch(self, patch_pixels, patch_snr_pixels, global_std):
        """
        Processes a single patch, computing various statistics (bispectrum, power spectrum, peak counts, etc.).
        """
        # Process kappa (convergence) map
        conv_map = ConvergenceMap(patch_pixels, angle=self.patch_size * u.deg)
        equilateral, isosceles, squeezed = self._compute_bispectrum(conv_map)
        clkk = self._compute_power_spectrum(conv_map)
        sk0, sk1, sk2, kur0, kur1, kur2, kur3 = self._compute_moments(conv_map, global_std)
        
        # Process SNR map
        snr_map = ConvergenceMap(patch_snr_pixels/global_std, angle=self.patch_size * u.deg)
        pdf_vals = self._compute_pdf(snr_map)
        peaks = self._compute_peak_statistics(snr_map, is_minima=False)
        minima = self._compute_peak_statistics(snr_map, is_minima=True)
        v0,v1,v2 = self._compute_minkowski_functionals(snr_map)

        # Flatten all statistics into a single array
        stats = np.hstack([
            equilateral, isosceles, squeezed, clkk,
            pdf_vals, peaks, minima, v0, v1, v2,
            sk0, sk1, sk2, kur0, kur1, kur2, kur3
        ])
        
        return stats

    def _compute_bispectrum(self, conv_map: ConvergenceMap):
        equilateral = conv_map.bispectrum(self.l_edges, configuration='equilateral')[1]
        isosceles = conv_map.bispectrum(self.l_edges, ratio=0.5, configuration='folded')[1]
        squeezed = conv_map.bispectrum(self.l_edges, ratio=0.1, configuration='folded')[1]

        equilateral = np.abs(PatchAnalyser._dimensionless_bispectrum(equilateral, self.ell))
        isosceles = np.abs(PatchAnalyser._dimensionless_bispectrum(isosceles, self.ell))
        squeezed = np.abs(PatchAnalyser._dimensionless_bispectrum(squeezed, self.ell))

        return equilateral, isosceles, squeezed
    
    def _compute_power_spectrum(self, conv_map: ConvergenceMap):
        _, cl = conv_map.powerSpectrum(self.l_edges)

        cl = PatchAnalyser._dimensionless_cl(cl, self.ell)
        return cl
    
    def _compute_pdf(self, snr_map: ConvergenceMap):
        _, pdf_vals = snr_map.pdf(self.bins)
        return pdf_vals
    
    def _compute_peak_statistics(self, snr_map: ConvergenceMap, is_minima=False):
        if is_minima:
            # Invert the map for minima computation
            snr_map = ConvergenceMap(-snr_map.data, angle=self.patch_size * u.deg)

        height, positions = snr_map.locatePeaks(self.bins)
        height, positions = self._exclude_edges(height, positions)
        peaks = np.histogram(height, bins=self.bins)[0]
        peaks = peaks / np.sum(peaks) / self.binwidth
        return peaks
    
    def _compute_minkowski_functionals(self, snr_map: ConvergenceMap):
        _, v0,v1,v2 = snr_map.minkowskiFunctionals(self.bins)
        return v0, v1, v2
    
    def _compute_moments(self, snr_map: ConvergenceMap, global_std: float):
        moments = snr_map.moments()
        sk0, sk1, sk2, kur0, kur1, kur2, kur3 = self._dimensionless_moments(moments, global_std)
        return sk0, sk1, sk2, kur0, kur1, kur2, kur3
    
    def _exclude_edges(self, heights, positions):
        """
        Excludes edge values from the peak or minima positions to avoid boundary issues.
        """
        # Scale positions to the patch size and apply boundary mask
        tmp_positions = positions.value * self.xsize / self.patch_size
        mask = (tmp_positions[:, 0] > 0) & (tmp_positions[:, 0] < self.xsize - 1) & \
               (tmp_positions[:, 1] > 0) & (tmp_positions[:, 1] < self.xsize - 1)
        return heights[mask], tmp_positions[mask].astype(int)
    
    def _dimensionless_moments(self, moments, global_std):
        """
        Convert the raw moments to dimensionless form.
        taken from lenstools.ConvergenceMap.moments
        """
        _,sigma1,S0,S1,S2,K0,K1,K2,K3 = moments
        sigma0 = global_std

        S0 /= sigma0**3
        S1 /= (sigma0 * sigma1**2)
        S2 *= (sigma0 / sigma1**4)

        K0 /= sigma0**4
        K1 /= (sigma0**2 * sigma1**2)
        K2 /= sigma1**4
        K3 /= sigma1**4

        return S0,S1,S2,K0,K1,K2,K3
    
    @staticmethod
    def _dimensionless_cl(cl, ell):
        return ell * (ell+1) * cl / (2*np.pi)
    
    @staticmethod
    def _dimensionless_bispectrum(bispec, ell):
        return bispec * ell**4 / (2*np.pi)**2
        