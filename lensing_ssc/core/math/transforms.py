"""
Mathematical transforms with minimal dependencies.
"""

import numpy as np
from typing import Optional, Tuple, Union, Callable
from scipy import fft
import logging

from ..base.exceptions import StatisticsError


class FourierTransforms:
    """Fourier transform utilities."""
    
    @staticmethod
    def fft_1d(data: np.ndarray, sampling_rate: float = 1.0,
               center_dc: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """1D Fast Fourier Transform."""
        n = len(data)
        
        # Apply FFT
        fft_data = fft.fft(data)
        freqs = fft.fftfreq(n, 1/sampling_rate)
        
        if center_dc:
            fft_data = fft.fftshift(fft_data)
            freqs = fft.fftshift(freqs)
        
        return freqs, fft_data
    
    @staticmethod
    def fft_2d(data: np.ndarray, pixel_scale: float = 1.0,
               center_dc: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2D Fast Fourier Transform."""
        ny, nx = data.shape
        
        # Apply 2D FFT
        fft_data = fft.fft2(data)
        
        # Create frequency grids
        kx = fft.fftfreq(nx, pixel_scale)
        ky = fft.fftfreq(ny, pixel_scale)
        
        if center_dc:
            fft_data = fft.fftshift(fft_data)
            kx = fft.fftshift(kx)
            ky = fft.fftshift(ky)
        
        return kx, ky, fft_data
    
    @staticmethod
    def power_spectrum_2d(data: np.ndarray, pixel_scale: float = 1.0,
                         radial_average: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], 
                                                             Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Calculate 2D power spectrum."""
        ny, nx = data.shape
        
        # Remove mean
        data_centered = data - np.mean(data)
        
        # Calculate FFT
        kx, ky, fft_data = FourierTransforms.fft_2d(data_centered, pixel_scale, center_dc=True)
        
        # Power spectrum
        power_2d = np.abs(fft_data)**2 / (nx * ny * pixel_scale**2)
        
        if not radial_average:
            return kx, ky, power_2d
        
        # Radial averaging
        kx_2d, ky_2d = np.meshgrid(kx, ky)
        k_radial = np.sqrt(kx_2d**2 + ky_2d**2)
        
        # Define radial bins
        k_max = np.max(k_radial)
        k_bins = np.linspace(0, k_max, min(nx, ny) // 2)
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        
        # Radially average
        power_1d = np.zeros(len(k_centers))
        for i, k_center in enumerate(k_centers):
            mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i+1])
            if np.any(mask):
                power_1d[i] = np.mean(power_2d[mask])
        
        return k_centers, power_1d
    
    @staticmethod
    def cross_power_spectrum(data1: np.ndarray, data2: np.ndarray,
                           pixel_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate cross power spectrum between two 2D fields."""
        if data1.shape != data2.shape:
            raise StatisticsError("Input arrays must have same shape")
        
        ny, nx = data1.shape
        
        # Remove means
        data1_centered = data1 - np.mean(data1)
        data2_centered = data2 - np.mean(data2)
        
        # Calculate FFTs
        _, _, fft1 = FourierTransforms.fft_2d(data1_centered, pixel_scale, center_dc=True)
        kx, ky, fft2 = FourierTransforms.fft_2d(data2_centered, pixel_scale, center_dc=True)
        
        # Cross power spectrum
        cross_power_2d = (fft1 * np.conj(fft2)).real / (nx * ny * pixel_scale**2)
        
        # Radial averaging
        kx_2d, ky_2d = np.meshgrid(kx, ky)
        k_radial = np.sqrt(kx_2d**2 + ky_2d**2)
        
        k_max = np.max(k_radial)
        k_bins = np.linspace(0, k_max, min(nx, ny) // 2)
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        
        cross_power_1d = np.zeros(len(k_centers))
        for i, k_center in enumerate(k_centers):
            mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i+1])
            if np.any(mask):
                cross_power_1d[i] = np.mean(cross_power_2d[mask])
        
        return k_centers, cross_power_1d


class SphericalHarmonics:
    """Spherical harmonics utilities (basic implementation)."""
    
    @staticmethod
    def legendre_polynomial(l: int, x: np.ndarray) -> np.ndarray:
        """Calculate Legendre polynomial P_l(x)."""
        if l == 0:
            return np.ones_like(x)
        elif l == 1:
            return x
        else:
            # Recurrence relation: (l+1)P_{l+1} = (2l+1)xP_l - lP_{l-1}
            P_lminus1 = np.ones_like(x)
            P_l = x
            
            for i in range(2, l + 1):
                P_lplus1 = ((2*i - 1) * x * P_l - (i - 1) * P_lminus1) / i
                P_lminus1, P_l = P_l, P_lplus1
            
            return P_l
    
    @staticmethod
    def associated_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
        """Calculate associated Legendre polynomial P_l^m(x)."""
        from scipy.special import lpmv
        return lpmv(m, l, x)
    
    @staticmethod
    def spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Calculate spherical harmonic Y_l^m(theta, phi)."""
        from scipy.special import sph_harm
        # Note: scipy uses (m, l, phi, theta) convention
        return sph_harm(m, l, phi, theta)
    
    @staticmethod
    def power_spectrum_estimator(alm: np.ndarray, lmax: int) -> np.ndarray:
        """Estimate power spectrum from spherical harmonic coefficients."""
        cl = np.zeros(lmax + 1)
        
        for l in range(lmax + 1):
            power = 0.0
            count = 0
            
            for m in range(-l, l + 1):
                idx = l * (l + 1) + m  # Standard indexing
                if idx < len(alm):
                    if m == 0:
                        power += np.abs(alm[idx])**2
                    else:
                        power += 2 * np.abs(alm[idx])**2  # Account for m and -m
                    count += 1 if m == 0 else 2
            
            if count > 0:
                cl[l] = power / count
        
        return cl


class WindowFunctions:
    """Window functions for data analysis."""
    
    @staticmethod
    def top_hat(x: np.ndarray, width: float) -> np.ndarray:
        """Top-hat window function."""
        return np.where(np.abs(x) <= width/2, 1.0, 0.0)
    
    @staticmethod
    def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian window function."""
        return np.exp(-0.5 * (x/sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    
    @staticmethod
    def cosine_bell(x: np.ndarray, width: float, roll_off: float = 0.1) -> np.ndarray:
        """Cosine bell (Tukey) window function."""
        half_width = width / 2
        roll_width = roll_off * width / 2
        
        window = np.zeros_like(x)
        
        # Central region
        center_mask = np.abs(x) <= (half_width - roll_width)
        window[center_mask] = 1.0
        
        # Roll-off regions
        for sign in [-1, 1]:
            roll_mask = (sign * x > half_width - roll_width) & (sign * x <= half_width)
            if np.any(roll_mask):
                arg = np.pi * (sign * x[roll_mask] - (half_width - roll_width)) / roll_width
                window[roll_mask] = 0.5 * (1 + np.cos(arg))
        
        return window
    
    @staticmethod
    def apply_apodization(data: np.ndarray, window_func: Callable,
                         boundary_fraction: float = 0.1) -> np.ndarray:
        """Apply apodization window to 2D data."""
        ny, nx = data.shape
        
        # Create coordinate grids
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # Distance from edge
        edge_dist = np.minimum(
            np.minimum(1 + X, 1 - X),
            np.minimum(1 + Y, 1 - Y)
        )
        
        # Apply window
        window_2d = window_func(edge_dist / boundary_fraction)
        window_2d = np.clip(window_2d, 0, 1)
        
        return data * window_2d
    
    @staticmethod
    def compensation_factor(window: np.ndarray) -> float:
        """Calculate compensation factor for window function."""
        return len(window) / np.sum(window)


class FilterOperations:
    """Filtering operations for data analysis."""
    
    @staticmethod
    def gaussian_filter_1d(data: np.ndarray, sigma: float) -> np.ndarray:
        """Apply 1D Gaussian filter."""
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(data, sigma)
    
    @staticmethod
    def gaussian_filter_2d(data: np.ndarray, sigma: Union[float, Tuple[float, float]]) -> np.ndarray:
        """Apply 2D Gaussian filter."""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(data, sigma)
    
    @staticmethod
    def median_filter(data: np.ndarray, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """Apply median filter."""
        from scipy.ndimage import median_filter
        return median_filter(data, size)
    
    @staticmethod
    def wiener_filter(data: np.ndarray, noise_power: float,
                     signal_power: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply Wiener filter for denoising."""
        if signal_power is None:
            # Estimate signal power from data
            data_fft = fft.fft2(data)
            signal_power = np.abs(data_fft)**2
        
        # Wiener filter in frequency domain
        wiener_filter_freq = signal_power / (signal_power + noise_power)
        
        # Apply filter
        data_fft = fft.fft2(data)
        filtered_fft = data_fft * wiener_filter_freq
        filtered_data = fft.ifft2(filtered_fft).real
        
        return filtered_data
    
    @staticmethod
    def butterworth_filter(data: np.ndarray, cutoff_freq: float,
                          order: int = 2, filter_type: str = 'low',
                          sampling_rate: float = 1.0) -> np.ndarray:
        """Apply Butterworth filter."""
        from scipy.signal import butter, filtfilt
        
        nyquist = sampling_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        
        b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
        filtered_data = filtfilt(b, a, data)
        
        return filtered_data