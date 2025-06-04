"""
Mathematical transforms with minimal dependencies.

This module provides Fourier transforms, spherical harmonics, window functions,
and filtering operations using only numpy and scipy.
"""

import numpy as np
from typing import Optional, Tuple, Union, Callable, List
from scipy import fft, signal
import logging

from ..base.exceptions import StatisticsError


class FourierTransforms:
    """Fourier transform utilities."""
    
    @staticmethod
    def fft_1d(data: np.ndarray, sampling_rate: float = 1.0,
               center_dc: bool = True, detrend: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """1D Fast Fourier Transform.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        sampling_rate : float
            Sampling rate
        center_dc : bool
            Whether to center DC component
        detrend : bool
            Whether to remove linear trend
            
        Returns
        -------
        tuple
            (frequencies, fft_data)
        """
        if detrend:
            # Remove linear trend
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 1)
            data = data - np.polyval(coeffs, x)
        
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
               center_dc: bool = True, detrend: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2D Fast Fourier Transform.
        
        Parameters
        ----------
        data : np.ndarray
            Input 2D data
        pixel_scale : float
            Pixel scale (spacing between samples)
        center_dc : bool
            Whether to center DC component
        detrend : bool
            Whether to remove 2D linear trend
            
        Returns
        -------
        tuple
            (kx, ky, fft_data)
        """
        ny, nx = data.shape
        
        if detrend:
            # Remove 2D linear trend
            y, x = np.mgrid[0:ny, 0:nx]
            A = np.column_stack([np.ones(nx*ny), x.ravel(), y.ravel()])
            coeffs, _, _, _ = np.linalg.lstsq(A, data.ravel(), rcond=None)
            trend = coeffs[0] + coeffs[1] * x + coeffs[2] * y
            data = data - trend
        
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
    def power_spectrum_1d(data: np.ndarray, sampling_rate: float = 1.0,
                         window: Optional[str] = None, detrend: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate 1D power spectrum.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        sampling_rate : float
            Sampling rate
        window : str, optional
            Window function name
        detrend : bool
            Whether to detrend data
            
        Returns
        -------
        tuple
            (frequencies, power_spectrum)
        """
        if detrend:
            # Remove mean and linear trend
            data = data - np.mean(data)
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 1)
            data = data - np.polyval(coeffs, x)
        
        # Apply window function
        if window is not None:
            if window == 'hanning':
                window_func = np.hanning(len(data))
            elif window == 'hamming':
                window_func = np.hamming(len(data))
            elif window == 'blackman':
                window_func = np.blackman(len(data))
            elif window == 'tukey':
                window_func = signal.tukey(len(data))
            else:
                window_func = np.ones(len(data))
            
            data = data * window_func
            window_norm = np.sum(window_func**2)
        else:
            window_norm = len(data)
        
        # Calculate power spectrum
        freqs, fft_data = FourierTransforms.fft_1d(data, sampling_rate, center_dc=False)
        power = np.abs(fft_data)**2 / (sampling_rate * window_norm)
        
        # Return positive frequencies only
        n_pos = len(data) // 2
        return freqs[:n_pos], power[:n_pos]
    
    @staticmethod
    def power_spectrum_2d(data: np.ndarray, pixel_scale: float = 1.0,
                         radial_average: bool = True, k_bins: Optional[np.ndarray] = None) -> Union[Tuple[np.ndarray, np.ndarray], 
                                                                                                    Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Calculate 2D power spectrum.
        
        Parameters
        ----------
        data : np.ndarray
            Input 2D data
        pixel_scale : float
            Pixel scale
        radial_average : bool
            Whether to perform radial averaging
        k_bins : np.ndarray, optional
            Custom k bins for radial averaging
            
        Returns
        -------
        tuple
            If radial_average: (k_centers, power_1d)
            Else: (kx, ky, power_2d)
        """
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
        if k_bins is None:
            k_max = np.max(k_radial)
            n_bins = min(nx, ny) // 2
            k_bins = np.linspace(0, k_max, n_bins)
        
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
                           pixel_scale: float = 1.0, radial_average: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate cross power spectrum between two 2D fields.
        
        Parameters
        ----------
        data1, data2 : np.ndarray
            Input 2D data arrays
        pixel_scale : float
            Pixel scale
        radial_average : bool
            Whether to perform radial averaging
            
        Returns
        -------
        tuple
            (k_centers, cross_power) if radial_average else (kx, ky, cross_power_2d)
        """
        if data1.shape != data2.shape:
            raise StatisticsError("Input arrays must have same shape")
        
        ny, nx = data1.shape
        
        # Remove means
        data1_centered = data1 - np.mean(data1)
        data2_centered = data2 - np.mean(data2)
        
        # Calculate FFTs
        _, _, fft1 = FourierTransforms.fft_2d(data1_centered, pixel_scale, center_dc=True)
        kx, ky, fft2 = FourierTransforms.fft_2d(data2_centered, pixel_scale, center_dc=True)
        
        # Cross power spectrum (real part)
        cross_power_2d = (fft1 * np.conj(fft2)).real / (nx * ny * pixel_scale**2)
        
        if not radial_average:
            return kx, ky, cross_power_2d
        
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
    """Spherical harmonics utilities."""
    
    @staticmethod
    def legendre_polynomial(l: int, x: np.ndarray) -> np.ndarray:
        """Calculate Legendre polynomial P_l(x).
        
        Parameters
        ----------
        l : int
            Degree of polynomial
        x : np.ndarray
            Input values (typically cos(theta))
            
        Returns
        -------
        np.ndarray
            Legendre polynomial values
        """
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
        """Calculate associated Legendre polynomial P_l^m(x).
        
        Parameters
        ----------
        l : int
            Degree
        m : int
            Order
        x : np.ndarray
            Input values
            
        Returns
        -------
        np.ndarray
            Associated Legendre polynomial values
        """
        from scipy.special import lpmv
        return lpmv(m, l, x)
    
    @staticmethod
    def spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Calculate spherical harmonic Y_l^m(theta, phi).
        
        Parameters
        ----------
        l : int
            Degree
        m : int
            Order
        theta : np.ndarray
            Polar angles
        phi : np.ndarray
            Azimuthal angles
            
        Returns
        -------
        np.ndarray
            Spherical harmonic values (complex)
        """
        from scipy.special import sph_harm
        # Note: scipy uses (m, l, phi, theta) convention
        return sph_harm(m, l, phi, theta)
    
    @staticmethod
    def ylm_transform(data: np.ndarray, theta: np.ndarray, phi: np.ndarray,
                     lmax: int) -> np.ndarray:
        """Transform spherical data to spherical harmonic coefficients.
        
        Parameters
        ----------
        data : np.ndarray
            Data on sphere
        theta : np.ndarray
            Polar angles
        phi : np.ndarray
            Azimuthal angles
        lmax : int
            Maximum l value
            
        Returns
        -------
        np.ndarray
            Spherical harmonic coefficients a_lm
        """
        # This is a simplified implementation
        # For full implementation, would need proper integration weights
        n_coeffs = (lmax + 1)**2
        alm = np.zeros(n_coeffs, dtype=complex)
        
        idx = 0
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                ylm = SphericalHarmonics.spherical_harmonic(l, m, theta, phi)
                # Simplified integration (should use proper quadrature)
                alm[idx] = np.mean(data * np.conj(ylm)) * 4 * np.pi
                idx += 1
        
        return alm
    
    @staticmethod
    def power_spectrum_from_alm(alm: np.ndarray, lmax: int) -> np.ndarray:
        """Calculate power spectrum C_l from spherical harmonic coefficients.
        
        Parameters
        ----------
        alm : np.ndarray
            Spherical harmonic coefficients
        lmax : int
            Maximum l value
            
        Returns
        -------
        np.ndarray
            Power spectrum C_l
        """
        cl = np.zeros(lmax + 1)
        
        idx = 0
        for l in range(lmax + 1):
            power = 0.0
            count = 0
            
            for m in range(-l, l + 1):
                if idx < len(alm):
                    if m == 0:
                        power += np.abs(alm[idx])**2
                        count += 1
                    else:
                        power += 2 * np.abs(alm[idx])**2  # Account for m and -m
                        count += 2
                idx += 1
            
            if count > 0:
                cl[l] = power / (2 * l + 1)  # Normalize by degeneracy
        
        return cl


class WindowFunctions:
    """Window functions for data analysis."""
    
    @staticmethod
    def top_hat(x: np.ndarray, width: float, center: float = 0.0) -> np.ndarray:
        """Top-hat window function.
        
        Parameters
        ----------
        x : np.ndarray
            Input coordinates
        width : float
            Window width
        center : float
            Window center
            
        Returns
        -------
        np.ndarray
            Window function values
        """
        return np.where(np.abs(x - center) <= width/2, 1.0, 0.0)
    
    @staticmethod
    def gaussian(x: np.ndarray, sigma: float, center: float = 0.0,
                amplitude: float = 1.0) -> np.ndarray:
        """Gaussian window function.
        
        Parameters
        ----------
        x : np.ndarray
            Input coordinates
        sigma : float
            Standard deviation
        center : float
            Window center
        amplitude : float
            Peak amplitude
            
        Returns
        -------
        np.ndarray
            Window function values
        """
        return amplitude * np.exp(-0.5 * ((x - center)/sigma)**2)
    
    @staticmethod
    def cosine_bell(x: np.ndarray, width: float, roll_off: float = 0.1,
                   center: float = 0.0) -> np.ndarray:
        """Cosine bell (Tukey) window function.
        
        Parameters
        ----------
        x : np.ndarray
            Input coordinates
        width : float
            Window width
        roll_off : float
            Roll-off fraction (0-0.5)
        center : float
            Window center
            
        Returns
        -------
        np.ndarray
            Window function values
        """
        x_centered = x - center
        half_width = width / 2
        roll_width = roll_off * width / 2
        
        window = np.zeros_like(x)
        
        # Central region
        center_mask = np.abs(x_centered) <= (half_width - roll_width)
        window[center_mask] = 1.0
        
        # Roll-off regions
        for sign in [-1, 1]:
            roll_mask = (sign * x_centered > half_width - roll_width) & (sign * x_centered <= half_width)
            if np.any(roll_mask):
                arg = np.pi * (sign * x_centered[roll_mask] - (half_width - roll_width)) / roll_width
                window[roll_mask] = 0.5 * (1 + np.cos(arg))
        
        return window
    
    @staticmethod
    def exponential(x: np.ndarray, scale: float, center: float = 0.0) -> np.ndarray:
        """Exponential window function.
        
        Parameters
        ----------
        x : np.ndarray
            Input coordinates
        scale : float
            Decay scale
        center : float
            Window center
            
        Returns
        -------
        np.ndarray
            Window function values
        """
        return np.exp(-np.abs(x - center) / scale)
    
    @staticmethod
    def apply_apodization_2d(data: np.ndarray, window_func: Callable,
                           boundary_fraction: float = 0.1, **window_kwargs) -> np.ndarray:
        """Apply apodization window to 2D data.
        
        Parameters
        ----------
        data : np.ndarray
            Input 2D data
        window_func : callable
            Window function
        boundary_fraction : float
            Fraction of boundary to apodize
        **window_kwargs
            Additional arguments for window function
            
        Returns
        -------
        np.ndarray
            Apodized data
        """
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
        
        # Apply window based on distance from edge
        window_arg = edge_dist / boundary_fraction
        window_2d = window_func(window_arg, **window_kwargs)
        window_2d = np.clip(window_2d, 0, 1)
        
        return data * window_2d
    
    @staticmethod
    def compensation_factor(window: np.ndarray) -> float:
        """Calculate compensation factor for window function.
        
        Parameters
        ----------
        window : np.ndarray
            Window function values
            
        Returns
        -------
        float
            Compensation factor
        """
        return len(window) / np.sum(window)


class FilterOperations:
    """Filtering operations for data analysis."""
    
    @staticmethod
    def gaussian_filter_1d(data: np.ndarray, sigma: float, mode: str = 'reflect') -> np.ndarray:
        """Apply 1D Gaussian filter.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        sigma : float
            Standard deviation for Gaussian kernel
        mode : str
            Boundary condition mode
            
        Returns
        -------
        np.ndarray
            Filtered data
        """
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(data, sigma, mode=mode)
    
    @staticmethod
    def gaussian_filter_2d(data: np.ndarray, sigma: Union[float, Tuple[float, float]],
                          mode: str = 'reflect') -> np.ndarray:
        """Apply 2D Gaussian filter.
        
        Parameters
        ----------
        data : np.ndarray
            Input 2D data
        sigma : float or tuple
            Standard deviation(s) for Gaussian kernel
        mode : str
            Boundary condition mode
            
        Returns
        -------
        np.ndarray
            Filtered data
        """
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(data, sigma, mode=mode)
    
    @staticmethod
    def median_filter(data: np.ndarray, size: Union[int, Tuple[int, ...]],
                     mode: str = 'reflect') -> np.ndarray:
        """Apply median filter.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        size : int or tuple
            Filter size
        mode : str
            Boundary condition mode
            
        Returns
        -------
        np.ndarray
            Filtered data
        """
        from scipy.ndimage import median_filter
        return median_filter(data, size, mode=mode)
    
    @staticmethod
    def wiener_filter(data: np.ndarray, noise_variance: float,
                     signal_psd: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply Wiener filter for denoising.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        noise_variance : float
            Noise variance
        signal_psd : np.ndarray, optional
            Signal power spectral density
            
        Returns
        -------
        np.ndarray
            Filtered data
        """
        # Take FFT
        data_fft = fft.fft2(data) if data.ndim == 2 else fft.fft(data)
        
        if signal_psd is None:
            # Estimate signal PSD from data
            signal_psd = np.abs(data_fft)**2
        
        # Wiener filter in frequency domain
        wiener_filter_freq = signal_psd / (signal_psd + noise_variance)
        
        # Apply filter
        filtered_fft = data_fft * wiener_filter_freq
        
        # Transform back
        if data.ndim == 2:
            filtered_data = fft.ifft2(filtered_fft).real
        else:
            filtered_data = fft.ifft(filtered_fft).real
        
        return filtered_data
    
    @staticmethod
    def butterworth_filter(data: np.ndarray, cutoff_freq: float,
                          order: int = 2, filter_type: str = 'low',
                          sampling_rate: float = 1.0) -> np.ndarray:
        """Apply Butterworth filter.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (1D)
        cutoff_freq : float
            Cutoff frequency
        order : int
            Filter order
        filter_type : str
            Filter type ('low', 'high', 'band', 'bandstop')
        sampling_rate : float
            Sampling rate
            
        Returns
        -------
        np.ndarray
            Filtered data
        """
        from scipy.signal import butter, filtfilt
        
        nyquist = sampling_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        
        if normal_cutoff >= 1.0:
            logging.warning("Cutoff frequency exceeds Nyquist frequency")
            return data
        
        b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
        filtered_data = filtfilt(b, a, data)
        
        return filtered_data
    
    @staticmethod
    def matched_filter(data: np.ndarray, template: np.ndarray,
                      noise_psd: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply matched filter to detect template in data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        template : np.ndarray
            Template to match
        noise_psd : np.ndarray, optional
            Noise power spectral density
            
        Returns
        -------
        np.ndarray
            Matched filter output
        """
        # Ensure same length
        if len(template) != len(data):
            raise StatisticsError("Data and template must have same length")
        
        # Take FFTs
        data_fft = fft.fft(data)
        template_fft = fft.fft(template)
        
        if noise_psd is None:
            # White noise assumption
            matched_filter_freq = np.conj(template_fft)
        else:
            # Colored noise
            matched_filter_freq = np.conj(template_fft) / noise_psd
        
        # Apply matched filter
        output_fft = data_fft * matched_filter_freq
        output = fft.ifft(output_fft).real
        
        # Normalize
        norm_factor = np.sqrt(np.sum(np.abs(template)**2))
        if norm_factor > 0:
            output = output / norm_factor
        
        return output