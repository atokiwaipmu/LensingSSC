"""
LensTools provider implementation.

This module provides lensing analysis capabilities using lenstools with lazy loading,
comprehensive statistical analysis tools, and efficient computation of lensing
observables including power spectra, bispectra, PDFs, and peak statistics.
"""

import numpy as np
from typing import Tuple, Any, Union, Optional, Dict, List, Callable
from pathlib import Path
import logging

from ..interfaces.data_interface import ConvergenceMapProvider
from ..base.data_structures import MapData, StatisticsData
from ..base.exceptions import ProviderError, StatisticsError
from .base_provider import LazyProvider, CachedProvider


class LenstoolsProvider(LazyProvider, CachedProvider, ConvergenceMapProvider):
    """Provider for lensing analysis using lenstools.
    
    This provider offers:
    - Lazy loading of lenstools and related packages
    - Comprehensive convergence map analysis
    - Statistical analysis tools (power spectra, bispectra, PDFs, peaks)
    - Smoothing and filtering operations
    - Shape measurement and weak lensing statistics
    - Memory-efficient processing with caching
    """
    
    def __init__(self, cache_size: int = 30, cache_ttl: Optional[float] = 3600):
        """Initialize lenstools provider.
        
        Parameters
        ----------
        cache_size : int
            Maximum number of cached results
        cache_ttl : float, optional
            Cache time-to-live in seconds (default: 1 hour)
        """
        LazyProvider.__init__(self)
        CachedProvider.__init__(self, cache_size=cache_size, cache_ttl=cache_ttl)
        
        self._lenstools = None
        self._convergence_map = None
        self._shear_map = None
        self._noise = None
        self._statistics = None
        self._simulations = None
        
        # Analysis configuration
        self._default_analysis_params = {
            'smoothing_scales': [1.0, 2.0, 5.0, 10.0],  # arcmin
            'l_edges': np.logspace(np.log10(300), np.log10(3000), 9),
            'pdf_bins': np.linspace(-0.1, 0.1, 50),
            'peak_thresholds': np.linspace(-3, 5, 50),
        }
    
    @property
    def name(self) -> str:
        """Provider name."""
        return "LenstoolsProvider"
    
    @property
    def version(self) -> str:
        """Provider version."""
        return "1.0.0"
    
    def _check_dependencies(self) -> None:
        """Check if lenstools and related packages are available."""
        try:
            self._lenstools = self._lazy_import('lenstools')
            self._convergence_map = self._get_module_attribute('lenstools', 'ConvergenceMap')
            self._shear_map = self._get_module_attribute('lenstools', 'ShearMap')
            
            # Optional modules
            try:
                self._noise = self._lazy_import('lenstools.noise', 'noise')
            except:
                self._logger.debug("lenstools.noise not available")
            
            try:
                self._statistics = self._lazy_import('lenstools.statistics', 'statistics')
            except:
                self._logger.debug("lenstools.statistics not available")
                
            try:
                self._simulations = self._lazy_import('lenstools.simulations', 'simulations')
            except:
                self._logger.debug("lenstools.simulations not available")
                
        except Exception as e:
            raise ImportError(f"lenstools is required for LenstoolsProvider: {e}")
    
    def _initialize_backend(self, **kwargs) -> None:
        """Initialize lenstools backend."""
        self._check_dependencies()
        
        # Update default analysis parameters
        analysis_params = kwargs.get('analysis_params', {})
        self._default_analysis_params.update(analysis_params)
        
        # Set random seed if provided
        random_seed = kwargs.get('random_seed', None)
        if random_seed is not None:
            np.random.seed(random_seed)
            self._logger.debug(f"Set random seed: {random_seed}")
    
    def _get_backend_info(self) -> Dict[str, Any]:
        """Get backend-specific information."""
        info = super()._get_backend_info()
        
        if self._lenstools is not None:
            info.update({
                'lenstools_version': getattr(self._lenstools, '__version__', 'unknown'),
                'available_modules': {
                    'noise': self._noise is not None,
                    'statistics': self._statistics is not None,
                    'simulations': self._simulations is not None,
                },
                'default_analysis_params': self._default_analysis_params,
            })
        
        return info
    
    def create_convergence_map(self, data: np.ndarray, angle_deg: float, **kwargs) -> Any:
        """Create a convergence map object.
        
        Parameters
        ----------
        data : np.ndarray
            Map data (2D array)
        angle_deg : float
            Map angular size in degrees
        **kwargs
            Additional arguments for ConvergenceMap
            
        Returns
        -------
        lenstools.ConvergenceMap
            Convergence map object
        """
        self.ensure_initialized()
        self._track_usage()
        
        cache_key = f"conv_map_{id(data)}_{angle_deg}_{hash(frozenset(kwargs.items()))}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            # Import astropy units for angle specification
            try:
                from astropy import units as u
                angle = angle_deg * u.deg
            except ImportError:
                # Fallback if astropy not available
                angle = angle_deg
                self._logger.warning("astropy not available, angle may not be handled correctly")
            
            # Additional parameters
            conv_kwargs = {
                'angle': angle,
                'cosmology': kwargs.get('cosmology', None),
                'unit': kwargs.get('unit', None),
                'z': kwargs.get('z', None),
                'filename': kwargs.get('filename', None),
            }
            
            # Remove None values
            conv_kwargs = {k: v for k, v in conv_kwargs.items() if v is not None}
            
            conv_map = self._convergence_map(data, **conv_kwargs)
            
            # Cache the result
            self._cache_set(cache_key, conv_map)
            
            self._logger.debug(f"Created convergence map: {data.shape}, {angle_deg}°")
            return conv_map
            
        except Exception as e:
            raise ProviderError(f"Failed to create convergence map: {e}")
    
    def power_spectrum(self, conv_map: Any, l_edges: np.ndarray = None,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power spectrum.
        
        Parameters
        ----------
        conv_map : lenstools.ConvergenceMap
            Convergence map object
        l_edges : np.ndarray, optional
            Multipole bin edges
        **kwargs
            Additional arguments for powerSpectrum
            
        Returns
        -------
        tuple
            (ell_centers, power_spectrum)
        """
        self.ensure_initialized()
        self._track_usage()
        
        if l_edges is None:
            l_edges = self._default_analysis_params['l_edges']
        
        cache_key = f"power_{id(conv_map)}_{hash(l_edges.tobytes())}_{hash(frozenset(kwargs.items()))}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            power_kwargs = {
                'l_edges': l_edges,
                'scale': kwargs.get('scale', 'log'),
                'method': kwargs.get('method', 'fft'),
            }
            
            # Remove None values
            power_kwargs = {k: v for k, v in power_kwargs.items() if v is not None}
            
            # Calculate power spectrum
            l_centers, power = conv_map.powerSpectrum(**power_kwargs)
            
            result = (l_centers, power)
            self._cache_set(cache_key, result)
            
            self._logger.debug(f"Calculated power spectrum: {len(l_centers)} l-bins")
            return result
            
        except Exception as e:
            raise StatisticsError(f"Failed to calculate power spectrum: {e}")
    
    def bispectrum(self, conv_map: Any, l_edges: np.ndarray = None,
                  configuration: str = "equilateral", **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bispectrum.
        
        Parameters
        ----------
        conv_map : lenstools.ConvergenceMap
            Convergence map object
        l_edges : np.ndarray, optional
            Multipole bin edges
        configuration : str
            Bispectrum configuration ('equilateral', 'squeezed', 'folded')
        **kwargs
            Additional arguments for bispectrum
            
        Returns
        -------
        tuple
            (ell_centers, bispectrum)
        """
        self.ensure_initialized()
        self._track_usage()
        
        if l_edges is None:
            l_edges = self._default_analysis_params['l_edges']
        
        cache_key = f"bispec_{id(conv_map)}_{hash(l_edges.tobytes())}_{configuration}_{hash(frozenset(kwargs.items()))}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            bispectrum_kwargs = {
                'l_edges': l_edges,
                'configuration': configuration,
                'scale': kwargs.get('scale', 'log'),
            }
            
            # Remove None values
            bispectrum_kwargs = {k: v for k, v in bispectrum_kwargs.items() if v is not None}
            
            # Calculate bispectrum
            l_centers, bispectrum = conv_map.bispectrum(**bispectrum_kwargs)
            
            result = (l_centers, bispectrum)
            self._cache_set(cache_key, result)
            
            self._logger.debug(f"Calculated {configuration} bispectrum: {len(l_centers)} l-bins")
            return result
            
        except Exception as e:
            raise StatisticsError(f"Failed to calculate bispectrum: {e}")
    
    def pdf(self, conv_map: Any, bins: np.ndarray = None,
           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate probability density function.
        
        Parameters
        ----------
        conv_map : lenstools.ConvergenceMap
            Convergence map object
        bins : np.ndarray, optional
            Bin edges for PDF
        **kwargs
            Additional arguments for pdf
            
        Returns
        -------
        tuple
            (bin_centers, pdf_values)
        """
        self.ensure_initialized()
        self._track_usage()
        
        if bins is None:
            bins = self._default_analysis_params['pdf_bins']
        
        cache_key = f"pdf_{id(conv_map)}_{hash(bins.tobytes())}_{hash(frozenset(kwargs.items()))}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            pdf_kwargs = {
                'bins': bins,
                'norm': kwargs.get('norm', True),
            }
            
            # Remove None values
            pdf_kwargs = {k: v for k, v in pdf_kwargs.items() if v is not None}
            
            # Calculate PDF
            bin_centers, pdf_values = conv_map.pdf(**pdf_kwargs)
            
            result = (bin_centers, pdf_values)
            self._cache_set(cache_key, result)
            
            self._logger.debug(f"Calculated PDF: {len(bin_centers)} bins")
            return result
            
        except Exception as e:
            raise StatisticsError(f"Failed to calculate PDF: {e}")
    
    def locate_peaks(self, conv_map: Any, threshold: float = 0.0,
                    **kwargs) -> Tuple[np.ndarray, Any]:
        """Locate peaks in the map.
        
        Parameters
        ----------
        conv_map : lenstools.ConvergenceMap
            Convergence map object
        threshold : float
            Peak threshold
        **kwargs
            Additional arguments for locatePeaks
            
        Returns
        -------
        tuple
            (peak_heights, peak_positions)
        """
        self.ensure_initialized()
        self._track_usage()
        
        cache_key = f"peaks_{id(conv_map)}_{threshold}_{hash(frozenset(kwargs.items()))}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            peak_kwargs = {
                'threshold': threshold,
                'norm': kwargs.get('norm', False),
                'save_thresholds': kwargs.get('save_thresholds', True),
            }
            
            # Remove None values
            peak_kwargs = {k: v for k, v in peak_kwargs.items() if v is not None}
            
            # Locate peaks
            peaks = conv_map.locatePeaks(**peak_kwargs)
            
            # Extract peak heights and positions
            if hasattr(peaks, 'kappa_max'):
                peak_heights = peaks.kappa_max
                peak_positions = np.column_stack([peaks.x, peaks.y])
            else:
                # Fallback for different lenstools versions
                peak_heights = peaks[:, 0] if peaks.ndim == 2 else peaks
                peak_positions = peaks[:, 1:3] if peaks.ndim == 2 and peaks.shape[1] >= 3 else None
            
            result = (peak_heights, peak_positions)
            self._cache_set(cache_key, result)
            
            self._logger.debug(f"Located {len(peak_heights)} peaks above threshold {threshold}")
            return result
            
        except Exception as e:
            raise StatisticsError(f"Failed to locate peaks: {e}")
    
    def peak_counts(self, conv_map: Any, threshold_bins: np.ndarray = None,
                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate peak counts as a function of threshold.
        
        Parameters
        ----------
        conv_map : lenstools.ConvergenceMap
            Convergence map object
        threshold_bins : np.ndarray, optional
            Threshold bin edges
        **kwargs
            Additional arguments
            
        Returns
        -------
        tuple
            (thresholds, peak_counts)
        """
        self.ensure_initialized()
        self._track_usage()
        
        if threshold_bins is None:
            threshold_bins = self._default_analysis_params['peak_thresholds']
        
        cache_key = f"peak_counts_{id(conv_map)}_{hash(threshold_bins.tobytes())}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            peak_counts = np.zeros(len(threshold_bins))
            
            # Count peaks for each threshold
            for i, threshold in enumerate(threshold_bins):
                peak_heights, _ = self.locate_peaks(conv_map, threshold=threshold)
                peak_counts[i] = len(peak_heights)
            
            result = (threshold_bins, peak_counts)
            self._cache_set(cache_key, result)
            
            self._logger.debug(f"Calculated peak counts for {len(threshold_bins)} thresholds")
            return result
            
        except Exception as e:
            raise StatisticsError(f"Failed to calculate peak counts: {e}")
    
    def smooth(self, conv_map: Any, scale_arcmin: float, **kwargs) -> Any:
        """Smooth the convergence map.
        
        Parameters
        ----------
        conv_map : lenstools.ConvergenceMap
            Convergence map object
        scale_arcmin : float
            Smoothing scale in arcminutes
        **kwargs
            Additional arguments for smooth
            
        Returns
        -------
        lenstools.ConvergenceMap
            Smoothed convergence map
        """
        self.ensure_initialized()
        self._track_usage()
        
        cache_key = f"smooth_{id(conv_map)}_{scale_arcmin}_{hash(frozenset(kwargs.items()))}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            # Import astropy units
            try:
                from astropy import units as u
                scale = scale_arcmin * u.arcmin
            except ImportError:
                scale = scale_arcmin
                self._logger.warning("astropy not available, scale may not be handled correctly")
            
            smooth_kwargs = {
                'scale': scale,
                'kind': kwargs.get('kind', 'gaussian'),
                'inplace': kwargs.get('inplace', False),
            }
            
            # Remove None values
            smooth_kwargs = {k: v for k, v in smooth_kwargs.items() if v is not None}
            
            # Smooth the map
            smoothed_map = conv_map.smooth(**smooth_kwargs)
            
            self._cache_set(cache_key, smoothed_map)
            
            self._logger.debug(f"Smoothed map with scale {scale_arcmin} arcmin")
            return smoothed_map
            
        except Exception as e:
            raise ProviderError(f"Failed to smooth map: {e}")
    
    def add_noise(self, conv_map: Any, noise_level: float, **kwargs) -> Any:
        """Add noise to convergence map.
        
        Parameters
        ----------
        conv_map : lenstools.ConvergenceMap
            Convergence map object
        noise_level : float
            Noise RMS level
        **kwargs
            Additional noise parameters
            
        Returns
        -------
        lenstools.ConvergenceMap
            Noisy convergence map
        """
        self.ensure_initialized()
        self._track_usage()
        
        try:
            # Create noise map
            noise_map = np.random.normal(0, noise_level, conv_map.data.shape)
            
            # Add noise to map
            noisy_data = conv_map.data + noise_map
            
            # Create new convergence map
            noisy_conv_map = self.create_convergence_map(
                noisy_data, 
                conv_map.side_angle.to('deg').value if hasattr(conv_map, 'side_angle') else 10.0,
                **kwargs
            )
            
            self._logger.debug(f"Added noise with RMS {noise_level}")
            return noisy_conv_map
            
        except Exception as e:
            raise ProviderError(f"Failed to add noise: {e}")
    
    def minkowski_functionals(self, conv_map: Any, threshold_bins: np.ndarray = None,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Minkowski functionals.
        
        Parameters
        ----------
        conv_map : lenstools.ConvergenceMap
            Convergence map object
        threshold_bins : np.ndarray, optional
            Threshold bin edges
        **kwargs
            Additional arguments
            
        Returns
        -------
        tuple
            (thresholds, V0, V1, V2) - Minkowski functionals
        """
        self.ensure_initialized()
        self._track_usage()
        
        if threshold_bins is None:
            threshold_bins = self._default_analysis_params['pdf_bins']
        
        cache_key = f"minkowski_{id(conv_map)}_{hash(threshold_bins.tobytes())}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            # Check if lenstools has Minkowski functionals
            if hasattr(conv_map, 'minkowskiFunctionals'):
                mink_kwargs = {
                    'thresholds': threshold_bins,
                    'norm': kwargs.get('norm', True),
                }
                
                v0, v1, v2 = conv_map.minkowskiFunctionals(**mink_kwargs)
                result = (threshold_bins, v0, v1, v2)
            else:
                # Fallback implementation
                result = self._compute_minkowski_fallback(conv_map, threshold_bins, **kwargs)
            
            self._cache_set(cache_key, result)
            
            self._logger.debug(f"Calculated Minkowski functionals for {len(threshold_bins)} thresholds")
            return result
            
        except Exception as e:
            raise StatisticsError(f"Failed to calculate Minkowski functionals: {e}")
    
    def _compute_minkowski_fallback(self, conv_map: Any, threshold_bins: np.ndarray,
                                  **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fallback implementation for Minkowski functionals."""
        # Simplified implementation
        data = conv_map.data
        v0 = np.zeros(len(threshold_bins))  # Area
        v1 = np.zeros(len(threshold_bins))  # Perimeter
        v2 = np.zeros(len(threshold_bins))  # Genus
        
        for i, threshold in enumerate(threshold_bins):
            # Binary map above threshold
            binary_map = data > threshold
            
            # V0: Area (fraction of pixels above threshold)
            v0[i] = np.sum(binary_map) / binary_map.size
            
            # V1 and V2 would require more sophisticated edge detection
            # For now, set to zero as placeholder
            v1[i] = 0.0
            v2[i] = 0.0
        
        return threshold_bins, v0, v1, v2
    
    def cross_correlation(self, conv_map1: Any, conv_map2: Any, 
                         l_edges: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate cross-correlation power spectrum.
        
        Parameters
        ----------
        conv_map1, conv_map2 : lenstools.ConvergenceMap
            Convergence map objects
        l_edges : np.ndarray, optional
            Multipole bin edges
        **kwargs
            Additional arguments
            
        Returns
        -------
        tuple
            (ell_centers, cross_power)
        """
        self.ensure_initialized()
        self._track_usage()
        
        if l_edges is None:
            l_edges = self._default_analysis_params['l_edges']
        
        cache_key = f"cross_{id(conv_map1)}_{id(conv_map2)}_{hash(l_edges.tobytes())}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            if hasattr(conv_map1, 'cross') and hasattr(conv_map1.cross, 'powerSpectrum'):
                # Use lenstools cross-correlation if available
                cross_kwargs = {
                    'l_edges': l_edges,
                    'scale': kwargs.get('scale', 'log'),
                }
                
                l_centers, cross_power = conv_map1.cross(conv_map2).powerSpectrum(**cross_kwargs)
            else:
                # Fallback: compute cross-power manually via FFT
                l_centers, cross_power = self._compute_cross_power_fft(
                    conv_map1, conv_map2, l_edges, **kwargs
                )
            
            result = (l_centers, cross_power)
            self._cache_set(cache_key, result)
            
            self._logger.debug(f"Calculated cross-correlation: {len(l_centers)} l-bins")
            return result
            
        except Exception as e:
            raise StatisticsError(f"Failed to calculate cross-correlation: {e}")
    
    def _compute_cross_power_fft(self, conv_map1: Any, conv_map2: Any,
                                l_edges: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cross-power spectrum using FFT."""
        try:
            # Get map data
            data1 = conv_map1.data
            data2 = conv_map2.data
            
            if data1.shape != data2.shape:
                raise ValueError("Maps must have the same shape for cross-correlation")
            
            # Compute FFTs
            fft1 = np.fft.fft2(data1)
            fft2 = np.fft.fft2(data2)
            
            # Cross power
            cross_power_2d = (fft1 * np.conj(fft2)).real
            
            # Convert to 1D power spectrum (simplified)
            # This is a basic implementation - full version would need proper k-binning
            ny, nx = data1.shape
            kx = np.fft.fftfreq(nx)
            ky = np.fft.fftfreq(ny)
            kx_2d, ky_2d = np.meshgrid(kx, ky)
            k_2d = np.sqrt(kx_2d**2 + ky_2d**2)
            
            # Simple radial averaging (placeholder)
            l_centers = (l_edges[:-1] + l_edges[1:]) / 2
            cross_power_1d = np.zeros(len(l_centers))
            
            # This is a simplified binning - full implementation would be more sophisticated
            for i in range(len(l_centers)):
                mask = (k_2d >= l_edges[i]) & (k_2d < l_edges[i+1])
                if np.any(mask):
                    cross_power_1d[i] = np.mean(cross_power_2d[mask])
            
            return l_centers, cross_power_1d
            
        except Exception as e:
            raise StatisticsError(f"Failed to compute cross-power via FFT: {e}")
    
    def comprehensive_analysis(self, conv_map: Any, 
                             analysis_types: List[str] = None,
                             **kwargs) -> StatisticsData:
        """Perform comprehensive statistical analysis.
        
        Parameters
        ----------
        conv_map : lenstools.ConvergenceMap
            Convergence map object
        analysis_types : List[str], optional
            Types of analysis to perform
        **kwargs
            Additional arguments for individual analyses
            
        Returns
        -------
        StatisticsData
            Comprehensive statistics results
        """
        self.ensure_initialized()
        self._track_usage()
        
        if analysis_types is None:
            analysis_types = ['power_spectrum', 'pdf', 'peak_counts']
        
        try:
            statistics = {}
            bins = {}
            
            # Power spectrum
            if 'power_spectrum' in analysis_types:
                l_centers, power = self.power_spectrum(conv_map, **kwargs)
                statistics['power_spectrum'] = power
                bins['power_spectrum'] = l_centers
            
            # Bispectrum
            if 'bispectrum' in analysis_types:
                l_centers_bi, bispectrum = self.bispectrum(conv_map, **kwargs)
                statistics['bispectrum'] = bispectrum
                bins['bispectrum'] = l_centers_bi
            
            # PDF
            if 'pdf' in analysis_types:
                bin_centers, pdf_values = self.pdf(conv_map, **kwargs)
                statistics['pdf'] = pdf_values
                bins['pdf'] = bin_centers
            
            # Peak counts
            if 'peak_counts' in analysis_types:
                thresholds, counts = self.peak_counts(conv_map, **kwargs)
                statistics['peak_counts'] = counts
                bins['peak_counts'] = thresholds
            
            # Minkowski functionals
            if 'minkowski' in analysis_types:
                thresh, v0, v1, v2 = self.minkowski_functionals(conv_map, **kwargs)
                statistics['minkowski_v0'] = v0
                statistics['minkowski_v1'] = v1
                statistics['minkowski_v2'] = v2
                bins['minkowski_v0'] = thresh
                bins['minkowski_v1'] = thresh
                bins['minkowski_v2'] = thresh
            
            # Create StatisticsData object
            metadata = {
                'provider': self.name,
                'analysis_types': analysis_types,
                'map_shape': conv_map.data.shape,
                'map_angle': getattr(conv_map, 'side_angle', 'unknown'),
            }
            
            stats_data = StatisticsData(
                statistics=statistics,
                bins=bins,
                metadata=metadata
            )
            
            self._logger.info(f"Completed comprehensive analysis: {len(analysis_types)} types")
            return stats_data
            
        except Exception as e:
            raise StatisticsError(f"Failed to perform comprehensive analysis: {e}")