# lensing_ssc/processing/steps/statistics.py
"""
Statistics calculation steps for LensingSSC processing pipelines.

Provides steps for computing various statistical measures from patch data including
power spectra, bispectra, PDFs, peak counts, and correlation analysis.
"""

import logging
import time
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import numpy as np

from ..pipeline import ProcessingStep, StepResult, PipelineContext, StepStatus
from lensing_ssc.core.base import ValidationError, ProcessingError, StatisticsError


logger = logging.getLogger(__name__)


class BaseStatisticsStep(ProcessingStep):
    """Base class for statistics calculation steps."""
    
    def __init__(
        self,
        name: str,
        ngal_list: Optional[List[int]] = None,
        smoothing_lengths: Optional[List[float]] = None,
        num_processes: Optional[int] = None,
        chunk_size: int = 10,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.ngal_list = ngal_list or [0, 7, 15, 30, 50]
        self.smoothing_lengths = smoothing_lengths or [2.0, 5.0, 8.0, 10.0]
        self.num_processes = num_processes or mp.cpu_count()
        self.chunk_size = chunk_size
        
        # Common parameters
        self.epsilon_noise = 0.26  # Galaxy shape noise
        self.requires_lenstools = True
    
    def validate_inputs(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> bool:
        """Validate inputs with statistics-specific checks."""
        if not super().validate_inputs(context, inputs):
            return False
        
        if self.requires_lenstools:
            try:
                import lenstools
                from astropy import units as u
            except ImportError:
                self.logger.error("lenstools and astropy are required for statistics calculations")
                return False
        
        return True
    
    def _get_patch_data(self, inputs: Dict[str, StepResult]) -> Optional[np.ndarray]:
        """Get patch data from inputs."""
        for step_result in inputs.values():
            if step_result.is_successful() and 'patches' in step_result.data:
                return step_result.data['patches']
        return None
    
    def _get_patch_config(self, inputs: Dict[str, StepResult]) -> Dict[str, Any]:
        """Get patch configuration from inputs."""
        config = {'patch_size_deg': 10.0, 'xsize': 2048}
        
        for step_result in inputs.values():
            if step_result.is_successful():
                data = step_result.data
                if 'patch_size_deg' in data:
                    config['patch_size_deg'] = data['patch_size_deg']
                if 'xsize' in step_result.metadata:
                    config['xsize'] = step_result.metadata['xsize']
                elif 'patch_shape' in data and data['patch_shape']:
                    config['xsize'] = data['patch_shape'][-1]
        
        return config
    
    def _add_noise_to_patch(self, patch: np.ndarray, ngal: int, pixarea_arcmin2: float) -> np.ndarray:
        """Add galaxy shape noise to a patch."""
        if ngal <= 0:
            return patch.copy()
        
        noise_sigma = self.epsilon_noise / np.sqrt(ngal * pixarea_arcmin2)
        noise = np.random.normal(0, noise_sigma, patch.shape)
        return patch + noise
    
    def _create_convergence_map(self, patch_data: np.ndarray, patch_size_deg: float):
        """Create ConvergenceMap object from patch data."""
        try:
            from lenstools import ConvergenceMap
            from astropy import units as u
            
            return ConvergenceMap(patch_data, angle=patch_size_deg * u.deg)
        except ImportError:
            raise ProcessingError("lenstools is required to create ConvergenceMap")


class PowerSpectrumStep(BaseStatisticsStep):
    """Calculate power spectra from patch data."""
    
    def __init__(
        self,
        name: str,
        lmin: int = 300,
        lmax: int = 3000,
        n_bins: int = 8,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.lmin = lmin
        self.lmax = lmax
        self.n_bins = n_bins
        
        # Prepare ell bins
        self.l_edges = np.logspace(np.log10(lmin), np.log10(lmax), n_bins + 1)
        self.l_mids = (self.l_edges[:-1] + self.l_edges[1:]) / 2
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute power spectrum calculation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Get patch data
            patches = self._get_patch_data(inputs)
            if patches is None or patches.size == 0:
                raise ProcessingError("No patch data found for power spectrum calculation")
            
            patch_config = self._get_patch_config(inputs)
            patch_size_deg = patch_config['patch_size_deg']
            xsize = patch_config['xsize']
            pixarea_arcmin2 = (patch_size_deg * 60.0 / xsize) ** 2
            
            self.logger.info(f"Calculating power spectra for {len(patches)} patches")
            
            # Calculate power spectra for all noise levels
            power_spectra = {}
            
            for ngal in self.ngal_list:
                self.logger.info(f"Processing ngal = {ngal}")
                
                patch_power_spectra = []
                
                for i, patch in enumerate(patches):
                    if i % 100 == 0:
                        self.logger.debug(f"Processing patch {i}/{len(patches)} for ngal={ngal}")
                    
                    try:
                        # Add noise if needed
                        noisy_patch = self._add_noise_to_patch(patch, ngal, pixarea_arcmin2)
                        
                        # Create ConvergenceMap
                        conv_map = self._create_convergence_map(noisy_patch, patch_size_deg)
                        
                        # Calculate power spectrum
                        ps = self._calculate_power_spectrum(conv_map)
                        patch_power_spectra.append(ps)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process patch {i} for ngal {ngal}: {e}")
                        continue
                
                if patch_power_spectra:
                    power_spectra[ngal] = np.array(patch_power_spectra)
                else:
                    self.logger.warning(f"No power spectra calculated for ngal={ngal}")
                    power_spectra[ngal] = np.array([])
            
            result.data = {
                'power_spectra': power_spectra,
                'l_edges': self.l_edges,
                'l_mids': self.l_mids,
                'ngal_list': self.ngal_list,
                'patch_config': patch_config
            }
            
            result.metadata = {
                'n_patches_processed': len(patches),
                'ngal_values': self.ngal_list,
                'lmin': self.lmin,
                'lmax': self.lmax,
                'n_ell_bins': self.n_bins,
                'spectra_shapes': {ngal: ps.shape for ngal, ps in power_spectra.items()}
            }
            
            self.logger.info("Power spectrum calculation completed")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _calculate_power_spectrum(self, conv_map) -> np.ndarray:
        """Calculate power spectrum for a single convergence map."""
        try:
            from lensing_ssc.stats.power_spectrum import calculate_power_spectrum
            
            return calculate_power_spectrum(conv_map, self.l_edges, self.l_mids)
            
        except ImportError:
            # Fallback implementation
            return self._fallback_power_spectrum(conv_map)
    
    def _fallback_power_spectrum(self, conv_map) -> np.ndarray:
        """Fallback power spectrum calculation."""
        try:
            # Use lenstools built-in power spectrum calculation
            ps = conv_map.powerSpectrum(self.l_edges)
            return ps
        except Exception as e:
            self.logger.warning(f"Fallback power spectrum calculation failed: {e}")
            return np.full(len(self.l_mids), np.nan)


class BispectrumStep(BaseStatisticsStep):
    """Calculate bispectra from patch data."""
    
    def __init__(
        self,
        name: str,
        lmin: int = 300,
        lmax: int = 3000,
        n_bins: int = 8,
        bispectrum_types: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.lmin = lmin
        self.lmax = lmax
        self.n_bins = n_bins
        self.bispectrum_types = bispectrum_types or ['equilateral', 'isosceles', 'squeezed']
        
        # Prepare ell bins
        self.l_edges = np.logspace(np.log10(lmin), np.log10(lmax), n_bins + 1)
        self.l_mids = (self.l_edges[:-1] + self.l_edges[1:]) / 2
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute bispectrum calculation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Get patch data
            patches = self._get_patch_data(inputs)
            if patches is None or patches.size == 0:
                raise ProcessingError("No patch data found for bispectrum calculation")
            
            patch_config = self._get_patch_config(inputs)
            patch_size_deg = patch_config['patch_size_deg']
            xsize = patch_config['xsize']
            pixarea_arcmin2 = (patch_size_deg * 60.0 / xsize) ** 2
            
            self.logger.info(f"Calculating bispectra for {len(patches)} patches")
            
            # Calculate bispectra for all noise levels
            bispectra = {}
            
            for ngal in self.ngal_list:
                self.logger.info(f"Processing ngal = {ngal}")
                
                bispectrum_results = {btype: [] for btype in self.bispectrum_types}
                
                for i, patch in enumerate(patches):
                    if i % 100 == 0:
                        self.logger.debug(f"Processing patch {i}/{len(patches)} for ngal={ngal}")
                    
                    try:
                        # Add noise if needed
                        noisy_patch = self._add_noise_to_patch(patch, ngal, pixarea_arcmin2)
                        
                        # Create ConvergenceMap
                        conv_map = self._create_convergence_map(noisy_patch, patch_size_deg)
                        
                        # Calculate bispectrum
                        bs_results = self._calculate_bispectrum(conv_map)
                        
                        for btype, bs_values in bs_results.items():
                            if btype in bispectrum_results:
                                bispectrum_results[btype].append(bs_values)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process patch {i} for ngal {ngal}: {e}")
                        continue
                
                # Convert to arrays
                for btype in self.bispectrum_types:
                    if bispectrum_results[btype]:
                        bispectra.setdefault(ngal, {})[btype] = np.array(bispectrum_results[btype])
                    else:
                        self.logger.warning(f"No {btype} bispectrum calculated for ngal={ngal}")
                        bispectra.setdefault(ngal, {})[btype] = np.array([])
            
            result.data = {
                'bispectra': bispectra,
                'l_edges': self.l_edges,
                'l_mids': self.l_mids,
                'ngal_list': self.ngal_list,
                'bispectrum_types': self.bispectrum_types,
                'patch_config': patch_config
            }
            
            result.metadata = {
                'n_patches_processed': len(patches),
                'ngal_values': self.ngal_list,
                'bispectrum_types': self.bispectrum_types,
                'lmin': self.lmin,
                'lmax': self.lmax,
                'n_ell_bins': self.n_bins,
            }
            
            self.logger.info("Bispectrum calculation completed")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _calculate_bispectrum(self, conv_map) -> Dict[str, np.ndarray]:
        """Calculate bispectrum for a single convergence map."""
        try:
            from lensing_ssc.stats.bispectrum import calculate_bispectrum
            
            equ, iso, sq = calculate_bispectrum(conv_map, self.l_edges, self.l_mids)
            
            return {
                'equilateral': equ,
                'isosceles': iso,
                'squeezed': sq
            }
            
        except ImportError:
            # Fallback implementation
            return self._fallback_bispectrum(conv_map)
    
    def _fallback_bispectrum(self, conv_map) -> Dict[str, np.ndarray]:
        """Fallback bispectrum calculation."""
        self.logger.warning("Bispectrum calculation module not available, using placeholder")
        
        # Return placeholder arrays with correct shape
        placeholder = np.full(len(self.l_mids), np.nan)
        
        return {
            'equilateral': placeholder.copy(),
            'isosceles': placeholder.copy(),
            'squeezed': placeholder.copy()
        }


class PDFAnalysisStep(BaseStatisticsStep):
    """Calculate probability density functions from smoothed patch data."""
    
    def __init__(
        self,
        name: str,
        nu_min: float = -5.0,
        nu_max: float = 5.0,
        n_bins: int = 50,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.n_bins = n_bins
        
        # Prepare nu bins
        self.nu_bins = np.linspace(nu_min, nu_max, n_bins + 1)
        self.nu_mids = (self.nu_bins[:-1] + self.nu_bins[1:]) / 2
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute PDF analysis."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Get patch data
            patches = self._get_patch_data(inputs)
            if patches is None or patches.size == 0:
                raise ProcessingError("No patch data found for PDF analysis")
            
            patch_config = self._get_patch_config(inputs)
            patch_size_deg = patch_config['patch_size_deg']
            xsize = patch_config['xsize']
            pixarea_arcmin2 = (patch_size_deg * 60.0 / xsize) ** 2
            
            self.logger.info(f"Calculating PDFs for {len(patches)} patches")
            
            # Calculate PDFs for all noise levels and smoothing lengths
            pdfs = {}
            
            for ngal in self.ngal_list:
                pdfs[ngal] = {}
                
                for sl in self.smoothing_lengths:
                    self.logger.info(f"Processing ngal={ngal}, smoothing={sl} arcmin")
                    
                    patch_pdfs = []
                    patch_sigma0s = []
                    
                    for i, patch in enumerate(patches):
                        if i % 100 == 0:
                            self.logger.debug(f"Processing patch {i}/{len(patches)}")
                        
                        try:
                            # Add noise if needed
                            noisy_patch = self._add_noise_to_patch(patch, ngal, pixarea_arcmin2)
                            
                            # Create ConvergenceMap and smooth
                            conv_map = self._create_convergence_map(noisy_patch, patch_size_deg)
                            smoothed_map = self._smooth_map(conv_map, sl)
                            
                            # Calculate sigma0 and SNR map
                            sigma0 = np.std(smoothed_map.data)
                            if sigma0 > 1e-9:
                                snr_data = smoothed_map.data / sigma0
                            else:
                                snr_data = smoothed_map.data
                            
                            # Create SNR map and calculate PDF
                            snr_map = self._create_convergence_map(snr_data, patch_size_deg)
                            pdf = self._calculate_pdf(snr_map)
                            
                            patch_pdfs.append(pdf)
                            patch_sigma0s.append(sigma0)
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to process patch {i}: {e}")
                            continue
                    
                    if patch_pdfs:
                        pdfs[ngal][sl] = {
                            'pdf': np.array(patch_pdfs),
                            'sigma0': np.array(patch_sigma0s)
                        }
                    else:
                        self.logger.warning(f"No PDFs calculated for ngal={ngal}, sl={sl}")
                        pdfs[ngal][sl] = {
                            'pdf': np.array([]),
                            'sigma0': np.array([])
                        }
            
            result.data = {
                'pdfs': pdfs,
                'nu_bins': self.nu_bins,
                'nu_mids': self.nu_mids,
                'ngal_list': self.ngal_list,
                'smoothing_lengths': self.smoothing_lengths,
                'patch_config': patch_config
            }
            
            result.metadata = {
                'n_patches_processed': len(patches),
                'ngal_values': self.ngal_list,
                'smoothing_lengths': self.smoothing_lengths,
                'nu_range': (self.nu_min, self.nu_max),
                'n_nu_bins': self.n_bins,
            }
            
            self.logger.info("PDF analysis completed")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _smooth_map(self, conv_map, smoothing_length: float):
        """Smooth convergence map with given smoothing length."""
        try:
            from astropy import units as u
            return conv_map.smooth(smoothing_length * u.arcmin)
        except Exception as e:
            raise ProcessingError(f"Map smoothing failed: {e}")
    
    def _calculate_pdf(self, snr_map) -> np.ndarray:
        """Calculate PDF for SNR map."""
        try:
            from lensing_ssc.stats.pdf import calculate_pdf
            
            return calculate_pdf(snr_map, self.nu_bins)
            
        except ImportError:
            # Fallback implementation using numpy histogram
            return self._fallback_pdf(snr_map)
    
    def _fallback_pdf(self, snr_map) -> np.ndarray:
        """Fallback PDF calculation using histogram."""
        try:
            data = snr_map.data.flatten()
            finite_data = data[np.isfinite(data)]
            
            if len(finite_data) == 0:
                return np.zeros(len(self.nu_bins) - 1)
            
            counts, _ = np.histogram(finite_data, bins=self.nu_bins, density=True)
            return counts
            
        except Exception as e:
            self.logger.warning(f"Fallback PDF calculation failed: {e}")
            return np.full(len(self.nu_bins) - 1, np.nan)


class PeakCountingStep(BaseStatisticsStep):
    """Calculate peak and minima counts from smoothed patch data."""
    
    def __init__(
        self,
        name: str,
        nu_min: float = -5.0,
        nu_max: float = 5.0,
        n_bins: int = 50,
        count_minima: bool = True,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.n_bins = n_bins
        self.count_minima = count_minima
        
        # Prepare nu bins
        self.nu_bins = np.linspace(nu_min, nu_max, n_bins + 1)
        self.nu_mids = (self.nu_bins[:-1] + self.nu_bins[1:]) / 2
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute peak counting analysis."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Get patch data
            patches = self._get_patch_data(inputs)
            if patches is None or patches.size == 0:
                raise ProcessingError("No patch data found for peak counting")
            
            patch_config = self._get_patch_config(inputs)
            patch_size_deg = patch_config['patch_size_deg']
            xsize = patch_config['xsize']
            pixarea_arcmin2 = (patch_size_deg * 60.0 / xsize) ** 2
            
            self.logger.info(f"Counting peaks for {len(patches)} patches")
            
            # Calculate peak counts for all noise levels and smoothing lengths
            peak_counts = {}
            
            for ngal in self.ngal_list:
                peak_counts[ngal] = {}
                
                for sl in self.smoothing_lengths:
                    self.logger.info(f"Processing ngal={ngal}, smoothing={sl} arcmin")
                    
                    patch_peaks = []
                    patch_minima = []
                    patch_sigma0s = []
                    
                    for i, patch in enumerate(patches):
                        if i % 100 == 0:
                            self.logger.debug(f"Processing patch {i}/{len(patches)}")
                        
                        try:
                            # Add noise if needed
                            noisy_patch = self._add_noise_to_patch(patch, ngal, pixarea_arcmin2)
                            
                            # Create ConvergenceMap and smooth
                            conv_map = self._create_convergence_map(noisy_patch, patch_size_deg)
                            smoothed_map = self._smooth_map(conv_map, sl)
                            
                            # Calculate sigma0 and SNR map
                            sigma0 = np.std(smoothed_map.data)
                            if sigma0 > 1e-9:
                                snr_data = smoothed_map.data / sigma0
                            else:
                                snr_data = smoothed_map.data
                            
                            # Create SNR map and count peaks
                            snr_map = self._create_convergence_map(snr_data, patch_size_deg)
                            
                            peaks = self._count_peaks(snr_map, is_minima=False)
                            patch_peaks.append(peaks)
                            patch_sigma0s.append(sigma0)
                            
                            if self.count_minima:
                                minima = self._count_peaks(snr_map, is_minima=True)
                                patch_minima.append(minima)
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to process patch {i}: {e}")
                            continue
                    
                    if patch_peaks:
                        peak_counts[ngal][sl] = {
                            'peaks': np.array(patch_peaks),
                            'sigma0': np.array(patch_sigma0s)
                        }
                        if self.count_minima and patch_minima:
                            peak_counts[ngal][sl]['minima'] = np.array(patch_minima)
                    else:
                        self.logger.warning(f"No peak counts calculated for ngal={ngal}, sl={sl}")
                        peak_counts[ngal][sl] = {
                            'peaks': np.array([]),
                            'sigma0': np.array([])
                        }
                        if self.count_minima:
                            peak_counts[ngal][sl]['minima'] = np.array([])
            
            result.data = {
                'peak_counts': peak_counts,
                'nu_bins': self.nu_bins,
                'nu_mids': self.nu_mids,
                'ngal_list': self.ngal_list,
                'smoothing_lengths': self.smoothing_lengths,
                'count_minima': self.count_minima,
                'patch_config': patch_config
            }
            
            result.metadata = {
                'n_patches_processed': len(patches),
                'ngal_values': self.ngal_list,
                'smoothing_lengths': self.smoothing_lengths,
                'nu_range': (self.nu_min, self.nu_max),
                'n_nu_bins': self.n_bins,
                'count_minima': self.count_minima,
            }
            
            self.logger.info("Peak counting analysis completed")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _smooth_map(self, conv_map, smoothing_length: float):
        """Smooth convergence map with given smoothing length."""
        try:
            from astropy import units as u
            return conv_map.smooth(smoothing_length * u.arcmin)
        except Exception as e:
            raise ProcessingError(f"Map smoothing failed: {e}")
    
    def _count_peaks(self, snr_map, is_minima: bool = False) -> np.ndarray:
        """Count peaks or minima in SNR map."""
        try:
            from lensing_ssc.stats.peak_counts import calculate_peak_counts
            
            return calculate_peak_counts(snr_map, self.nu_bins, is_minima=is_minima)
            
        except ImportError:
            # Fallback implementation
            return self._fallback_peak_counting(snr_map, is_minima)
    
    def _fallback_peak_counting(self, snr_map, is_minima: bool = False) -> np.ndarray:
        """Fallback peak counting implementation."""
        try:
            from scipy import ndimage
            
            data = snr_map.data
            
            # Find local extrema
            if is_minima:
                # For minima, use minimum filter
                local_min = (data == ndimage.minimum_filter(data, size=3))
                extrema_values = data[local_min]
            else:
                # For maxima, use maximum filter
                local_max = (data == ndimage.maximum_filter(data, size=3))
                extrema_values = data[local_max]
            
            # Remove edge effects
            if len(extrema_values) > 0:
                finite_extrema = extrema_values[np.isfinite(extrema_values)]
                counts, _ = np.histogram(finite_extrema, bins=self.nu_bins)
                return counts.astype(float)
            else:
                return np.zeros(len(self.nu_bins) - 1)
                
        except ImportError:
            self.logger.warning("scipy not available for peak counting, using placeholder")
            return np.full(len(self.nu_bins) - 1, np.nan)
        except Exception as e:
            self.logger.warning(f"Fallback peak counting failed: {e}")
            return np.full(len(self.nu_bins) - 1, np.nan)


class CorrelationAnalysisStep(BaseStatisticsStep):
    """Calculate correlation functions and related statistics."""
    
    def __init__(
        self,
        name: str,
        correlation_types: Optional[List[str]] = None,
        max_separation: float = 10.0,
        n_bins: int = 20,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.correlation_types = correlation_types or ['2pt', 'shear']
        self.max_separation = max_separation
        self.n_bins = n_bins
        
        # Prepare separation bins
        self.theta_bins = np.logspace(-2, np.log10(max_separation), n_bins + 1)
        self.theta_mids = (self.theta_bins[:-1] + self.theta_bins[1:]) / 2
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute correlation analysis."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Get patch data
            patches = self._get_patch_data(inputs)
            if patches is None or patches.size == 0:
                raise ProcessingError("No patch data found for correlation analysis")
            
            patch_config = self._get_patch_config(inputs)
            patch_size_deg = patch_config['patch_size_deg']
            xsize = patch_config['xsize']
            pixarea_arcmin2 = (patch_size_deg * 60.0 / xsize) ** 2
            
            self.logger.info(f"Calculating correlations for {len(patches)} patches")
            
            # Calculate correlations for all noise levels
            correlations = {}
            
            for ngal in self.ngal_list:
                self.logger.info(f"Processing ngal = {ngal}")
                
                correlation_results = {ctype: [] for ctype in self.correlation_types}
                
                for i, patch in enumerate(patches):
                    if i % 50 == 0:
                        self.logger.debug(f"Processing patch {i}/{len(patches)}")
                    
                    try:
                        # Add noise if needed
                        noisy_patch = self._add_noise_to_patch(patch, ngal, pixarea_arcmin2)
                        
                        # Create ConvergenceMap
                        conv_map = self._create_convergence_map(noisy_patch, patch_size_deg)
                        
                        # Calculate correlations
                        corr_results = self._calculate_correlations(conv_map)
                        
                        for ctype, corr_values in corr_results.items():
                            if ctype in correlation_results:
                                correlation_results[ctype].append(corr_values)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process patch {i}: {e}")
                        continue
                
                # Convert to arrays
                for ctype in self.correlation_types:
                    if correlation_results[ctype]:
                        correlations.setdefault(ngal, {})[ctype] = np.array(correlation_results[ctype])
                    else:
                        self.logger.warning(f"No {ctype} correlations calculated for ngal={ngal}")
                        correlations.setdefault(ngal, {})[ctype] = np.array([])
            
            result.data = {
                'correlations': correlations,
                'theta_bins': self.theta_bins,
                'theta_mids': self.theta_mids,
                'ngal_list': self.ngal_list,
                'correlation_types': self.correlation_types,
                'patch_config': patch_config
            }
            
            result.metadata = {
                'n_patches_processed': len(patches),
                'ngal_values': self.ngal_list,
                'correlation_types': self.correlation_types,
                'max_separation': self.max_separation,
                'n_theta_bins': self.n_bins,
            }
            
            self.logger.info("Correlation analysis completed")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _calculate_correlations(self, conv_map) -> Dict[str, np.ndarray]:
        """Calculate correlation functions for a convergence map."""
        correlations = {}
        
        try:
            # 2-point correlation function
            if '2pt' in self.correlation_types:
                correlations['2pt'] = self._calculate_2pt_correlation(conv_map)
            
            # Shear correlation (if applicable)
            if 'shear' in self.correlation_types:
                correlations['shear'] = self._calculate_shear_correlation(conv_map)
            
        except Exception as e:
            self.logger.warning(f"Correlation calculation failed: {e}")
            # Return placeholder arrays
            for ctype in self.correlation_types:
                correlations[ctype] = np.full(len(self.theta_mids), np.nan)
        
        return correlations
    
    def _calculate_2pt_correlation(self, conv_map) -> np.ndarray:
        """Calculate 2-point correlation function."""
        try:
            # Use lenstools correlation function if available
            correlation = conv_map.correlation(self.theta_bins)
            return correlation
        except Exception:
            # Fallback to simple implementation
            return self._fallback_2pt_correlation(conv_map)
    
    def _fallback_2pt_correlation(self, conv_map) -> np.ndarray:
        """Fallback 2-point correlation calculation."""
        # Simplified implementation using FFT-based approach
        try:
            data = conv_map.data
            fft_data = np.fft.fft2(data)
            power_spectrum = np.abs(fft_data) ** 2
            correlation = np.fft.ifft2(power_spectrum).real
            
            # Extract radial profile (simplified)
            center = np.array(data.shape) // 2
            y, x = np.ogrid[:data.shape[0], :data.shape[1]]
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            
            # Bin by radius and calculate mean correlation
            correlations = []
            for i in range(len(self.theta_mids)):
                mask = (r >= self.theta_bins[i]) & (r < self.theta_bins[i+1])
                if np.any(mask):
                    correlations.append(np.mean(correlation[mask]))
                else:
                    correlations.append(0.0)
            
            return np.array(correlations)
            
        except Exception as e:
            self.logger.warning(f"Fallback 2pt correlation failed: {e}")
            return np.full(len(self.theta_mids), np.nan)
    
    def _calculate_shear_correlation(self, conv_map) -> np.ndarray:
        """Calculate shear correlation function."""
        try:
            # Convert convergence to shear
            shear = conv_map.toShear()
            
            # Calculate shear correlation
            correlation = shear.correlation(self.theta_bins)
            return correlation
            
        except Exception as e:
            self.logger.warning(f"Shear correlation calculation failed: {e}")
            return np.full(len(self.theta_mids), np.nan)


class CompositeStatisticsStep(BaseStatisticsStep):
    """Calculate multiple statistics in a single step for efficiency."""
    
    def __init__(
        self,
        name: str,
        statistics_types: Optional[List[str]] = None,
        lmin: int = 300,
        lmax: int = 3000,
        n_ell_bins: int = 8,
        nu_min: float = -5.0,
        nu_max: float = 5.0,
        n_nu_bins: int = 50,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.statistics_types = statistics_types or ['power_spectrum', 'pdf', 'peaks']
        self.lmin = lmin
        self.lmax = lmax
        self.n_ell_bins = n_ell_bins
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.n_nu_bins = n_nu_bins
        
        # Prepare bins
        self.l_edges = np.logspace(np.log10(lmin), np.log10(lmax), n_ell_bins + 1)
        self.l_mids = (self.l_edges[:-1] + self.l_edges[1:]) / 2
        self.nu_bins = np.linspace(nu_min, nu_max, n_nu_bins + 1)
        self.nu_mids = (self.nu_bins[:-1] + self.nu_bins[1:]) / 2
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute composite statistics calculation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Get patch data
            patches = self._get_patch_data(inputs)
            if patches is None or patches.size == 0:
                raise ProcessingError("No patch data found for statistics calculation")
            
            patch_config = self._get_patch_config(inputs)
            patch_size_deg = patch_config['patch_size_deg']
            xsize = patch_config['xsize']
            pixarea_arcmin2 = (patch_size_deg * 60.0 / xsize) ** 2
            
            self.logger.info(f"Calculating composite statistics for {len(patches)} patches")
            
            # Initialize results structure
            all_statistics = {}
            
            for ngal in self.ngal_list:
                all_statistics[ngal] = {}
                
                # Initialize statistic containers
                if 'power_spectrum' in self.statistics_types:
                    all_statistics[ngal]['power_spectrum'] = []
                
                if 'bispectrum' in self.statistics_types:
                    all_statistics[ngal]['bispectrum'] = {
                        'equilateral': [],
                        'isosceles': [],
                        'squeezed': []
                    }
                
                # Initialize smoothed statistics containers
                for sl in self.smoothing_lengths:
                    sl_key = f'sl_{sl}'
                    all_statistics[ngal][sl_key] = {}
                    
                    if 'pdf' in self.statistics_types:
                        all_statistics[ngal][sl_key]['pdf'] = []
                    if 'peaks' in self.statistics_types:
                        all_statistics[ngal][sl_key]['peaks'] = []
                        all_statistics[ngal][sl_key]['minima'] = []
                    
                    all_statistics[ngal][sl_key]['sigma0'] = []
                
                # Process patches
                for i, patch in enumerate(patches):
                    if i % 50 == 0:
                        self.logger.debug(f"Processing patch {i}/{len(patches)} for ngal={ngal}")
                    
                    try:
                        # Add noise if needed
                        noisy_patch = self._add_noise_to_patch(patch, ngal, pixarea_arcmin2)
                        conv_map = self._create_convergence_map(noisy_patch, patch_size_deg)
                        
                        # Calculate non-smoothed statistics
                        if 'power_spectrum' in self.statistics_types:
                            ps = self._calculate_power_spectrum_simple(conv_map)
                            all_statistics[ngal]['power_spectrum'].append(ps)
                        
                        if 'bispectrum' in self.statistics_types:
                            bs_results = self._calculate_bispectrum_simple(conv_map)
                            for btype, bs_values in bs_results.items():
                                all_statistics[ngal]['bispectrum'][btype].append(bs_values)
                        
                        # Calculate smoothed statistics
                        for sl in self.smoothing_lengths:
                            sl_key = f'sl_{sl}'
                            
                            # Smooth the map
                            smoothed_map = self._smooth_map(conv_map, sl)
                            sigma0 = np.std(smoothed_map.data)
                            all_statistics[ngal][sl_key]['sigma0'].append(sigma0)
                            
                            # Create SNR map
                            if sigma0 > 1e-9:
                                snr_data = smoothed_map.data / sigma0
                            else:
                                snr_data = smoothed_map.data
                            
                            snr_map = self._create_convergence_map(snr_data, patch_size_deg)
                            
                            # Calculate smoothed statistics
                            if 'pdf' in self.statistics_types:
                                pdf = self._calculate_pdf_simple(snr_map)
                                all_statistics[ngal][sl_key]['pdf'].append(pdf)
                            
                            if 'peaks' in self.statistics_types:
                                peaks = self._count_peaks_simple(snr_map, is_minima=False)
                                minima = self._count_peaks_simple(snr_map, is_minima=True)
                                all_statistics[ngal][sl_key]['peaks'].append(peaks)
                                all_statistics[ngal][sl_key]['minima'].append(minima)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process patch {i}: {e}")
                        continue
                
                # Convert lists to arrays
                for stat_key, stat_data in all_statistics[ngal].items():
                    if isinstance(stat_data, list):
                        all_statistics[ngal][stat_key] = np.array(stat_data) if stat_data else np.array([])
                    elif isinstance(stat_data, dict):
                        for sub_key, sub_data in stat_data.items():
                            if isinstance(sub_data, list):
                                all_statistics[ngal][stat_key][sub_key] = np.array(sub_data) if sub_data else np.array([])
            
            result.data = {
                'statistics': all_statistics,
                'l_edges': self.l_edges,
                'l_mids': self.l_mids,
                'nu_bins': self.nu_bins,
                'nu_mids': self.nu_mids,
                'ngal_list': self.ngal_list,
                'smoothing_lengths': self.smoothing_lengths,
                'statistics_types': self.statistics_types,
                'patch_config': patch_config
            }
            
            result.metadata = {
                'n_patches_processed': len(patches),
                'ngal_values': self.ngal_list,
                'smoothing_lengths': self.smoothing_lengths,
                'statistics_types': self.statistics_types,
                'n_ell_bins': self.n_ell_bins,
                'n_nu_bins': self.n_nu_bins,
            }
            
            self.logger.info("Composite statistics calculation completed")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _smooth_map(self, conv_map, smoothing_length: float):
        """Smooth convergence map."""
        try:
            from astropy import units as u
            return conv_map.smooth(smoothing_length * u.arcmin)
        except Exception as e:
            raise ProcessingError(f"Map smoothing failed: {e}")
    
    def _calculate_power_spectrum_simple(self, conv_map) -> np.ndarray:
        """Simple power spectrum calculation."""
        try:
            return conv_map.powerSpectrum(self.l_edges)
        except Exception:
            return np.full(len(self.l_mids), np.nan)
    
    def _calculate_bispectrum_simple(self, conv_map) -> Dict[str, np.ndarray]:
        """Simple bispectrum calculation."""
        # Placeholder implementation
        placeholder = np.full(len(self.l_mids), np.nan)
        return {
            'equilateral': placeholder.copy(),
            'isosceles': placeholder.copy(),
            'squeezed': placeholder.copy()
        }
    
    def _calculate_pdf_simple(self, snr_map) -> np.ndarray:
        """Simple PDF calculation."""
        try:
            data = snr_map.data.flatten()
            finite_data = data[np.isfinite(data)]
            if len(finite_data) == 0:
                return np.zeros(len(self.nu_bins) - 1)
            counts, _ = np.histogram(finite_data, bins=self.nu_bins, density=True)
            return counts
        except Exception:
            return np.full(len(self.nu_bins) - 1, np.nan)
    
    def _count_peaks_simple(self, snr_map, is_minima: bool = False) -> np.ndarray:
        """Simple peak counting."""
        try:
            from scipy import ndimage
            
            data = snr_map.data
            
            if is_minima:
                local_extrema = (data == ndimage.minimum_filter(data, size=3))
            else:
                local_extrema = (data == ndimage.maximum_filter(data, size=3))
            
            extrema_values = data[local_extrema]
            
            if len(extrema_values) > 0:
                finite_extrema = extrema_values[np.isfinite(extrema_values)]
                counts, _ = np.histogram(finite_extrema, bins=self.nu_bins)
                return counts.astype(float)
            else:
                return np.zeros(len(self.nu_bins) - 1)
                
        except ImportError:
            return np.full(len(self.nu_bins) - 1, np.nan)
        except Exception:
            return np.full(len(self.nu_bins) - 1, np.nan)


__all__ = [
    'BaseStatisticsStep',
    'PowerSpectrumStep',
    'BispectrumStep',
    'PDFAnalysisStep',
    'PeakCountingStep',
    'CorrelationAnalysisStep',
    'CompositeStatisticsStep',
]