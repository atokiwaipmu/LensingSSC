"""
LensTools provider implementation.
"""

import numpy as np
from typing import Tuple, Any, Union
from pathlib import Path

from ..core.interfaces.data_interface import ConvergenceMapProvider
from ..core.base.exceptions import ProviderError
from .base_provider import LazyProvider


class LenstoolsProvider(LazyProvider):
    """Provider for lensing analysis using lenstools."""
    
    def __init__(self):
        super().__init__()
        self._lenstools = None
    
    @property
    def name(self) -> str:
        return "LenstoolsProvider"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def _check_dependencies(self) -> None:
        """Check if lenstools is available."""
        try:
            import lenstools
            self._lenstools = lenstools
        except ImportError:
            raise ImportError("lenstools is required for LenstoolsProvider")
    
    def _initialize_backend(self, **kwargs) -> None:
        """Initialize lenstools backend."""
        self._check_dependencies()
    
    def create_convergence_map(self, data: np.ndarray, angle_deg: float) -> Any:
        """Create convergence map object."""
        self.ensure_initialized()
        
        try:
            from astropy import units as u
            return self._lenstools.ConvergenceMap(data, angle=angle_deg * u.deg)
        except Exception as e:
            raise ProviderError(f"Failed to create convergence map: {e}")
    
    def power_spectrum(self, conv_map: Any, l_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power spectrum."""
        self.ensure_initialized()
        
        try:
            return conv_map.powerSpectrum(l_edges)
        except Exception as e:
            raise ProviderError(f"Failed to calculate power spectrum: {e}")
    
    def bispectrum(self, conv_map: Any, l_edges: np.ndarray, 
                  configuration: str = "equilateral", **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bispectrum."""
        self.ensure_initialized()
        
        try:
            return conv_map.bispectrum(l_edges, configuration=configuration, **kwargs)
        except Exception as e:
            raise ProviderError(f"Failed to calculate bispectrum: {e}")
    
    def pdf(self, conv_map: Any, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate PDF."""
        self.ensure_initialized()
        
        try:
            return conv_map.pdf(bins)
        except Exception as e:
            raise ProviderError(f"Failed to calculate PDF: {e}")
    
    def locate_peaks(self, conv_map: Any, threshold: float) -> Tuple[np.ndarray, Any]:
        """Locate peaks in map."""
        self.ensure_initialized()
        
        try:
            return conv_map.locatePeaks(threshold)
        except Exception as e:
            raise ProviderError(f"Failed to locate peaks: {e}")
    
    def smooth(self, conv_map: Any, scale_arcmin: float) -> Any:
        """Smooth convergence map."""
        self.ensure_initialized()
        
        try:
            from astropy import units as u
            return conv_map.smooth(scale_arcmin * u.arcmin)
        except Exception as e:
            raise ProviderError(f"Failed to smooth map: {e}")