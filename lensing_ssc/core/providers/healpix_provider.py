"""
HEALPix provider implementation.
"""

import numpy as np
from typing import Tuple, Any, Union
from pathlib import Path

from ..core.interfaces.data_interface import MapProvider
from ..core.base.data_structures import MapData
from ..core.base.exceptions import ProviderError
from .base_provider import LazyProvider


class HealpixProvider(LazyProvider):
    """Provider for HEALPix operations using healpy."""
    
    def __init__(self):
        super().__init__()
        self._healpy = None
    
    @property
    def name(self) -> str:
        return "HealpixProvider"
    
    @property 
    def version(self) -> str:
        return "1.0.0"
    
    def _check_dependencies(self) -> None:
        """Check if healpy is available."""
        try:
            import healpy
            self._healpy = healpy
        except ImportError:
            raise ImportError("healpy is required for HealpixProvider")
    
    def _initialize_backend(self, **kwargs) -> None:
        """Initialize healpy backend."""
        self._check_dependencies()
    
    def read_map(self, path: Union[str, Path], **kwargs) -> MapData:
        """Read HEALPix map from file."""
        self.ensure_initialized()
        
        try:
            data = self._healpy.read_map(str(path), **kwargs)
            return MapData(
                data=data,
                shape=data.shape,
                dtype=data.dtype,
                metadata={"source": str(path), "provider": self.name}
            )
        except Exception as e:
            raise ProviderError(f"Failed to read map from {path}: {e}")
    
    def write_map(self, map_data: MapData, path: Union[str, Path], **kwargs) -> None:
        """Write HEALPix map to file."""
        self.ensure_initialized()
        
        try:
            self._healpy.write_map(str(path), map_data.data, **kwargs)
        except Exception as e:
            raise ProviderError(f"Failed to write map to {path}: {e}")
    
    def get_nside(self, map_data: MapData) -> int:
        """Get NSIDE parameter from map data."""
        self.ensure_initialized()
        return self._healpy.npix2nside(map_data.size)
    
    def get_npix(self, nside: int) -> int:
        """Get number of pixels for given NSIDE."""
        self.ensure_initialized()
        return self._healpy.nside2npix(nside)
    
    def reorder_map(self, map_data: MapData, input_ordering: str, 
                   output_ordering: str) -> MapData:
        """Reorder map between RING and NEST."""
        self.ensure_initialized()
        
        if input_ordering == output_ordering:
            return map_data
        
        try:
            if input_ordering == "NEST" and output_ordering == "RING":
                reordered_data = self._healpy.reorder(map_data.data, n2r=True)
            elif input_ordering == "RING" and output_ordering == "NEST":
                reordered_data = self._healpy.reorder(map_data.data, r2n=True)
            else:
                raise ValueError(f"Invalid ordering: {input_ordering} -> {output_ordering}")
            
            return MapData(
                data=reordered_data,
                shape=reordered_data.shape,
                dtype=reordered_data.dtype,
                metadata={**map_data.metadata, "ordering": output_ordering}
            )
        except Exception as e:
            raise ProviderError(f"Failed to reorder map: {e}")
    
    def gnomonic_projection(self, map_data: MapData, center_coords: Tuple[float, float],
                          xsize: int, reso_arcmin: float, **kwargs) -> np.ndarray:
        """Create gnomonic projection."""
        self.ensure_initialized()
        
        try:
            return self._healpy.gnomview(
                map_data.data,
                rot=center_coords,
                xsize=xsize,
                reso=reso_arcmin,
                return_projected_map=True,
                no_plot=True,
                **kwargs
            )
        except Exception as e:
            raise ProviderError(f"Failed to create gnomonic projection: {e}")
    
    def query_polygon(self, nside: int, vertices: np.ndarray, 
                     nest: bool = False) -> np.ndarray:
        """Query pixels inside polygon."""
        self.ensure_initialized()
        
        try:
            return self._healpy.query_polygon(nside, vertices, nest=nest)
        except Exception as e:
            raise ProviderError(f"Failed to query polygon: {e}")
    
    def ang2pix(self, nside: int, theta: np.ndarray, phi: np.ndarray,
               nest: bool = False) -> np.ndarray:
        """Convert angles to pixel indices."""
        self.ensure_initialized()
        
        try:
            return self._healpy.ang2pix(nside, theta, phi, nest=nest)
        except Exception as e:
            raise ProviderError(f"Failed to convert angles to pixels: {e}")
    
    def pix2ang(self, nside: int, pix: np.ndarray, 
               nest: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pixel indices to angles."""
        self.ensure_initialized()
        
        try:
            return self._healpy.pix2ang(nside, pix, nest=nest)
        except Exception as e:
            raise ProviderError(f"Failed to convert pixels to angles: {e}")