"""
HEALPix provider implementation.

This module provides HEALPix operations using healpy with lazy loading,
caching, and comprehensive error handling. The provider abstracts healpy
functionality and provides a consistent interface for map operations.
"""

import numpy as np
from typing import Tuple, Any, Union, Optional, Dict, List
from pathlib import Path
import logging

from ..interfaces.data_interface import MapProvider
from ..base.data_structures import MapData
from ..base.exceptions import ProviderError, DataError
from .base_provider import LazyProvider, CachedProvider


class HealpixProvider(LazyProvider, CachedProvider):
    """Provider for HEALPix operations using healpy.
    
    This provider offers:
    - Lazy loading of healpy (only imported when needed)
    - Caching of expensive operations
    - Comprehensive map I/O operations
    - Coordinate transformations
    - Pixel queries and projections
    - Memory-efficient processing
    """
    
    def __init__(self, cache_size: int = 50, cache_ttl: Optional[float] = 3600):
        """Initialize HEALPix provider.
        
        Parameters
        ----------
        cache_size : int
            Maximum number of cached items
        cache_ttl : float, optional
            Cache time-to-live in seconds (default: 1 hour)
        """
        LazyProvider.__init__(self)
        CachedProvider.__init__(self, cache_size=cache_size, cache_ttl=cache_ttl)
        self._healpy = None
        self._rotator = None
    
    @property
    def name(self) -> str:
        """Provider name."""
        return "HealpixProvider"
    
    @property 
    def version(self) -> str:
        """Provider version."""
        return "1.0.0"
    
    def _check_dependencies(self) -> None:
        """Check if healpy is available."""
        try:
            self._healpy = self._lazy_import('healpy', 'healpy')
        except Exception as e:
            raise ImportError(f"healpy is required for HealpixProvider: {e}")
    
    def _initialize_backend(self, **kwargs) -> None:
        """Initialize healpy backend."""
        self._check_dependencies()
        
        # Initialize rotator if needed
        if kwargs.get('enable_rotator', True):
            try:
                rotator_cls = self._get_module_attribute('healpy', 'rotator')
                self._rotator = rotator_cls
                self._logger.debug("HEALPix rotator initialized")
            except Exception as e:
                self._logger.warning(f"Could not initialize rotator: {e}")
        
        # Set healpy verbosity
        verbose_level = kwargs.get('verbose', False)
        if hasattr(self._healpy, 'disable_warnings') and not verbose_level:
            self._healpy.disable_warnings()
    
    def _get_backend_info(self) -> Dict[str, Any]:
        """Get backend-specific information."""
        info = super()._get_backend_info()
        
        if self._healpy is not None:
            info.update({
                'healpy_version': getattr(self._healpy, '__version__', 'unknown'),
                'rotator_available': self._rotator is not None,
                'pixel_functions': {
                    'nside2npix': hasattr(self._healpy, 'nside2npix'),
                    'npix2nside': hasattr(self._healpy, 'npix2nside'),
                    'ang2pix': hasattr(self._healpy, 'ang2pix'),
                    'pix2ang': hasattr(self._healpy, 'pix2ang'),
                }
            })
        
        return info
    
    def read_map(self, path: Union[str, Path], **kwargs) -> MapData:
        """Read HEALPix map from file.
        
        Parameters
        ----------
        path : str or Path
            Path to the map file
        **kwargs
            Additional arguments passed to healpy.read_map
            
        Returns
        -------
        MapData
            Loaded map data with metadata
            
        Raises
        ------
        ProviderError
            If reading fails
        """
        self.ensure_initialized()
        self._track_usage()
        
        path_str = str(path)
        cache_key = f"read_map_{path_str}_{hash(frozenset(kwargs.items()))}"
        
        # Check cache first
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            self._logger.debug(f"Retrieved map from cache: {path_str}")
            return cached_result
        
        try:
            # Set default parameters
            read_kwargs = {
                'nest': kwargs.get('nest', None),
                'hdu': kwargs.get('hdu', 1),
                'field': kwargs.get('field', None),
                'dtype': kwargs.get('dtype', np.float64),
                'partial': kwargs.get('partial', False),
                'verbose': kwargs.get('verbose', False),
            }
            
            # Remove None values
            read_kwargs = {k: v for k, v in read_kwargs.items() if v is not None}
            
            # Read the map
            data = self._healpy.read_map(path_str, **read_kwargs)
            
            # Ensure data is numpy array
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
            
            # Create MapData object
            map_data = MapData(
                data=data,
                shape=data.shape,
                dtype=data.dtype,
                metadata={
                    "source": path_str,
                    "provider": self.name,
                    "nside": self.get_nside_from_npix(data.size),
                    "npix": data.size,
                    "ordering": kwargs.get('nest', 'unknown'),
                    "read_parameters": read_kwargs
                }
            )
            
            # Cache the result
            self._cache_set(cache_key, map_data)
            
            self._logger.debug(f"Successfully read map: {path_str}")
            return map_data
            
        except Exception as e:
            raise ProviderError(f"Failed to read map from {path}: {e}")
    
    def write_map(self, map_data: MapData, path: Union[str, Path], **kwargs) -> None:
        """Write HEALPix map to file.
        
        Parameters
        ----------
        map_data : MapData
            Map data to write
        path : str or Path
            Output file path
        **kwargs
            Additional arguments passed to healpy.write_map
            
        Raises
        ------
        ProviderError
            If writing fails
        """
        self.ensure_initialized()
        self._track_usage()
        
        path_str = str(path)
        
        try:
            # Set default parameters
            write_kwargs = {
                'nest': kwargs.get('nest', False),
                'dtype': kwargs.get('dtype', map_data.dtype),
                'fits_IDL': kwargs.get('fits_IDL', True),
                'coord': kwargs.get('coord', None),
                'partial': kwargs.get('partial', False),
                'overwrite': kwargs.get('overwrite', False),
            }
            
            # Remove None values
            write_kwargs = {k: v for k, v in write_kwargs.items() if v is not None}
            
            # Ensure output directory exists
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)
            
            # Write the map
            self._healpy.write_map(path_str, map_data.data, **write_kwargs)
            
            self._logger.debug(f"Successfully wrote map: {path_str}")
            
        except Exception as e:
            raise ProviderError(f"Failed to write map to {path}: {e}")
    
    def get_nside(self, map_data: MapData) -> int:
        """Get NSIDE parameter from map data.
        
        Parameters
        ----------
        map_data : MapData
            Map data
            
        Returns
        -------
        int
            NSIDE parameter
        """
        self.ensure_initialized()
        
        if 'nside' in map_data.metadata:
            return map_data.metadata['nside']
        
        return self.get_nside_from_npix(map_data.size)
    
    def get_nside_from_npix(self, npix: int) -> int:
        """Get NSIDE from number of pixels.
        
        Parameters
        ----------
        npix : int
            Number of pixels
            
        Returns
        -------
        int
            NSIDE parameter
        """
        self.ensure_initialized()
        
        try:
            return self._healpy.npix2nside(npix)
        except Exception as e:
            raise ProviderError(f"Failed to compute NSIDE from npix={npix}: {e}")
    
    def get_npix(self, nside: int) -> int:
        """Get number of pixels for given NSIDE.
        
        Parameters
        ----------
        nside : int
            NSIDE parameter
            
        Returns
        -------
        int
            Number of pixels
        """
        self.ensure_initialized()
        
        cache_key = f"npix_{nside}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            npix = self._healpy.nside2npix(nside)
            self._cache_set(cache_key, npix)
            return npix
        except Exception as e:
            raise ProviderError(f"Failed to compute npix for nside={nside}: {e}")
    
    def reorder_map(self, map_data: MapData, input_ordering: str, 
                   output_ordering: str) -> MapData:
        """Reorder map between RING and NEST orderings.
        
        Parameters
        ----------
        map_data : MapData
            Input map data
        input_ordering : str
            Input ordering ('RING' or 'NEST')
        output_ordering : str
            Output ordering ('RING' or 'NEST')
            
        Returns
        -------
        MapData
            Reordered map data
        """
        self.ensure_initialized()
        self._track_usage()
        
        if input_ordering.upper() == output_ordering.upper():
            return map_data
        
        cache_key = f"reorder_{id(map_data.data)}_{input_ordering}_{output_ordering}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            input_ord = input_ordering.upper()
            output_ord = output_ordering.upper()
            
            if input_ord == "NEST" and output_ord == "RING":
                reordered_data = self._healpy.reorder(map_data.data, n2r=True)
            elif input_ord == "RING" and output_ord == "NEST":
                reordered_data = self._healpy.reorder(map_data.data, r2n=True)
            else:
                raise ValueError(
                    f"Invalid ordering combination: {input_ordering} -> {output_ordering}. "
                    "Valid options are 'RING' and 'NEST'."
                )
            
            # Create new MapData with updated metadata
            new_metadata = map_data.metadata.copy()
            new_metadata.update({
                "ordering": output_ord,
                "reordered_from": input_ord,
                "provider": self.name
            })
            
            reordered_map = MapData(
                data=reordered_data,
                shape=reordered_data.shape,
                dtype=reordered_data.dtype,
                metadata=new_metadata
            )
            
            self._cache_set(cache_key, reordered_map)
            return reordered_map
            
        except Exception as e:
            raise ProviderError(f"Failed to reorder map from {input_ordering} to {output_ordering}: {e}")
    
    def gnomonic_projection(self, map_data: MapData, center_coords: Tuple[float, float],
                          xsize: int, reso_arcmin: float, **kwargs) -> np.ndarray:
        """Create gnomonic projection of the map.
        
        Parameters
        ----------
        map_data : MapData
            Input map data
        center_coords : tuple
            Center coordinates (lon, lat) in degrees
        xsize : int
            Output image size in pixels
        reso_arcmin : float
            Resolution in arcminutes per pixel
        **kwargs
            Additional arguments for gnomview
            
        Returns
        -------
        np.ndarray
            Projected map
        """
        self.ensure_initialized()
        self._track_usage()
        
        cache_key = f"gnomonic_{id(map_data.data)}_{center_coords}_{xsize}_{reso_arcmin}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            projection_kwargs = {
                'rot': center_coords,
                'xsize': xsize,
                'reso': reso_arcmin,
                'return_projected_map': True,
                'no_plot': True,
                'nest': kwargs.get('nest', False),
                'coord': kwargs.get('coord', None),
                'unit': kwargs.get('unit', ''),
                'title': kwargs.get('title', ''),
                'norm': kwargs.get('norm', None),
                'min': kwargs.get('min', None),
                'max': kwargs.get('max', None),
            }
            
            # Remove None values
            projection_kwargs = {k: v for k, v in projection_kwargs.items() if v is not None}
            
            projected_map = self._healpy.gnomview(map_data.data, **projection_kwargs)
            
            if projected_map is None:
                raise ProviderError("gnomview returned None - projection failed")
            
            self._cache_set(cache_key, projected_map)
            return projected_map
            
        except Exception as e:
            raise ProviderError(f"Failed to create gnomonic projection: {e}")
    
    def query_polygon(self, nside: int, vertices: np.ndarray, 
                     nest: bool = False, **kwargs) -> np.ndarray:
        """Query pixels inside a polygon.
        
        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter
        vertices : np.ndarray
            Polygon vertices as unit vectors (shape: N x 3)
        nest : bool
            Whether to use NEST ordering
        **kwargs
            Additional arguments for query_polygon
            
        Returns
        -------
        np.ndarray
            Pixel indices inside the polygon
        """
        self.ensure_initialized()
        self._track_usage()
        
        if vertices.shape[1] != 3:
            raise DataError("Vertices must be unit vectors with shape (N, 3)")
        
        try:
            query_kwargs = {
                'inclusive': kwargs.get('inclusive', False),
                'fact': kwargs.get('fact', 4),
            }
            
            pixel_indices = self._healpy.query_polygon(
                nside, vertices, nest=nest, **query_kwargs
            )
            
            return pixel_indices
            
        except Exception as e:
            raise ProviderError(f"Failed to query polygon: {e}")
    
    def ang2pix(self, nside: int, theta: Union[float, np.ndarray], 
               phi: Union[float, np.ndarray], nest: bool = False) -> Union[int, np.ndarray]:
        """Convert angles to pixel indices.
        
        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter
        theta : float or array
            Polar angle(s) in radians
        phi : float or array
            Azimuthal angle(s) in radians
        nest : bool
            Whether to use NEST ordering
            
        Returns
        -------
        int or np.ndarray
            Pixel index/indices
        """
        self.ensure_initialized()
        
        try:
            return self._healpy.ang2pix(nside, theta, phi, nest=nest)
        except Exception as e:
            raise ProviderError(f"Failed to convert angles to pixels: {e}")
    
    def pix2ang(self, nside: int, pix: Union[int, np.ndarray], 
               nest: bool = False) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert pixel indices to angles.
        
        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter
        pix : int or array
            Pixel index/indices
        nest : bool
            Whether to use NEST ordering
            
        Returns
        -------
        tuple
            (theta, phi) angles in radians
        """
        self.ensure_initialized()
        
        try:
            return self._healpy.pix2ang(nside, pix, nest=nest)
        except Exception as e:
            raise ProviderError(f"Failed to convert pixels to angles: {e}")
    
    def ang2vec(self, theta: Union[float, np.ndarray], 
               phi: Union[float, np.ndarray]) -> np.ndarray:
        """Convert angles to unit vectors.
        
        Parameters
        ----------
        theta : float or array
            Polar angle(s) in radians
        phi : float or array
            Azimuthal angle(s) in radians
            
        Returns
        -------
        np.ndarray
            Unit vector(s)
        """
        self.ensure_initialized()
        
        try:
            return self._healpy.ang2vec(theta, phi)
        except Exception as e:
            raise ProviderError(f"Failed to convert angles to vectors: {e}")
    
    def vec2ang(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert unit vectors to angles.
        
        Parameters
        ----------
        vectors : np.ndarray
            Unit vector(s) with shape (..., 3)
            
        Returns
        -------
        tuple
            (theta, phi) angles in radians
        """
        self.ensure_initialized()
        
        try:
            return self._healpy.vec2ang(vectors)
        except Exception as e:
            raise ProviderError(f"Failed to convert vectors to angles: {e}")
    
    def get_map_statistics(self, map_data: MapData, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate basic statistics for a HEALPix map.
        
        Parameters
        ----------
        map_data : MapData
            Map data
        mask : np.ndarray, optional
            Boolean mask (True for valid pixels)
            
        Returns
        -------
        Dict[str, float]
            Map statistics
        """
        self._track_usage()
        
        data = map_data.data
        if mask is not None:
            data = data[mask]
        
        # Remove invalid values
        valid_data = data[np.isfinite(data)]
        
        if len(valid_data) == 0:
            return {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'valid_pixels': 0,
                'total_pixels': len(data)
            }
        
        return {
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'valid_pixels': len(valid_data),
            'total_pixels': len(data)
        }
    
    def create_rotator(self, coord_in: str, coord_out: str, **kwargs) -> Any:
        """Create a coordinate rotator object.
        
        Parameters
        ----------
        coord_in : str
            Input coordinate system
        coord_out : str
            Output coordinate system
        **kwargs
            Additional rotator arguments
            
        Returns
        -------
        Any
            HEALPix rotator object
        """
        self.ensure_initialized()
        
        if self._rotator is None:
            raise ProviderError("Rotator not available - check healpy installation")
        
        try:
            rotator = self._rotator.Rotator(coord=[coord_in, coord_out], **kwargs)
            return rotator
        except Exception as e:
            raise ProviderError(f"Failed to create rotator: {e}")
    
    def validate_map(self, map_data: MapData) -> bool:
        """Validate that map data is a proper HEALPix map.
        
        Parameters
        ----------
        map_data : MapData
            Map data to validate
            
        Returns
        -------
        bool
            True if map is valid
        """
        try:
            # Check if npix is valid for HEALPix
            npix = map_data.size
            nside = self.get_nside_from_npix(npix)
            expected_npix = self.get_npix(nside)
            
            if npix != expected_npix:
                self._logger.warning(f"Invalid npix {npix} for HEALPix map")
                return False
            
            # Check for reasonable value ranges
            if np.all(np.isnan(map_data.data)):
                self._logger.warning("Map contains only NaN values")
                return False
            
            return True
            
        except Exception as e:
            self._logger.warning(f"Map validation failed: {e}")
            return False