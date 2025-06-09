"""
Abstract interfaces for data providers.

This module defines the interfaces that data providers must implement,
allowing for dependency injection and easy swapping of implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from ..base.data_structures import MapData, PatchData


class DataProvider(ABC):
    """Abstract base class for all data providers.
    
    This class defines the common interface that all data providers must implement,
    ensuring consistent behavior across different data backend implementations.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier.
        
        Returns
        -------
        str
            Unique name for this provider
        """
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Provider version string.
        
        Returns
        -------
        str
            Version of the provider implementation
        """
        pass
    
    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """List of required dependencies.
        
        Returns
        -------
        List[str]
            Names of required Python packages
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available (dependencies installed).
        
        Returns
        -------
        bool
            True if all dependencies are available and provider can be used
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: Any, **kwargs) -> bool:
        """Validate input data before processing.
        
        Parameters
        ----------
        data : Any
            Input data to validate
        **kwargs
            Additional validation parameters
            
        Returns
        -------
        bool
            True if input is valid
            
        Raises
        ------
        ValidationError
            If input validation fails
        """
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get comprehensive provider information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing provider details including name, version,
            dependencies, capabilities, and configuration options
        """
        pass


class MapProvider(DataProvider):
    """Abstract interface for map data providers (e.g., HEALPix)."""
    
    @abstractmethod
    def read_map(self, path: Union[str, Path], **kwargs) -> MapData:
        """Read a map from file.
        
        Parameters
        ----------
        path : str or Path
            Path to the map file
        **kwargs
            Additional arguments for the specific implementation
            
        Returns
        -------
        MapData
            Loaded map data
        """
        pass
    
    @abstractmethod
    def write_map(self, map_data: MapData, path: Union[str, Path], **kwargs) -> None:
        """Write a map to file.
        
        Parameters
        ----------
        map_data : MapData
            Map data to write
        path : str or Path
            Output path
        **kwargs
            Additional arguments for the specific implementation
        """
        pass
    
    @abstractmethod
    def get_nside(self, map_data: MapData) -> int:
        """Get NSIDE parameter from map data."""
        pass
    
    @abstractmethod
    def get_npix(self, nside: int) -> int:
        """Get number of pixels for given NSIDE."""
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Projected map
        """
        pass
    
    @abstractmethod
    def query_polygon(self, nside: int, vertices: np.ndarray, 
                     nest: bool = False) -> np.ndarray:
        """Query pixels inside a polygon.
        
        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter
        vertices : np.ndarray
            Polygon vertices as unit vectors (shape: N x 3)
        nest : bool
            Whether to use NEST ordering
            
        Returns
        -------
        np.ndarray
            Pixel indices inside the polygon
        """
        pass
    
    @abstractmethod
    def ang2pix(self, nside: int, theta: np.ndarray, phi: np.ndarray,
               nest: bool = False) -> np.ndarray:
        """Convert angles to pixel indices."""
        pass
    
    @abstractmethod
    def pix2ang(self, nside: int, pix: np.ndarray, 
               nest: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pixel indices to angles."""
        pass


class CatalogProvider(DataProvider):
    """Abstract interface for catalog data providers (e.g., nbodykit)."""
    
    @abstractmethod
    def read_catalog(self, path: Union[str, Path], dataset: str = None, **kwargs) -> Any:
        """Read a catalog from file.
        
        Parameters
        ----------
        path : str or Path
            Path to the catalog file
        dataset : str, optional
            Dataset name within the file
        **kwargs
            Additional arguments
            
        Returns
        -------
        Any
            Catalog object (implementation-specific)
        """
        pass
    
    @abstractmethod
    def get_column(self, catalog: Any, column: str, 
                  start: Optional[int] = None, end: Optional[int] = None) -> np.ndarray:
        """Get a column from the catalog.
        
        Parameters
        ----------
        catalog : Any
            Catalog object
        column : str
            Column name
        start, end : int, optional
            Slice indices
            
        Returns
        -------
        np.ndarray
            Column data
        """
        pass
    
    @abstractmethod
    def get_attributes(self, catalog: Any) -> Dict[str, Any]:
        """Get catalog attributes/metadata."""
        pass
    
    @abstractmethod
    def get_size(self, catalog: Any) -> int:
        """Get catalog size (number of entries)."""
        pass


class ConvergenceMapProvider(DataProvider):
    """Abstract interface for convergence map providers (e.g., lenstools)."""
    
    @abstractmethod
    def create_convergence_map(self, data: np.ndarray, angle_deg: float) -> Any:
        """Create a convergence map object.
        
        Parameters
        ----------
        data : np.ndarray
            Map data
        angle_deg : float
            Map angular size in degrees
            
        Returns
        -------
        Any
            Convergence map object (implementation-specific)
        """
        pass
    
    @abstractmethod
    def power_spectrum(self, conv_map: Any, l_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power spectrum.
        
        Parameters
        ----------
        conv_map : Any
            Convergence map object
        l_edges : np.ndarray
            Multipole bin edges
            
        Returns
        -------
        tuple
            (ell_centers, power_spectrum)
        """
        pass
    
    @abstractmethod
    def bispectrum(self, conv_map: Any, l_edges: np.ndarray, 
                  configuration: str = "equilateral", **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bispectrum.
        
        Parameters
        ----------
        conv_map : Any
            Convergence map object
        l_edges : np.ndarray
            Multipole bin edges
        configuration : str
            Bispectrum configuration
        **kwargs
            Additional arguments
            
        Returns
        -------
        tuple
            (ell_centers, bispectrum)
        """
        pass
    
    @abstractmethod
    def pdf(self, conv_map: Any, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate probability density function.
        
        Parameters
        ----------
        conv_map : Any
            Convergence map object
        bins : np.ndarray
            Bin edges
            
        Returns
        -------
        tuple
            (bin_centers, pdf_values)
        """
        pass
    
    @abstractmethod
    def locate_peaks(self, conv_map: Any, threshold: float) -> Tuple[np.ndarray, Any]:
        """Locate peaks in the map.
        
        Parameters
        ----------
        conv_map : Any
            Convergence map object
        threshold : float
            Peak threshold
            
        Returns
        -------
        tuple
            (peak_heights, peak_positions)
        """
        pass
    
    @abstractmethod
    def smooth(self, conv_map: Any, scale_arcmin: float) -> Any:
        """Smooth the convergence map.
        
        Parameters
        ----------
        conv_map : Any
            Convergence map object
        scale_arcmin : float
            Smoothing scale in arcminutes
            
        Returns
        -------
        Any
            Smoothed convergence map
        """
        pass