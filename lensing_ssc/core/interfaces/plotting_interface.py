"""
Abstract interfaces for plotting and visualization providers.

This module defines interfaces for plotting operations, allowing for different
visualization backends (matplotlib, plotly, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from ..base.data_structures import MapData, PatchData, StatisticsData
from .data_interface import DataProvider


class PlottingProvider(DataProvider):
    """Abstract base class for plotting providers.
    
    This class defines interfaces for plotting and visualization operations,
    supporting multiple backends (matplotlib, plotly, bokeh, etc.).
    """
    
    @property
    @abstractmethod
    def supported_backends(self) -> List[str]:
        """List of supported plotting backends.
        
        Returns
        -------
        List[str]
            Names of supported backends (e.g., ['matplotlib', 'plotly', 'bokeh'])
        """
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """List of supported output formats.
        
        Returns
        -------
        List[str]
            Supported file formats (e.g., ['png', 'pdf', 'svg', 'html'])
        """
        pass
    
    @abstractmethod
    def set_backend(self, backend: str) -> None:
        """Set plotting backend.
        
        Parameters
        ----------
        backend : str
            Backend name
            
        Raises
        ------
        ValueError
            If backend is not supported
        """
        pass
    
    @abstractmethod
    def get_backend(self) -> str:
        """Get current plotting backend.
        
        Returns
        -------
        str
            Name of currently active backend
        """
        pass
    
    @abstractmethod
    def create_figure(self, figsize: Tuple[int, int] = (10, 8), **kwargs) -> Any:
        """Create a new figure.
        
        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size (width, height) in inches
        **kwargs
            Additional figure creation arguments
            
        Returns
        -------
        Any
            Figure object (backend-specific)
        """
        pass
    
    @abstractmethod
    def save_figure(self, fig: Any, path: Union[str, Path], **kwargs) -> None:
        """Save figure to file.
        
        Parameters
        ----------
        fig : Any
            Figure object
        path : str or Path
            Output file path
        **kwargs
            Additional save arguments (dpi, format, etc.)
            
        Raises
        ------
        OSError
            If file cannot be saved
        """
        pass
    
    @abstractmethod
    def show_figure(self, fig: Any) -> None:
        """Display figure.
        
        Parameters
        ----------
        fig : Any
            Figure object to display
        """
        pass
    
    @abstractmethod
    def close_figure(self, fig: Any) -> None:
        """Close figure and free resources.
        
        Parameters
        ----------
        fig : Any
            Figure object to close
        """
        pass
    
    @abstractmethod
    def set_style(self, style: str) -> None:
        """Set plotting style.
        
        Parameters
        ----------
        style : str
            Style name or path to style file
        """
        pass
    
    @abstractmethod
    def get_style_options(self) -> List[str]:
        """Get available plotting styles.
        
        Returns
        -------
        List[str]
            Available style names
        """
        pass


class MapPlottingProvider(PlottingProvider):
    """Abstract interface for map plotting providers."""
    
    @abstractmethod
    def plot_map(self, map_data: MapData, title: Optional[str] = None,
                colorbar: bool = True, **kwargs) -> Any:
        """Plot a full-sky map.
        
        Parameters
        ----------
        map_data : MapData
            Map data to plot
        title : str, optional
            Plot title
        colorbar : bool
            Whether to show colorbar
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        Any
            Figure object
        """
        pass
    
    @abstractmethod
    def plot_patch(self, patch_data: np.ndarray, title: Optional[str] = None,
                  colorbar: bool = True, **kwargs) -> Any:
        """Plot a single patch.
        
        Parameters
        ----------
        patch_data : np.ndarray
            Patch data to plot
        title : str, optional
            Plot title
        colorbar : bool
            Whether to show colorbar
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        Any
            Figure object
        """
        pass
    
    @abstractmethod
    def plot_patches_grid(self, patches: PatchData, n_cols: int = 4,
                         title: Optional[str] = None, **kwargs) -> Any:
        """Plot multiple patches in a grid.
        
        Parameters
        ----------
        patches : PatchData
            Patch data
        n_cols : int
            Number of columns in grid
        title : str, optional
            Overall title
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        Any
            Figure object
        """
        pass
    
    @abstractmethod
    def plot_mollweide(self, map_data: MapData, title: Optional[str] = None,
                      **kwargs) -> Any:
        """Plot map in Mollweide projection."""
        pass
    
    @abstractmethod
    def plot_orthographic(self, map_data: MapData, center: Tuple[float, float],
                         title: Optional[str] = None, **kwargs) -> Any:
        """Plot map in orthographic projection."""
        pass
    
    @abstractmethod
    def plot_gnomonic(self, map_data: MapData, center: Tuple[float, float],
                     size_deg: float, title: Optional[str] = None, **kwargs) -> Any:
        """Plot map in gnomonic projection."""
        pass


class StatisticsPlottingProvider(PlottingProvider):
    """Abstract interface for statistics plotting providers."""
    
    @abstractmethod
    def plot_power_spectrum(self, ell: np.ndarray, cl: np.ndarray,
                           error: Optional[np.ndarray] = None,
                           title: Optional[str] = None, **kwargs) -> Any:
        """Plot power spectrum.
        
        Parameters
        ----------
        ell : np.ndarray
            Multipole values
        cl : np.ndarray
            Power spectrum values
        error : np.ndarray, optional
            Error bars
        title : str, optional
            Plot title
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        Any
            Figure object
        """
        pass
    
    @abstractmethod
    def plot_bispectrum(self, ell: np.ndarray, bispectrum: np.ndarray,
                       configuration: str = "equilateral", 
                       title: Optional[str] = None, **kwargs) -> Any:
        """Plot bispectrum."""
        pass
    
    @abstractmethod
    def plot_pdf(self, bins: np.ndarray, pdf: np.ndarray,
                title: Optional[str] = None, **kwargs) -> Any:
        """Plot probability density function."""
        pass
    
    @abstractmethod
    def plot_peak_counts(self, thresholds: np.ndarray, counts: np.ndarray,
                        title: Optional[str] = None, **kwargs) -> Any:
        """Plot peak counts."""
        pass
    
    @abstractmethod
    def plot_correlation_matrix(self, matrix: np.ndarray, labels: Optional[List[str]] = None,
                               title: Optional[str] = None, **kwargs) -> Any:
        """Plot correlation matrix."""
        pass
    
    @abstractmethod
    def plot_comparison(self, x: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                       labels: Tuple[str, str], title: Optional[str] = None,
                       **kwargs) -> Any:
        """Plot comparison between two datasets."""
        pass
    
    @abstractmethod
    def plot_ratio(self, x: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                  title: Optional[str] = None, **kwargs) -> Any:
        """Plot ratio between two datasets."""
        pass


class InteractivePlottingProvider(PlottingProvider):
    """Abstract interface for interactive plotting providers."""
    
    @abstractmethod
    def create_interactive_map(self, map_data: MapData, **kwargs) -> Any:
        """Create interactive map visualization."""
        pass
    
    @abstractmethod
    def create_dashboard(self, data: Dict[str, Any], **kwargs) -> Any:
        """Create interactive dashboard."""
        pass
    
    @abstractmethod
    def add_widget(self, plot: Any, widget_type: str, **kwargs) -> Any:
        """Add interactive widget to plot."""
        pass


class VisualizationProvider(PlottingProvider):
    """High-level visualization provider combining multiple plot types."""
    
    @abstractmethod
    def create_summary_plot(self, statistics_data: StatisticsData, 
                           title: Optional[str] = None, **kwargs) -> Any:
        """Create comprehensive summary plot."""
        pass
    
    @abstractmethod
    def create_comparison_plot(self, data1: StatisticsData, data2: StatisticsData,
                              labels: Tuple[str, str], title: Optional[str] = None,
                              **kwargs) -> Any:
        """Create comparison plot between two datasets."""
        pass
    
    @abstractmethod
    def create_analysis_report(self, results: Dict[str, Any], 
                              output_path: Union[str, Path], **kwargs) -> None:
        """Create comprehensive analysis report."""
        pass
    
    @abstractmethod
    def create_animation(self, data_sequence: List[Any], 
                        output_path: Union[str, Path], **kwargs) -> None:
        """Create animation from data sequence."""
        pass