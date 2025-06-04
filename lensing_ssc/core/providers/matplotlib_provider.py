"""
Matplotlib provider implementation.

This module provides plotting and visualization capabilities using matplotlib
with lazy loading, style management, and comprehensive plotting functions
for lensing analysis.
"""

import io
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import numpy as np
import logging

from ..interfaces.plotting_interface import (
    PlottingProvider, MapPlottingProvider, StatisticsPlottingProvider, 
    VisualizationProvider
)
from ..base.data_structures import MapData, PatchData, StatisticsData
from ..base.exceptions import ProviderError, VisualizationError
from .base_provider import LazyProvider, CachedProvider


class MatplotlibProvider(LazyProvider, CachedProvider, PlottingProvider, 
                        MapPlottingProvider, StatisticsPlottingProvider, 
                        VisualizationProvider):
    """Provider for plotting and visualization using matplotlib.
    
    This provider offers:
    - Lazy loading of matplotlib and related packages
    - Comprehensive plotting functions for lensing analysis
    - Style management and customization
    - Interactive and static plotting capabilities
    - Export to multiple formats
    - Memory-efficient plot caching
    """
    
    def __init__(self, cache_size: int = 20, cache_ttl: Optional[float] = 1800):
        """Initialize matplotlib provider.
        
        Parameters
        ----------
        cache_size : int
            Maximum number of cached plots
        cache_ttl : float, optional
            Cache time-to-live in seconds (default: 30 minutes)
        """
        LazyProvider.__init__(self)
        CachedProvider.__init__(self, cache_size=cache_size, cache_ttl=cache_ttl)
        
        self._matplotlib = None
        self._pyplot = None
        self._colors = None
        self._cm = None
        self._patches = None
        self._gridspec = None
        
        self._backend_set = False
        self._style_context = None
        self._default_style = {
            'figure.figsize': (12, 8),
            'figure.dpi': 150,
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'axes.grid': True,
            'grid.alpha': 0.3,
        }
    
    @property
    def name(self) -> str:
        """Provider name."""
        return "MatplotlibProvider"
    
    @property
    def version(self) -> str:
        """Provider version."""
        return "1.0.0"
    
    def _check_dependencies(self) -> None:
        """Check if matplotlib and related packages are available."""
        try:
            self._matplotlib = self._lazy_import('matplotlib')
            self._pyplot = self._lazy_import('matplotlib.pyplot', 'pyplot')
            self._colors = self._lazy_import('matplotlib.colors', 'colors')
            self._cm = self._lazy_import('matplotlib.cm', 'cm')
            self._patches = self._lazy_import('matplotlib.patches', 'patches')
            self._gridspec = self._lazy_import('matplotlib.gridspec', 'gridspec')
        except Exception as e:
            raise ImportError(f"matplotlib is required for MatplotlibProvider: {e}")
    
    def _initialize_backend(self, **kwargs) -> None:
        """Initialize matplotlib backend and settings."""
        self._check_dependencies()
        
        # Set backend if specified
        backend = kwargs.get('backend', None)
        if backend and not self._backend_set:
            try:
                self._matplotlib.use(backend)
                self._backend_set = True
                self._logger.debug(f"Set matplotlib backend to {backend}")
            except Exception as e:
                self._logger.warning(f"Failed to set backend {backend}: {e}")
        
        # Apply default style
        self._apply_style(kwargs.get('style', self._default_style))
        
        # Set up interactive mode
        interactive = kwargs.get('interactive', False)
        if interactive:
            self._pyplot.ion()
        else:
            self._pyplot.ioff()
        
        # Configure warnings
        if not kwargs.get('show_warnings', False):
            warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    def _apply_style(self, style: Union[str, Dict[str, Any]]) -> None:
        """Apply matplotlib style settings."""
        try:
            if isinstance(style, str):
                self._pyplot.style.use(style)
            elif isinstance(style, dict):
                self._matplotlib.rcParams.update(style)
            self._logger.debug(f"Applied matplotlib style: {type(style).__name__}")
        except Exception as e:
            self._logger.warning(f"Failed to apply style: {e}")
    
    def _get_backend_info(self) -> Dict[str, Any]:
        """Get backend-specific information."""
        info = super()._get_backend_info()
        
        if self._matplotlib is not None:
            info.update({
                'matplotlib_version': getattr(self._matplotlib, '__version__', 'unknown'),
                'backend': self._matplotlib.get_backend(),
                'interactive': self._pyplot.isinteractive() if self._pyplot else False,
                'available_backends': getattr(self._matplotlib, 'backend_bases', {}).keys() if hasattr(self._matplotlib, 'backend_bases') else [],
            })
        
        return info
    
    def set_backend(self, backend: str) -> None:
        """Set plotting backend."""
        self.ensure_initialized()
        
        try:
            self._matplotlib.use(backend)
            self._backend_set = True
            self._logger.info(f"Changed matplotlib backend to {backend}")
        except Exception as e:
            raise ProviderError(f"Failed to set backend {backend}: {e}")
    
    def get_backend(self) -> str:
        """Get current plotting backend."""
        self.ensure_initialized()
        return self._matplotlib.get_backend()
    
    def create_figure(self, figsize: Tuple[int, int] = (10, 8), **kwargs) -> Any:
        """Create a new figure.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches
        **kwargs
            Additional arguments for plt.figure()
            
        Returns
        -------
        matplotlib.figure.Figure
            Created figure object
        """
        self.ensure_initialized()
        self._track_usage()
        
        try:
            fig_kwargs = {
                'figsize': figsize,
                'dpi': kwargs.get('dpi', 150),
                'facecolor': kwargs.get('facecolor', 'white'),
                'edgecolor': kwargs.get('edgecolor', 'black'),
                'tight_layout': kwargs.get('tight_layout', True),
            }
            
            fig = self._pyplot.figure(**fig_kwargs)
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create figure: {e}")
    
    def save_figure(self, fig: Any, path: Union[str, Path], **kwargs) -> None:
        """Save figure to file.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        path : str or Path
            Output file path
        **kwargs
            Additional arguments for savefig()
        """
        self.ensure_initialized()
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            save_kwargs = {
                'dpi': kwargs.get('dpi', 300),
                'bbox_inches': kwargs.get('bbox_inches', 'tight'),
                'facecolor': kwargs.get('facecolor', 'white'),
                'edgecolor': kwargs.get('edgecolor', 'none'),
                'transparent': kwargs.get('transparent', False),
                'pad_inches': kwargs.get('pad_inches', 0.1),
            }
            
            fig.savefig(str(path), **save_kwargs)
            self._logger.debug(f"Saved figure to {path}")
            
        except Exception as e:
            raise VisualizationError(f"Failed to save figure to {path}: {e}")
    
    def show_figure(self, fig: Any) -> None:
        """Display figure."""
        self.ensure_initialized()
        
        try:
            self._pyplot.show()
        except Exception as e:
            self._logger.warning(f"Failed to show figure: {e}")
    
    def close_figure(self, fig: Any) -> None:
        """Close figure and free memory."""
        try:
            self._pyplot.close(fig)
        except Exception as e:
            self._logger.warning(f"Failed to close figure: {e}")
    
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
        matplotlib.figure.Figure
            Figure object
        """
        self.ensure_initialized()
        self._track_usage()
        
        cache_key = f"plot_map_{id(map_data.data)}_{title}_{colorbar}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            # For full-sky maps, we need healpy for visualization
            # This is a fallback implementation using matplotlib directly
            fig = self.create_figure(kwargs.get('figsize', (12, 8)))
            ax = fig.add_subplot(111)
            
            # Reshape data if needed for visualization
            data = map_data.data
            if data.ndim == 1:
                # Convert HEALPix data to 2D for basic visualization
                # This is a simplified approach - full HEALPix plotting needs healpy
                side_length = int(np.sqrt(len(data)))
                if side_length * side_length != len(data):
                    # Use mollweide projection placeholder
                    theta = np.linspace(0, np.pi, 180)
                    phi = np.linspace(0, 2*np.pi, 360)
                    data_2d = np.zeros((len(theta), len(phi)))
                    # Simple mapping for demonstration
                    for i, t in enumerate(theta):
                        for j, p in enumerate(phi):
                            # Map to nearest HEALPix pixel (simplified)
                            idx = min(int((i * len(phi) + j) * len(data) / (len(theta) * len(phi))), len(data) - 1)
                            data_2d[i, j] = data[idx]
                    data = data_2d
                else:
                    data = data.reshape(side_length, side_length)
            
            # Create the plot
            plot_kwargs = {
                'cmap': kwargs.get('cmap', kwargs.get('colormap', 'viridis')),
                'vmin': kwargs.get('vmin'),
                'vmax': kwargs.get('vmax'),
                'aspect': kwargs.get('aspect', 'auto'),
            }
            plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}
            
            im = ax.imshow(data, **plot_kwargs)
            
            if title:
                ax.set_title(title, fontsize=kwargs.get('title_fontsize', 16))
            
            if colorbar:
                cbar = fig.colorbar(im, ax=ax, shrink=kwargs.get('colorbar_shrink', 0.8))
                if 'colorbar_label' in kwargs:
                    cbar.set_label(kwargs['colorbar_label'])
            
            ax.set_xlabel(kwargs.get('xlabel', ''))
            ax.set_ylabel(kwargs.get('ylabel', ''))
            
            self._cache_set(cache_key, fig)
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to plot map: {e}")
    
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
        matplotlib.figure.Figure
            Figure object
        """
        self.ensure_initialized()
        self._track_usage()
        
        try:
            fig = self.create_figure(kwargs.get('figsize', (8, 8)))
            ax = fig.add_subplot(111)
            
            plot_kwargs = {
                'cmap': kwargs.get('cmap', kwargs.get('colormap', 'viridis')),
                'vmin': kwargs.get('vmin'),
                'vmax': kwargs.get('vmax'),
                'aspect': kwargs.get('aspect', 'equal'),
                'origin': kwargs.get('origin', 'lower'),
            }
            plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}
            
            im = ax.imshow(patch_data, **plot_kwargs)
            
            if title:
                ax.set_title(title, fontsize=kwargs.get('title_fontsize', 14))
            
            if colorbar:
                cbar = fig.colorbar(im, ax=ax, shrink=kwargs.get('colorbar_shrink', 0.8))
                if 'colorbar_label' in kwargs:
                    cbar.set_label(kwargs['colorbar_label'])
            
            ax.set_xlabel(kwargs.get('xlabel', 'Pixels'))
            ax.set_ylabel(kwargs.get('ylabel', 'Pixels'))
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to plot patch: {e}")
    
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
        matplotlib.figure.Figure
            Figure object
        """
        self.ensure_initialized()
        self._track_usage()
        
        try:
            n_patches = patches.n_patches
            n_rows = (n_patches + n_cols - 1) // n_cols
            
            figsize = kwargs.get('figsize', (4 * n_cols, 4 * n_rows))
            fig = self.create_figure(figsize)
            
            if title:
                fig.suptitle(title, fontsize=kwargs.get('title_fontsize', 16))
            
            # Common colormap limits
            if kwargs.get('common_colorbar', True):
                vmin = kwargs.get('vmin', np.min(patches.patches))
                vmax = kwargs.get('vmax', np.max(patches.patches))
            else:
                vmin = vmax = None
            
            for i in range(n_patches):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                
                plot_kwargs = {
                    'cmap': kwargs.get('cmap', 'viridis'),
                    'vmin': vmin,
                    'vmax': vmax,
                    'aspect': 'equal',
                    'origin': 'lower',
                }
                
                im = ax.imshow(patches.patches[i], **plot_kwargs)
                ax.set_title(f'Patch {i}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Add common colorbar
            if kwargs.get('colorbar', True) and kwargs.get('common_colorbar', True):
                fig.colorbar(im, ax=fig.get_axes(), shrink=0.6, aspect=20)
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to plot patches grid: {e}")
    
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
        matplotlib.figure.Figure
            Figure object
        """
        self.ensure_initialized()
        self._track_usage()
        
        try:
            fig = self.create_figure(kwargs.get('figsize', (10, 6)))
            ax = fig.add_subplot(111)
            
            # Main plot
            plot_kwargs = {
                'color': kwargs.get('color', 'blue'),
                'linewidth': kwargs.get('linewidth', 2),
                'marker': kwargs.get('marker', 'o'),
                'markersize': kwargs.get('markersize', 4),
                'label': kwargs.get('label', 'Power Spectrum'),
            }
            
            if kwargs.get('loglog', True):
                ax.loglog(ell, cl, **plot_kwargs)
            elif kwargs.get('semilogx', False):
                ax.semilogx(ell, cl, **plot_kwargs)
            elif kwargs.get('semilogy', False):
                ax.semilogy(ell, cl, **plot_kwargs)
            else:
                ax.plot(ell, cl, **plot_kwargs)
            
            # Error bars
            if error is not None:
                ax.errorbar(ell, cl, yerr=error, fmt='none', 
                           capsize=kwargs.get('capsize', 3),
                           color=kwargs.get('error_color', plot_kwargs['color']),
                           alpha=kwargs.get('error_alpha', 0.7))
            
            # Labels and title
            ax.set_xlabel(kwargs.get('xlabel', r'$\ell$'), fontsize=14)
            ax.set_ylabel(kwargs.get('ylabel', r'$C_\ell$'), fontsize=14)
            
            if title:
                ax.set_title(title, fontsize=16)
            
            # Grid and legend
            ax.grid(True, alpha=0.3)
            if kwargs.get('legend', True):
                ax.legend()
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to plot power spectrum: {e}")
    
    def plot_correlation_matrix(self, matrix: np.ndarray, labels: Optional[List[str]] = None,
                               title: Optional[str] = None, **kwargs) -> Any:
        """Plot correlation matrix.
        
        Parameters
        ----------
        matrix : np.ndarray
            Correlation matrix
        labels : List[str], optional
            Axis labels
        title : str, optional
            Plot title
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        self.ensure_initialized()
        self._track_usage()
        
        try:
            fig = self.create_figure(kwargs.get('figsize', (10, 8)))
            ax = fig.add_subplot(111)
            
            plot_kwargs = {
                'cmap': kwargs.get('cmap', 'RdBu_r'),
                'vmin': kwargs.get('vmin', -1),
                'vmax': kwargs.get('vmax', 1),
                'aspect': 'equal',
            }
            
            im = ax.imshow(matrix, **plot_kwargs)
            
            # Add text annotations
            if kwargs.get('annotate', True):
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                     ha="center", va="center",
                                     color="black" if abs(matrix[i, j]) < 0.5 else "white",
                                     fontsize=kwargs.get('annotation_fontsize', 10))
            
            # Labels
            if labels:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_yticklabels(labels)
            
            if title:
                ax.set_title(title, fontsize=16, pad=20)
            
            # Colorbar
            if kwargs.get('colorbar', True):
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Correlation', fontsize=12)
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to plot correlation matrix: {e}")
    
    def create_summary_plot(self, statistics_data: StatisticsData, 
                           title: Optional[str] = None, **kwargs) -> Any:
        """Create comprehensive summary plot.
        
        Parameters
        ----------
        statistics_data : StatisticsData
            Statistics data to plot
        title : str, optional
            Overall title
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        self.ensure_initialized()
        self._track_usage()
        
        try:
            # Determine subplot layout based on available statistics
            stats = statistics_data.statistics
            n_stats = len(stats)
            
            if n_stats == 0:
                raise VisualizationError("No statistics to plot")
            
            # Calculate grid layout
            n_cols = min(3, n_stats)
            n_rows = (n_stats + n_cols - 1) // n_cols
            
            figsize = kwargs.get('figsize', (6 * n_cols, 4 * n_rows))
            fig = self.create_figure(figsize)
            
            if title:
                fig.suptitle(title, fontsize=18, y=0.98)
            
            # Plot each statistic
            for i, (stat_name, stat_data) in enumerate(stats.items()):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                
                # Get corresponding bins if available
                bins = statistics_data.bins.get(stat_name, np.arange(len(stat_data)))
                
                # Plot based on statistic type
                if 'power' in stat_name.lower() or 'spectrum' in stat_name.lower():
                    ax.loglog(bins, stat_data, 'o-', linewidth=2, markersize=4)
                    ax.set_xlabel(r'$\ell$')
                    ax.set_ylabel(r'$C_\ell$')
                elif 'pdf' in stat_name.lower():
                    ax.plot(bins, stat_data, '-', linewidth=2)
                    ax.set_xlabel('Value')
                    ax.set_ylabel('PDF')
                else:
                    ax.plot(bins, stat_data, 'o-', linewidth=2, markersize=4)
                    ax.set_xlabel('Bin')
                    ax.set_ylabel(stat_name)
                
                ax.set_title(stat_name.replace('_', ' ').title(), fontsize=12)
                ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create summary plot: {e}")
    
    # Additional convenience methods
    def mollweide_projection(self, map_data: MapData, **kwargs) -> Any:
        """Plot map in Mollweide projection (placeholder implementation)."""
        # This would typically require healpy for proper Mollweide projection
        # For now, provide a basic implementation
        return self.plot_map(map_data, title="Mollweide Projection", **kwargs)
    
    def orthographic_projection(self, map_data: MapData, center: Tuple[float, float],
                               **kwargs) -> Any:
        """Plot map in orthographic projection (placeholder implementation)."""
        return self.plot_map(map_data, title=f"Orthographic Projection (center: {center})", **kwargs)
    
    def gnomonic_projection(self, map_data: MapData, center: Tuple[float, float],
                           size_deg: float, **kwargs) -> Any:
        """Plot map in gnomonic projection (placeholder implementation)."""
        return self.plot_map(map_data, title=f"Gnomonic Projection (center: {center}, size: {size_deg}°)", **kwargs)