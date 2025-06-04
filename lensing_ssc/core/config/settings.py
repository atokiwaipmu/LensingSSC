"""
Configuration settings and management for LensingSSC.

This module provides centralized configuration management with validation,
type checking, and environment variable support. All configuration classes
use dataclasses for clean, type-safe configuration handling.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

from ..base.exceptions import ConfigurationError


# Global configuration instance
_global_config: Optional["ProcessingConfig"] = None


@dataclass
class ProcessingConfig:
    """Main configuration for preprocessing and data processing.
    
    This configuration class handles all parameters related to data processing,
    including file paths, processing parameters, memory management, and provider settings.
    """
    
    # Data paths
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "results")
    cache_dir: Path = field(default_factory=lambda: Path.cwd() / ".cache")
    
    # Processing parameters
    patch_size_deg: float = 10.0
    xsize: int = 2048
    nside: int = 8192
    num_workers: Optional[int] = None
    
    # Data processing
    sheet_range: Tuple[int, int] = (20, 100)
    extra_index: int = 100
    overwrite: bool = False
    resume: bool = True
    
    # Memory management
    chunk_size: int = 50000
    cache_size_mb: int = 1024
    memory_limit_mb: Optional[int] = None
    cleanup_interval: int = 50
    mmap_threshold: int = 1000000
    max_cache_entries: int = 1000
    
    # Parallel processing
    batch_size: int = 10
    
    # Provider settings
    providers: Dict[str, str] = field(default_factory=lambda: {
        'healpix': 'lensing_ssc.core.providers.healpix_provider.HealpixProvider',
        'lenstools': 'lensing_ssc.core.providers.lenstools_provider.LenstoolsProvider',
        'nbodykit': 'lensing_ssc.core.providers.nbodykit_provider.NbodykitProvider',
        'plotting': 'lensing_ssc.core.providers.matplotlib_provider.MatplotlibProvider',
    })
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    enable_progress_bar: bool = True
    checkpoint_interval: int = 10
    
    # Data validation
    validate_input: bool = True
    strict_validation: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Convert string paths to Path objects
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        
        if self.log_file is not None:
            self.log_file = Path(self.log_file)
        
        # Set number of workers if not specified
        if self.num_workers is None:
            self.num_workers = os.cpu_count()
        
        # Validate configuration
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises
        ------
        ConfigurationError
            If any configuration parameter is invalid
        """
        errors = []
        
        # Validate numeric parameters
        if self.patch_size_deg <= 0 or self.patch_size_deg > 180:
            errors.append("patch_size_deg must be between 0 and 180 degrees")
        
        if self.xsize <= 0:
            errors.append("xsize must be positive")
        
        if self.nside <= 0 or (self.nside & (self.nside - 1)) != 0:
            errors.append("nside must be a positive power of 2")
        
        if self.num_workers is not None and self.num_workers <= 0:
            errors.append("num_workers must be positive or None")
        
        if self.sheet_range[0] >= self.sheet_range[1]:
            errors.append("sheet_range must be (start, end) with start < end")
        
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.cache_size_mb <= 0:
            errors.append("cache_size_mb must be positive")
        
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            errors.append("memory_limit_mb must be positive or None")
        
        if self.cleanup_interval <= 0:
            errors.append("cleanup_interval must be positive")
        
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.checkpoint_interval <= 0:
            errors.append("checkpoint_interval must be positive")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}")
        
        # Validate sheet range
        if not isinstance(self.sheet_range, (tuple, list)) or len(self.sheet_range) != 2:
            errors.append("sheet_range must be a tuple or list of length 2")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of configuration
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingConfig":
        """Create configuration from dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary containing configuration data
            
        Returns
        -------
        ProcessingConfig
            New configuration instance
        """
        # Handle path conversion
        path_fields = ['data_dir', 'output_dir', 'cache_dir', 'log_file']
        for field_name in path_fields:
            if field_name in data and data[field_name] is not None:
                data[field_name] = Path(data[field_name])
        
        # Handle tuple conversion
        if 'sheet_range' in data and isinstance(data['sheet_range'], list):
            data['sheet_range'] = tuple(data['sheet_range'])
        
        return cls(**data)
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters.
        
        Parameters
        ----------
        **kwargs
            Configuration parameters to update
            
        Raises
        ------
        ConfigurationError
            If unknown parameter or validation fails
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ConfigurationError(f"Unknown configuration parameter: {key}")
        
        # Re-validate after update
        self.validate()
    
    def get_provider_class(self, provider_type: str) -> str:
        """Get provider class name for a given type.
        
        Parameters
        ----------
        provider_type : str
            Type of provider ('healpix', 'lenstools', etc.)
            
        Returns
        -------
        str
            Fully qualified class name
            
        Raises
        ------
        ConfigurationError
            If provider type is unknown
        """
        if provider_type not in self.providers:
            available = list(self.providers.keys())
            raise ConfigurationError(f"Unknown provider type: {provider_type}. Available: {available}")
        return self.providers[provider_type]
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data_dir, self.output_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_memory_limit_bytes(self) -> Optional[int]:
        """Get memory limit in bytes.
        
        Returns
        -------
        int or None
            Memory limit in bytes, or None if not set
        """
        if self.memory_limit_mb is None:
            return None
        return self.memory_limit_mb * 1024 * 1024
    
    def get_cache_size_bytes(self) -> int:
        """Get cache size in bytes.
        
        Returns
        -------
        int
            Cache size in bytes
        """
        return self.cache_size_mb * 1024 * 1024


@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis.
    
    This configuration class handles parameters for statistical analysis
    including source redshifts, galaxy densities, and analysis parameters.
    """
    
    # Source redshifts
    zs_list: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5])
    
    # Galaxy densities (per arcmin^2)
    ngal_list: List[int] = field(default_factory=lambda: [0, 7, 15, 30, 50])
    
    # Smoothing lengths (arcmin)
    sl_list: List[float] = field(default_factory=lambda: [2.0, 5.0, 8.0, 10.0])
    
    # Power spectrum/bispectrum parameters
    lmin: int = 300
    lmax: int = 3000
    nbin_ps_bs: int = 8
    
    # PDF/peak counts parameters
    nbin_pdf_peaks: int = 50
    pdf_peaks_range: Tuple[float, float] = (-5.0, 5.0)
    
    # Minkowski functionals parameters
    nbin_mf: int = 50
    mf_range: Tuple[float, float] = (-3.0, 3.0)
    
    # Noise parameters
    epsilon_noise: float = 0.26
    shape_noise_std: float = 0.3
    
    # Statistical parameters
    bootstrap_samples: int = 1000
    confidence_level: float = 0.68
    
    # Cosmological parameters
    cosmo_params: Dict[str, float] = field(default_factory=lambda: {
        'H0': 67.74,
        'Om0': 0.309,
        'Ob0': 0.0486,
        'sigma8': 0.8159,
        'ns': 0.9667,
    })
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.validate()
    
    def validate(self) -> None:
        """Validate analysis configuration.
        
        Raises
        ------
        ConfigurationError
            If any configuration parameter is invalid
        """
        errors = []
        
        # Validate multipole range
        if self.lmin >= self.lmax:
            errors.append("lmin must be less than lmax")
        
        if self.lmin <= 0:
            errors.append("lmin must be positive")
        
        # Validate bin numbers
        if self.nbin_ps_bs <= 0:
            errors.append("nbin_ps_bs must be positive")
        
        if self.nbin_pdf_peaks <= 0:
            errors.append("nbin_pdf_peaks must be positive")
        
        if self.nbin_mf <= 0:
            errors.append("nbin_mf must be positive")
        
        # Validate ranges
        if self.pdf_peaks_range[0] >= self.pdf_peaks_range[1]:
            errors.append("pdf_peaks_range must be (min, max) with min < max")
        
        if self.mf_range[0] >= self.mf_range[1]:
            errors.append("mf_range must be (min, max) with min < max")
        
        # Validate noise parameters
        if self.epsilon_noise <= 0:
            errors.append("epsilon_noise must be positive")
        
        if self.shape_noise_std <= 0:
            errors.append("shape_noise_std must be positive")
        
        # Validate statistical parameters
        if self.bootstrap_samples <= 0:
            errors.append("bootstrap_samples must be positive")
        
        if not 0 < self.confidence_level < 1:
            errors.append("confidence_level must be between 0 and 1")
        
        # Validate redshift list
        if not self.zs_list or any(z <= 0 for z in self.zs_list):
            errors.append("zs_list must contain positive redshift values")
        
        # Validate galaxy density list
        if not self.ngal_list or any(n < 0 for n in self.ngal_list):
            errors.append("ngal_list must contain non-negative values")
        
        # Validate smoothing length list
        if not self.sl_list or any(sl <= 0 for sl in self.sl_list):
            errors.append("sl_list must contain positive smoothing lengths")
        
        # Validate cosmological parameters
        cosmo_errors = self._validate_cosmology()
        errors.extend(cosmo_errors)
        
        if errors:
            raise ConfigurationError(f"Analysis configuration validation failed: {'; '.join(errors)}")
    
    def _validate_cosmology(self) -> List[str]:
        """Validate cosmological parameters.
        
        Returns
        -------
        list
            List of validation error messages
        """
        errors = []
        
        required_params = ['H0', 'Om0']
        for param in required_params:
            if param not in self.cosmo_params:
                errors.append(f"Missing required cosmological parameter: {param}")
        
        # Validate parameter ranges
        if 'H0' in self.cosmo_params:
            if not 50 <= self.cosmo_params['H0'] <= 100:
                errors.append("H0 should be between 50 and 100 km/s/Mpc")
        
        if 'Om0' in self.cosmo_params:
            if not 0.1 <= self.cosmo_params['Om0'] <= 1.0:
                errors.append("Om0 should be between 0.1 and 1.0")
        
        if 'Ob0' in self.cosmo_params:
            if not 0.01 <= self.cosmo_params['Ob0'] <= 0.1:
                errors.append("Ob0 should be between 0.01 and 0.1")
        
        if 'sigma8' in self.cosmo_params:
            if not 0.5 <= self.cosmo_params['sigma8'] <= 1.5:
                errors.append("sigma8 should be between 0.5 and 1.5")
        
        if 'ns' in self.cosmo_params:
            if not 0.8 <= self.cosmo_params['ns'] <= 1.2:
                errors.append("ns should be between 0.8 and 1.2")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {key: value for key, value in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisConfig":
        """Create from dictionary representation."""
        # Handle tuple conversion
        if 'pdf_peaks_range' in data and isinstance(data['pdf_peaks_range'], list):
            data['pdf_peaks_range'] = tuple(data['pdf_peaks_range'])
        
        if 'mf_range' in data and isinstance(data['mf_range'], list):
            data['mf_range'] = tuple(data['mf_range'])
        
        return cls(**data)
    
    def get_l_bins(self) -> Tuple[List[float], List[float]]:
        """Get multipole bins for power spectrum analysis.
        
        Returns
        -------
        tuple
            (l_edges, l_centers) for binning
        """
        l_edges = [self.lmin * (self.lmax / self.lmin) ** (i / self.nbin_ps_bs) 
                  for i in range(self.nbin_ps_bs + 1)]
        l_centers = [(l_edges[i] + l_edges[i+1]) / 2 for i in range(self.nbin_ps_bs)]
        return l_edges, l_centers


@dataclass
class VisualizationConfig:
    """Configuration for visualization and plotting.
    
    This configuration class handles all parameters related to plotting
    and visualization, including figure settings, colors, and output formats.
    """
    
    # Figure settings
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 150
    fontsize: int = 14
    
    # Color settings
    colormap: str = "viridis"
    color_palette: List[str] = field(default_factory=lambda: [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
    ])
    
    # Plot types and formats
    plot_formats: List[str] = field(default_factory=lambda: ["pdf", "png"])
    save_plots: bool = True
    show_plots: bool = False
    
    # Specific plot settings
    correlation_vmin: float = -1.0
    correlation_vmax: float = 1.0
    ratio_vmin: float = 0.6
    ratio_vmax: float = 1.4
    
    # Map plotting
    map_projection: str = "mollweide"
    map_colormap: str = "RdBu_r"
    map_symmetric: bool = True
    
    # Layout settings
    subplot_hspace: float = 0.3
    subplot_wspace: float = 0.2
    tight_layout: bool = True
    legend_fontsize: int = 12
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "plots")
    file_prefix: str = ""
    file_suffix: str = ""
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Convert string path to Path object
        self.output_dir = Path(self.output_dir)
        self.validate()
    
    def validate(self) -> None:
        """Validate visualization configuration.
        
        Raises
        ------
        ConfigurationError
            If any configuration parameter is invalid
        """
        errors = []
        
        # Validate figure size
        if len(self.figsize) != 2 or any(s <= 0 for s in self.figsize):
            errors.append("figsize must be a tuple of two positive numbers")
        
        # Validate DPI and font sizes
        if self.dpi <= 0:
            errors.append("dpi must be positive")
        
        if self.fontsize <= 0:
            errors.append("fontsize must be positive")
        
        if self.legend_fontsize <= 0:
            errors.append("legend_fontsize must be positive")
        
        # Validate ranges
        if self.correlation_vmin >= self.correlation_vmax:
            errors.append("correlation_vmin must be less than correlation_vmax")
        
        if self.ratio_vmin >= self.ratio_vmax:
            errors.append("ratio_vmin must be less than ratio_vmax")
        
        # Validate spacing
        if not 0 <= self.subplot_hspace <= 1:
            errors.append("subplot_hspace must be between 0 and 1")
        
        if not 0 <= self.subplot_wspace <= 1:
            errors.append("subplot_wspace must be between 0 and 1")
        
        # Validate color palette
        if not self.color_palette:
            errors.append("color_palette cannot be empty")
        
        # Validate plot formats
        valid_formats = ["pdf", "png", "svg", "eps", "jpg", "jpeg", "tiff"]
        invalid_formats = [fmt for fmt in self.plot_formats if fmt.lower() not in valid_formats]
        if invalid_formats:
            errors.append(f"Invalid plot formats: {invalid_formats}. Valid: {valid_formats}")
        
        # Validate map projection
        valid_projections = ["mollweide", "aitoff", "hammer", "lambert", "polar"]
        if self.map_projection not in valid_projections:
            errors.append(f"Invalid map projection: {self.map_projection}. Valid: {valid_projections}")
        
        if errors:
            raise ConfigurationError(f"Visualization configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualizationConfig":
        """Create from dictionary representation."""
        # Handle path conversion
        if 'output_dir' in data:
            data['output_dir'] = Path(data['output_dir'])
        
        # Handle tuple conversion
        if 'figsize' in data and isinstance(data['figsize'], list):
            data['figsize'] = tuple(data['figsize'])
        
        return cls(**data)
    
    def get_figure_path(self, name: str, format: str = None) -> Path:
        """Get full path for a figure file.
        
        Parameters
        ----------
        name : str
            Base name for the figure
        format : str, optional
            File format (if None, uses first format in plot_formats)
            
        Returns
        -------
        Path
            Full path for the figure file
        """
        if format is None:
            format = self.plot_formats[0]
        
        filename = f"{self.file_prefix}{name}{self.file_suffix}.{format}"
        return self.output_dir / filename


def get_config() -> ProcessingConfig:
    """Get the global configuration instance.
    
    Returns
    -------
    ProcessingConfig
        Global configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ProcessingConfig()
    return _global_config


def set_config(config: ProcessingConfig) -> None:
    """Set the global configuration instance.
    
    Parameters
    ----------
    config : ProcessingConfig
        Configuration instance to set as global
        
    Raises
    ------
    TypeError
        If config is not a ProcessingConfig instance
    """
    global _global_config
    if not isinstance(config, ProcessingConfig):
        raise TypeError("config must be a ProcessingConfig instance")
    config.validate()
    _global_config = config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _global_config
    _global_config = ProcessingConfig()


def update_config(**kwargs) -> None:
    """Update global configuration parameters.
    
    Parameters
    ----------
    **kwargs
        Configuration parameters to update
    """
    config = get_config()
    config.update(**kwargs)


def load_config_from_env() -> ProcessingConfig:
    """Load configuration from environment variables.
    
    Returns
    -------
    ProcessingConfig
        Configuration loaded from environment
    """
    config = ProcessingConfig()
    
    # Map environment variables to config attributes
    env_mapping = {
        'LENSING_SSC_DATA_DIR': 'data_dir',
        'LENSING_SSC_OUTPUT_DIR': 'output_dir',
        'LENSING_SSC_CACHE_DIR': 'cache_dir',
        'LENSING_SSC_PATCH_SIZE': 'patch_size_deg',
        'LENSING_SSC_XSIZE': 'xsize',
        'LENSING_SSC_NSIDE': 'nside',
        'LENSING_SSC_NUM_WORKERS': 'num_workers',
        'LENSING_SSC_LOG_LEVEL': 'log_level',
        'LENSING_SSC_LOG_FILE': 'log_file',
        'LENSING_SSC_OVERWRITE': 'overwrite',
        'LENSING_SSC_RESUME': 'resume',
        'LENSING_SSC_MEMORY_LIMIT': 'memory_limit_mb',
        'LENSING_SSC_CHUNK_SIZE': 'chunk_size',
        'LENSING_SSC_CACHE_SIZE': 'cache_size_mb',
    }
    
    updates = {}
    for env_var, attr_name in env_mapping.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            
            # Type conversion based on attribute
            if attr_name in ['data_dir', 'output_dir', 'cache_dir', 'log_file']:
                if value:  # Only convert if not empty
                    updates[attr_name] = Path(value)
            elif attr_name in ['patch_size_deg']:
                updates[attr_name] = float(value)
            elif attr_name in ['xsize', 'nside', 'num_workers', 'memory_limit_mb', 'chunk_size', 'cache_size_mb']:
                updates[attr_name] = int(value) if value else None
            elif attr_name in ['overwrite', 'resume']:
                updates[attr_name] = value.lower() in ('true', '1', 'yes', 'on')
            else:
                updates[attr_name] = value
    
    if updates:
        config.update(**updates)
        logging.info(f"Updated configuration from environment variables: {list(updates.keys())}")
    
    return config


# Configuration schema for validation
CONFIG_SCHEMA = {
    'processing': {
        'required': ['data_dir', 'output_dir'],
        'types': {
            'data_dir': (str, Path),
            'output_dir': (str, Path),
            'cache_dir': (str, Path),
            'patch_size_deg': (int, float),
            'xsize': int,
            'nside': int,
            'num_workers': (int, type(None)),
            'sheet_range': (tuple, list),
            'overwrite': bool,
            'resume': bool,
            'log_level': str,
            'chunk_size': int,
            'cache_size_mb': int,
            'memory_limit_mb': (int, type(None)),
            'cleanup_interval': int,
            'batch_size': int,
            'providers': dict,
        },
        'validators': {
            'patch_size_deg': lambda x: 0 < x <= 180,
            'xsize': lambda x: x > 0,
            'nside': lambda x: x > 0 and (x & (x - 1)) == 0,
            'num_workers': lambda x: x is None or x > 0,
            'chunk_size': lambda x: x > 0,
            'cache_size_mb': lambda x: x > 0,
            'memory_limit_mb': lambda x: x is None or x > 0,
            'cleanup_interval': lambda x: x > 0,
            'batch_size': lambda x: x > 0,
        }
    },
    'analysis': {
        'required': ['zs_list', 'ngal_list', 'sl_list'],
        'types': {
            'zs_list': list,
            'ngal_list': list,
            'sl_list': list,
            'lmin': int,
            'lmax': int,
            'nbin_ps_bs': int,
            'nbin_pdf_peaks': int,
            'epsilon_noise': (int, float),
            'bootstrap_samples': int,
            'confidence_level': float,
            'cosmo_params': dict,
        },
        'validators': {
            'lmin': lambda x: x > 0,
            'lmax': lambda x: x > 0,
            'nbin_ps_bs': lambda x: x > 0,
            'nbin_pdf_peaks': lambda x: x > 0,
            'epsilon_noise': lambda x: x > 0,
            'bootstrap_samples': lambda x: x > 0,
            'confidence_level': lambda x: 0 < x < 1,
        }
    },
    'visualization': {
        'required': [],
        'types': {
            'figsize': (tuple, list),
            'dpi': int,
            'fontsize': int,
            'save_plots': bool,
            'show_plots': bool,
            'color_palette': list,
            'plot_formats': list,
            'output_dir': (str, Path),
        },
        'validators': {
            'dpi': lambda x: x > 0,
            'fontsize': lambda x: x > 0,
        }
    }
}