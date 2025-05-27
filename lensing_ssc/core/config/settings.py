"""
Configuration settings and management for LensingSSC.

This module provides centralized configuration management with validation,
type checking, and environment variable support.
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
    """Main configuration for preprocessing and data processing."""
    
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
    
    # Provider settings
    providers: Dict[str, str] = field(default_factory=lambda: {
        'healpix': 'lensing_ssc.providers.healpix_provider.HealpixProvider',
        'lenstools': 'lensing_ssc.providers.lenstools_provider.LenstoolsProvider',
        'nbodykit': 'lensing_ssc.providers.nbodykit_provider.NbodykitProvider',
        'plotting': 'lensing_ssc.providers.matplotlib_provider.MatplotlibProvider',
    })
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    enable_progress_bar: bool = True
    
    # Validation
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
        """Validate configuration parameters."""
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
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
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
        """Create configuration from dictionary."""
        # Handle path conversion
        path_fields = ['data_dir', 'output_dir', 'cache_dir', 'log_file']
        for field_name in path_fields:
            if field_name in data and data[field_name] is not None:
                data[field_name] = Path(data[field_name])
        
        return cls(**data)
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ConfigurationError(f"Unknown configuration parameter: {key}")
        
        # Re-validate after update
        self.validate()
    
    def get_provider_class(self, provider_type: str) -> str:
        """Get provider class name for a given type."""
        if provider_type not in self.providers:
            raise ConfigurationError(f"Unknown provider type: {provider_type}")
        return self.providers[provider_type]


@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis."""
    
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
    
    # Noise parameters
    epsilon_noise: float = 0.26
    
    # Cosmological parameters
    cosmo_params: Dict[str, float] = field(default_factory=lambda: {
        'H0': 67.74,
        'Om0': 0.309,
    })
    
    def validate(self) -> None:
        """Validate analysis configuration."""
        errors = []
        
        if self.lmin >= self.lmax:
            errors.append("lmin must be less than lmax")
        
        if self.nbin_ps_bs <= 0:
            errors.append("nbin_ps_bs must be positive")
        
        if self.nbin_pdf_peaks <= 0:
            errors.append("nbin_pdf_peaks must be positive")
        
        if self.pdf_peaks_range[0] >= self.pdf_peaks_range[1]:
            errors.append("pdf_peaks_range must be (min, max) with min < max")
        
        if self.epsilon_noise <= 0:
            errors.append("epsilon_noise must be positive")
        
        # Validate redshift list
        if not self.zs_list or any(z <= 0 for z in self.zs_list):
            errors.append("zs_list must contain positive redshift values")
        
        # Validate galaxy density list
        if not self.ngal_list or any(n < 0 for n in self.ngal_list):
            errors.append("ngal_list must contain non-negative values")
        
        # Validate smoothing length list
        if not self.sl_list or any(sl <= 0 for sl in self.sl_list):
            errors.append("sl_list must contain positive smoothing lengths")
        
        if errors:
            raise ConfigurationError(f"Analysis configuration validation failed: {'; '.join(errors)}")


@dataclass
class VisualizationConfig:
    """Configuration for visualization and plotting."""
    
    # Figure settings
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 150
    fontsize: int = 14
    
    # Color settings
    colormap: str = "viridis"
    color_palette: List[str] = field(default_factory=lambda: [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"
    ])
    
    # Plot types
    plot_formats: List[str] = field(default_factory=lambda: ["pdf", "png"])
    save_plots: bool = True
    show_plots: bool = False
    
    # Specific plot settings
    correlation_vmin: float = -1.0
    correlation_vmax: float = 1.0
    ratio_vmin: float = 0.6
    ratio_vmax: float = 1.4
    
    # Layout settings
    subplot_hspace: float = 0.3
    subplot_wspace: float = 0.2
    tight_layout: bool = True
    
    def validate(self) -> None:
        """Validate visualization configuration."""
        errors = []
        
        if len(self.figsize) != 2 or any(s <= 0 for s in self.figsize):
            errors.append("figsize must be a tuple of two positive numbers")
        
        if self.dpi <= 0:
            errors.append("dpi must be positive")
        
        if self.fontsize <= 0:
            errors.append("fontsize must be positive")
        
        if self.correlation_vmin >= self.correlation_vmax:
            errors.append("correlation_vmin must be less than correlation_vmax")
        
        if self.ratio_vmin >= self.ratio_vmax:
            errors.append("ratio_vmin must be less than ratio_vmax")
        
        if errors:
            raise ConfigurationError(f"Visualization configuration validation failed: {'; '.join(errors)}")


def get_config() -> ProcessingConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = ProcessingConfig()
    return _global_config


def set_config(config: ProcessingConfig) -> None:
    """Set the global configuration instance."""
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
    """Update global configuration parameters."""
    config = get_config()
    config.update(**kwargs)


def load_config_from_env() -> ProcessingConfig:
    """Load configuration from environment variables."""
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
            elif attr_name in ['xsize', 'nside', 'num_workers']:
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
            'patch_size_deg': (int, float),
            'xsize': int,
            'nside': int,
            'num_workers': (int, type(None)),
            'sheet_range': tuple,
            'overwrite': bool,
            'log_level': str,
        },
        'validators': {
            'patch_size_deg': lambda x: 0 < x <= 180,
            'xsize': lambda x: x > 0,
            'nside': lambda x: x > 0 and (x & (x - 1)) == 0,
            'num_workers': lambda x: x is None or x > 0,
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
            'epsilon_noise': (int, float),
        },
        'validators': {
            'lmin': lambda x: x > 0,
            'lmax': lambda x: x > 0,
            'nbin_ps_bs': lambda x: x > 0,
            'epsilon_noise': lambda x: x > 0,
        }
    },
    'visualization': {
        'required': [],
        'types': {
            'figsize': tuple,
            'dpi': int,
            'fontsize': int,
            'save_plots': bool,
            'show_plots': bool,
        },
        'validators': {
            'dpi': lambda x: x > 0,
            'fontsize': lambda x: x > 0,
        }
    }
}