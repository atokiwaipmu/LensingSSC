"""
Configuration manager for LensingSSC.

This module provides utilities for loading, validating, and managing
configuration from various sources including files, environment variables,
and programmatic updates.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

from ..base.exceptions import ConfigurationError
from ..base.validation import ConfigValidator
from .settings import ProcessingConfig, AnalysisConfig, VisualizationConfig, CONFIG_SCHEMA
from .loader import ConfigLoader, YAMLConfigLoader, JSONConfigLoader


class ConfigManager:
    """Manager for configuration loading and validation.
    
    This class provides a centralized interface for loading, validating,
    saving, and merging configurations from various sources and formats.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.loaders = {
            '.yaml': YAMLConfigLoader(),
            '.yml': YAMLConfigLoader(),
            '.json': JSONConfigLoader(),
        }
        self.validators = {
            'processing': ConfigValidator(CONFIG_SCHEMA['processing']),
            'analysis': ConfigValidator(CONFIG_SCHEMA['analysis']),
            'visualization': ConfigValidator(CONFIG_SCHEMA['visualization']),
        }
        self._config_cache: Dict[str, Any] = {}
        self._last_modified: Dict[str, float] = {}
    
    def register_loader(self, extension: str, loader: ConfigLoader) -> None:
        """Register a new configuration loader.
        
        Parameters
        ----------
        extension : str
            File extension (including the dot, e.g., '.toml')
        loader : ConfigLoader
            Loader instance for this file type
        """
        self.loaders[extension.lower()] = loader
        logging.debug(f"Registered config loader for {extension}")
    
    def load_config(self, config_path: Union[str, Path], 
                   config_type: str = 'processing',
                   use_cache: bool = True) -> Union[ProcessingConfig, AnalysisConfig, VisualizationConfig]:
        """Load configuration from file.
        
        Parameters
        ----------
        config_path : str or Path
            Path to configuration file
        config_type : str
            Type of configuration ('processing', 'analysis', 'visualization')
        use_cache : bool
            Whether to use cached configuration if available
            
        Returns
        -------
        Config object
            Loaded and validated configuration
            
        Raises
        ------
        ConfigurationError
            If configuration cannot be loaded or is invalid
        """
        config_path = Path(config_path)
        cache_key = f"{config_path}:{config_type}"
        
        # Check cache if enabled
        if use_cache and self._is_cached_valid(config_path, cache_key):
            logging.debug(f"Using cached configuration for {config_path}")
            return self._config_cache[cache_key]
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        # Get appropriate loader
        suffix = config_path.suffix.lower()
        if suffix not in self.loaders:
            available = list(self.loaders.keys())
            raise ConfigurationError(f"Unsupported configuration file format: {suffix}. Available: {available}")
        
        loader = self.loaders[suffix]
        
        # Load configuration data
        try:
            config_data = loader.load(config_path)
            logging.debug(f"Loaded raw config data from {config_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}")
        
        # Validate configuration
        if config_type in self.validators:
            validator = self.validators[config_type]
            if not validator.validate(config_data):
                errors = validator.get_errors()
                raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
            
            # Log warnings if any
            warnings = validator.get_warnings()
            if warnings:
                for warning in warnings:
                    logging.warning(f"Config warning: {warning}")
        
        # Create appropriate configuration object
        config_classes = {
            'processing': ProcessingConfig,
            'analysis': AnalysisConfig,
            'visualization': VisualizationConfig,
        }
        
        if config_type not in config_classes:
            available = list(config_classes.keys())
            raise ConfigurationError(f"Unknown configuration type: {config_type}. Available: {available}")
        
        config_class = config_classes[config_type]
        
        try:
            if hasattr(config_class, 'from_dict'):
                config = config_class.from_dict(config_data)
            else:
                config = config_class(**config_data)
            
            # Cache the result
            if use_cache:
                self._config_cache[cache_key] = config
                self._last_modified[cache_key] = config_path.stat().st_mtime
            
            logging.info(f"Successfully loaded {config_type} configuration from {config_path}")
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create {config_type} configuration: {e}")
    
    def save_config(self, config: Union[ProcessingConfig, AnalysisConfig, VisualizationConfig],
                   config_path: Union[str, Path], overwrite: bool = False) -> None:
        """Save configuration to file.
        
        Parameters
        ----------
        config : Config object
            Configuration to save
        config_path : str or Path
            Output path
        overwrite : bool
            Whether to overwrite existing files
            
        Raises
        ------
        ConfigurationError
            If save operation fails
        """
        config_path = Path(config_path)
        
        # Check if file exists and overwrite is False
        if config_path.exists() and not overwrite:
            raise ConfigurationError(f"Configuration file already exists: {config_path}. Use overwrite=True to replace.")
        
        # Ensure output directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get appropriate loader
        suffix = config_path.suffix.lower()
        if suffix not in self.loaders:
            available = list(self.loaders.keys())
            raise ConfigurationError(f"Unsupported configuration file format: {suffix}. Available: {available}")
        
        loader = self.loaders[suffix]
        
        # Convert config to dictionary
        try:
            if hasattr(config, 'to_dict'):
                config_data = config.to_dict()
            else:
                config_data = config.__dict__
        except Exception as e:
            raise ConfigurationError(f"Failed to convert configuration to dictionary: {e}")
        
        # Save configuration
        try:
            loader.save(config_data, config_path)
            logging.info(f"Configuration saved to {config_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {config_path}: {e}")
    
    def merge_configs(self, *configs: Union[ProcessingConfig, AnalysisConfig, VisualizationConfig]) -> Dict[str, Any]:
        """Merge multiple configurations into a single dictionary.
        
        Parameters
        ----------
        *configs : Config objects
            Configurations to merge
            
        Returns
        -------
        dict
            Merged configuration data
        """
        merged = {}
        
        for config in configs:
            if hasattr(config, 'to_dict'):
                config_data = config.to_dict()
            else:
                config_data = config.__dict__
            
            # Determine section name from config type
            config_type = type(config).__name__.replace('Config', '').lower()
            merged[config_type] = config_data
        
        return merged
    
    def load_multiple_configs(self, config_paths: Dict[str, Union[str, Path]],
                            use_cache: bool = True) -> Dict[str, Any]:
        """Load multiple configuration files.
        
        Parameters
        ----------
        config_paths : dict
            Dictionary mapping config types to file paths
        use_cache : bool
            Whether to use cached configurations
            
        Returns
        -------
        dict
            Dictionary of loaded configurations
        """
        configs = {}
        errors = []
        
        for config_type, config_path in config_paths.items():
            try:
                configs[config_type] = self.load_config(config_path, config_type, use_cache)
                logging.info(f"Loaded {config_type} configuration from {config_path}")
            except ConfigurationError as e:
                error_msg = f"Failed to load {config_type} configuration: {e}"
                errors.append(error_msg)
                logging.error(error_msg)
        
        if errors and not configs:
            # If all configs failed to load, raise an error
            raise ConfigurationError(f"Failed to load any configurations: {'; '.join(errors)}")
        elif errors:
            # If some configs failed, log warnings but continue
            logging.warning(f"Some configurations failed to load: {'; '.join(errors)}")
        
        return configs
    
    def create_default_config(self, config_type: str, 
                            output_path: Optional[Union[str, Path]] = None) -> Union[ProcessingConfig, AnalysisConfig, VisualizationConfig]:
        """Create and optionally save default configuration.
        
        Parameters
        ----------
        config_type : str
            Type of configuration to create
        output_path : str or Path, optional
            Path to save the default configuration
            
        Returns
        -------
        Config object
            Default configuration
            
        Raises
        ------
        ConfigurationError
            If config type is unknown
        """
        config_classes = {
            'processing': ProcessingConfig,
            'analysis': AnalysisConfig,
            'visualization': VisualizationConfig,
        }
        
        if config_type not in config_classes:
            available = list(config_classes.keys())
            raise ConfigurationError(f"Unknown configuration type: {config_type}. Available: {available}")
        
        config_class = config_classes[config_type]
        config = config_class()
        
        if output_path is not None:
            self.save_config(config, output_path, overwrite=True)
            logging.info(f"Default {config_type} configuration saved to {output_path}")
        
        return config
    
    def validate_config_file(self, config_path: Union[str, Path], 
                           config_type: str = 'processing') -> Tuple[bool, List[str], List[str]]:
        """Validate a configuration file without loading it.
        
        Parameters
        ----------
        config_path : str or Path
            Path to configuration file
        config_type : str
            Type of configuration
            
        Returns
        -------
        tuple
            (is_valid, errors, warnings)
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                return False, [f"Configuration file not found: {config_path}"], []
            
            # Get loader and load data
            suffix = config_path.suffix.lower()
            if suffix not in self.loaders:
                return False, [f"Unsupported file format: {suffix}"], []
            
            loader = self.loaders[suffix]
            config_data = loader.load(config_path)
            
            # Validate using schema
            if config_type in self.validators:
                validator = self.validators[config_type]
                is_valid = validator.validate(config_data)
                return is_valid, validator.get_errors(), validator.get_warnings()
            else:
                return True, [], [f"No validator available for config type: {config_type}"]
                
        except Exception as e:
            return False, [f"Validation failed: {e}"], []
    
    def get_config_template(self, config_type: str) -> Dict[str, Any]:
        """Get a configuration template with documentation.
        
        Parameters
        ----------
        config_type : str
            Type of configuration
            
        Returns
        -------
        dict
            Configuration template with comments
            
        Raises
        ------
        ConfigurationError
            If config type is unknown
        """
        templates = {
            'processing': {
                'data_dir': './data',
                'output_dir': './results',
                'cache_dir': './.cache',
                'patch_size_deg': 10.0,
                'xsize': 2048,
                'nside': 8192,
                'num_workers': None,
                'sheet_range': [20, 100],
                'overwrite': False,
                'log_level': 'INFO',
                'chunk_size': 50000,
                'cache_size_mb': 1024,
                'providers': {
                    'healpix': 'lensing_ssc.core.providers.healpix_provider.HealpixProvider',
                    'lenstools': 'lensing_ssc.core.providers.lenstools_provider.LenstoolsProvider',
                },
                '_comments': {
                    'data_dir': 'Directory containing input data',
                    'output_dir': 'Directory for output files',
                    'cache_dir': 'Directory for temporary cache files',
                    'patch_size_deg': 'Patch size in degrees',
                    'xsize': 'Patch resolution in pixels',
                    'nside': 'HEALPix NSIDE parameter (must be power of 2)',
                    'num_workers': 'Number of worker processes (None for auto)',
                    'sheet_range': 'Range of mass sheets to process [start, end]',
                    'chunk_size': 'Number of items to process in each chunk',
                    'cache_size_mb': 'Maximum cache size in megabytes',
                }
            },
            'analysis': {
                'zs_list': [0.5, 1.0, 1.5, 2.0, 2.5],
                'ngal_list': [0, 7, 15, 30, 50],
                'sl_list': [2.0, 5.0, 8.0, 10.0],
                'lmin': 300,
                'lmax': 3000,
                'nbin_ps_bs': 8,
                'nbin_pdf_peaks': 50,
                'epsilon_noise': 0.26,
                'cosmo_params': {
                    'H0': 67.74,
                    'Om0': 0.309,
                },
                '_comments': {
                    'zs_list': 'Source redshift values',
                    'ngal_list': 'Galaxy number densities (per arcmin^2)',
                    'sl_list': 'Smoothing lengths (arcmin)',
                    'lmin': 'Minimum multipole for power spectrum',
                    'lmax': 'Maximum multipole for power spectrum',
                    'nbin_ps_bs': 'Number of bins for power spectrum/bispectrum',
                    'epsilon_noise': 'Intrinsic ellipticity noise',
                }
            },
            'visualization': {
                'figsize': [12, 8],
                'dpi': 150,
                'fontsize': 14,
                'colormap': 'viridis',
                'color_palette': ['tab:blue', 'tab:orange', 'tab:green'],
                'plot_formats': ['pdf', 'png'],
                'save_plots': True,
                'show_plots': False,
                'output_dir': './plots',
                '_comments': {
                    'figsize': 'Default figure size [width, height]',
                    'dpi': 'Figure resolution',
                    'fontsize': 'Default font size',
                    'colormap': 'Default colormap for plots',
                    'plot_formats': 'File formats for saved plots',
                    'output_dir': 'Directory for plot output',
                }
            }
        }
        
        if config_type not in templates:
            available = list(templates.keys())
            raise ConfigurationError(f"Unknown configuration type: {config_type}. Available: {available}")
        
        return templates[config_type]
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._config_cache.clear()
        self._last_modified.clear()
        logging.debug("Configuration cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the configuration cache.
        
        Returns
        -------
        dict
            Cache information including size and keys
        """
        return {
            'cached_configs': len(self._config_cache),
            'cache_keys': list(self._config_cache.keys()),
            'memory_usage_estimate': sum(
                len(str(config)) for config in self._config_cache.values()
            )
        }
    
    def _is_cached_valid(self, config_path: Path, cache_key: str) -> bool:
        """Check if cached configuration is still valid.
        
        Parameters
        ----------
        config_path : Path
            Path to configuration file
        cache_key : str
            Cache key for the configuration
            
        Returns
        -------
        bool
            True if cached config is valid
        """
        if cache_key not in self._config_cache:
            return False
        
        try:
            current_mtime = config_path.stat().st_mtime
            cached_mtime = self._last_modified.get(cache_key, 0)
            return current_mtime <= cached_mtime
        except OSError:
            # File doesn't exist or can't be accessed
            return False


class EnvironmentConfigManager:
    """Manager for environment-based configuration.
    
    This class provides utilities for loading configuration from environment
    variables and setting environment defaults from configuration objects.
    """
    
    def __init__(self, prefix: str = "LENSING_SSC"):
        """Initialize environment config manager.
        
        Parameters
        ----------
        prefix : str
            Environment variable prefix
        """
        self.prefix = prefix
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables.
        
        Returns
        -------
        dict
            Configuration data from environment
        """
        config_data = {}
        prefix_len = len(self.prefix) + 1  # +1 for underscore
        
        for key, value in os.environ.items():
            if key.startswith(f"{self.prefix}_"):
                config_key = key[prefix_len:].lower()
                config_data[config_key] = self._convert_env_value(value)
                self.logger.debug(f"Loaded {config_key}={value} from environment")
        
        return config_data
    
    def set_environment_defaults(self, config: Union[ProcessingConfig, AnalysisConfig, VisualizationConfig]) -> None:
        """Set environment variables from configuration.
        
        Parameters
        ----------
        config : Config object
            Configuration to export to environment
        """
        if hasattr(config, 'to_dict'):
            config_data = config.to_dict()
        else:
            config_data = config.__dict__
        
        count = 0
        for key, value in config_data.items():
            env_key = f"{self.prefix}_{key.upper()}"
            if env_key not in os.environ:
                os.environ[env_key] = str(value)
                count += 1
                self.logger.debug(f"Set environment variable {env_key}={value}")
        
        self.logger.info(f"Set {count} environment variables from configuration")
    
    def get_environment_overrides(self) -> Dict[str, str]:
        """Get all environment variables that override configuration.
        
        Returns
        -------
        dict
            Dictionary of environment overrides
        """
        overrides = {}
        for key, value in os.environ.items():
            if key.startswith(f"{self.prefix}_"):
                overrides[key] = value
        return overrides
    
    def clear_environment_overrides(self) -> int:
        """Clear all environment variables with the configured prefix.
        
        Returns
        -------
        int
            Number of variables cleared
        """
        to_remove = [key for key in os.environ.keys() if key.startswith(f"{self.prefix}_")]
        
        for key in to_remove:
            del os.environ[key]
        
        self.logger.info(f"Cleared {len(to_remove)} environment variables")
        return len(to_remove)
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type.
        
        Parameters
        ----------
        value : str
            Environment variable value
            
        Returns
        -------
        Any
            Converted value
        """
        # Try boolean conversion
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        
        # Try None conversion
        if value.lower() in ('none', 'null', ''):
            return None
        
        # Try integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try list conversion (comma-separated)
        if ',' in value:
            try:
                items = [item.strip() for item in value.split(',')]
                return [self._convert_env_value(item) for item in items]
            except Exception:
                pass
        
        # Try path conversion for path-like strings
        if ('/' in value or '\\' in value) and not value.startswith('http'):
            try:
                return Path(value)
            except Exception:
                pass
        
        # Return as string
        return value


# Convenience functions for common operations
def create_config_from_template(config_type: str, template_overrides: Optional[Dict[str, Any]] = None) -> Union[ProcessingConfig, AnalysisConfig, VisualizationConfig]:
    """Create a configuration from template with optional overrides.
    
    Parameters
    ----------
    config_type : str
        Type of configuration
    template_overrides : dict, optional
        Values to override in the template
        
    Returns
    -------
    Config object
        Created configuration
    """
    manager = ConfigManager()
    template = manager.get_config_template(config_type)
    
    # Remove comments from template
    template = {k: v for k, v in template.items() if not k.startswith('_')}
    
    # Apply overrides
    if template_overrides:
        template.update(template_overrides)
    
    # Create config object
    config_classes = {
        'processing': ProcessingConfig,
        'analysis': AnalysisConfig,
        'visualization': VisualizationConfig,
    }
    
    config_class = config_classes[config_type]
    return config_class.from_dict(template)


def auto_detect_config_files(directory: Union[str, Path], 
                            patterns: Optional[Dict[str, str]] = None) -> Dict[str, Optional[Path]]:
    """Auto-detect configuration files in a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search
    patterns : dict, optional
        Mapping of config types to filename patterns
        
    Returns
    -------
    dict
        Mapping of config types to found file paths
    """
    directory = Path(directory)
    
    if patterns is None:
        patterns = {
            'processing': 'processing',
            'analysis': 'analysis', 
            'visualization': 'visualization',
        }
    
    found_configs = {}
    
    for config_type, pattern in patterns.items():
        found_file = None
        
        # Try different extensions
        for ext in ['.yaml', '.yml', '.json']:
            candidate = directory / f"{pattern}{ext}"
            if candidate.exists():
                found_file = candidate
                break
        
        found_configs[config_type] = found_file
    
    return found_configs