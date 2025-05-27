"""
Configuration manager for LensingSSC.

This module provides utilities for loading, validating, and managing
configuration from various sources.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
import logging

from ..base.exceptions import ConfigurationError
from ..base.validation import ConfigValidator
from .settings import ProcessingConfig, AnalysisConfig, VisualizationConfig, CONFIG_SCHEMA
from .loader import ConfigLoader, YAMLConfigLoader, JSONConfigLoader


class ConfigManager:
    """Manager for configuration loading and validation."""
    
    def __init__(self):
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
    
    def load_config(self, config_path: Union[str, Path], 
                   config_type: str = 'processing') -> Union[ProcessingConfig, AnalysisConfig, VisualizationConfig]:
        """Load configuration from file.
        
        Parameters
        ----------
        config_path : str or Path
            Path to configuration file
        config_type : str
            Type of configuration ('processing', 'analysis', 'visualization')
            
        Returns
        -------
        Config object
            Loaded and validated configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        # Get appropriate loader
        suffix = config_path.suffix.lower()
        if suffix not in self.loaders:
            raise ConfigurationError(f"Unsupported configuration file format: {suffix}")
        
        loader = self.loaders[suffix]
        
        # Load configuration data
        try:
            config_data = loader.load(config_path)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}")
        
        # Validate configuration
        if config_type in self.validators:
            validator = self.validators[config_type]
            if not validator.validate(config_data):
                errors = validator.get_errors()
                raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
        
        # Create appropriate configuration object
        config_classes = {
            'processing': ProcessingConfig,
            'analysis': AnalysisConfig,
            'visualization': VisualizationConfig,
        }
        
        if config_type not in config_classes:
            raise ConfigurationError(f"Unknown configuration type: {config_type}")
        
        config_class = config_classes[config_type]
        
        try:
            if hasattr(config_class, 'from_dict'):
                return config_class.from_dict(config_data)
            else:
                return config_class(**config_data)
        except Exception as e:
            raise ConfigurationError(f"Failed to create {config_type} configuration: {e}")
    
    def save_config(self, config: Union[ProcessingConfig, AnalysisConfig, VisualizationConfig],
                   config_path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Parameters
        ----------
        config : Config object
            Configuration to save
        config_path : str or Path
            Output path
        """
        config_path = Path(config_path)
        
        # Ensure output directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get appropriate loader
        suffix = config_path.suffix.lower()
        if suffix not in self.loaders:
            raise ConfigurationError(f"Unsupported configuration file format: {suffix}")
        
        loader = self.loaders[suffix]
        
        # Convert config to dictionary
        if hasattr(config, 'to_dict'):
            config_data = config.to_dict()
        else:
            config_data = config.__dict__
        
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
    
    def load_multiple_configs(self, config_paths: Dict[str, Union[str, Path]]) -> Dict[str, Any]:
        """Load multiple configuration files.
        
        Parameters
        ----------
        config_paths : dict
            Dictionary mapping config types to file paths
            
        Returns
        -------
        dict
            Dictionary of loaded configurations
        """
        configs = {}
        
        for config_type, config_path in config_paths.items():
            try:
                configs[config_type] = self.load_config(config_path, config_type)
                logging.info(f"Loaded {config_type} configuration from {config_path}")
            except Exception as e:
                logging.error(f"Failed to load {config_type} configuration: {e}")
                # Continue loading other configs
        
        return configs
    
    def create_default_config(self, config_type: str, output_path: Optional[Union[str, Path]] = None) -> Union[ProcessingConfig, AnalysisConfig, VisualizationConfig]:
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
        """
        config_classes = {
            'processing': ProcessingConfig,
            'analysis': AnalysisConfig,
            'visualization': VisualizationConfig,
        }
        
        if config_type not in config_classes:
            raise ConfigurationError(f"Unknown configuration type: {config_type}")
        
        config_class = config_classes[config_type]
        config = config_class()
        
        if output_path is not None:
            self.save_config(config, output_path)
        
        return config
    
    def validate_config_file(self, config_path: Union[str, Path], 
                           config_type: str = 'processing') -> bool:
        """Validate a configuration file without loading it.
        
        Parameters
        ----------
        config_path : str or Path
            Path to configuration file
        config_type : str
            Type of configuration
            
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        try:
            self.load_config(config_path, config_type)
            return True
        except ConfigurationError:
            return False
    
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
                '_comments': {
                    'data_dir': 'Directory containing input data',
                    'output_dir': 'Directory for output files',
                    'patch_size_deg': 'Patch size in degrees',
                    'xsize': 'Patch resolution in pixels',
                    'nside': 'HEALPix NSIDE parameter (must be power of 2)',
                    'num_workers': 'Number of worker processes (None for auto)',
                }
            },
            'analysis': {
                'zs_list': [0.5, 1.0, 1.5, 2.0, 2.5],
                'ngal_list': [0, 7, 15, 30, 50],
                'sl_list': [2.0, 5.0, 8.0, 10.0],
                'lmin': 300,
                'lmax': 3000,
                'nbin_ps_bs': 8,
                'epsilon_noise': 0.26,
                '_comments': {
                    'zs_list': 'Source redshift values',
                    'ngal_list': 'Galaxy number densities (per arcmin^2)',
                    'sl_list': 'Smoothing lengths (arcmin)',
                    'lmin': 'Minimum multipole for power spectrum',
                    'lmax': 'Maximum multipole for power spectrum',
                }
            },
            'visualization': {
                'figsize': [12, 8],
                'dpi': 150,
                'fontsize': 14,
                'colormap': 'viridis',
                'save_plots': True,
                'show_plots': False,
                '_comments': {
                    'figsize': 'Default figure size [width, height]',
                    'dpi': 'Figure resolution',
                    'fontsize': 'Default font size',
                    'colormap': 'Default colormap for plots',
                }
            }
        }
        
        if config_type not in templates:
            raise ConfigurationError(f"Unknown configuration type: {config_type}")
        
        return templates[config_type]


class EnvironmentConfigManager:
    """Manager for environment-based configuration."""
    
    def __init__(self, prefix: str = "LENSING_SSC"):
        self.prefix = prefix
    
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
        
        return config_data
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Try boolean conversion
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        
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
                return [self._convert_env_value(item.strip()) for item in value.split(',')]
            except:
                pass
        
        # Return as string
        return value
    
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
        
        for key, value in config_data.items():
            env_key = f"{self.prefix}_{key.upper()}"
            if env_key not in os.environ:
                os.environ[env_key] = str(value)