"""
Configuration management for LensingSSC.

This module provides centralized configuration management with validation,
type checking, and environment variable support. The configuration system
supports multiple file formats and provides both global and local configuration
management capabilities.
"""

from .settings import (
    ProcessingConfig,
    AnalysisConfig,
    VisualizationConfig,
    get_config,
    set_config,
    reset_config,
    update_config,
    load_config_from_env,
    CONFIG_SCHEMA,
)
from .manager import (
    ConfigManager,
    EnvironmentConfigManager,
)
from .loader import (
    ConfigLoader,
    YAMLConfigLoader,
    JSONConfigLoader,
    TOMLConfigLoader,
    INIConfigLoader,
    get_config_loader,
)

__all__ = [
    # Configuration classes
    "ProcessingConfig",
    "AnalysisConfig", 
    "VisualizationConfig",
    # Global config functions
    "get_config",
    "set_config",
    "reset_config",
    "update_config",
    "load_config_from_env",
    "CONFIG_SCHEMA",
    # Management classes
    "ConfigManager",
    "EnvironmentConfigManager",
    # Loader classes
    "ConfigLoader",
    "YAMLConfigLoader",
    "JSONConfigLoader",
    "TOMLConfigLoader",
    "INIConfigLoader",
    "get_config_loader",
]