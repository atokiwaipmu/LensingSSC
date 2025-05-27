"""
Core functionality for LensingSSC with minimal dependencies.

This module provides the foundational components that other modules build upon,
including base classes, interfaces, mathematical utilities, and configuration.
"""

from lensing_ssc.core.base import (
    LensingSSCError,
    ValidationError,
    ConfigurationError,
    DataStructure,
    Coordinates,
)
from lensing_ssc.core.config import (
    get_config,
    set_config,
    ConfigManager,
    ProcessingConfig,
)

__all__ = [
    # Exceptions
    "LensingSSCError",
    "ValidationError", 
    "ConfigurationError",
    # Base classes
    "DataStructure",
    "Coordinates",
    # Configuration
    "get_config",
    "set_config", 
    "ConfigManager",
    "ProcessingConfig",
]