"""
Base classes and data structures with minimal dependencies.

This module provides the foundational components that other modules build upon,
including base classes, data structures, coordinate systems, and validation utilities.
"""

from .exceptions import (
    LensingSSCError,
    ValidationError,
    ConfigurationError,
    ProviderError,
    ProcessingError,
    DataError,
    GeometryError,
    StatisticsError,
    IOError,
    VisualizationError,
)
from .data_structures import (
    DataStructure,
    MapData,
    PatchData,
    StatisticsData,
)
from .coordinates import (
    Coordinates,
    SphericalCoordinates,
    CartesianCoordinates,
    CoordinateTransformer,
    RotationMatrix,
)
from .validation import (
    Validator,
    DataValidator,
    ConfigValidator,
    PathValidator,
    RangeValidator,
    validate_spherical_coordinates,
    validate_patch_size,
    validate_nside,
)

__all__ = [
    # Exceptions
    "LensingSSCError",
    "ValidationError",
    "ConfigurationError", 
    "ProviderError",
    "ProcessingError",
    "DataError",
    "GeometryError",
    "StatisticsError",
    "IOError",
    "VisualizationError",
    # Data structures
    "DataStructure",
    "MapData",
    "PatchData", 
    "StatisticsData",
    # Coordinates
    "Coordinates",
    "SphericalCoordinates",
    "CartesianCoordinates",
    "CoordinateTransformer",
    "RotationMatrix",
    # Validation
    "Validator",
    "DataValidator",
    "ConfigValidator",
    "PathValidator",
    "RangeValidator",
    "validate_spherical_coordinates",
    "validate_patch_size",
    "validate_nside",
]