"""
Base classes and data structures with minimal dependencies.
"""

from .exceptions import (
    LensingSSCError,
    ValidationError,
    ConfigurationError,
    ProviderError,
    ProcessingError,
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
)
from .validation import (
    Validator,
    DataValidator,
    ConfigValidator,
)

__all__ = [
    # Exceptions
    "LensingSSCError",
    "ValidationError",
    "ConfigurationError", 
    "ProviderError",
    "ProcessingError",
    # Data structures
    "DataStructure",
    "MapData",
    "PatchData", 
    "StatisticsData",
    # Coordinates
    "Coordinates",
    "SphericalCoordinates",
    "CartesianCoordinates",
    # Validation
    "Validator",
    "DataValidator",
    "ConfigValidator",
]