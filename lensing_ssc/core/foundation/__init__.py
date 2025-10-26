"""Foundation components with minimal dependencies."""

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
    reraise_with_context,
    validate_not_none,
    validate_type,
    validate_positive,
    validate_range,
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
    "reraise_with_context",
    "validate_not_none",
    "validate_type",
    "validate_positive",
    "validate_range",
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
