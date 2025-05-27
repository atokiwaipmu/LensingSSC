"""
Abstract interfaces for dependency injection and provider pattern.
"""

from .data_interface import (
    DataProvider,
    MapProvider,
    CatalogProvider,
)
from .compute_interface import (
    ComputeProvider,
    StatisticsProvider,
    GeometryProvider,
)
from .storage_interface import (
    StorageProvider,
    FileFormatProvider,
)
from .plotting_interface import (
    PlottingProvider,
    VisualizationProvider,
)

__all__ = [
    # Data interfaces
    "DataProvider",
    "MapProvider", 
    "CatalogProvider",
    # Compute interfaces
    "ComputeProvider",
    "StatisticsProvider",
    "GeometryProvider",
    # Storage interfaces
    "StorageProvider",
    "FileFormatProvider",
    # Plotting interfaces
    "PlottingProvider",
    "VisualizationProvider",
]