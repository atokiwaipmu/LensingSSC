"""
Abstract interfaces for dependency injection and provider pattern.
"""

from .data_interface import (
    DataProvider,
    MapProvider,
    CatalogProvider,
    ConvergenceMapProvider,
)
from .compute_interface import (
    ComputeProvider,
    StatisticsProvider,
    GeometryProvider,
    OptimizationProvider,
    InterpolationProvider,
    FilteringProvider,
)
from .storage_interface import (
    StorageProvider,
    FileFormatProvider,
    FITSProvider,
    HDF5Provider,
    CSVProvider,
    NPYProvider,
    CacheProvider,
    CheckpointProvider,
    CompressionProvider,
)
from .plotting_interface import (
    PlottingProvider,
    MapPlottingProvider,
    StatisticsPlottingProvider,
    InteractivePlottingProvider,
    VisualizationProvider,
)

__all__ = [
    # Data interfaces
    "DataProvider",
    "MapProvider", 
    "CatalogProvider",
    "ConvergenceMapProvider",
    # Compute interfaces
    "ComputeProvider",
    "StatisticsProvider",
    "GeometryProvider",
    "OptimizationProvider",
    "InterpolationProvider",
    "FilteringProvider",
    # Storage interfaces
    "StorageProvider",
    "FileFormatProvider",
    "FITSProvider",
    "HDF5Provider",
    "CSVProvider",
    "NPYProvider",
    "CacheProvider",
    "CheckpointProvider",
    "CompressionProvider",
    # Plotting interfaces
    "PlottingProvider",
    "MapPlottingProvider",
    "StatisticsPlottingProvider",
    "InteractivePlottingProvider",
    "VisualizationProvider",
]