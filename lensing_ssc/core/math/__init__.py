"""
Mathematical utilities for LensingSSC with minimal dependencies.

This module provides core mathematical functions for statistical analysis,
transforms, and interpolation using only numpy and scipy.
"""

from .statistics import (
    BasicStatistics,
    RobustStatistics,
    CorrelationAnalysis,
    CovarianceEstimator,
    PowerSpectrumEstimator,
)
from .transforms import (
    FourierTransforms,
    SphericalHarmonics,
    WindowFunctions,
    FilterOperations,
)
from .interpolation import (
    Interpolator1D,
    Interpolator2D,
    SphericalInterpolator,
    AdaptiveInterpolator,
    MultiScaleInterpolator,
)

__all__ = [
    # Statistics
    "BasicStatistics",
    "RobustStatistics", 
    "CorrelationAnalysis",
    "CovarianceEstimator",
    "PowerSpectrumEstimator",
    # Transforms
    "FourierTransforms",
    "SphericalHarmonics",
    "WindowFunctions",
    "FilterOperations",
    # Interpolation
    "Interpolator1D",
    "Interpolator2D",
    "SphericalInterpolator",
    "AdaptiveInterpolator",
    "MultiScaleInterpolator",
]