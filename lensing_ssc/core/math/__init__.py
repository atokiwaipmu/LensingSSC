"""
Mathematical utilities for LensingSSC with minimal dependencies.
"""

from .statistics import (
    BasicStatistics,
    RobustStatistics,
    CorrelationAnalysis,
    CovarianceEstimator,
)
from .transforms import (
    FourierTransforms,
    SphericalHarmonics,
    WindowFunctions,
)
from .interpolation import (
    Interpolator1D,
    Interpolator2D,
    SphericalInterpolator,
)

__all__ = [
    # Statistics
    "BasicStatistics",
    "RobustStatistics", 
    "CorrelationAnalysis",
    "CovarianceEstimator",
    # Transforms
    "FourierTransforms",
    "SphericalHarmonics",
    "WindowFunctions",
    # Interpolation
    "Interpolator1D",
    "Interpolator2D",
    "SphericalInterpolator",
]