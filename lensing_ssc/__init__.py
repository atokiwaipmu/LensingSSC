"""
LensingSSC: A Python package for studying Super-Sample Covariance effects 
in weak gravitational lensing simulations.
"""

__version__ = "0.2.0"
__author__ = "Akira Tokiwa, Adrian E. Bayer, Jia Liu, Masahiro Takada"
__email__ = "akira.tokiwa@example.com"

# Core imports that should always be available
from lensing_ssc.core.base.exceptions import LensingSSCError
from lensing_ssc.core.config import get_config, Config
from lensing_ssc.api.client import LensingSSCClient
from lensing_ssc.core.base import MapData, PatchData

# Conditional imports for heavy dependencies
try:
    from lensing_ssc.api.client import LensingSSCClient
    _HAS_FULL_API = True
except ImportError:
    _HAS_FULL_API = False

# Public API
__all__ = [
    "__version__",
    "LensingSSCError",
    "get_config",
    "Config",
    "LensingSSCClient",
    "MapData",
    "PatchData",
]

if _HAS_FULL_API:
    __all__.append("LensingSSCClient")


def require_heavy_dependencies():
    """Check if heavy dependencies are available."""
    missing = []
    
    try:
        import healpy
    except ImportError:
        missing.append("healpy")
    
    try:
        import lenstools
    except ImportError:
        missing.append("lenstools")
    
    try:
        import nbodykit
    except ImportError:
        missing.append("nbodykit")
    
    if missing:
        raise ImportError(
            f"Heavy dependencies missing: {', '.join(missing)}. "
            "Install with: pip install lensing-ssc[heavy]"
        )


def get_version():
    """Get the version string."""
    return __version__


def get_info():
    """Get package information."""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "has_full_api": _HAS_FULL_API,
    }