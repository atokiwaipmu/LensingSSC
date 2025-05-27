"""
Provider implementations for dependency injection.

This module contains concrete implementations of the abstract interfaces
defined in lensing_ssc.core.interfaces.
"""

from .factory import ProviderFactory, get_provider
from .base_provider import BaseProvider

# Import available providers
_available_providers = {}

try:
    from .healpix_provider import HealpixProvider
    _available_providers['healpix'] = HealpixProvider
except ImportError:
    pass

try:
    from .lenstools_provider import LenstoolsProvider
    _available_providers['lenstools'] = LenstoolsProvider
except ImportError:
    pass

try:
    from .nbodykit_provider import NbodykitProvider
    _available_providers['nbodykit'] = NbodykitProvider
except ImportError:
    pass

try:
    from .matplotlib_provider import MatplotlibProvider
    _available_providers['matplotlib'] = MatplotlibProvider
except ImportError:
    pass

# Register available providers
factory = ProviderFactory()
for name, provider_class in _available_providers.items():
    factory.register_provider(name, provider_class)

__all__ = [
    'ProviderFactory',
    'get_provider',
    'BaseProvider',
] + list(_available_providers.keys())