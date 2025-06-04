"""
Provider implementations for dependency injection.

This module contains concrete implementations of the abstract interfaces
defined in lensing_ssc.core.interfaces. The provider pattern allows for
heavy dependencies (healpy, lenstools, nbodykit, matplotlib) to be
dynamically loaded only when needed, reducing startup time and memory usage.

The provider system supports:
- Lazy loading of heavy dependencies
- Automatic provider discovery and registration
- Fallback providers when dependencies are missing
- Plugin-style architecture for extending functionality

Usage:
    from lensing_ssc.providers import get_provider
    
    # Get a provider (auto-detection of available implementation)
    healpix = get_provider('healpix')
    
    # Use the provider
    map_data = healpix.read_map('kappa_map.fits')

Available Providers:
    - healpix: HEALPix operations using healpy
    - lenstools: Lensing analysis using lenstools
    - nbodykit: N-body simulation data using nbodykit
    - matplotlib: Plotting and visualization using matplotlib
"""

import logging
from typing import Dict, List, Optional, Type, Any

from .factory import ProviderFactory, get_provider as _get_provider_from_factory, _global_factory as global_provider_factory
from .base_provider import BaseProvider, LazyProvider, CachedProvider

# _factory = ProviderFactory() # Removed: Using global_provider_factory from factory.py

# Registry of available providers
_available_providers: Dict[str, Type[BaseProvider]] = {}

# Provider discovery and registration
def _discover_providers():
    """Discover and register available providers with the global factory."""
    global _available_providers # Removed _factory from global
    
    logger = logging.getLogger(__name__)
    
    # HEALPix provider
    try:
        from .healpix_provider import HealpixProvider
        _available_providers['healpix'] = HealpixProvider
        global_provider_factory.register_provider('healpix', HealpixProvider)
        logger.debug("Registered HealpixProvider with global factory")
    except ImportError as e:
        logger.debug(f"HealpixProvider not available: {e}")

    # LensTools provider
    try:
        from .lenstools_provider import LenstoolsProvider
        _available_providers['lenstools'] = LenstoolsProvider
        global_provider_factory.register_provider('lenstools', LenstoolsProvider)
        logger.debug("Registered LenstoolsProvider with global factory")
    except ImportError as e:
        logger.debug(f"LenstoolsProvider not available: {e}")

    # NBBodyKit provider
    try:
        from .nbodykit_provider import NbodykitProvider
        _available_providers['nbodykit'] = NbodykitProvider
        global_provider_factory.register_provider('nbodykit', NbodykitProvider)
        logger.debug("Registered NbodykitProvider with global factory")
    except ImportError as e:
        logger.debug(f"NbodykitProvider not available: {e}")

    # Matplotlib provider
    try:
        from .matplotlib_provider import MatplotlibProvider
        _available_providers['matplotlib'] = MatplotlibProvider
        global_provider_factory.register_provider('matplotlib', MatplotlibProvider)
        logger.debug("Registered MatplotlibProvider with global factory")
    except ImportError as e:
        logger.debug(f"MatplotlibProvider not available: {e}")

    # MySQL provider
    try:
        from .mysql_provider import MySQLProvider
        _available_providers['mysql'] = MySQLProvider
        global_provider_factory.register_provider('mysql', MySQLProvider)
        logger.debug("Registered MySQLProvider with global factory")
    except ImportError as e:
        logger.debug(f"MySQLProvider not available: {e}")

# Perform provider discovery on import
_discover_providers()

# Public API functions
def list_available_providers() -> List[str]:
    """List names of available providers.
    
    Returns
    -------
    List[str]
        List of provider names that can be instantiated
    """
    return global_provider_factory.list_providers(available_only=True)

def list_all_providers() -> List[str]:
    """List names of all registered providers (available or not).
    
    Returns
    -------
    List[str]
        List of all registered provider names
    """
    return global_provider_factory.list_providers(available_only=False)

# The get_provider function to be exported is taken directly from the factory
# (which uses the global factory)
get_provider = _get_provider_from_factory


def get_provider_info(name: str) -> Dict[str, Any]:
    """Get information about a provider.
    
    Parameters
    ----------
    name : str
        Provider name
        
    Returns
    -------
    Dict[str, Any]
        Provider information including availability, version, etc.
    """
    return global_provider_factory.get_provider_info(name)

def is_provider_available(name: str) -> bool:
    """Check if a provider is available.
    
    Parameters
    ----------
    name : str
        Provider name
        
    Returns
    -------
    bool
        True if provider can be instantiated
    """
    return global_provider_factory.is_provider_available(name)

def register_provider(name: str, provider_class: Type[BaseProvider]) -> None:
    """Register a custom provider.
    
    Parameters
    ----------
    name : str
        Provider name
    provider_class : Type[BaseProvider]
        Provider class
    """
    _available_providers[name] = provider_class
    global_provider_factory.register_provider(name, provider_class)

def auto_detect_providers() -> Dict[str, bool]:
    """Auto-detect available providers and return status.
    
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping provider names to availability status
    """
    return {name: is_provider_available(name) for name in _available_providers.keys()}

def get_default_provider(provider_type: str) -> Optional[str]:
    """Get the default provider name for a given type.
    
    Parameters
    ----------
    provider_type : str
        Type of provider ('map', 'catalog', 'plotting', etc.)
        
    Returns
    -------
    Optional[str]
        Default provider name or None if no suitable provider available
    """
    type_mapping = {
        'map': ['healpix'],
        'catalog': ['nbodykit'],
        'plotting': ['matplotlib'],
        'lensing': ['lenstools'],
        'convergence': ['lenstools'],
        'statistics': ['lenstools'],
        'database': ['mysql'],
    }
    
    candidates = type_mapping.get(provider_type, [])
    for candidate in candidates:
        if is_provider_available(candidate):
            return candidate
    
    return None

def create_provider_config() -> Dict[str, Any]:
    """Create a configuration dictionary with available providers.
    
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary suitable for use with ProcessingConfig
    """
    config = {}
    
    # Map provider types to their default implementations
    if is_provider_available('healpix'):
        config['healpix'] = 'lensing_ssc.providers.healpix_provider.HealpixProvider'
    
    if is_provider_available('lenstools'):
        config['lenstools'] = 'lensing_ssc.providers.lenstools_provider.LenstoolsProvider'
    
    if is_provider_available('nbodykit'):
        config['nbodykit'] = 'lensing_ssc.providers.nbodykit_provider.NbodykitProvider'
    
    if is_provider_available('matplotlib'):
        config['matplotlib'] = 'lensing_ssc.providers.matplotlib_provider.MatplotlibProvider'

    if is_provider_available('mysql'):
        config['mysql'] = 'lensing_ssc.providers.mysql_provider.MySQLProvider'
    
    return config

def validate_dependencies() -> Dict[str, Dict[str, Any]]:
    """Validate all provider dependencies and return detailed status.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Detailed validation results for each provider
    """
    results = {}
    
    for name in _available_providers.keys():
        try:
            # create_provider is suitable here for a temporary instance if needed for full info
            # However, get_provider_info from the factory often does this, or can be enhanced to.
            # For just availability and version, get_provider_info might be enough and safer.
            # Let's use create_provider for now as per original structure for validate_dependencies
            provider = global_provider_factory.create_provider(name)
            results[name] = {
                'available': provider.is_available(),
                'version': provider.version,
                'info': provider.get_info() if hasattr(provider, 'get_info') else {},
                'error': None
            }
        except Exception as e:
            results[name] = {
                'available': False,
                'version': None,
                'info': {},
                'error': str(e)
            }
    
    return results

# Export main factory function and classes
__all__ = [
    # Main factory function
    'get_provider',
    
    # Base provider classes
    'BaseProvider',
    'LazyProvider', 
    'CachedProvider',
    
    # Factory class
    'ProviderFactory',
    
    # Provider discovery functions
    'list_available_providers',
    'list_all_providers',
    'get_provider_info',
    'is_provider_available',
    'register_provider',
    'auto_detect_providers',
    'get_default_provider',
    
    # Configuration helpers
    'create_provider_config',
    'validate_dependencies',
]

# Import specific providers if available (for backward compatibility)
# These will only be available if the dependencies are installed
try:
    from .healpix_provider import HealpixProvider
    __all__.append('HealpixProvider')
except ImportError:
    pass

try:
    from .lenstools_provider import LenstoolsProvider
    __all__.append('LenstoolsProvider')
except ImportError:
    pass

try:
    from .nbodykit_provider import NbodykitProvider
    __all__.append('NbodykitProvider')
except ImportError:
    pass

try:
    from .matplotlib_provider import MatplotlibProvider
    __all__.append('MatplotlibProvider')
except ImportError:
    pass

try:
    from .mysql_provider import MySQLProvider
    __all__.append('MySQLProvider')
except ImportError:
    pass