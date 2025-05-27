"""
Factory for creating and managing providers.
"""

from typing import Any, Dict, Type, Optional, List
import logging

from ..core.base.exceptions import ProviderError
from ..core.interfaces.data_interface import DataProvider
from .base_provider import BaseProvider


class ProviderFactory:
    """Factory for creating and managing providers."""
    
    def __init__(self):
        self._providers: Dict[str, Type[DataProvider]] = {}
        self._instances: Dict[str, DataProvider] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def register_provider(self, name: str, provider_class: Type[DataProvider]) -> None:
        """Register a provider class.
        
        Parameters
        ----------
        name : str
            Provider name
        provider_class : Type[DataProvider]
            Provider class
        """
        if not issubclass(provider_class, DataProvider):
            raise ProviderError(f"Provider class must inherit from DataProvider")
        
        self._providers[name] = provider_class
        self._logger.debug(f"Registered provider: {name}")
    
    def unregister_provider(self, name: str) -> None:
        """Unregister a provider.
        
        Parameters
        ----------
        name : str
            Provider name
        """
        if name in self._providers:
            del self._providers[name]
            
        if name in self._instances:
            del self._instances[name]
            
        self._logger.debug(f"Unregistered provider: {name}")
    
    def create_provider(self, name: str, **kwargs) -> DataProvider:
        """Create a provider instance.
        
        Parameters
        ----------
        name : str
            Provider name
        **kwargs
            Provider initialization arguments
            
        Returns
        -------
        DataProvider
            Provider instance
        """
        if name not in self._providers:
            raise ProviderError(f"Unknown provider: {name}")
        
        provider_class = self._providers[name]
        
        try:
            provider = provider_class()
            if hasattr(provider, 'initialize'):
                provider.initialize(**kwargs)
            return provider
        except Exception as e:
            raise ProviderError(f"Failed to create provider {name}: {e}")
    
    def get_provider(self, name: str, **kwargs) -> DataProvider:
        """Get a provider instance (singleton pattern).
        
        Parameters
        ----------
        name : str
            Provider name
        **kwargs
            Provider initialization arguments (only used on first creation)
            
        Returns
        -------
        DataProvider
            Provider instance
        """
        if name not in self._instances:
            self._instances[name] = self.create_provider(name, **kwargs)
        
        return self._instances[name]
    
    def is_provider_available(self, name: str) -> bool:
        """Check if a provider is available.
        
        Parameters
        ----------
        name : str
            Provider name
            
        Returns
        -------
        bool
            True if provider is available
        """
        if name not in self._providers:
            return False
        
        try:
            provider = self.create_provider(name)
            return provider.is_available()
        except Exception:
            return False
    
    def list_providers(self, available_only: bool = False) -> List[str]:
        """List registered providers.
        
        Parameters
        ----------
        available_only : bool
            If True, only list available providers
            
        Returns
        -------
        list
            List of provider names
        """
        if not available_only:
            return list(self._providers.keys())
        
        available = []
        for name in self._providers.keys():
            if self.is_provider_available(name):
                available.append(name)
        
        return available
    
    def get_provider_info(self, name: str) -> Dict[str, Any]:
        """Get information about a provider.
        
        Parameters
        ----------
        name : str
            Provider name
            
        Returns
        -------
        dict
            Provider information
        """
        if name not in self._providers:
            raise ProviderError(f"Unknown provider: {name}")
        
        provider_class = self._providers[name]
        info = {
            'name': name,
            'class': provider_class.__name__,
            'module': provider_class.__module__,
            'registered': True,
        }
        
        try:
            provider = self.create_provider(name)
            info.update({
                'available': provider.is_available(),
                'version': provider.version,
            })
            
            if hasattr(provider, 'get_info'):
                info.update(provider.get_info())
        except Exception as e:
            info.update({
                'available': False,
                'error': str(e),
            })
        
        return info
    
    def clear_instances(self) -> None:
        """Clear all provider instances."""
        self._instances.clear()
        self._logger.debug("Cleared all provider instances")
    
    def configure_provider(self, name: str, **config) -> None:
        """Configure a provider.
        
        Parameters
        ----------
        name : str
            Provider name
        **config
            Configuration parameters
        """
        if name in self._instances:
            provider = self._instances[name]
            if hasattr(provider, 'configure'):
                provider.configure(**config)
            else:
                self._logger.warning(f"Provider {name} does not support configuration")
        else:
            self._logger.warning(f"Provider {name} not instantiated yet")


# Global factory instance
_global_factory = ProviderFactory()


def get_provider(name: str, **kwargs) -> DataProvider:
    """Get a provider from the global factory.
    
    Parameters
    ----------
    name : str
        Provider name
    **kwargs
        Provider initialization arguments
        
    Returns
    -------
    DataProvider
        Provider instance
    """
    return _global_factory.get_provider(name, **kwargs)


def register_provider(name: str, provider_class: Type[DataProvider]) -> None:
    """Register a provider with the global factory.
    
    Parameters
    ----------
    name : str
        Provider name
    provider_class : Type[DataProvider]
        Provider class
    """
    _global_factory.register_provider(name, provider_class)


def list_available_providers() -> List[str]:
    """List available providers.
    
    Returns
    -------
    list
        List of available provider names
    """
    return _global_factory.list_providers(available_only=True)


def get_provider_info(name: str) -> Dict[str, Any]:
    """Get provider information.
    
    Parameters
    ----------
    name : str
        Provider name
        
    Returns
    -------
    dict
        Provider information
    """
    return _global_factory.get_provider_info(name)


def auto_detect_providers() -> Dict[str, bool]:
    """Auto-detect available providers.
    
    Returns
    -------
    dict
        Dictionary mapping provider names to availability
    """
    # List of providers to check
    providers_to_check = [
        ('healpix', 'lensing_ssc.providers.healpix_provider.HealpixProvider'),
        ('lenstools', 'lensing_ssc.providers.lenstools_provider.LenstoolsProvider'),
        ('nbodykit', 'lensing_ssc.providers.nbodykit_provider.NbodykitProvider'),
        ('matplotlib', 'lensing_ssc.providers.matplotlib_provider.MatplotlibProvider'),
    ]
    
    availability = {}
    
    for name, class_path in providers_to_check:
        try:
            # Try to import and instantiate the provider
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            provider_class = getattr(module, class_name)
            
            # Register and check availability
            _global_factory.register_provider(name, provider_class)
            availability[name] = _global_factory.is_provider_available(name)
        except Exception:
            availability[name] = False
    
    return availability