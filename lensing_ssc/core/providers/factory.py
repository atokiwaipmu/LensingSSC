"""
Factory for creating and managing providers.

This module implements the factory pattern for provider creation and management,
providing a centralized way to instantiate providers, manage their lifecycle,
and handle dependency resolution.
"""

import importlib
import threading
from typing import Any, Dict, Type, Optional, List, Union, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..interfaces.data_interface import DataProvider
from ..base.exceptions import ProviderError
from .base_provider import BaseProvider, ProviderRegistry, get_global_registry


class ProviderFactory:
    """Factory for creating and managing providers.
    
    This factory provides:
    - Provider class registration and discovery
    - Lazy instantiation with singleton pattern
    - Dependency resolution and validation
    - Thread-safe provider creation
    - Automatic fallback to alternative providers
    """
    
    def __init__(self, use_global_registry: bool = True):
        """Initialize the provider factory.
        
        Parameters
        ----------
        use_global_registry : bool
            Whether to use the global provider registry
        """
        self._provider_classes: Dict[str, Type[DataProvider]] = {}
        self._provider_configs: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, DataProvider] = {}
        self._creation_lock = threading.RLock()
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Use global registry or create local one
        self._registry = get_global_registry() if use_global_registry else ProviderRegistry()
        
        # Provider aliases for backward compatibility
        self._aliases: Dict[str, str] = {}
        
        # Fallback chains for provider types
        self._fallback_chains: Dict[str, List[str]] = {}
        
        # Provider validation functions
        self._validators: Dict[str, Callable[[DataProvider], bool]] = {}
    
    def register_provider(self, name: str, provider_class: Type[DataProvider],
                         config: Optional[Dict[str, Any]] = None,
                         aliases: Optional[List[str]] = None) -> None:
        """Register a provider class with the factory.
        
        Parameters
        ----------
        name : str
            Provider name
        provider_class : Type[DataProvider]
            Provider class
        config : Dict[str, Any], optional
            Default configuration for the provider
        aliases : List[str], optional
            Alternative names for the provider
        """
        if not issubclass(provider_class, DataProvider):
            raise ProviderError(f"Provider class must inherit from DataProvider")
        
        with self._creation_lock:
            self._provider_classes[name] = provider_class
            self._provider_configs[name] = config or {}
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = name
            
            self._logger.debug(f"Registered provider: {name} ({provider_class.__name__})")
    
    def unregister_provider(self, name: str) -> None:
        """Unregister a provider and clean up its instances.
        
        Parameters
        ----------
        name : str
            Provider name
        """
        with self._creation_lock:
            # Remove from registry
            self._registry.remove(name)
            
            # Clean up factory state
            if name in self._provider_classes:
                del self._provider_classes[name]
            if name in self._provider_configs:
                del self._provider_configs[name]
            if name in self._instances:
                del self._instances[name]
            
            # Remove aliases
            aliases_to_remove = [alias for alias, target in self._aliases.items() if target == name]
            for alias in aliases_to_remove:
                del self._aliases[alias]
            
            self._logger.debug(f"Unregistered provider: {name}")
    
    def register_from_string(self, name: str, class_path: str,
                           config: Optional[Dict[str, Any]] = None) -> None:
        """Register a provider from a string class path.
        
        Parameters
        ----------
        name : str
            Provider name
        class_path : str
            Full path to provider class (e.g., 'package.module.ClassName')
        config : Dict[str, Any], optional
            Default configuration
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)
            
            self.register_provider(name, provider_class, config)
            
        except Exception as e:
            raise ProviderError(f"Failed to register provider from {class_path}: {e}")
    
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
        # Resolve aliases
        resolved_name = self._aliases.get(name, name)
        
        if resolved_name not in self._provider_classes:
            raise ProviderError(f"Unknown provider: {name}")
        
        provider_class = self._provider_classes[resolved_name]
        
        try:
            # Merge default config with provided kwargs
            init_kwargs = self._provider_configs[resolved_name].copy()
            init_kwargs.update(kwargs)
            
            # Create and initialize provider
            provider = provider_class()
            if hasattr(provider, 'initialize'):
                provider.initialize(**init_kwargs)
            
            # Validate provider if validator is registered
            if resolved_name in self._validators:
                if not self._validators[resolved_name](provider):
                    raise ProviderError(f"Provider {resolved_name} failed validation")
            
            self._logger.debug(f"Created provider: {resolved_name}")
            return provider
            
        except Exception as e:
            raise ProviderError(f"Failed to create provider {resolved_name}: {e}")
    
    def get_provider(self, name: str, singleton: bool = True, **kwargs) -> DataProvider:
        """Get a provider instance using singleton pattern.
        
        Parameters
        ----------
        name : str
            Provider name
        singleton : bool
            Whether to use singleton pattern (default: True)
        **kwargs
            Provider initialization arguments (only used for first creation)
            
        Returns
        -------
        DataProvider
            Provider instance
        """
        # Resolve aliases
        resolved_name = self._aliases.get(name, resolved_name)
        
        if not singleton:
            return self.create_provider(resolved_name, **kwargs)
        
        with self._creation_lock:
            if resolved_name not in self._instances:
                self._instances[resolved_name] = self.create_provider(resolved_name, **kwargs)
                
                # Register with registry for lifecycle management
                self._registry.register(resolved_name, self._instances[resolved_name])
            
            return self._instances[resolved_name]
    
    def get_or_fallback(self, name: str, fallback_chain: Optional[List[str]] = None,
                       **kwargs) -> Optional[DataProvider]:
        """Get a provider with automatic fallback to alternatives.
        
        Parameters
        ----------
        name : str
            Primary provider name
        fallback_chain : List[str], optional
            List of fallback provider names
        **kwargs
            Provider initialization arguments
            
        Returns
        -------
        DataProvider or None
            Provider instance or None if all providers fail
        """
        # Try primary provider first
        if self.is_provider_available(name):
            try:
                return self.get_provider(name, **kwargs)
            except Exception as e:
                self._logger.warning(f"Primary provider {name} failed: {e}")
        
        # Try fallback chain
        fallbacks = fallback_chain or self._fallback_chains.get(name, [])
        for fallback_name in fallbacks:
            if self.is_provider_available(fallback_name):
                try:
                    provider = self.get_provider(fallback_name, **kwargs)
                    self._logger.info(f"Using fallback provider {fallback_name} for {name}")
                    return provider
                except Exception as e:
                    self._logger.warning(f"Fallback provider {fallback_name} failed: {e}")
        
        self._logger.error(f"No available providers for {name}")
        return None
    
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
        # Resolve aliases
        resolved_name = self._aliases.get(name, name)
        
        if resolved_name not in self._provider_classes:
            return False
        
        try:
            provider = self.create_provider(resolved_name)
            available = provider.is_available()
            
            # Clean up test instance
            if hasattr(provider, 'shutdown'):
                provider.shutdown()
            
            return available
            
        except Exception:
            return False
    
    def list_providers(self, available_only: bool = False,
                      include_aliases: bool = False) -> List[str]:
        """List registered providers.
        
        Parameters
        ----------
        available_only : bool
            If True, only list available providers
        include_aliases : bool
            If True, include provider aliases
            
        Returns
        -------
        List[str]
            List of provider names
        """
        providers = list(self._provider_classes.keys())
        
        if available_only:
            providers = [name for name in providers if self.is_provider_available(name)]
        
        if include_aliases:
            providers.extend(self._aliases.keys())
        
        return sorted(providers)
    
    def get_provider_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive information about a provider.
        
        Parameters
        ----------
        name : str
            Provider name
            
        Returns
        -------
        Dict[str, Any]
            Provider information
        """
        # Resolve aliases
        resolved_name = self._aliases.get(name, name)
        
        if resolved_name not in self._provider_classes:
            raise ProviderError(f"Unknown provider: {name}")
        
        provider_class = self._provider_classes[resolved_name]
        config = self._provider_configs[resolved_name]
        
        info = {
            'name': resolved_name,
            'original_name': name,
            'class': provider_class.__name__,
            'module': provider_class.__module__,
            'registered': True,
            'config': config,
            'aliases': [alias for alias, target in self._aliases.items() if target == resolved_name],
        }
        
        # Add runtime information if possible
        try:
            provider = self.create_provider(resolved_name)
            info.update({
                'available': provider.is_available(),
                'version': provider.version if hasattr(provider, 'version') else 'unknown',
            })
            
            if hasattr(provider, 'get_info'):
                runtime_info = provider.get_info()
                info.update(runtime_info)
            
            # Clean up test instance
            if hasattr(provider, 'shutdown'):
                provider.shutdown()
                
        except Exception as e:
            info.update({
                'available': False,
                'error': str(e),
            })
        
        return info
    
    def set_fallback_chain(self, primary: str, fallbacks: List[str]) -> None:
        """Set fallback chain for a provider type.
        
        Parameters
        ----------
        primary : str
            Primary provider name
        fallbacks : List[str]
            List of fallback provider names in order of preference
        """
        self._fallback_chains[primary] = fallbacks
        self._logger.debug(f"Set fallback chain for {primary}: {fallbacks}")
    
    def register_validator(self, name: str, validator: Callable[[DataProvider], bool]) -> None:
        """Register a validation function for a provider.
        
        Parameters
        ----------
        name : str
            Provider name
        validator : Callable
            Validation function that takes a provider and returns bool
        """
        self._validators[name] = validator
        self._logger.debug(f"Registered validator for {name}")
    
    def bulk_availability_check(self, provider_names: Optional[List[str]] = None,
                              max_workers: int = 4) -> Dict[str, bool]:
        """Check availability of multiple providers in parallel.
        
        Parameters
        ----------
        provider_names : List[str], optional
            List of provider names to check (default: all registered)
        max_workers : int
            Maximum number of threads for parallel checking
            
        Returns
        -------
        Dict[str, bool]
            Availability status for each provider
        """
        if provider_names is None:
            provider_names = list(self._provider_classes.keys())
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit availability checks
            futures = {
                executor.submit(self.is_provider_available, name): name
                for name in provider_names
            }
            
            # Collect results
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    self._logger.warning(f"Availability check failed for {name}: {e}")
                    results[name] = False
        
        return results
    
    def clear_instances(self) -> None:
        """Clear all provider instances (for testing/cleanup)."""
        with self._creation_lock:
            # Shutdown all instances
            for name, instance in self._instances.items():
                try:
                    if hasattr(instance, 'shutdown'):
                        instance.shutdown()
                except Exception as e:
                    self._logger.warning(f"Error shutting down {name}: {e}")
            
            self._instances.clear()
            self._registry.shutdown_all()
            
            self._logger.debug("Cleared all provider instances")
    
    def configure_provider(self, name: str, **config) -> None:
        """Update configuration for a provider.
        
        Parameters
        ----------
        name : str
            Provider name
        **config
            Configuration parameters
        """
        resolved_name = self._aliases.get(name, name)
        
        if resolved_name not in self._provider_configs:
            self._provider_configs[resolved_name] = {}
        
        self._provider_configs[resolved_name].update(config)
        
        # If instance exists, try to reconfigure it
        if resolved_name in self._instances:
            instance = self._instances[resolved_name]
            if hasattr(instance, 'configure'):
                try:
                    instance.configure(**config)
                except Exception as e:
                    self._logger.warning(f"Failed to reconfigure {resolved_name}: {e}")
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics and status.
        
        Returns
        -------
        Dict[str, Any]
            Factory statistics
        """
        return {
            'registered_providers': len(self._provider_classes),
            'active_instances': len(self._instances),
            'aliases': len(self._aliases),
            'fallback_chains': len(self._fallback_chains),
            'validators': len(self._validators),
            'providers': list(self._provider_classes.keys()),
            'instances': list(self._instances.keys()),
            'registry_status': self._registry.get_status(),
        }


# Global factory instance
_global_factory = ProviderFactory()


def get_provider(name: str, singleton: bool = True, **kwargs) -> DataProvider:
    """Get a provider from the global factory.
    
    Parameters
    ----------
    name : str
        Provider name
    singleton : bool
        Whether to use singleton pattern
    **kwargs
        Provider initialization arguments
        
    Returns
    -------
    DataProvider
        Provider instance
    """
    return _global_factory.get_provider(name, singleton=singleton, **kwargs)


def get_or_fallback(name: str, fallback_chain: Optional[List[str]] = None,
                   **kwargs) -> Optional[DataProvider]:
    """Get a provider with fallback from the global factory.
    
    Parameters
    ----------
    name : str
        Primary provider name
    fallback_chain : List[str], optional
        Fallback provider names
    **kwargs
        Provider initialization arguments
        
    Returns
    -------
    DataProvider or None
        Provider instance or None if all fail
    """
    return _global_factory.get_or_fallback(name, fallback_chain, **kwargs)


def register_provider(name: str, provider_class: Type[DataProvider],
                     config: Optional[Dict[str, Any]] = None) -> None:
    """Register a provider with the global factory.
    
    Parameters
    ----------
    name : str
        Provider name
    provider_class : Type[DataProvider]
        Provider class
    config : Dict[str, Any], optional
        Default configuration
    """
    _global_factory.register_provider(name, provider_class, config)


def list_available_providers() -> List[str]:
    """List available providers from the global factory.
    
    Returns
    -------
    List[str]
        List of available provider names
    """
    return _global_factory.list_providers(available_only=True)


def get_provider_info(name: str) -> Dict[str, Any]:
    """Get provider information from the global factory.
    
    Parameters
    ----------
    name : str
        Provider name
        
    Returns
    -------
    Dict[str, Any]
        Provider information
    """
    return _global_factory.get_provider_info(name)


def auto_configure_fallbacks() -> None:
    """Automatically configure fallback chains for common provider types."""
    # Map provider fallback chains
    fallback_configs = {
        'healpix': ['healpix'],  # Only one HEALPix provider typically
        'lensing': ['lenstools'],  # Primary lensing analysis
        'catalog': ['nbodykit'],  # Primary catalog analysis
        'plotting': ['matplotlib'],  # Primary plotting
    }
    
    for primary, fallbacks in fallback_configs.items():
        _global_factory.set_fallback_chain(primary, fallbacks)


def validate_all_providers() -> Dict[str, Dict[str, Any]]:
    """Validate all registered providers.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Validation results for all providers
    """
    results = {}
    
    for provider_name in _global_factory.list_providers():
        try:
            info = _global_factory.get_provider_info(provider_name)
            results[provider_name] = {
                'available': info.get('available', False),
                'version': info.get('version', 'unknown'),
                'error': info.get('error'),
                'validation_passed': True
            }
        except Exception as e:
            results[provider_name] = {
                'available': False,
                'version': None,
                'error': str(e),
                'validation_passed': False
            }
    
    return results


# Initialize fallback configurations
auto_configure_fallbacks()