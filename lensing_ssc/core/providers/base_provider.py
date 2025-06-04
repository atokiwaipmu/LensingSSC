"""
Base provider implementation with common functionality.

This module provides the foundational classes that all providers inherit from,
implementing common patterns like lazy loading, caching, and dependency checking.
"""

import gc
import sys
import time
import weakref
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, Callable
import logging

from ..interfaces.data_interface import DataProvider
from ..base.exceptions import ProviderError


class BaseProvider(DataProvider):
    """Base implementation for all providers.
    
    This class provides common functionality for all providers including:
    - Initialization tracking
    - Logging setup
    - Basic dependency checking
    - Error handling
    - Provider metadata management
    """
    
    def __init__(self):
        self._backend = None
        self._initialized = False
        self._logger = logging.getLogger(self.__class__.__name__)
        self._initialization_time = None
        self._usage_count = 0
        self._last_used = None
    
    @property
    def name(self) -> str:
        """Provider name (derived from class name)."""
        return self.__class__.__name__
    
    @property
    def version(self) -> str:
        """Provider version (override in subclasses)."""
        return "1.0.0"
    
    @property
    def initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized
    
    @property
    def usage_count(self) -> int:
        """Get usage count."""
        return self._usage_count
    
    @property
    def last_used(self) -> Optional[float]:
        """Get timestamp of last usage."""
        return self._last_used
    
    def is_available(self) -> bool:
        """Check if the provider is available.
        
        Returns
        -------
        bool
            True if all dependencies are satisfied
        """
        try:
            self._check_dependencies()
            return True
        except ImportError:
            return False
        except Exception as e:
            self._logger.debug(f"Provider {self.name} availability check failed: {e}")
            return False
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available.
        
        Override in subclasses to check specific dependencies.
        
        Raises
        ------
        ImportError
            If required dependencies are not available
        """
        # Base implementation does nothing
        pass
    
    def initialize(self, **kwargs) -> None:
        """Initialize the provider.
        
        Parameters
        ----------
        **kwargs
            Provider-specific initialization arguments
        """
        if self._initialized:
            self._logger.debug(f"Provider {self.name} already initialized")
            return
        
        start_time = time.perf_counter()
        
        try:
            self._check_dependencies()
            self._initialize_backend(**kwargs)
            self._initialized = True
            self._initialization_time = time.perf_counter() - start_time
            
            self._logger.info(
                f"Provider {self.name} initialized successfully "
                f"in {self._initialization_time:.3f}s"
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize provider {self.name}: {e}")
    
    def _initialize_backend(self, **kwargs) -> None:
        """Initialize the backend. Override in subclasses.
        
        Parameters
        ----------
        **kwargs
            Backend-specific initialization arguments
        """
        pass
    
    def ensure_initialized(self) -> None:
        """Ensure provider is initialized, initializing if necessary."""
        if not self._initialized:
            self.initialize()
    
    def _track_usage(self) -> None:
        """Track provider usage for monitoring."""
        self._usage_count += 1
        self._last_used = time.time()
    
    def shutdown(self) -> None:
        """Shutdown the provider and cleanup resources."""
        if hasattr(self, '_backend') and self._backend is not None:
            try:
                if hasattr(self._backend, 'close'):
                    self._backend.close()
                elif hasattr(self._backend, 'shutdown'):
                    self._backend.shutdown()
            except Exception as e:
                self._logger.warning(f"Error during provider shutdown: {e}")
        
        self._backend = None
        self._initialized = False
        self._logger.debug(f"Provider {self.name} shutdown complete")
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive provider information.
        
        Returns
        -------
        Dict[str, Any]
            Provider information dictionary
        """
        info = {
            'name': self.name,
            'version': self.version,
            'available': self.is_available(),
            'initialized': self._initialized,
            'usage_count': self._usage_count,
            'last_used': self._last_used,
        }
        
        if self._initialization_time is not None:
            info['initialization_time'] = self._initialization_time
        
        # Add backend-specific info if available
        if hasattr(self, '_get_backend_info'):
            try:
                backend_info = self._get_backend_info()
                info.update(backend_info)
            except Exception as e:
                self._logger.debug(f"Failed to get backend info: {e}")
        
        return info
    
    def __enter__(self):
        """Context manager entry."""
        self.ensure_initialized()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Don't shutdown automatically in context manager
        # Let the provider persist for potential reuse
        pass
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception:
            # Ignore errors during cleanup
            pass


class LazyProvider(BaseProvider):
    """Base class for providers with lazy loading capabilities.
    
    This class provides utilities for lazy importing of modules and
    accessing their attributes only when needed.
    """
    
    def __init__(self):
        super().__init__()
        self._modules: Dict[str, Any] = {}
        self._module_versions: Dict[str, str] = {}
    
    def _lazy_import(self, module_name: str, alias: Optional[str] = None) -> Any:
        """Lazy import a module.
        
        Parameters
        ----------
        module_name : str
            Name of module to import
        alias : str, optional
            Alias to store module under
            
        Returns
        -------
        Any
            Imported module
            
        Raises
        ------
        ProviderError
            If module cannot be imported
        """
        if alias is None:
            alias = module_name
        
        if alias not in self._modules:
            try:
                # Handle nested imports like 'numpy.fft'
                if '.' in module_name:
                    parts = module_name.split('.')
                    module = __import__(parts[0])
                    for part in parts[1:]:
                        module = getattr(module, part)
                else:
                    module = __import__(module_name)
                
                self._modules[alias] = module
                
                # Try to get version if available
                version = getattr(module, '__version__', 'unknown')
                self._module_versions[alias] = version
                
                self._logger.debug(f"Lazy imported {module_name} as {alias} (v{version})")
                
            except ImportError as e:
                raise ProviderError(f"Failed to import {module_name}: {e}")
            except Exception as e:
                raise ProviderError(f"Error importing {module_name}: {e}")
        
        return self._modules[alias]
    
    def _get_module_attribute(self, module_name: str, attribute: str, 
                            alias: Optional[str] = None) -> Any:
        """Get an attribute from a lazily imported module.
        
        Parameters
        ----------
        module_name : str
            Name of module
        attribute : str
            Attribute name
        alias : str, optional
            Module alias
            
        Returns
        -------
        Any
            Module attribute
            
        Raises
        ------
        ProviderError
            If module or attribute cannot be accessed
        """
        module = self._lazy_import(module_name, alias)
        
        try:
            return getattr(module, attribute)
        except AttributeError as e:
            raise ProviderError(f"Module {module_name} has no attribute {attribute}: {e}")
    
    def _get_backend_info(self) -> Dict[str, Any]:
        """Get backend-specific information."""
        info = super().get_info() if hasattr(super(), 'get_info') else {}
        
        if self._modules:
            info['modules'] = {}
            for alias, module in self._modules.items():
                module_info = {
                    'loaded': True,
                    'version': self._module_versions.get(alias, 'unknown')
                }
                
                # Add module-specific info if available
                if hasattr(module, '__file__'):
                    module_info['file'] = module.__file__
                
                info['modules'][alias] = module_info
        
        return info


class CachedProvider(BaseProvider):
    """Base class for providers with caching capabilities.
    
    This class provides LRU-style caching for expensive operations
    with automatic cleanup and memory management.
    """
    
    def __init__(self, cache_size: int = 100, cache_ttl: Optional[float] = None):
        """Initialize cached provider.
        
        Parameters
        ----------
        cache_size : int
            Maximum number of items in cache
        cache_ttl : float, optional
            Time-to-live for cache entries in seconds
        """
        super().__init__()
        self._cache: Dict[str, Any] = {}
        self._cache_times: Dict[str, float] = {}
        self._cache_order: List[str] = []
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        Any or None
            Cached item or None if not found/expired
        """
        if key not in self._cache:
            self._cache_misses += 1
            return None
        
        # Check TTL if specified
        if self._cache_ttl is not None:
            age = time.time() - self._cache_times[key]
            if age > self._cache_ttl:
                self._cache_delete(key)
                self._cache_misses += 1
                return None
        
        # Move to end (most recently used)
        self._cache_order.remove(key)
        self._cache_order.append(key)
        
        self._cache_hits += 1
        return self._cache[key]
    
    def _cache_set(self, key: str, value: Any) -> None:
        """Set item in cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        current_time = time.time()
        
        if key in self._cache:
            # Update existing item
            self._cache[key] = value
            self._cache_times[key] = current_time
            self._cache_order.remove(key)
            self._cache_order.append(key)
        else:
            # Add new item
            if len(self._cache) >= self._cache_size:
                # Remove least recently used item
                oldest_key = self._cache_order.pop(0)
                self._cache_delete(oldest_key)
            
            self._cache[key] = value
            self._cache_times[key] = current_time
            self._cache_order.append(key)
    
    def _cache_delete(self, key: str) -> None:
        """Delete item from cache.
        
        Parameters
        ----------
        key : str
            Cache key to delete
        """
        if key in self._cache:
            del self._cache[key]
            del self._cache_times[key]
            if key in self._cache_order:
                self._cache_order.remove(key)
    
    def _cache_clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._cache_times.clear()
        self._cache_order.clear()
        gc.collect()  # Force garbage collection
    
    def _cache_cleanup(self) -> None:
        """Clean up expired cache entries."""
        if self._cache_ttl is None:
            return
        
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_times.items()
            if current_time - timestamp > self._cache_ttl
        ]
        
        for key in expired_keys:
            self._cache_delete(key)
        
        if expired_keys:
            self._logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information and statistics.
        
        Returns
        -------
        Dict[str, Any]
            Cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self._cache),
            'max_size': self._cache_size,
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'ttl': self._cache_ttl,
            'keys': list(self._cache_order),
        }
    
    def _get_backend_info(self) -> Dict[str, Any]:
        """Get backend-specific information including cache stats."""
        info = super()._get_backend_info() if hasattr(super(), '_get_backend_info') else {}
        info['cache'] = self.get_cache_info()
        return info
    
    def shutdown(self) -> None:
        """Shutdown provider and clear cache."""
        self._cache_clear()
        super().shutdown()


class ProviderRegistry:
    """Registry for managing provider instances with automatic cleanup.
    
    This class manages provider instances, handles initialization,
    and provides automatic cleanup of unused providers.
    """
    
    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}
        self._weak_refs: Dict[str, weakref.ref] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def register(self, name: str, provider: BaseProvider) -> None:
        """Register a provider instance.
        
        Parameters
        ----------
        name : str
            Provider name
        provider : BaseProvider
            Provider instance
        """
        self._providers[name] = provider
        
        # Create weak reference for automatic cleanup
        def cleanup_callback(ref):
            if name in self._weak_refs and self._weak_refs[name] is ref:
                del self._weak_refs[name]
                self._logger.debug(f"Provider {name} garbage collected")
        
        self._weak_refs[name] = weakref.ref(provider, cleanup_callback)
        self._logger.debug(f"Registered provider: {name}")
    
    def get(self, name: str) -> Optional[BaseProvider]:
        """Get a provider instance.
        
        Parameters
        ----------
        name : str
            Provider name
            
        Returns
        -------
        BaseProvider or None
            Provider instance or None if not found
        """
        return self._providers.get(name)
    
    def remove(self, name: str) -> None:
        """Remove a provider from registry.
        
        Parameters
        ----------
        name : str
            Provider name
        """
        if name in self._providers:
            provider = self._providers[name]
            try:
                provider.shutdown()
            except Exception as e:
                self._logger.warning(f"Error shutting down provider {name}: {e}")
            
            del self._providers[name]
            if name in self._weak_refs:
                del self._weak_refs[name]
            
            self._logger.debug(f"Removed provider: {name}")
    
    def cleanup_unused(self) -> int:
        """Clean up unused providers.
        
        Returns
        -------
        int
            Number of providers cleaned up
        """
        unused = []
        for name, ref in list(self._weak_refs.items()):
            if ref() is None:  # Object has been garbage collected
                unused.append(name)
        
        for name in unused:
            if name in self._providers:
                self.remove(name)
        
        if unused:
            self._logger.info(f"Cleaned up {len(unused)} unused providers")
        
        return len(unused)
    
    def shutdown_all(self) -> None:
        """Shutdown all registered providers."""
        for name in list(self._providers.keys()):
            self.remove(name)
        
        self._logger.info("All providers shutdown")
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered providers.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Status information for each provider
        """
        status = {}
        for name, provider in self._providers.items():
            try:
                status[name] = provider.get_info()
            except Exception as e:
                status[name] = {'error': str(e), 'available': False}
        
        return status


# Global provider registry instance
_global_registry = ProviderRegistry()

def get_global_registry() -> ProviderRegistry:
    """Get the global provider registry.
    
    Returns
    -------
    ProviderRegistry
        Global provider registry instance
    """
    return _global_registry