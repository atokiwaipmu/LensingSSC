"""
Base provider implementation with common functionality.
"""

from abc import ABC
from typing import Any, Dict, Optional
import logging

from ..core.interfaces.data_interface import DataProvider
from ..core.base.exceptions import ProviderError


class BaseProvider(DataProvider):
    """Base implementation for all providers."""
    
    def __init__(self):
        self._backend = None
        self._initialized = False
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def name(self) -> str:
        """Provider name."""
        return self.__class__.__name__
    
    @property
    def version(self) -> str:
        """Provider version."""
        return "1.0.0"
    
    def is_available(self) -> bool:
        """Check if the provider is available."""
        try:
            self._check_dependencies()
            return True
        except ImportError:
            return False
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        # Override in subclasses
        pass
    
    def initialize(self, **kwargs) -> None:
        """Initialize the provider."""
        if self._initialized:
            return
        
        try:
            self._check_dependencies()
            self._initialize_backend(**kwargs)
            self._initialized = True
            self._logger.info(f"Provider {self.name} initialized successfully")
        except Exception as e:
            raise ProviderError(f"Failed to initialize provider {self.name}: {e}")
    
    def _initialize_backend(self, **kwargs) -> None:
        """Initialize the backend. Override in subclasses."""
        pass
    
    def ensure_initialized(self) -> None:
        """Ensure provider is initialized."""
        if not self._initialized:
            self.initialize()
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            'name': self.name,
            'version': self.version,
            'available': self.is_available(),
            'initialized': self._initialized,
        }


class LazyProvider(BaseProvider):
    """Base class for providers with lazy loading."""
    
    def __init__(self):
        super().__init__()
        self._modules = {}
    
    def _lazy_import(self, module_name: str, alias: Optional[str] = None) -> Any:
        """Lazy import a module."""
        if alias is None:
            alias = module_name
        
        if alias not in self._modules:
            try:
                self._modules[alias] = __import__(module_name)
                self._logger.debug(f"Lazy imported {module_name} as {alias}")
            except ImportError as e:
                raise ProviderError(f"Failed to import {module_name}: {e}")
        
        return self._modules[alias]
    
    def _get_module_attribute(self, module_name: str, attribute: str, alias: Optional[str] = None) -> Any:
        """Get an attribute from a lazily imported module."""
        module = self._lazy_import(module_name, alias)
        
        try:
            return getattr(module, attribute)
        except AttributeError as e:
            raise ProviderError(f"Module {module_name} has no attribute {attribute}: {e}")


class CachedProvider(BaseProvider):
    """Base class for providers with caching capabilities."""
    
    def __init__(self, cache_size: int = 100):
        super().__init__()
        self._cache = {}
        self._cache_size = cache_size
        self._cache_order = []
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]
        return None
    
    def _cache_set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        if key in self._cache:
            # Update existing item
            self._cache[key] = value
            self._cache_order.remove(key)
            self._cache_order.append(key)
        else:
            # Add new item
            if len(self._cache) >= self._cache_size:
                # Remove least recently used item
                oldest_key = self._cache_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[key] = value
            self._cache_order.append(key)
    
    def _cache_clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._cache_order.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        return {
            'size': len(self._cache),
            'max_size': self._cache_size,
            'keys': list(self._cache_order),
        }