"""
Factory for creating and managing providers.
"""

from typing import Type, Dict, Any
import logging

class ProviderRegistry:
    _providers: Dict[str, Type] = {}
    _instances: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type):
        cls._providers[name] = provider_class
    
    @classmethod
    def get(cls, name: str, **kwargs):
        if name not in cls._instances:
            if name not in cls._providers:
                raise ValueError(f"Unknown provider: {name}")
            cls._instances[name] = cls._providers[name]()
            cls._instances[name].initialize(**kwargs)
        return cls._instances[name]

def register_provider(name: str):
    def decorator(cls):
        ProviderRegistry.register(name, cls)
        return cls
    return decorator

# Single function export
get_provider = ProviderRegistry.get