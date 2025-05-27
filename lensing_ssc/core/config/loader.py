"""
Configuration loaders for different file formats.

This module provides loaders for YAML, JSON, and other configuration formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union
import json

from ..base.exceptions import ConfigurationError

# Optional imports for different formats
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    import toml
    _HAS_TOML = True
except ImportError:
    _HAS_TOML = False


class ConfigLoader(ABC):
    """Abstract base class for configuration loaders."""
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file.
        
        Parameters
        ----------
        path : str or Path
            Path to configuration file
            
        Returns
        -------
        dict
            Configuration data
        """
        pass
    
    @abstractmethod
    def save(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Parameters
        ----------
        data : dict
            Configuration data
        path : str or Path
            Output path
        """
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> list:
        """List of supported file extensions."""
        pass


class JSONConfigLoader(ConfigLoader):
    """JSON configuration loader."""
    
    @property
    def supported_extensions(self) -> list:
        return ['.json']
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON configuration file."""
        path = Path(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load JSON config from {path}: {e}")
    
    def save(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            raise ConfigurationError(f"Failed to save JSON config to {path}: {e}")


class YAMLConfigLoader(ConfigLoader):
    """YAML configuration loader."""
    
    def __init__(self):
        if not _HAS_YAML:
            raise ImportError("PyYAML is required for YAML configuration support. Install with: pip install PyYAML")
    
    @property
    def supported_extensions(self) -> list:
        return ['.yaml', '.yml']
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file."""
        path = Path(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data if data is not None else {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load YAML config from {path}: {e}")
    
    def save(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to save YAML config to {path}: {e}")


class TOMLConfigLoader(ConfigLoader):
    """TOML configuration loader."""
    
    def __init__(self):
        if not _HAS_TOML:
            raise ImportError("toml is required for TOML configuration support. Install with: pip install toml")
    
    @property
    def supported_extensions(self) -> list:
        return ['.toml']
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load TOML configuration file."""
        path = Path(path)
        
        try:
            return toml.load(path)
        except Exception as e:
            raise ConfigurationError(f"Failed to load TOML config from {path}: {e}")
    
    def save(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to TOML file."""
        path = Path(path)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                toml.dump(data, f)
        except Exception as e:
            raise ConfigurationError(f"Failed to save TOML config to {path}: {e}")


class INIConfigLoader(ConfigLoader):
    """INI configuration loader."""
    
    def __init__(self):
        import configparser
        self.configparser = configparser
    
    @property
    def supported_extensions(self) -> list:
        return ['.ini', '.cfg']
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load INI configuration file."""
        path = Path(path)
        
        try:
            config = self.configparser.ConfigParser()
            config.read(path)
            
            # Convert to nested dictionary
            result = {}
            for section_name in config.sections():
                result[section_name] = dict(config[section_name])
            
            return result
        except Exception as e:
            raise ConfigurationError(f"Failed to load INI config from {path}: {e}")
    
    def save(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to INI file."""
        path = Path(path)
        
        try:
            config = self.configparser.ConfigParser()
            
            for section_name, section_data in data.items():
                if isinstance(section_data, dict):
                    config[section_name] = {k: str(v) for k, v in section_data.items()}
                else:
                    # Handle non-section data
                    if 'DEFAULT' not in config:
                        config['DEFAULT'] = {}
                    config['DEFAULT'][section_name] = str(section_data)
            
            with open(path, 'w', encoding='utf-8') as f:
                config.write(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to save INI config to {path}: {e}")


def get_config_loader(file_path: Union[str, Path]) -> ConfigLoader:
    """Get appropriate config loader for file extension.
    
    Parameters
    ----------
    file_path : str or Path
        Path to configuration file
        
    Returns
    -------
    ConfigLoader
        Appropriate loader for the file format
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    loaders = {
        '.json': JSONConfigLoader,
        '.yaml': YAMLConfigLoader,
        '.yml': YAMLConfigLoader,
    }
    
    # Optional loaders
    if _HAS_TOML:
        loaders['.toml'] = TOMLConfigLoader
    
    loaders.update({
        '.ini': INIConfigLoader,
        '.cfg': INIConfigLoader,
    })
    
    if suffix not in loaders:
        raise ConfigurationError(f"Unsupported configuration file format: {suffix}")
    
    return loaders[suffix]()