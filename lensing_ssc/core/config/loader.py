"""
Configuration loaders for different file formats.

This module provides loaders for YAML, JSON, TOML, and INI configuration formats,
with optional dependency handling and graceful fallbacks.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union, List
import json
import logging

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

try:
    import configparser
    _HAS_CONFIGPARSER = True
except ImportError:
    _HAS_CONFIGPARSER = False


class ConfigLoader(ABC):
    """Abstract base class for configuration loaders.
    
    This class defines the interface that all configuration loaders must implement,
    ensuring consistent behavior across different file formats.
    """
    
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
            
        Raises
        ------
        ConfigurationError
            If loading fails
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
            
        Raises
        ------
        ConfigurationError
            If saving fails
        """
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions.
        
        Returns
        -------
        list
            List of supported extensions (including the dot)
        """
        pass
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Human-readable format name.
        
        Returns
        -------
        str
            Format name for display purposes
        """
        pass
    
    def is_available(self) -> bool:
        """Check if the loader is available (dependencies installed).
        
        Returns
        -------
        bool
            True if loader can be used
        """
        return True  # Override in subclasses that require optional dependencies
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate configuration data before saving.
        
        Parameters
        ----------
        data : dict
            Configuration data to validate
            
        Returns
        -------
        bool
            True if data is valid for this format
        """
        # Basic validation - ensure it's a dictionary
        return isinstance(data, dict)
    
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data before saving (e.g., convert Path objects).
        
        Parameters
        ----------
        data : dict
            Original configuration data
            
        Returns
        -------
        dict
            Preprocessed data ready for saving
        """
        def convert_paths(obj):
            """Recursively convert Path objects to strings."""
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_paths(item) for item in obj]
            else:
                return obj
        
        return convert_paths(data)


class JSONConfigLoader(ConfigLoader):
    """JSON configuration loader.
    
    Provides loading and saving of configuration data in JSON format.
    JSON is always available as it's part of the Python standard library.
    """
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.json']
    
    @property
    def format_name(self) -> str:
        return "JSON"
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON configuration file.
        
        Parameters
        ----------
        path : str or Path
            Path to JSON file
            
        Returns
        -------
        dict
            Loaded configuration data
            
        Raises
        ------
        ConfigurationError
            If file cannot be loaded or parsed
        """
        path = Path(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Ensure we return a dictionary
            if not isinstance(data, dict):
                raise ConfigurationError(f"JSON file must contain an object/dictionary, got {type(data).__name__}")
                
            logging.debug(f"Successfully loaded JSON config from {path}")
            return data
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {path}: {e}")
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {path}")
        except PermissionError:
            raise ConfigurationError(f"Permission denied reading {path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load JSON config from {path}: {e}")
    
    def save(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to JSON file.
        
        Parameters
        ----------
        data : dict
            Configuration data
        path : str or Path
            Output path
            
        Raises
        ------
        ConfigurationError
            If file cannot be saved
        """
        path = Path(path)
        
        if not self.validate_data(data):
            raise ConfigurationError("Data must be a dictionary for JSON format")
        
        # Preprocess data to handle special types
        processed_data = self.preprocess_data(data)
        
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False, sort_keys=True)
                
            logging.debug(f"Successfully saved JSON config to {path}")
            
        except PermissionError:
            raise ConfigurationError(f"Permission denied writing to {path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save JSON config to {path}: {e}")


class YAMLConfigLoader(ConfigLoader):
    """YAML configuration loader.
    
    Provides loading and saving of configuration data in YAML format.
    Requires PyYAML to be installed.
    """
    
    def __init__(self):
        """Initialize YAML loader and check dependencies."""
        if not _HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML configuration support. "
                "Install with: pip install PyYAML"
            )
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.yaml', '.yml']
    
    @property
    def format_name(self) -> str:
        return "YAML"
    
    def is_available(self) -> bool:
        """Check if YAML loader is available."""
        return _HAS_YAML
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Parameters
        ----------
        path : str or Path
            Path to YAML file
            
        Returns
        -------
        dict
            Loaded configuration data
            
        Raises
        ------
        ConfigurationError
            If file cannot be loaded or parsed
        """
        path = Path(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            # Handle empty files
            if data is None:
                data = {}
                
            # Ensure we return a dictionary
            if not isinstance(data, dict):
                raise ConfigurationError(f"YAML file must contain a mapping/dictionary, got {type(data).__name__}")
                
            logging.debug(f"Successfully loaded YAML config from {path}")
            return data
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}")
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {path}")
        except PermissionError:
            raise ConfigurationError(f"Permission denied reading {path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load YAML config from {path}: {e}")
    
    def save(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Parameters
        ----------
        data : dict
            Configuration data
        path : str or Path
            Output path
            
        Raises
        ------
        ConfigurationError
            If file cannot be saved
        """
        path = Path(path)
        
        if not self.validate_data(data):
            raise ConfigurationError("Data must be a dictionary for YAML format")
        
        # Preprocess data to handle special types
        processed_data = self.preprocess_data(data)
        
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(processed_data, f, default_flow_style=False, indent=2, 
                         allow_unicode=True, sort_keys=True)
                
            logging.debug(f"Successfully saved YAML config to {path}")
            
        except PermissionError:
            raise ConfigurationError(f"Permission denied writing to {path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save YAML config to {path}: {e}")


class TOMLConfigLoader(ConfigLoader):
    """TOML configuration loader.
    
    Provides loading and saving of configuration data in TOML format.
    Requires toml package to be installed.
    """
    
    def __init__(self):
        """Initialize TOML loader and check dependencies."""
        if not _HAS_TOML:
            raise ImportError(
                "toml is required for TOML configuration support. "
                "Install with: pip install toml"
            )
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.toml']
    
    @property
    def format_name(self) -> str:
        return "TOML"
    
    def is_available(self) -> bool:
        """Check if TOML loader is available."""
        return _HAS_TOML
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load TOML configuration file.
        
        Parameters
        ----------
        path : str or Path
            Path to TOML file
            
        Returns
        -------
        dict
            Loaded configuration data
            
        Raises
        ------
        ConfigurationError
            If file cannot be loaded or parsed
        """
        path = Path(path)
        
        try:
            data = toml.load(path)
            logging.debug(f"Successfully loaded TOML config from {path}")
            return data
            
        except toml.TomlDecodeError as e:
            raise ConfigurationError(f"Invalid TOML in {path}: {e}")
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {path}")
        except PermissionError:
            raise ConfigurationError(f"Permission denied reading {path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load TOML config from {path}: {e}")
    
    def save(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to TOML file.
        
        Parameters
        ----------
        data : dict
            Configuration data
        path : str or Path
            Output path
            
        Raises
        ------
        ConfigurationError
            If file cannot be saved
        """
        path = Path(path)
        
        if not self.validate_data(data):
            raise ConfigurationError("Data must be a dictionary for TOML format")
        
        # Preprocess data to handle special types
        processed_data = self.preprocess_data(data)
        
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                toml.dump(processed_data, f)
                
            logging.debug(f"Successfully saved TOML config to {path}")
            
        except PermissionError:
            raise ConfigurationError(f"Permission denied writing to {path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save TOML config to {path}: {e}")


class INIConfigLoader(ConfigLoader):
    """INI configuration loader.
    
    Provides loading and saving of configuration data in INI format.
    Uses Python's standard library configparser module.
    """
    
    def __init__(self):
        """Initialize INI loader."""
        if not _HAS_CONFIGPARSER:
            raise ImportError("configparser is required for INI support")
        self.configparser = configparser
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.ini', '.cfg']
    
    @property
    def format_name(self) -> str:
        return "INI"
    
    def is_available(self) -> bool:
        """Check if INI loader is available."""
        return _HAS_CONFIGPARSER
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load INI configuration file.
        
        Parameters
        ----------
        path : str or Path
            Path to INI file
            
        Returns
        -------
        dict
            Loaded configuration data
            
        Raises
        ------
        ConfigurationError
            If file cannot be loaded or parsed
        """
        path = Path(path)
        
        try:
            config = self.configparser.ConfigParser()
            config.read(path, encoding='utf-8')
            
            # Convert to nested dictionary
            result = {}
            
            # Handle DEFAULT section specially
            if config.defaults():
                result['DEFAULT'] = dict(config.defaults())
            
            # Process other sections
            for section_name in config.sections():
                section_dict = {}
                for key, value in config[section_name].items():
                    # Try to convert values to appropriate types
                    section_dict[key] = self._convert_ini_value(value)
                result[section_name] = section_dict
            
            logging.debug(f"Successfully loaded INI config from {path}")
            return result
            
        except self.configparser.Error as e:
            raise ConfigurationError(f"Invalid INI format in {path}: {e}")
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {path}")
        except PermissionError:
            raise ConfigurationError(f"Permission denied reading {path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load INI config from {path}: {e}")
    
    def save(self, data: Dict[str, Any], path: Union[str, Path]) -> None:
        """Save configuration to INI file.
        
        Parameters
        ----------
        data : dict
            Configuration data
        path : str or Path
            Output path
            
        Raises
        ------
        ConfigurationError
            If file cannot be saved
        """
        path = Path(path)
        
        if not self.validate_data(data):
            raise ConfigurationError("Data must be a dictionary for INI format")
        
        try:
            config = self.configparser.ConfigParser()
            
            # Process nested dictionary structure
            for section_name, section_data in data.items():
                if isinstance(section_data, dict):
                    # Regular section
                    config[section_name] = {}
                    for key, value in section_data.items():
                        config[section_name][key] = str(value)
                else:
                    # Handle non-section data by putting it in DEFAULT
                    if 'DEFAULT' not in config:
                        config['DEFAULT'] = {}
                    config['DEFAULT'][section_name] = str(section_data)
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                config.write(f)
                
            logging.debug(f"Successfully saved INI config to {path}")
            
        except PermissionError:
            raise ConfigurationError(f"Permission denied writing to {path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save INI config to {path}: {e}")
    
    def _convert_ini_value(self, value: str) -> Any:
        """Convert INI string value to appropriate Python type.
        
        Parameters
        ----------
        value : str
            String value from INI file
            
        Returns
        -------
        Any
            Converted value
        """
        # Try boolean conversion
        if value.lower() in ('true', 'yes', 'on', '1'):
            return True
        elif value.lower() in ('false', 'no', 'off', '0'):
            return False
        
        # Try numeric conversions
        try:
            # Try integer first
            return int(value)
        except ValueError:
            pass
        
        try:
            # Try float
            return float(value)
        except ValueError:
            pass
        
        # Try list conversion (comma-separated)
        if ',' in value:
            try:
                items = [item.strip() for item in value.split(',')]
                return [self._convert_ini_value(item) for item in items]
            except Exception:
                pass
        
        # Return as string
        return value


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
        
    Raises
    ------
    ConfigurationError
        If file format is not supported or loader is not available
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    # Define available loaders
    loaders = {
        '.json': JSONConfigLoader,
    }
    
    # Add optional loaders if dependencies are available
    if _HAS_YAML:
        loaders['.yaml'] = YAMLConfigLoader
        loaders['.yml'] = YAMLConfigLoader
    
    if _HAS_TOML:
        loaders['.toml'] = TOMLConfigLoader
    
    if _HAS_CONFIGPARSER:
        loaders['.ini'] = INIConfigLoader
        loaders['.cfg'] = INIConfigLoader
    
    if suffix not in loaders:
        available = list(loaders.keys())
        raise ConfigurationError(f"Unsupported configuration file format: {suffix}. Available: {available}")
    
    try:
        return loaders[suffix]()
    except ImportError as e:
        raise ConfigurationError(f"Required dependency not available for {suffix} format: {e}")


def get_available_loaders() -> Dict[str, ConfigLoader]:
    """Get all available configuration loaders.
    
    Returns
    -------
    dict
        Dictionary mapping format names to loader instances
    """
    available_loaders = {}
    
    # JSON is always available
    try:
        json_loader = JSONConfigLoader()
        available_loaders['json'] = json_loader
    except Exception:
        pass
    
    # Optional loaders
    if _HAS_YAML:
        try:
            yaml_loader = YAMLConfigLoader()
            available_loaders['yaml'] = yaml_loader
        except Exception:
            pass
    
    if _HAS_TOML:
        try:
            toml_loader = TOMLConfigLoader()
            available_loaders['toml'] = toml_loader
        except Exception:
            pass
    
    if _HAS_CONFIGPARSER:
        try:
            ini_loader = INIConfigLoader()
            available_loaders['ini'] = ini_loader
        except Exception:
            pass
    
    return available_loaders


def get_loader_info() -> Dict[str, Dict[str, Any]]:
    """Get information about all potential loaders.
    
    Returns
    -------
    dict
        Information about each loader format
    """
    info = {
        'json': {
            'available': True,
            'extensions': ['.json'],
            'description': 'JavaScript Object Notation - always available',
            'dependency': 'Built-in (json module)',
        },
        'yaml': {
            'available': _HAS_YAML,
            'extensions': ['.yaml', '.yml'],
            'description': 'YAML Ain\'t Markup Language - human-readable',
            'dependency': 'PyYAML (pip install PyYAML)',
        },
        'toml': {
            'available': _HAS_TOML,
            'extensions': ['.toml'],
            'description': 'Tom\'s Obvious, Minimal Language - simple and readable',
            'dependency': 'toml (pip install toml)',
        },
        'ini': {
            'available': _HAS_CONFIGPARSER,
            'extensions': ['.ini', '.cfg'],
            'description': 'INI configuration format - simple key-value pairs',
            'dependency': 'Built-in (configparser module)',
        }
    }
    
    return info


def suggest_format(data: Dict[str, Any]) -> str:
    """Suggest the best configuration format for given data.
    
    Parameters
    ----------
    data : dict
        Configuration data to analyze
        
    Returns
    -------
    str
        Suggested format name
    """
    # Count nesting levels and complexity
    max_depth = _get_dict_depth(data)
    has_lists = _contains_lists(data)
    has_complex_types = _contains_complex_types(data)
    
    # Decision logic
    if max_depth <= 2 and not has_lists and not has_complex_types:
        return 'ini'  # Simple, flat structure
    elif _HAS_TOML and max_depth <= 3 and not has_complex_types:
        return 'toml'  # Good for moderate complexity
    elif _HAS_YAML:
        return 'yaml'  # Best human readability
    else:
        return 'json'  # Always available fallback


def _get_dict_depth(d: Dict[str, Any], depth: int = 1) -> int:
    """Calculate maximum nesting depth of a dictionary."""
    if not isinstance(d, dict):
        return depth
    
    if not d:
        return depth
    
    return max(_get_dict_depth(v, depth + 1) for v in d.values())


def _contains_lists(obj: Any) -> bool:
    """Check if object contains lists."""
    if isinstance(obj, list):
        return True
    elif isinstance(obj, dict):
        return any(_contains_lists(v) for v in obj.values())
    return False


def _contains_complex_types(obj: Any) -> bool:
    """Check if object contains complex types (beyond str, int, float, bool)."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return False
    elif isinstance(obj, (list, tuple)):
        return any(_contains_complex_types(item) for item in obj)
    elif isinstance(obj, dict):
        return any(_contains_complex_types(v) for v in obj.values())
    else:
        return True  # Complex type found