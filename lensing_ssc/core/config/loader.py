import json
import os
from pathlib import Path

# Custom Exceptions
class UnsupportedFormatError(Exception):
    pass

class ConfigLoadError(Exception):
    pass

class ConfigSaveError(Exception):
    pass

# Base Loader
class ConfigLoader:
    name = "AbstractConfig"
    extensions = []

    def load(self, filepath):
        raise NotImplementedError

    def save(self, data, filepath):
        raise NotImplementedError

    @classmethod
    def get_info(cls):
        return {
            "name": cls.name,
            "extensions": cls.extensions,
            "loader_class": cls,
        }

    def __str__(self):
        return f"{self.__class__.__name__} (handles: {', '.join(self.extensions)})"

    def __repr__(self):
        return f"<{self.__class__.__name__} (handles: {', '.join(self.extensions)})>"

# Concrete Loaders
class JSONConfigLoader(ConfigLoader):
    name = "JSON"
    extensions = [".json"]

    def load(self, filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ConfigLoadError(f"File not found: {filepath}")
        except PermissionError:
            raise ConfigLoadError(f"Permission denied for file: {filepath}")
        except json.JSONDecodeError as e:
            raise ConfigLoadError(f"Error decoding JSON file {filepath}: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Could not load JSON file {filepath}: {e}")

    def save(self, data, filepath):
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)
        except PermissionError:
            raise ConfigSaveError(f"Permission denied for file: {filepath}")
        except Exception as e:
            raise ConfigSaveError(f"Could not save JSON file {filepath}: {e}")

class YAMLConfigLoader(ConfigLoader):
    name = "YAML"
    extensions = [".yaml", ".yml"]

    def load(self, filepath):
        # Placeholder for actual YAML loading logic
        # Requires PyYAML to be installed
        try:
            import yaml # type: ignore
            with open(filepath, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigLoadError(f"File not found: {filepath}")
        except PermissionError:
            raise ConfigLoadError(f"Permission denied for file: {filepath}")
        except ImportError:
            raise ConfigLoadError("PyYAML library is not installed. Please install it to use YAMLConfigLoader.")
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Error decoding YAML file {filepath}: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Could not load YAML file {filepath}: {e}")

    def save(self, data, filepath):
        # Placeholder for actual YAML saving logic
        try:
            import yaml # type: ignore
            with open(filepath, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        except PermissionError:
            raise ConfigSaveError(f"Permission denied for file: {filepath}")
        except ImportError:
            raise ConfigSaveError("PyYAML library is not installed. Please install it to use YAMLConfigLoader.")
        except Exception as e:
            raise ConfigSaveError(f"Could not save YAML file {filepath}: {e}")


class TOMLConfigLoader(ConfigLoader):
    name = "TOML"
    extensions = [".toml"]

    def load(self, filepath):
        # Placeholder for actual TOML loading logic
        # Requires toml library to be installed
        try:
            import toml # type: ignore
            with open(filepath, "r") as f:
                return toml.load(f)
        except FileNotFoundError:
            raise ConfigLoadError(f"File not found: {filepath}")
        except PermissionError:
            raise ConfigLoadError(f"Permission denied for file: {filepath}")
        except ImportError:
            raise ConfigLoadError("toml library is not installed. Please install it to use TOMLConfigLoader.")
        except toml.TomlDecodeError as e:
            raise ConfigLoadError(f"Error decoding TOML file {filepath}: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Could not load TOML file {filepath}: {e}")

    def save(self, data, filepath):
        # Placeholder for actual TOML saving logic
        try:
            import toml # type: ignore
            with open(filepath, "w") as f:
                toml.dump(data, f)
        except PermissionError:
            raise ConfigSaveError(f"Permission denied for file: {filepath}")
        except ImportError:
            raise ConfigSaveError("toml library is not installed. Please install it to use TOMLConfigLoader.")
        except Exception as e:
            raise ConfigSaveError(f"Could not save TOML file {filepath}: {e}")


class INIConfigLoader(ConfigLoader):
    name = "INI"
    extensions = [".ini"]

    def load(self, filepath):
        # Placeholder for actual INI loading logic
        # Requires configparser library (standard library)
        import configparser
        try:
            config = configparser.ConfigParser()
            # Read the file, ensuring it's treated as text
            with open(filepath, "r", encoding='utf-8') as f:
                config.read_file(f)

            # Convert to dict
            data = {section: dict(config.items(section)) for section in config.sections()}
            if not data and os.path.getsize(filepath) > 0: # File has content but no sections parsed
                 with open(filepath, "r", encoding='utf-8') as f:
                    content = f.read()
                 if not content.strip().startswith("["): # Heuristic: if not starting with section, likely invalid
                    raise ConfigLoadError(f"INI file {filepath} has no sections or is malformed.")
            return data
        except FileNotFoundError:
            raise ConfigLoadError(f"File not found: {filepath}")
        except PermissionError:
            raise ConfigLoadError(f"Permission denied for file: {filepath}")
        except configparser.Error as e:
            raise ConfigLoadError(f"Error decoding INI file {filepath}: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Could not load INI file {filepath}: {e}")

    def save(self, data, filepath):
        # Placeholder for actual INI saving logic
        import configparser
        config = configparser.ConfigParser()
        try:
            # Ensure data is in the format configparser expects:
            # A dictionary where keys are section names and values are dictionaries of key-value pairs.
            if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
                raise ConfigSaveError("Invalid data format for INI. Must be a dict of dicts (sections).")

            config.read_dict(data)
            with open(filepath, "w") as f:
                config.write(f)
        except PermissionError:
            raise ConfigSaveError(f"Permission denied for file: {filepath}")
        except Exception as e:
            raise ConfigSaveError(f"Could not save INI file {filepath}: {e}")


# Helper functions
_loaders = {
    "json": JSONConfigLoader,
    "yaml": YAMLConfigLoader,
    "toml": TOMLConfigLoader,
    "ini": INIConfigLoader,
}

def get_config_loader(filepath_or_ext: str | Path) -> ConfigLoader:
    ext = Path(filepath_or_ext).suffix.lower()
    if not ext: # If no suffix, assume it's an extension string itself
        ext = f".{filepath_or_ext.lower()}"

    if ext == ".yml": ext = ".yaml" # Normalize .yml to .yaml

    for loader_ext_key, loader_class in _loaders.items():
        if f".{loader_ext_key}" in loader_class.extensions:
            if ext == f".{loader_ext_key}":
                return loader_class()
    
    # Check full extensions list again for cases like .yml
    for loader_class in _loaders.values():
        if ext in loader_class.extensions:
            return loader_class()

    raise UnsupportedFormatError(f"No loader available for extension: {ext}")


def get_available_loaders():
    return {
        key: loader.get_info() for key, loader in _loaders.items()
    }

def get_loader_info(format_name: str):
    format_name = format_name.lower()
    if format_name not in _loaders:
        raise UnsupportedFormatError(f"No loader info for format: {format_name}")
    return _loaders[format_name].get_info()

def suggest_format(data, filename: str | Path | None = None):
    if filename:
        try:
            loader = get_config_loader(filename)
            return loader.name.lower()
        except UnsupportedFormatError:
            pass # Fallback to content-based suggestion or default

    # Basic suggestion logic (can be expanded)
    if isinstance(data, (list, dict)): # JSON is a good default for common data structures
        return "json"
    return "json" # Default fallback
