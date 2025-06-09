# Core Configuration Module

This module provides a comprehensive configuration management system for LensingSSC with **support for multiple file formats**, **environment variable integration**, and **robust validation**. The system uses dataclasses for type-safe configuration handling and supports hot-reloading with caching.

## Design Philosophy

The configuration module follows these principles:
- **Type Safety**: All configuration uses dataclasses with full type annotations
- **Multi-Format Support**: YAML, JSON, TOML, and INI formats with graceful fallbacks
- **Environment Integration**: Seamless environment variable loading and override support
- **Validation**: Comprehensive validation with detailed error messages
- **Caching**: Intelligent caching with file modification tracking
- **Extensibility**: Pluggable loaders and validators for new formats

## Module Structure

### 📁 Core Components

#### `settings.py` (788 lines) ⚠️
**Type-safe configuration classes with comprehensive validation**

- **`ProcessingConfig`**: Main processing configuration with 25+ parameters
  - **Data paths**: `data_dir`, `output_dir`, `cache_dir` with automatic Path conversion
  - **Processing parameters**: `patch_size_deg`, `xsize`, `nside`, `num_workers`
  - **Memory management**: `chunk_size`, `cache_size_mb`, `memory_limit_mb`
  - **Provider settings**: Dynamic provider class mapping
  - **Validation**: Comprehensive parameter validation with detailed error messages

- **`AnalysisConfig`**: Statistical analysis configuration
  - **Redshift parameters**: `zs_list`, `ngal_list`, `sl_list`
  - **Spectrum parameters**: `lmin`, `lmax`, `nbin_ps_bs`
  - **Noise parameters**: `epsilon_noise`, `shape_noise_std`
  - **Cosmological parameters**: Validated `cosmo_params` dictionary
  - **Statistical parameters**: `bootstrap_samples`, `confidence_level`

- **`VisualizationConfig`**: Plotting and visualization settings
  - **Figure settings**: `figsize`, `dpi`, `fontsize`
  - **Color settings**: `colormap`, `color_palette`
  - **Output settings**: `plot_formats`, `save_plots`, `show_plots`
  - **Map plotting**: `map_projection`, `map_colormap`, `map_symmetric`

- **Global Configuration Functions**:
  - `get_config()`, `set_config()`, `reset_config()`: Global state management
  - `update_config()`, `load_config_from_env()`: Dynamic updates
  - `CONFIG_SCHEMA`: Comprehensive validation schema

#### `loader.py` (816 lines) ⚠️
**Multi-format configuration loaders with optional dependencies**

- **`ConfigLoader`** (ABC): Base loader interface
  - Abstract methods: `load()`, `save()`, `supported_extensions`, `format_name`
  - Utility methods: `is_available()`, `validate_data()`, `preprocess_data()`

- **Format-Specific Loaders**:
  - **`JSONConfigLoader`**: Always available, standard library JSON support
  - **`YAMLConfigLoader`**: Optional PyYAML dependency with safe loading
  - **`TOMLConfigLoader`**: Optional toml dependency for TOML files
  - **`INIConfigLoader`**: Standard library configparser with type conversion

- **Smart Loading Features**:
  - `get_config_loader()`: Auto-detects format from file extension
  - `get_available_loaders()`: Dynamic loader discovery
  - `suggest_format()`: Recommends best format for data complexity
  - Intelligent type conversion (Path objects, booleans, lists)

- **Dependency Management**:
  - Graceful fallback when optional dependencies unavailable
  - Runtime availability checking with helpful error messages
  - Optional dependency flags: `_HAS_YAML`, `_HAS_TOML`, `_HAS_CONFIGPARSER`

#### `manager.py` (713 lines)
**Configuration management with caching and validation**

- **`ConfigManager`**: Central configuration management
  - **Dynamic loader registration**: Auto-detects available format loaders
  - **Intelligent caching**: File modification tracking with cache invalidation
  - **Multi-config loading**: Load multiple configurations simultaneously
  - **Validation integration**: Schema-based validation with detailed reporting
  - **Template system**: Generate configuration templates with documentation

- **`EnvironmentConfigManager`**: Environment variable integration
  - **Prefix-based loading**: `LENSING_SSC_*` environment variables
  - **Type conversion**: Automatic string-to-type conversion for env vars
  - **Override management**: Track and clear environment overrides
  - **Bidirectional sync**: Set environment from config objects

- **Management Features**:
  - `load_config()`: Load with caching and validation
  - `save_config()`: Save with format auto-detection
  - `merge_configs()`: Combine multiple configuration objects
  - `validate_config_file()`: Pre-validation without loading
  - `get_config_template()`: Template generation with comments

#### `__init__.py` (56 lines)
**Clean public API with organized imports**

Exports comprehensive configuration API:
- **Configuration classes**: 3 main config classes
- **Global functions**: 6 global config management functions
- **Management classes**: 2 manager classes with full functionality
- **Loader classes**: 5 loader classes with format support

## Usage Examples

### Basic Configuration
```python
from lensing_ssc.core.config import ProcessingConfig, get_config

# Create default configuration
config = ProcessingConfig()
config.data_dir = "/path/to/data"
config.num_workers = 8
config.validate()  # Comprehensive validation

# Use as global configuration
from lensing_ssc.core.config import set_config, get_config
set_config(config)
current_config = get_config()
```

### Multi-Format Loading
```python
from lensing_ssc.core.config import ConfigManager

manager = ConfigManager()

# Load different formats automatically
proc_config = manager.load_config("config.yaml", "processing")
viz_config = manager.load_config("plots.json", "visualization")
analysis_config = manager.load_config("analysis.toml", "analysis")

# Multi-config loading
configs = manager.load_multiple_configs({
    "processing": "config.yaml",
    "analysis": "analysis.json"
})
```

### Environment Integration
```python
from lensing_ssc.core.config import EnvironmentConfigManager, load_config_from_env

# Load from environment variables
config = load_config_from_env()  # Loads LENSING_SSC_* variables

# Advanced environment management
env_manager = EnvironmentConfigManager()
env_data = env_manager.load_from_environment()
env_manager.set_environment_defaults(config)
```

### Template and Validation
```python
from lensing_ssc.core.config import ConfigManager

manager = ConfigManager()

# Generate templates
template = manager.get_config_template("processing")
# template includes comments and documentation

# Validate without loading
is_valid, errors, warnings = manager.validate_config_file("config.yaml")
if not is_valid:
    print("Validation errors:", errors)
```

## File Size Analysis
- **`settings.py`**: 788 lines - **EXCEEDS 500-line guideline** (comprehensive configs)
- **`loader.py`**: 816 lines - **EXCEEDS 500-line guideline** (4 loader implementations)
- **`manager.py`**: 713 lines - **EXCEEDS 500-line guideline** (full management features)
- **`__init__.py`**: 56 lines - Clean API export
- **Total**: 2,373 lines across 4 implementation files

## Architectural Benefits

1. **Type Safety**: Dataclass-based configs with full type annotations
2. **Flexibility**: Multiple formats with automatic detection and fallback
3. **Robustness**: Comprehensive validation with detailed error reporting
4. **Performance**: Intelligent caching with modification tracking
5. **Environment Integration**: Seamless environment variable support
6. **Extensibility**: Pluggable loaders and validators
7. **Developer Experience**: Templates, validation, and helpful error messages

## Integration Points

This module integrates with:
- **Base Module**: Uses validation framework and exception hierarchy
- **Provider System**: Dynamic provider class loading
- **Processing Pipelines**: Configuration injection throughout workflows
- **Environment Variables**: Runtime configuration override
- **File System**: Multi-format configuration persistence

## Configuration Schema

The module supports three main configuration types:

### Processing Configuration
- **Core Parameters**: 25+ processing and memory management settings
- **Path Management**: Automatic Path object handling
- **Provider Integration**: Dynamic provider class mapping
- **Validation**: Range checking, file existence, and logical consistency

### Analysis Configuration  
- **Scientific Parameters**: Redshift lists, galaxy densities, cosmology
- **Statistical Settings**: Bootstrap, confidence intervals, binning
- **Validation**: Cosmological parameter ranges and consistency

### Visualization Configuration
- **Figure Settings**: Size, DPI, fonts, colors
- **Output Control**: Formats, saving, display options
- **Map Plotting**: Projections, colormaps, layout settings

## Development Recommendations

1. **File Size**: Consider splitting large files by functionality
2. **Documentation**: Already comprehensive with examples
3. **Performance**: Caching system is well-implemented
4. **Testing**: Add comprehensive test coverage for all loaders
5. **Extension**: Plugin architecture ready for new formats
