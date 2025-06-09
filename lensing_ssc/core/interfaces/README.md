# Core Interfaces Module

This module provides **abstract interfaces for dependency injection and provider pattern implementation** across the LensingSSC package. It enables **pluggable backend systems** for data access, computation, storage, and visualization with **consistent APIs** and **graceful fallback mechanisms**.

## Design Philosophy

The interfaces module follows these key principles:
- **Provider Pattern**: Pluggable implementations with consistent interfaces
- **Dependency Injection**: Abstract away heavy external dependencies
- **Backend Agnostic**: Support multiple computational and visualization backends
- **Error Resilience**: Graceful handling of missing dependencies
- **Extensibility**: Easy addition of new providers and backends
- **Type Safety**: Comprehensive type hints and validation interfaces

## Module Structure

### 📁 Core Components

#### `data_interface.py` (370+ lines)
**Abstract interfaces for data providers with enhanced functionality**

- **`DataProvider`** (Enhanced Base): Universal data provider interface
  - **Core Methods**: `name`, `version`, `dependencies`, `is_available()`
  - **New Features**: `validate_input()`, `get_provider_info()`
  - **Provider Discovery**: Dynamic provider information and capabilities
  - **Dependency Tracking**: List of required Python packages

- **`MapProvider`**: HEALPix map operations with comprehensive functionality
  - **Core Operations**: `read_map()`, `write_map()`, `get_nside()`, `get_npix()`
  - **Coordinate Operations**: `ang2pix()`, `pix2ang()`, `reorder_map()`
  - **Projection Operations**: `gnomonic_projection()`, `query_polygon()`
  - **Advanced Features**: Coordinate transformations and spatial queries

- **`CatalogProvider`**: N-body catalog operations for simulation data
  - **Data Access**: `read_catalog()`, `get_column()`, `get_attributes()`
  - **Metadata Operations**: `get_size()`, catalog attribute extraction
  - **Flexible Loading**: Support for multiple file formats and datasets

- **`ConvergenceMapProvider`**: Lensing-specific operations and statistics
  - **Map Creation**: `create_convergence_map()` for lensing analysis
  - **Statistical Analysis**: `power_spectrum()`, `bispectrum()`, `pdf()`
  - **Peak Analysis**: `locate_peaks()` with threshold detection
  - **Map Processing**: `smooth()` with configurable smoothing scales

#### `compute_interface.py` (530+ lines)
**Abstract interfaces for computational providers with backend management**

- **`ComputeProvider`** (Enhanced Base): Multi-backend computational interface
  - **Backend Management**: `supported_backends`, `set_backend()`, `get_backend()`
  - **Performance Tools**: `benchmark_backend()`, `get_backend_info()`
  - **Runtime Optimization**: Backend-specific performance hints

- **`StatisticsProvider`**: Statistical computation with multiple backends
  - **Power Analysis**: `power_spectrum()`, `bispectrum()` with configurable binning
  - **Distribution Analysis**: `probability_density_function()`, `peak_counts()`
  - **Morphology Analysis**: `minkowski_functionals()` for map topology
  - **Correlation Analysis**: `correlation_function()`, `covariance_matrix()`

- **`GeometryProvider`**: Geometric operations for spherical analysis
  - **Grid Generation**: `fibonacci_grid()` for optimal sampling
  - **Patch Operations**: `patch_extraction()` from full-sky maps
  - **Coordinate Operations**: `spherical_distance()`, `coordinate_transform()`
  - **Rotation Operations**: `rotation_matrix()` with multiple conventions

- **`OptimizationProvider`**: Mathematical optimization interface
  - **Function Minimization**: `minimize()` with multiple algorithms
  - **Curve Fitting**: `curve_fit()` with parameter estimation

- **`InterpolationProvider`**: Multi-dimensional interpolation interface
  - **1D/2D Interpolation**: `interpolate_1d()`, `interpolate_2d()`
  - **Spherical Interpolation**: `spherical_interpolation()` for sky maps

- **`FilteringProvider`**: Signal processing and filtering interface
  - **Spatial Filtering**: `gaussian_filter()`, `median_filter()`
  - **Frequency Filtering**: `fourier_filter()` with custom functions

#### `storage_interface.py` (320+ lines)
**Abstract interfaces for storage and I/O providers with enhanced file operations**

- **`StorageProvider`** (Enhanced Base): File system operations interface
  - **Core Operations**: `exists()`, `mkdir()`, `remove()`, `list_files()`
  - **Enhanced Features**: `get_file_info()`, `copy_file()`
  - **Metadata Support**: File size, permissions, modification times

- **`FileFormatProvider`**: Generic file format interface
  - **Format Detection**: `supported_extensions`, `can_read()`, `can_write()`
  - **Data Operations**: `read()`, `write()`, `get_metadata()`
  - **Format Validation**: Extension and content validation

- **`FITSProvider`**: FITS file operations for astronomical data
  - **Map Operations**: `read_fits_map()`, `write_fits_map()`
  - **Table Operations**: `read_fits_table()`, `write_fits_table()`
  - **HDU Management**: Multi-extension FITS support

- **`HDF5Provider`**: HDF5 operations for large dataset management
  - **Dataset Operations**: `read_hdf5_dataset()`, `write_hdf5_dataset()`
  - **Group Operations**: `read_hdf5_group()`, `write_hdf5_group()`
  - **Metadata Operations**: `list_datasets()`, `get_attributes()`

- **`CSVProvider`**: CSV file operations for tabular data
- **`NPYProvider`**: NumPy array serialization with NPY/NPZ support

- **`CacheProvider`**: Intelligent caching interface
  - **Cache Operations**: `get()`, `set()`, `delete()`, `clear()`
  - **Cache Management**: `keys()`, `size()`, TTL support

- **`CheckpointProvider`**: Checkpoint management for long-running processes
  - **Checkpoint Operations**: `save_checkpoint()`, `load_checkpoint()`
  - **Checkpoint Management**: `list_checkpoints()`, `checkpoint_exists()`

- **`CompressionProvider`**: Data compression interface
  - **Data Compression**: `compress()`, `decompress()` with multiple algorithms
  - **Array Compression**: `compress_array()`, `decompress_array()`

#### `plotting_interface.py` (355+ lines)
**Abstract interfaces for plotting and visualization providers with multi-backend support**

- **`PlottingProvider`** (Enhanced Base): Multi-backend plotting interface
  - **Backend Management**: `supported_backends`, `supported_formats`
  - **Figure Operations**: `create_figure()`, `save_figure()`, `show_figure()`
  - **Style Management**: `set_style()`, `get_style_options()`
  - **Enhanced Documentation**: Comprehensive parameter descriptions

- **`MapPlottingProvider`**: Specialized map visualization interface
  - **Sky Projections**: `plot_mollweide()`, `plot_orthographic()`, `plot_gnomonic()`
  - **Patch Visualization**: `plot_patch()`, `plot_patches_grid()`
  - **Full-Sky Plotting**: `plot_map()` with colorbar and title support

- **`StatisticsPlottingProvider`**: Statistical visualization interface
  - **Spectrum Plotting**: `plot_power_spectrum()`, `plot_bispectrum()`
  - **Distribution Plotting**: `plot_pdf()`, `plot_peak_counts()`
  - **Comparison Plotting**: `plot_comparison()`, `plot_ratio()`
  - **Matrix Visualization**: `plot_correlation_matrix()`

- **`InteractivePlottingProvider`**: Interactive visualization interface
  - **Interactive Maps**: `create_interactive_map()`
  - **Dashboard Creation**: `create_dashboard()`
  - **Widget Management**: `add_widget()`

- **`VisualizationProvider`**: High-level visualization interface
  - **Summary Plots**: `create_summary_plot()` for comprehensive analysis
  - **Comparison Analysis**: `create_comparison_plot()` between datasets
  - **Report Generation**: `create_analysis_report()` with multi-format output
  - **Animation Creation**: `create_animation()` from data sequences

#### `__init__.py` (67 lines)
**Clean public API with comprehensive exports**

Exports all interface classes organized by category:
- **Data Interfaces**: 4 provider classes for data access
- **Compute Interfaces**: 6 provider classes for computation
- **Storage Interfaces**: 9 provider classes for I/O operations
- **Plotting Interfaces**: 5 provider classes for visualization

## Usage Examples

### Provider Pattern Implementation
```python
from lensing_ssc.core.interfaces import MapProvider
from lensing_ssc.core.base import MapData

class HealpixMapProvider(MapProvider):
    @property
    def name(self) -> str:
        return "healpy"
    
    @property
    def version(self) -> str:
        return "1.16.0"
    
    @property
    def dependencies(self) -> List[str]:
        return ["healpy", "numpy"]
    
    def is_available(self) -> bool:
        try:
            import healpy
            return True
        except ImportError:
            return False
    
    def read_map(self, path: Union[str, Path], **kwargs) -> MapData:
        import healpy as hp
        data = hp.read_map(str(path), **kwargs)
        return MapData(data=data, metadata={"source": "healpy"})
```

### Multi-Backend Computation
```python
from lensing_ssc.core.interfaces import StatisticsProvider

class NumpyStatisticsProvider(StatisticsProvider):
    def __init__(self):
        self._backend = "numpy"
    
    @property
    def supported_backends(self) -> List[str]:
        return ["numpy", "cupy"]
    
    def set_backend(self, backend: str) -> None:
        if backend not in self.supported_backends:
            raise ValueError(f"Unsupported backend: {backend}")
        self._backend = backend
    
    def power_spectrum(self, data: np.ndarray, l_edges: np.ndarray, **kwargs):
        # Implementation with selected backend
        if self._backend == "cupy":
            import cupy as cp
            data = cp.asarray(data)
        # Compute power spectrum...
```

### Storage Provider Implementation
```python
from lensing_ssc.core.interfaces import HDF5Provider

class H5pyHDF5Provider(HDF5Provider):
    @property
    def dependencies(self) -> List[str]:
        return ["h5py", "numpy"]
    
    def read_hdf5_dataset(self, path: Union[str, Path], dataset: str, **kwargs):
        import h5py
        with h5py.File(path, 'r') as f:
            return f[dataset][...]
    
    def write_hdf5_dataset(self, data: np.ndarray, path: Union[str, Path], 
                          dataset: str, overwrite: bool = False, **kwargs):
        import h5py
        mode = 'w' if overwrite else 'a'
        with h5py.File(path, mode) as f:
            f.create_dataset(dataset, data=data, **kwargs)
```

### Visualization Provider Implementation
```python
from lensing_ssc.core.interfaces import MapPlottingProvider

class MatplotlibMapProvider(MapPlottingProvider):
    @property
    def supported_backends(self) -> List[str]:
        return ["matplotlib"]
    
    @property
    def supported_formats(self) -> List[str]:
        return ["png", "pdf", "svg", "eps"]
    
    def plot_mollweide(self, map_data: MapData, title: Optional[str] = None, **kwargs):
        import matplotlib.pyplot as plt
        import healpy as hp
        
        fig = plt.figure(figsize=(12, 8))
        hp.mollview(map_data.data, title=title, **kwargs)
        return fig
```

## File Size Analysis
- **`data_interface.py`**: 370+ lines (enhanced with validation and discovery)
- **`compute_interface.py`**: 530+ lines (enhanced with backend management)
- **`storage_interface.py`**: 320+ lines (enhanced with file operations)
- **`plotting_interface.py`**: 355+ lines (enhanced with style management)
- **`__init__.py`**: 67 lines - Clean API export
- **Total**: 1,642+ lines across 5 files

## Architectural Benefits

1. **Dependency Abstraction**: Clean separation from heavy external libraries
2. **Backend Flexibility**: Multiple computational and visualization backends
3. **Provider Discovery**: Runtime capability detection and information
4. **Error Resilience**: Graceful handling of missing dependencies
5. **Type Safety**: Comprehensive type hints throughout all interfaces
6. **Extensibility**: Easy addition of new providers and capabilities
7. **Performance**: Backend-specific optimizations and benchmarking
8. **Validation**: Input validation patterns across all providers

## Integration Points

This module integrates with:
- **Provider System**: Concrete implementations of all interfaces
- **Base Module**: Uses data structures and exception hierarchy
- **Configuration System**: Provider selection and configuration
- **Processing Pipelines**: Computational provider usage
- **I/O Operations**: Storage provider utilization
- **Visualization System**: Plotting provider implementations

## Enhanced Features

### Provider Discovery and Information
- **Dynamic Capabilities**: Runtime discovery of provider features
- **Dependency Tracking**: Automatic dependency validation
- **Version Management**: Provider version compatibility checking
- **Performance Metrics**: Backend benchmarking capabilities

### Backend Management
- **Multi-Backend Support**: NumPy, CuPy, JAX for computation
- **Dynamic Switching**: Runtime backend selection
- **Performance Optimization**: Backend-specific optimizations
- **Fallback Mechanisms**: Graceful degradation to available backends

### Validation and Error Handling
- **Input Validation**: Standardized validation patterns
- **Error Context**: Rich error information with context
- **Dependency Checking**: Comprehensive dependency validation
- **Graceful Fallbacks**: Robust error handling

## Development Recommendations

1. **Implementation Completeness**: All abstract methods must be implemented
2. **Error Handling**: Use base exception classes for consistency
3. **Documentation**: Maintain comprehensive docstrings
4. **Testing**: Mock providers for testing without heavy dependencies
5. **Performance**: Consider backend-specific optimizations
6. **Validation**: Implement robust input validation in concrete providers

## Provider Implementation Guidelines

### Required Methods
All providers must implement:
- Basic provider info (`name`, `version`, `dependencies`)
- Availability checking (`is_available()`)
- Input validation (`validate_input()`)
- Provider information (`get_provider_info()`)

### Error Handling
Use appropriate exception types:
- `ValidationError` for input validation failures
- `RuntimeError` for backend initialization failures  
- `OSError` for file operation failures
- `ValueError` for invalid parameter values

### Documentation Standards
- Comprehensive docstrings for all methods
- Parameter and return type documentation
- Exception documentation
- Usage examples where appropriate

This interfaces module provides the foundation for a flexible, extensible, and robust provider system that enables LensingSSC to work with multiple backend implementations while maintaining consistent APIs and graceful error handling.