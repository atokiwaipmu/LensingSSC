# Core Base Module

This module provides the foundational components of the LensingSSC package with **minimal dependencies** (numpy and standard library only). It implements a lightweight, dependency-abstracted architecture that other modules can build upon.

## Design Philosophy

The base module follows these key principles:
- **Minimal Dependencies**: Only numpy and Python standard library
- **Provider Pattern Ready**: Designed to work with the provider abstraction system
- **Comprehensive Validation**: Robust data validation and error handling
- **Type Safety**: Full type hints and validation throughout
- **Extensible Architecture**: Abstract base classes for consistent interfaces

## Module Structure

### 📁 Core Components

#### `data_structures.py` (748 lines)
**Primary data containers with built-in validation and serialization**

- **`DataStructure`** (ABC): Base class for all data structures
  - Abstract methods: `validate()`, `to_dict()`, `from_dict()`
  - Deep copy support and consistent string representation

- **`MapData`**: Container for astronomical map data
  - Automatic numpy array conversion and validation
  - Built-in statistics computation (`get_statistics()`)
  - Masking support with metadata tracking
  - Finite value validation and sanity checks

- **`PatchData`**: Container for extracted map patches  
  - 3D array storage: `(n_patches, height, width)`
  - Spherical coordinate center tracking
  - Subset extraction and patch-level statistics
  - Size and coordinate validation

- **`StatisticsData`**: Container for statistical analysis results
  - Dictionary-based storage for multiple statistics
  - Bin arrays and error propagation support
  - Monotonic bin validation and consistency checks
  - Dynamic statistic addition/removal

- **Utility Functions**:
  - `combine_map_data()`: Combine multiple maps with various operations

#### `coordinates.py` (710 lines)
**Complete coordinate system implementation with transformations**

- **`Coordinates`** (ABC): Base coordinate interface
- **`SphericalCoordinates`**: Full spherical coordinate implementation
  - Automatic validation and normalization
  - Angular distance calculation (haversine formula)
  - Great circle bearing computation
  - Degree/radian conversion utilities

- **`CartesianCoordinates`**: Complete Cartesian implementation
  - Vector operations (dot, cross, normalize, project)
  - Magnitude calculations and angle computations
  - Operator overloading for vector arithmetic

- **`CoordinateTransformer`**: Batch transformation utilities
  - `spherical_to_cartesian_batch()`: Vectorized conversions
  - `cartesian_to_spherical_batch()`: Efficient batch processing
  - `angular_distance_batch()`: Mass distance calculations

- **`RotationMatrix`**: Advanced rotation utilities
  - Axis-angle rotation matrices
  - Euler angle support (ZYZ, XYZ, ZXZ conventions)
  - Vector-to-vector alignment
  - Spherical coordinate rotation
  - Rotation composition

#### `exceptions.py` (541 lines)
**Hierarchical exception system with enhanced error reporting**

- **`LensingSSCError`** (base): Enhanced exception with context
  - Detail dictionary for structured error information
  - Cause tracking for exception chaining
  - Method chaining for error enrichment

- **Specialized Exceptions**:
  - `ValidationError`: Data validation failures
  - `ConfigurationError`: Configuration issues  
  - `ProviderError`: External library provider failures
  - `ProcessingError`: Pipeline processing failures
  - `DataError`: Data format/corruption issues
  - `GeometryError`: Coordinate/geometric operation failures
  - `StatisticsError`: Statistical analysis failures
  - `IOError`: Input/output operation failures
  - `VisualizationError`: Plotting and visualization failures

- **Utility Functions**:
  - `reraise_with_context()`: Exception enhancement
  - `validate_not_none()`, `validate_type()`: Quick validators
  - `validate_positive()`, `validate_range()`: Numeric validation

#### `validation.py` (952 lines)
**Comprehensive validation framework with multiple validator types**

- **`Validator`** (ABC): Base validator with error/warning collection
- **`DataValidator`**: Validates data structures and arrays
  - Automatic type detection and appropriate validation
  - Strict/permissive modes
  - NaN/infinity detection and statistical validation

- **`ConfigValidator`**: Schema-based configuration validation
  - Required field checking
  - Type validation with complex type support
  - Custom validator function support
  - Unknown field detection in strict mode

- **`PathValidator`**: File system path validation
  - Existence and permission checking
  - File extension and size validation
  - Directory content and writeability testing
  - Pattern matching support

- **`RangeValidator`**: Numeric range and constraint validation
  - Scalar and array support
  - Inclusive/exclusive range checking
  - Finite value validation
  - Positive value validation

- **Domain-Specific Validators**:
  - `validate_spherical_coordinates()`: Coordinate validation
  - `validate_patch_size()`: Patch size validation  
  - `validate_nside()`: HEALPix NSIDE validation
  - `validate_redshift()`: Cosmological redshift validation
  - `validate_angular_scale()`: Angular scale validation

#### `__init__.py` (76 lines)
**Clean public API with organized imports**

Exports all major classes and functions with logical grouping:
- Exception hierarchy (10 exception types)
- Data structures (4 main classes)
- Coordinate systems (5 classes)
- Validation framework (8 validators + utility functions)

## Usage Examples

### Data Structures
```python
from lensing_ssc.core.base import MapData, PatchData

# Create and validate map data
map_data = MapData(
    data=kappa_array,
    shape=kappa_array.shape,
    dtype=kappa_array.dtype,
    metadata={"source": "simulation"}
)
map_data.validate()  # Comprehensive validation
stats = map_data.get_statistics()  # Built-in statistics
```

### Coordinate Transformations
```python
from lensing_ssc.core.base import SphericalCoordinates, CoordinateTransformer

# Single coordinate
sphere_coord = SphericalCoordinates.from_degrees(r=1.0, theta_deg=45, phi_deg=90)
cart_coord = sphere_coord.to_cartesian()

# Batch operations
coords_array = np.array([[1, 0.5, 1.2], [1, 1.0, 2.4]])  # [r, theta, phi]
cartesian_batch = CoordinateTransformer.spherical_to_cartesian_batch(coords_array)
```

### Validation Framework
```python
from lensing_ssc.core.base import DataValidator, validate_spherical_coordinates

# Validate complex data
validator = DataValidator(strict=False)
is_valid = validator.validate(my_data)
if not is_valid:
    print("Errors:", validator.get_errors())
    print("Warnings:", validator.get_warnings())

# Domain-specific validation
coords_valid = validate_spherical_coordinates(theta_phi_array)
```

### Exception Handling
```python
from lensing_ssc.core.base import ValidationError, reraise_with_context

try:
    # Some operation
    pass
except Exception as e:
    reraise_with_context(e, "During patch extraction", {"patch_id": 123})
```

## File Size Analysis
- **`data_structures.py`**: 748 lines - Comprehensive but within limits
- **`coordinates.py`**: 710 lines - Approaching limit, well-structured
- **`validation.py`**: 952 lines - **EXCEEDS 500-line guideline**
- **`exceptions.py`**: 541 lines - Slightly over guideline
- **Total**: 2,951 lines across 4 implementation files

## Architectural Benefits

1. **Independence**: Works without heavy astronomical libraries
2. **Consistency**: Unified interfaces across all data structures
3. **Robustness**: Comprehensive validation and error handling
4. **Performance**: Vectorized operations where appropriate
5. **Extensibility**: Abstract base classes enable easy extension
6. **Interoperability**: Serialization support for data persistence

## Integration Points

This module integrates with:
- **Provider System**: Through abstract interfaces
- **Configuration System**: Via validation framework  
- **Processing Pipelines**: Through data structures
- **I/O Operations**: Via serialization methods
- **Error Handling**: Throughout the entire package
