"""
Validation utilities for data and configuration.

This module provides validation classes and functions to ensure data integrity
and configuration validity throughout the package. All validation uses minimal
dependencies (numpy and standard library only).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Type
import numpy as np
from pathlib import Path
import re

from .exceptions import ValidationError, ConfigurationError, DataError
from .data_structures import DataStructure, MapData, PatchData, StatisticsData


class Validator(ABC):
    """Abstract base class for validators."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate the given data.
        
        Parameters
        ----------
        data : Any
            Data to validate
            
        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        pass
    
    def get_errors(self) -> List[str]:
        """Get validation error messages.
        
        Returns
        -------
        list
            List of error messages
        """
        return self.errors.copy()
    
    def get_warnings(self) -> List[str]:
        """Get validation warning messages.
        
        Returns
        -------
        list
            List of warning messages
        """
        return self.warnings.copy()
    
    def clear_messages(self) -> None:
        """Clear all error and warning messages."""
        self.errors.clear()
        self.warnings.clear()
    
    def add_error(self, message: str) -> None:
        """Add an error message.
        
        Parameters
        ----------
        message : str
            Error message to add
        """
        self.errors.append(message)
    
    def add_warning(self, message: str) -> None:
        """Add a warning message.
        
        Parameters
        ----------
        message : str
            Warning message to add
        """
        self.warnings.append(message)
    
    def has_errors(self) -> bool:
        """Check if there are any errors.
        
        Returns
        -------
        bool
            True if there are errors
        """
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings.
        
        Returns
        -------
        bool
            True if there are warnings
        """
        return len(self.warnings) > 0


class DataValidator(Validator):
    """Validator for data structures and arrays."""
    
    def __init__(self, strict: bool = False):
        """Initialize data validator.
        
        Parameters
        ----------
        strict : bool
            If True, treat warnings as errors
        """
        super().__init__()
        self.strict = strict
    
    def validate(self, data: Any) -> bool:
        """Validate data structure or array.
        
        Parameters
        ----------
        data : Any
            Data to validate
            
        Returns
        -------
        bool
            True if validation passes
        """
        self.clear_messages()
        
        try:
            if isinstance(data, DataStructure):
                return self._validate_data_structure(data)
            elif isinstance(data, np.ndarray):
                return self._validate_array(data)
            elif isinstance(data, (list, tuple)):
                return self._validate_sequence(data)
            elif isinstance(data, dict):
                return self._validate_dictionary(data)
            else:
                self.add_error(f"Unsupported data type: {type(data)}")
                return False
        except Exception as e:
            self.add_error(f"Validation failed with exception: {e}")
            return False
    
    def _validate_data_structure(self, data: DataStructure) -> bool:
        """Validate a DataStructure object.
        
        Parameters
        ----------
        data : DataStructure
            Data structure to validate
            
        Returns
        -------
        bool
            True if validation passes
        """
        try:
            is_valid = data.validate()
            if not is_valid:
                self.add_error("Data structure validation returned False")
            return is_valid
        except ValidationError as e:
            self.add_error(str(e))
            return False
        except Exception as e:
            self.add_error(f"Unexpected error during data structure validation: {e}")
            return False
    
    def _validate_array(self, data: np.ndarray) -> bool:
        """Validate a numpy array.
        
        Parameters
        ----------
        data : np.ndarray
            Array to validate
            
        Returns
        -------
        bool
            True if validation passes
        """
        is_valid = True
        
        # Check for empty arrays
        if data.size == 0:
            self.add_error("Array is empty")
            return False
        
        # Check for NaN values
        if np.any(np.isnan(data)):
            n_nan = np.sum(np.isnan(data))
            if self.strict:
                self.add_error(f"Array contains {n_nan} NaN values")
                is_valid = False
            else:
                self.add_warning(f"Array contains {n_nan} NaN values")
        
        # Check for infinite values
        if np.any(np.isinf(data)):
            n_inf = np.sum(np.isinf(data))
            if self.strict:
                self.add_error(f"Array contains {n_inf} infinite values")
                is_valid = False
            else:
                self.add_warning(f"Array contains {n_inf} infinite values")
        
        # Check for suspicious values
        if data.dtype.kind in ['f', 'c']:  # floating point or complex
            max_abs = np.abs(data[np.isfinite(data)]).max() if np.any(np.isfinite(data)) else 0
            if max_abs > 1e10:
                self.add_warning(f"Array contains very large values (max: {max_abs})")
            elif max_abs == 0:
                self.add_warning("All finite values in array are zero")
        
        return is_valid
    
    def _validate_sequence(self, data: Union[List, Tuple]) -> bool:
        """Validate a sequence (list or tuple).
        
        Parameters
        ----------
        data : list or tuple
            Sequence to validate
            
        Returns
        -------
        bool
            True if validation passes
        """
        if len(data) == 0:
            self.add_warning("Sequence is empty")
            return not self.strict
        
        # If all elements are arrays, validate each
        if all(isinstance(item, np.ndarray) for item in data):
            is_valid = True
            for i, item in enumerate(data):
                if not self._validate_array(item):
                    self.add_error(f"Array at index {i} failed validation")
                    is_valid = False
            return is_valid
        
        # Check for mixed types
        types = set(type(item) for item in data)
        if len(types) > 1:
            self.add_warning(f"Sequence contains mixed types: {types}")
        
        return True
    
    def _validate_dictionary(self, data: Dict[str, Any]) -> bool:
        """Validate a dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary to validate
            
        Returns
        -------
        bool
            True if validation passes
        """
        if len(data) == 0:
            self.add_warning("Dictionary is empty")
            return not self.strict
        
        # Check for None values
        none_keys = [k for k, v in data.items() if v is None]
        if none_keys:
            self.add_warning(f"Dictionary contains None values for keys: {none_keys}")
        
        # Validate numpy arrays in the dictionary
        is_valid = True
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if not self._validate_array(value):
                    self.add_error(f"Array for key '{key}' failed validation")
                    is_valid = False
        
        return is_valid


class ConfigValidator(Validator):
    """Validator for configuration objects and dictionaries."""
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None, strict: bool = True):
        """Initialize config validator.
        
        Parameters
        ----------
        schema : dict, optional
            Validation schema
        strict : bool
            If True, unknown fields cause errors
        """
        super().__init__()
        self.schema = schema or {}
        self.strict = strict
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary.
        
        Parameters
        ----------
        config : dict
            Configuration to validate
            
        Returns
        -------
        bool
            True if validation passes
        """
        self.clear_messages()
        
        try:
            # Check required fields
            if not self._check_required_fields(config):
                return False
            
            # Validate field types
            if not self._validate_field_types(config):
                return False
            
            # Validate field values
            if not self._validate_field_values(config):
                return False
            
            # Check for unknown fields
            if self.strict:
                self._check_unknown_fields(config)
            
            return not self.has_errors()
        except Exception as e:
            self.add_error(f"Configuration validation failed: {e}")
            return False
    
    def _check_required_fields(self, config: Dict[str, Any]) -> bool:
        """Check that all required fields are present.
        
        Parameters
        ----------
        config : dict
            Configuration to check
            
        Returns
        -------
        bool
            True if all required fields present
        """
        required_fields = self.schema.get('required', [])
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            self.add_error(f"Missing required fields: {missing_fields}")
            return False
        
        return True
    
    def _validate_field_types(self, config: Dict[str, Any]) -> bool:
        """Validate field types against schema.
        
        Parameters
        ----------
        config : dict
            Configuration to validate
            
        Returns
        -------
        bool
            True if all types are correct
        """
        field_types = self.schema.get('types', {})
        is_valid = True
        
        for field, expected_type in field_types.items():
            if field in config:
                value = config[field]
                if not self._check_type(value, expected_type):
                    self.add_error(
                        f"Field '{field}' has type {type(value).__name__}, "
                        f"expected {self._type_name(expected_type)}"
                    )
                    is_valid = False
        
        return is_valid
    
    def _validate_field_values(self, config: Dict[str, Any]) -> bool:
        """Validate field values against constraints.
        
        Parameters
        ----------
        config : dict
            Configuration to validate
            
        Returns
        -------
        bool
            True if all values are valid
        """
        validators = self.schema.get('validators', {})
        is_valid = True
        
        for field, validator_func in validators.items():
            if field in config:
                try:
                    if not validator_func(config[field]):
                        self.add_error(f"Field '{field}' failed validation")
                        is_valid = False
                except Exception as e:
                    self.add_error(f"Validation of field '{field}' raised exception: {e}")
                    is_valid = False
        
        return is_valid
    
    def _check_unknown_fields(self, config: Dict[str, Any]) -> None:
        """Check for unknown fields in strict mode.
        
        Parameters
        ----------
        config : dict
            Configuration to check
        """
        known_fields = set()
        known_fields.update(self.schema.get('required', []))
        known_fields.update(self.schema.get('types', {}).keys())
        known_fields.update(self.schema.get('validators', {}).keys())
        known_fields.update(self.schema.get('optional', []))
        
        unknown_fields = set(config.keys()) - known_fields
        if unknown_fields:
            self.add_warning(f"Unknown fields (will be ignored): {unknown_fields}")
    
    def _check_type(self, value: Any, expected_type: Union[Type, Tuple[Type, ...]]) -> bool:
        """Check if value matches expected type.
        
        Parameters
        ----------
        value : Any
            Value to check
        expected_type : type or tuple of types
            Expected type(s)
            
        Returns
        -------
        bool
            True if type matches
        """
        if isinstance(expected_type, tuple):
            return isinstance(value, expected_type)
        else:
            return isinstance(value, expected_type)
    
    def _type_name(self, type_spec: Union[Type, Tuple[Type, ...]]) -> str:
        """Get human-readable type name.
        
        Parameters
        ----------
        type_spec : type or tuple of types
            Type specification
            
        Returns
        -------
        str
            Human-readable type name
        """
        if isinstance(type_spec, tuple):
            names = [t.__name__ for t in type_spec]
            return " or ".join(names)
        else:
            return type_spec.__name__


class PathValidator(Validator):
    """Validator for file and directory paths."""
    
    def __init__(self, check_permissions: bool = True):
        """Initialize path validator.
        
        Parameters
        ----------
        check_permissions : bool
            Whether to check file/directory permissions
        """
        super().__init__()
        self.check_permissions = check_permissions
    
    def validate(self, path: Union[str, Path]) -> bool:
        """Validate a file or directory path.
        
        Parameters
        ----------
        path : str or Path
            Path to validate
            
        Returns
        -------
        bool
            True if validation passes
        """
        self.clear_messages()
        
        try:
            path_obj = Path(path)
            
            # Check if path exists
            if not path_obj.exists():
                self.add_error(f"Path does not exist: {path}")
                return False
            
            return True
        except Exception as e:
            self.add_error(f"Path validation failed: {e}")
            return False
    
    def validate_file(self, path: Union[str, Path], extensions: Optional[List[str]] = None,
                     min_size: Optional[int] = None, max_size: Optional[int] = None) -> bool:
        """Validate a file path with optional constraints.
        
        Parameters
        ----------
        path : str or Path
            File path to validate
        extensions : list, optional
            Allowed file extensions
        min_size : int, optional
            Minimum file size in bytes
        max_size : int, optional
            Maximum file size in bytes
            
        Returns
        -------
        bool
            True if validation passes
        """
        if not self.validate(path):
            return False
        
        path_obj = Path(path)
        
        if not path_obj.is_file():
            self.add_error(f"Path is not a file: {path}")
            return False
        
        # Check file extension
        if extensions:
            file_ext = path_obj.suffix.lower()
            allowed_exts = [ext.lower() for ext in extensions]
            if file_ext not in allowed_exts:
                self.add_error(f"File extension must be one of {extensions}, got {path_obj.suffix}")
                return False
        
        # Check file size
        try:
            file_size = path_obj.stat().st_size
            
            if min_size is not None and file_size < min_size:
                self.add_error(f"File size {file_size} bytes is below minimum {min_size} bytes")
                return False
            
            if max_size is not None and file_size > max_size:
                self.add_error(f"File size {file_size} bytes exceeds maximum {max_size} bytes")
                return False
        except OSError as e:
            self.add_error(f"Cannot access file size: {e}")
            return False
        
        # Check permissions
        if self.check_permissions:
            if not path_obj.is_readable():
                self.add_error(f"File is not readable: {path}")
                return False
        
        return True
    
    def validate_directory(self, path: Union[str, Path], must_be_writable: bool = False,
                          must_be_empty: bool = False) -> bool:
        """Validate a directory path.
        
        Parameters
        ----------
        path : str or Path
            Directory path to validate
        must_be_writable : bool
            Whether directory must be writable
        must_be_empty : bool
            Whether directory must be empty
            
        Returns
        -------
        bool
            True if validation passes
        """
        if not self.validate(path):
            return False
        
        path_obj = Path(path)
        
        if not path_obj.is_dir():
            self.add_error(f"Path is not a directory: {path}")
            return False
        
        # Check if directory is empty
        if must_be_empty:
            try:
                if any(path_obj.iterdir()):
                    self.add_error(f"Directory is not empty: {path}")
                    return False
            except OSError as e:
                self.add_error(f"Cannot check directory contents: {e}")
                return False
        
        # Check write permissions
        if must_be_writable and self.check_permissions:
            try:
                # Test writeability by creating a temporary file
                test_file = path_obj / ".write_test_temp"
                test_file.touch()
                test_file.unlink()
            except Exception:
                self.add_error(f"Directory is not writable: {path}")
                return False
        
        return True
    
    def validate_path_pattern(self, path: Union[str, Path], pattern: str) -> bool:
        """Validate path against a pattern.
        
        Parameters
        ----------
        path : str or Path
            Path to validate
        pattern : str
            Regular expression pattern
            
        Returns
        -------
        bool
            True if path matches pattern
        """
        self.clear_messages()
        
        try:
            path_str = str(path)
            if not re.match(pattern, path_str):
                self.add_error(f"Path '{path_str}' does not match pattern '{pattern}'")
                return False
            return True
        except re.error as e:
            self.add_error(f"Invalid regex pattern '{pattern}': {e}")
            return False


class RangeValidator(Validator):
    """Validator for numeric ranges and constraints."""
    
    def __init__(self):
        super().__init__()
    
    def validate(self, value: Union[int, float, np.ndarray]) -> bool:
        """Validate a numeric value or array.
        
        Parameters
        ----------
        value : int, float, or np.ndarray
            Value to validate
            
        Returns
        -------
        bool
            True if validation passes
        """
        self.clear_messages()
        
        if isinstance(value, np.ndarray):
            return self._validate_array_basic(value)
        else:
            return self._validate_scalar_basic(value)
    
    def validate_range(self, value: Union[int, float, np.ndarray], 
                      min_val: Optional[Union[int, float]] = None,
                      max_val: Optional[Union[int, float]] = None, 
                      inclusive: bool = True) -> bool:
        """Validate that a value is within a specified range.
        
        Parameters
        ----------
        value : int, float, or np.ndarray
            Value to validate
        min_val : int, float, or None
            Minimum allowed value
        max_val : int, float, or None
            Maximum allowed value
        inclusive : bool
            Whether range bounds are inclusive
            
        Returns
        -------
        bool
            True if value is in range
        """
        self.clear_messages()
        
        if isinstance(value, np.ndarray):
            return self._validate_array_range(value, min_val, max_val, inclusive)
        else:
            return self._validate_scalar_range(value, min_val, max_val, inclusive)
    
    def validate_positive(self, value: Union[int, float, np.ndarray], 
                         strict: bool = True) -> bool:
        """Validate that a value is positive.
        
        Parameters
        ----------
        value : int, float, or np.ndarray
            Value to validate
        strict : bool
            If True, value must be > 0; if False, value must be >= 0
            
        Returns
        -------
        bool
            True if value is positive
        """
        min_val = 0.0 if not strict else 1e-10
        return self.validate_range(value, min_val=min_val, inclusive=not strict)
    
    def validate_finite(self, value: Union[int, float, np.ndarray]) -> bool:
        """Validate that a value is finite.
        
        Parameters
        ----------
        value : int, float, or np.ndarray
            Value to validate
            
        Returns
        -------
        bool
            True if value is finite
        """
        self.clear_messages()
        
        if isinstance(value, np.ndarray):
            if not np.all(np.isfinite(value)):
                n_invalid = np.sum(~np.isfinite(value))
                self.add_error(f"Array contains {n_invalid} non-finite values")
                return False
        else:
            if not np.isfinite(value):
                self.add_error(f"Value {value} is not finite")
                return False
        
        return True
    
    def _validate_scalar_basic(self, value: Union[int, float]) -> bool:
        """Basic validation for scalar values."""
        if not isinstance(value, (int, float, np.integer, np.floating)):
            self.add_error(f"Value must be numeric, got {type(value)}")
            return False
        return True
    
    def _validate_array_basic(self, value: np.ndarray) -> bool:
        """Basic validation for array values."""
        if not np.issubdtype(value.dtype, np.number):
            self.add_error(f"Array must be numeric, got dtype {value.dtype}")
            return False
        return True
    
    def _validate_scalar_range(self, value: Union[int, float], 
                              min_val: Optional[Union[int, float]],
                              max_val: Optional[Union[int, float]], 
                              inclusive: bool) -> bool:
        """Validate scalar range."""
        if not self._validate_scalar_basic(value):
            return False
        
        if min_val is not None:
            if (inclusive and value < min_val) or (not inclusive and value <= min_val):
                op = ">=" if inclusive else ">"
                self.add_error(f"Value {value} must be {op} {min_val}")
                return False
        
        if max_val is not None:
            if (inclusive and value > max_val) or (not inclusive and value >= max_val):
                op = "<=" if inclusive else "<"
                self.add_error(f"Value {value} must be {op} {max_val}")
                return False
        
        return True
    
    def _validate_array_range(self, value: np.ndarray,
                             min_val: Optional[Union[int, float]],
                             max_val: Optional[Union[int, float]], 
                             inclusive: bool) -> bool:
        """Validate array range."""
        if not self._validate_array_basic(value):
            return False
        
        if min_val is not None:
            if inclusive:
                invalid_mask = value < min_val
                op = ">="
            else:
                invalid_mask = value <= min_val
                op = ">"
            
            if np.any(invalid_mask):
                n_invalid = np.sum(invalid_mask)
                self.add_error(f"{n_invalid} array elements must be {op} {min_val}")
                return False
        
        if max_val is not None:
            if inclusive:
                invalid_mask = value > max_val
                op = "<="
            else:
                invalid_mask = value >= max_val
                op = "<"
            
            if np.any(invalid_mask):
                n_invalid = np.sum(invalid_mask)
                self.add_error(f"{n_invalid} array elements must be {op} {max_val}")
                return False
        
        return True


# Convenience validation functions
def validate_spherical_coordinates(coords: np.ndarray, strict: bool = True) -> bool:
    """Validate spherical coordinates array.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinates array with shape (..., 2) or (..., 3) for (theta, phi) or (r, theta, phi)
    strict : bool
        Whether to use strict validation
        
    Returns
    -------
    bool
        True if coordinates are valid
    """
    validator = DataValidator(strict=strict)
    
    if not validator._validate_array(coords):
        return False
    
    if coords.shape[-1] < 2:
        validator.add_error("Coordinates must have at least 2 columns")
        return False
    
    # Check theta range [0, π]
    theta = coords[..., -2]
    range_validator = RangeValidator()
    if not range_validator.validate_range(theta, 0, np.pi):
        validator.errors.extend(range_validator.get_errors())
        return False
    
    # Check phi range [0, 2π]
    phi = coords[..., -1]
    if not range_validator.validate_range(phi, 0, 2*np.pi):
        validator.errors.extend(range_validator.get_errors())
        return False
    
    return True


def validate_patch_size(patch_size: float) -> bool:
    """Validate patch size in degrees.
    
    Parameters
    ----------
    patch_size : float
        Patch size in degrees
        
    Returns
    -------
    bool
        True if patch size is valid
    """
    validator = RangeValidator()
    return validator.validate_range(patch_size, 0.1, 90.0)


def validate_nside(nside: int) -> bool:
    """Validate HEALPix NSIDE parameter.
    
    Parameters
    ----------
    nside : int
        HEALPix NSIDE parameter
        
    Returns
    -------
    bool
        True if NSIDE is valid
    """
    validator = RangeValidator()
    
    # NSIDE must be positive
    if not validator.validate_positive(nside):
        return False
    
    # NSIDE must be a power of 2
    if nside > 0 and (nside & (nside - 1)) != 0:
        validator.add_error(f"NSIDE must be a power of 2, got {nside}")
        return False
    
    return True


def validate_redshift(z: Union[float, np.ndarray]) -> bool:
    """Validate redshift values.
    
    Parameters
    ----------
    z : float or np.ndarray
        Redshift value(s)
        
    Returns
    -------
    bool
        True if redshift is valid
    """
    validator = RangeValidator()
    return validator.validate_range(z, min_val=0.0, max_val=1000.0)  # Reasonable cosmological range


def validate_angular_scale(scale_arcmin: Union[float, np.ndarray]) -> bool:
    """Validate angular scale in arcminutes.
    
    Parameters
    ----------
    scale_arcmin : float or np.ndarray
        Angular scale in arcminutes
        
    Returns
    -------
    bool
        True if scale is valid
    """
    validator = RangeValidator()
    return validator.validate_range(scale_arcmin, min_val=0.01, max_val=3600.0)  # 0.01 arcmin to 1 degree