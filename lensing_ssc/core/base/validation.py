"""
Validation utilities for data and configuration.

This module provides validation classes and functions to ensure data integrity
and configuration validity throughout the package.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import numpy as np
from pathlib import Path

from .exceptions import ValidationError, ConfigurationError
from .data_structures import DataStructure, MapData, PatchData, StatisticsData


class Validator(ABC):
    """Abstract base class for validators."""
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate the given data."""
        pass
    
    @abstractmethod
    def get_errors(self) -> List[str]:
        """Get validation error messages."""
        pass


class DataValidator(Validator):
    """Validator for data structures and arrays."""
    
    def __init__(self):
        self.errors: List[str] = []
    
    def validate(self, data: Any) -> bool:
        """Validate data structure or array."""
        self.errors.clear()
        
        try:
            if isinstance(data, DataStructure):
                return self._validate_data_structure(data)
            elif isinstance(data, np.ndarray):
                return self._validate_array(data)
            elif isinstance(data, (list, tuple)):
                return self._validate_sequence(data)
            else:
                self.errors.append(f"Unsupported data type: {type(data)}")
                return False
        except Exception as e:
            self.errors.append(f"Validation failed with exception: {e}")
            return False
    
    def get_errors(self) -> List[str]:
        """Get validation error messages."""
        return self.errors.copy()
    
    def _validate_data_structure(self, data: DataStructure) -> bool:
        """Validate a DataStructure object."""
        try:
            return data.validate()
        except ValidationError as e:
            self.errors.append(str(e))
            return False
    
    def _validate_array(self, data: np.ndarray) -> bool:
        """Validate a numpy array."""
        # Check for NaN/inf values
        if np.any(np.isnan(data)):
            self.errors.append("Array contains NaN values")
            return False
        
        if np.any(np.isinf(data)):
            self.errors.append("Array contains infinite values")
            return False
        
        # Check for empty arrays
        if data.size == 0:
            self.errors.append("Array is empty")
            return False
        
        return True
    
    def _validate_sequence(self, data: Union[List, Tuple]) -> bool:
        """Validate a sequence (list or tuple)."""
        if len(data) == 0:
            self.errors.append("Sequence is empty")
            return False
        
        # If all elements are arrays, validate each
        if all(isinstance(item, np.ndarray) for item in data):
            for i, item in enumerate(data):
                if not self._validate_array(item):
                    self.errors.append(f"Array at index {i} failed validation")
                    return False
        
        return True


class ConfigValidator(Validator):
    """Validator for configuration objects and dictionaries."""
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.errors: List[str] = []
        self.schema = schema or {}
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary."""
        self.errors.clear()
        
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
            
            return True
        except Exception as e:
            self.errors.append(f"Configuration validation failed: {e}")
            return False
    
    def get_errors(self) -> List[str]:
        """Get validation error messages."""
        return self.errors.copy()
    
    def _check_required_fields(self, config: Dict[str, Any]) -> bool:
        """Check that all required fields are present."""
        required_fields = self.schema.get('required', [])
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            self.errors.append(f"Missing required fields: {missing_fields}")
            return False
        
        return True
    
    def _validate_field_types(self, config: Dict[str, Any]) -> bool:
        """Validate field types against schema."""
        field_types = self.schema.get('types', {})
        
        for field, expected_type in field_types.items():
            if field in config:
                if not isinstance(config[field], expected_type):
                    self.errors.append(
                        f"Field '{field}' has type {type(config[field]).__name__}, "
                        f"expected {expected_type.__name__}"
                    )
                    return False
        
        return True
    
    def _validate_field_values(self, config: Dict[str, Any]) -> bool:
        """Validate field values against constraints."""
        validators = self.schema.get('validators', {})
        
        for field, validator_func in validators.items():
            if field in config:
                try:
                    if not validator_func(config[field]):
                        self.errors.append(f"Field '{field}' failed validation")
                        return False
                except Exception as e:
                    self.errors.append(f"Validation of field '{field}' raised exception: {e}")
                    return False
        
        return True


class PathValidator(Validator):
    """Validator for file and directory paths."""
    
    def __init__(self):
        self.errors: List[str] = []
    
    def validate(self, path: Union[str, Path]) -> bool:
        """Validate a file or directory path."""
        self.errors.clear()
        
        try:
            path_obj = Path(path)
            
            # Check if path exists
            if not path_obj.exists():
                self.errors.append(f"Path does not exist: {path}")
                return False
            
            return True
        except Exception as e:
            self.errors.append(f"Path validation failed: {e}")
            return False
    
    def get_errors(self) -> List[str]:
        """Get validation error messages."""
        return self.errors.copy()
    
    def validate_file(self, path: Union[str, Path], extensions: Optional[List[str]] = None) -> bool:
        """Validate a file path with optional extension checking."""
        if not self.validate(path):
            return False
        
        path_obj = Path(path)
        
        if not path_obj.is_file():
            self.errors.append(f"Path is not a file: {path}")
            return False
        
        if extensions:
            if path_obj.suffix.lower() not in [ext.lower() for ext in extensions]:
                self.errors.append(f"File extension must be one of {extensions}, got {path_obj.suffix}")
                return False
        
        return True
    
    def validate_directory(self, path: Union[str, Path], must_be_writable: bool = False) -> bool:
        """Validate a directory path."""
        if not self.validate(path):
            return False
        
        path_obj = Path(path)
        
        if not path_obj.is_dir():
            self.errors.append(f"Path is not a directory: {path}")
            return False
        
        if must_be_writable:
            try:
                # Test writeability by creating a temporary file
                test_file = path_obj / ".write_test"
                test_file.touch()
                test_file.unlink()
            except Exception:
                self.errors.append(f"Directory is not writable: {path}")
                return False
        
        return True


class RangeValidator:
    """Utility class for validating numeric ranges."""
    
    @staticmethod
    def validate_range(value: Union[int, float], min_val: Optional[Union[int, float]] = None,
                      max_val: Optional[Union[int, float]] = None, 
                      inclusive: bool = True) -> bool:
        """Validate that a value is within a specified range."""
        if min_val is not None:
            if inclusive and value < min_val:
                return False
            elif not inclusive and value <= min_val:
                return False
        
        if max_val is not None:
            if inclusive and value > max_val:
                return False
            elif not inclusive and value >= max_val:
                return False
        
        return True
    
    @staticmethod
    def validate_array_range(array: np.ndarray, min_val: Optional[Union[int, float]] = None,
                           max_val: Optional[Union[int, float]] = None,
                           inclusive: bool = True) -> bool:
        """Validate that all values in an array are within a specified range."""
        if min_val is not None:
            if inclusive and np.any(array < min_val):
                return False
            elif not inclusive and np.any(array <= min_val):
                return False
        
        if max_val is not None:
            if inclusive and np.any(array > max_val):
                return False
            elif not inclusive and np.any(array >= max_val):
                return False
        
        return True


# Convenience validation functions
def validate_spherical_coordinates(coords: np.ndarray) -> bool:
    """Validate spherical coordinates array."""
    validator = DataValidator()
    
    if not validator._validate_array(coords):
        return False
    
    if coords.shape[-1] < 2:
        return False
    
    # Check theta range [0, π]
    theta = coords[..., -2]
    if not RangeValidator.validate_array_range(theta, 0, np.pi):
        return False
    
    # Check phi range [0, 2π]
    phi = coords[..., -1]
    if not RangeValidator.validate_array_range(phi, 0, 2*np.pi):
        return False
    
    return True


def validate_patch_size(patch_size: float) -> bool:
    """Validate patch size in degrees."""
    return RangeValidator.validate_range(patch_size, 0.1, 90.0)


def validate_nside(nside: int) -> bool:
    """Validate HEALPix NSIDE parameter."""
    # NSIDE must be a power of 2
    return nside > 0 and (nside & (nside - 1)) == 0