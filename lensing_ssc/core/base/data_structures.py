"""
Core data structures for LensingSSC with minimal dependencies.

This module defines the fundamental data structures used throughout the package,
independent of heavy external libraries like healpy or lenstools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from .exceptions import DataError, ValidationError


class DataStructure(ABC):
    """Abstract base class for all data structures."""
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the data structure."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataStructure":
        """Create from dictionary representation."""
        pass


@dataclass
class MapData(DataStructure):
    """Container for map data with metadata."""
    
    data: np.ndarray
    shape: Tuple[int, ...]
    dtype: np.dtype
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not isinstance(self.data, np.ndarray):
            raise DataError("Data must be a numpy array")
        
        if self.data.shape != self.shape:
            raise DataError(f"Data shape {self.data.shape} does not match specified shape {self.shape}")
        
        if self.data.dtype != self.dtype:
            self.data = self.data.astype(self.dtype)
    
    def validate(self) -> bool:
        """Validate the map data."""
        try:
            # Check for NaN or infinite values
            if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
                raise ValidationError("Map data contains NaN or infinite values")
            
            # Check data range (basic sanity check)
            if np.abs(self.data).max() > 1e10:
                raise ValidationError("Map data values are suspiciously large")
            
            return True
        except Exception as e:
            raise ValidationError(f"Map validation failed: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "data": self.data.tolist(),
            "shape": self.shape,
            "dtype": str(self.dtype),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MapData":
        """Create from dictionary representation."""
        return cls(
            data=np.array(data["data"], dtype=data["dtype"]),
            shape=tuple(data["shape"]),
            dtype=np.dtype(data["dtype"]),
            metadata=data.get("metadata", {}),
        )
    
    @property
    def size(self) -> int:
        """Get the total number of elements."""
        return self.data.size
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return self.data.ndim
    
    def copy(self) -> "MapData":
        """Create a deep copy of the map data."""
        return MapData(
            data=self.data.copy(),
            shape=self.shape,
            dtype=self.dtype,
            metadata=self.metadata.copy(),
        )


@dataclass
class PatchData(DataStructure):
    """Container for patch data extracted from maps."""
    
    patches: np.ndarray  # Shape: (n_patches, patch_height, patch_width)
    centers: np.ndarray  # Shape: (n_patches, 2) - (theta, phi) coordinates
    patch_size_deg: float
    xsize: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.patches.ndim != 3:
            raise DataError("Patches must be a 3D array (n_patches, height, width)")
        
        if self.centers.shape != (self.patches.shape[0], 2):
            raise DataError("Centers shape must match number of patches")
        
        if self.patches.shape[1] != self.xsize or self.patches.shape[2] != self.xsize:
            raise DataError(f"Patch dimensions must be {self.xsize}x{self.xsize}")
    
    def validate(self) -> bool:
        """Validate the patch data."""
        try:
            # Check for valid patch centers (spherical coordinates)
            theta, phi = self.centers[:, 0], self.centers[:, 1]
            if np.any(theta < 0) or np.any(theta > np.pi):
                raise ValidationError("Invalid theta coordinates (must be in [0, π])")
            
            if np.any(phi < 0) or np.any(phi > 2 * np.pi):
                raise ValidationError("Invalid phi coordinates (must be in [0, 2π])")
            
            # Check patch data
            if np.any(np.isnan(self.patches)) or np.any(np.isinf(self.patches)):
                raise ValidationError("Patch data contains NaN or infinite values")
            
            return True
        except Exception as e:
            raise ValidationError(f"Patch validation failed: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "patches": self.patches.tolist(),
            "centers": self.centers.tolist(),
            "patch_size_deg": self.patch_size_deg,
            "xsize": self.xsize,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatchData":
        """Create from dictionary representation."""
        return cls(
            patches=np.array(data["patches"]),
            centers=np.array(data["centers"]),
            patch_size_deg=data["patch_size_deg"],
            xsize=data["xsize"],
            metadata=data.get("metadata", {}),
        )
    
    @property
    def n_patches(self) -> int:
        """Get the number of patches."""
        return self.patches.shape[0]
    
    def get_patch(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a specific patch and its center."""
        if index >= self.n_patches:
            raise IndexError(f"Patch index {index} out of range")
        return self.patches[index], self.centers[index]


@dataclass
class StatisticsData(DataStructure):
    """Container for statistical analysis results."""
    
    statistics: Dict[str, np.ndarray]
    bins: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate the statistics data."""
        try:
            # Check that all statistics have finite values
            for name, values in self.statistics.items():
                if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                    raise ValidationError(f"Statistic '{name}' contains NaN or infinite values")
            
            # Check that bins are monotonically increasing
            for name, bin_values in self.bins.items():
                if len(bin_values) > 1 and not np.all(np.diff(bin_values) > 0):
                    raise ValidationError(f"Bins '{name}' are not monotonically increasing")
            
            return True
        except Exception as e:
            raise ValidationError(f"Statistics validation failed: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "statistics": {k: v.tolist() for k, v in self.statistics.items()},
            "bins": {k: v.tolist() for k, v in self.bins.items()},
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatisticsData":
        """Create from dictionary representation."""
        return cls(
            statistics={k: np.array(v) for k, v in data["statistics"].items()},
            bins={k: np.array(v) for k, v in data["bins"].items()},
            metadata=data.get("metadata", {}),
        )
    
    def get_statistic(self, name: str) -> np.ndarray:
        """Get a specific statistic."""
        if name not in self.statistics:
            raise KeyError(f"Statistic '{name}' not found")
        return self.statistics[name]
    
    def get_bins(self, name: str) -> np.ndarray:
        """Get bins for a specific statistic."""
        if name not in self.bins:
            raise KeyError(f"Bins '{name}' not found")
        return self.bins[name]
    
    def add_statistic(self, name: str, values: np.ndarray, bins: Optional[np.ndarray] = None):
        """Add a new statistic."""
        self.statistics[name] = np.asarray(values)
        if bins is not None:
            self.bins[name] = np.asarray(bins)