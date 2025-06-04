"""
Core data structures for LensingSSC with minimal dependencies.

This module defines the fundamental data structures used throughout the package,
independent of heavy external libraries like healpy or lenstools. These structures
provide a consistent interface for data handling and validation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import numpy as np
from pathlib import Path
import copy

from .exceptions import DataError, ValidationError


class DataStructure(ABC):
    """Abstract base class for all data structures.
    
    This class defines the interface that all LensingSSC data structures
    must implement, ensuring consistent behavior across the package.
    """
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the data structure.
        
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns
        -------
        dict
            Dictionary representation of the data structure
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataStructure":
        """Create from dictionary representation.
        
        Parameters
        ----------
        data : dict
            Dictionary containing data structure information
            
        Returns
        -------
        DataStructure
            New instance created from dictionary
        """
        pass
    
    def copy(self) -> "DataStructure":
        """Create a deep copy of the data structure.
        
        Returns
        -------
        DataStructure
            Deep copy of the data structure
        """
        return copy.deepcopy(self)
    
    def __repr__(self) -> str:
        """String representation of the data structure."""
        return f"{self.__class__.__name__}({self._repr_info()})"
    
    def _repr_info(self) -> str:
        """Information for string representation."""
        return ""


@dataclass
class MapData(DataStructure):
    """Container for map data with metadata.
    
    This class provides a standardized container for map data, supporting
    various map formats and coordinate systems while maintaining metadata.
    """
    
    data: np.ndarray
    shape: Tuple[int, ...]
    dtype: np.dtype
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        # Ensure data is numpy array
        if not isinstance(self.data, np.ndarray):
            try:
                self.data = np.asarray(self.data)
            except Exception as e:
                raise DataError("Data must be convertible to numpy array") from e
        
        # Validate shape consistency
        if self.data.shape != self.shape:
            raise DataError(f"Data shape {self.data.shape} does not match specified shape {self.shape}")
        
        # Ensure dtype consistency
        if self.data.dtype != self.dtype:
            self.data = self.data.astype(self.dtype)
        
        # Initialize metadata if needed
        if not isinstance(self.metadata, dict):
            self.metadata = {}
    
    def validate(self) -> bool:
        """Validate the map data.
        
        Returns
        -------
        bool
            True if validation passes
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        try:
            # Check for finite values
            if not np.all(np.isfinite(self.data)):
                finite_mask = np.isfinite(self.data)
                n_invalid = np.sum(~finite_mask)
                raise ValidationError(f"Map data contains {n_invalid} non-finite values")
            
            # Check data range (basic sanity check)
            data_range = np.ptp(self.data)  # peak-to-peak
            if data_range == 0:
                raise ValidationError("Map data has zero range (all values identical)")
            
            # Check for reasonable data magnitude
            max_abs = np.abs(self.data).max()
            if max_abs > 1e10:
                raise ValidationError(f"Map data values are suspiciously large (max: {max_abs})")
            
            return True
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Map validation failed: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns
        -------
        dict
            Dictionary representation
        """
        return {
            "data": self.data.tolist(),
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "metadata": self.metadata.copy(),
            "class": self.__class__.__name__
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MapData":
        """Create from dictionary representation.
        
        Parameters
        ----------
        data : dict
            Dictionary containing map data
            
        Returns
        -------
        MapData
            New MapData instance
        """
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
    
    @property
    def nbytes(self) -> int:
        """Get the number of bytes used by the data."""
        return self.data.nbytes
    
    def get_statistics(self) -> Dict[str, float]:
        """Get basic statistics of the map data.
        
        Returns
        -------
        dict
            Dictionary with basic statistics
        """
        finite_data = self.data[np.isfinite(self.data)]
        
        if len(finite_data) == 0:
            return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
        
        return {
            "mean": float(np.mean(finite_data)),
            "std": float(np.std(finite_data)),
            "min": float(np.min(finite_data)),
            "max": float(np.max(finite_data)),
            "median": float(np.median(finite_data)),
            "n_finite": len(finite_data),
            "n_total": self.data.size
        }
    
    def apply_mask(self, mask: np.ndarray) -> "MapData":
        """Apply a mask to the map data.
        
        Parameters
        ----------
        mask : np.ndarray
            Boolean mask array
            
        Returns
        -------
        MapData
            New MapData with masked values set to NaN
        """
        if mask.shape != self.data.shape:
            raise DataError(f"Mask shape {mask.shape} doesn't match data shape {self.data.shape}")
        
        masked_data = self.data.copy()
        masked_data[~mask] = np.nan
        
        new_metadata = self.metadata.copy()
        new_metadata["masked"] = True
        new_metadata["n_masked"] = np.sum(~mask)
        
        return MapData(
            data=masked_data,
            shape=self.shape,
            dtype=self.dtype,
            metadata=new_metadata
        )
    
    def _repr_info(self) -> str:
        """Information for string representation."""
        return f"shape={self.shape}, dtype={self.dtype}"


@dataclass
class PatchData(DataStructure):
    """Container for patch data extracted from maps.
    
    This class stores multiple patches extracted from larger maps,
    along with their center coordinates and extraction parameters.
    """
    
    patches: np.ndarray  # Shape: (n_patches, patch_height, patch_width)
    centers: np.ndarray  # Shape: (n_patches, 2) - (theta, phi) coordinates
    patch_size_deg: float
    xsize: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        # Ensure patches is 3D array
        self.patches = np.asarray(self.patches)
        if self.patches.ndim != 3:
            raise DataError("Patches must be a 3D array (n_patches, height, width)")
        
        # Ensure centers is 2D array
        self.centers = np.asarray(self.centers)
        if self.centers.shape != (self.patches.shape[0], 2):
            raise DataError("Centers shape must be (n_patches, 2)")
        
        # Validate patch dimensions
        if self.patches.shape[1] != self.xsize or self.patches.shape[2] != self.xsize:
            raise DataError(f"Patch dimensions must be {self.xsize}x{self.xsize}")
        
        # Validate patch size
        if self.patch_size_deg <= 0:
            raise DataError("Patch size must be positive")
        
        # Initialize metadata
        if not isinstance(self.metadata, dict):
            self.metadata = {}
    
    def validate(self) -> bool:
        """Validate the patch data.
        
        Returns
        -------
        bool
            True if validation passes
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        try:
            # Check for valid patch centers (spherical coordinates)
            theta, phi = self.centers[:, 0], self.centers[:, 1]
            
            if np.any(theta < 0) or np.any(theta > np.pi):
                invalid_theta = np.sum((theta < 0) | (theta > np.pi))
                raise ValidationError(f"Invalid theta coordinates: {invalid_theta} points outside [0, π]")
            
            if np.any(phi < 0) or np.any(phi > 2 * np.pi):
                invalid_phi = np.sum((phi < 0) | (phi > 2 * np.pi))
                raise ValidationError(f"Invalid phi coordinates: {invalid_phi} points outside [0, 2π]")
            
            # Check patch data for finite values
            if not np.all(np.isfinite(self.patches)):
                n_invalid = np.sum(~np.isfinite(self.patches))
                raise ValidationError(f"Patch data contains {n_invalid} non-finite values")
            
            return True
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Patch validation failed: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns
        -------
        dict
            Dictionary representation
        """
        return {
            "patches": self.patches.tolist(),
            "centers": self.centers.tolist(),
            "patch_size_deg": self.patch_size_deg,
            "xsize": self.xsize,
            "metadata": self.metadata.copy(),
            "class": self.__class__.__name__
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatchData":
        """Create from dictionary representation.
        
        Parameters
        ----------
        data : dict
            Dictionary containing patch data
            
        Returns
        -------
        PatchData
            New PatchData instance
        """
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
    
    @property
    def patch_shape(self) -> Tuple[int, int]:
        """Get the shape of individual patches."""
        return (self.patches.shape[1], self.patches.shape[2])
    
    def get_patch(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a specific patch and its center.
        
        Parameters
        ----------
        index : int
            Patch index
            
        Returns
        -------
        tuple
            (patch_data, center_coordinates)
        """
        if index >= self.n_patches:
            raise IndexError(f"Patch index {index} out of range (0-{self.n_patches-1})")
        return self.patches[index], self.centers[index]
    
    def get_patches_subset(self, indices: Union[List[int], np.ndarray]) -> "PatchData":
        """Get a subset of patches.
        
        Parameters
        ----------
        indices : list or np.ndarray
            Patch indices to extract
            
        Returns
        -------
        PatchData
            New PatchData with subset of patches
        """
        indices = np.asarray(indices)
        if np.any(indices >= self.n_patches) or np.any(indices < 0):
            raise IndexError("Some indices are out of range")
        
        subset_metadata = self.metadata.copy()
        subset_metadata["subset_of"] = self.n_patches
        subset_metadata["subset_indices"] = indices.tolist()
        
        return PatchData(
            patches=self.patches[indices],
            centers=self.centers[indices],
            patch_size_deg=self.patch_size_deg,
            xsize=self.xsize,
            metadata=subset_metadata
        )
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute statistics for all patches.
        
        Returns
        -------
        dict
            Dictionary with patch statistics
        """
        # Statistics per patch
        patch_means = np.mean(self.patches, axis=(1, 2))
        patch_stds = np.std(self.patches, axis=(1, 2))
        patch_mins = np.min(self.patches, axis=(1, 2))
        patch_maxs = np.max(self.patches, axis=(1, 2))
        
        # Overall statistics
        all_data = self.patches.flatten()
        finite_data = all_data[np.isfinite(all_data)]
        
        return {
            "n_patches": self.n_patches,
            "patch_shape": self.patch_shape,
            "patch_size_deg": self.patch_size_deg,
            "per_patch": {
                "means": patch_means,
                "stds": patch_stds,
                "mins": patch_mins,
                "maxs": patch_maxs
            },
            "overall": {
                "mean": float(np.mean(finite_data)) if len(finite_data) > 0 else np.nan,
                "std": float(np.std(finite_data)) if len(finite_data) > 0 else np.nan,
                "min": float(np.min(finite_data)) if len(finite_data) > 0 else np.nan,
                "max": float(np.max(finite_data)) if len(finite_data) > 0 else np.nan,
                "n_finite": len(finite_data),
                "n_total": len(all_data)
            }
        }
    
    def _repr_info(self) -> str:
        """Information for string representation."""
        return f"n_patches={self.n_patches}, patch_size={self.patch_size_deg}°, xsize={self.xsize}"


@dataclass
class StatisticsData(DataStructure):
    """Container for statistical analysis results.
    
    This class stores results from statistical analyses such as power spectra,
    bispectra, PDFs, and other statistical measures.
    """
    
    statistics: Dict[str, np.ndarray]
    bins: Dict[str, np.ndarray]
    errors: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        # Ensure dictionaries
        if not isinstance(self.statistics, dict):
            raise DataError("Statistics must be a dictionary")
        if not isinstance(self.bins, dict):
            raise DataError("Bins must be a dictionary")
        if not isinstance(self.errors, dict):
            self.errors = {}
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        # Convert values to numpy arrays
        for key, value in self.statistics.items():
            self.statistics[key] = np.asarray(value)
        
        for key, value in self.bins.items():
            self.bins[key] = np.asarray(value)
        
        for key, value in self.errors.items():
            self.errors[key] = np.asarray(value)
    
    def validate(self) -> bool:
        """Validate the statistics data.
        
        Returns
        -------
        bool
            True if validation passes
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        try:
            # Check that all statistics have finite values
            for name, values in self.statistics.items():
                if not np.all(np.isfinite(values)):
                    n_invalid = np.sum(~np.isfinite(values))
                    raise ValidationError(f"Statistic '{name}' contains {n_invalid} non-finite values")
            
            # Check that bins are monotonically increasing
            for name, bin_values in self.bins.items():
                if len(bin_values) > 1:
                    diffs = np.diff(bin_values)
                    if not np.all(diffs > 0):
                        raise ValidationError(f"Bins '{name}' are not monotonically increasing")
            
            # Check consistency between statistics and errors
            for name, error_values in self.errors.items():
                if name not in self.statistics:
                    raise ValidationError(f"Error array '{name}' has no corresponding statistic")
                
                if error_values.shape != self.statistics[name].shape:
                    raise ValidationError(f"Error array '{name}' shape mismatch with statistic")
                
                if not np.all(error_values >= 0):
                    raise ValidationError(f"Error array '{name}' contains negative values")
            
            return True
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Statistics validation failed: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns
        -------
        dict
            Dictionary representation
        """
        return {
            "statistics": {k: v.tolist() for k, v in self.statistics.items()},
            "bins": {k: v.tolist() for k, v in self.bins.items()},
            "errors": {k: v.tolist() for k, v in self.errors.items()},
            "metadata": self.metadata.copy(),
            "class": self.__class__.__name__
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatisticsData":
        """Create from dictionary representation.
        
        Parameters
        ----------
        data : dict
            Dictionary containing statistics data
            
        Returns
        -------
        StatisticsData
            New StatisticsData instance
        """
        return cls(
            statistics={k: np.array(v) for k, v in data["statistics"].items()},
            bins={k: np.array(v) for k, v in data["bins"].items()},
            errors={k: np.array(v) for k, v in data.get("errors", {}).items()},
            metadata=data.get("metadata", {}),
        )
    
    def get_statistic(self, name: str) -> np.ndarray:
        """Get a specific statistic.
        
        Parameters
        ----------
        name : str
            Name of the statistic
            
        Returns
        -------
        np.ndarray
            Statistic values
        """
        if name not in self.statistics:
            raise KeyError(f"Statistic '{name}' not found. Available: {list(self.statistics.keys())}")
        return self.statistics[name]
    
    def get_bins(self, name: str) -> np.ndarray:
        """Get bins for a specific statistic.
        
        Parameters
        ----------
        name : str
            Name of the bins
            
        Returns
        -------
        np.ndarray
            Bin values
        """
        if name not in self.bins:
            raise KeyError(f"Bins '{name}' not found. Available: {list(self.bins.keys())}")
        return self.bins[name]
    
    def get_errors(self, name: str) -> Optional[np.ndarray]:
        """Get errors for a specific statistic.
        
        Parameters
        ----------
        name : str
            Name of the statistic
            
        Returns
        -------
        np.ndarray or None
            Error values if available, None otherwise
        """
        return self.errors.get(name)
    
    def add_statistic(self, name: str, values: np.ndarray, 
                     bins: Optional[np.ndarray] = None, 
                     errors: Optional[np.ndarray] = None) -> None:
        """Add a new statistic.
        
        Parameters
        ----------
        name : str
            Name of the statistic
        values : np.ndarray
            Statistic values
        bins : np.ndarray, optional
            Bin values
        errors : np.ndarray, optional
            Error values
        """
        self.statistics[name] = np.asarray(values)
        
        if bins is not None:
            self.bins[name] = np.asarray(bins)
        
        if errors is not None:
            self.errors[name] = np.asarray(errors)
    
    def remove_statistic(self, name: str) -> None:
        """Remove a statistic and its associated data.
        
        Parameters
        ----------
        name : str
            Name of the statistic to remove
        """
        self.statistics.pop(name, None)
        self.bins.pop(name, None)
        self.errors.pop(name, None)
    
    def list_statistics(self) -> List[str]:
        """List all available statistics.
        
        Returns
        -------
        list
            List of statistic names
        """
        return list(self.statistics.keys())
    
    def _repr_info(self) -> str:
        """Information for string representation."""
        n_stats = len(self.statistics)
        stat_names = ", ".join(list(self.statistics.keys())[:3])
        if n_stats > 3:
            stat_names += f", ... ({n_stats} total)"
        return f"statistics=[{stat_names}]"


# Utility functions for data structure operations
def combine_map_data(maps: List[MapData], operation: str = "mean") -> MapData:
    """Combine multiple MapData objects.
    
    Parameters
    ----------
    maps : list
        List of MapData objects to combine
    operation : str
        Combination operation ('mean', 'sum', 'median', 'std')
        
    Returns
    -------
    MapData
        Combined map data
    """
    if not maps:
        raise DataError("Cannot combine empty list of maps")
    
    # Check compatibility
    reference = maps[0]
    for i, map_data in enumerate(maps[1:], 1):
        if map_data.shape != reference.shape:
            raise DataError(f"Map {i} shape {map_data.shape} doesn't match reference {reference.shape}")
        if map_data.dtype != reference.dtype:
            raise DataError(f"Map {i} dtype {map_data.dtype} doesn't match reference {reference.dtype}")
    
    # Stack data
    stacked_data = np.stack([m.data for m in maps], axis=0)
    
    # Apply operation
    if operation == "mean":
        combined_data = np.mean(stacked_data, axis=0)
    elif operation == "sum":
        combined_data = np.sum(stacked_data, axis=0)
    elif operation == "median":
        combined_data = np.median(stacked_data, axis=0)
    elif operation == "std":
        combined_data = np.std(stacked_data, axis=0)
    else:
        raise DataError(f"Unknown operation: {operation}")
    
    # Combine metadata
    combined_metadata = reference.metadata.copy()
    combined_metadata["combined_from"] = len(maps)
    combined_metadata["combination_operation"] = operation
    
    return MapData(
        data=combined_data,
        shape=reference.shape,
        dtype=reference.dtype,
        metadata=combined_metadata
    )