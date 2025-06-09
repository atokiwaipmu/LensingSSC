"""
Abstract interfaces for storage and I/O providers.

This module defines interfaces for file I/O operations, allowing for different
storage backends and file format implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO
import numpy as np
from pathlib import Path

from ..base.data_structures import DataStructure, MapData, PatchData, StatisticsData
from .data_interface import DataProvider


class StorageProvider(DataProvider):
    """Abstract base class for storage providers.
    
    This class defines interfaces for file system operations and storage management,
    supporting both local and remote storage backends.
    """
    
    @abstractmethod
    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a file or directory exists.
        
        Parameters
        ----------
        path : str or Path
            Path to check
            
        Returns
        -------
        bool
            True if path exists
        """
        pass
    
    @abstractmethod
    def mkdir(self, path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> None:
        """Create directory.
        
        Parameters
        ----------
        path : str or Path
            Directory path to create
        parents : bool
            Create parent directories if needed
        exist_ok : bool
            Do not raise error if directory exists
            
        Raises
        ------
        OSError
            If directory creation fails
        """
        pass
    
    @abstractmethod
    def remove(self, path: Union[str, Path]) -> None:
        """Remove file or directory.
        
        Parameters
        ----------
        path : str or Path
            Path to remove
            
        Raises
        ------
        OSError
            If removal fails
        FileNotFoundError
            If path does not exist
        """
        pass
    
    @abstractmethod
    def list_files(self, path: Union[str, Path], pattern: Optional[str] = None) -> List[Path]:
        """List files in directory.
        
        Parameters
        ----------
        path : str or Path
            Directory path
        pattern : str, optional
            Glob pattern to filter files
            
        Returns
        -------
        List[Path]
            List of file paths
            
        Raises
        ------
        OSError
            If directory cannot be accessed
        """
        pass
    
    @abstractmethod
    def get_file_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get file information.
        
        Parameters
        ----------
        path : str or Path
            File path
            
        Returns
        -------
        Dict[str, Any]
            File information including size, modification time, permissions
        """
        pass
    
    @abstractmethod
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> None:
        """Copy file from source to destination.
        
        Parameters
        ----------
        src : str or Path
            Source file path
        dst : str or Path
            Destination file path
            
        Raises
        ------
        OSError
            If copy operation fails
        """
        pass


class FileFormatProvider(DataProvider):
    """Abstract interface for file format providers."""
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions."""
        pass
    
    @abstractmethod
    def can_read(self, path: Union[str, Path]) -> bool:
        """Check if the provider can read this file."""
        pass
    
    @abstractmethod
    def can_write(self, path: Union[str, Path]) -> bool:
        """Check if the provider can write to this file."""
        pass
    
    @abstractmethod
    def read(self, path: Union[str, Path], **kwargs) -> Any:
        """Read data from file."""
        pass
    
    @abstractmethod
    def write(self, data: Any, path: Union[str, Path], **kwargs) -> None:
        """Write data to file."""
        pass
    
    @abstractmethod
    def get_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get file metadata."""
        pass


class FITSProvider(FileFormatProvider):
    """Abstract interface for FITS file providers."""
    
    @abstractmethod
    def read_fits_map(self, path: Union[str, Path], hdu: int = 0, **kwargs) -> MapData:
        """Read map from FITS file."""
        pass
    
    @abstractmethod
    def write_fits_map(self, map_data: MapData, path: Union[str, Path], 
                      overwrite: bool = False, **kwargs) -> None:
        """Write map to FITS file."""
        pass
    
    @abstractmethod
    def read_fits_table(self, path: Union[str, Path], hdu: int = 1, **kwargs) -> Dict[str, np.ndarray]:
        """Read table from FITS file."""
        pass
    
    @abstractmethod
    def write_fits_table(self, data: Dict[str, np.ndarray], path: Union[str, Path],
                        overwrite: bool = False, **kwargs) -> None:
        """Write table to FITS file."""
        pass


class HDF5Provider(FileFormatProvider):
    """Abstract interface for HDF5 file providers."""
    
    @abstractmethod
    def read_hdf5_dataset(self, path: Union[str, Path], dataset: str, **kwargs) -> np.ndarray:
        """Read dataset from HDF5 file."""
        pass
    
    @abstractmethod
    def write_hdf5_dataset(self, data: np.ndarray, path: Union[str, Path], 
                          dataset: str, overwrite: bool = False, **kwargs) -> None:
        """Write dataset to HDF5 file."""
        pass
    
    @abstractmethod
    def read_hdf5_group(self, path: Union[str, Path], group: str, **kwargs) -> Dict[str, Any]:
        """Read group from HDF5 file."""
        pass
    
    @abstractmethod
    def write_hdf5_group(self, data: Dict[str, Any], path: Union[str, Path],
                        group: str, overwrite: bool = False, **kwargs) -> None:
        """Write group to HDF5 file."""
        pass
    
    @abstractmethod
    def list_datasets(self, path: Union[str, Path], group: Optional[str] = None) -> List[str]:
        """List datasets in HDF5 file."""
        pass
    
    @abstractmethod
    def get_attributes(self, path: Union[str, Path], item: Optional[str] = None) -> Dict[str, Any]:
        """Get attributes from HDF5 file or dataset."""
        pass


class CSVProvider(FileFormatProvider):
    """Abstract interface for CSV file providers."""
    
    @abstractmethod
    def read_csv(self, path: Union[str, Path], **kwargs) -> Dict[str, np.ndarray]:
        """Read CSV file."""
        pass
    
    @abstractmethod
    def write_csv(self, data: Dict[str, np.ndarray], path: Union[str, Path], **kwargs) -> None:
        """Write CSV file."""
        pass


class NPYProvider(FileFormatProvider):
    """Abstract interface for NumPy file providers."""
    
    @abstractmethod
    def read_npy(self, path: Union[str, Path], **kwargs) -> np.ndarray:
        """Read NPY file."""
        pass
    
    @abstractmethod
    def write_npy(self, data: np.ndarray, path: Union[str, Path], **kwargs) -> None:
        """Write NPY file."""
        pass
    
    @abstractmethod
    def read_npz(self, path: Union[str, Path], **kwargs) -> Dict[str, np.ndarray]:
        """Read NPZ file."""
        pass
    
    @abstractmethod
    def write_npz(self, data: Dict[str, np.ndarray], path: Union[str, Path], 
                 compressed: bool = True, **kwargs) -> None:
        """Write NPZ file."""
        pass


class CacheProvider(StorageProvider):
    """Abstract interface for caching providers."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """Get all cache keys."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get cache size in bytes."""
        pass


class CheckpointProvider(StorageProvider):
    """Abstract interface for checkpoint providers."""
    
    @abstractmethod
    def save_checkpoint(self, data: Dict[str, Any], checkpoint_id: str) -> None:
        """Save checkpoint data."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data."""
        pass
    
    @abstractmethod
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint."""
        pass
    
    @abstractmethod
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        pass
    
    @abstractmethod
    def checkpoint_exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists."""
        pass


class CompressionProvider(DataProvider):
    """Abstract interface for compression providers."""
    
    @abstractmethod
    def compress(self, data: bytes, method: str = "gzip", level: int = 6) -> bytes:
        """Compress data."""
        pass
    
    @abstractmethod
    def decompress(self, data: bytes, method: str = "gzip") -> bytes:
        """Decompress data."""
        pass
    
    @abstractmethod
    def compress_array(self, array: np.ndarray, method: str = "gzip", 
                      level: int = 6) -> bytes:
        """Compress numpy array."""
        pass
    
    @abstractmethod
    def decompress_array(self, data: bytes, method: str = "gzip") -> np.ndarray:
        """Decompress to numpy array."""
        pass