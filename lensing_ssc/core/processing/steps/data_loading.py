# lensing_ssc/core/processing/steps/data_loading.py
"""
Data loading processing steps for LensingSSC pipelines.

This module provides steps for discovering, loading, and validating data files
used in lensing super-sample covariance analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import numpy as np

from ..pipeline import ProcessingStep, StepResult, PipelineContext, StepStatus
from ...base import ValidationError, ProcessingError, DataError
from . import BaseDataStep

logger = logging.getLogger(__name__)


class FileDiscoveryStep(BaseDataStep):
    """Step for discovering data files based on patterns and filters.
    
    Searches specified directories for files matching given patterns,
    applies size and format filters, and returns organized file lists.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.search_patterns = kwargs.get('search_patterns', ['*.fits', '*.hdf5'])
        self.min_file_size_mb = kwargs.get('min_file_size_mb', 0.1)
        self.max_file_size_gb = kwargs.get('max_file_size_gb', 50.0)
        self.recursive = kwargs.get('recursive', True)
        self.sort_by = kwargs.get('sort_by', 'name')  # 'name', 'size', 'modified'
    
    def validate_inputs(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> bool:
        """Validate that required configuration is present."""
        if not super().validate_inputs(context, inputs):
            return False
        
        # Check for data directory in context
        if not hasattr(context.config, 'data_dir'):
            self.logger.error("data_dir not specified in configuration")
            return False
        
        data_dir = Path(context.config.data_dir)
        if not data_dir.exists():
            self.logger.error(f"Data directory does not exist: {data_dir}")
            return False
        
        return True
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute file discovery process."""
        self.logger.info(f"Starting file discovery with patterns: {self.search_patterns}")
        
        data_dir = Path(context.config.data_dir)
        discovered_files = []
        file_metadata = {}
        
        try:
            # Search for files matching each pattern
            for pattern in self.search_patterns:
                if self.recursive:
                    pattern_files = list(data_dir.rglob(pattern))
                else:
                    pattern_files = list(data_dir.glob(pattern))
                
                self.logger.debug(f"Pattern '{pattern}' found {len(pattern_files)} files")
                
                for file_path in pattern_files:
                    if self._validate_file_path(file_path) and self._apply_size_filters(file_path):
                        discovered_files.append(file_path)
                        file_metadata[str(file_path)] = self._get_file_info(file_path)
            
            # Remove duplicates and sort
            discovered_files = list(set(discovered_files))
            discovered_files = self._sort_files(discovered_files)
            
            self.logger.info(f"Discovered {len(discovered_files)} valid files")
            
            # Organize results
            result_data = {
                'files': [str(f) for f in discovered_files],
                'file_paths': discovered_files,
                'metadata': file_metadata,
                'search_patterns': self.search_patterns,
                'search_directory': str(data_dir),
                'total_files': len(discovered_files),
                'total_size_gb': sum(info['size_mb'] for info in file_metadata.values()) / 1024
            }
            
            return StepResult(
                status=StepStatus.SUCCESS,
                data=result_data,
                metadata={
                    'step_name': self.name,
                    'files_found': len(discovered_files),
                    'patterns_used': self.search_patterns
                }
            )
            
        except Exception as e:
            self.logger.error(f"File discovery failed: {e}")
            return StepResult(
                status=StepStatus.FAILED,
                error=f"File discovery error: {e}",
                data={}
            )
    
    def _apply_size_filters(self, file_path: Path) -> bool:
        """Apply size-based filters to files."""
        size_mb = file_path.stat().st_size / (1024**2)
        size_gb = size_mb / 1024
        
        if size_mb < self.min_file_size_mb:
            self.logger.debug(f"File too small ({size_mb:.2f} MB): {file_path}")
            return False
        
        if size_gb > self.max_file_size_gb:
            self.logger.warning(f"Large file ({size_gb:.1f} GB): {file_path}")
            
        return True
    
    def _sort_files(self, files: List[Path]) -> List[Path]:
        """Sort files based on specified criteria."""
        if self.sort_by == 'name':
            return sorted(files, key=lambda f: f.name)
        elif self.sort_by == 'size':
            return sorted(files, key=lambda f: f.stat().st_size)
        elif self.sort_by == 'modified':
            return sorted(files, key=lambda f: f.stat().st_mtime)
        else:
            return files


class DataLoadingStep(BaseDataStep):
    """Step for loading data from discovered files.
    
    Loads data from various formats (FITS, HDF5, NumPy) with format-specific
    handling and memory management.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.load_format = kwargs.get('load_format', 'auto')  # 'auto', 'fits', 'hdf5', 'npy'
        self.memory_limit_gb = kwargs.get('memory_limit_gb', 8.0)
        self.lazy_loading = kwargs.get('lazy_loading', True)
        self.chunk_size = kwargs.get('chunk_size', 1000)
        self.fits_hdu = kwargs.get('fits_hdu', 0)
        self.hdf5_dataset = kwargs.get('hdf5_dataset', None)
    
    def validate_inputs(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> bool:
        """Validate inputs for data loading."""
        if not super().validate_inputs(context, inputs):
            return False
        
        # Check for file discovery results
        if not any('files' in inp.data for inp in inputs.values()):
            self.logger.error("No file discovery results found in inputs")
            return False
        
        return True
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute data loading process."""
        # Get file list from discovery step
        file_input = next(inp for inp in inputs.values() if 'files' in inp.data)
        file_paths = [Path(f) for f in file_input.data['files']]
        
        self.logger.info(f"Loading {len(file_paths)} data files")
        
        try:
            loaded_data = {}
            data_info = {}
            
            for file_path in file_paths:
                self.logger.debug(f"Loading file: {file_path}")
                
                # Determine format
                format_type = self._detect_format(file_path)
                
                # Load data based on format
                data, info = self._load_file(file_path, format_type)
                
                file_key = file_path.stem
                loaded_data[file_key] = data
                data_info[file_key] = info
            
            result_data = {
                'data': loaded_data,
                'data_info': data_info,
                'file_count': len(file_paths),
                'formats_loaded': list(set(info['format'] for info in data_info.values()))
            }
            
            return StepResult(
                status=StepStatus.SUCCESS,
                data=result_data,
                metadata={
                    'step_name': self.name,
                    'files_loaded': len(file_paths),
                    'total_memory_gb': sum(info.get('size_gb', 0) for info in data_info.values())
                }
            )
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            return StepResult(
                status=StepStatus.FAILED,
                error=f"Data loading error: {e}",
                data={}
            )
    
    def _detect_format(self, file_path: Path) -> str:
        """Detect file format from extension or content."""
        if self.load_format != 'auto':
            return self.load_format
        
        suffix = file_path.suffix.lower()
        if suffix in ['.fits', '.fit']:
            return 'fits'
        elif suffix in ['.hdf5', '.h5']:
            return 'hdf5'
        elif suffix in ['.npy', '.npz']:
            return 'numpy'
        else:
            self.logger.warning(f"Unknown format for {file_path}, trying FITS")
            return 'fits'
    
    def _load_file(self, file_path: Path, format_type: str) -> tuple:
        """Load file based on format type."""
        if format_type == 'fits':
            return self._load_fits(file_path)
        elif format_type == 'hdf5':
            return self._load_hdf5(file_path)
        elif format_type == 'numpy':
            return self._load_numpy(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _load_fits(self, file_path: Path) -> tuple:
        """Load FITS file."""
        try:
            from astropy.io import fits
        except ImportError:
            raise ImportError("astropy is required for FITS file loading")
        
        with fits.open(file_path) as hdul:
            data = hdul[self.fits_hdu].data
            header = dict(hdul[self.fits_hdu].header)
            
            info = {
                'format': 'fits',
                'shape': data.shape if hasattr(data, 'shape') else None,
                'dtype': str(data.dtype) if hasattr(data, 'dtype') else None,
                'size_gb': data.nbytes / (1024**3) if hasattr(data, 'nbytes') else 0,
                'header': header
            }
            
        return data, info
    
    def _load_hdf5(self, file_path: Path) -> tuple:
        """Load HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 file loading")
        
        with h5py.File(file_path, 'r') as f:
            if self.hdf5_dataset:
                data = f[self.hdf5_dataset][:]
                attrs = dict(f[self.hdf5_dataset].attrs)
            else:
                # Load first dataset if not specified
                first_key = list(f.keys())[0]
                data = f[first_key][:]
                attrs = dict(f[first_key].attrs)
            
            info = {
                'format': 'hdf5',
                'shape': data.shape,
                'dtype': str(data.dtype),
                'size_gb': data.nbytes / (1024**3),
                'attributes': attrs
            }
        
        return data, info
    
    def _load_numpy(self, file_path: Path) -> tuple:
        """Load NumPy file."""
        if file_path.suffix == '.npz':
            data = np.load(file_path)
            # Get first array if multiple
            first_key = list(data.keys())[0]
            array_data = data[first_key]
        else:
            array_data = np.load(file_path)
        
        info = {
            'format': 'numpy',
            'shape': array_data.shape,
            'dtype': str(array_data.dtype),
            'size_gb': array_data.nbytes / (1024**3)
        }
        
        return array_data, info


class DataValidationStep(BaseDataStep):
    """Step for validating loaded data quality and consistency.
    
    Performs comprehensive checks on data arrays including range validation,
    NaN/infinity detection, and statistical outlier analysis.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.check_finite = kwargs.get('check_finite', True)
        self.check_range = kwargs.get('check_range', True)
        self.expected_min = kwargs.get('expected_min', None)
        self.expected_max = kwargs.get('expected_max', None)
        self.outlier_threshold = kwargs.get('outlier_threshold', 5.0)  # sigma
        self.required_shape = kwargs.get('required_shape', None)
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute data validation process."""
        # Get loaded data
        data_input = next(inp for inp in inputs.values() if 'data' in inp.data)
        loaded_data = data_input.data['data']
        
        self.logger.info(f"Validating {len(loaded_data)} data arrays")
        
        validation_results = {}
        overall_valid = True
        
        try:
            for data_key, data_array in loaded_data.items():
                self.logger.debug(f"Validating data: {data_key}")
                
                result = self._validate_array(data_array, data_key)
                validation_results[data_key] = result
                
                if not result['valid']:
                    overall_valid = False
                    self.logger.warning(f"Validation failed for {data_key}: {result['issues']}")
            
            result_data = {
                'validation_results': validation_results,
                'overall_valid': overall_valid,
                'valid_count': sum(1 for r in validation_results.values() if r['valid']),
                'total_count': len(validation_results)
            }
            
            status = StepStatus.SUCCESS if overall_valid else StepStatus.WARNING
            
            return StepResult(
                status=status,
                data=result_data,
                metadata={
                    'step_name': self.name,
                    'arrays_validated': len(validation_results),
                    'validation_passed': overall_valid
                }
            )
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return StepResult(
                status=StepStatus.FAILED,
                error=f"Validation error: {e}",
                data={}
            )
    
    def _validate_array(self, data_array: np.ndarray, data_key: str) -> Dict[str, Any]:
        """Validate individual data array."""
        issues = []
        
        # Check if data is numpy array
        if not isinstance(data_array, np.ndarray):
            return {'valid': False, 'issues': ['Data is not a numpy array']}
        
        # Check shape requirements
        if self.required_shape and data_array.shape != self.required_shape:
            issues.append(f"Shape mismatch: {data_array.shape} != {self.required_shape}")
        
        # Check for finite values
        if self.check_finite:
            if not np.isfinite(data_array).all():
                nan_count = np.isnan(data_array).sum()
                inf_count = np.isinf(data_array).sum()
                issues.append(f"Non-finite values: {nan_count} NaN, {inf_count} Inf")
        
        # Check value ranges
        if self.check_range and np.isfinite(data_array).any():
            finite_data = data_array[np.isfinite(data_array)]
            data_min, data_max = finite_data.min(), finite_data.max()
            
            if self.expected_min is not None and data_min < self.expected_min:
                issues.append(f"Minimum value {data_min} below expected {self.expected_min}")
            
            if self.expected_max is not None and data_max > self.expected_max:
                issues.append(f"Maximum value {data_max} above expected {self.expected_max}")
        
        # Check for statistical outliers
        if np.isfinite(data_array).any():
            finite_data = data_array[np.isfinite(data_array)]
            mean_val = np.mean(finite_data)
            std_val = np.std(finite_data)
            
            if std_val > 0:
                outliers = np.abs(finite_data - mean_val) > self.outlier_threshold * std_val
                outlier_count = outliers.sum()
                outlier_fraction = outlier_count / len(finite_data)
                
                if outlier_fraction > 0.01:  # More than 1% outliers
                    issues.append(f"High outlier fraction: {outlier_fraction:.3f}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'shape': data_array.shape,
            'dtype': str(data_array.dtype),
            'finite_fraction': np.isfinite(data_array).mean() if data_array.size > 0 else 0,
            'value_range': [float(np.nanmin(data_array)), float(np.nanmax(data_array))] if data_array.size > 0 else [0, 0]
        }