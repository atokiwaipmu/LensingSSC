# lensing_ssc/processing/steps/data_loading.py
"""
Data loading and validation steps for LensingSSC processing.

This module provides processing steps for discovering, loading, and validating
various types of data files used in lensing analysis, including:

- Generic file discovery and validation
- Kappa map loading from FITS files  
- N-body simulation data loading from BigFile catalogs
- Data structure validation and metadata extraction

Steps in this module:
    - FileDiscoveryStep: Generic file discovery and metadata extraction
    - DataLoadingStep: Generic data loading with format detection
    - DataValidationStep: Data structure and content validation
    - KappaMapLoadingStep: Specialized kappa map loading from FITS
    - USMeshLoadingStep: N-body simulation data loading from BigFile

Usage:
    from lensing_ssc.processing.steps.data_loading import KappaMapLoadingStep
    
    # Load kappa maps
    kappa_step = KappaMapLoadingStep(
        "load_kappa",
        input_directory="/path/to/kappa/maps",
        file_pattern="kappa_*.fits"
    )
    
    # Add to pipeline
    pipeline.add_step(kappa_step)
"""

import logging
from typing import Dict, List, Any, Optional, Union, Pattern
from pathlib import Path
import re
import time
from dataclasses import dataclass

from . import BaseDataStep, StepResult, PipelineContext, StepStatus
from lensing_ssc.core.base import ValidationError, ProcessingError, DataError


logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Container for file metadata and information."""
    path: Path
    name: str
    suffix: str
    size_bytes: int
    size_mb: float
    modified_time: float
    exists: bool
    metadata: Dict[str, Any]
    
    @classmethod
    def from_path(cls, file_path: Path) -> 'FileInfo':
        """Create FileInfo from a file path."""
        if file_path.exists():
            stat = file_path.stat()
            return cls(
                path=file_path,
                name=file_path.name,
                suffix=file_path.suffix,
                size_bytes=stat.st_size,
                size_mb=stat.st_size / (1024**2),
                modified_time=stat.st_mtime,
                exists=True,
                metadata={}
            )
        else:
            return cls(
                path=file_path,
                name=file_path.name,
                suffix=file_path.suffix,
                size_bytes=0,
                size_mb=0.0,
                modified_time=0.0,
                exists=False,
                metadata={}
            )


class FileDiscoveryStep(BaseDataStep):
    """Discover files matching specified patterns and extract metadata.
    
    This step provides generic file discovery capabilities with pattern matching,
    metadata extraction, and file validation.
    
    Parameters
    ----------
    name : str
        Step instance name
    input_directory : Union[str, Path]
        Directory to search for files
    file_patterns : Union[str, List[str]]
        File patterns to match (glob-style)
    recursive : bool, optional
        Whether to search recursively
    required_files : bool, optional
        Whether files are required to be found
    metadata_extractors : Dict[str, callable], optional
        Custom metadata extraction functions
    """
    
    def __init__(
        self,
        name: str,
        input_directory: Union[str, Path],
        file_patterns: Union[str, List[str]],
        recursive: bool = False,
        required_files: bool = True,
        metadata_extractors: Optional[Dict[str, callable]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.input_directory = Path(input_directory)
        self.file_patterns = file_patterns if isinstance(file_patterns, list) else [file_patterns]
        self.recursive = recursive
        self.required_files = required_files
        self.metadata_extractors = metadata_extractors or {}
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute file discovery."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Validate input directory
            if not self.input_directory.exists():
                if self.required_files:
                    raise ValidationError(f"Input directory does not exist: {self.input_directory}")
                else:
                    self.logger.warning(f"Input directory does not exist: {self.input_directory}")
                    result.data = {'discovered_files': {}, 'file_count': 0}
                    result.status = StepStatus.COMPLETED
                    return result
            
            # Discover files
            discovered_files = self._discover_files()
            
            # Extract metadata
            self._extract_metadata(discovered_files)
            
            # Validate discovery results
            if self.required_files and not discovered_files:
                raise ValidationError("No files found matching the specified patterns")
            
            result.data = {
                'discovered_files': discovered_files,
                'file_count': len(discovered_files),
                'input_directory': str(self.input_directory),
                'patterns_used': self.file_patterns
            }
            
            result.metadata = {
                'n_files_found': len(discovered_files),
                'total_size_mb': sum(info.size_mb for info in discovered_files.values()),
                'file_extensions': list(set(info.suffix for info in discovered_files.values())),
                'discovery_successful': True
            }
            
            self.logger.info(f"Discovered {len(discovered_files)} files in {self.input_directory}")
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _discover_files(self) -> Dict[Path, FileInfo]:
        """Discover files matching patterns."""
        discovered_files = {}
        
        for pattern in self.file_patterns:
            if self.recursive:
                matches = self.input_directory.rglob(pattern)
            else:
                matches = self.input_directory.glob(pattern)
            
            for file_path in matches:
                if file_path.is_file():
                    file_info = FileInfo.from_path(file_path)
                    discovered_files[file_path] = file_info
                    self.logger.debug(f"Found file: {file_path} ({file_info.size_mb:.2f} MB)")
        
        return discovered_files
    
    def _extract_metadata(self, discovered_files: Dict[Path, FileInfo]) -> None:
        """Extract metadata from discovered files."""
        for file_path, file_info in discovered_files.items():
            # Apply custom metadata extractors
            for extractor_name, extractor_func in self.metadata_extractors.items():
                try:
                    metadata = extractor_func(file_path)
                    file_info.metadata[extractor_name] = metadata
                except Exception as e:
                    self.logger.warning(f"Metadata extraction '{extractor_name}' failed for {file_path}: {e}")
                    file_info.metadata[extractor_name] = None


class DataLoadingStep(BaseDataStep):
    """Generic data loading step with format detection.
    
    This step provides generic data loading capabilities with automatic
    format detection and validation.
    
    Parameters
    ----------
    name : str
        Step instance name
    file_dependency : str
        Name of step that provides file information
    file_key : str, optional
        Key in dependency output to get file information
    load_all : bool, optional
        Whether to load all discovered files
    max_files : int, optional
        Maximum number of files to load
    validate_data : bool, optional
        Whether to validate loaded data
    """
    
    def __init__(
        self,
        name: str,
        file_dependency: str,
        file_key: str = "discovered_files",
        load_all: bool = True,
        max_files: Optional[int] = None,
        validate_data: bool = True,
        **kwargs
    ):
        super().__init__(name, dependencies=[file_dependency], **kwargs)
        self.file_dependency = file_dependency
        self.file_key = file_key
        self.load_all = load_all
        self.max_files = max_files
        self.validate_data = validate_data
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute data loading."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Get file information from dependency
            file_input = inputs[self.file_dependency]
            if not file_input.is_successful():
                raise ProcessingError(f"File dependency '{self.file_dependency}' failed")
            
            discovered_files = file_input.data.get(self.file_key, {})
            if not discovered_files:
                raise ValidationError("No files to load")
            
            # Select files to load
            files_to_load = self._select_files_to_load(discovered_files)
            
            # Load data from files
            loaded_data = self._load_data_files(files_to_load)
            
            result.data = {
                'loaded_data': loaded_data,
                'file_count': len(loaded_data),
                'loading_successful': True
            }
            
            result.metadata = {
                'n_files_loaded': len(loaded_data),
                'n_files_available': len(discovered_files),
                'loading_errors': sum(1 for data in loaded_data.values() if data.get('error')),
                'data_types': list(set(data.get('data_type') for data in loaded_data.values() if data.get('data_type')))
            }
            
            self.logger.info(f"Loaded data from {len(loaded_data)} files")
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _select_files_to_load(self, discovered_files: Dict[Path, FileInfo]) -> Dict[Path, FileInfo]:
        """Select which files to load based on configuration."""
        files_to_load = discovered_files
        
        if not self.load_all:
            # Take first file only
            files_to_load = {list(discovered_files.keys())[0]: list(discovered_files.values())[0]}
        
        if self.max_files and len(files_to_load) > self.max_files:
            # Limit to max_files
            items = list(files_to_load.items())[:self.max_files]
            files_to_load = dict(items)
        
        return files_to_load
    
    def _load_data_files(self, files_to_load: Dict[Path, FileInfo]) -> Dict[Path, Dict[str, Any]]:
        """Load data from files with format detection."""
        loaded_data = {}
        
        for file_path, file_info in files_to_load.items():
            try:
                self.logger.debug(f"Loading data from {file_path}")
                
                # Detect format and load data
                data_info = self._load_single_file(file_path, file_info)
                
                # Validate data if requested
                if self.validate_data and data_info.get('data') is not None:
                    validation_result = self._validate_loaded_data(data_info['data'], file_path)
                    data_info.update(validation_result)
                
                loaded_data[file_path] = data_info
                
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
                loaded_data[file_path] = {
                    'data': None,
                    'error': str(e),
                    'file_info': file_info,
                    'loading_successful': False
                }
        
        return loaded_data
    
    def _load_single_file(self, file_path: Path, file_info: FileInfo) -> Dict[str, Any]:
        """Load a single file with format detection."""
        file_format = self._detect_file_format(file_path)
        
        data_info = {
            'file_path': file_path,
            'file_info': file_info,
            'file_format': file_format,
            'loading_successful': False,
            'data': None
        }
        
        if file_format == 'fits':
            data_info.update(self._load_fits_file(file_path))
        elif file_format == 'npy':
            data_info.update(self._load_npy_file(file_path))
        elif file_format == 'csv':
            data_info.update(self._load_csv_file(file_path))
        elif file_format == 'hdf5':
            data_info.update(self._load_hdf5_file(file_path))
        else:
            raise ProcessingError(f"Unsupported file format: {file_format}")
        
        return data_info
    
    def _detect_file_format(self, file_path: Path) -> str:
        """Detect file format from extension."""
        suffix = file_path.suffix.lower()
        
        format_mapping = {
            '.fits': 'fits',
            '.fit': 'fits',
            '.npy': 'npy',
            '.npz': 'npz',
            '.csv': 'csv',
            '.txt': 'txt',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
        }
        
        return format_mapping.get(suffix, 'unknown')
    
    def _load_fits_file(self, file_path: Path) -> Dict[str, Any]:
        """Load FITS file."""
        try:
            import healpy as hp
            
            data = hp.read_map(str(file_path), nest=None)
            
            return {
                'data': data,
                'data_type': 'healpix_map',
                'shape': data.shape,
                'dtype': str(data.dtype),
                'loading_successful': True
            }
        except ImportError:
            raise ProcessingError("healpy is required to load FITS files")
    
    def _load_npy_file(self, file_path: Path) -> Dict[str, Any]:
        """Load NumPy file."""
        try:
            import numpy as np
            
            data = np.load(file_path)
            
            return {
                'data': data,
                'data_type': 'numpy_array',
                'shape': data.shape,
                'dtype': str(data.dtype),
                'loading_successful': True
            }
        except ImportError:
            raise ProcessingError("numpy is required to load .npy files")
    
    def _load_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """Load CSV file."""
        try:
            import pandas as pd
            
            data = pd.read_csv(file_path)
            
            return {
                'data': data,
                'data_type': 'dataframe',
                'shape': data.shape,
                'columns': list(data.columns),
                'loading_successful': True
            }
        except ImportError:
            raise ProcessingError("pandas is required to load CSV files")
    
    def _load_hdf5_file(self, file_path: Path) -> Dict[str, Any]:
        """Load HDF5 file."""
        try:
            import h5py
            
            with h5py.File(file_path, 'r') as f:
                # Get basic structure info
                keys = list(f.keys())
                
            return {
                'data': file_path,  # Store path for lazy loading
                'data_type': 'hdf5_file',
                'keys': keys,
                'loading_successful': True
            }
        except ImportError:
            raise ProcessingError("h5py is required to load HDF5 files")
    
    def _validate_loaded_data(self, data: Any, file_path: Path) -> Dict[str, Any]:
        """Validate loaded data."""
        validation_result = {
            'validation_passed': False,
            'validation_errors': []
        }
        
        try:
            import numpy as np
            
            if isinstance(data, np.ndarray):
                # Validate numpy array
                if data.size == 0:
                    validation_result['validation_errors'].append("Array is empty")
                elif not np.isfinite(data).any():
                    validation_result['validation_errors'].append("Array contains no finite values")
                else:
                    validation_result['validation_passed'] = True
            
            elif hasattr(data, 'shape'):
                # Validate pandas DataFrame or similar
                if data.empty:
                    validation_result['validation_errors'].append("Data is empty")
                else:
                    validation_result['validation_passed'] = True
            
            else:
                # Basic validation for other types
                if data is None:
                    validation_result['validation_errors'].append("Data is None")
                else:
                    validation_result['validation_passed'] = True
                    
        except Exception as e:
            validation_result['validation_errors'].append(f"Validation error: {e}")
        
        return validation_result


class DataValidationStep(BaseDataStep):
    """Validate data structure and content.
    
    Parameters
    ----------
    name : str
        Step instance name
    data_dependency : str
        Name of step that provides data
    validation_rules : Dict[str, Any], optional
        Custom validation rules
    strict_mode : bool, optional
        Whether to fail on any validation error
    """
    
    def __init__(
        self,
        name: str,
        data_dependency: str,
        validation_rules: Optional[Dict[str, Any]] = None,
        strict_mode: bool = False,
        **kwargs
    ):
        super().__init__(name, dependencies=[data_dependency], **kwargs)
        self.data_dependency = data_dependency
        self.validation_rules = validation_rules or {}
        self.strict_mode = strict_mode
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute data validation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Get data from dependency
            data_input = inputs[self.data_dependency]
            if not data_input.is_successful():
                raise ProcessingError(f"Data dependency '{self.data_dependency}' failed")
            
            loaded_data = data_input.data.get('loaded_data', {})
            
            # Validate each data item
            validation_results = self._validate_all_data(loaded_data)
            
            # Aggregate results
            total_items = len(validation_results)
            passed_items = sum(1 for v in validation_results.values() if v['validation_passed'])
            failed_items = total_items - passed_items
            
            result.data = {
                'validation_results': validation_results,
                'validation_summary': {
                    'total_items': total_items,
                    'passed_items': passed_items,
                    'failed_items': failed_items,
                    'success_rate': passed_items / total_items if total_items > 0 else 0
                }
            }
            
            result.metadata = {
                'validation_passed': failed_items == 0,
                'n_items_validated': total_items,
                'n_failed_validations': failed_items,
                'validation_success_rate': passed_items / total_items if total_items > 0 else 0
            }
            
            # Check if we should fail in strict mode
            if self.strict_mode and failed_items > 0:
                raise ValidationError(f"Validation failed for {failed_items} items in strict mode")
            
            self.logger.info(f"Validated {total_items} items: {passed_items} passed, {failed_items} failed")
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _validate_all_data(self, loaded_data: Dict[Path, Dict[str, Any]]) -> Dict[Path, Dict[str, Any]]:
        """Validate all loaded data items."""
        validation_results = {}
        
        for file_path, data_info in loaded_data.items():
            validation_result = self._validate_single_item(data_info, file_path)
            validation_results[file_path] = validation_result
        
        return validation_results
    
    def _validate_single_item(self, data_info: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Validate a single data item."""
        validation_result = {
            'file_path': file_path,
            'validation_passed': True,
            'validation_errors': [],
            'validation_warnings': []
        }
        
        try:
            # Check if data loaded successfully
            if not data_info.get('loading_successful', False):
                validation_result['validation_errors'].append("Data loading failed")
                validation_result['validation_passed'] = False
                return validation_result
            
            data = data_info.get('data')
            if data is None:
                validation_result['validation_errors'].append("Data is None")
                validation_result['validation_passed'] = False
                return validation_result
            
            # Apply custom validation rules
            for rule_name, rule_config in self.validation_rules.items():
                rule_result = self._apply_validation_rule(data, rule_name, rule_config)
                if not rule_result['passed']:
                    validation_result['validation_errors'].extend(rule_result['errors'])
                    validation_result['validation_passed'] = False
                validation_result['validation_warnings'].extend(rule_result.get('warnings', []))
            
        except Exception as e:
            validation_result['validation_errors'].append(f"Validation exception: {e}")
            validation_result['validation_passed'] = False
        
        return validation_result
    
    def _apply_validation_rule(self, data: Any, rule_name: str, rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a custom validation rule."""
        rule_result = {
            'passed': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            rule_type = rule_config.get('type', 'custom')
            
            if rule_type == 'shape':
                # Validate data shape
                expected_shape = rule_config['expected_shape']
                if hasattr(data, 'shape') and data.shape != expected_shape:
                    rule_result['errors'].append(f"Shape mismatch: expected {expected_shape}, got {data.shape}")
                    rule_result['passed'] = False
            
            elif rule_type == 'range':
                # Validate value range
                import numpy as np
                if isinstance(data, np.ndarray):
                    min_val = rule_config.get('min_value')
                    max_val = rule_config.get('max_value')
                    
                    if min_val is not None and np.min(data) < min_val:
                        rule_result['errors'].append(f"Values below minimum: {np.min(data)} < {min_val}")
                        rule_result['passed'] = False
                    
                    if max_val is not None and np.max(data) > max_val:
                        rule_result['errors'].append(f"Values above maximum: {np.max(data)} > {max_val}")
                        rule_result['passed'] = False
            
            elif rule_type == 'finite':
                # Check for finite values
                import numpy as np
                if isinstance(data, np.ndarray):
                    if not np.isfinite(data).all():
                        n_invalid = np.sum(~np.isfinite(data))
                        rule_result['warnings'].append(f"Found {n_invalid} non-finite values")
                        
                        if rule_config.get('require_all_finite', False):
                            rule_result['errors'].append("Non-finite values found when all finite required")
                            rule_result['passed'] = False
            
            elif rule_type == 'custom':
                # Apply custom validation function
                validator_func = rule_config.get('validator')
                if validator_func:
                    custom_result = validator_func(data)
                    rule_result.update(custom_result)
            
        except Exception as e:
            rule_result['errors'].append(f"Rule '{rule_name}' failed: {e}")
            rule_result['passed'] = False
        
        return rule_result


class KappaMapLoadingStep(BaseDataStep):
    """Specialized step for loading kappa maps from FITS files.
    
    Parameters
    ----------
    name : str
        Step instance name
    input_directory : Union[str, Path]
        Directory containing kappa FITS files
    file_pattern : str, optional
        Pattern to match kappa files
    validate_healpix : bool, optional
        Whether to validate HEALPix structure
    reorder_to_ring : bool, optional
        Whether to reorder maps to RING ordering
    """
    
    def __init__(
        self,
        name: str,
        input_directory: Union[str, Path],
        file_pattern: str = "kappa_*.fits",
        validate_healpix: bool = True,
        reorder_to_ring: bool = True,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.input_directory = Path(input_directory)
        self.file_pattern = file_pattern
        self.validate_healpix = validate_healpix
        self.reorder_to_ring = reorder_to_ring
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute kappa map loading."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Discover kappa files
            kappa_files = self._discover_kappa_files()
            
            # Load and process kappa maps
            loaded_maps = self._load_kappa_maps(kappa_files)
            
            result.data = {
                'kappa_maps': loaded_maps,
                'file_count': len(loaded_maps),
                'input_directory': str(self.input_directory)
            }
            
            result.metadata = {
                'n_maps_loaded': len(loaded_maps),
                'n_maps_failed': sum(1 for m in loaded_maps.values() if m.get('error')),
                'redshifts': list(set(m.get('redshift') for m in loaded_maps.values() if m.get('redshift'))),
                'seeds': list(set(m.get('seed') for m in loaded_maps.values() if m.get('seed'))),
                'nsides': list(set(m.get('nside') for m in loaded_maps.values() if m.get('nside')))
            }
            
            self.logger.info(f"Loaded {len(loaded_maps)} kappa maps")
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _discover_kappa_files(self) -> Dict[Path, Dict[str, Any]]:
        """Discover kappa FITS files and extract metadata."""
        if not self.input_directory.exists():
            raise ValidationError(f"Kappa input directory does not exist: {self.input_directory}")
        
        kappa_files = {}
        for fits_file in self.input_directory.glob(self.file_pattern):
            file_info = self._parse_kappa_filename(fits_file)
            if file_info:
                kappa_files[fits_file] = file_info
            else:
                self.logger.warning(f"Could not parse kappa filename: {fits_file.name}")
        
        if not kappa_files:
            raise ValidationError(f"No kappa files found matching pattern '{self.file_pattern}'")
        
        return kappa_files
    
    def _parse_kappa_filename(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse kappa filename to extract metadata."""
        # Pattern: kappa_zs{redshift}_s{seed}_nside{nside}.fits
        pattern = r"kappa_zs(\d+\.?\d*)_s(\w+)_nside(\d+)\.fits"
        match = re.match(pattern, file_path.name)
        
        if match:
            return {
                'redshift': float(match.group(1)),
                'seed': str(match.group(2)),
                'nside': int(match.group(3)),
                'file_path': file_path
            }
        return None
    
    def _load_kappa_maps(self, kappa_files: Dict[Path, Dict[str, Any]]) -> Dict[Path, Dict[str, Any]]:
        """Load kappa maps from FITS files."""
        try:
            import healpy as hp
            import numpy as np
        except ImportError:
            raise ProcessingError("healpy and numpy are required for kappa map loading")
        
        loaded_maps = {}
        
        for file_path, file_info in kappa_files.items():
            try:
                self.logger.debug(f"Loading kappa map: {file_path.name}")
                
                # Load the map
                kappa_map = hp.read_map(str(file_path), nest=None)
                
                # Reorder to RING if requested
                if self.reorder_to_ring:
                    # Check if map is in NEST ordering and reorder if needed
                    # Note: This is a heuristic since healpy doesn't always detect ordering correctly
                    try:
                        # Try to reorder - if it's already RING, this should be safe
                        kappa_map = hp.reorder(kappa_map, n2r=True)
                    except:
                        # If reordering fails, assume it's already in correct format
                        pass
                
                # Validate HEALPix structure if requested
                if self.validate_healpix:
                    self._validate_healpix_map(kappa_map, file_info)
                
                map_info = {
                    'data': kappa_map,
                    'file_path': file_path,
                    'redshift': file_info.get('redshift'),
                    'seed': file_info.get('seed'),
                    'nside': file_info.get('nside'),
                    'npix': len(kappa_map),
                    'shape': kappa_map.shape,
                    'dtype': str(kappa_map.dtype),
                    'mean': np.mean(kappa_map),
                    'std': np.std(kappa_map),
                    'min': np.min(kappa_map),
                    'max': np.max(kappa_map),
                    'n_finite': np.sum(np.isfinite(kappa_map)),
                    'loading_successful': True
                }
                
                loaded_maps[file_path] = map_info
                
            except Exception as e:
                self.logger.error(f"Failed to load kappa map {file_path}: {e}")
                loaded_maps[file_path] = {
                    'data': None,
                    'file_path': file_path,
                    'error': str(e),
                    'loading_successful': False,
                    **file_info
                }
        
        return loaded_maps
    
    def _validate_healpix_map(self, kappa_map: Any, file_info: Dict[str, Any]) -> None:
        """Validate HEALPix map structure."""
        try:
            import healpy as hp
            import numpy as np
        except ImportError:
            return
        
        # Check if it's a valid HEALPix map
        if not isinstance(kappa_map, np.ndarray):
            raise ValidationError("Kappa map is not a numpy array")
        
        if kappa_map.ndim != 1:
            raise ValidationError(f"Kappa map must be 1D, got {kappa_map.ndim}D")
        
        # Check if npix is valid for HEALPix
        npix = len(kappa_map)
        if not hp.isnpix(npix):
            raise ValidationError(f"Invalid number of pixels for HEALPix: {npix}")
        
        # Check expected nside if provided
        expected_nside = file_info.get('nside')
        if expected_nside:
            actual_nside = hp.npix2nside(npix)
            if actual_nside != expected_nside:
                raise ValidationError(f"NSIDE mismatch: expected {expected_nside}, got {actual_nside}")
        
        # Check for reasonable values
        if not np.isfinite(kappa_map).any():
            raise ValidationError("Kappa map contains no finite values")


class USMeshLoadingStep(BaseDataStep):
    """Specialized step for loading N-body simulation data from BigFile.
    
    Parameters
    ----------
    name : str
        Step instance name
    data_directory : Union[str, Path]
        Directory containing usmesh data
    dataset_name : str, optional
        BigFile dataset name
    validate_structure : bool, optional
        Whether to validate data structure
    """
    
    def __init__(
        self,
        name: str,
        data_directory: Union[str, Path],
        dataset_name: str = "HEALPIX/",
        validate_structure: bool = True,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.data_directory = Path(data_directory)
        self.dataset_name = dataset_name
        self.validate_structure = validate_structure
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute USMesh data loading."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Validate directory structure
            usmesh_dir = self.data_directory / "usmesh"
            if not usmesh_dir.exists():
                raise ValidationError(f"USMesh directory not found: {usmesh_dir}")
            
            # Load BigFile catalog
            catalog_info = self._load_bigfile_catalog(usmesh_dir)
            
            result.data = {
                'usmesh_catalog': catalog_info['catalog'],
                'catalog_attrs': catalog_info['attrs'],
                'data_directory': str(self.data_directory),
                'usmesh_directory': str(usmesh_dir)
            }
            
            result.metadata = {
                'catalog_loaded': True,
                'seed': catalog_info['attrs'].get('seed', [None])[0],
                'box_size': catalog_info['attrs'].get('BoxSize', [None])[0],
                'nc': catalog_info['attrs'].get('NC', [None])[0],
                'n_mass_sheets': len(catalog_info['attrs'].get('aemitIndex.edges', [])) - 1,
                'healpix_npix': catalog_info['attrs'].get('healpix.npix', [None])[0]
            }
            
            self.logger.info(f"Loaded USMesh catalog with {result.metadata['n_mass_sheets']} mass sheets")
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _load_bigfile_catalog(self, usmesh_dir: Path) -> Dict[str, Any]:
        """Load BigFile catalog and extract attributes."""
        try:
            from nbodykit.lab import BigFileCatalog
        except ImportError:
            raise ProcessingError("nbodykit is required for USMesh data loading")
        
        try:
            # Load the catalog
            catalog = BigFileCatalog(str(usmesh_dir), dataset=self.dataset_name)
            attrs = catalog.attrs
            
            # Validate structure if requested
            if self.validate_structure:
                self._validate_catalog_structure(catalog, attrs)
            
            return {
                'catalog': catalog,
                'attrs': attrs
            }
            
        except Exception as e:
            raise ProcessingError(f"Failed to load BigFile catalog: {e}")
    
    def _validate_catalog_structure(self, catalog: Any, attrs: Dict[str, Any]) -> None:
        """Validate BigFile catalog structure."""
        # Check required attributes
        required_attrs = ['aemitIndex.edges', 'aemitIndex.offset', 'healpix.npix', 'BoxSize', 'NC', 'MassTable']
        missing_attrs = []
        
        for attr in required_attrs:
            if attr not in attrs:
                missing_attrs.append(attr)
        
        if missing_attrs:
            raise ValidationError(f"Missing required attributes: {missing_attrs}")
        
        # Check required columns
        required_columns = ['ID', 'Mass', 'Aemit']
        missing_columns = []
        
        for col in required_columns:
            if col not in catalog:
                missing_columns.append(col)
        
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
        
        # Validate data consistency
        aemit_edges = attrs['aemitIndex.edges']
        aemit_offset = attrs['aemitIndex.offset']
        
        if len(aemit_offset) != len(aemit_edges) + 1:
            raise ValidationError("Inconsistent aemitIndex arrays")
        
        self.logger.debug("BigFile catalog structure validation passed")


__all__ = [
    'FileInfo',
    'FileDiscoveryStep',
    'DataLoadingStep',
    'DataValidationStep',
    'KappaMapLoadingStep',
    'USMeshLoadingStep',
]