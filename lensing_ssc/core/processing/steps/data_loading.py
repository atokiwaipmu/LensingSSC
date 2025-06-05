# lensing_ssc/processing/steps/data_loading.py
"""
Data loading and validation steps for LensingSSC processing pipelines.

Provides steps for file discovery, data loading, and validation with support
for FITS, HDF5, NumPy, and CSV formats commonly used in weak lensing analysis.
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from ..pipeline import ProcessingStep, StepResult, PipelineContext, StepStatus
from lensing_ssc.core.base import ValidationError, ProcessingError, DataError, IOError


logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Container for file metadata and information."""
    path: Path
    name: str
    stem: str
    suffix: str
    size_bytes: int
    size_mb: float
    modified_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_path(cls, path: Path) -> 'FileInfo':
        """Create FileInfo from a file path."""
        stat = path.stat()
        return cls(
            path=path,
            name=path.name,
            stem=path.stem,
            suffix=path.suffix,
            size_bytes=stat.st_size,
            size_mb=stat.st_size / (1024**2),
            modified_time=datetime.fromtimestamp(stat.st_mtime)
        )
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata entry."""
        self.metadata[key] = value
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the file."""
        if tag not in self.tags:
            self.tags.append(tag)


class FileDiscoveryStep(ProcessingStep):
    """Discover and catalog data files based on patterns and criteria."""
    
    def __init__(
        self,
        name: str,
        file_patterns: Optional[List[str]] = None,
        search_dirs: Optional[List[Union[str, Path]]] = None,
        recursive: bool = True,
        extract_metadata: bool = True,
        min_file_size: int = 0,
        max_file_size: Optional[int] = None,
        required_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.file_patterns = file_patterns or ["*.fits", "*.hdf5", "*.npy"]
        self.search_dirs = [Path(d) for d in (search_dirs or [])]
        self.recursive = recursive
        self.extract_metadata = extract_metadata
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.required_extensions = required_extensions or []
        self.exclude_patterns = exclude_patterns or []
        
        # Metadata extraction patterns
        self.metadata_patterns = {
            'kappa_maps': r"kappa_zs(\d+\.?\d*)_s(\w+)_nside(\d+)\.fits",
            'patch_files': r"(.+)_patches_oa(\d+)_x(\d+)\.npy",
            'stats_files': r"(.+)_stats_oa(\d+)_x(\d+)\.hdf5",
            'mass_sheets': r"delta-sheet-(\d+)\.fits",
        }
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute file discovery."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            if not self.search_dirs:
                self.search_dirs = self._get_search_dirs_from_context(context)
            
            discovered_files = []
            total_size = 0
            
            for search_dir in self.search_dirs:
                if not search_dir.exists():
                    self.logger.warning(f"Search directory does not exist: {search_dir}")
                    continue
                
                dir_files = self._discover_files_in_directory(search_dir)
                discovered_files.extend(dir_files)
                total_size += sum(f.size_bytes for f in dir_files)
            
            valid_files = self._filter_and_validate_files(discovered_files)
            
            if self.extract_metadata:
                self._extract_file_metadata(valid_files)
            
            organized_files = self._organize_files_by_type(valid_files)
            
            result.data = {
                'files': valid_files,
                'organized_files': organized_files,
                'file_count': len(valid_files),
                'total_size_gb': total_size / (1024**3),
                'search_dirs': [str(d) for d in self.search_dirs]
            }
            
            result.metadata = {
                'n_discovered': len(discovered_files),
                'n_valid': len(valid_files),
                'n_filtered': len(discovered_files) - len(valid_files),
                'total_size_gb': total_size / (1024**3),
                'file_types': list(organized_files.keys()),
                'search_patterns': self.file_patterns,
            }
            
            if len(valid_files) == 0:
                result.warnings.append("No valid files discovered")
            
            self.logger.info(f"Discovered {len(valid_files)} valid files ({total_size/(1024**3):.2f} GB)")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _get_search_dirs_from_context(self, context: PipelineContext) -> List[Path]:
        """Extract search directories from pipeline context."""
        search_dirs = []
        config = context.config
        
        for attr in ['data_dir', 'input_dir', 'kappa_input_dir', 'patch_output_dir']:
            if hasattr(config, attr):
                path = getattr(config, attr)
                if path and Path(path).exists():
                    search_dirs.append(Path(path))
        
        if not search_dirs:
            search_dirs = [Path.cwd()]
            self.logger.warning("No search directories specified, using current directory")
        
        return search_dirs
    
    def _discover_files_in_directory(self, directory: Path) -> List[FileInfo]:
        """Discover files in a single directory."""
        discovered = []
        
        for pattern in self.file_patterns:
            files = directory.rglob(pattern) if self.recursive else directory.glob(pattern)
            
            for file_path in files:
                if file_path.is_file():
                    try:
                        file_info = FileInfo.from_path(file_path)
                        discovered.append(file_info)
                    except Exception as e:
                        self.logger.warning(f"Failed to process file {file_path}: {e}")
        
        return discovered
    
    def _filter_and_validate_files(self, files: List[FileInfo]) -> List[FileInfo]:
        """Filter and validate discovered files."""
        valid_files = []
        
        for file_info in files:
            # Size filters
            if file_info.size_bytes < self.min_file_size:
                continue
            if self.max_file_size and file_info.size_bytes > self.max_file_size:
                continue
            
            # Extension filters
            if self.required_extensions and file_info.suffix.lower() not in self.required_extensions:
                continue
            
            # Exclude patterns
            if any(re.search(pattern, file_info.name) for pattern in self.exclude_patterns):
                continue
            
            # Basic validation
            if not file_info.path.exists() or file_info.size_bytes == 0:
                continue
            
            valid_files.append(file_info)
        
        return valid_files
    
    def _extract_file_metadata(self, files: List[FileInfo]) -> None:
        """Extract metadata from filenames using regex patterns."""
        for file_info in files:
            filename = file_info.name
            
            for pattern_name, pattern in self.metadata_patterns.items():
                match = re.match(pattern, filename)
                if match:
                    file_info.add_tag(pattern_name)
                    
                    if pattern_name == 'kappa_maps':
                        file_info.add_metadata('redshift', float(match.group(1)))
                        file_info.add_metadata('seed', match.group(2))
                        file_info.add_metadata('nside', int(match.group(3)))
                    elif pattern_name in ['patch_files', 'stats_files']:
                        file_info.add_metadata('base_name', match.group(1))
                        file_info.add_metadata('patch_size_deg', int(match.group(2)))
                        file_info.add_metadata('xsize', int(match.group(3)))
                    elif pattern_name == 'mass_sheets':
                        file_info.add_metadata('sheet_id', int(match.group(1)))
                    break
    
    def _organize_files_by_type(self, files: List[FileInfo]) -> Dict[str, List[FileInfo]]:
        """Organize files by type/category."""
        organized = {}
        
        for file_info in files:
            # Organize by tags first, fallback to extension
            categories = file_info.tags if file_info.tags else [file_info.suffix.lower()]
            
            for category in categories:
                if category not in organized:
                    organized[category] = []
                organized[category].append(file_info)
        
        # Sort files within each category
        for category in organized:
            organized[category].sort(key=lambda f: f.path)
        
        return organized


class DataLoadingStep(ProcessingStep):
    """Load data from discovered files with format detection and validation."""
    
    def __init__(
        self,
        name: str,
        data_types: Optional[List[str]] = None,
        load_strategy: str = "eager",
        validate_on_load: bool = True,
        max_memory_gb: float = 8.0,
        cache_loaded_data: bool = True,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.data_types = data_types or ['healpix_map', 'patches', 'catalog']
        self.load_strategy = load_strategy
        self.validate_on_load = validate_on_load
        self.max_memory_gb = max_memory_gb
        self.cache_loaded_data = cache_loaded_data
        
        # Data loaders for different formats
        self.loaders = {
            '.fits': self._load_fits_file,
            '.hdf5': self._load_hdf5_file,
            '.h5': self._load_hdf5_file,
            '.npy': self._load_npy_file,
            '.npz': self._load_npz_file,
            '.csv': self._load_csv_file,
        }
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute data loading."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            discovery_result = self._get_discovery_result(inputs)
            if not discovery_result:
                raise ProcessingError("File discovery step result not found or failed")
            
            files_to_load = self._select_files_to_load(discovery_result.data)
            
            if not files_to_load:
                result.status = StepStatus.SKIPPED
                result.warnings.append("No files selected for loading")
                return result
            
            # Estimate memory and check limits
            estimated_memory = self._estimate_memory_requirements(files_to_load)
            if estimated_memory > self.max_memory_gb:
                self.logger.warning(
                    f"Estimated memory ({estimated_memory:.2f} GB) exceeds limit ({self.max_memory_gb} GB)"
                )
            
            # Load data
            loaded_data = {}
            loading_errors = []
            total_loaded = 0
            
            for file_info in files_to_load:
                try:
                    data = self._load_file(file_info)
                    
                    if self.validate_on_load and not self._validate_loaded_data(data, file_info):
                        loading_errors.append(f"Validation failed: {file_info.name}")
                        continue
                    
                    loaded_data[str(file_info.path)] = {
                        'data': data,
                        'file_info': file_info,
                        'load_time': time.time(),
                        'data_type': self._detect_data_type(data, file_info)
                    }
                    
                    total_loaded += 1
                    
                    if self.cache_loaded_data:
                        cache_key = f"loaded_data_{file_info.stem}"
                        context.shared_data[cache_key] = data
                    
                except Exception as e:
                    error_msg = f"Failed to load {file_info.name}: {e}"
                    loading_errors.append(error_msg)
                    self.logger.error(error_msg)
            
            result.data = {
                'loaded_data': loaded_data,
                'loading_errors': loading_errors,
                'files_loaded': total_loaded,
                'files_requested': len(files_to_load),
                'estimated_memory_gb': estimated_memory
            }
            
            result.metadata = {
                'n_files_loaded': total_loaded,
                'n_files_failed': len(loading_errors),
                'success_rate': total_loaded / len(files_to_load) if files_to_load else 0,
                'load_strategy': self.load_strategy,
                'data_types_loaded': list(set(
                    item['data_type'] for item in loaded_data.values()
                )),
            }
            
            if loading_errors:
                result.warnings.extend(loading_errors[:5])
                if len(loading_errors) > 5:
                    result.warnings.append(f"... and {len(loading_errors) - 5} more errors")
            
            self.logger.info(f"Successfully loaded {total_loaded}/{len(files_to_load)} files")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _get_discovery_result(self, inputs: Dict[str, StepResult]) -> Optional[StepResult]:
        """Get file discovery result from inputs."""
        for step_result in inputs.values():
            if (step_result.is_successful() and 
                'files' in step_result.data and
                'organized_files' in step_result.data):
                return step_result
        return None
    
    def _select_files_to_load(self, discovery_data: Dict[str, Any]) -> List[FileInfo]:
        """Select files to load based on data types."""
        files_to_load = []
        organized_files = discovery_data.get('organized_files', {})
        
        for data_type in self.data_types:
            if data_type in organized_files:
                files_to_load.extend(organized_files[data_type])
            else:
                # Match by file tags
                for file_list in organized_files.values():
                    for file_info in file_list:
                        if data_type in file_info.tags:
                            files_to_load.append(file_info)
        
        # Remove duplicates
        seen_paths = set()
        unique_files = []
        for file_info in files_to_load:
            if file_info.path not in seen_paths:
                unique_files.append(file_info)
                seen_paths.add(file_info.path)
        
        return unique_files
    
    def _estimate_memory_requirements(self, files: List[FileInfo]) -> float:
        """Estimate memory requirements for loading files."""
        total_size_gb = sum(f.size_mb for f in files) / 1024
        
        # Add overhead based on file types
        overhead_factor = 1.5
        for file_info in files:
            if file_info.suffix.lower() == '.fits':
                overhead_factor = max(overhead_factor, 2.0)  # FITS may be compressed
            elif file_info.suffix.lower() in ['.hdf5', '.h5']:
                overhead_factor = max(overhead_factor, 1.2)  # HDF5 is efficient
        
        return total_size_gb * overhead_factor
    
    def _load_file(self, file_info: FileInfo) -> Any:
        """Load a single file using appropriate loader."""
        loader = self.loaders.get(file_info.suffix.lower())
        if not loader:
            raise IOError(f"No loader available for file type: {file_info.suffix}")
        return loader(file_info.path)
    
    def _load_fits_file(self, file_path: Path) -> Any:
        """Load FITS file."""
        try:
            import healpy as hp
            return hp.read_map(str(file_path), nest=None)
        except ImportError:
            raise ProcessingError("healpy is required to load FITS files")
        except Exception:
            try:
                from astropy.io import fits
                with fits.open(file_path) as hdul:
                    return hdul[0].data
            except ImportError:
                raise ProcessingError("astropy is required as fallback for FITS files")
    
    def _load_hdf5_file(self, file_path: Path) -> Any:
        """Load HDF5 file."""
        try:
            import h5py
            return h5py.File(file_path, 'r')
        except ImportError:
            raise ProcessingError("h5py is required to load HDF5 files")
    
    def _load_npy_file(self, file_path: Path) -> np.ndarray:
        """Load NumPy .npy file."""
        return np.load(file_path)
    
    def _load_npz_file(self, file_path: Path) -> Any:
        """Load NumPy .npz file."""
        return np.load(file_path)
    
    def _load_csv_file(self, file_path: Path) -> Any:
        """Load CSV file."""
        try:
            import pandas as pd
            return pd.read_csv(file_path)
        except ImportError:
            return np.loadtxt(file_path, delimiter=',')
    
    def _detect_data_type(self, data: Any, file_info: FileInfo) -> str:
        """Detect the type of loaded data."""
        if file_info.tags:
            return file_info.tags[0]
        
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return 'vector_data'
            elif data.ndim == 2:
                return 'image_or_patch'
            elif data.ndim == 3:
                return 'patch_collection'
            return 'multidimensional_array'
        
        try:
            import h5py
            if isinstance(data, h5py.File):
                return 'hdf5_file'
        except ImportError:
            pass
        
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                return 'tabular_data'
        except ImportError:
            pass
        
        return 'unknown'
    
    def _validate_loaded_data(self, data: Any, file_info: FileInfo) -> bool:
        """Basic validation of loaded data."""
        if data is None:
            return False
        
        if isinstance(data, np.ndarray):
            return data.size > 0 and np.isfinite(data).any()
        
        return True


class DataValidationStep(ProcessingStep):
    """Validate loaded data for quality and consistency."""
    
    def __init__(
        self,
        name: str,
        validation_rules: Optional[Dict[str, Any]] = None,
        strict_mode: bool = False,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.validation_rules = validation_rules or {}
        self.strict_mode = strict_mode
        
        # Default validation rules
        self.default_rules = {
            'healpix_map': {
                'finite_values': True,
                'value_range': (-10, 10),
            },
            'patches': {
                'finite_values': True,
                'min_patches': 1,
            },
            'tabular_data': {
                'no_empty_rows': True,
            }
        }
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute data validation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            loading_result = self._get_loading_result(inputs)
            if not loading_result:
                raise ProcessingError("Data loading step result not found or failed")
            
            loaded_data = loading_result.data['loaded_data']
            
            if not loaded_data:
                result.status = StepStatus.SKIPPED
                result.warnings.append("No loaded data to validate")
                return result
            
            # Validate each dataset
            validation_results = {}
            total_issues = 0
            
            for file_path, data_info in loaded_data.items():
                validation_result = self._validate_dataset(
                    data_info['data'], 
                    data_info['data_type'], 
                    data_info['file_info']
                )
                validation_results[file_path] = validation_result
                total_issues += len(validation_result['issues'])
            
            # Generate summary
            valid_count = sum(1 for vr in validation_results.values() if vr['is_valid'])
            invalid_count = len(validation_results) - valid_count
            
            result.data = {
                'validation_results': validation_results,
                'summary': {
                    'total_datasets': len(validation_results),
                    'valid_datasets': valid_count,
                    'invalid_datasets': invalid_count,
                    'total_issues': total_issues,
                }
            }
            
            result.metadata = {
                'n_validated': len(validation_results),
                'n_valid': valid_count,
                'n_invalid': invalid_count,
                'total_issues': total_issues,
                'validation_passed': invalid_count == 0,
                'strict_mode': self.strict_mode,
            }
            
            if self.strict_mode and invalid_count > 0:
                raise ValidationError(f"Validation failed: {invalid_count} invalid datasets")
            
            if invalid_count > 0:
                result.warnings.append(f"{invalid_count} datasets failed validation")
            
            self.logger.info(f"Validation complete: {valid_count}/{len(validation_results)} datasets valid")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _get_loading_result(self, inputs: Dict[str, StepResult]) -> Optional[StepResult]:
        """Get data loading result from inputs."""
        for step_result in inputs.values():
            if (step_result.is_successful() and 'loaded_data' in step_result.data):
                return step_result
        return None
    
    def _validate_dataset(self, data: Any, data_type: str, file_info: FileInfo) -> Dict[str, Any]:
        """Validate a single dataset."""
        issues = []
        warnings = []
        
        rules = self.validation_rules.get(data_type, self.default_rules.get(data_type, {}))
        
        try:
            if data is None:
                issues.append("Data is None")
                return {'is_valid': False, 'issues': issues, 'warnings': warnings}
            
            # Type-specific validation
            if isinstance(data, np.ndarray):
                issues.extend(self._validate_numpy_array(data, rules))
            elif hasattr(data, 'keys'):
                issues.extend(self._validate_structured_data(data, rules))
            
            # Data type specific validation
            if data_type == 'healpix_map':
                issues.extend(self._validate_healpix_map(data, rules))
            elif data_type in ['patches', 'patch_collection']:
                issues.extend(self._validate_patches(data, rules))
            elif data_type == 'tabular_data':
                issues.extend(self._validate_tabular_data(data, rules))
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'data_type': data_type,
            'file_info': file_info.name,
            'validation_time': datetime.now()
        }
    
    def _validate_numpy_array(self, data: np.ndarray, rules: Dict[str, Any]) -> List[str]:
        """Validate numpy array data."""
        issues = []
        
        if rules.get('finite_values', True) and not np.isfinite(data).all():
            n_bad = np.sum(~np.isfinite(data))
            issues.append(f"Array contains {n_bad} non-finite values")
        
        if 'value_range' in rules:
            min_val, max_val = rules['value_range']
            data_min, data_max = np.min(data), np.max(data)
            if data_min < min_val or data_max > max_val:
                issues.append(f"Values outside expected range [{min_val}, {max_val}]: actual [{data_min:.3f}, {data_max:.3f}]")
        
        if data.size == 0:
            issues.append("Array is empty")
        
        return issues
    
    def _validate_structured_data(self, data: Any, rules: Dict[str, Any]) -> List[str]:
        """Validate structured data."""
        issues = []
        
        try:
            keys = list(data.keys()) if hasattr(data, 'keys') else []
            
            if not keys:
                issues.append("No data keys found")
            
            if 'required_keys' in rules:
                missing_keys = set(rules['required_keys']) - set(keys)
                if missing_keys:
                    issues.append(f"Missing required keys: {missing_keys}")
        
        except Exception as e:
            issues.append(f"Error accessing structured data: {str(e)}")
        
        return issues
    
    def _validate_healpix_map(self, data: np.ndarray, rules: Dict[str, Any]) -> List[str]:
        """Validate HEALPix map data."""
        issues = []
        
        if not isinstance(data, np.ndarray):
            issues.append("HEALPix map must be numpy array")
            return issues
        
        try:
            import healpy as hp
            if not hp.isnpix(data.size):
                issues.append(f"Invalid HEALPix map size: {data.size}")
            else:
                nside = hp.npix2nside(data.size)
                if nside < 1 or nside > 8192:
                    issues.append(f"Unusual NSIDE value: {nside}")
        except ImportError:
            issues.append("Cannot validate HEALPix map without healpy")
        
        # Check for typical kappa map characteristics
        if data.size > 0:
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if abs(mean_val) > 0.1:
                issues.append(f"Unusual mean value for kappa map: {mean_val:.6f}")
            
            if std_val < 1e-6:
                issues.append(f"Very small standard deviation: {std_val:.6e}")
            elif std_val > 1.0:
                issues.append(f"Very large standard deviation: {std_val:.6f}")
        
        return issues
    
    def _validate_patches(self, data: np.ndarray, rules: Dict[str, Any]) -> List[str]:
        """Validate patch data."""
        issues = []
        
        if not isinstance(data, np.ndarray):
            issues.append("Patches must be numpy array")
            return issues
        
        min_patches = rules.get('min_patches', 1)
        if data.ndim >= 1 and data.shape[0] < min_patches:
            issues.append(f"Too few patches: {data.shape[0]}, minimum: {min_patches}")
        
        if data.ndim == 3:  # Collection of 2D patches
            n_patches, height, width = data.shape
            
            if height != width:
                issues.append(f"Non-square patches: {height}x{width}")
            
            if height < 16 or height > 4096:
                issues.append(f"Unusual patch size: {height}x{width}")
        
        elif data.ndim == 2:  # Single patch
            height, width = data.shape
            if height != width:
                issues.append(f"Non-square patch: {height}x{width}")
        
        else:
            issues.append(f"Unexpected patch data dimensions: {data.ndim}")
        
        return issues
    
    def _validate_tabular_data(self, data: Any, rules: Dict[str, Any]) -> List[str]:
        """Validate tabular data."""
        issues = []
        
        try:
            import pandas as pd
            is_dataframe = isinstance(data, pd.DataFrame)
        except ImportError:
            is_dataframe = False
        
        if is_dataframe:
            if data.empty:
                issues.append("DataFrame is empty")
            
            if 'required_columns' in rules:
                missing_cols = set(rules['required_columns']) - set(data.columns)
                if missing_cols:
                    issues.append(f"Missing required columns: {missing_cols}")
            
            if rules.get('no_empty_rows', False):
                empty_rows = data.isnull().all(axis=1).sum()
                if empty_rows > 0:
                    issues.append(f"Found {empty_rows} empty rows")
        
        elif isinstance(data, np.ndarray) and data.dtype.names:
            if data.size == 0:
                issues.append("Structured array is empty")
            
            if 'required_columns' in rules:
                missing_fields = set(rules['required_columns']) - set(data.dtype.names)
                if missing_fields:
                    issues.append(f"Missing required fields: {missing_fields}")
        
        else:
            issues.append("Tabular data must be pandas DataFrame or structured numpy array")
        
        return issues


__all__ = [
    'FileInfo',
    'FileDiscoveryStep',
    'DataLoadingStep', 
    'DataValidationStep',
]