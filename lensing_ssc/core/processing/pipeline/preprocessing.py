"""
Preprocessing pipeline for mass sheet data processing.

This module implements the specialized preprocessing pipeline for converting
raw simulation data to processed mass sheets. It handles the complete workflow
from data discovery and validation through coordinate transformation and
mass sheet generation with comprehensive error handling and progress tracking.
"""

import os
import time
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from .base_pipeline import (
    BasePipeline, 
    PipelineResult, 
    PipelineError, 
    PipelineState,
    pipeline_step
)
from ...base.exceptions import (
    ProcessingError, 
    ValidationError, 
    ConfigurationError,
    DataError
)
from ...base.validation import (
    PathValidator, 
    RangeValidator,
    validate_not_none,
    validate_positive
)
from ...base.data_structures import MapData
from ...config.settings import ProcessingConfig
from ...providers.factory import get_provider


@dataclass
class PreprocessingMetrics:
    """Container for preprocessing performance metrics."""
    
    total_sheets: int = 0
    processed_sheets: int = 0
    failed_sheets: int = 0
    skipped_sheets: int = 0
    total_processing_time: float = 0.0
    average_sheet_time: float = 0.0
    peak_memory_mb: float = 0.0
    io_read_time: float = 0.0
    io_write_time: float = 0.0
    coordinate_transform_time: float = 0.0
    validation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_sheets': self.total_sheets,
            'processed_sheets': self.processed_sheets,
            'failed_sheets': self.failed_sheets,
            'skipped_sheets': self.skipped_sheets,
            'success_rate': self.processed_sheets / max(self.total_sheets, 1),
            'total_processing_time': self.total_processing_time,
            'average_sheet_time': self.average_sheet_time,
            'peak_memory_mb': self.peak_memory_mb,
            'throughput_sheets_per_hour': self.processed_sheets / max(self.total_processing_time / 3600, 1e-6),
            'timing_breakdown': {
                'io_read_time': self.io_read_time,
                'io_write_time': self.io_write_time,
                'coordinate_transform_time': self.coordinate_transform_time,
                'validation_time': self.validation_time,
            }
        }


class PreprocessingPipeline(BasePipeline):
    """Specialized pipeline for mass sheet preprocessing.
    
    This pipeline handles the complete preprocessing workflow:
    1. Data discovery and validation
    2. Input file processing and coordinate transformation
    3. Mass sheet generation and binning
    4. Quality control and output validation
    5. Result aggregation and reporting
    
    The pipeline supports:
    - Parallel processing of multiple sheets
    - Incremental processing with resume capability
    - Memory-efficient processing for large datasets
    - Comprehensive validation and quality control
    - Detailed progress tracking and performance metrics
    """
    
    def __init__(self, config: ProcessingConfig, **kwargs):
        """Initialize preprocessing pipeline.
        
        Parameters
        ----------
        config : ProcessingConfig
            Configuration object with preprocessing parameters
        **kwargs
            Additional arguments passed to BasePipeline
        """
        # Set preprocessing-specific defaults
        kwargs.setdefault('checkpoint_interval', 10)
        kwargs.setdefault('max_retries', 2)
        kwargs.setdefault('retry_delay', 2.0)
        
        super().__init__(config, **kwargs)
        
        # Preprocessing-specific state
        self.metrics = PreprocessingMetrics()
        self.input_files: List[Path] = []
        self.output_files: List[Path] = []
        self.failed_files: List[Tuple[Path, Exception]] = []
        
        # Providers
        self.healpix_provider = None
        self.nbodykit_provider = None
        
        # Validation
        self.path_validator = PathValidator()
        self.range_validator = RangeValidator()
        
        # Processing parameters
        self.sheet_range = config.sheet_range
        self.extra_index = getattr(config, 'extra_index', 100)
        self.chunk_size = config.chunk_size
        self.num_workers = config.num_workers
        
        # Output directory setup
        self.output_dir = config.output_dir / "mass_sheets"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self) -> None:
        """Validate preprocessing-specific configuration."""
        errors = []
        
        # Validate sheet range
        if not isinstance(self.sheet_range, (tuple, list)) or len(self.sheet_range) != 2:
            errors.append("sheet_range must be a tuple/list of length 2")
        elif self.sheet_range[0] >= self.sheet_range[1]:
            errors.append("sheet_range start must be less than end")
        elif self.sheet_range[0] < 0:
            errors.append("sheet_range values must be non-negative")
        
        # Validate directories
        if not self.config.data_dir.exists():
            errors.append(f"Data directory does not exist: {self.config.data_dir}")
        
        usmesh_dir = self.config.data_dir / "usmesh"
        if not usmesh_dir.exists():
            errors.append(f"Required usmesh directory not found: {usmesh_dir}")
        
        # Validate processing parameters
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.num_workers <= 0:
            errors.append("num_workers must be positive")
        
        if hasattr(self.config, 'extra_index') and self.extra_index < 0:
            errors.append("extra_index must be non-negative")
        
        if errors:
            raise ConfigurationError(f"Preprocessing configuration invalid: {'; '.join(errors)}")
    
    def _create_steps(self) -> None:
        """Create preprocessing pipeline steps."""
        self._add_step(self._step_initialize_providers, "Initialize Providers")
        self._add_step(self._step_discover_input_files, "Discover Input Files")
        self._add_step(self._step_validate_input_data, "Validate Input Data")
        self._add_step(self._step_setup_output_structure, "Setup Output Structure")
        self._add_step(self._step_process_mass_sheets, "Process Mass Sheets")
        self._add_step(self._step_validate_outputs, "Validate Outputs")
        self._add_step(self._step_generate_summary, "Generate Summary")
    
    def _execute_step(self, step_index: int, input_data: Any) -> Any:
        """Execute a single preprocessing step."""
        step_func = self._steps[step_index]
        step_name = self._step_names[step_index]
        
        self.context.logger.info(f"Executing step: {step_name}")
        
        try:
            result = step_func(input_data)
            self.context.logger.debug(f"Step '{step_name}' completed successfully")
            return result
        except Exception as e:
            self.context.logger.error(f"Step '{step_name}' failed: {e}")
            raise PipelineError(
                f"Preprocessing step '{step_name}' failed: {e}",
                pipeline_state=self.context.state,
                current_step=step_name,
                step_index=step_index
            ) from e
    
    @pipeline_step("Initialize Providers")
    def _step_initialize_providers(self, input_data: Any) -> Any:
        """Initialize required providers for preprocessing."""
        try:
            # Initialize HEALPix provider for map operations
            self.healpix_provider = get_provider('healpix')
            if not self.healpix_provider.is_available():
                raise ProviderError("HEALPix provider not available - required for preprocessing")
            
            self.context.logger.info("HEALPix provider initialized successfully")
            
            # Initialize NBBodyKit provider for catalog operations (optional)
            try:
                self.nbodykit_provider = get_provider('nbodykit')
                if self.nbodykit_provider.is_available():
                    self.context.logger.info("NBBodyKit provider initialized successfully")
                else:
                    self.context.logger.warning("NBBodyKit provider not available - some features may be limited")
                    self.nbodykit_provider = None
            except Exception as e:
                self.context.logger.warning(f"NBBodyKit provider initialization failed: {e}")
                self.nbodykit_provider = None
            
            return input_data
            
        except Exception as e:
            raise ProcessingError(f"Provider initialization failed: {e}")
    
    @pipeline_step("Discover Input Files")
    def _step_discover_input_files(self, input_data: Any) -> Dict[str, Any]:
        """Discover and catalog input files for processing."""
        usmesh_dir = self.config.data_dir / "usmesh"
        
        try:
            # Find all usmesh files
            all_usmesh_files = sorted(list(usmesh_dir.glob("usmesh_*.bin")))
            
            if not all_usmesh_files:
                raise DataError(f"No usmesh files found in {usmesh_dir}")
            
            # Filter by sheet range
            self.input_files = []
            sheet_start, sheet_end = self.sheet_range
            
            for file_path in all_usmesh_files:
                try:
                    # Extract sheet number from filename
                    filename = file_path.stem
                    if filename.startswith("usmesh_"):
                        sheet_num_str = filename[7:]  # Remove "usmesh_" prefix
                        sheet_num = int(sheet_num_str)
                        
                        if sheet_start <= sheet_num < sheet_end:
                            self.input_files.append(file_path)
                            
                except (ValueError, IndexError) as e:
                    self.context.logger.warning(f"Could not parse sheet number from {file_path}: {e}")
                    continue
            
            if not self.input_files:
                raise DataError(
                    f"No usmesh files found in range [{sheet_start}, {sheet_end}) "
                    f"out of {len(all_usmesh_files)} total files"
                )
            
            self.metrics.total_sheets = len(self.input_files)
            
            self.context.logger.info(
                f"Discovered {len(self.input_files)} input files "
                f"(range: {sheet_start}-{sheet_end-1})"
            )
            
            return {
                'input_files': self.input_files,
                'total_files': len(all_usmesh_files),
                'filtered_files': len(self.input_files),
                'sheet_range': self.sheet_range
            }
            
        except Exception as e:
            raise ProcessingError(f"Input file discovery failed: {e}")
    
    @pipeline_step("Validate Input Data") 
    def _step_validate_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input files and data structure."""
        validation_start = time.time()
        
        try:
            valid_files = []
            invalid_files = []
            
            for file_path in self.input_files:
                try:
                    # Basic file validation
                    if not self.path_validator.validate_file(
                        file_path, 
                        extensions=['.bin'],
                        min_size=1024  # Minimum 1KB
                    ):
                        errors = self.path_validator.get_errors()
                        invalid_files.append((file_path, f"File validation failed: {'; '.join(errors)}"))
                        continue
                    
                    # Check file accessibility
                    with open(file_path, 'rb') as f:
                        # Try to read first few bytes
                        header = f.read(32)
                        if len(header) < 32:
                            invalid_files.append((file_path, "File too small or corrupted"))
                            continue
                    
                    valid_files.append(file_path)
                    
                except Exception as e:
                    invalid_files.append((file_path, str(e)))
            
            if invalid_files:
                self.context.logger.warning(f"Found {len(invalid_files)} invalid files:")
                for file_path, error in invalid_files:
                    self.context.logger.warning(f"  {file_path}: {error}")
            
            if not valid_files:
                raise ValidationError("No valid input files found")
            
            self.input_files = valid_files
            self.metrics.total_sheets = len(valid_files)
            
            validation_time = time.time() - validation_start
            self.metrics.validation_time += validation_time
            
            self.context.logger.info(
                f"Validated {len(valid_files)} files "
                f"({len(invalid_files)} invalid) in {validation_time:.2f}s"
            )
            
            return {
                **input_data,
                'valid_files': valid_files,
                'invalid_files': invalid_files,
                'validation_time': validation_time
            }
            
        except Exception as e:
            raise ValidationError(f"Input data validation failed: {e}")
    
    @pipeline_step("Setup Output Structure")
    def _step_setup_output_structure(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Setup output directory structure and files."""
        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plan output files
            self.output_files = []
            for input_file in self.input_files:
                # Extract sheet number
                filename = input_file.stem
                sheet_num_str = filename[7:]  # Remove "usmesh_" prefix
                
                # Create output filename
                output_filename = f"delta-sheet-{sheet_num_str}.fits"
                output_path = self.output_dir / output_filename
                
                # Check if we should skip existing files
                if output_path.exists() and not self.config.overwrite:
                    self.context.logger.debug(f"Output file exists, will skip: {output_path}")
                    self.metrics.skipped_sheets += 1
                
                self.output_files.append(output_path)
            
            # Filter out files to skip if not overwriting
            if not self.config.overwrite:
                files_to_process = []
                outputs_to_create = []
                
                for inp_file, out_file in zip(self.input_files, self.output_files):
                    if not out_file.exists():
                        files_to_process.append(inp_file)
                        outputs_to_create.append(out_file)
                
                self.input_files = files_to_process
                self.output_files = outputs_to_create
                self.metrics.total_sheets = len(files_to_process)
            
            self.context.logger.info(
                f"Setup output structure: {len(self.output_files)} files to process, "
                f"{self.metrics.skipped_sheets} files to skip"
            )
            
            return {
                **input_data,
                'output_files': self.output_files,
                'skipped_sheets': self.metrics.skipped_sheets
            }
            
        except Exception as e:
            raise ProcessingError(f"Output structure setup failed: {e}")
    
    @pipeline_step("Process Mass Sheets")
    def _step_process_mass_sheets(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process mass sheets in parallel."""
        processing_start = time.time()
        
        if not self.input_files:
            self.context.logger.warning("No files to process")
            return {**input_data, 'processing_results': []}
        
        try:
            # Process files in parallel
            processing_results = []
            
            if self.num_workers == 1:
                # Sequential processing
                for i, (input_file, output_file) in enumerate(zip(self.input_files, self.output_files)):
                    if self.context.is_cancelled():
                        break
                    
                    result = self._process_single_sheet(input_file, output_file, i)
                    processing_results.append(result)
                    
                    # Update progress
                    progress = (i + 1) / len(self.input_files) * 100
                    self.context.update_progress(f"Processing sheet {i+1}/{len(self.input_files)}", progress)
            
            else:
                # Parallel processing
                processing_results = self._process_sheets_parallel()
            
            # Aggregate results
            successful_results = [r for r in processing_results if r['success']]
            failed_results = [r for r in processing_results if not r['success']]
            
            self.metrics.processed_sheets = len(successful_results)
            self.metrics.failed_sheets = len(failed_results)
            self.metrics.total_processing_time = time.time() - processing_start
            
            if successful_results:
                # Calculate timing statistics
                sheet_times = [r['processing_time'] for r in successful_results]
                self.metrics.average_sheet_time = np.mean(sheet_times)
                
                # Aggregate timing breakdowns
                self.metrics.io_read_time = sum(r.get('io_read_time', 0) for r in successful_results)
                self.metrics.io_write_time = sum(r.get('io_write_time', 0) for r in successful_results)
                self.metrics.coordinate_transform_time = sum(r.get('transform_time', 0) for r in successful_results)
            
            # Store failed files for reporting
            self.failed_files = [(r['input_file'], r['error']) for r in failed_results]
            
            self.context.logger.info(
                f"Mass sheet processing completed: {self.metrics.processed_sheets} successful, "
                f"{self.metrics.failed_sheets} failed in {self.metrics.total_processing_time:.2f}s"
            )
            
            return {
                **input_data,
                'processing_results': processing_results,
                'successful_sheets': self.metrics.processed_sheets,
                'failed_sheets': self.metrics.failed_sheets
            }
            
        except Exception as e:
            raise ProcessingError(f"Mass sheet processing failed: {e}")
    
    def _process_sheets_parallel(self) -> List[Dict[str, Any]]:
        """Process sheets using parallel workers."""
        results = []
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(
                        _process_sheet_worker,
                        input_file,
                        output_file,
                        i,
                        self.chunk_size,
                        self.extra_index
                    ): i
                    for i, (input_file, output_file) in enumerate(zip(self.input_files, self.output_files))
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_index):
                    if self.context.is_cancelled():
                        # Cancel remaining tasks
                        for f in future_to_index:
                            f.cancel()
                        break
                    
                    index = future_to_index[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        # Create error result
                        error_result = {
                            'index': index,
                            'input_file': self.input_files[index],
                            'output_file': self.output_files[index],
                            'success': False,
                            'error': e,
                            'processing_time': 0.0
                        }
                        results.append(error_result)
                        self.context.logger.error(f"Sheet {index} processing failed: {e}")
                    
                    completed += 1
                    progress = completed / len(self.input_files) * 100
                    self.context.update_progress(f"Processed sheet {completed}/{len(self.input_files)}", progress)
            
            # Sort results by index to maintain order
            results.sort(key=lambda x: x['index'])
            return results
            
        except Exception as e:
            raise ProcessingError(f"Parallel processing failed: {e}")
    
    def _process_single_sheet(self, input_file: Path, output_file: Path, index: int) -> Dict[str, Any]:
        """Process a single mass sheet file."""
        sheet_start = time.time()
        
        result = {
            'index': index,
            'input_file': input_file,
            'output_file': output_file,
            'success': False,
            'error': None,
            'processing_time': 0.0,
            'io_read_time': 0.0,
            'io_write_time': 0.0,
            'transform_time': 0.0,
            'output_size': 0,
        }
        
        try:
            # Read input data
            read_start = time.time()
            sheet_data = self._read_usmesh_file(input_file)
            result['io_read_time'] = time.time() - read_start
            
            # Transform coordinates and create mass sheet
            transform_start = time.time()
            mass_sheet = self._transform_to_mass_sheet(sheet_data)
            result['transform_time'] = time.time() - transform_start
            
            # Write output
            write_start = time.time()
            self._write_mass_sheet(mass_sheet, output_file)
            result['io_write_time'] = time.time() - write_start
            
            # Get output file size
            result['output_size'] = output_file.stat().st_size
            result['success'] = True
            
            self.context.logger.debug(f"Successfully processed sheet {index}: {input_file.name}")
            
        except Exception as e:
            result['error'] = e
            self.context.logger.error(f"Failed to process sheet {index} ({input_file.name}): {e}")
        
        result['processing_time'] = time.time() - sheet_start
        return result
    
    def _read_usmesh_file(self, file_path: Path) -> np.ndarray:
        """Read usmesh binary file."""
        try:
            # This is a simplified implementation
            # In practice, would need to handle specific usmesh format
            with open(file_path, 'rb') as f:
                # Read file size and determine array shape
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                f.seek(0)  # Seek to beginning
                
                # Assume float32 data format
                n_elements = file_size // 4
                data = np.frombuffer(f.read(), dtype=np.float32)
                
                if len(data) != n_elements:
                    raise DataError(f"File size mismatch in {file_path}")
                
                return data
                
        except Exception as e:
            raise DataError(f"Failed to read usmesh file {file_path}: {e}")
    
    def _transform_to_mass_sheet(self, raw_data: np.ndarray) -> np.ndarray:
        """Transform raw data to mass sheet format."""
        try:
            # This is a simplified implementation
            # In practice, would involve coordinate transformations,
            # binning, and mass assignment
            
            # Example: reshape and normalize
            n_particles = len(raw_data) // 6  # Assuming 6 values per particle (x,y,z,vx,vy,vz)
            
            if len(raw_data) % 6 != 0:
                raise DataError("Invalid data format: not divisible by 6")
            
            particles = raw_data.reshape(n_particles, 6)
            positions = particles[:, :3]  # x, y, z coordinates
            
            # Create a simple mass sheet (in practice would use proper gridding)
            # For demonstration, create a small grid
            grid_size = 64
            mass_sheet = np.histogram2d(
                positions[:, 0], 
                positions[:, 1], 
                bins=grid_size,
                range=[[-50, 50], [-50, 50]]  # Example coordinate range
            )[0]
            
            # Flatten to 1D for HEALPix format
            mass_sheet_1d = mass_sheet.flatten()
            
            return mass_sheet_1d.astype(np.float64)
            
        except Exception as e:
            raise ProcessingError(f"Mass sheet transformation failed: {e}")
    
    def _write_mass_sheet(self, mass_sheet: np.ndarray, output_file: Path) -> None:
        """Write mass sheet to FITS file."""
        try:
            # Create MapData object
            map_data = MapData(
                data=mass_sheet,
                shape=mass_sheet.shape,
                dtype=mass_sheet.dtype,
                metadata={
                    'NSIDE': 64,  # Example NSIDE
                    'ORDERING': 'RING',
                    'COORDSYS': 'C',
                    'UNITS': 'dimensionless',
                    'CREATOR': 'LensingSSC-PreprocessingPipeline',
                    'DATE': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            )
            
            # Write using HEALPix provider
            self.healpix_provider.write_map(map_data, output_file, overwrite=True)
            
        except Exception as e:
            raise ProcessingError(f"Failed to write mass sheet to {output_file}: {e}")
    
    @pipeline_step("Validate Outputs")
    def _step_validate_outputs(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated output files."""
        validation_start = time.time()
        
        try:
            valid_outputs = []
            invalid_outputs = []
            
            for output_file in self.output_files:
                if not output_file.exists():
                    invalid_outputs.append((output_file, "File not created"))
                    continue
                
                try:
                    # Validate FITS file structure
                    map_data = self.healpix_provider.read_map(output_file)
                    
                    # Basic validation
                    if map_data.size == 0:
                        invalid_outputs.append((output_file, "Empty map"))
                        continue
                    
                    if not np.any(np.isfinite(map_data.data)):
                        invalid_outputs.append((output_file, "No finite values"))
                        continue
                    
                    valid_outputs.append(output_file)
                    
                except Exception as e:
                    invalid_outputs.append((output_file, str(e)))
            
            validation_time = time.time() - validation_start
            self.metrics.validation_time += validation_time
            
            self.context.logger.info(
                f"Output validation completed: {len(valid_outputs)} valid, "
                f"{len(invalid_outputs)} invalid in {validation_time:.2f}s"
            )
            
            if invalid_outputs:
                self.context.logger.warning("Invalid outputs found:")
                for output_file, error in invalid_outputs:
                    self.context.logger.warning(f"  {output_file}: {error}")
            
            return {
                **input_data,
                'valid_outputs': valid_outputs,
                'invalid_outputs': invalid_outputs,
                'output_validation_time': validation_time
            }
            
        except Exception as e:
            raise ValidationError(f"Output validation failed: {e}")
    
    @pipeline_step("Generate Summary")
    def _step_generate_summary(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate processing summary and final report."""
        try:
            # Create comprehensive summary
            summary = {
                'preprocessing_metrics': self.metrics.to_dict(),
                'file_summary': {
                    'total_discovered': len(self.input_files) + self.metrics.skipped_sheets,
                    'processed': self.metrics.processed_sheets,
                    'failed': self.metrics.failed_sheets,
                    'skipped': self.metrics.skipped_sheets,
                },
                'timing_summary': {
                    'total_time': self.metrics.total_processing_time,
                    'average_per_sheet': self.metrics.average_sheet_time,
                    'validation_time': self.metrics.validation_time,
                },
                'failed_files': [
                    {'file': str(file_path), 'error': str(error)}
                    for file_path, error in self.failed_files
                ],
                'output_directory': str(self.output_dir),
                'configuration': {
                    'sheet_range': self.sheet_range,
                    'chunk_size': self.chunk_size,
                    'num_workers': self.num_workers,
                    'overwrite': self.config.overwrite,
                }
            }
            
            # Write summary to file
            summary_file = self.output_dir / "preprocessing_summary.json"
            try:
                import json
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                self.context.logger.info(f"Processing summary written to: {summary_file}")
            except Exception as e:
                self.context.logger.warning(f"Could not write summary file: {e}")
            
            self.context.logger.info(
                f"Preprocessing pipeline completed successfully:\n"
                f"  Processed: {self.metrics.processed_sheets} sheets\n"
                f"  Failed: {self.metrics.failed_sheets} sheets\n"
                f"  Skipped: {self.metrics.skipped_sheets} sheets\n"
                f"  Success rate: {self.metrics.processed_sheets / max(self.metrics.total_sheets, 1):.1%}\n"
                f"  Total time: {self.metrics.total_processing_time:.2f}s"
            )
            
            return {
                **input_data,
                'summary': summary,
                'summary_file': summary_file
            }
            
        except Exception as e:
            raise ProcessingError(f"Summary generation failed: {e}")


def _process_sheet_worker(input_file: Path, output_file: Path, index: int, 
                         chunk_size: int, extra_index: int) -> Dict[str, Any]:
    """Worker function for parallel processing of mass sheets.
    
    This function is designed to be run in a separate process and handles
    the complete processing of a single mass sheet file.
    
    Parameters
    ----------
    input_file : Path
        Path to input usmesh file
    output_file : Path
        Path to output FITS file
    index : int
        Processing index for tracking
    chunk_size : int
        Chunk size for memory management
    extra_index : int
        Extra index parameter for processing
        
    Returns
    -------
    Dict[str, Any]
        Processing result dictionary
    """
    import time
    import numpy as np
    from pathlib import Path
    
    sheet_start = time.time()
    
    result = {
        'index': index,
        'input_file': input_file,
        'output_file': output_file,
        'success': False,
        'error': None,
        'processing_time': 0.0,
        'io_read_time': 0.0,
        'io_write_time': 0.0,
        'transform_time': 0.0,
        'output_size': 0,
        'memory_peak': 0.0,
    }
    
    try:
        # Monitor memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Read input data
        read_start = time.time()
        sheet_data = _read_usmesh_worker(input_file, chunk_size)
        result['io_read_time'] = time.time() - read_start
        
        # Monitor memory after read
        post_read_memory = process.memory_info().rss / 1024 / 1024
        
        # Transform to mass sheet
        transform_start = time.time()
        mass_sheet = _transform_worker(sheet_data, extra_index)
        result['transform_time'] = time.time() - transform_start
        
        # Monitor memory after transform
        post_transform_memory = process.memory_info().rss / 1024 / 1024
        
        # Write output
        write_start = time.time()
        _write_fits_worker(mass_sheet, output_file)
        result['io_write_time'] = time.time() - write_start
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024
        result['memory_peak'] = max(initial_memory, post_read_memory, post_transform_memory, final_memory)
        
        # Get output file size
        if output_file.exists():
            result['output_size'] = output_file.stat().st_size
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = e
    
    result['processing_time'] = time.time() - sheet_start
    return result


def _read_usmesh_worker(file_path: Path, chunk_size: int) -> np.ndarray:
    """Worker function to read usmesh file with memory management."""
    try:
        with open(file_path, 'rb') as f:
            # Get file size
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(0)
            
            # Calculate number of elements
            n_elements = file_size // 4  # Assuming float32
            
            if n_elements == 0:
                raise ValueError("Empty file")
            
            # Read in chunks if file is large
            if n_elements > chunk_size:
                data_chunks = []
                elements_read = 0
                
                while elements_read < n_elements:
                    elements_to_read = min(chunk_size, n_elements - elements_read)
                    bytes_to_read = elements_to_read * 4
                    
                    chunk_data = f.read(bytes_to_read)
                    if len(chunk_data) != bytes_to_read:
                        raise ValueError("Unexpected end of file")
                    
                    chunk_array = np.frombuffer(chunk_data, dtype=np.float32)
                    data_chunks.append(chunk_array)
                    elements_read += elements_to_read
                
                data = np.concatenate(data_chunks)
            else:
                # Read entire file at once
                data = np.frombuffer(f.read(), dtype=np.float32)
            
            if len(data) != n_elements:
                raise ValueError("File size mismatch")
            
            return data
            
    except Exception as e:
        raise RuntimeError(f"Failed to read usmesh file {file_path}: {e}")


def _transform_worker(raw_data: np.ndarray, extra_index: int) -> np.ndarray:
    """Worker function to transform raw data to mass sheet."""
    try:
        # Validate input data
        if len(raw_data) == 0:
            raise ValueError("Empty input data")
        
        # Assume data format: [x, y, z, vx, vy, vz] per particle
        n_particles = len(raw_data) // 6
        if len(raw_data) % 6 != 0:
            raise ValueError("Invalid data format: length not divisible by 6")
        
        # Reshape to particle format
        particles = raw_data.reshape(n_particles, 6)
        positions = particles[:, :3]  # Extract positions
        
        # Apply coordinate transformations
        # This is a simplified example - real implementation would be more complex
        
        # Normalize coordinates to [0, 1] range
        for i in range(3):
            pos_min = np.min(positions[:, i])
            pos_max = np.max(positions[:, i])
            if pos_max > pos_min:
                positions[:, i] = (positions[:, i] - pos_min) / (pos_max - pos_min)
        
        # Create mass assignment grid
        # For HEALPix format, we need to project to sphere
        grid_size = 256  # Example grid size
        
        # Simple 2D projection for demonstration
        hist, _, _ = np.histogram2d(
            positions[:, 0], 
            positions[:, 1],
            bins=grid_size,
            range=[[0, 1], [0, 1]]
        )
        
        # Apply extra index transformation if needed
        if extra_index > 0:
            hist = hist + extra_index * np.random.normal(0, 0.01, hist.shape)
        
        # Convert to proper HEALPix map format
        # This is simplified - real implementation would properly handle spherical geometry
        nside = 512  # Example NSIDE
        npix = 12 * nside * nside
        
        # Flatten and resize to correct HEALPix size
        flat_hist = hist.flatten()
        if len(flat_hist) < npix:
            # Pad with zeros
            mass_sheet = np.zeros(npix, dtype=np.float64)
            mass_sheet[:len(flat_hist)] = flat_hist
        else:
            # Truncate or downsample
            mass_sheet = flat_hist[:npix].astype(np.float64)
        
        # Normalize to physical units (simplified)
        mass_sheet = mass_sheet / np.mean(mass_sheet[mass_sheet > 0]) - 1.0
        
        return mass_sheet
        
    except Exception as e:
        raise RuntimeError(f"Mass sheet transformation failed: {e}")


def _write_fits_worker(mass_sheet: np.ndarray, output_file: Path) -> None:
    """Worker function to write mass sheet to FITS file."""
    try:
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use a simple FITS writer since we can't import complex providers in worker
        # This is a simplified implementation
        try:
            import astropy.io.fits as fits
            
            # Create HDU
            hdu = fits.PrimaryHDU(mass_sheet)
            
            # Add headers
            hdu.header['NSIDE'] = 512
            hdu.header['ORDERING'] = 'RING'
            hdu.header['COORDSYS'] = 'C'
            hdu.header['UNITS'] = 'dimensionless'
            hdu.header['CREATOR'] = 'LensingSSC-PreprocessingPipeline'
            hdu.header['EXTNAME'] = 'xtension'
            
            # Write file
            hdu.writeto(output_file, overwrite=True)
            
        except ImportError:
            # Fallback to basic binary write if astropy not available
            with open(output_file, 'wb') as f:
                # Write a minimal FITS-like header
                header = b'SIMPLE  =                    T / file does conform to FITS standard             '
                header += b'BITPIX  =                  -64 / number of bits per data pixel                '
                header += b'NAXIS   =                    1 / number of data axes                          '
                header += b'NAXIS1  =            %8d / length of data axis 1                       ' % len(mass_sheet)
                header += b'EXTEND  =                    T / FITS dataset may contain extensions         '
                header += b'END' + b' ' * 77
                
                # Pad header to multiple of 2880 bytes
                header_size = len(header)
                padding_needed = 2880 - (header_size % 2880)
                if padding_needed != 2880:
                    header += b' ' * padding_needed
                
                f.write(header)
                
                # Write data
                mass_sheet.astype('>f8').tobytes()  # Big-endian float64
                f.write(mass_sheet.astype('>f8').tobytes())
                
                # Pad data to multiple of 2880 bytes
                data_size = len(mass_sheet) * 8
                padding_needed = 2880 - (data_size % 2880)
                if padding_needed != 2880:
                    f.write(b'\x00' * padding_needed)
        
    except Exception as e:
        raise RuntimeError(f"Failed to write FITS file {output_file}: {e}")


# Additional utility functions for preprocessing pipeline

class PreprocessingMonitor:
    """Monitor for preprocessing pipeline with resource tracking."""
    
    def __init__(self, pipeline: PreprocessingPipeline):
        """Initialize monitor.
        
        Parameters
        ----------
        pipeline : PreprocessingPipeline
            Pipeline to monitor
        """
        self.pipeline = pipeline
        self.start_time = None
        self.monitoring_active = False
        self._memory_samples = []
        self._cpu_samples = []
        
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        self.start_time = time.time()
        self.monitoring_active = True
        self._memory_samples = []
        self._cpu_samples = []
        
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary.
        
        Returns
        -------
        Dict[str, Any]
            Monitoring summary
        """
        self.monitoring_active = False
        end_time = time.time()
        
        return {
            'total_time': end_time - self.start_time if self.start_time else 0,
            'peak_memory_mb': max(self._memory_samples) if self._memory_samples else 0,
            'average_memory_mb': np.mean(self._memory_samples) if self._memory_samples else 0,
            'peak_cpu_percent': max(self._cpu_samples) if self._cpu_samples else 0,
            'average_cpu_percent': np.mean(self._cpu_samples) if self._cpu_samples else 0,
            'memory_samples': len(self._memory_samples),
            'cpu_samples': len(self._cpu_samples),
        }
    
    def sample_resources(self) -> None:
        """Sample current resource usage."""
        if not self.monitoring_active:
            return
        
        try:
            import psutil
            
            # Memory usage
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self._memory_samples.append(memory_mb)
            
            # CPU usage
            cpu_percent = psutil.Process().cpu_percent()
            self._cpu_samples.append(cpu_percent)
            
            # Update pipeline metrics
            if memory_mb > self.pipeline.metrics.peak_memory_mb:
                self.pipeline.metrics.peak_memory_mb = memory_mb
                
        except ImportError:
            # psutil not available, skip resource monitoring
            pass
        except Exception:
            # Error during monitoring, continue silently
            pass


def validate_preprocessing_config(config: ProcessingConfig) -> List[str]:
    """Validate preprocessing configuration and return list of issues.
    
    Parameters
    ----------
    config : ProcessingConfig
        Configuration to validate
        
    Returns
    -------
    List[str]
        List of validation issues (empty if valid)
    """
    issues = []
    
    # Check required directories
    if not config.data_dir.exists():
        issues.append(f"Data directory does not exist: {config.data_dir}")
    else:
        usmesh_dir = config.data_dir / "usmesh"
        if not usmesh_dir.exists():
            issues.append(f"Required usmesh directory not found: {usmesh_dir}")
    
    # Check output directory writability
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        test_file = config.output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        issues.append(f"Output directory not writable: {config.output_dir} ({e})")
    
    # Check sheet range
    if hasattr(config, 'sheet_range'):
        sheet_range = config.sheet_range
        if not isinstance(sheet_range, (tuple, list)) or len(sheet_range) != 2:
            issues.append("sheet_range must be a tuple/list of length 2")
        elif sheet_range[0] >= sheet_range[1]:
            issues.append("sheet_range start must be less than end")
        elif sheet_range[0] < 0:
            issues.append("sheet_range values must be non-negative")
    
    # Check processing parameters
    if config.chunk_size <= 0:
        issues.append("chunk_size must be positive")
    
    if config.num_workers <= 0:
        issues.append("num_workers must be positive")
    
    # Check memory limits
    if hasattr(config, 'memory_limit_mb') and config.memory_limit_mb is not None:
        if config.memory_limit_mb <= 0:
            issues.append("memory_limit_mb must be positive")
        elif config.memory_limit_mb < 1024:
            issues.append("memory_limit_mb should be at least 1024 MB for preprocessing")
    
    return issues


def estimate_preprocessing_requirements(config: ProcessingConfig) -> Dict[str, Any]:
    """Estimate resource requirements for preprocessing.
    
    Parameters
    ----------
    config : ProcessingConfig
        Configuration to analyze
        
    Returns
    -------
    Dict[str, Any]
        Resource requirement estimates
    """
    estimates = {
        'processing_time_hours': 0.0,
        'memory_required_mb': 0.0,
        'disk_space_required_gb': 0.0,
        'recommended_workers': 1,
        'warnings': [],
    }
    
    try:
        # Count input files
        usmesh_dir = config.data_dir / "usmesh"
        if usmesh_dir.exists():
            usmesh_files = list(usmesh_dir.glob("usmesh_*.bin"))
            
            # Filter by sheet range if specified
            if hasattr(config, 'sheet_range'):
                sheet_start, sheet_end = config.sheet_range
                filtered_files = []
                for file_path in usmesh_files:
                    try:
                        filename = file_path.stem
                        if filename.startswith("usmesh_"):
                            sheet_num = int(filename[7:])
                            if sheet_start <= sheet_num < sheet_end:
                                filtered_files.append(file_path)
                    except ValueError:
                        continue
                usmesh_files = filtered_files
            
            n_files = len(usmesh_files)
            
            if n_files > 0:
                # Estimate based on typical file sizes
                sample_size = min(5, n_files)
                sample_files = usmesh_files[:sample_size]
                
                total_sample_size = sum(f.stat().st_size for f in sample_files)
                avg_file_size_mb = (total_sample_size / sample_size) / (1024 * 1024)
                
                # Processing time estimation (rough)
                # Assume ~1-5 seconds per MB depending on complexity
                time_per_file = max(1.0, avg_file_size_mb * 2.0)  # seconds
                total_time_seq = n_files * time_per_file
                
                # Parallel speedup estimation
                recommended_workers = min(config.num_workers, max(1, n_files // 4))
                parallel_efficiency = 0.8  # Assume 80% efficiency
                total_time_parallel = total_time_seq / (recommended_workers * parallel_efficiency)
                
                estimates['processing_time_hours'] = total_time_parallel / 3600
                estimates['recommended_workers'] = recommended_workers
                
                # Memory estimation
                # Peak memory per worker + overhead
                memory_per_worker = max(512, avg_file_size_mb * 3)  # 3x file size
                estimates['memory_required_mb'] = memory_per_worker * recommended_workers + 1024
                
                # Disk space estimation
                # Output files are typically similar size to input
                estimates['disk_space_required_gb'] = (total_sample_size * n_files / sample_size) / (1024**3)
                
                # Generate warnings
                if estimates['memory_required_mb'] > 16384:  # > 16 GB
                    estimates['warnings'].append("High memory usage expected - consider reducing workers or file batch size")
                
                if estimates['processing_time_hours'] > 24:  # > 1 day
                    estimates['warnings'].append("Long processing time expected - consider increasing workers")
                
                if estimates['disk_space_required_gb'] > 100:  # > 100 GB
                    estimates['warnings'].append("Large disk space required - ensure sufficient storage")
            
    except Exception as e:
        estimates['warnings'].append(f"Could not estimate requirements: {e}")
    
    return estimates


# Legacy compatibility
MassSheetPreprocessingPipeline = PreprocessingPipeline