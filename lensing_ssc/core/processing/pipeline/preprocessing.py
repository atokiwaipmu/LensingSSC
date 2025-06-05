# lensing_ssc/processing/pipeline/preprocessing.py
"""
Preprocessing pipeline for mass sheet data.

This module implements the preprocessing workflow that transforms raw N-body
simulation data into processed mass sheets ready for kappa map generation.
The pipeline handles the complete workflow from index finding through mass
sheet processing with robust error handling and checkpoint support.

Pipeline Steps:
1. Data Discovery: Find and validate input data files
2. Index Finding: Determine processing indices for mass sheets
3. Mass Sheet Processing: Extract and process density contrast maps
4. Validation: Verify output quality and completeness
5. Cleanup: Remove temporary files and organize outputs

The pipeline supports:
- Resume from checkpoints for long-running jobs
- Resource monitoring and memory management
- Parallel processing of multiple sheets
- Comprehensive validation and error recovery

Usage:
    from lensing_ssc.processing.pipeline import PreprocessingPipeline
    from lensing_ssc.config import ProcessingConfig
    
    config = ProcessingConfig(
        data_dir="/path/to/usmesh/data",
        output_dir="/path/to/output",
        overwrite=False
    )
    
    pipeline = PreprocessingPipeline(config)
    results = pipeline.run(
        resume_from_checkpoint=True,
        memory_limit_mb=8000
    )

Advanced Usage:
    # Custom step configuration
    pipeline = PreprocessingPipeline(config)
    
    # Modify steps before running
    validation_step = pipeline.get_step("validation")
    validation_step.strict_mode = True
    
    # Run with custom callbacks
    def on_sheet_processed(step_name, result):
        print(f"Processed sheet: {result.metadata.get('sheet_id')}")
    
    pipeline.on_step_complete = on_sheet_processed
    results = pipeline.run()
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import time
import pandas as pd
import numpy as np

from . import BasePipeline, ProcessingStep, StepResult, PipelineContext, StepStatus
from lensing_ssc.core.base import ValidationError, ProcessingError, ConfigurationError


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline.
    
    This is a lightweight wrapper that standardizes configuration
    access regardless of the input config object type.
    """
    data_dir: Path
    output_dir: Optional[Path] = None
    overwrite: bool = False
    sheet_range: Optional[Tuple[int, int]] = None
    seed: Optional[int] = None
    extra_index: int = 100
    validate_outputs: bool = True
    cleanup_temp: bool = True
    
    @classmethod
    def from_config(cls, config: Any) -> 'PreprocessingConfig':
        """Create from various config object types."""
        if isinstance(config, cls):
            return config
        
        # Extract common attributes
        data_dir = getattr(config, 'data_dir', getattr(config, 'datadir', None))
        if data_dir is None:
            raise ConfigurationError("data_dir or datadir must be specified in config")
        
        return cls(
            data_dir=Path(data_dir),
            output_dir=Path(getattr(config, 'output_dir', data_dir)) if hasattr(config, 'output_dir') else None,
            overwrite=getattr(config, 'overwrite', False),
            sheet_range=getattr(config, 'sheet_range', None),
            seed=getattr(config, 'seed', None),
            extra_index=getattr(config, 'extra_index', 100),
            validate_outputs=getattr(config, 'validate_outputs', True),
            cleanup_temp=getattr(config, 'cleanup_temp', True),
        )


class DataDiscoveryStep(ProcessingStep):
    """Discover and validate input data files."""
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute data discovery."""
        config = PreprocessingConfig.from_config(context.config)
        
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Check if data directory exists
            if not config.data_dir.exists():
                raise ValidationError(f"Data directory does not exist: {config.data_dir}")
            
            # Look for usmesh directory
            usmesh_dir = config.data_dir / "usmesh"
            if not usmesh_dir.exists():
                raise ValidationError(f"usmesh directory not found: {usmesh_dir}")
            
            # Validate usmesh data structure
            try:
                # Import here to avoid heavy dependency at module level
                from nbodykit.lab import BigFileCatalog
                
                msheets = BigFileCatalog(str(usmesh_dir), dataset="HEALPIX/")
                attrs = msheets.attrs
                
                # Extract key attributes
                seed = attrs.get('seed', [config.seed or 0])[0]
                nc = attrs['NC'][0]
                box_size = attrs['BoxSize'][0]
                aemit_edges = attrs['aemitIndex.edges']
                aemit_offset = attrs['aemitIndex.offset']
                
                result.data = {
                    'usmesh_dir': usmesh_dir,
                    'msheets_catalog': msheets,
                    'attrs': attrs
                }
                
                result.metadata = {
                    'seed': seed,
                    'nc': nc,
                    'box_size': box_size,
                    'n_sheets': len(aemit_edges) - 1,
                    'data_valid': True
                }
                
                self.logger.info(f"Data discovery completed: seed={seed}, n_sheets={result.metadata['n_sheets']}")
                
            except ImportError:
                raise ProcessingError("nbodykit is required for data discovery")
            except Exception as e:
                raise ProcessingError(f"Failed to validate usmesh data: {e}")
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result


class IndexFindingStep(ProcessingStep):
    """Find and save processing indices for mass sheets."""
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute index finding."""
        config = PreprocessingConfig.from_config(context.config)
        
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Get data from discovery step
            discovery_result = inputs.get('data_discovery')
            if not discovery_result or not discovery_result.is_successful():
                raise ProcessingError("Data discovery step failed or missing")
            
            msheets = discovery_result.data['msheets_catalog']
            attrs = discovery_result.data['attrs']
            seed = discovery_result.metadata['seed']
            
            # Determine sheet range
            max_sheets = discovery_result.metadata['n_sheets']
            if config.sheet_range:
                i_start, i_end = config.sheet_range
                i_end = min(i_end, max_sheets)
            else:
                i_start, i_end = 20, min(100, max_sheets)
            
            # Create indices finder
            indices_finder = self._create_indices_finder(
                config.data_dir, msheets, attrs, seed, config.extra_index
            )
            
            # Find indices
            indices = self._find_indices(indices_finder, i_start, i_end)
            
            # Save indices
            indices_file = config.data_dir / f"preproc_s{seed}_indices.csv"
            if indices:
                indices_df = pd.DataFrame(indices)
                indices_df.to_csv(indices_file, index=False)
                self.logger.info(f"Saved {len(indices)} indices to {indices_file}")
            else:
                self.logger.warning("No valid indices found")
                indices_df = pd.DataFrame(columns=['sheet', 'start', 'end'])
            
            result.data = {
                'indices_file': indices_file,
                'indices_df': indices_df,
                'sheet_range': (i_start, i_end)
            }
            
            result.metadata = {
                'n_indices': len(indices),
                'sheet_range': (i_start, i_end),
                'indices_file': str(indices_file)
            }
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _create_indices_finder(self, data_dir: Path, msheets, attrs, seed: int, extra_index: int):
        """Create indices finder object."""
        # Simplified version of IndicesFinder functionality
        class IndicesFinder:
            def __init__(self, datadir, msheets, attrs, seed, extra_index):
                self.datadir = datadir
                self.msheets = msheets
                self.seed = seed
                self.extra_index = extra_index
                self.aemit_index_offset = attrs['aemitIndex.offset']
                self.aemit_index_edges = attrs['aemitIndex.edges']
                self.a_interval = self.aemit_index_edges[1] - self.aemit_index_edges[0]
            
            def is_sheet_empty(self, sheet: int) -> bool:
                return self.aemit_index_offset[sheet + 1] == self.aemit_index_offset[sheet + 2]
            
            def find_index(self, sheet: int, start=None) -> Tuple[int, int]:
                if start is not None:
                    start_index = start
                    end_index = self.aemit_index_offset[sheet + 2]
                else:
                    start_index = self.aemit_index_offset[sheet + 1]
                    end_index = self.aemit_index_offset[sheet + 2]
                    
                    if self.extra_index and start_index < end_index:
                        search_start = min(start_index + self.extra_index, end_index)
                        aemit_slice = self.msheets['Aemit'][start_index:search_start].compute()
                        diff = np.diff(aemit_slice)
                        change_indices = np.where(diff == self.a_interval)[0]
                        if change_indices.size > 0:
                            delta = change_indices[0]
                            start_index += delta
                
                if start_index == end_index:
                    return start_index, end_index
                
                if self.extra_index and start_index < end_index:
                    search_end = max(end_index - self.extra_index, start_index)
                    aemit_slice_end = self.msheets['Aemit'][search_end:end_index].compute()
                    diff_end = np.round(np.diff(aemit_slice_end), 2)
                    change_indices_end = np.where(diff_end == self.a_interval)[0]
                    if change_indices_end.size > 0:
                        delta = change_indices_end[0]
                        end_index -= delta
                
                return start_index, end_index
        
        return IndicesFinder(data_dir, msheets, attrs, seed, extra_index)
    
    def _find_indices(self, finder, i_start: int, i_end: int) -> List[Dict]:
        """Find indices for sheet range."""
        indices = []
        prev_end = None
        
        for i in range(i_start, i_end):
            if finder.is_sheet_empty(i):
                self.logger.info(f"Sheet {i} is empty. Skipping...")
                continue
            
            try:
                start, end = finder.find_index(i, start=prev_end)
                prev_end = end
                indices.append({"sheet": i, "start": start, "end": end})
                self.logger.debug(f"Found indices for sheet {i}: start={start}, end={end}")
            except Exception as e:
                self.logger.warning(f"Failed to find indices for sheet {i}: {e}")
                continue
        
        return indices


class MassSheetProcessingStep(ProcessingStep):
    """Process mass sheets to create density contrast maps."""
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute mass sheet processing."""
        config = PreprocessingConfig.from_config(context.config)
        
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Get inputs
            discovery_result = inputs.get('data_discovery')
            indices_result = inputs.get('index_finding')
            
            if not all([discovery_result, indices_result]):
                raise ProcessingError("Missing required input steps")
            
            # Setup output directory
            output_dir = config.output_dir or (config.data_dir / "mass_sheets")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get processing data
            msheets = discovery_result.data['msheets_catalog']
            attrs = discovery_result.data['attrs']
            indices_df = indices_result.data['indices_df']
            
            # Create processor
            processor = self._create_mass_sheet_processor(
                config.data_dir, output_dir, msheets, attrs, config.overwrite
            )
            
            # Process sheets
            processed_sheets = []
            failed_sheets = []
            
            for _, row in indices_df.iterrows():
                sheet = int(row['sheet'])
                start = int(row['start'])
                end = int(row['end'])
                
                try:
                    success = processor.process_sheet(sheet, start, end)
                    if success:
                        processed_sheets.append(sheet)
                        self.logger.info(f"Processed sheet {sheet}")
                    else:
                        failed_sheets.append(sheet)
                        self.logger.warning(f"Failed to process sheet {sheet}")
                except Exception as e:
                    failed_sheets.append(sheet)
                    self.logger.error(f"Error processing sheet {sheet}: {e}")
            
            result.data = {
                'output_dir': output_dir,
                'processed_sheets': processed_sheets,
                'failed_sheets': failed_sheets
            }
            
            result.metadata = {
                'n_processed': len(processed_sheets),
                'n_failed': len(failed_sheets),
                'success_rate': len(processed_sheets) / len(indices_df) if len(indices_df) > 0 else 0,
                'output_dir': str(output_dir)
            }
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _create_mass_sheet_processor(self, data_dir: Path, output_dir: Path, msheets, attrs, overwrite: bool):
        """Create mass sheet processor."""
        try:
            import healpy as hp
            from astropy.cosmology import FlatLambdaCDM
        except ImportError:
            raise ProcessingError("healpy and astropy are required for mass sheet processing")
        
        class MassSheetProcessor:
            def __init__(self, data_dir, output_dir, msheets, attrs, overwrite):
                self.data_dir = data_dir
                self.output_dir = output_dir
                self.msheets = msheets
                self.overwrite = overwrite
                
                # Extract attributes
                self.aemit_index_edges = attrs['aemitIndex.edges']
                self.aemit_index_offset = attrs['aemitIndex.offset']
                self.npix = attrs['healpix.npix'][0]
                self.box_size = attrs['BoxSize'][0]
                self.m_cdm = attrs['MassTable'][1]
                self.nc = attrs['NC'][0]
                self.rhobar = self.m_cdm * (self.nc / self.box_size) ** 3
                
                # Cosmology
                self.cosmo = FlatLambdaCDM(H0=67.74, Om0=0.309)
            
            def process_sheet(self, sheet: int, start: int, end: int) -> bool:
                """Process a single mass sheet."""
                save_path = self.output_dir / f"delta-sheet-{sheet:02d}.fits"
                
                if save_path.exists() and not self.overwrite:
                    logger.info(f"File {save_path} exists and overwrite is False. Skipping...")
                    return True
                
                try:
                    delta = self._get_mass_sheet(sheet, start, end)
                    hp.write_map(str(save_path), delta, nest=True, dtype=np.float32)
                    return True
                except Exception as e:
                    logger.error(f"Failed to process sheet {sheet}: {e}")
                    return False
            
            def _get_mass_sheet(self, sheet: int, start: int, end: int) -> np.ndarray:
                """Compute density contrast map for a mass sheet."""
                # Read data
                pid = self.msheets['ID'][start:end].compute()
                mass = self.msheets['Mass'][start:end].compute()
                
                # Compute map
                ipix = pid % self.npix
                map_slice = np.bincount(ipix, weights=mass, minlength=self.npix)
                
                # Compute volume and density contrast
                a1, a2 = self.aemit_index_edges[sheet:sheet + 2]
                z1, z2 = 1.0 / a1 - 1.0, 1.0 / a2 - 1.0
                chi1, chi2 = self.cosmo.comoving_distance([z1, z2]).value * self.cosmo.h
                volume_diff = (4.0 * np.pi * (chi1**3 - chi2**3)) / (3 * self.npix)
                delta = map_slice / (volume_diff * self.rhobar) - 1.0
                
                return delta
        
        return MassSheetProcessor(data_dir, output_dir, msheets, attrs, overwrite)


class ValidationStep(ProcessingStep):
    """Validate processed mass sheets."""
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute validation."""
        config = PreprocessingConfig.from_config(context.config)
        
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        if not config.validate_outputs:
            result.status = StepStatus.SKIPPED
            result.metadata = {'validation_skipped': True}
            return result
        
        try:
            # Get processing results
            processing_result = inputs.get('mass_sheet_processing')
            if not processing_result or not processing_result.is_successful():
                raise ProcessingError("Mass sheet processing step failed or missing")
            
            output_dir = processing_result.data['output_dir']
            processed_sheets = processing_result.data['processed_sheets']
            
            # Validate files
            validation_results = self._validate_output_files(output_dir, processed_sheets)
            
            result.data = validation_results
            result.metadata = {
                'n_validated': validation_results['n_valid_files'],
                'n_invalid': validation_results['n_invalid_files'],
                'validation_passed': validation_results['n_invalid_files'] == 0
            }
            
            if validation_results['n_invalid_files'] > 0:
                self.logger.warning(f"Validation found {validation_results['n_invalid_files']} invalid files")
            else:
                self.logger.info("All output files passed validation")
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _validate_output_files(self, output_dir: Path, processed_sheets: List[int]) -> Dict[str, Any]:
        """Validate output files."""
        try:
            import healpy as hp
        except ImportError:
            raise ProcessingError("healpy is required for validation")
        
        valid_files = []
        invalid_files = []
        
        for sheet in processed_sheets:
            file_path = output_dir / f"delta-sheet-{sheet:02d}.fits"
            
            try:
                if not file_path.exists():
                    invalid_files.append({'sheet': sheet, 'error': 'File does not exist'})
                    continue
                
                # Try to read the file
                data = hp.read_map(str(file_path), nest=None)
                
                # Basic validation
                if not isinstance(data, np.ndarray):
                    invalid_files.append({'sheet': sheet, 'error': 'Invalid data type'})
                    continue
                
                if data.size == 0:
                    invalid_files.append({'sheet': sheet, 'error': 'Empty data'})
                    continue
                
                if not np.isfinite(data).all():
                    invalid_files.append({'sheet': sheet, 'error': 'Contains non-finite values'})
                    continue
                
                valid_files.append({'sheet': sheet, 'size': data.size, 'mean': np.mean(data), 'std': np.std(data)})
                
            except Exception as e:
                invalid_files.append({'sheet': sheet, 'error': str(e)})
        
        return {
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'n_valid_files': len(valid_files),
            'n_invalid_files': len(invalid_files)
        }


class CleanupStep(ProcessingStep):
    """Cleanup temporary files and organize outputs."""
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute cleanup."""
        config = PreprocessingConfig.from_config(context.config)
        
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            cleaned_files = []
            
            if config.cleanup_temp:
                # Clean up temporary files
                temp_dir = context.temp_dir
                if temp_dir.exists():
                    for temp_file in temp_dir.glob("*"):
                        try:
                            if temp_file.is_file():
                                temp_file.unlink()
                                cleaned_files.append(str(temp_file))
                        except Exception as e:
                            self.logger.warning(f"Failed to remove temp file {temp_file}: {e}")
            
            result.data = {
                'cleaned_files': cleaned_files
            }
            
            result.metadata = {
                'n_cleaned_files': len(cleaned_files),
                'cleanup_enabled': config.cleanup_temp
            }
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result


class PreprocessingPipeline(BasePipeline):
    """Pipeline for preprocessing mass sheet data.
    
    This pipeline handles the complete workflow from raw N-body simulation
    data to processed mass sheets ready for kappa map generation.
    
    Parameters
    ----------
    config : Any
        Configuration object with preprocessing settings
    name : str, optional
        Pipeline name
    """
    
    def __init__(self, config: Any, name: str = "PreprocessingPipeline"):
        super().__init__(config, name)
        
        # Convert to standardized config
        self._preprocessing_config = PreprocessingConfig.from_config(config)
    
    def setup(self) -> None:
        """Setup the preprocessing pipeline steps."""
        # Add steps in order
        self.add_step(DataDiscoveryStep("data_discovery"))
        
        self.add_step(IndexFindingStep(
            "index_finding",
            dependencies=["data_discovery"]
        ))
        
        self.add_step(MassSheetProcessingStep(
            "mass_sheet_processing",
            dependencies=["data_discovery", "index_finding"]
        ))
        
        self.add_step(ValidationStep(
            "validation",
            dependencies=["mass_sheet_processing"],
            skip_on_failure=True  # Validation can be skipped if processing fails
        ))
        
        self.add_step(CleanupStep(
            "cleanup",
            dependencies=["mass_sheet_processing"],
            skip_on_failure=True  # Always try to cleanup
        ))
    
    def validate_inputs(self) -> bool:
        """Validate pipeline inputs."""
        try:
            config = self._preprocessing_config
            
            # Check required paths
            if not config.data_dir.exists():
                self.logger.error(f"Data directory does not exist: {config.data_dir}")
                return False
            
            # Check for usmesh subdirectory
            usmesh_dir = config.data_dir / "usmesh"
            if not usmesh_dir.exists():
                self.logger.error(f"usmesh directory not found: {usmesh_dir}")
                return False
            
            # Validate sheet range if provided
            if config.sheet_range:
                start, end = config.sheet_range
                if start < 0 or end <= start:
                    self.logger.error(f"Invalid sheet range: {config.sheet_range}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing configuration.
        
        Returns
        -------
        Dict[str, Any]
            Processing information
        """
        config = self._preprocessing_config
        
        # Try to get additional info from data
        info = {
            'data_dir': str(config.data_dir),
            'output_dir': str(config.output_dir) if config.output_dir else None,
            'overwrite': config.overwrite,
            'sheet_range': config.sheet_range,
            'seed': config.seed,
        }
        
        # Add data discovery info if available
        try:
            from nbodykit.lab import BigFileCatalog
            usmesh_dir = config.data_dir / "usmesh"
            if usmesh_dir.exists():
                msheets = BigFileCatalog(str(usmesh_dir), dataset="HEALPIX/")
                attrs = msheets.attrs
                
                info.update({
                    'total_sheets': len(attrs['aemitIndex.edges']) - 1,
                    'box_size': attrs['BoxSize'][0],
                    'nc': attrs['NC'][0],
                    'data_seed': attrs.get('seed', [None])[0],
                })
        except (ImportError, Exception):
            pass
        
        return info


__all__ = [
    'PreprocessingConfig',
    'PreprocessingPipeline',
    'DataDiscoveryStep',
    'IndexFindingStep', 
    'MassSheetProcessingStep',
    'ValidationStep',
    'CleanupStep',
]