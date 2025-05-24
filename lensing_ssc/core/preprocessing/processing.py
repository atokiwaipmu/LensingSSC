# preprocessing/processing.py
import logging
import gc
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

import numpy as np
import pandas as pd
import healpy as hp
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm

from .config import ProcessingConfig
from .data_access import OptimizedDataAccess
from .indices import OptimizedIndicesFinder
from .validation import DataValidator
from .utils import PerformanceMonitor, ProgressTracker, CheckpointManager


class ProcessingResult:
    """Container for processing results."""
    def __init__(self, sheet_id: int, success: bool, error: Optional[str] = None, 
                 processing_time: Optional[float] = None):
        self.sheet_id = sheet_id
        self.success = success
        self.error = error
        self.processing_time = processing_time


class MassSheetProcessor:
    """Optimized processor for mass sheets with robust error handling and progress tracking."""

    def __init__(self, datadir: Path, config: ProcessingConfig) -> None:
        self.datadir = datadir
        self.config = config
        self.output_dir = datadir / "mass_sheets"
        # self.checkpoint_file = datadir / "processing_checkpoint.json" # Managed by CheckpointManager
        
        # Initialize components
        self.data_access = OptimizedDataAccess(datadir, config)
        self.validator = DataValidator()
        self.monitor = PerformanceMonitor()
        self.checkpoint_manager = CheckpointManager(datadir) # Initialize CheckpointManager
        self.checkpoint_key = f"mass_sheet_processing_s{self.data_access.seed}" # Unique key
        
        # Initialize cosmology and processing parameters
        self.cosmo = FlatLambdaCDM(H0=67.74, Om0=0.309)
        self._initialize_processing_parameters()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate data structure
        self.validator.validate_usmesh_structure(self.data_access.msheets)
        
        # Load or create indices
        self.indices_df = self._get_or_create_indices()
        
        logging.info(f"Initialized processor for {len(self.indices_df)} sheets")

    def _initialize_processing_parameters(self) -> None:
        """Initialize processing parameters from data attributes."""
        attrs = self.data_access.msheets.attrs
        self.aemit_index_edges = attrs['aemitIndex.edges']
        self.aemit_index_offset = attrs['aemitIndex.offset']
        self.a_interval = self.aemit_index_edges[1] - self.aemit_index_edges[0]
        self.npix = attrs['healpix.npix'][0]
        self.box_size = attrs['BoxSize'][0]
        self.m_cdm = attrs['MassTable'][1]
        self.nc = attrs['NC'][0]
        self.rhobar = self.m_cdm * (self.nc / self.box_size) ** 3
        self.seed = attrs.get('seed', [0])[0]

    def _get_or_create_indices(self) -> pd.DataFrame:
        """Get indices from existing file or create new ones."""
        csv_path = self.datadir / f"preproc_s{self.seed}_indices.csv"
        
        if csv_path.exists() and not self.config.overwrite:
            logging.info(f"Loading existing indices from {csv_path}")
            return pd.read_csv(csv_path)
        
        logging.info("Creating new indices...")
        finder = OptimizedIndicesFinder(
            self.datadir, 
            self.config
        )
        finder.find_indices(*self.config.sheet_range)
        return pd.read_csv(csv_path)

    def preprocess(self) -> Dict[str, any]:
        """Main preprocessing method with progress tracking and error recovery."""
        # Load checkpoint if exists using CheckpointManager
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        completed_sheets = set()
        if checkpoint_data and self.checkpoint_key in checkpoint_data:
            data_for_key = checkpoint_data[self.checkpoint_key]
            if isinstance(data_for_key, dict):
                 completed_sheets = set(data_for_key.get('completed_sheets', []))
            elif isinstance(data_for_key, list): # older format compatibility
                 completed_sheets = set(data_for_key)

        # Filter sheets to process
        sheets_to_process = [
            row for _, row in self.indices_df.iterrows() 
            if int(row['sheet']) not in completed_sheets
        ]
        
        if not sheets_to_process:
            logging.info("All sheets already processed")
            return {"status": "complete", "processed": len(completed_sheets)}
        
        logging.info(f"Processing {len(sheets_to_process)} sheets")
        
        # Process sheets with progress tracking
        progress = ProgressTracker(len(sheets_to_process), "Processing mass sheets")
        results = []
        
        try:
            if self.config.num_workers and self.config.num_workers > 1:
                results = self._process_parallel(sheets_to_process, progress)
            else:
                results = self._process_sequential(sheets_to_process, progress)
                
        except KeyboardInterrupt:
            logging.info("Processing interrupted by user")
        finally:
            progress.close()
            
        # Update checkpoint
        successful_sheets = [r.sheet_id for r in results if r.success]
        all_completed = completed_sheets.union(successful_sheets)
        self._save_checkpoint_entry(list(all_completed)) # Use new save method
        
        # Cleanup and return summary
        self._cleanup_memory()
        return self._generate_summary(results, len(completed_sheets))

    def _process_sequential(self, sheets_to_process: List, progress: ProgressTracker) -> List[ProcessingResult]:
        """Process sheets sequentially."""
        results = []
        for row in sheets_to_process:
            result = self._process_single_sheet(row)
            results.append(result)
            
            # Update progress
            status = "✓" if result.success else "✗"
            progress.update(1, f"Sheet {result.sheet_id} {status}")
            
            # Periodic cleanup
            if len(results) % 10 == 0:
                self._cleanup_memory()
                
        return results

    def _process_parallel(self, sheets_to_process: List, progress: ProgressTracker) -> List[ProcessingResult]:
        """Process sheets in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit all tasks
            # Prepare parameters that are constant for all tasks
            config_dict_for_static = asdict(self.config)
            data_access_init_params = {'datadir': str(self.datadir)} # ODA will use config_dict_for_static
            cosmo_init_params = {"H0": self.cosmo.H0.value, "Om0": self.cosmo.Om0} # Pass H0 value
            
            # These processing parameters are derived once in __init__ and can be passed as a dict
            processing_params_dict = {
                'aemit_index_edges': self.aemit_index_edges,
                'aemit_index_offset': self.aemit_index_offset,
                'npix': self.npix,
                'rhobar': self.rhobar,
                'a_interval': self.a_interval, # Make sure this is serializable if it's a complex object
                'box_size': self.box_size,
                'm_cdm': self.m_cdm,
                'nc': self.nc
            }
            output_dir_str_for_static = str(self.output_dir)

            future_to_sheet = {
                executor.submit(self._process_single_sheet_static, 
                              dict(row), 
                              # self.datadir, # datadir is part of data_access_init_params now
                              config_dict_for_static, 
                              data_access_init_params, 
                              cosmo_init_params,
                              processing_params_dict,
                              output_dir_str_for_static,
                              None # monitor_shared, pass None for now to avoid pickling issues
                              ): row
                for row in sheets_to_process
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_sheet):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                    
                    status = "✓" if result.success else "✗"
                    progress.update(1, f"Sheet {result.sheet_id} {status}")
                    
                except Exception as e:
                    row = future_to_sheet[future]
                    sheet_id = int(row['sheet'])
                    result = ProcessingResult(sheet_id, False, str(e))
                    results.append(result)
                    progress.update(1, f"Sheet {sheet_id} ✗")
                    
        return results

    @staticmethod
    def _process_single_sheet_static(row_dict: Dict, 
                                   # datadir: Path, # No longer passed directly here
                                   config_dict: Dict, 
                                   data_access_params: Dict, 
                                   cosmo_params: Dict, 
                                   processing_params: Dict,
                                   output_dir_str: str, 
                                   monitor_shared # Remains for potential future use, but is None for now
                                   ) -> ProcessingResult:
        """Static method for parallel processing. 
           Now designed to be more lightweight by re-initializing only what's needed or using passed components.
        """
        # Reconstruct necessary components. Ideally, data_access could be shared if picklable or through a manager.
        # For now, we re-initialize a minimal set. Config is passed as dict.
        # This still isn't perfect for cache sharing across processes unless data_access is managed.
        
        # Create a minimal config object from dict for components that expect it.
        # This assumes ProcessingConfig can be initialized from a dict, which it can via **data.
        # However, we need to be careful about what parts of config are truly needed by _process_single_sheet_core
        # and its callees (like _compute_mass_sheet).
        
        # A more robust approach for cache sharing would be to initialize data_access once per worker process,
        # or use a shared memory cache. This is a step towards reducing re-initialization overhead.

        # Simplified temporary config for this static method's scope
        # We need to ensure that the config passed to OptimizedDataAccess is the full ProcessingConfig
        # if it relies on more than just chunk_size and cache_size_mb (which it does).
        
        temp_config = ProcessingConfig(**config_dict) 

        # Re-initialize OptimizedDataAccess 
        # OptimizedDataAccess constructor expects: datadir: Path, config: ProcessingConfig
        temp_config_for_oda = ProcessingConfig(**config_dict) # Create the config object
        temp_data_access = OptimizedDataAccess(Path(data_access_params['datadir']), temp_config_for_oda)

        # Re-initialize FlatLambdaCDM
        temp_cosmo = FlatLambdaCDM(H0=cosmo_params['H0'], Om0=cosmo_params['Om0'])
        
        output_dir = Path(output_dir_str)

        # Call the core logic, passing necessary pre-calculated or re-initialized components.
        # The core logic needs to be adapted to take these instead of relying on `self` for everything.
        return MassSheetProcessor._process_single_sheet_core_static(
            row_dict, temp_data_access, temp_cosmo, processing_params, output_dir, temp_config.overwrite
        )

    @staticmethod
    def _process_single_sheet_core_static(row_dict: Dict, 
                                        data_access: OptimizedDataAccess, 
                                        cosmo: FlatLambdaCDM, 
                                        processing_params: Dict,
                                        output_dir: Path,
                                        overwrite: bool) -> ProcessingResult:
        """Core processing logic for a single sheet, designed to be static and receive dependencies."""
        sheet = int(row_dict['sheet'])
        start = int(row_dict['start'])
        end = int(row_dict['end'])
        start_time = time.perf_counter()

        try:
            # Accessing aemit_index_offset and other params from the passed processing_params dict
            aemit_index_offset = processing_params['aemit_index_offset']

            if aemit_index_offset[sheet + 1] == aemit_index_offset[sheet + 2]:
                logging.debug(f"Sheet {sheet} is empty, skipping")
                return ProcessingResult(sheet, True, processing_time=0)

            save_path = output_dir / f"delta-sheet-{sheet:02d}.fits"
            if save_path.exists() and not overwrite:
                logging.debug(f"Sheet {sheet} already exists, skipping")
                return ProcessingResult(sheet, True, processing_time=0)

            # monitor.timer would ideally use a shared monitor if available/picklable
            # For now, timing is local to this static call.
            delta = MassSheetProcessor._compute_mass_sheet_static(
                sheet, start, end, data_access, cosmo, processing_params
            )

            hp.write_map(str(save_path), delta, nest=True, dtype=np.float32, overwrite=True)
            processing_time = time.perf_counter() - start_time
            logging.debug(f"Processed sheet {sheet} in {processing_time:.2f}s")
            return ProcessingResult(sheet, True, processing_time=processing_time)

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            error_msg = f"Error processing sheet {sheet} (static): {str(e)}"
            logging.error(error_msg)
            return ProcessingResult(sheet, False, error_msg, processing_time)

    def _process_single_sheet_from_dict(self, row_dict: Dict) -> ProcessingResult:
        """Process single sheet from dictionary (for parallel execution)."""
        # This method might become obsolete if _process_single_sheet_static is called directly.
        # Or it can call the static version with self.components if needed for sequential path.
        # For now, it calls the original _process_single_sheet_core which relies on `self`.
        sheet = int(row_dict['sheet'])
        start = int(row_dict['start'])
        end = int(row_dict['end'])
        # Call the original instance method for sequential processing or if static not used by parallel caller
        return self._process_single_sheet_core(sheet, start, end)

    def _process_single_sheet(self, row) -> ProcessingResult:
        """Process single sheet from pandas row."""
        sheet = int(row['sheet'])
        start = int(row['start'])
        end = int(row['end'])
        return self._process_single_sheet_core(sheet, start, end)

    def _process_single_sheet_core(self, sheet: int, start: int, end: int) -> ProcessingResult:
        """Core processing logic for a single sheet."""
        start_time = time.perf_counter()
        
        try:
            # Check if sheet is empty
            if self.aemit_index_offset[sheet + 1] == self.aemit_index_offset[sheet + 2]:
                logging.debug(f"Sheet {sheet} is empty, skipping")
                return ProcessingResult(sheet, True, processing_time=0)
            
            # Check if output already exists
            save_path = self.output_dir / f"delta-sheet-{sheet:02d}.fits"
            if save_path.exists() and not self.config.overwrite:
                logging.debug(f"Sheet {sheet} already exists, skipping")
                return ProcessingResult(sheet, True, processing_time=0)
            
            # Process the sheet
            with self.monitor.timer(f"process_sheet_{sheet}"):
                delta = self._compute_mass_sheet(sheet, start, end)
                
            # Save the result
            with self.monitor.timer(f"save_sheet_{sheet}"):
                hp.write_map(str(save_path), delta, nest=True, dtype=np.float32, overwrite=True)
            
            processing_time = time.perf_counter() - start_time
            logging.debug(f"Processed sheet {sheet} in {processing_time:.2f}s")
            
            return ProcessingResult(sheet, True, processing_time=processing_time)
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            error_msg = f"Error processing sheet {sheet}: {str(e)}"
            logging.error(error_msg)
            return ProcessingResult(sheet, False, error_msg, processing_time)

    def _compute_mass_sheet(self, sheet: int, start: int, end: int) -> np.ndarray:
        """Compute density contrast map for a mass sheet."""
        # Read data efficiently
        pid = self.data_access.get_column_slice('ID', start, end)
        mass = self.data_access.get_column_slice('Mass', start, end)
        
        # Compute pixel indices
        ipix = pid % self.npix
        
        # Bin the masses
        map_slice = np.bincount(ipix, weights=mass, minlength=self.npix)
        
        # Compute volume and density contrast
        a1, a2 = self.aemit_index_edges[sheet:sheet + 2]
        z1, z2 = 1.0 / a1 - 1.0, 1.0 / a2 - 1.0
        chi1, chi2 = self.cosmo.comoving_distance([z1, z2]).value * self.cosmo.h
        volume_diff = (4.0 * np.pi * (chi1**3 - chi2**3)) / (3 * self.npix)
        delta = map_slice / (volume_diff * self.rhobar) - 1.0
        
        return delta.astype(np.float32)

    @staticmethod
    def _compute_mass_sheet_static(sheet: int, start: int, end: int, 
                                   data_access: OptimizedDataAccess, 
                                   cosmo: FlatLambdaCDM, 
                                   processing_params: Dict) -> np.ndarray:
        """Compute density contrast map for a mass sheet (static version)."""
        pid = data_access.get_column_slice('ID', start, end)
        mass = data_access.get_column_slice('Mass', start, end)
        
        npix = processing_params['npix']
        ipix = pid % npix
        map_slice = np.bincount(ipix, weights=mass, minlength=npix)
        
        aemit_index_edges = processing_params['aemit_index_edges']
        rhobar = processing_params['rhobar']
        
        a1, a2 = aemit_index_edges[sheet:sheet + 2]
        z1, z2 = 1.0 / a1 - 1.0, 1.0 / a2 - 1.0
        chi1, chi2 = cosmo.comoving_distance([z1, z2]).value * cosmo.h # Use passed cosmo
        volume_diff = (4.0 * np.pi * (chi1**3 - chi2**3)) / (3 * npix)
        delta = map_slice / (volume_diff * rhobar) - 1.0
        
        return delta.astype(np.float32)

    def _save_checkpoint_entry(self, completed_sheets_list: List[int]) -> None:
        """Save processing checkpoint for this processor's task."""
        current_checkpoint_data = self.checkpoint_manager.load_checkpoint() or {}
        
        entry_data = {
            "completed_sheets": completed_sheets_list,
            "timestamp": time.time(),
            "config": asdict(self.config),
            "total_sheets": len(self.indices_df)
        }
        current_checkpoint_data[self.checkpoint_key] = entry_data
        
        # The `completed_sheets` argument for save_checkpoint might be for the primary calling context
        # if CheckpointManager is to be generic. For now, we pass this task's list.
        # The `metadata` field is used to store the whole multi-key dictionary.
        self.checkpoint_manager.save_checkpoint(
            completed_sheets=completed_sheets_list, 
            metadata=current_checkpoint_data
        )
        logging.debug(f"Saved checkpoint for {self.checkpoint_key} with {len(completed_sheets_list)} completed sheets")

    def _cleanup_memory(self) -> None:
        """Clean up memory caches."""
        self.data_access.cleanup_cache()
        gc.collect()

    def _generate_summary(self, results: List[ProcessingResult], previously_completed: int) -> Dict[str, any]:
        """Generate processing summary."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_time = sum(r.processing_time for r in results if r.processing_time)
        avg_time = total_time / len(results) if results else 0
        
        summary = {
            "status": "complete" if not failed else "partial",
            "total_sheets": len(self.indices_df),
            "previously_completed": previously_completed,
            "newly_processed": len(successful),
            "failed": len(failed),
            "total_time": total_time,
            "average_time_per_sheet": avg_time,
            "failed_sheets": [r.sheet_id for r in failed],
            "performance_metrics": self.monitor.get_summary()
        }
        
        logging.info(f"Processing summary: {len(successful)} successful, {len(failed)} failed")
        return summary

    def get_processing_status(self) -> Dict[str, any]:
        """Get current processing status."""
        checkpoint = self.checkpoint_manager.load_checkpoint() # Use CheckpointManager
        if not checkpoint or self.checkpoint_key not in checkpoint:
            return {"status": "not_started", "progress": 0, "key": self.checkpoint_key}
        
        data_for_key = checkpoint[self.checkpoint_key]
        completed_sheets_list = []
        if isinstance(data_for_key, dict):
            completed_sheets_list = data_for_key.get('completed_sheets', [])
        elif isinstance(data_for_key, list): # Support old format
            completed_sheets_list = data_for_key
            
        completed = len(completed_sheets_list)
        total = len(self.indices_df)
        progress = completed / total if total > 0 else 0
        
        timestamp = None
        if isinstance(data_for_key, dict):
            timestamp = data_for_key.get('timestamp')

        return {
            "status": "in_progress" if progress < 1.0 else "complete",
            "progress": progress,
            "completed_sheets": completed,
            "total_sheets": total,
            "last_update": timestamp,
            "key": self.checkpoint_key
        }