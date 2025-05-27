# ====================
# lensing_ssc/core/preprocessing/processing.py
# ====================
import logging
import gc
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

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

    def __init__(self, datadir: Path, config: Optional[ProcessingConfig] = None, overwrite: bool = False) -> None:
        self.datadir = Path(datadir)
        self.config = config or ProcessingConfig()
        
        # Override config with parameters if provided
        if overwrite:
            self.config.overwrite = overwrite
            
        self.output_dir = self.datadir / "mass_sheets"
        
        # Initialize components
        self.data_access = OptimizedDataAccess(self.datadir, self.config)
        self.validator = DataValidator()
        self.monitor = PerformanceMonitor()
        self.checkpoint_manager = CheckpointManager(self.datadir)
        self.checkpoint_key = f"mass_sheet_processing_s{self.data_access.seed}"
        
        # Initialize cosmology and processing parameters
        self.cosmo = FlatLambdaCDM(H0=67.74, Om0=0.309)
        self._initialize_processing_parameters()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate data structure
        if not self.validator.validate_usmesh_structure(self.data_access.msheets):
            raise ValueError("Data validation failed")
        
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
        self.seed = self.data_access.seed

    def _get_or_create_indices(self) -> pd.DataFrame:
        """Get indices from existing file or create new ones."""
        csv_path = self.datadir / f"preproc_s{self.seed}_indices.csv"
        
        if csv_path.exists() and not self.config.overwrite:
            logging.info(f"Loading existing indices from {csv_path}")
            return pd.read_csv(csv_path)
        
        logging.info("Creating new indices...")
        finder = OptimizedIndicesFinder(self.datadir, self.config)
        finder.find_indices(*self.config.sheet_range)
        return pd.read_csv(csv_path)

    def preprocess(self, resume: bool = False) -> Dict[str, Any]:
        """Main preprocessing method with progress tracking and error recovery."""
        # Load checkpoint if resuming
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        completed_sheets = set()
        if resume and checkpoint_data and self.checkpoint_key in checkpoint_data:
            data_for_key = checkpoint_data[self.checkpoint_key]
            if isinstance(data_for_key, dict):
                completed_sheets = set(data_for_key.get('completed_sheets', []))
            elif isinstance(data_for_key, list):
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
            results = self._process_sequential(sheets_to_process, progress)
        except KeyboardInterrupt:
            logging.info("Processing interrupted by user")
        finally:
            progress.close()
            
        # Update checkpoint
        successful_sheets = [r.sheet_id for r in results if r.success]
        all_completed = completed_sheets.union(successful_sheets)
        self._save_checkpoint(list(all_completed))
        
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

    def _save_checkpoint(self, completed_sheets_list: List[int]) -> None:
        """Save processing checkpoint."""
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        checkpoint_data[self.checkpoint_key] = {
            'completed_sheets': completed_sheets_list,
            'last_updated': time.time()
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_data)
        logging.info(f"Checkpoint saved for {len(completed_sheets_list)} completed sheets.")

    def _cleanup_memory(self) -> None:
        """Cleanup memory by clearing caches and running garbage collection."""
        if hasattr(self, 'data_access') and hasattr(self.data_access, 'cleanup_cache'):
            self.data_access.cleanup_cache()
        gc.collect()

    def _generate_summary(self, results: List[ProcessingResult], previously_completed: int) -> Dict[str, Any]:
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

    def validate_data(self) -> bool:
        """Validate data structure."""
        return self.validator.validate_data(self.datadir)

    def get_processing_info(self) -> Dict[str, Any]:
        """Get processing information for dry-run."""
        return {
            "seed": self.seed,
            "total_sheets": len(self.indices_df),
            "sheets_to_process": len(self.indices_df),
            "sheet_range": self.config.sheet_range,
            "existing_files": list(self.output_dir.glob("*.fits")) if self.output_dir.exists() else []
        }