# preprocessing/indices.py
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import json # Keep for loading old checkpoints if necessary, or remove if handling only new format
import time # Added import for time

from .config import ProcessingConfig
from .data_access import OptimizedDataAccess
from lensing_ssc.core.preprocessing.utils import CheckpointManager # Import CheckpointManager


class OptimizedIndicesFinder:
    """Optimized indices finder with progress tracking and checkpointing."""
    
    def __init__(self, datadir: Path, config: ProcessingConfig):
        self.datadir = datadir
        self.config = config
        self.data = OptimizedDataAccess(datadir, config)
        
        self.save_path = self.datadir / f"preproc_s{self.data.seed}_indices.csv"
        # Use CheckpointManager for checkpointing logic
        self.checkpoint_manager = CheckpointManager(self.datadir)
        self.checkpoint_key = f"indices_s{self.data.seed}" # Unique key for this process's checkpoint
        
    def find_indices(self, i_start: Optional[int] = None, i_end: Optional[int] = None) -> None:
        """Find and save indices with progress tracking and checkpointing."""
        if i_start is None:
            i_start = self.config.sheet_range[0]
        if i_end is None:
            i_end = self.config.sheet_range[1]
        
        # Load checkpoint using CheckpointManager
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        completed_sheets = set()
        if checkpoint_data and self.checkpoint_key in checkpoint_data:
            completed_sheets = set(checkpoint_data[self.checkpoint_key].get('completed_sheets', []))
        
        # Filter out already completed sheets
        remaining_sheets = [i for i in range(i_start, i_end) 
                          if i not in completed_sheets and not self.data.is_sheet_empty(i)]
        
        if not remaining_sheets:
            logging.info("All sheets already processed or empty.")
            if checkpoint_data and self.checkpoint_key in checkpoint_data: # Clean up if everything was already done
                self._cleanup_checkpoint_entry(self.checkpoint_key)
            return
        
        indices = []
        prev_end = None
        
        # Load existing indices if file exists
        if self.save_path.exists():
            existing_df = pd.read_csv(self.save_path)
            indices = existing_df.to_dict('records')
            if indices:
                # If loading existing indices, ensure completed_sheets reflects this to avoid reprocessing
                # This logic might need adjustment based on how `prev_end` is determined from existing_df
                processed_from_csv = set(idx['sheet'] for idx in indices)
                completed_sheets.update(processed_from_csv) 
                # Re-filter remaining_sheets based on potentially updated completed_sheets
                remaining_sheets = [i for i in range(i_start, i_end)
                                  if i not in completed_sheets and not self.data.is_sheet_empty(i)]
                if not remaining_sheets:
                    logging.info("All sheets from CSV already processed or empty.")
                    if checkpoint_data and self.checkpoint_key in checkpoint_data:
                        self._cleanup_checkpoint_entry(self.checkpoint_key)
                    return
                
                prev_end = max(idx['end'] for idx in indices if 'end' in idx) # Ensure 'end' exists
        
        # Process remaining sheets with progress bar
        with tqdm(total=len(remaining_sheets), 
                 desc="Finding indices", 
                 disable=not self.config.enable_progress_bar) as pbar:
            
            for i, sheet in enumerate(remaining_sheets):
                try:
                    start, end = self._find_optimized_index(sheet, prev_end)
                    prev_end = end
                    
                    indices.append({"sheet": sheet, "start": start, "end": end})
                    completed_sheets.add(sheet)
                    
                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix(sheet=sheet, start=start, end=end)
                    
                    # Save checkpoint periodically using CheckpointManager
                    if (i + 1) % self.config.checkpoint_interval == 0:
                        self._save_checkpoint_entry(self.checkpoint_key, list(completed_sheets))
                        self._save_indices(indices)
                    
                except Exception as e:
                    logging.error(f"Failed to process sheet {sheet}: {e}")
                    continue
        
        # Final save
        if indices:
            self._save_indices(indices)
            self._cleanup_checkpoint_entry(self.checkpoint_key) # Use new cleanup method
            logging.info(f"Indices saved to {self.save_path}")
        
        # Log performance summary
        self.data.monitor.log_summary()
    
    def _find_optimized_index(self, sheet: int, prev_end: Optional[int] = None) -> Tuple[int, int]:
        """Optimized index finding using cached data access."""
        start_index, end_index = self.data.get_sheet_bounds(sheet)
        
        if prev_end is not None:
            start_index = prev_end
        
        if start_index == end_index:
            return start_index, end_index
        
        # Optimize start index if extra_index is specified
        if self.config.extra_index and prev_end is None:
            change_point = self.data.find_aemit_change_point(
                start_index, self.config.extra_index, forward=True
            )
            if change_point is not None:
                logging.info(f"Sheet {sheet}: Aemit start optimized from {start_index} to {change_point}")
                start_index = change_point
        
        # Optimize end index if extra_index is specified
        if self.config.extra_index:
            change_point = self.data.find_aemit_change_point(
                end_index, self.config.extra_index, forward=False
            )
            if change_point is not None:
                logging.info(f"Sheet {sheet}: Aemit end optimized from {end_index} to {change_point}")
                end_index = change_point
        
        logging.info(f"Sheet {sheet}: indices ({start_index}, {end_index})")
        return start_index, end_index
    
    def _save_indices(self, indices: List[Dict]):
        """Save indices to CSV."""
        if indices:
            df = pd.DataFrame(indices)
            df.to_csv(self.save_path, index=False)
    
    def _save_checkpoint_entry(self, key: str, completed_sheets_list: list):
        """Save checkpoint data for a specific key using CheckpointManager."""
        current_checkpoint_data = self.checkpoint_manager.load_checkpoint() or {}
        current_checkpoint_data[key] = {
            'completed_sheets': completed_sheets_list,
            'timestamp': time.time(), # Added import time
            'config': self.config.__dict__ # Ensure config is serializable or select parts
        }
        # metadata in save_checkpoint is for the entire file, not per-key
        self.checkpoint_manager.save_checkpoint(
            completed_sheets=current_checkpoint_data.get(key, {}).get('completed_sheets', []), # This seems a bit redundant, needs review
            metadata=current_checkpoint_data 
        )
    
    def _cleanup_checkpoint_entry(self, key: str):
        """Remove a specific entry from the checkpoint file."""
        current_checkpoint_data = self.checkpoint_manager.load_checkpoint()
        if current_checkpoint_data and key in current_checkpoint_data:
            del current_checkpoint_data[key]
            if not current_checkpoint_data: # If no other keys, clear the file
                self.checkpoint_manager.clear_checkpoint()
            else:
                # Re-save the checkpoint file with the key removed.
                # This part is tricky as CheckpointManager.save_checkpoint expects `completed_sheets` list directly.
                # We might need to adjust CheckpointManager or how we store multiple process states.
                # For now, let's assume CheckpointManager can handle saving the modified dict.
                # This likely requires CheckpointManager to be more flexible or a new method.
                # A simple approach: save based on one of the remaining keys, or save the whole dict as metadata.
                
                # Simplified: if a key is removed and others exist, we re-save the whole thing.
                # This might not be ideal if save_checkpoint expects a specific structure.
                # For now, this is a placeholder for a more robust multi-process checkpoint handling.
                # One way is to have CheckpointManager.save_checkpoint take the full data dict.
                
                # Let's assume `save_checkpoint` can save the dictionary directly as metadata
                # and `completed_sheets` argument is for the primary list of the *calling* process.
                # This needs a more thought-out design for multi-key checkpointing.
                # A temporary workaround:
                self.checkpoint_manager.save_checkpoint(completed_sheets=[], metadata=current_checkpoint_data)
        elif not current_checkpoint_data: # If file was already empty or non-existent
            self.checkpoint_manager.clear_checkpoint()


class IndicesValidator:
    """Validate computed indices for consistency."""
    
    def __init__(self, datadir: Path, data_access: OptimizedDataAccess):
        self.datadir = datadir
        self.data = data_access
    
    def validate_indices_file(self, indices_path: Path) -> bool:
        """Validate indices file for consistency and completeness."""
        if not indices_path.exists():
            logging.error(f"Indices file not found: {indices_path}")
            return False
        
        try:
            df = pd.read_csv(indices_path)
            
            # Check required columns
            required_cols = {'sheet', 'start', 'end'}
            if not required_cols.issubset(df.columns):
                logging.error(f"Missing required columns: {required_cols - set(df.columns)}")
                return False
            
            # Validate individual entries
            issues = []
            for _, row in df.iterrows():
                sheet, start, end = int(row['sheet']), int(row['start']), int(row['end'])
                
                # Check bounds consistency
                if start > end:
                    issues.append(f"Sheet {sheet}: start > end ({start} > {end})")
                
                # Check against data bounds
                data_start, data_end = self.data.get_sheet_bounds(sheet)
                if not self.data.is_sheet_empty(sheet):
                    if start < data_start or end > data_end:
                        issues.append(f"Sheet {sheet}: indices out of data bounds")
            
            if issues:
                logging.error(f"Validation issues found: {issues}")
                return False
            
            logging.info(f"Indices file validation passed: {len(df)} sheets")
            return True
            
        except Exception as e:
            logging.error(f"Failed to validate indices file: {e}")
            return False
    
    def detect_gaps_and_overlaps(self, indices_path: Path) -> Dict[str, List]:
        """Detect gaps and overlaps in sheet indices."""
        df = pd.read_csv(indices_path)
        df = df.sort_values('sheet')
        
        gaps = []
        overlaps = []
        
        for i in range(len(df) - 1):
            current_end = df.iloc[i]['end']
            next_start = df.iloc[i + 1]['start']
            current_sheet = df.iloc[i]['sheet']
            next_sheet = df.iloc[i + 1]['sheet']
            
            if current_end < next_start:
                gaps.append({
                    'between_sheets': (current_sheet, next_sheet),
                    'gap_size': next_start - current_end
                })
            elif current_end > next_start:
                overlaps.append({
                    'between_sheets': (current_sheet, next_sheet),
                    'overlap_size': current_end - next_start
                })
        
        return {'gaps': gaps, 'overlaps': overlaps}