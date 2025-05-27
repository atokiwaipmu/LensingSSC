# ====================
# lensing_ssc/core/preprocessing/indices.py
# ====================
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import time

from .config import ProcessingConfig
from .data_access import OptimizedDataAccess
from .utils import CheckpointManager


class OptimizedIndicesFinder:
    """Optimized indices finder with progress tracking and checkpointing."""
    
    def __init__(self, datadir: Path, config: ProcessingConfig):
        self.datadir = datadir
        self.config = config
        self.data = OptimizedDataAccess(datadir, config)
        
        self.save_path = self.datadir / f"preproc_s{self.data.seed}_indices.csv"
        self.checkpoint_manager = CheckpointManager(self.datadir)
        self.checkpoint_key = f"indices_s{self.data.seed}"
        
    def find_indices(self, i_start: Optional[int] = None, i_end: Optional[int] = None) -> None:
        """Find and save indices with progress tracking and checkpointing."""
        if i_start is None:
            i_start = self.config.sheet_range[0]
        if i_end is None:
            i_end = self.config.sheet_range[1]
        
        # Load checkpoint
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        completed_sheets = set(checkpoint_data.get(self.checkpoint_key, {}).get('completed_sheets', []))
        
        # Filter out already completed sheets
        remaining_sheets = [i for i in range(i_start, i_end) 
                          if i not in completed_sheets and not self.data.is_sheet_empty(i)]
        
        if not remaining_sheets:
            logging.info("All sheets already processed or empty.")
            return
        
        indices = []
        
        # Load existing indices if file exists
        if self.save_path.exists():
            existing_df = pd.read_csv(self.save_path)
            indices = existing_df.to_dict('records')
        
        # Process remaining sheets with progress bar
        with tqdm(total=len(remaining_sheets), 
                 desc="Finding indices", 
                 disable=not self.config.enable_progress_bar) as pbar:
            
            for i, sheet in enumerate(remaining_sheets):
                try:
                    start, end = self._find_optimized_index(sheet)
                    
                    indices.append({"sheet": sheet, "start": start, "end": end})
                    completed_sheets.add(sheet)
                    
                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix(sheet=sheet, start=start, end=end)
                    
                    # Save checkpoint periodically
                    if (i + 1) % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(list(completed_sheets))
                        self._save_indices(indices)
                    
                except Exception as e:
                    logging.error(f"Failed to process sheet {sheet}: {e}")
                    continue
        
        # Final save
        if indices:
            self._save_indices(indices)
            self._cleanup_checkpoint()
            logging.info(f"Indices saved to {self.save_path}")
    
    def _find_optimized_index(self, sheet: int) -> Tuple[int, int]:
        """Find optimized index for a sheet."""
        start_index, end_index = self.data.get_sheet_bounds(sheet)
        logging.info(f"Sheet {sheet}: indices ({start_index}, {end_index})")
        return start_index, end_index
    
    def _save_indices(self, indices: List[Dict]):
        """Save indices to CSV."""
        if indices:
            df = pd.DataFrame(indices)
            df.to_csv(self.save_path, index=False)
    
    def _save_checkpoint(self, completed_sheets_list: list):
        """Save checkpoint data."""
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        checkpoint_data[self.checkpoint_key] = {
            'completed_sheets': completed_sheets_list,
            'timestamp': time.time()
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_data)
    
    def _cleanup_checkpoint(self):
        """Remove checkpoint entry."""
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        if self.checkpoint_key in checkpoint_data:
            del checkpoint_data[self.checkpoint_key]
            if not checkpoint_data:
                self.checkpoint_manager.clear_checkpoint()
            else:
                self.checkpoint_manager.save_checkpoint(checkpoint_data)