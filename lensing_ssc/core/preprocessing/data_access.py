# ====================
# lensing_ssc/core/preprocessing/data_access.py
# ====================
import logging
import gc
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
from nbodykit.lab import BigFileCatalog
import re

from .config import ProcessingConfig
from .utils import PerformanceMonitor


class OptimizedDataAccess:
    """Optimized data access for mass sheets with caching and chunking."""
    
    def __init__(self, datadir: Path, config: ProcessingConfig):
        self.datadir = datadir
        self.config = config
        self.monitor = PerformanceMonitor()
        
        # Initialize BigFile catalog
        self.msheets = BigFileCatalog(str(self.datadir / "usmesh"), dataset="HEALPIX/")
        
        # Cache frequently accessed attributes
        self._cache_attributes()
        
        # Operation counters for cleanup
        self.operation_count = 0
        
        logging.info(f"Initialized data access for {self.total_records:,} records")
    
    def _cache_attributes(self):
        """Cache frequently accessed catalog attributes."""
        attrs = self.msheets.attrs
        self.aemit_index_edges = attrs['aemitIndex.edges']
        self.aemit_index_offset = attrs['aemitIndex.offset']
        
        # Handle the case where offset has more elements than expected
        if len(self.aemit_index_offset) > len(self.aemit_index_edges) + 1:
            logging.warning(f"Offset array longer than expected. Using first {len(self.aemit_index_edges) + 1} elements.")
            self.aemit_index_offset = self.aemit_index_offset[:len(self.aemit_index_edges) + 1]
        
        self.a_interval = self.aemit_index_edges[1] - self.aemit_index_edges[0]
        self.total_records = len(self.msheets)
        
        # Try to get seed from attributes
        try:
            self.seed = attrs['seed'][0]
        except KeyError:
            import re
            dataset_name = str(self.datadir)
            seed_match = re.search(r's(\d+)', dataset_name)
            self.seed = int(seed_match.group(1)) if seed_match else 0
            logging.warning(f"No seed in attributes, extracted: {self.seed}")
    
    def get_column_slice(self, column: str, start: int, end: int) -> np.ndarray:
        """Get data column slice."""
        with self.monitor.timer(f'{column}_slice'):
            return self.msheets[column][start:end].compute()
    
    def get_sheet_bounds(self, sheet: int) -> Tuple[int, int]:
        """Get sheet boundary indices with bounds checking."""
        # Ensure we don't go out of bounds
        max_sheet = len(self.aemit_index_offset) - 2
        if sheet >= max_sheet:
            logging.warning(f"Sheet {sheet} exceeds maximum available sheet {max_sheet}")
            return self.aemit_index_offset[-1], self.aemit_index_offset[-1]  # Empty range
        
        start_idx = self.aemit_index_offset[sheet + 1]
        end_idx = self.aemit_index_offset[sheet + 2] if sheet + 2 < len(self.aemit_index_offset) else self.total_records
        
        # Ensure indices are within data bounds
        start_idx = min(start_idx, self.total_records)
        end_idx = min(end_idx, self.total_records)
        
        return start_idx, end_idx
    
    def is_sheet_empty(self, sheet: int) -> bool:
        """Check if sheet is empty."""
        start, end = self.get_sheet_bounds(sheet)
        return start == end
    
    def cleanup_cache(self):
        """Clean up caches to free memory."""
        import gc
        gc.collect()
        logging.debug(f"Cache cleaned after {self.operation_count} operations")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return self.monitor.get_summary()