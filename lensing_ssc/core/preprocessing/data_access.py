# preprocessing/data_access.py
import logging
import gc
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
from nbodykit.lab import BigFileCatalog
from cachetools import TTLCache
from functools import lru_cache
from contextlib import contextmanager

from .config import ProcessingConfig
from lensing_ssc.core.preprocessing.utils import PerformanceMonitor


class OptimizedDataAccess:
    """Optimized data access for mass sheets with caching and chunking."""
    
    def __init__(self, datadir: Path, config: ProcessingConfig):
        self.datadir = datadir
        self.config = config
        self.monitor = PerformanceMonitor()
        
        # Initialize BigFile catalog
        self.msheets = BigFileCatalog(str(self.datadir / "usmesh"), dataset="HEALPIX/")
        
        # Initialize caches
        self.chunk_cache = TTLCache(maxsize=config.max_cache_entries, ttl=3600)
        self.metadata_cache = {}
        
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
    
    def get_aemit_slice(self, start: int, end: int) -> np.ndarray:
        """Get Aemit values with intelligent caching."""
        with self.monitor.timer('aemit_slice'):
            return self._get_cached_slice('Aemit', start, end)
    
    def get_data_slice(self, column: str, start: int, end: int) -> np.ndarray:
        """Get data column slice with caching."""
        with self.monitor.timer(f'{column}_slice'):
            return self._get_cached_slice(column, start, end)
    
    def _get_cached_slice(self, column: str, start: int, end: int) -> np.ndarray:
        """Get column slice with intelligent chunking and caching."""
        self.operation_count += 1
        
        # Cleanup cache periodically
        if self.operation_count % self.config.cleanup_interval == 0:
            self.cleanup_cache()
        
        # For small requests, use direct access
        if end - start <= 1000:
            return self.msheets[column][start:end].compute()
        
        # For larger requests, check cache first
        chunk_key = (column, start // self.config.chunk_size, end // self.config.chunk_size)
        
        if chunk_key in self.chunk_cache:
            cached_data = self.chunk_cache[chunk_key]
            local_start = start % self.config.chunk_size
            local_end = local_start + (end - start)
            return cached_data[local_start:local_end]
        
        # Read and cache chunk
        chunk_start = (start // self.config.chunk_size) * self.config.chunk_size
        chunk_end = min(chunk_start + self.config.chunk_size, self.total_records)
        
        chunk_data = self.msheets[column][chunk_start:chunk_end].compute()
        self.chunk_cache[chunk_key] = chunk_data
        
        # Extract requested slice
        local_start = start - chunk_start
        local_end = local_start + (end - start)
        return chunk_data[local_start:local_end]
    
    @lru_cache(maxsize=1000)
    def get_sheet_bounds(self, sheet: int) -> Tuple[int, int]:
        """Get cached sheet boundary indices."""
        start_idx = self.aemit_index_offset[sheet + 1]
        end_idx = self.aemit_index_offset[sheet + 2]
        return start_idx, end_idx
    
    def is_sheet_empty(self, sheet: int) -> bool:
        """Check if sheet is empty using cached bounds."""
        start, end = self.get_sheet_bounds(sheet)
        return start == end
    
    def find_aemit_change_point(self, start: int, search_range: int, 
                               forward: bool = True) -> Optional[int]:
        """Find where Aemit changes by a_interval, optimized version."""
        if search_range <= 0:
            return None
        
        search_end = start + search_range if forward else start
        search_start = start if forward else start - search_range
        
        # Ensure bounds are valid
        search_start = max(0, search_start)
        search_end = min(self.total_records, search_end)
        
        if search_start >= search_end:
            return None
        
        # Get the slice efficiently
        aemit_slice = self.get_aemit_slice(search_start, search_end)
        
        if len(aemit_slice) < 2:
            return None
        
        # Find differences
        diff = np.diff(aemit_slice)
        change_indices = np.where(np.abs(diff - self.a_interval) < 1e-10)[0]
        
        if change_indices.size > 0:
            relative_idx = change_indices[0]
            return search_start + relative_idx if forward else search_start + relative_idx
        
        return None
    
    def cleanup_cache(self):
        """Clean up caches to free memory."""
        # Clear chunk cache
        self.chunk_cache.clear()
        
        # Clear LRU cache
        self.get_sheet_bounds.cache_clear()
        
        # Force garbage collection
        gc.collect()
        
        logging.debug(f"Cache cleaned after {self.operation_count} operations")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = self.monitor.get_summary()
        summary['cache_stats'] = {
            'chunk_cache_size': len(self.chunk_cache),
            'lru_cache_info': self.get_sheet_bounds.cache_info()._asdict(),
            'total_operations': self.operation_count
        }
        return summary
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'monitor'):
            self.monitor.log_summary()


class BatchDataProcessor:
    """Process multiple data operations in batches for efficiency."""
    
    def __init__(self, data_access: OptimizedDataAccess):
        self.data = data_access
    
    def batch_read_sheets(self, sheet_indices: list) -> Dict[int, Dict[str, np.ndarray]]:
        """Read multiple sheets efficiently in batch."""
        results = {}
        
        # Sort sheets by their data location for sequential access
        sorted_sheets = sorted(sheet_indices, key=lambda s: self.data.get_sheet_bounds(s)[0])
        
        for sheet in sorted_sheets:
            start, end = self.data.get_sheet_bounds(sheet)
            
            if start == end:  # Empty sheet
                results[sheet] = {'empty': True}
                continue
            
            # Read required columns
            results[sheet] = {
                'ID': self.data.get_data_slice('ID', start, end),
                'Mass': self.data.get_data_slice('Mass', start, end),
                'Aemit': self.data.get_data_slice('Aemit', start, end),
                'bounds': (start, end)
            }
        
        return results