# lensing_ssc/core/utils.py
import gc
import json
import time
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import numpy as np
from tqdm import tqdm
from cachetools import TTLCache


class ProgressTracker:
    """Enhanced progress tracking with timing and memory info."""
    
    def __init__(self, total_operations: int, description: str, unit: str = "it"):
        self.pbar = tqdm(total=total_operations, desc=description, unit=unit)
        self.start_time = time.perf_counter()
        
    def update(self, n: int = 1, info: str = ""):
        self.pbar.update(n)
        if info:
            self.pbar.set_postfix_str(info)
            
    def close(self):
        self.pbar.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record_timing(operation, duration)
            
    def record_timing(self, operation: str, duration: float):
        """Record timing for an operation."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
        
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary statistics."""
        summary = {}
        for op, times in self.metrics.items():
            times_array = np.array(times)
            summary[op] = {
                'count': len(times),
                'mean': float(np.mean(times_array)),
                'std': float(np.std(times_array)),
                'min': float(np.min(times_array)),
                'max': float(np.max(times_array)),
                'total': float(np.sum(times_array))
            }
        return summary
        
    def log_summary(self):
        """Log performance summary."""
        summary = self.get_summary()
        logging.info("Performance Summary:")
        for op, stats in summary.items():
            logging.info(f"  {op}: {stats['count']} ops, "
                        f"avg={stats['mean']:.3f}s, total={stats['total']:.3f}s")


class MemoryManager:
    """Manage memory usage and cleanup."""
    
    def __init__(self, cache_size_mb: int = 1024):
        self.cache_size_mb = cache_size_mb
        self.cached_chunks = TTLCache(maxsize=100, ttl=3600)  # 1 hour TTL
        
    def cleanup_memory(self):
        """Clean up cached data and force garbage collection."""
        self.cached_chunks.clear()
        gc.collect()
        
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage estimate in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
        
    @contextmanager
    def memory_limit_context(self, max_memory_mb: int = None):
        """Context manager to monitor memory usage."""
        max_memory_mb = max_memory_mb or self.cache_size_mb * 2
        initial_memory = self.get_memory_usage_mb()
        
        try:
            yield
        finally:
            current_memory = self.get_memory_usage_mb()
            if current_memory - initial_memory > max_memory_mb:
                logging.warning(f"Memory usage increased by {current_memory - initial_memory:.1f}MB")
                self.cleanup_memory()


class CheckpointManager:
    """Manage processing checkpoints for recovery."""
    
    def __init__(self, datadir: Path):
        self.datadir = Path(datadir)
        self.checkpoint_file = self.datadir / "processing_checkpoint.json"
        
    def save_checkpoint(self, completed_sheets: List[int], 
                       failed_sheets: List[int] = None,
                       metadata: Dict[str, Any] = None):
        """Save processing state for recovery."""
        checkpoint = {
            "completed_sheets": completed_sheets,
            "failed_sheets": failed_sheets or [],
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Ensure directory exists
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write atomically
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        temp_file.rename(self.checkpoint_file)
        
        logging.info(f"Checkpoint saved: {len(completed_sheets)} completed, "
                    f"{len(failed_sheets or [])} failed")
        
    def load_checkpoint(self) -> Optional[Dict]:
        """Load previous processing state."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                logging.info(f"Loaded checkpoint: {len(checkpoint.get('completed_sheets', []))} "
                           f"completed sheets")
                return checkpoint
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Failed to load checkpoint: {e}")
        return None
        
    def clear_checkpoint(self):
        """Remove checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logging.info("Checkpoint cleared")


def extract_seed_from_path(path: Path) -> int:
    """Extract seed number from dataset path."""
    import re
    path_str = str(path)
    seed_match = re.search(r's(\d+)', path_str)
    if seed_match:
        return int(seed_match.group(1))
    return 0


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup enhanced logging configuration."""
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
        
    # Configure logging
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s',
        force=True
    )
    
    # Reduce noise from external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)


def format_memory_size(size_bytes: int) -> str:
    """Format memory size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


class BatchProcessor:
    """Process items in batches with error handling."""
    
    def __init__(self, batch_size: int = 10, max_retries: int = 3):
        self.batch_size = batch_size
        self.max_retries = max_retries
        
    def process_batches(self, items: List[Any], process_func, 
                       progress_desc: str = "Processing") -> Tuple[List[Any], List[Any]]:
        """Process items in batches with retry logic."""
        successful = []
        failed = []
        
        batches = [items[i:i + self.batch_size] 
                  for i in range(0, len(items), self.batch_size)]
        
        with ProgressTracker(len(batches), progress_desc, "batch") as pbar:
            for batch in batches:
                batch_success, batch_failed = self._process_batch_with_retry(
                    batch, process_func
                )
                successful.extend(batch_success)
                failed.extend(batch_failed)
                
                pbar.update(1, f"Success: {len(successful)}, Failed: {len(failed)}")
                
        return successful, failed
        
    def _process_batch_with_retry(self, batch: List[Any], process_func) -> Tuple[List[Any], List[Any]]:
        """Process a single batch with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                results = process_func(batch)
                return results, []
            except Exception as e:
                if attempt == self.max_retries:
                    logging.error(f"Batch failed after {self.max_retries} retries: {e}")
                    return [], batch
                else:
                    logging.warning(f"Batch attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return [], batch