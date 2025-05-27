# ====================
# lensing_ssc/core/preprocessing/utils.py  
# ====================
import gc
import json
import time
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from tqdm import tqdm


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


class CheckpointManager:
    """Manage processing checkpoints for recovery."""
    
    def __init__(self, datadir: Path):
        self.datadir = Path(datadir)
        self.checkpoint_file = self.datadir / "processing_checkpoint.json"
        
    def save_checkpoint(self, data: Dict[str, Any]):
        """Save checkpoint data."""
        checkpoint = {
            "data": data,
            "timestamp": time.time()
        }
        
        # Ensure directory exists
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write atomically
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        temp_file.rename(self.checkpoint_file)
        
    def load_checkpoint(self) -> Optional[Dict]:
        """Load previous checkpoint data."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                return checkpoint.get('data', {})
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Failed to load checkpoint: {e}")
        return {}
        
    def clear_checkpoint(self):
        """Remove checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logging.info("Checkpoint cleared")


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