"""
Resource management for processing operations.
"""

import gc
import psutil
import time
from typing import Dict, Any, Optional
import logging
from contextlib import contextmanager

from ...core.base.exceptions import ProcessingError


class ResourceManager:
    """Manage system resources during processing."""
    
    def __init__(self, memory_limit_mb: Optional[int] = None):
        self.memory_limit_mb = memory_limit_mb
        self.logger = logging.getLogger(self.__class__.__name__)
        self._monitoring = False
        self._start_time = None
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        self._monitoring = True
        self._start_time = time.time()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._start_time:
            duration = time.time() - self._start_time
            self.logger.info(f"Resource monitoring stopped (duration: {duration:.2f}s)")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    
    def check_memory_limit(self) -> None:
        """Check if memory usage exceeds limit."""
        if not self.memory_limit_mb:
            return
        
        memory_usage = self.get_memory_usage()
        if memory_usage["rss_mb"] > self.memory_limit_mb:
            raise ProcessingError(
                f"Memory usage ({memory_usage['rss_mb']:.1f} MB) "
                f"exceeds limit ({self.memory_limit_mb} MB)"
            )
    
    def cleanup_memory(self) -> None:
        """Force garbage collection and memory cleanup."""
        self.logger.debug("Performing memory cleanup")
        gc.collect()
    
    @contextmanager
    def memory_monitor(self):
        """Context manager for memory monitoring."""
        initial_memory = self.get_memory_usage()
        self.logger.debug(f"Initial memory: {initial_memory['rss_mb']:.1f} MB")
        
        try:
            yield
        finally:
            final_memory = self.get_memory_usage()
            self.logger.debug(f"Final memory: {final_memory['rss_mb']:.1f} MB")
            
            memory_diff = final_memory["rss_mb"] - initial_memory["rss_mb"]
            if abs(memory_diff) > 10:  # Log if significant change
                self.logger.info(f"Memory change: {memory_diff:+.1f} MB")