"""
Resource manager for monitoring and controlling system resources.

This module provides comprehensive resource monitoring and management capabilities
for processing operations, including memory, CPU, and disk usage tracking with
automatic limits and recovery mechanisms.
"""

import psutil
import threading
import time
import logging
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager

from .exceptions import ResourceError
from ...config.settings import ProcessingConfig


logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limit configuration."""
    memory_mb: Optional[int] = None
    memory_percent: Optional[float] = None
    cpu_percent: Optional[float] = None
    disk_gb: Optional[float] = None
    disk_percent: Optional[float] = None
    swap_percent: Optional[float] = None


@dataclass
class ResourceUsage:
    """Current resource usage information."""
    memory_mb: float
    memory_percent: float
    cpu_percent: float
    disk_gb: float
    disk_percent: float
    swap_percent: float
    timestamp: float = field(default_factory=time.time)
    
    def exceeds_limits(self, limits: ResourceLimits) -> Dict[str, bool]:
        """Check which limits are exceeded."""
        exceeded = {}
        
        if limits.memory_mb and self.memory_mb > limits.memory_mb:
            exceeded['memory_mb'] = True
        if limits.memory_percent and self.memory_percent > limits.memory_percent:
            exceeded['memory_percent'] = True
        if limits.cpu_percent and self.cpu_percent > limits.cpu_percent:
            exceeded['cpu_percent'] = True
        if limits.disk_gb and self.disk_gb > limits.disk_gb:
            exceeded['disk_gb'] = True
        if limits.disk_percent and self.disk_percent > limits.disk_percent:
            exceeded['disk_percent'] = True
        if limits.swap_percent and self.swap_percent > limits.swap_percent:
            exceeded['swap_percent'] = True
            
        return exceeded


class ResourceManager:
    """Manager for system resource monitoring and control.
    
    This manager provides:
    - Real-time resource monitoring (memory, CPU, disk)
    - Configurable resource limits with automatic enforcement
    - Background monitoring with callback support
    - Context manager support for scoped resource management
    - Detailed resource usage reporting and logging
    """
    
    def __init__(
        self,
        memory_limit_mb: Optional[int] = None,
        memory_limit_percent: Optional[float] = None,
        cpu_limit_percent: Optional[float] = None,
        disk_limit_gb: Optional[float] = None,
        disk_limit_percent: Optional[float] = None,
        swap_limit_percent: Optional[float] = None,
        monitor_interval: float = 1.0,
        warning_threshold: float = 0.8,
        config: Optional[ProcessingConfig] = None
    ):
        """Initialize resource manager.
        
        Parameters
        ----------
        memory_limit_mb : int, optional
            Memory limit in megabytes
        memory_limit_percent : float, optional
            Memory limit as percentage of total
        cpu_limit_percent : float, optional
            CPU usage limit as percentage
        disk_limit_gb : float, optional
            Disk usage limit in gigabytes
        disk_limit_percent : float, optional
            Disk usage limit as percentage
        swap_limit_percent : float, optional
            Swap usage limit as percentage
        monitor_interval : float
            Monitoring interval in seconds
        warning_threshold : float
            Warning threshold as fraction of limits
        config : ProcessingConfig, optional
            Configuration object
        """
        # Load from config if provided
        if config:
            memory_limit_mb = memory_limit_mb or config.memory_limit_mb
            
        # Set up limits
        self.limits = ResourceLimits(
            memory_mb=memory_limit_mb,
            memory_percent=memory_limit_percent,
            cpu_percent=cpu_limit_percent,
            disk_gb=disk_limit_gb,
            disk_percent=disk_limit_percent,
            swap_percent=swap_limit_percent
        )
        
        self.monitor_interval = monitor_interval
        self.warning_threshold = warning_threshold
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_usage: Optional[ResourceUsage] = None
        self._usage_history: list = []
        self._max_history = 1000
        
        # Callbacks
        self._warning_callbacks: list = []
        self._limit_callbacks: list = []
        
        # Performance tracking
        self._start_usage: Optional[ResourceUsage] = None
        self._peak_usage: Optional[ResourceUsage] = None
        
        logger.debug(f"ResourceManager initialized with limits: {self.limits}")
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current system resource usage."""
        try:
            # Memory info
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 ** 2)
            memory_percent = memory.percent
            
            # CPU info
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Disk info (root partition)
            disk = psutil.disk_usage('/')
            disk_gb = disk.used / (1024 ** 3)
            disk_percent = (disk.used / disk.total) * 100
            
            # Swap info
            swap = psutil.swap_memory()
            swap_percent = swap.percent
            
            usage = ResourceUsage(
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                disk_gb=disk_gb,
                disk_percent=disk_percent,
                swap_percent=swap_percent
            )
            
            self._current_usage = usage
            return usage
            
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            raise ResourceError(f"Resource monitoring failed: {e}")
    
    def check_limits(self, raise_on_exceed: bool = True) -> Dict[str, bool]:
        """Check if current usage exceeds limits.
        
        Parameters
        ----------
        raise_on_exceed : bool
            Whether to raise exception if limits exceeded
            
        Returns
        -------
        Dict[str, bool]
            Dictionary of exceeded limits
            
        Raises
        ------
        ResourceError
            If limits exceeded and raise_on_exceed is True
        """
        usage = self.get_current_usage()
        exceeded = usage.exceeds_limits(self.limits)
        
        if exceeded:
            logger.warning(f"Resource limits exceeded: {exceeded}")
            
            # Trigger callbacks
            for callback in self._limit_callbacks:
                try:
                    callback(usage, exceeded)
                except Exception as e:
                    logger.error(f"Limit callback failed: {e}")
            
            if raise_on_exceed:
                raise ResourceError(f"Resource limits exceeded: {list(exceeded.keys())}")
        
        return exceeded
    
    def check_warnings(self) -> Dict[str, bool]:
        """Check if current usage exceeds warning thresholds."""
        usage = self.get_current_usage()
        warnings = {}
        
        # Check each limit type
        if self.limits.memory_mb:
            threshold = self.limits.memory_mb * self.warning_threshold
            if usage.memory_mb > threshold:
                warnings['memory_mb'] = True
                
        if self.limits.memory_percent:
            threshold = self.limits.memory_percent * self.warning_threshold
            if usage.memory_percent > threshold:
                warnings['memory_percent'] = True
                
        if self.limits.cpu_percent:
            threshold = self.limits.cpu_percent * self.warning_threshold
            if usage.cpu_percent > threshold:
                warnings['cpu_percent'] = True
                
        if self.limits.disk_gb:
            threshold = self.limits.disk_gb * self.warning_threshold
            if usage.disk_gb > threshold:
                warnings['disk_gb'] = True
                
        if self.limits.disk_percent:
            threshold = self.limits.disk_percent * self.warning_threshold
            if usage.disk_percent > threshold:
                warnings['disk_percent'] = True
                
        if self.limits.swap_percent:
            threshold = self.limits.swap_percent * self.warning_threshold
            if usage.swap_percent > threshold:
                warnings['swap_percent'] = True
        
        if warnings:
            logger.warning(f"Resource warning thresholds exceeded: {warnings}")
            
            # Trigger callbacks
            for callback in self._warning_callbacks:
                try:
                    callback(usage, warnings)
                except Exception as e:
                    logger.error(f"Warning callback failed: {e}")
        
        return warnings
    
    def start_monitoring(self) -> None:
        """Start background resource monitoring."""
        if self._monitoring:
            logger.warning("Resource monitoring already active")
            return
        
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="ResourceMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background resource monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
            
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        logger.debug("Resource monitoring loop started")
        
        while not self._stop_event.wait(self.monitor_interval):
            try:
                usage = self.get_current_usage()
                
                # Update peak usage
                if not self._peak_usage or self._is_higher_usage(usage, self._peak_usage):
                    self._peak_usage = usage
                
                # Add to history
                self._usage_history.append(usage)
                if len(self._usage_history) > self._max_history:
                    self._usage_history.pop(0)
                
                # Check warnings and limits
                self.check_warnings()
                
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                
        logger.debug("Resource monitoring loop stopped")
    
    def _is_higher_usage(self, usage1: ResourceUsage, usage2: ResourceUsage) -> bool:
        """Check if usage1 represents higher resource usage than usage2."""
        return (usage1.memory_mb > usage2.memory_mb or 
                usage1.cpu_percent > usage2.cpu_percent or
                usage1.disk_gb > usage2.disk_gb)
    
    def add_warning_callback(self, callback: Callable[[ResourceUsage, Dict[str, bool]], None]) -> None:
        """Add callback for warning threshold events."""
        self._warning_callbacks.append(callback)
    
    def add_limit_callback(self, callback: Callable[[ResourceUsage, Dict[str, bool]], None]) -> None:
        """Add callback for limit exceeded events."""
        self._limit_callbacks.append(callback)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_mb': memory.total / (1024 ** 2),
                'available_mb': memory.available / (1024 ** 2),
                'used_mb': memory.used / (1024 ** 2),
                'free_mb': memory.free / (1024 ** 2),
                'percent': memory.percent,
                'swap_total_mb': swap.total / (1024 ** 2),
                'swap_used_mb': swap.used / (1024 ** 2),
                'swap_percent': swap.percent,
            }
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}
    
    def get_cpu_usage(self) -> Dict[str, Any]:
        """Get detailed CPU usage information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1.0, percpu=True)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            return {
                'percent_total': sum(cpu_percent) / len(cpu_percent),
                'percent_per_cpu': cpu_percent,
                'logical_count': psutil.cpu_count(logical=True),
                'physical_count': psutil.cpu_count(logical=False),
                'frequency_mhz': cpu_freq.current if cpu_freq else None,
            }
        except Exception as e:
            logger.error(f"Failed to get CPU usage: {e}")
            return {}
    
    def get_disk_usage(self, path: Union[str, Path] = '/') -> Dict[str, Any]:
        """Get disk usage information for given path."""
        try:
            usage = psutil.disk_usage(str(path))
            
            return {
                'total_gb': usage.total / (1024 ** 3),
                'used_gb': usage.used / (1024 ** 3),
                'free_gb': usage.free / (1024 ** 3),
                'percent': (usage.used / usage.total) * 100,
            }
        except Exception as e:
            logger.error(f"Failed to get disk usage for {path}: {e}")
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive resource manager status."""
        usage = self.get_current_usage()
        
        return {
            'monitoring': self._monitoring,
            'current_usage': {
                'memory_mb': usage.memory_mb,
                'memory_percent': usage.memory_percent,
                'cpu_percent': usage.cpu_percent,
                'disk_gb': usage.disk_gb,
                'disk_percent': usage.disk_percent,
                'swap_percent': usage.swap_percent,
            },
            'limits': {
                'memory_mb': self.limits.memory_mb,
                'memory_percent': self.limits.memory_percent,
                'cpu_percent': self.limits.cpu_percent,
                'disk_gb': self.limits.disk_gb,
                'disk_percent': self.limits.disk_percent,
                'swap_percent': self.limits.swap_percent,
            },
            'peak_usage': {
                'memory_mb': self._peak_usage.memory_mb if self._peak_usage else None,
                'cpu_percent': self._peak_usage.cpu_percent if self._peak_usage else None,
                'disk_gb': self._peak_usage.disk_gb if self._peak_usage else None,
            },
            'history_length': len(self._usage_history),
            'callback_count': {
                'warning': len(self._warning_callbacks),
                'limit': len(self._limit_callbacks),
            }
        }
    
    @contextmanager
    def monitor_context(self):
        """Context manager for scoped resource monitoring."""
        self._start_usage = self.get_current_usage()
        
        if not self._monitoring:
            self.start_monitoring()
            stop_on_exit = True
        else:
            stop_on_exit = False
        
        try:
            yield self
        finally:
            if stop_on_exit:
                self.stop_monitoring()
    
    def __enter__(self):
        """Enter context manager."""
        return self.monitor_context().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if hasattr(self, '_context_manager'):
            return self._context_manager.__exit__(exc_type, exc_val, exc_tb)
        return False
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop_monitoring()
        except Exception:
            pass