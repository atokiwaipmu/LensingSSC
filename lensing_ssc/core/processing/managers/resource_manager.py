"""
Resource management for processing operations.
"""

import gc
import os
import sys
import time
import psutil
import threading
import warnings
from typing import Dict, Any, Optional, Callable, List, Union
from contextlib import contextmanager
from dataclasses import dataclass
import logging

import numpy as np

from ...base.exceptions import ProcessingError


@dataclass
class ResourceLimits:
    """Container for resource limits configuration."""
    memory_limit_mb: Optional[int] = None
    cpu_limit_percent: Optional[float] = None
    disk_limit_mb: Optional[int] = None
    max_workers: Optional[int] = None
    timeout_seconds: Optional[int] = None


@dataclass
class ResourceUsage:
    """Container for current resource usage."""
    memory_mb: float
    memory_percent: float
    cpu_percent: float
    disk_usage_mb: float
    disk_percent: float
    num_threads: int
    num_processes: int
    timestamp: float


class ResourceManager:
    """Manager for monitoring and controlling system resources.
    
    Provides comprehensive resource management including:
    - Memory usage monitoring and limits
    - CPU usage tracking
    - Disk space monitoring
    - Process/thread tracking
    - Automatic cleanup and garbage collection
    - Resource alerts and warnings
    - Performance optimization hints
    
    Parameters
    ----------
    memory_limit_mb : int, optional
        Memory limit in megabytes
    cpu_limit_percent : float, optional
        CPU usage limit as percentage (0-100)
    disk_limit_mb : int, optional
        Disk usage limit in megabytes
    max_workers : int, optional
        Maximum number of worker processes/threads
    timeout_seconds : int, optional
        Timeout for operations in seconds
    monitoring_interval : float, optional
        Resource monitoring interval in seconds (default: 1.0)
    enable_alerts : bool, optional
        Whether to enable resource alerts (default: True)
    cleanup_threshold : float, optional
        Memory usage threshold for automatic cleanup (default: 0.8)
    """
    
    def __init__(
        self,
        memory_limit_mb: Optional[int] = None,
        cpu_limit_percent: Optional[float] = None,
        disk_limit_mb: Optional[int] = None,
        max_workers: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        monitoring_interval: float = 1.0,
        enable_alerts: bool = True,
        cleanup_threshold: float = 0.8
    ):
        self.limits = ResourceLimits(
            memory_limit_mb=memory_limit_mb,
            cpu_limit_percent=cpu_limit_percent,
            disk_limit_mb=disk_limit_mb,
            max_workers=max_workers,
            timeout_seconds=timeout_seconds
        )
        
        self.monitoring_interval = monitoring_interval
        self.enable_alerts = enable_alerts
        self.cleanup_threshold = cleanup_threshold
        
        # Internal state
        self._monitoring = False
        self._monitor_thread = None
        self._start_time = None
        self._lock = threading.RLock()
        
        # Resource tracking
        self._current_usage = None
        self._usage_history = []
        self._max_history_size = 1000
        
        # Alert tracking
        self._alert_counts = {
            'memory': 0,
            'cpu': 0,
            'disk': 0,
            'cleanup': 0
        }
        self._last_alert_time = {}
        self._alert_cooldown = 60  # seconds
        
        # Cleanup tracking
        self._cleanup_count = 0
        self._last_cleanup_time = 0
        self._cleanup_interval = 30  # seconds
        
        # Performance tracking
        self._performance_metrics = {
            'gc_time': 0,
            'gc_count': 0,
            'peak_memory_mb': 0,
            'peak_cpu_percent': 0
        }
        
        # Process information
        self._process = psutil.Process()
        self._initial_memory = self._process.memory_info().rss / (1024 * 1024)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Start monitoring if limits are set
        if any([memory_limit_mb, cpu_limit_percent, disk_limit_mb]):
            self.start_monitoring()
    
    def start_monitoring(self) -> None:
        """Start resource monitoring in background thread."""
        with self._lock:
            if self._monitoring:
                return
            
            self._monitoring = True
            self._start_time = time.time()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="ResourceMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            
            self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        with self._lock:
            if not self._monitoring:
                return
            
            self._monitoring = False
            
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
            
            self.logger.info("Resource monitoring stopped")
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage.
        
        Returns
        -------
        ResourceUsage
            Current resource usage snapshot
        """
        try:
            # Memory usage
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # System memory
            sys_memory = psutil.virtual_memory()
            memory_percent = (memory_mb / (sys_memory.total / (1024 * 1024))) * 100
            
            # CPU usage
            cpu_percent = self._process.cpu_percent()
            
            # Disk usage for current working directory
            disk_usage = psutil.disk_usage(os.getcwd())
            disk_usage_mb = disk_usage.used / (1024 * 1024)
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Thread/process counts
            num_threads = self._process.num_threads()
            num_processes = len(psutil.pids())
            
            usage = ResourceUsage(
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                disk_usage_mb=disk_usage_mb,
                disk_percent=disk_percent,
                num_threads=num_threads,
                num_processes=num_processes,
                timestamp=time.time()
            )
            
            # Update tracking
            with self._lock:
                self._current_usage = usage
                self._usage_history.append(usage)
                
                # Limit history size
                if len(self._usage_history) > self._max_history_size:
                    self._usage_history = self._usage_history[-self._max_history_size:]
                
                # Update peak metrics
                self._performance_metrics['peak_memory_mb'] = max(
                    self._performance_metrics['peak_memory_mb'], memory_mb
                )
                self._performance_metrics['peak_cpu_percent'] = max(
                    self._performance_metrics['peak_cpu_percent'], cpu_percent
                )
            
            return usage
            
        except Exception as e:
            self.logger.error(f"Error getting resource usage: {e}")
            # Return minimal usage info
            return ResourceUsage(
                memory_mb=0, memory_percent=0, cpu_percent=0,
                disk_usage_mb=0, disk_percent=0, num_threads=0,
                num_processes=0, timestamp=time.time()
            )
    
    def check_limits(self) -> Dict[str, bool]:
        """Check if current usage exceeds limits.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary indicating which limits are exceeded
        """
        usage = self.get_current_usage()
        violations = {}
        
        # Check memory limit
        if self.limits.memory_limit_mb:
            violations['memory'] = usage.memory_mb > self.limits.memory_limit_mb
        
        # Check CPU limit
        if self.limits.cpu_limit_percent:
            violations['cpu'] = usage.cpu_percent > self.limits.cpu_limit_percent
        
        # Check disk limit
        if self.limits.disk_limit_mb:
            violations['disk'] = usage.disk_usage_mb > self.limits.disk_limit_mb
        
        # Log violations
        for resource, violated in violations.items():
            if violated:
                self._handle_limit_violation(resource, usage)
        
        return violations
    
    def check_memory_limit(self) -> None:
        """Check memory limit and raise exception if exceeded.
        
        Raises
        ------
        ProcessingError
            If memory limit is exceeded
        """
        if not self.limits.memory_limit_mb:
            return
        
        usage = self.get_current_usage()
        if usage.memory_mb > self.limits.memory_limit_mb:
            raise ProcessingError(
                f"Memory usage ({usage.memory_mb:.1f} MB) "
                f"exceeds limit ({self.limits.memory_limit_mb} MB)"
            )
    
    def cleanup_memory(self, force: bool = False) -> Dict[str, Any]:
        """Perform memory cleanup and garbage collection.
        
        Parameters
        ----------
        force : bool, optional
            Force cleanup even if not needed (default: False)
            
        Returns
        -------
        Dict[str, Any]
            Cleanup statistics
        """
        current_time = time.time()
        
        # Check if cleanup is needed
        if not force:
            if current_time - self._last_cleanup_time < self._cleanup_interval:
                return {'skipped': True, 'reason': 'too_recent'}
            
            usage = self.get_current_usage()
            if (self.limits.memory_limit_mb and 
                usage.memory_mb < self.limits.memory_limit_mb * self.cleanup_threshold):
                return {'skipped': True, 'reason': 'below_threshold'}
        
        self.logger.debug("Performing memory cleanup")
        
        # Record pre-cleanup state
        pre_usage = self.get_current_usage()
        gc_start = time.perf_counter()
        
        # Perform cleanup
        collected_objects = []
        
        # Clear numpy cache if available
        try:
            import numpy as np
            if hasattr(np, 'ndarray'):
                # Force numpy to release memory
                collected_objects.append(('numpy_cache', 0))
        except ImportError:
            pass
        
        # Run garbage collection
        for generation in range(3):
            collected = gc.collect(generation)
            collected_objects.append((f'gc_gen_{generation}', collected))
        
        # Force memory release
        try:
            # Try to release memory back to OS (Linux/Unix)
            if hasattr(os, 'sync'):
                os.sync()
        except Exception:
            pass
        
        gc_time = time.perf_counter() - gc_start
        
        # Record post-cleanup state
        post_usage = self.get_current_usage()
        
        # Update tracking
        with self._lock:
            self._cleanup_count += 1
            self._last_cleanup_time = current_time
            self._performance_metrics['gc_time'] += gc_time
            self._performance_metrics['gc_count'] += 1
            self._alert_counts['cleanup'] += 1
        
        # Calculate cleanup effectiveness
        memory_freed = pre_usage.memory_mb - post_usage.memory_mb
        
        cleanup_stats = {
            'cleanup_count': self._cleanup_count,
            'gc_time': gc_time,
            'memory_freed_mb': memory_freed,
            'pre_memory_mb': pre_usage.memory_mb,
            'post_memory_mb': post_usage.memory_mb,
            'collected_objects': dict(collected_objects),
            'total_collected': sum(count for _, count in collected_objects),
            'effectiveness_percent': (memory_freed / pre_usage.memory_mb * 100) if pre_usage.memory_mb > 0 else 0
        }
        
        self.logger.info(f"Memory cleanup completed: freed {memory_freed:.1f} MB in {gc_time:.3f}s")
        
        return cleanup_stats
    
    @contextmanager
    def memory_monitor(self, operation_name: str = "operation"):
        """Context manager for monitoring memory usage during operations.
        
        Parameters
        ----------
        operation_name : str
            Name of the operation being monitored
        """
        start_usage = self.get_current_usage()
        start_time = time.perf_counter()
        
        self.logger.debug(f"Starting memory monitoring for: {operation_name}")
        
        try:
            yield start_usage
        finally:
            end_usage = self.get_current_usage()
            end_time = time.perf_counter()
            
            memory_delta = end_usage.memory_mb - start_usage.memory_mb
            duration = end_time - start_time
            
            self.logger.info(
                f"Memory monitoring - {operation_name}: "
                f"Δ{memory_delta:+.1f} MB over {duration:.1f}s "
                f"({start_usage.memory_mb:.1f} → {end_usage.memory_mb:.1f} MB)"
            )
            
            # Check if cleanup is needed
            if (self.limits.memory_limit_mb and 
                end_usage.memory_mb > self.limits.memory_limit_mb * self.cleanup_threshold):
                self.cleanup_memory()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information.
        
        Returns
        -------
        Dict[str, Any]
            System information
        """
        try:
            cpu_info = {}
            try:
                cpu_info = {
                    'physical_cores': psutil.cpu_count(logical=False),
                    'logical_cores': psutil.cpu_count(logical=True),
                    'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                    'usage_per_cpu': psutil.cpu_percent(percpu=True),
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
                }
            except Exception as e:
                cpu_info = {'error': str(e)}
            
            memory_info = {}
            try:
                mem = psutil.virtual_memory()
                swap = psutil.swap_memory()
                memory_info = {
                    'total_gb': mem.total / (1024**3),
                    'available_gb': mem.available / (1024**3),
                    'used_gb': mem.used / (1024**3),
                    'percent_used': mem.percent,
                    'swap_total_gb': swap.total / (1024**3),
                    'swap_used_gb': swap.used / (1024**3),
                    'swap_percent': swap.percent
                }
            except Exception as e:
                memory_info = {'error': str(e)}
            
            disk_info = {}
            try:
                disk = psutil.disk_usage('/')
                disk_info = {
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'percent_used': (disk.used / disk.total) * 100
                }
            except Exception as e:
                disk_info = {'error': str(e)}
            
            process_info = {}
            try:
                process_info = {
                    'pid': self._process.pid,
                    'memory_mb': self._process.memory_info().rss / (1024**2),
                    'cpu_percent': self._process.cpu_percent(),
                    'num_threads': self._process.num_threads(),
                    'create_time': self._process.create_time(),
                    'status': self._process.status()
                }
            except Exception as e:
                process_info = {'error': str(e)}
            
            return {
                'cpu': cpu_info,
                'memory': memory_info,
                'disk': disk_info,
                'process': process_info,
                'python': {
                    'version': sys.version,
                    'executable': sys.executable,
                    'platform': sys.platform
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {'error': str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resource management statistics.
        
        Returns
        -------
        Dict[str, Any]
            Comprehensive statistics
        """
        with self._lock:
            current_usage = self.get_current_usage()
            
            # Calculate averages from history
            if self._usage_history:
                avg_memory = np.mean([u.memory_mb for u in self._usage_history])
                avg_cpu = np.mean([u.cpu_percent for u in self._usage_history])
                max_memory = np.max([u.memory_mb for u in self._usage_history])
                max_cpu = np.max([u.cpu_percent for u in self._usage_history])
            else:
                avg_memory = avg_cpu = max_memory = max_cpu = 0
            
            uptime = time.time() - self._start_time if self._start_time else 0
            
            return {
                'current_usage': current_usage.__dict__,
                'limits': self.limits.__dict__,
                'averages': {
                    'memory_mb': float(avg_memory),
                    'cpu_percent': float(avg_cpu)
                },
                'peaks': {
                    'memory_mb': float(max_memory),
                    'cpu_percent': float(max_cpu)
                },
                'performance_metrics': self._performance_metrics.copy(),
                'alert_counts': self._alert_counts.copy(),
                'cleanup_stats': {
                    'count': self._cleanup_count,
                    'last_time': self._last_cleanup_time,
                    'interval': self._cleanup_interval
                },
                'monitoring': {
                    'active': self._monitoring,
                    'uptime': uptime,
                    'interval': self.monitoring_interval,
                    'history_size': len(self._usage_history)
                },
                'memory_growth': {
                    'initial_mb': self._initial_memory,
                    'current_mb': current_usage.memory_mb,
                    'growth_mb': current_usage.memory_mb - self._initial_memory,
                    'growth_percent': ((current_usage.memory_mb / self._initial_memory) - 1) * 100 if self._initial_memory > 0 else 0
                }
            }
    
    def estimate_memory_for_operation(self, operation_type: str, **params) -> Dict[str, float]:
        """Estimate memory requirements for an operation.
        
        Parameters
        ----------
        operation_type : str
            Type of operation
        **params
            Operation parameters
            
        Returns
        -------
        Dict[str, float]
            Memory estimates in MB
        """
        estimates = {'base_memory_mb': self.get_current_usage().memory_mb}
        
        if operation_type == 'healpix_map':
            nside = params.get('nside', 2048)
            npix = 12 * nside * nside
            dtype_size = params.get('dtype_size', 8)  # float64
            estimates['map_memory_mb'] = (npix * dtype_size) / (1024**2)
            estimates['processing_overhead_mb'] = estimates['map_memory_mb'] * 0.5
            
        elif operation_type == 'patch_extraction':
            n_patches = params.get('n_patches', 100)
            patch_size = params.get('patch_size', 2048)
            dtype_size = params.get('dtype_size', 4)  # float32
            patch_memory = n_patches * patch_size * patch_size * dtype_size
            estimates['patches_memory_mb'] = patch_memory / (1024**2)
            estimates['processing_overhead_mb'] = estimates['patches_memory_mb'] * 0.3
            
        elif operation_type == 'fft_processing':
            data_size_mb = params.get('data_size_mb', 100)
            estimates['data_memory_mb'] = data_size_mb
            estimates['fft_working_memory_mb'] = data_size_mb * 2  # Complex arrays
            estimates['processing_overhead_mb'] = data_size_mb * 0.5
            
        else:
            self.logger.warning(f"Unknown operation type: {operation_type}")
            estimates['unknown_operation_mb'] = 100
        
        estimates['total_estimated_mb'] = sum(
            v for k, v in estimates.items() 
            if k != 'base_memory_mb' and k.endswith('_mb')
        )
        estimates['peak_estimated_mb'] = estimates['base_memory_mb'] + estimates['total_estimated_mb']
        
        # Check if estimate exceeds limits
        if self.limits.memory_limit_mb:
            estimates['within_limits'] = estimates['peak_estimated_mb'] <= self.limits.memory_limit_mb
            estimates['safety_margin_mb'] = self.limits.memory_limit_mb - estimates['peak_estimated_mb']
        
        return estimates
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in background thread)."""
        self.logger.debug("Resource monitoring loop started")
        
        while self._monitoring:
            try:
                # Update current usage
                self.get_current_usage()
                
                # Check limits
                violations = self.check_limits()
                
                # Automatic cleanup if memory usage is high
                if self._current_usage and self.limits.memory_limit_mb:
                    usage_ratio = self._current_usage.memory_mb / self.limits.memory_limit_mb
                    if usage_ratio > self.cleanup_threshold:
                        self.cleanup_memory()
                
                # Sleep until next check
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
        
        self.logger.debug("Resource monitoring loop stopped")
    
    def _handle_limit_violation(self, resource: str, usage: ResourceUsage) -> None:
        """Handle resource limit violation."""
        current_time = time.time()
        last_alert = self._last_alert_time.get(resource, 0)
        
        # Rate limit alerts
        if current_time - last_alert < self._alert_cooldown:
            return
        
        with self._lock:
            self._alert_counts[resource] += 1
            self._last_alert_time[resource] = current_time
        
        if self.enable_alerts:
            if resource == 'memory':
                self.logger.warning(
                    f"Memory limit exceeded: {usage.memory_mb:.1f} MB > "
                    f"{self.limits.memory_limit_mb} MB"
                )
            elif resource == 'cpu':
                self.logger.warning(
                    f"CPU limit exceeded: {usage.cpu_percent:.1f}% > "
                    f"{self.limits.cpu_limit_percent}%"
                )
            elif resource == 'disk':
                self.logger.warning(
                    f"Disk usage limit exceeded: {usage.disk_usage_mb:.1f} MB > "
                    f"{self.limits.disk_limit_mb} MB"
                )
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()