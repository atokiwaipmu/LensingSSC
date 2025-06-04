"""
Management utilities for processing operations.

This module provides managers for various aspects of processing operations:

- ResourceManager: Monitor and control system resources (memory, CPU)
- CheckpointManager: Save and restore processing state for recovery
- ProgressManager: Track and report processing progress
- CacheManager: Manage data caching and temporary storage
- LogManager: Centralized logging management

These managers work together to provide robust, monitorable, and recoverable
processing workflows.

Usage:
    from lensing_ssc.processing.managers import ResourceManager, CheckpointManager
    
    # Basic usage
    with ResourceManager(memory_limit_mb=8000) as resource_mgr:
        checkpoint_mgr = CheckpointManager("./checkpoints")
        
        # Save progress
        checkpoint_mgr.save_checkpoint({"step": 1, "data": results})
        
        # Process with monitoring
        resource_mgr.check_memory_limit()
        # ... processing code ...

Advanced Usage:
    from lensing_ssc.processing.managers import ManagerContext
    
    # Use multiple managers in a coordinated way
    with ManagerContext(
        resource_limit_mb=8000,
        checkpoint_dir="./checkpoints",
        progress_total=100
    ) as ctx:
        for i in range(100):
            # Processing with automatic management
            ctx.progress.update(1)
            ctx.resource.check_limits()
            
            if i % 10 == 0:
                ctx.checkpoint.save({"step": i})
"""

import logging
from typing import Dict, Any, Optional, Type, Union
from pathlib import Path



logger = logging.getLogger(__name__)


class ManagerContext:
    """Coordinated context manager for multiple processing managers.
    
    This class provides a unified interface for managing multiple aspects
    of processing operations in a single context.
    
    Parameters
    ----------
    resource_limit_mb : int, optional
        Memory limit in megabytes
    checkpoint_dir : str or Path, optional
        Directory for checkpoint files
    checkpoint_name : str, optional
        Base name for checkpoint files
    progress_total : int, optional
        Total number of operations for progress tracking
    progress_desc : str, optional
        Description for progress bar
    cache_dir : str or Path, optional
        Directory for cache files
    cache_size_mb : int, optional
        Maximum cache size in megabytes
    log_level : str, optional
        Logging level
    log_file : str or Path, optional
        Log file path
    """
    
    def __init__(
        self,
        resource_limit_mb: Optional[int] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_name: str = "checkpoint",
        progress_total: Optional[int] = None,
        progress_desc: str = "Processing",
        cache_dir: Optional[Union[str, Path]] = None,
        cache_size_mb: Optional[int] = None,
        log_level: Optional[str] = None,
        log_file: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        # Initialize managers
        self.resource = ResourceManager(
            memory_limit_mb=resource_limit_mb
        ) if resource_limit_mb else None
        
        self.checkpoint = CheckpointManager(
            checkpoint_dir=checkpoint_dir or Path.cwd() / ".checkpoints",
            checkpoint_name=checkpoint_name
        ) if checkpoint_dir or resource_limit_mb else None
        
        self.progress = ProgressManager(
            total=progress_total,
            description=progress_desc
        ) if progress_total else None
        
        self.cache = CacheManager(
            cache_dir=cache_dir,
            max_size_mb=cache_size_mb
        ) if cache_dir or cache_size_mb else None
        
        self.log = LogManager(
            level=log_level,
            file_path=log_file
        ) if log_level or log_file else None
        
        # Store additional kwargs for custom managers
        self.kwargs = kwargs
        
        # Track active managers
        self._active_managers = []
        for name, manager in [
            ('resource', self.resource),
            ('checkpoint', self.checkpoint),
            ('progress', self.progress),
            ('cache', self.cache),
            ('log', self.log)
        ]:
            if manager is not None:
                self._active_managers.append((name, manager))
    
    def __enter__(self):
        """Enter context and initialize all managers."""
        logger.debug(f"Entering ManagerContext with {len(self._active_managers)} managers")
        
        # Start managers that support context management
        for name, manager in self._active_managers:
            if hasattr(manager, '__enter__'):
                try:
                    manager.__enter__()
                    logger.debug(f"Started {name} manager")
                except Exception as e:
                    logger.warning(f"Failed to start {name} manager: {e}")
            elif hasattr(manager, 'start'):
                try:
                    manager.start()
                    logger.debug(f"Started {name} manager")
                except Exception as e:
                    logger.warning(f"Failed to start {name} manager: {e}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup all managers."""
        logger.debug("Exiting ManagerContext")
        
        # Stop managers in reverse order
        for name, manager in reversed(self._active_managers):
            if hasattr(manager, '__exit__'):
                try:
                    manager.__exit__(exc_type, exc_val, exc_tb)
                    logger.debug(f"Stopped {name} manager")
                except Exception as e:
                    logger.warning(f"Failed to stop {name} manager: {e}")
            elif hasattr(manager, 'stop'):
                try:
                    manager.stop()
                    logger.debug(f"Stopped {name} manager")
                except Exception as e:
                    logger.warning(f"Failed to stop {name} manager: {e}")
    
    def save_checkpoint(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save checkpoint data if checkpoint manager is available."""
        if self.checkpoint:
            self.checkpoint.save_checkpoint(data, metadata)
        else:
            logger.warning("No checkpoint manager available")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint data if checkpoint manager is available."""
        if self.checkpoint:
            return self.checkpoint.load_checkpoint()
        else:
            logger.warning("No checkpoint manager available")
            return None
    
    def update_progress(self, n: int = 1, **kwargs) -> None:
        """Update progress if progress manager is available."""
        if self.progress:
            self.progress.update(n, **kwargs)
    
    def check_resources(self) -> Dict[str, Any]:
        """Check resource usage if resource manager is available."""
        if self.resource:
            return self.resource.get_memory_usage()
        else:
            return {}
    
    def cleanup_cache(self) -> None:
        """Cleanup cache if cache manager is available."""
        if self.cache:
            self.cache.cleanup()
    
    def get_manager_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active managers."""
        status = {}
        
        for name, manager in self._active_managers:
            try:
                if hasattr(manager, 'get_status'):
                    status[name] = manager.get_status()
                elif hasattr(manager, 'get_info'):
                    status[name] = manager.get_info()
                else:
                    status[name] = {"active": True, "type": type(manager).__name__}
            except Exception as e:
                status[name] = {"active": False, "error": str(e)}
        
        return status


def create_manager_context(config: Optional[Any] = None, **kwargs) -> ManagerContext:
    """Create a manager context from configuration.
    
    Parameters
    ----------
    config : Any, optional
        Configuration object with manager settings
    **kwargs
        Override settings
        
    Returns
    -------
    ManagerContext
        Configured manager context
    """
    # Extract settings from config if provided
    settings = {}
    
    if config is not None:
        # Try to extract common configuration attributes
        for attr in [
            'memory_limit_mb', 'checkpoint_dir', 'cache_dir', 
            'cache_size_mb', 'log_level', 'log_file'
        ]:
            if hasattr(config, attr):
                settings[attr] = getattr(config, attr)
    
    # Override with provided kwargs
    settings.update(kwargs)
    
    return ManagerContext(**settings)


def get_system_info() -> Dict[str, Any]:
    """Get system information relevant to processing.
    
    Returns
    -------
    Dict[str, Any]
        System information
    """
    import psutil
    import platform
    
    return {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
        'cpu': {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        },
        'memory': {
            'total_gb': psutil.virtual_memory().total / (1024**3),
            'available_gb': psutil.virtual_memory().available / (1024**3),
            'percent_used': psutil.virtual_memory().percent,
        },
        'disk': {
            'total_gb': psutil.disk_usage('/').total / (1024**3),
            'free_gb': psutil.disk_usage('/').free / (1024**3),
            'percent_used': (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100,
        }
    }


def estimate_memory_requirements(operation: str, **params) -> Dict[str, float]:
    """Estimate memory requirements for common operations.
    
    Parameters
    ----------
    operation : str
        Operation type ('healpix_map', 'patch_extraction', etc.)
    **params
        Operation-specific parameters
        
    Returns
    -------
    Dict[str, float]
        Memory estimates in MB
    """
    estimates = {}
    
    if operation == 'healpix_map':
        nside = params.get('nside', 2048)
        npix = 12 * nside * nside
        dtype_size = params.get('dtype_size', 8)  # float64
        estimates['map_memory_mb'] = (npix * dtype_size) / (1024**2)
        estimates['processing_overhead_mb'] = estimates['map_memory_mb'] * 0.5
        
    elif operation == 'patch_extraction':
        n_patches = params.get('n_patches', 100)
        patch_size = params.get('patch_size', 2048)
        dtype_size = params.get('dtype_size', 4)  # float32
        patch_memory = n_patches * patch_size * patch_size * dtype_size
        estimates['patches_memory_mb'] = patch_memory / (1024**2)
        estimates['processing_overhead_mb'] = estimates['patches_memory_mb'] * 0.3
        
    elif operation == 'statistics':
        data_size_mb = params.get('data_size_mb', 100)
        estimates['data_memory_mb'] = data_size_mb
        estimates['computation_overhead_mb'] = data_size_mb * 2.0  # For FFTs, etc.
        
    else:
        logger.warning(f"Unknown operation for memory estimation: {operation}")
        estimates['unknown_operation_mb'] = 100  # Conservative estimate
    
    # Add base Python overhead
    estimates['python_overhead_mb'] = 50
    estimates['total_estimated_mb'] = sum(estimates.values())
    
    return estimates


__all__ = [
    # Individual managers
    'ResourceManager',
    'CheckpointManager', 
    'ProgressManager',
    'CacheManager',
    'LogManager',
    
    # Coordinated management
    'ManagerContext',
    'create_manager_context',
    
    # Utility functions
    'get_system_info',
    'estimate_memory_requirements',
]