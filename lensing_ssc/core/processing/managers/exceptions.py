"""
Additional exception classes for processing managers.

This module extends the base exception hierarchy with manager-specific
error types for resource management, checkpointing, progress tracking,
caching, and logging operations.
"""

from typing import Optional, Any, Dict
from ...base.exceptions import LensingSSCError


class ResourceError(LensingSSCError):
    """Raised when resource management operations fail.
    
    This exception is used for errors in resource monitoring,
    limit enforcement, or system resource access.
    """
    
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 limit_exceeded: Optional[str] = None, **kwargs):
        """Initialize resource error.
        
        Parameters
        ----------
        message : str
            Resource error message
        resource_type : str, optional
            Type of resource (memory, cpu, disk)
        limit_exceeded : str, optional
            Specific limit that was exceeded
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if resource_type is not None:
            details['resource_type'] = resource_type
        if limit_exceeded is not None:
            details['limit_exceeded'] = limit_exceeded
        
        super().__init__(message, details=details, **kwargs)
        self.resource_type = resource_type
        self.limit_exceeded = limit_exceeded


class CheckpointError(LensingSSCError):
    """Raised when checkpoint operations fail.
    
    This exception is used for errors in checkpoint save/load operations,
    validation failures, or checkpoint file corruption.
    """
    
    def __init__(self, message: str, checkpoint_file: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        """Initialize checkpoint error.
        
        Parameters
        ----------
        message : str
            Checkpoint error message
        checkpoint_file : str, optional
            Path to the problematic checkpoint file
        operation : str, optional
            Checkpoint operation that failed (save, load, validate)
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if checkpoint_file is not None:
            details['checkpoint_file'] = checkpoint_file
        if operation is not None:
            details['operation'] = operation
        
        super().__init__(message, details=details, **kwargs)
        self.checkpoint_file = checkpoint_file
        self.operation = operation


class ProgressError(LensingSSCError):
    """Raised when progress tracking operations fail.
    
    This exception is used for errors in progress management,
    tracker creation, or progress display operations.
    """
    
    def __init__(self, message: str, tracker_name: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        """Initialize progress error.
        
        Parameters
        ----------
        message : str
            Progress error message
        tracker_name : str, optional
            Name of the progress tracker
        operation : str, optional
            Progress operation that failed
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if tracker_name is not None:
            details['tracker_name'] = tracker_name
        if operation is not None:
            details['operation'] = operation
        
        super().__init__(message, details=details, **kwargs)
        self.tracker_name = tracker_name
        self.operation = operation


class CacheError(LensingSSCError):
    """Raised when cache operations fail.
    
    This exception is used for errors in cache management,
    eviction policies, or cache storage operations.
    """
    
    def __init__(self, message: str, cache_key: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        """Initialize cache error.
        
        Parameters
        ----------
        message : str
            Cache error message
        cache_key : str, optional
            Cache key involved in the error
        operation : str, optional
            Cache operation that failed (get, put, delete)
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if cache_key is not None:
            details['cache_key'] = cache_key
        if operation is not None:
            details['operation'] = operation
        
        super().__init__(message, details=details, **kwargs)
        self.cache_key = cache_key
        self.operation = operation


class LoggingError(LensingSSCError):
    """Raised when logging operations fail.
    
    This exception is used for errors in log configuration,
    handler setup, or log formatting operations.
    """
    
    def __init__(self, message: str, logger_name: Optional[str] = None,
                 handler_type: Optional[str] = None, **kwargs):
        """Initialize logging error.
        
        Parameters
        ----------
        message : str
            Logging error message
        logger_name : str, optional
            Name of the logger
        handler_type : str, optional
            Type of log handler (file, console, etc.)
        **kwargs
            Additional arguments for base class
        """
        details = kwargs.pop('details', {})
        if logger_name is not None:
            details['logger_name'] = logger_name
        if handler_type is not None:
            details['handler_type'] = handler_type
        
        super().__init__(message, details=details, **kwargs)
        self.logger_name = logger_name
        self.handler_type = handler_type