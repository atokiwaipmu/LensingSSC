"""
Centralized logging management for processing operations.
"""

import os
import sys
import time
import logging
import logging.handlers
import threading
from typing import Optional, Dict, Any, List, Union, TextIO
from pathlib import Path
from datetime import datetime
import json
import traceback

from ...base.exceptions import ProcessingError


class LogManager:
    """Centralized logging manager for processing operations.
    
    Provides advanced logging features including:
    - Multiple output destinations (console, file, rotating files)
    - Structured logging with JSON format
    - Performance tracking
    - Error aggregation and reporting
    - Context-aware logging
    - Thread-safe operations
    
    Parameters
    ----------
    level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    file_path : str or Path, optional
        Log file path
    console_output : bool, optional
        Whether to output to console (default: True)
    json_format : bool, optional
        Whether to use JSON formatting (default: False)
    max_file_size_mb : int, optional
        Maximum log file size in MB before rotation (default: 10)
    backup_count : int, optional
        Number of backup files to keep (default: 5)
    enable_performance_logging : bool, optional
        Whether to enable performance metrics logging (default: True)
    """
    
    # Log level mappings
    LEVEL_MAP = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(
        self,
        level: str = 'INFO',
        file_path: Optional[Union[str, Path]] = None,
        console_output: bool = True,
        json_format: bool = False,
        max_file_size_mb: int = 10,
        backup_count: int = 5,
        enable_performance_logging: bool = True
    ):
        self.level = level.upper()
        self.file_path = Path(file_path) if file_path else None
        self.console_output = console_output
        self.json_format = json_format
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        self.enable_performance_logging = enable_performance_logging
        
        # Internal state
        self._loggers = {}
        self._handlers = []
        self._formatters = {}
        self._start_time = time.time()
        self._lock = threading.RLock()
        
        # Performance tracking
        self._performance_data = {}
        self._error_counts = {}
        self._log_counts = {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0}
        
        # Context stack for structured logging
        self._context_stack = threading.local()
        
        # Initialize logging system
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup the logging system with handlers and formatters."""
        with self._lock:
            # Create formatters
            self._create_formatters()
            
            # Setup root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(self.LEVEL_MAP[self.level])
            
            # Clear existing handlers to avoid duplicates
            root_logger.handlers.clear()
            
            # Add console handler
            if self.console_output:
                self._add_console_handler(root_logger)
            
            # Add file handler
            if self.file_path:
                self._add_file_handler(root_logger)
            
            # Add custom handler for statistics
            self._add_stats_handler(root_logger)
            
            # Set up performance logger if enabled
            if self.enable_performance_logging:
                self._setup_performance_logger()
    
    def _create_formatters(self) -> None:
        """Create different log formatters."""
        if self.json_format:
            self._formatters['json'] = JsonFormatter()
            self._formatters['console'] = JsonFormatter()
        else:
            # Standard text formatters
            detailed_format = (
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(filename)s:%(lineno)d - %(funcName)s - %(message)s'
            )
            simple_format = '%(asctime)s - %(levelname)s - %(message)s'
            
            self._formatters['detailed'] = logging.Formatter(
                detailed_format,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self._formatters['simple'] = logging.Formatter(
                simple_format,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self._formatters['console'] = logging.Formatter(
                '%(levelname)s - %(name)s - %(message)s'
            )
    
    def _add_console_handler(self, logger: logging.Logger) -> None:
        """Add console handler to logger."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.LEVEL_MAP[self.level])
        console_handler.setFormatter(self._formatters['console'])
        
        logger.addHandler(console_handler)
        self._handlers.append(console_handler)
    
    def _add_file_handler(self, logger: logging.Logger) -> None:
        """Add file handler with rotation to logger."""
        # Ensure log directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            self.file_path,
            maxBytes=self.max_file_size_mb * 1024 * 1024,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.LEVEL_MAP[self.level])
        
        # Use detailed formatter for files
        formatter_key = 'json' if self.json_format else 'detailed'
        file_handler.setFormatter(self._formatters[formatter_key])
        
        logger.addHandler(file_handler)
        self._handlers.append(file_handler)
    
    def _add_stats_handler(self, logger: logging.Logger) -> None:
        """Add custom handler for collecting statistics."""
        stats_handler = StatisticsHandler(self)
        stats_handler.setLevel(logging.DEBUG)  # Capture all levels
        
        logger.addHandler(stats_handler)
        self._handlers.append(stats_handler)
    
    def _setup_performance_logger(self) -> None:
        """Setup separate logger for performance metrics."""
        perf_logger = logging.getLogger('performance')
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False  # Don't propagate to root logger
        
        if self.file_path:
            # Create separate performance log file
            perf_file = self.file_path.parent / f"{self.file_path.stem}_performance.log"
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_file,
                maxBytes=self.max_file_size_mb * 1024 * 1024,
                backupCount=self.backup_count
            )
            perf_handler.setFormatter(JsonFormatter())
            perf_logger.addHandler(perf_handler)
            self._handlers.append(perf_handler)
        
        self._loggers['performance'] = perf_logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a named logger with proper configuration.
        
        Parameters
        ----------
        name : str
            Logger name
            
        Returns
        -------
        logging.Logger
            Configured logger
        """
        with self._lock:
            if name not in self._loggers:
                logger = logging.getLogger(name)
                # Logger inherits root configuration
                self._loggers[name] = logger
            
            return self._loggers[name]
    
    def log_performance(self, operation: str, duration: float, **metadata) -> None:
        """Log performance metrics.
        
        Parameters
        ----------
        operation : str
            Operation name
        duration : float
            Duration in seconds
        **metadata
            Additional metadata
        """
        if not self.enable_performance_logging:
            return
        
        with self._lock:
            # Update performance data
            if operation not in self._performance_data:
                self._performance_data[operation] = {
                    'count': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'avg_time': 0.0
                }
            
            perf_data = self._performance_data[operation]
            perf_data['count'] += 1
            perf_data['total_time'] += duration
            perf_data['min_time'] = min(perf_data['min_time'], duration)
            perf_data['max_time'] = max(perf_data['max_time'], duration)
            perf_data['avg_time'] = perf_data['total_time'] / perf_data['count']
            
            # Log to performance logger
            if 'performance' in self._loggers:
                log_data = {
                    'timestamp': datetime.now().isoformat(),
                    'operation': operation,
                    'duration': duration,
                    'statistics': perf_data.copy(),
                    **metadata
                }
                self._loggers['performance'].info(json.dumps(log_data))
    
    def push_context(self, **context) -> None:
        """Push logging context for structured logging.
        
        Parameters
        ----------
        **context
            Context key-value pairs
        """
        if not hasattr(self._context_stack, 'contexts'):
            self._context_stack.contexts = []
        
        self._context_stack.contexts.append(context)
    
    def pop_context(self) -> Dict[str, Any]:
        """Pop logging context.
        
        Returns
        -------
        Dict[str, Any]
            Popped context
        """
        if (hasattr(self._context_stack, 'contexts') and 
            self._context_stack.contexts):
            return self._context_stack.contexts.pop()
        return {}
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get current logging context.
        
        Returns
        -------
        Dict[str, Any]
            Combined context from stack
        """
        if not hasattr(self._context_stack, 'contexts'):
            return {}
        
        # Merge all contexts in stack
        combined = {}
        for context in self._context_stack.contexts:
            combined.update(context)
        
        return combined
    
    def log_with_context(self, level: str, message: str, logger_name: str = None, **extra) -> None:
        """Log message with current context.
        
        Parameters
        ----------
        level : str
            Log level
        message : str
            Log message
        logger_name : str, optional
            Logger name (default: root)
        **extra
            Additional context
        """
        logger = self.get_logger(logger_name) if logger_name else logging.getLogger()
        
        # Combine context
        context = self.get_current_context()
        context.update(extra)
        
        # Add context to message if not using JSON format
        if not self.json_format and context:
            context_str = ' | '.join(f'{k}={v}' for k, v in context.items())
            message = f"{message} | {context_str}"
        
        # Log with appropriate level
        log_level = self.LEVEL_MAP.get(level.upper(), logging.INFO)
        logger.log(log_level, message, extra=context if self.json_format else {})
    
    def log_exception(self, exception: Exception, logger_name: str = None, **context) -> None:
        """Log exception with full traceback and context.
        
        Parameters
        ----------
        exception : Exception
            Exception to log
        logger_name : str, optional
            Logger name
        **context
            Additional context
        """
        logger = self.get_logger(logger_name) if logger_name else logging.getLogger()
        
        # Track error counts
        exc_type = type(exception).__name__
        with self._lock:
            self._error_counts[exc_type] = self._error_counts.get(exc_type, 0) + 1
        
        # Prepare exception data
        exc_data = {
            'exception_type': exc_type,
            'exception_message': str(exception),
            'traceback': traceback.format_exc(),
            'error_count': self._error_counts[exc_type],
            **context
        }
        
        if self.json_format:
            logger.error("Exception occurred", extra=exc_data)
        else:
            logger.error(f"Exception occurred: {exc_type}: {exception}\n{traceback.format_exc()}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics.
        
        Returns
        -------
        Dict[str, Any]
            Logging statistics
        """
        with self._lock:
            uptime = time.time() - self._start_time
            
            return {
                'uptime_seconds': uptime,
                'log_counts': self._log_counts.copy(),
                'error_counts': self._error_counts.copy(),
                'performance_data': self._performance_data.copy(),
                'total_logs': sum(self._log_counts.values()),
                'logs_per_second': sum(self._log_counts.values()) / uptime if uptime > 0 else 0
            }
    
    def create_performance_context(self, operation: str, **metadata):
        """Create context manager for automatic performance logging.
        
        Parameters
        ----------
        operation : str
            Operation name
        **metadata
            Additional metadata
            
        Returns
        -------
        PerformanceContext
            Context manager
        """
        return PerformanceContext(self, operation, metadata)
    
    def set_level(self, level: str) -> None:
        """Change logging level.
        
        Parameters
        ----------
        level : str
            New logging level
        """
        self.level = level.upper()
        log_level = self.LEVEL_MAP[self.level]
        
        with self._lock:
            # Update all handlers
            for handler in self._handlers:
                handler.setLevel(log_level)
            
            # Update root logger
            logging.getLogger().setLevel(log_level)
    
    def close(self) -> None:
        """Close all handlers and cleanup."""
        with self._lock:
            for handler in self._handlers:
                try:
                    handler.close()
                except Exception as e:
                    print(f"Error closing handler: {e}")
            
            self._handlers.clear()
            self._loggers.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Parameters
        ----------
        record : logging.LogRecord
            Log record
            
        Returns
        -------
        str
            JSON formatted log message
        """
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message', 'exc_info', 'exc_text',
                'stack_info'
            }
        }
        
        if extra_fields:
            log_data['extra'] = extra_fields
        
        return json.dumps(log_data, default=str)


class StatisticsHandler(logging.Handler):
    """Custom handler for collecting logging statistics."""
    
    def __init__(self, log_manager):
        super().__init__()
        self.log_manager = log_manager
    
    def emit(self, record: logging.LogRecord) -> None:
        """Process log record for statistics.
        
        Parameters
        ----------
        record : logging.LogRecord
            Log record
        """
        try:
            with self.log_manager._lock:
                level = record.levelname
                if level in self.log_manager._log_counts:
                    self.log_manager._log_counts[level] += 1
        except Exception:
            # Ignore errors in statistics collection
            pass


class PerformanceContext:
    """Context manager for automatic performance logging."""
    
    def __init__(self, log_manager: LogManager, operation: str, metadata: Dict[str, Any]):
        self.log_manager = log_manager
        self.operation = operation
        self.metadata = metadata
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log performance."""
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            
            # Add exception info if occurred
            if exc_type is not None:
                self.metadata['exception'] = True
                self.metadata['exception_type'] = exc_type.__name__
            
            self.log_manager.log_performance(self.operation, duration, **self.metadata)


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[Union[str, Path]] = None,
    json_format: bool = False,
    **kwargs
) -> LogManager:
    """Convenience function to setup logging.
    
    Parameters
    ----------
    level : str
        Logging level
    log_file : str or Path, optional
        Log file path
    json_format : bool
        Whether to use JSON formatting
    **kwargs
        Additional LogManager arguments
        
    Returns
    -------
    LogManager
        Configured log manager
    """
    return LogManager(
        level=level,
        file_path=log_file,
        json_format=json_format,
        **kwargs
    )