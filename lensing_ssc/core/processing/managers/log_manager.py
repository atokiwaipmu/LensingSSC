"""
Log manager for centralized logging configuration and management.

Provides structured logging with multiple handlers, performance tracking,
and comprehensive log management for processing operations.
"""

import logging
import logging.handlers
import json
import time
import threading
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
import functools

from .exceptions import LoggingError
from ...config.settings import ProcessingConfig


@dataclass
class LogContext:
    """Context information for structured logging."""
    operation: Optional[str] = None
    step: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add context if available
        if hasattr(record, 'context'):
            context = record.context
            if context.operation:
                log_entry['operation'] = context.operation
            if context.step:
                log_entry['step'] = context.step
            if context.request_id:
                log_entry['request_id'] = context.request_id
            if context.extra:
                log_entry.update(context.extra)
        
        # Add exception info
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._timers: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def start_timer(self, name: str) -> None:
        """Start timing operation."""
        with self._lock:
            self._timers[name] = time.time()
    
    def end_timer(self, name: str, log_level: int = logging.INFO) -> float:
        """End timing and log duration."""
        with self._lock:
            if name not in self._timers:
                self.logger.warning(f"Timer '{name}' not found")
                return 0.0
            
            duration = time.time() - self._timers.pop(name)
            self.logger.log(log_level, f"Operation '{name}' completed in {duration:.3f}s")
            return duration
    
    @contextmanager
    def time_operation(self, name: str, log_level: int = logging.INFO):
        """Context manager for timing operations."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name, log_level)
    
    def time_function(self, name: Optional[str] = None, log_level: int = logging.INFO):
        """Decorator for timing functions."""
        def decorator(func):
            timer_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.time_operation(timer_name, log_level):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


class LogManager:
    """Centralized logging manager with structured logging support."""
    
    def __init__(
        self,
        level: Union[str, int] = logging.INFO,
        file_path: Optional[Union[str, Path]] = None,
        max_file_size: int = 10,  # MB
        backup_count: int = 5,
        format_type: str = "standard",
        enable_console: bool = True,
        enable_performance: bool = True,
        config: Optional[ProcessingConfig] = None
    ):
        """Initialize log manager.
        
        Parameters
        ----------
        level : str or int
            Logging level
        file_path : str or Path, optional
            Log file path
        max_file_size : int
            Max file size in MB for rotation
        backup_count : int
            Number of backup files to keep
        format_type : str
            Format type ("standard", "json", "detailed")
        enable_console : bool
            Enable console logging
        enable_performance : bool
            Enable performance logging
        config : ProcessingConfig, optional
            Configuration object
        """
        # Load from config
        if config:
            level = getattr(logging, config.log_level.upper(), logging.INFO)
            file_path = file_path or config.log_file
        
        self.level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.file_path = Path(file_path) if file_path else None
        self.max_file_size = max_file_size * 1024 * 1024  # Convert to bytes
        self.backup_count = backup_count
        self.format_type = format_type
        self.enable_console = enable_console
        self.enable_performance = enable_performance
        
        # State
        self._loggers: Dict[str, logging.Logger] = {}
        self._handlers: List[logging.Handler] = []
        self._context_stack: List[LogContext] = []
        self._lock = threading.Lock()
        
        # Setup root logger
        self._setup_root_logger()
        
        # Performance logger
        if self.enable_performance:
            self.performance = PerformanceLogger(self.get_logger('performance'))
        
        logging.info("LogManager initialized")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create logger with consistent configuration."""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self.level)
            logger.propagate = True
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def set_level(self, level: Union[str, int]) -> None:
        """Set logging level for all loggers."""
        self.level = level if isinstance(level, int) else getattr(logging, level.upper())
        
        # Update all loggers
        for logger in self._loggers.values():
            logger.setLevel(self.level)
        
        # Update root logger
        logging.getLogger().setLevel(self.level)
    
    def add_file_handler(
        self,
        file_path: Union[str, Path],
        level: Optional[Union[str, int]] = None,
        format_type: Optional[str] = None
    ) -> logging.Handler:
        """Add file handler with rotation."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        
        if level:
            handler_level = level if isinstance(level, int) else getattr(logging, level.upper())
            handler.setLevel(handler_level)
        else:
            handler.setLevel(self.level)
        
        formatter = self._create_formatter(format_type or self.format_type)
        handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(handler)
        self._handlers.append(handler)
        
        return handler
    
    def add_console_handler(
        self,
        level: Optional[Union[str, int]] = None,
        format_type: Optional[str] = None
    ) -> logging.Handler:
        """Add console handler."""
        handler = logging.StreamHandler()
        
        if level:
            handler_level = level if isinstance(level, int) else getattr(logging, level.upper())
            handler.setLevel(handler_level)
        else:
            handler.setLevel(self.level)
        
        formatter = self._create_formatter(format_type or self.format_type)
        handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(handler)
        self._handlers.append(handler)
        
        return handler
    
    @contextmanager
    def log_context(
        self,
        operation: Optional[str] = None,
        step: Optional[str] = None,
        request_id: Optional[str] = None,
        **extra
    ):
        """Context manager for structured logging."""
        context = LogContext(
            operation=operation,
            step=step,
            request_id=request_id,
            extra=extra
        )
        
        self._context_stack.append(context)
        
        # Create context filter
        old_makeRecord = logging.getLoggerClass().makeRecord
        
        def makeRecord_with_context(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
            record = old_makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)
            if self._context_stack:
                record.context = self._context_stack[-1]
            return record
        
        logging.getLoggerClass().makeRecord = makeRecord_with_context
        
        try:
            yield
        finally:
            self._context_stack.pop()
            logging.getLoggerClass().makeRecord = old_makeRecord
    
    def log_operation_start(self, operation: str, **details) -> None:
        """Log operation start."""
        logger = self.get_logger('operations')
        logger.info(f"Starting operation: {operation}", extra={'details': details})
        
        if self.enable_performance:
            self.performance.start_timer(operation)
    
    def log_operation_end(self, operation: str, success: bool = True, **details) -> None:
        """Log operation completion."""
        logger = self.get_logger('operations')
        status = "completed" if success else "failed"
        logger.info(f"Operation {status}: {operation}", extra={'details': details})
        
        if self.enable_performance:
            self.performance.end_timer(operation)
    
    def log_error(self, message: str, exception: Optional[Exception] = None, **context) -> None:
        """Log error with context."""
        logger = self.get_logger('errors')
        if exception:
            logger.error(message, exc_info=exception, extra=context)
        else:
            logger.error(message, extra=context)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "", **context) -> None:
        """Log performance metric."""
        if not self.enable_performance:
            return
        
        logger = self.get_logger('performance')
        logger.info(f"Metric {metric_name}: {value} {unit}", extra={
            'metric_name': metric_name,
            'metric_value': value,
            'metric_unit': unit,
            **context
        })
    
    def _setup_root_logger(self) -> None:
        """Setup root logger configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler
        if self.enable_console:
            self.add_console_handler()
        
        # Add file handler
        if self.file_path:
            self.add_file_handler(self.file_path)
    
    def _create_formatter(self, format_type: str) -> logging.Formatter:
        """Create formatter based on type."""
        if format_type == "json":
            return JSONFormatter()
        elif format_type == "detailed":
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            return logging.Formatter(fmt)
        else:  # standard
            fmt = '%(asctime)s - %(levelname)s - %(message)s'
            return logging.Formatter(fmt)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            'total_loggers': len(self._loggers),
            'active_handlers': len(self._handlers),
            'current_level': logging.getLevelName(self.level),
            'context_depth': len(self._context_stack),
        }
        
        # File handler stats
        for handler in self._handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                try:
                    stats['log_file_size'] = Path(handler.baseFilename).stat().st_size
                    break
                except Exception:
                    pass
        
        return stats
    
    def rotate_logs(self) -> None:
        """Manually rotate log files."""
        for handler in self._handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.doRollover()
    
    def cleanup_old_logs(self, days: int = 30) -> int:
        """Clean up old log files."""
        if not self.file_path:
            return 0
        
        log_dir = self.file_path.parent
        if not log_dir.exists():
            return 0
        
        cutoff_time = time.time() - (days * 24 * 3600)
        removed_count = 0
        
        # Find old log files
        pattern = f"{self.file_path.stem}.*"
        for log_file in log_dir.glob(pattern):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    removed_count += 1
            except Exception as e:
                logging.warning(f"Failed to remove old log file {log_file}: {e}")
        
        return removed_count
    
    def get_status(self) -> Dict[str, Any]:
        """Get log manager status."""
        return {
            'level': logging.getLevelName(self.level),
            'file_path': str(self.file_path) if self.file_path else None,
            'format_type': self.format_type,
            'console_enabled': self.enable_console,
            'performance_enabled': self.enable_performance,
            'handlers': len(self._handlers),
            'loggers': len(self._loggers),
            'stats': self.get_log_stats(),
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Close all handlers
        for handler in self._handlers:
            handler.close()
        self._handlers.clear()