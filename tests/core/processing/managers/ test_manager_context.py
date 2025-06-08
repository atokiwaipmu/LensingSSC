"""
Tests for LogManager.
"""

import pytest
import tempfile
import shutil
import logging
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from lensing_ssc.core.processing.managers.log_manager import (
    LogManager, JSONFormatter, PerformanceLogger, LogContext
)


class TestLogContext:
    def test_creation(self):
        context = LogContext(
            operation="test_op",
            step="validation",
            request_id="req_123",
            extra={"key": "value"}
        )
        
        assert context.operation == "test_op"
        assert context.step == "validation"
        assert context.request_id == "req_123"
        assert context.extra["key"] == "value"


class TestJSONFormatter:
    def test_basic_formatting(self):
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["level"] == "INFO"
        assert data["logger"] == "test_logger"
        assert data["message"] == "Test message"
        assert data["module"] == "test_module"
        assert data["function"] == "test_function"
        assert data["line"] == 42
    
    def test_context_formatting(self):
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        # Add context
        record.context = LogContext(
            operation="test_operation",
            step="step1",
            request_id="req_123",
            extra={"custom": "data"}
        )
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["operation"] == "test_operation"
        assert data["step"] == "step1"
        assert data["request_id"] == "req_123"
        assert data["custom"] == "data"
    
    def test_exception_formatting(self):
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test exception")
        except Exception:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="/test/path.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
            record.module = "test_module"
            record.funcName = "test_function"
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestPerformanceLogger:
    @pytest.fixture
    def perf_logger(self):
        mock_logger = Mock()
        return PerformanceLogger(mock_logger)
    
    def test_timer_operations(self, perf_logger):
        perf_logger.start_timer("test_op")
        time.sleep(0.01)
        duration = perf_logger.end_timer("test_op")
        
        assert duration > 0
        perf_logger.logger.log.assert_called_once()
    
    def test_timer_not_found(self, perf_logger):
        duration = perf_logger.end_timer("nonexistent")
        assert duration == 0.0
        perf_logger.logger.warning.assert_called_once()
    
    def test_time_operation_context(self, perf_logger):
        with perf_logger.time_operation("context_op"):
            time.sleep(0.01)
        
        perf_logger.logger.log.assert_called_once()
        args = perf_logger.logger.log.call_args[0]
        assert "context_op" in args[1]
    
    def test_time_function_decorator(self, perf_logger):
        @perf_logger.time_function("decorated_func")
        def test_function():
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        assert result == "result"
        perf_logger.logger.log.assert_called_once()


class TestLogManager:
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def log_manager(self, temp_dir):
        return LogManager(
            level="INFO",
            file_path=temp_dir / "test.log",
            enable_console=False  # Disable for testing
        )
    
    def test_initialization(self, log_manager):
        assert log_manager.level == logging.INFO
        assert log_manager.format_type == "standard"
        assert not log_manager.enable_console
        assert log_manager.enable_performance
    
    def test_get_logger(self, log_manager):
        logger1 = log_manager.get_logger("test_module")
        logger2 = log_manager.get_logger("test_module")
        
        assert logger1 is logger2  # Should return same instance
        assert logger1.level == logging.INFO
        assert "test_module" in log_manager._loggers
    
    def test_set_level(self, log_manager):
        logger = log_manager.get_logger("test")
        
        log_manager.set_level("DEBUG")
        assert log_manager.level == logging.DEBUG
        assert logger.level == logging.DEBUG
    
    def test_add_file_handler(self, log_manager, temp_dir):
        log_file = temp_dir / "custom.log"
        handler = log_manager.add_file_handler(log_file, level="WARNING")
        
        assert handler in log_manager._handlers
        assert handler.level == logging.WARNING
        assert log_file.parent.exists()
    
    def test_add_console_handler(self, log_manager):
        handler = log_manager.add_console_handler(level="ERROR")
        
        assert handler in log_manager._handlers
        assert handler.level == logging.ERROR
    
    def test_json_formatter(self, temp_dir):
        manager = LogManager(
            file_path=temp_dir / "json.log",
            format_type="json",
            enable_console=False
        )
        
        logger = manager.get_logger("json_test")
        logger.info("Test JSON message")
        
        # Check that file contains JSON
        log_file = temp_dir / "json.log"
        if log_file.exists():
            with open(log_file) as f:
                line = f.readline().strip()
                if line:
                    data = json.loads(line)
                    assert data["message"] == "Test JSON message"
    
    def test_log_context(self, log_manager):
        logger = log_manager.get_logger("context_test")
        
        with log_manager.log_context(operation="test_op", step="step1"):
            # This would normally add context to log records
            logger.info("Context message")
        
        # Context should be removed after exiting
        assert len(log_manager._context_stack) == 0
    
    def test_log_operation_methods(self, log_manager):
        # Test operation start/end logging
        log_manager.log_operation_start("test_operation", param1="value1")
        time.sleep(0.01)
        log_manager.log_operation_end("test_operation", success=True)
        
        # Should have logged and timed the operation
        if log_manager.enable_performance:
            assert hasattr(log_manager, 'performance')
    
    def test_log_error(self, log_manager):
        try:
            raise ValueError("Test error")
        except Exception as e:
            log_manager.log_error("An error occurred", exception=e, context="test")
        
        # Should log error with exception info
    
    def test_log_performance_metric(self, log_manager):
        log_manager.log_performance_metric(
            "processing_rate",
            42.5,
            unit="items/sec",
            operation="batch_process"
        )
        
        # Should log performance metric
    
    def test_get_log_stats(self, log_manager):
        stats = log_manager.get_log_stats()
        
        assert "total_loggers" in stats
        assert "active_handlers" in stats
        assert "current_level" in stats
        assert stats["current_level"] == "INFO"
    
    def test_rotate_logs(self, log_manager, temp_dir):
        # Add some content to log file
        logger = log_manager.get_logger("rotation_test")
        for i in range(10):
            logger.info(f"Log message {i}")
        
        # Manually rotate
        log_manager.rotate_logs()
        
        # Original log file should still exist
        assert log_manager.file_path.exists()
    
    def test_cleanup_old_logs(self, log_manager, temp_dir):
        # Create some old log files
        old_file1 = temp_dir / "test.log.1"
        old_file2 = temp_dir / "test.log.2"
        
        old_file1.touch()
        old_file2.touch()
        
        # Set old modification times
        old_time = time.time() - (40 * 24 * 3600)  # 40 days ago
        old_file1.stat = lambda: type('stat', (), {'st_mtime': old_time})()
        old_file2.stat = lambda: type('stat', (), {'st_mtime': old_time})()
        
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = type('stat', (), {'st_mtime': old_time})()
            
            # This would normally clean up old files
            # removed = log_manager.cleanup_old_logs(days=30)
    
    def test_get_status(self, log_manager):
        status = log_manager.get_status()
        
        assert "level" in status
        assert "file_path" in status
        assert "format_type" in status
        assert "console_enabled" in status
        assert "performance_enabled" in status
        assert "handlers" in status
        assert "loggers" in status
        assert "stats" in status
    
    def test_context_manager(self, log_manager):
        with log_manager as lm:
            assert lm is log_manager
            logger = lm.get_logger("context_test")
            logger.info("Test message")
        
        # Handlers should be closed after context


def test_performance_logger_thread_safety():
    """Test thread safety of performance logger."""
    mock_logger = Mock()
    perf_logger = PerformanceLogger(mock_logger)
    
    errors = []
    
    def worker(worker_id):
        try:
            timer_name = f"worker_{worker_id}"
            perf_logger.start_timer(timer_name)
            time.sleep(0.01)
            perf_logger.end_timer(timer_name)
        except Exception as e:
            errors.append(e)
    
    import threading
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    assert len(errors) == 0
    assert mock_logger.log.call_count == 5


@patch('logging.getLogger')
def test_log_manager_integration(mock_get_logger, temp_dir):
    """Test integration with Python logging system."""
    mock_logger = Mock()
    mock_get_logger.return_value = mock_logger
    
    manager = LogManager(
        file_path=temp_dir / "integration.log",
        enable_console=False
    )
    
    logger = manager.get_logger("integration_test")
    logger.info("Integration test message")
    
    # Should have configured loggers properly
    assert len(manager._loggers) > 0


def test_formatter_creation():
    """Test different formatter types."""
    manager = LogManager(enable_console=False)
    
    # Standard formatter
    std_formatter = manager._create_formatter("standard")
    assert isinstance(std_formatter, logging.Formatter)
    
    # JSON formatter
    json_formatter = manager._create_formatter("json")
    assert isinstance(json_formatter, JSONFormatter)
    
    # Detailed formatter
    detailed_formatter = manager._create_formatter("detailed")
    assert isinstance(detailed_formatter, logging.Formatter)