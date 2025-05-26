import pytest
from unittest.mock import patch, MagicMock, mock_open, call
import logging
from pathlib import Path
import time
import json
import numpy as np # For PerformanceMonitor

from lensing_ssc.core.preprocessing.utils import (
    setup_logging,
    ProgressTracker,
    PerformanceMonitor,
    MemoryManager,
    CheckpointManager,
    extract_seed_from_path,
    format_memory_size,
    format_duration,
    BatchProcessor
)

# --- Tests for setup_logging (existing tests, slightly adapted if needed) ---
@pytest.fixture
def mock_logger():
    logger = logging.getLogger("lensing_ssc") # or a more specific logger name if used
    # Ensure it's clean for each test that uses it indirectly
    logger.handlers = [] 
    return logger

@patch('logging.basicConfig')
@patch('logging.FileHandler')
@patch('logging.StreamHandler')
def test_setup_logging_basic_setup(
    MockStreamHandler, MockFileHandler, MockBasicConfig, mock_logger
):
    """Test basic logging setup with INFO level and console output."""
    mock_stream_instance = MockStreamHandler.return_value
    
    setup_logging(log_level="INFO", log_file=None)
    
    MockStreamHandler.assert_called_once()
    MockFileHandler.assert_not_called()
    MockBasicConfig.assert_called_once()
    args, kwargs = MockBasicConfig.call_args
    assert kwargs['level'] == logging.INFO
    assert mock_stream_instance in kwargs['handlers']
    assert 'force' in kwargs and kwargs['force'] is True

@patch('logging.basicConfig')
@patch('logging.FileHandler')
@patch('logging.StreamHandler')
def test_setup_logging_with_file(
    MockStreamHandler, MockFileHandler, MockBasicConfig, tmp_path, mock_logger
):
    """Test logging setup with a log file specified."""
    log_file = tmp_path / "test.log"
    mock_file_instance = MockFileHandler.return_value
    mock_stream_instance = MockStreamHandler.return_value
    
    setup_logging(log_level="DEBUG", log_file=log_file)
    
    MockFileHandler.assert_called_once_with(log_file)
    MockStreamHandler.assert_called_once()
    MockBasicConfig.assert_called_once()
    args, kwargs = MockBasicConfig.call_args
    assert kwargs['level'] == logging.DEBUG
    assert mock_file_instance in kwargs['handlers']
    assert mock_stream_instance in kwargs['handlers']

def test_setup_logging_invalid_level(mock_logger):
    """Test setup_logging with an invalid log level string."""
    with pytest.raises(AttributeError): # getattr will fail
        setup_logging(log_level="INVALID_LEVEL")

@patch('logging.getLogger')
def test_setup_logging_external_library_levels(mock_get_logger, mock_logger):
    """Test that external library log levels are set."""
    mock_mpl_logger = MagicMock()
    mock_numba_logger = MagicMock()

    def get_logger_side_effect(name):
        if name == 'matplotlib': return mock_mpl_logger
        if name == 'numba': return mock_numba_logger
        return MagicMock() # Default mock for other loggers
    mock_get_logger.side_effect = get_logger_side_effect

    with patch('logging.basicConfig'): # Don't care about basicConfig here
        setup_logging()
    
    mock_mpl_logger.setLevel.assert_called_once_with(logging.WARNING)
    mock_numba_logger.setLevel.assert_called_once_with(logging.WARNING)


# --- Tests for ProgressTracker ---
@patch('lensing_ssc.core.preprocessing.utils.tqdm')
def test_progress_tracker_init(mock_tqdm):
    mock_pbar_instance = mock_tqdm.return_value
    tracker = ProgressTracker(total_operations=100, description="Testing", unit="tests")
    mock_tqdm.assert_called_once_with(total=100, desc="Testing", unit="tests")
    assert tracker.pbar == mock_pbar_instance

@patch('lensing_ssc.core.preprocessing.utils.tqdm')
def test_progress_tracker_update(mock_tqdm):
    mock_pbar_instance = mock_tqdm.return_value
    tracker = ProgressTracker(10, "Desc")
    tracker.update(2, info="Step 2")
    mock_pbar_instance.update.assert_called_once_with(2)
    mock_pbar_instance.set_postfix_str.assert_called_once_with("Step 2")
    tracker.close()
    mock_pbar_instance.close.assert_called_once()

# --- Tests for PerformanceMonitor ---
def test_performance_monitor_record_timing():
    pm = PerformanceMonitor()
    pm.record_timing("op1", 0.5)
    pm.record_timing("op1", 0.7)
    pm.record_timing("op2", 1.2)
    assert "op1" in pm.metrics and len(pm.metrics["op1"]) == 2
    assert "op2" in pm.metrics and len(pm.metrics["op2"]) == 1

@patch("time.perf_counter", side_effect=[1.0, 1.5]) # Start, End
def test_performance_monitor_timer_context(mock_perf_counter):
    pm = PerformanceMonitor()
    with pm.timer("timed_op"):
        pass # Simulate work
    assert "timed_op" in pm.metrics
    assert pm.metrics["timed_op"][0] == 0.5 # 1.5 - 1.0

def test_performance_monitor_get_summary():
    pm = PerformanceMonitor()
    pm.record_timing("op_sum", 0.1)
    pm.record_timing("op_sum", 0.2)
    pm.record_timing("op_sum", 0.3)
    summary = pm.get_summary()
    assert "op_sum" in summary
    stats = summary["op_sum"]
    assert stats['count'] == 3
    assert np.isclose(stats['mean'], 0.2)
    assert np.isclose(stats['total'], 0.6)

@patch("logging.info")
def test_performance_monitor_log_summary(mock_log_info):
    pm = PerformanceMonitor()
    pm.record_timing("logged_op", 0.25)
    pm.log_summary()
    assert mock_log_info.call_count >= 2 # Header + one op line
    # Check if logged_op string is in any of the log calls
    assert any("logged_op" in call_args[0][0] for call_args in mock_log_info.call_args_list)


# --- Tests for MemoryManager ---
@patch("gc.collect")
def test_memory_manager_cleanup(mock_gc_collect):
    mm = MemoryManager()
    mm.cached_chunks["key"] = "value" # Add something to cache
    mm.cleanup_memory()
    mock_gc_collect.assert_called_once()
    assert not mm.cached_chunks # Cache should be cleared

@patch("psutil.Process")
def test_memory_manager_get_memory_usage(MockPsutilProcess):
    mock_process_instance = MockPsutilProcess.return_value
    mock_process_instance.memory_info.return_value.rss = 2048 * 1024 # 2MB
    mm = MemoryManager()
    assert mm.get_memory_usage_mb() == 2.0

# --- Tests for CheckpointManager ---
@pytest.fixture
def checkpoint_dir(tmp_path):
    chk_dir = tmp_path / "chk_data"
    # CheckpointManager will create parent if it doesn't exist during save
    return chk_dir

def test_checkpoint_manager_init(checkpoint_dir):
    cm = CheckpointManager(checkpoint_dir)
    assert cm.datadir == checkpoint_dir
    assert cm.checkpoint_file == checkpoint_dir / "processing_checkpoint.json"

@patch("json.dump")
@patch("pathlib.Path.rename")
@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.mkdir")
def test_checkpoint_manager_save(mock_mkdir, mock_file_open, mock_rename, mock_json_dump, checkpoint_dir):
    cm = CheckpointManager(checkpoint_dir)
    completed = [1, 2, 3]
    failed = [4]
    metadata = {"key": "value"}
    cm.save_checkpoint(completed, failed, metadata)
    
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    temp_file_path = cm.checkpoint_file.with_suffix('.tmp')
    mock_file_open.assert_called_once_with(temp_file_path, 'w')
    mock_json_dump.assert_called_once()
    saved_data = mock_json_dump.call_args[0][0]
    assert saved_data["completed_sheets"] == completed
    assert saved_data["failed_sheets"] == failed
    assert saved_data["metadata"] == metadata
    assert "timestamp" in saved_data
    mock_rename.assert_called_once_with(cm.checkpoint_file)

@patch("builtins.open", new_callable=mock_open)
@patch("json.load")
def test_checkpoint_manager_load_exists_valid(mock_json_load, mock_file_open, checkpoint_dir):
    cm = CheckpointManager(checkpoint_dir)
    mock_checkpoint_data = {"completed_sheets": [1], "timestamp": time.time()}
    mock_json_load.return_value = mock_checkpoint_data
    
    with patch.object(Path, 'exists', return_value=True):
        data = cm.load_checkpoint()
        assert data == mock_checkpoint_data
        mock_file_open.assert_called_once_with(cm.checkpoint_file, 'r')

def test_checkpoint_manager_load_not_exists(checkpoint_dir):
    cm = CheckpointManager(checkpoint_dir)
    with patch.object(Path, 'exists', return_value=False):
        assert cm.load_checkpoint() is None

@patch("pathlib.Path.unlink")
def test_checkpoint_manager_clear(mock_unlink, checkpoint_dir):
    cm = CheckpointManager(checkpoint_dir)
    with patch.object(Path, 'exists', return_value=True):
        cm.clear_checkpoint()
        mock_unlink.assert_called_once()

# --- Tests for extract_seed_from_path ---
@pytest.mark.parametrize("path_str, expected_seed", [
    ("/data/sims/run_s123_other/stuff", 123),
    ("s456_another_format", 456),
    ("/no/seed/here", 0),
    ("s0", 0),
    ("s", 0) # No digits after s
])
def test_extract_seed_from_path(path_str, expected_seed):
    assert extract_seed_from_path(Path(path_str)) == expected_seed

# --- Tests for format_memory_size ---
@pytest.mark.parametrize("bytes_val, expected_str", [
    (500, "500.0 B"),
    (2048, "2.0 KB"),
    (3 * 1024 * 1024, "3.0 MB"),
    (1.5 * 1024**3, "1.5 GB")
])
def test_format_memory_size(bytes_val, expected_str):
    assert format_memory_size(bytes_val) == expected_str

# --- Tests for format_duration ---
@pytest.mark.parametrize("seconds, expected_str", [
    (30.5, "30.5s"),
    (90, "1m 30s"),
    (3661, "1h 1m")
])
def test_format_duration(seconds, expected_str):
    assert format_duration(seconds) == expected_str

# --- Tests for BatchProcessor ---
@patch("lensing_ssc.core.preprocessing.utils.ProgressTracker")
def test_batch_processor_success(MockProgressTracker):
    items_to_process = list(range(5))
    # Mock process_func: adds 1 to each item
    mock_process_func = MagicMock(side_effect=lambda x: x + 1)
    
    bp = BatchProcessor(batch_size=2)
    successful, failed = bp.process_batches(items_to_process, mock_process_func)
    
    assert successful == [1, 2, 3, 4, 5]
    assert not failed
    assert mock_process_func.call_count == 5
    MockProgressTracker.assert_called_once_with(3, "Processing", "batch") # 5 items, batch 2 -> 3 batches

@patch("lensing_ssc.core.preprocessing.utils.ProgressTracker")
def test_batch_processor_with_failures_and_retries(MockProgressTracker):
    items = [1, 2, 3, 4, 5]
    # Process func: fails for 2, succeeds for others, succeeds for 2 on retry
    call_counts = {item: 0 for item in items}
    def process_func_with_retry(item):
        call_counts[item] += 1
        if item == 2 and call_counts[item] < 2: # Fails first time for item 2
            raise ValueError("Failed item 2")
        return item * 10

    bp = BatchProcessor(batch_size=2, max_retries=1) # Allow 1 retry
    successful, failed = bp.process_batches(items, process_func_with_retry)

    assert sorted(successful) == [10, 20, 30, 40, 50]
    assert not failed
    assert call_counts[2] == 2 # Item 2 was processed twice (original + 1 retry)
    assert call_counts[1] == 1 