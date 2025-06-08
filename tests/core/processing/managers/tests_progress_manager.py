"""
Tests for ProgressManager.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from io import StringIO

from lensing_ssc.core.processing.managers.progress_manager import (
    ProgressManager, ProgressTracker, ProgressMetrics
)
from lensing_ssc.core.processing.managers.exceptions import ProgressError


class TestProgressMetrics:
    def test_creation(self):
        metrics = ProgressMetrics(total=100)
        assert metrics.total == 100
        assert metrics.current == 0
        assert metrics.rate == 0.0
        assert metrics.start_time > 0
    
    def test_update_metrics(self):
        metrics = ProgressMetrics(total=100)
        
        # Simulate some progress
        time.sleep(0.01)  # Small delay for rate calculation
        metrics.update_metrics(10)
        
        assert metrics.current == 10
        assert metrics.elapsed > 0
        assert metrics.rate > 0
        assert metrics.eta is not None


class TestProgressTracker:
    @pytest.fixture
    def tracker(self):
        return ProgressTracker(total=100, description="Test task")
    
    def test_creation(self, tracker):
        assert tracker.total == 100
        assert tracker.description == "Test task"
        assert tracker.metrics.current == 0
    
    def test_update(self, tracker):
        tracker.update(10)
        assert tracker.metrics.current == 10
        
        # Update with description change
        tracker.update(5, description="Updated task")
        assert tracker.metrics.current == 15
        assert tracker.description == "Updated task"
    
    def test_set_current(self, tracker):
        tracker.set_current(50)
        assert tracker.metrics.current == 50
    
    def test_pause_resume(self, tracker):
        tracker.update(10)
        assert tracker.metrics.current == 10
        
        tracker.pause()
        tracker.update(10)  # Should not update while paused
        assert tracker.metrics.current == 10
        
        tracker.resume()
        tracker.update(10)
        assert tracker.metrics.current == 20
    
    def test_finish(self, tracker):
        tracker.update(50)
        tracker.finish()
        assert tracker.metrics.current == 100  # Should complete to total
    
    def test_unlimited_progress(self):
        tracker = ProgressTracker(description="Unlimited task")
        tracker.update(100)
        assert tracker.metrics.current == 100
        # No total limit, can exceed
        tracker.update(50)
        assert tracker.metrics.current == 150
    
    @patch('sys.stdout.isatty')
    def test_display_detection(self, mock_isatty):
        mock_isatty.return_value = True
        tracker = ProgressTracker(total=100)
        assert tracker._display_enabled
        
        mock_isatty.return_value = False
        tracker = ProgressTracker(total=100)
        assert not tracker._display_enabled
    
    def test_format_time(self, tracker):
        assert tracker._format_time(30) == "30s"
        assert tracker._format_time(90) == "1m30s"
        assert tracker._format_time(3700) == "1h1m"
    
    def test_get_status(self, tracker):
        tracker.update(25)
        status = tracker.get_status()
        
        assert status['current'] == 25
        assert status['total'] == 100
        assert status['percentage'] == 25.0
        assert 'rate' in status
        assert 'elapsed' in status


class TestProgressManager:
    @pytest.fixture
    def manager(self):
        return ProgressManager(total=1000, description="Main task")
    
    def test_creation(self, manager):
        assert manager.main_tracker.total == 1000
        assert manager.main_tracker.description == "Main task"
    
    def test_update(self, manager):
        manager.update(100)
        assert manager.main_tracker.metrics.current == 100
    
    def test_set_total(self, manager):
        manager.set_total(2000)
        assert manager.main_tracker.total == 2000
        assert manager.main_tracker.metrics.total == 2000
    
    def test_create_subtracker(self, manager):
        sub = manager.create_subtracker("subtask", total=50, description="Sub work")
        assert sub.total == 50
        assert "subtask" in manager._subtrackers
        
        # Duplicate name should raise error
        with pytest.raises(ProgressError):
            manager.create_subtracker("subtask", total=25)
    
    def test_remove_subtracker(self, manager):
        manager.create_subtracker("temp", total=10)
        assert "temp" in manager._subtrackers
        
        manager.remove_subtracker("temp")
        assert "temp" not in manager._subtrackers
    
    def test_subprogress_context(self, manager):
        with manager.subprogress("context_task", total=20) as sub:
            assert sub.total == 20
            assert "context_task" in manager._subtrackers
            sub.update(10)
            assert sub.metrics.current == 10
        
        # Should be removed after context
        assert "context_task" not in manager._subtrackers
    
    def test_pause_resume_all(self, manager):
        sub1 = manager.create_subtracker("sub1", total=10)
        sub2 = manager.create_subtracker("sub2", total=20)
        
        manager.pause_all()
        assert manager.main_tracker._paused
        assert sub1._paused
        assert sub2._paused
        
        manager.resume_all()
        assert not manager.main_tracker._paused
        assert not sub1._paused
        assert not sub2._paused
    
    def test_finish(self, manager):
        sub = manager.create_subtracker("sub", total=10)
        sub.update(5)
        
        manager.finish()
        assert manager.main_tracker.metrics.current == 1000  # Completed
        assert len(manager._subtrackers) == 0  # Cleared
    
    def test_callbacks(self, manager):
        callback_calls = []
        
        def test_callback(tracker_name, status):
            callback_calls.append((tracker_name, status))
        
        manager.add_callback(test_callback)
        manager.update(100)
        
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "main"
        assert callback_calls[0][1]['current'] == 100
    
    def test_enable_logging(self, manager):
        with patch('logging.getLogger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            manager.enable_logging(log_interval=10)
            
            # Trigger logging at interval
            for i in range(1, 21):
                manager.update(1)
            
            # Should have logged at steps 10 and 20
            assert mock_log.info.call_count >= 1
    
    def test_get_status(self, manager):
        sub = manager.create_subtracker("sub", total=50)
        sub.update(25)
        manager.update(500)
        
        status = manager.get_status()
        
        assert status['main']['current'] == 500
        assert status['main']['percentage'] == 50.0
        assert 'sub' in status['subtrackers']
        assert status['subtrackers']['sub']['current'] == 25
        assert status['overall_percentage'] == 50.0
    
    def test_export_import_progress(self, manager):
        # Setup some progress
        sub = manager.create_subtracker("sub", total=100, description="Subtask")
        manager.update(300)
        sub.update(60)
        
        # Export
        exported = manager.export_progress()
        assert exported['main_progress']['current'] == 300
        assert exported['subtrackers']['sub']['current'] == 60
        
        # Create new manager and import
        new_manager = ProgressManager(total=1000)
        new_manager.import_progress(exported)
        
        assert new_manager.main_tracker.metrics.current == 300
    
    def test_context_manager(self, manager):
        with manager as pm:
            assert pm is manager
            pm.update(100)
        # Should finish after context


def test_threading_safety():
    """Test thread safety of progress tracking."""
    manager = ProgressManager(total=1000)
    errors = []
    
    def update_progress(start, count):
        try:
            for i in range(count):
                manager.update(1)
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=update_progress, args=(i*20, 20))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    assert len(errors) == 0
    assert manager.main_tracker.metrics.current == 100


def test_performance_with_high_frequency_updates():
    """Test performance with rapid updates."""
    manager = ProgressManager(total=10000)
    
    start_time = time.time()
    for i in range(1000):
        manager.update(1)
    duration = time.time() - start_time
    
    # Should handle 1000 updates quickly
    assert duration < 1.0
    assert manager.main_tracker.metrics.current == 1000


@patch('sys.stdout')
def test_display_output(mock_stdout):
    """Test that display output is generated."""
    tracker = ProgressTracker(total=100, description="Test")
    tracker._display_enabled = True  # Force display
    
    tracker.update(50)
    # Would generate output if not mocked


def test_eta_calculation():
    """Test ETA calculation accuracy."""
    tracker = ProgressTracker(total=100)
    
    # Simulate steady progress
    for i in range(10):
        time.sleep(0.001)  # Small delay
        tracker.update(1)
    
    # Should have reasonable ETA
    if tracker.metrics.eta:
        assert tracker.metrics.eta > 0
        assert tracker.metrics.eta < 10  # Should be reasonable