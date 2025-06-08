"""
Tests for ResourceManager.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from lensing_ssc.core.processing.managers.resource_manager import (
    ResourceManager, ResourceLimits, ResourceUsage
)
from lensing_ssc.core.processing.managers.exceptions import ResourceError


class TestResourceUsage:
    def test_usage_creation(self):
        usage = ResourceUsage(
            memory_mb=1000.0,
            memory_percent=50.0,
            cpu_percent=25.0,
            disk_gb=100.0,
            disk_percent=30.0,
            swap_percent=10.0
        )
        assert usage.memory_mb == 1000.0
        assert usage.cpu_percent == 25.0
        assert usage.timestamp > 0
    
    def test_exceeds_limits(self):
        usage = ResourceUsage(
            memory_mb=1000.0,
            memory_percent=50.0,
            cpu_percent=80.0,
            disk_gb=100.0,
            disk_percent=30.0,
            swap_percent=20.0
        )
        
        limits = ResourceLimits(
            memory_mb=800,
            cpu_percent=70.0,
            disk_percent=40.0
        )
        
        exceeded = usage.exceeds_limits(limits)
        assert exceeded['memory_mb'] is True
        assert exceeded['cpu_percent'] is True
        assert 'disk_percent' not in exceeded  # Under limit


class TestResourceManager:
    @pytest.fixture
    def manager(self):
        return ResourceManager(
            memory_limit_mb=1000,
            cpu_limit_percent=80.0,
            monitor_interval=0.1
        )
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    @patch('psutil.swap_memory')
    def test_get_current_usage(self, mock_swap, mock_disk, mock_cpu, mock_memory, manager):
        # Mock system info
        mock_memory.return_value = Mock(
            total=8*1024**3, used=2*1024**3, available=6*1024**3, percent=25.0
        )
        mock_cpu.return_value = 30.0
        mock_disk.return_value = Mock(
            total=1000*1024**3, used=300*1024**3, free=700*1024**3
        )
        mock_swap.return_value = Mock(percent=5.0)
        
        usage = manager.get_current_usage()
        
        assert usage.memory_mb == pytest.approx(2048, rel=0.1)
        assert usage.memory_percent == 25.0
        assert usage.cpu_percent == 30.0
        assert usage.disk_gb == pytest.approx(300, rel=0.1)
        assert usage.swap_percent == 5.0
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    @patch('psutil.swap_memory')
    def test_check_limits_pass(self, mock_swap, mock_disk, mock_cpu, mock_memory, manager):
        # Mock usage under limits
        mock_memory.return_value = Mock(
            total=8*1024**3, used=512*1024**2, available=7.5*1024**3, percent=6.25
        )
        mock_cpu.return_value = 50.0
        mock_disk.return_value = Mock(
            total=1000*1024**3, used=100*1024**3, free=900*1024**3
        )
        mock_swap.return_value = Mock(percent=2.0)
        
        exceeded = manager.check_limits(raise_on_exceed=False)
        assert not exceeded
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    @patch('psutil.swap_memory')
    def test_check_limits_exceeded(self, mock_swap, mock_disk, mock_cpu, mock_memory, manager):
        # Mock usage over limits
        mock_memory.return_value = Mock(
            total=8*1024**3, used=2*1024**3, available=6*1024**3, percent=25.0
        )
        mock_cpu.return_value = 90.0  # Over 80% limit
        mock_disk.return_value = Mock(
            total=1000*1024**3, used=300*1024**3, free=700*1024**3
        )
        mock_swap.return_value = Mock(percent=5.0)
        
        with pytest.raises(ResourceError):
            manager.check_limits(raise_on_exceed=True)
        
        exceeded = manager.check_limits(raise_on_exceed=False)
        assert 'cpu_percent' in exceeded
    
    def test_callbacks(self, manager):
        warning_called = threading.Event()
        limit_called = threading.Event()
        
        def warning_callback(usage, warnings):
            warning_called.set()
        
        def limit_callback(usage, exceeded):
            limit_called.set()
        
        manager.add_warning_callback(warning_callback)
        manager.add_limit_callback(limit_callback)
        
        # Mock high usage to trigger callbacks
        with patch.object(manager, 'get_current_usage') as mock_usage:
            mock_usage.return_value = ResourceUsage(
                memory_mb=1200.0,  # Over limit
                memory_percent=80.0,
                cpu_percent=85.0,  # Over limit
                disk_gb=100.0,
                disk_percent=30.0,
                swap_percent=10.0
            )
            
            manager.check_limits(raise_on_exceed=False)
            assert limit_called.is_set()
    
    def test_monitoring(self, manager):
        assert not manager._monitoring
        
        manager.start_monitoring()
        assert manager._monitoring
        time.sleep(0.2)  # Let monitor run
        
        manager.stop_monitoring()
        assert not manager._monitoring
    
    def test_context_manager(self, manager):
        with manager.monitor_context() as ctx:
            assert isinstance(ctx, ResourceManager)
            # Should start monitoring if not already running
    
    def test_get_memory_usage(self, manager):
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.swap_memory') as mock_swap:
            
            mock_memory.return_value = Mock(
                total=8*1024**3, used=2*1024**3, available=6*1024**3,
                free=6*1024**3, percent=25.0
            )
            mock_swap.return_value = Mock(
                total=4*1024**3, used=512*1024**2, percent=12.5
            )
            
            usage = manager.get_memory_usage()
            assert 'total_mb' in usage
            assert 'used_mb' in usage
            assert 'percent' in usage
            assert 'swap_total_mb' in usage
    
    def test_get_status(self, manager):
        status = manager.get_status()
        
        assert 'monitoring' in status
        assert 'current_usage' in status
        assert 'limits' in status
        assert 'callback_count' in status
        
        # Check limits are properly reflected
        assert status['limits']['memory_mb'] == 1000
        assert status['limits']['cpu_percent'] == 80.0


class TestResourceLimits:
    def test_creation(self):
        limits = ResourceLimits(
            memory_mb=1000,
            cpu_percent=80.0,
            disk_gb=500.0
        )
        assert limits.memory_mb == 1000
        assert limits.cpu_percent == 80.0
        assert limits.disk_gb == 500.0
        assert limits.memory_percent is None


@pytest.fixture
def mock_psutil():
    """Mock psutil functions for testing."""
    with patch('psutil.virtual_memory') as mock_mem, \
         patch('psutil.cpu_percent') as mock_cpu, \
         patch('psutil.disk_usage') as mock_disk, \
         patch('psutil.swap_memory') as mock_swap:
        
        # Default mock values
        mock_mem.return_value = Mock(
            total=8*1024**3, used=1*1024**3, available=7*1024**3, 
            free=7*1024**3, percent=12.5
        )
        mock_cpu.return_value = 25.0
        mock_disk.return_value = Mock(
            total=1000*1024**3, used=200*1024**3, free=800*1024**3
        )
        mock_swap.return_value = Mock(
            total=4*1024**3, used=0, percent=0.0
        )
        
        yield {
            'memory': mock_mem,
            'cpu': mock_cpu,
            'disk': mock_disk,
            'swap': mock_swap
        }


def test_integration(mock_psutil):
    """Test full integration scenario."""
    manager = ResourceManager(
        memory_limit_mb=2000,
        cpu_limit_percent=90.0,
        monitor_interval=0.1
    )
    
    # Should pass with default mock values
    manager.check_limits()
    
    # Get comprehensive status
    status = manager.get_status()
    assert status['current_usage']['memory_mb'] < 2000
    assert status['current_usage']['cpu_percent'] < 90.0
    
    # Test context manager
    with manager:
        usage = manager.get_current_usage()
        assert usage.memory_mb > 0