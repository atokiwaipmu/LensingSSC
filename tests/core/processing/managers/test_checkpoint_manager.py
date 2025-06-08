"""
Tests for CheckpointManager.
"""

import pytest
import tempfile
import shutil
import pickle
import json
from pathlib import Path
from unittest.mock import Mock, patch

from lensing_ssc.core.processing.managers.checkpoint_manager import (
    CheckpointManager, CheckpointMetadata
)
from lensing_ssc.core.processing.managers.exceptions import CheckpointError


class TestCheckpointMetadata:
    def test_creation(self):
        meta = CheckpointMetadata(
            description="Test checkpoint",
            tags=["test", "demo"],
            data_keys=["key1", "key2"]
        )
        assert meta.description == "Test checkpoint"
        assert "test" in meta.tags
        assert meta.timestamp > 0
        assert meta.version == "1.0"
    
    def test_datetime_str(self):
        meta = CheckpointMetadata()
        datetime_str = meta.datetime_str
        assert "T" in datetime_str  # ISO format


class TestCheckpointManager:
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def manager(self, temp_dir):
        return CheckpointManager(
            checkpoint_dir=temp_dir,
            max_checkpoints=3
        )
    
    def test_initialization(self, temp_dir):
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            checkpoint_name="test_checkpoint",
            max_checkpoints=5
        )
        assert manager.checkpoint_dir == temp_dir
        assert manager.checkpoint_name == "test_checkpoint"
        assert manager.max_checkpoints == 5
        assert temp_dir.exists()
    
    def test_save_checkpoint(self, manager):
        data = {"step": 1, "results": [1, 2, 3], "config": {"param": "value"}}
        
        checkpoint_path = manager.save_checkpoint(
            data,
            description="Test save",
            tags=["test"]
        )
        
        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == ".pkl"
        
        # Check metadata file
        meta_file = checkpoint_path.with_suffix('.meta.json')
        assert meta_file.exists()
        
        with open(meta_file) as f:
            meta_data = json.load(f)
        assert meta_data["description"] == "Test save"
        assert "test" in meta_data["tags"]
    
    def test_load_checkpoint(self, manager):
        # Save first
        original_data = {"step": 5, "results": [10, 20, 30]}
        checkpoint_path = manager.save_checkpoint(original_data)
        
        # Load
        loaded_data = manager.load_checkpoint()
        assert loaded_data == original_data
        
        # Load specific checkpoint
        loaded_data = manager.load_checkpoint(checkpoint_path)
        assert loaded_data == original_data
    
    def test_load_nonexistent_checkpoint(self, manager):
        result = manager.load_checkpoint()
        assert result is None
    
    def test_get_latest_checkpoint(self, manager):
        # No checkpoints initially
        assert manager.get_latest_checkpoint() is None
        
        # Save multiple checkpoints
        manager.save_checkpoint({"step": 1})
        checkpoint2 = manager.save_checkpoint({"step": 2})
        
        latest = manager.get_latest_checkpoint()
        assert latest == checkpoint2
    
    def test_list_checkpoints(self, manager):
        # Empty initially
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 0
        
        # Add checkpoints
        manager.save_checkpoint({"step": 1}, description="First")
        manager.save_checkpoint({"step": 2}, description="Second")
        
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2
        assert checkpoints[0]["description"] == "Second"  # Most recent first
        assert "size_mb" in checkpoints[0]
        assert "modified" in checkpoints[0]
    
    def test_delete_checkpoint(self, manager):
        # Save checkpoint
        data = {"test": "data"}
        checkpoint_path = manager.save_checkpoint(data)
        meta_path = checkpoint_path.with_suffix('.meta.json')
        
        assert checkpoint_path.exists()
        assert meta_path.exists()
        
        # Delete
        success = manager.delete_checkpoint(checkpoint_path)
        assert success
        assert not checkpoint_path.exists()
        assert not meta_path.exists()
    
    def test_cleanup_old_checkpoints(self, manager):
        # Save more than max_checkpoints
        for i in range(5):
            manager.save_checkpoint({"step": i})
        
        # Should automatically cleanup
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) <= manager.max_checkpoints
    
    def test_compression(self, temp_dir):
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            compression=True
        )
        
        data = {"large_data": list(range(1000))}
        checkpoint_path = manager.save_checkpoint(data)
        
        # Should be able to load compressed data
        loaded_data = manager.load_checkpoint()
        assert loaded_data == data
    
    def test_validation(self, manager):
        # Save with validation enabled
        data = {"test": "validation"}
        manager.validate_on_load = True
        
        checkpoint_path = manager.save_checkpoint(data)
        
        # Should validate and load successfully
        loaded_data = manager.load_checkpoint()
        assert loaded_data == data
    
    def test_custom_validator(self, manager):
        validation_called = False
        
        def custom_validator(data):
            nonlocal validation_called
            validation_called = True
            return "required_key" in data
        
        manager.add_validator(custom_validator)
        
        # Save valid data
        data = {"required_key": "value", "other": "data"}
        checkpoint_path = manager.save_checkpoint(data)
        
        # Load should call validator
        loaded_data = manager.load_checkpoint()
        assert validation_called
        assert loaded_data == data
        
        # Invalid data should fail validation
        with patch.object(manager, '_load_data') as mock_load:
            mock_load.return_value = {"missing_required_key": "value"}
            
            with pytest.raises(CheckpointError):
                manager.load_checkpoint()
    
    def test_auto_save_check(self, manager):
        # No auto-save interval by default
        assert not manager.should_auto_save()
        
        # Set auto-save interval
        manager.auto_save_interval = 0.1
        assert manager.should_auto_save()  # Should trigger immediately
        
        # After save, should not trigger until interval passes
        manager._last_save_time = manager._last_save_time
        # Would need to wait for actual time to pass
    
    def test_checksum_validation(self, manager, temp_dir):
        # Save checkpoint
        data = {"test": "checksum"}
        checkpoint_path = manager.save_checkpoint(data)
        
        # Load metadata
        meta_file = checkpoint_path.with_suffix('.meta.json')
        with open(meta_file) as f:
            meta_data = json.load(f)
        
        original_checksum = meta_data["checksum"]
        assert original_checksum  # Should have checksum
        
        # Corrupt file
        with open(checkpoint_path, 'ab') as f:
            f.write(b"corruption")
        
        # Validation should fail
        metadata = CheckpointMetadata(**meta_data)
        assert not manager._validate_checkpoint(checkpoint_path, metadata)
    
    def test_context_manager(self, manager):
        with manager as cm:
            assert cm is manager
    
    def test_get_status(self, manager):
        status = manager.get_status()
        
        assert "checkpoint_dir" in status
        assert "checkpoint_count" in status
        assert "max_checkpoints" in status
        assert "compression" in status
        assert status["checkpoint_count"] == 0
        
        # After adding checkpoint
        manager.save_checkpoint({"test": "status"})
        status = manager.get_status()
        assert status["checkpoint_count"] == 1


def test_error_handling():
    """Test error conditions."""
    # Invalid directory permissions (simulate)
    with patch('pathlib.Path.mkdir') as mock_mkdir:
        mock_mkdir.side_effect = PermissionError("Access denied")
        
        with pytest.raises(PermissionError):
            CheckpointManager("/invalid/path")


def test_concurrent_access(temp_dir):
    """Test thread safety."""
    import threading
    
    manager = CheckpointManager(temp_dir)
    results = []
    errors = []
    
    def save_checkpoint(i):
        try:
            data = {"thread": i, "data": list(range(10))}
            path = manager.save_checkpoint(data, description=f"Thread {i}")
            results.append(path)
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=save_checkpoint, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    assert len(errors) == 0
    assert len(results) == 5
    
    # All checkpoints should be loadable
    checkpoints = manager.list_checkpoints()
    assert len(checkpoints) == 5