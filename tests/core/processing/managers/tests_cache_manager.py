"""
Tests for CacheManager.
"""

import pytest
import tempfile
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch

from lensing_ssc.core.processing.managers.cache_manager import (
    CacheManager, CacheEntry, CacheStats
)
from lensing_ssc.core.processing.managers.exceptions import CacheError


class TestCacheEntry:
    def test_creation(self):
        data = {"test": "data"}
        entry = CacheEntry(
            data=data,
            size_bytes=100,
            ttl=3600,
            tags=["test", "data"]
        )
        
        assert entry.data == data
        assert entry.size_bytes == 100
        assert entry.ttl == 3600
        assert "test" in entry.tags
        assert entry.timestamp > 0
        assert entry.access_count == 0
    
    def test_is_expired(self):
        # Non-expiring entry
        entry = CacheEntry(data="test")
        assert not entry.is_expired
        
        # Expired entry
        entry = CacheEntry(data="test", ttl=0.001)
        time.sleep(0.002)
        assert entry.is_expired
        
        # Not yet expired
        entry = CacheEntry(data="test", ttl=3600)
        assert not entry.is_expired
    
    def test_age(self):
        entry = CacheEntry(data="test")
        time.sleep(0.01)
        assert entry.age > 0


class TestCacheStats:
    def test_creation(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
    
    def test_hit_rate_calculation(self):
        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == 0.7
        
        # No requests
        stats = CacheStats()
        assert stats.hit_rate == 0.0


class TestCacheManager:
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def memory_cache(self):
        return CacheManager(
            max_size_mb=1,
            max_entries=10,
            enable_disk_cache=False
        )
    
    @pytest.fixture
    def disk_cache(self, temp_dir):
        return CacheManager(
            cache_dir=temp_dir,
            max_size_mb=1,
            max_entries=10,
            enable_disk_cache=True
        )
    
    def test_memory_cache_basic(self, memory_cache):
        # Put and get
        memory_cache.put("key1", "value1")
        assert memory_cache.get("key1") == "value1"
        
        # Miss
        assert memory_cache.get("nonexistent") is None
        assert memory_cache.get("nonexistent", "default") == "default"
        
        # Stats
        stats = memory_cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 2
    
    def test_ttl_expiration(self, memory_cache):
        # Put with short TTL
        memory_cache.put("expiring", "value", ttl=0.001)
        assert memory_cache.get("expiring") == "value"
        
        # Wait for expiration
        time.sleep(0.002)
        assert memory_cache.get("expiring") is None
        
        # Should be removed from cache
        assert "expiring" not in memory_cache._memory_cache
    
    def test_eviction_lru(self, memory_cache):
        memory_cache.eviction_policy = "lru"
        
        # Fill cache to capacity
        for i in range(memory_cache.max_entries):
            memory_cache.put(f"key{i}", f"value{i}")
        
        # Access some keys to change LRU order
        memory_cache.get("key0")  # Make key0 recently used
        
        # Add one more to trigger eviction
        memory_cache.put("overflow", "value")
        
        # key0 should still be there (recently used)
        assert memory_cache.get("key0") == "value0"
        # Some other key should be evicted
        assert len(memory_cache._memory_cache) == memory_cache.max_entries
    
    def test_eviction_lfu(self, memory_cache):
        memory_cache.eviction_policy = "lfu"
        
        # Add items
        memory_cache.put("frequent", "value1")
        memory_cache.put("rare", "value2")
        
        # Access frequent item multiple times
        for _ in range(5):
            memory_cache.get("frequent")
        
        # Fill to capacity
        for i in range(memory_cache.max_entries - 2):
            memory_cache.put(f"filler{i}", f"value{i}")
        
        # Add overflow item
        memory_cache.put("overflow", "value")
        
        # Frequent item should survive
        assert memory_cache.get("frequent") == "value1"
    
    def test_size_limit_eviction(self):
        # Small cache for testing size limits
        cache = CacheManager(max_size_mb=0.001, max_entries=100, enable_disk_cache=False)
        
        # Add large item that exceeds size limit
        large_data = "x" * 2048  # 2KB
        cache.put("large", large_data)
        
        # Should trigger size-based eviction
        small_data = "small"
        cache.put("small", small_data)
        
        # At least one should be evicted
        usage = cache.get_memory_usage()
        assert usage['size_mb'] <= cache.max_size_bytes / (1024 * 1024) * 1.1  # Allow small margin
    
    def test_delete(self, memory_cache):
        memory_cache.put("delete_me", "value")
        assert memory_cache.get("delete_me") == "value"
        
        success = memory_cache.delete("delete_me")
        assert success
        assert memory_cache.get("delete_me") is None
        
        # Delete non-existent
        success = memory_cache.delete("nonexistent")
        assert not success
    
    def test_clear(self, memory_cache):
        # Add multiple items
        for i in range(5):
            memory_cache.put(f"key{i}", f"value{i}")
        
        assert len(memory_cache._memory_cache) == 5
        
        memory_cache.clear()
        assert len(memory_cache._memory_cache) == 0
        stats = memory_cache.get_stats()
        assert stats.size_bytes == 0
    
    def test_cleanup_expired(self, memory_cache):
        # Add mix of expiring and non-expiring items
        memory_cache.put("permanent", "value")
        memory_cache.put("expiring1", "value1", ttl=0.001)
        memory_cache.put("expiring2", "value2", ttl=0.001)
        
        time.sleep(0.002)
        
        cleaned = memory_cache.cleanup_expired()
        assert cleaned == 2
        assert memory_cache.get("permanent") == "value"
        assert memory_cache.get("expiring1") is None
    
    def test_tag_invalidation(self, memory_cache):
        memory_cache.put("item1", "value1", tags=["group1", "type_a"])
        memory_cache.put("item2", "value2", tags=["group1", "type_b"])
        memory_cache.put("item3", "value3", tags=["group2", "type_a"])
        
        # Invalidate by tag
        invalidated = memory_cache.invalidate_by_tags(["group1"])
        assert invalidated == 2
        
        assert memory_cache.get("item1") is None
        assert memory_cache.get("item2") is None
        assert memory_cache.get("item3") == "value3"
    
    def test_disk_cache_basic(self, disk_cache):
        # Put item
        disk_cache.put("disk_key", "disk_value")
        assert disk_cache.get("disk_key") == "disk_value"
        
        # Remove from memory but keep on disk
        disk_cache._memory_cache.clear()
        
        # Should load from disk
        assert disk_cache.get("disk_key") == "disk_value"
    
    def test_disk_cache_compression(self, temp_dir):
        cache = CacheManager(
            cache_dir=temp_dir,
            enable_disk_cache=True,
            compression=True
        )
        
        data = {"large": list(range(1000))}
        cache.put("compressed", data)
        
        # Clear memory and reload
        cache._memory_cache.clear()
        loaded = cache.get("compressed")
        assert loaded == data
    
    def test_get_memory_usage(self, memory_cache):
        memory_cache.put("test", "value")
        
        usage = memory_cache.get_memory_usage()
        assert "size_mb" in usage
        assert "max_size_mb" in usage
        assert "utilization" in usage
        assert "entry_count" in usage
        assert usage["entry_count"] == 1
    
    def test_get_status(self, memory_cache):
        memory_cache.put("test", "value")
        memory_cache.get("test")
        memory_cache.get("miss")
        
        status = memory_cache.get_status()
        assert "stats" in status
        assert status["stats"]["hits"] == 1
        assert status["stats"]["misses"] == 1
        assert "memory" in status
        assert "eviction_policy" in status
    
    def test_concurrent_access(self, memory_cache):
        """Test thread safety."""
        errors = []
        results = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    key = f"worker_{worker_id}_item_{i}"
                    value = f"value_{worker_id}_{i}"
                    memory_cache.put(key, value)
                    retrieved = memory_cache.get(key)
                    if retrieved == value:
                        results.append(True)
                    else:
                        results.append(False)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0
        assert all(results)
    
    @patch('lensing_ssc.core.processing.managers.cache_manager.logger')
    def test_error_handling(self, mock_logger, temp_dir):
        cache = CacheManager(cache_dir=temp_dir, enable_disk_cache=True)
        
        # Simulate disk write error
        with patch('builtins.open', side_effect=IOError("Disk full")):
            with pytest.raises(CacheError):
                cache.put("error_key", "value")
    
    def test_periodic_cleanup(self, memory_cache):
        # Disable automatic cleanup for testing
        if memory_cache._cleanup_timer:
            memory_cache._cleanup_timer.cancel()
        
        # Add expiring items
        memory_cache.put("exp1", "value1", ttl=0.001)
        memory_cache.put("exp2", "value2", ttl=0.001)
        
        time.sleep(0.002)
        
        # Manual cleanup
        memory_cache._periodic_cleanup()
        
        # Items should be cleaned up
        assert memory_cache.get("exp1") is None
        assert memory_cache.get("exp2") is None


def test_disk_cache_file_corruption(temp_dir):
    """Test handling of corrupted disk cache files."""
    cache = CacheManager(cache_dir=temp_dir, enable_disk_cache=True)
    
    # Put item
    cache.put("test", "value")
    
    # Corrupt the disk file
    disk_files = list(temp_dir.glob("cache_*.pkl"))
    assert len(disk_files) == 1
    
    with open(disk_files[0], 'w') as f:
        f.write("corrupted data")
    
    # Clear memory cache
    cache._memory_cache.clear()
    
    # Should handle corruption gracefully
    result = cache.get("test")
    assert result is None


def test_cache_size_calculation():
    """Test cache size calculation accuracy."""
    cache = CacheManager(max_size_mb=1, enable_disk_cache=False)
    
    # Add known size data
    small_data = "x" * 100
    cache.put("small", small_data)
    
    large_data = "x" * 1000
    cache.put("large", large_data)
    
    usage = cache.get_memory_usage()
    assert usage["size_mb"] > 0
    assert usage["entry_count"] == 2


def test_cache_policy_edge_cases():
    """Test edge cases in eviction policies."""
    cache = CacheManager(max_entries=1, enable_disk_cache=False)
    
    # Single item cache
    cache.put("first", "value1")
    assert cache.get("first") == "value1"
    
    # Adding second item should evict first
    cache.put("second", "value2")
    assert cache.get("first") is None
    assert cache.get("second") == "value2"