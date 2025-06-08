"""
Cache manager for efficient data storage and retrieval.

Provides multi-tier caching with configurable backends, automatic cleanup,
and comprehensive cache management features for processing operations.
"""

import time
import pickle
import hashlib
import threading
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
import logging

from .exceptions import CacheError
from ...config.settings import ProcessingConfig


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl
    
    @property
    def age(self) -> float:
        """Age in seconds."""
        return time.time() - self.timestamp


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheManager:
    """Multi-tier cache manager with configurable backends.
    
    Features:
    - LRU/LFU eviction policies
    - TTL-based expiration
    - Disk persistence with memory overlay
    - Thread-safe operations
    - Comprehensive statistics
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        max_size_mb: int = 512,
        max_entries: int = 1000,
        default_ttl: Optional[float] = None,
        eviction_policy: str = "lru",
        enable_disk_cache: bool = True,
        cleanup_interval: float = 300.0,
        compression: bool = False,
        config: Optional[ProcessingConfig] = None
    ):
        """Initialize cache manager.
        
        Parameters
        ----------
        cache_dir : str or Path, optional
            Directory for disk cache
        max_size_mb : int
            Maximum cache size in MB
        max_entries : int
            Maximum number of cache entries
        default_ttl : float, optional
            Default TTL in seconds
        eviction_policy : str
            Eviction policy ("lru", "lfu", "fifo")
        enable_disk_cache : bool
            Enable disk-based caching
        cleanup_interval : float
            Cleanup interval in seconds
        compression : bool
            Enable compression for disk cache
        config : ProcessingConfig, optional
            Configuration object
        """
        # Load from config
        if config:
            cache_dir = cache_dir or config.cache_dir
            max_size_mb = max_size_mb or config.cache_size_mb
            max_entries = max_entries or config.max_cache_entries
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy.lower()
        self.enable_disk_cache = enable_disk_cache
        self.cleanup_interval = cleanup_interval
        self.compression = compression
        
        # Initialize storage
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._disk_cache_index: Dict[str, Path] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
        # Setup disk cache
        if self.enable_disk_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_index()
        
        # Cleanup timer
        self._cleanup_timer: Optional[threading.Timer] = None
        self._start_cleanup_timer()
        
        logger.debug(f"CacheManager initialized: {max_size_mb}MB, {max_entries} entries")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                if entry.is_expired:
                    self._remove_entry(key)
                    self._stats.misses += 1
                    return default
                
                # Update access info
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Move to end for LRU
                if self.eviction_policy == "lru":
                    self._memory_cache.move_to_end(key)
                
                self._stats.hits += 1
                return entry.data
            
            # Check disk cache
            if self.enable_disk_cache and key in self._disk_cache_index:
                try:
                    data = self._load_from_disk(key)
                    if data is not None:
                        # Promote to memory cache
                        self.put(key, data)
                        self._stats.hits += 1
                        return data
                except Exception as e:
                    logger.warning(f"Failed to load from disk cache: {e}")
                    self._disk_cache_index.pop(key, None)
            
            self._stats.misses += 1
            return default
    
    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Put value in cache."""
        with self._lock:
            ttl = ttl or self.default_ttl
            tags = tags or []
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 0
            
            # Create entry
            entry = CacheEntry(
                data=value,
                size_bytes=size_bytes,
                ttl=ttl,
                tags=tags
            )
            
            # Check if we need to evict
            self._ensure_capacity(size_bytes)
            
            # Store in memory
            self._memory_cache[key] = entry
            self._stats.size_bytes += size_bytes
            self._stats.entry_count += 1
            
            # Store to disk if enabled
            if self.enable_disk_cache:
                try:
                    self._save_to_disk(key, value)
                except Exception as e:
                    logger.warning(f"Failed to save to disk cache: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            removed = False
            
            # Remove from memory
            if key in self._memory_cache:
                entry = self._memory_cache.pop(key)
                self._stats.size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                removed = True
            
            # Remove from disk
            if key in self._disk_cache_index:
                disk_path = self._disk_cache_index.pop(key)
                try:
                    if disk_path.exists():
                        disk_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete disk cache file: {e}")
                removed = True
            
            return removed
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._memory_cache.clear()
            self._disk_cache_index.clear()
            self._stats = CacheStats()
            
            # Clear disk cache
            if self.enable_disk_cache and self.cache_dir:
                try:
                    for file_path in self.cache_dir.glob("cache_*"):
                        file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clear disk cache: {e}")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            expired_keys = []
            
            for key, entry in self._memory_cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
            return len(expired_keys)
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate entries by tags."""
        with self._lock:
            to_remove = []
            
            for key, entry in self._memory_cache.items():
                if any(tag in entry.tags for tag in tags):
                    to_remove.append(key)
            
            for key in to_remove:
                self.delete(key)
            
            return len(to_remove)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size_bytes=self._stats.size_bytes,
                entry_count=len(self._memory_cache)
            )
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage details."""
        with self._lock:
            return {
                'size_mb': self._stats.size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': self._stats.size_bytes / self.max_size_bytes,
                'entry_count': len(self._memory_cache),
                'max_entries': self.max_entries,
            }
    
    def _ensure_capacity(self, new_size: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Check size limit
        while (self._stats.size_bytes + new_size > self.max_size_bytes and 
               self._memory_cache):
            self._evict_one()
        
        # Check entry count limit
        while len(self._memory_cache) >= self.max_entries:
            self._evict_one()
    
    def _evict_one(self) -> None:
        """Evict one entry based on policy."""
        if not self._memory_cache:
            return
        
        if self.eviction_policy == "lru":
            # Remove least recently used (first in OrderedDict)
            key = next(iter(self._memory_cache))
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            key = min(self._memory_cache.keys(), 
                     key=lambda k: self._memory_cache[k].access_count)
        elif self.eviction_policy == "fifo":
            # Remove oldest entry
            key = min(self._memory_cache.keys(),
                     key=lambda k: self._memory_cache[k].timestamp)
        else:
            # Default to LRU
            key = next(iter(self._memory_cache))
        
        self._remove_entry(key)
        self._stats.evictions += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from memory cache."""
        if key in self._memory_cache:
            entry = self._memory_cache.pop(key)
            self._stats.size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
    
    def _save_to_disk(self, key: str, value: Any) -> None:
        """Save entry to disk cache."""
        if not self.cache_dir:
            return
        
        # Generate filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        filename = f"cache_{key_hash}.pkl"
        if self.compression:
            filename += ".gz"
        
        file_path = self.cache_dir / filename
        
        try:
            if self.compression:
                import gzip
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(value, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
            
            self._disk_cache_index[key] = file_path
            
        except Exception as e:
            logger.error(f"Failed to save to disk: {e}")
            raise CacheError(f"Disk cache save failed: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load entry from disk cache."""
        if key not in self._disk_cache_index:
            return None
        
        file_path = self._disk_cache_index[key]
        if not file_path.exists():
            self._disk_cache_index.pop(key, None)
            return None
        
        try:
            if self.compression:
                import gzip
                with gzip.open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Failed to load from disk: {e}")
            # Remove corrupted file
            try:
                file_path.unlink()
                self._disk_cache_index.pop(key, None)
            except Exception:
                pass
            return None
    
    def _load_disk_index(self) -> None:
        """Load disk cache index."""
        if not self.cache_dir or not self.cache_dir.exists():
            return
        
        pattern = "cache_*.pkl*"
        for file_path in self.cache_dir.glob(pattern):
            # Extract key hash from filename
            parts = file_path.stem.split('_', 1)
            if len(parts) == 2:
                key_hash = parts[1]
                # We can't reverse the hash, so we'll use the hash as key
                # In practice, you'd store a mapping file
                self._disk_cache_index[key_hash] = file_path
    
    def _start_cleanup_timer(self) -> None:
        """Start periodic cleanup timer."""
        if self.cleanup_interval > 0:
            self._cleanup_timer = threading.Timer(
                self.cleanup_interval,
                self._periodic_cleanup
            )
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup task."""
        try:
            self.cleanup_expired()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        finally:
            # Schedule next cleanup
            self._start_cleanup_timer()
    
    def get_status(self) -> Dict[str, Any]:
        """Get cache manager status."""
        stats = self.get_stats()
        memory_usage = self.get_memory_usage()
        
        return {
            'stats': {
                'hits': stats.hits,
                'misses': stats.misses,
                'hit_rate': stats.hit_rate,
                'evictions': stats.evictions,
            },
            'memory': memory_usage,
            'disk_cache_enabled': self.enable_disk_cache,
            'disk_entries': len(self._disk_cache_index),
            'eviction_policy': self.eviction_policy,
            'default_ttl': self.default_ttl,
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()