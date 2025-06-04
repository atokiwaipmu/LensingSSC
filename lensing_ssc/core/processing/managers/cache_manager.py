"""
Cache management for processing operations.
"""

import os
import gc
import json
import pickle
import hashlib
import tempfile
import time
from typing import Any, Dict, Optional, Union, Callable, List, Tuple
from pathlib import Path
import logging
import threading
from collections import OrderedDict

import numpy as np

from ...base.exceptions import ProcessingError


class CacheManager:
    """Manager for caching intermediate processing results.
    
    Provides both in-memory and disk-based caching with:
    - LRU eviction policy
    - Size limits
    - Automatic cleanup
    - Thread safety
    - Serialization support for various data types
    
    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for disk cache (default: temp directory)
    max_size_mb : int, optional
        Maximum cache size in MB (default: 1024)
    max_memory_items : int, optional
        Maximum number of items in memory cache (default: 100)
    cleanup_interval : int, optional
        Cleanup interval in seconds (default: 300)
    enable_disk_cache : bool, optional
        Whether to enable disk caching (default: True)
    enable_memory_cache : bool, optional
        Whether to enable memory caching (default: True)
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        max_size_mb: int = 1024,
        max_memory_items: int = 100,
        cleanup_interval: int = 300,
        enable_disk_cache: bool = True,
        enable_memory_cache: bool = True
    ):
        # Cache directories
        if cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir()) / "lensing_ssc_cache"
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_memory_items = max_memory_items
        self.cleanup_interval = cleanup_interval
        self.enable_disk_cache = enable_disk_cache
        self.enable_memory_cache = enable_memory_cache
        
        # Memory cache (LRU)
        self._memory_cache = OrderedDict()
        self._memory_cache_size = 0
        
        # Disk cache tracking
        self._disk_cache_index = {}
        self._disk_cache_size = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cleanup tracking
        self._last_cleanup = time.time()
        
        # Statistics
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'disk_hits': 0,
            'disk_misses': 0,
            'evictions': 0,
            'errors': 0
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load existing disk cache index
        self._load_disk_index()
        
        # Perform initial cleanup
        self._cleanup_if_needed()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache.
        
        Parameters
        ----------
        key : str
            Cache key
        default : Any, optional
            Default value if key not found
            
        Returns
        -------
        Any
            Cached value or default
        """
        try:
            with self._lock:
                # Check memory cache first
                if self.enable_memory_cache and key in self._memory_cache:
                    # Move to end (most recently used)
                    value = self._memory_cache.pop(key)
                    self._memory_cache[key] = value
                    self.stats['memory_hits'] += 1
                    self.logger.debug(f"Memory cache hit: {key}")
                    return value
                
                # Check disk cache
                if self.enable_disk_cache and key in self._disk_cache_index:
                    try:
                        value = self._load_from_disk(key)
                        
                        # Add to memory cache if enabled
                        if self.enable_memory_cache:
                            self._add_to_memory_cache(key, value)
                        
                        self.stats['disk_hits'] += 1
                        self.logger.debug(f"Disk cache hit: {key}")
                        return value
                    except Exception as e:
                        self.logger.warning(f"Failed to load from disk cache: {e}")
                        # Remove invalid entry
                        self._remove_from_disk_index(key)
                
                # Cache miss
                if key in self._memory_cache:
                    self.stats['memory_misses'] += 1
                else:
                    self.stats['disk_misses'] += 1
                
                return default
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Cache get error for key {key}: {e}")
            return default
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put item in cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        ttl : int, optional
            Time to live in seconds
            
        Returns
        -------
        bool
            True if successfully cached
        """
        try:
            with self._lock:
                current_time = time.time()
                
                # Calculate value size
                value_size = self._calculate_size(value)
                
                # Add to memory cache if enabled and fits
                if self.enable_memory_cache:
                    if value_size < self.max_size_bytes * 0.1:  # Don't cache huge items in memory
                        self._add_to_memory_cache(key, value, value_size)
                
                # Add to disk cache if enabled
                if self.enable_disk_cache:
                    self._save_to_disk(key, value, ttl, current_time)
                
                # Cleanup if needed
                self._cleanup_if_needed()
                
                return True
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Cache put error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from cache.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        bool
            True if item was deleted
        """
        try:
            with self._lock:
                deleted = False
                
                # Remove from memory cache
                if key in self._memory_cache:
                    value = self._memory_cache.pop(key)
                    self._memory_cache_size -= self._calculate_size(value)
                    deleted = True
                
                # Remove from disk cache
                if key in self._disk_cache_index:
                    self._remove_from_disk_cache(key)
                    deleted = True
                
                return deleted
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            with self._lock:
                # Clear memory cache
                self._memory_cache.clear()
                self._memory_cache_size = 0
                
                # Clear disk cache
                for key in list(self._disk_cache_index.keys()):
                    self._remove_from_disk_cache(key)
                
                self.logger.info("Cache cleared")
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Cache clear error: {e}")
    
    def cleanup(self) -> Dict[str, int]:
        """Perform cleanup and return statistics.
        
        Returns
        -------
        Dict[str, int]
            Cleanup statistics
        """
        try:
            with self._lock:
                stats = {
                    'memory_evicted': 0,
                    'disk_evicted': 0,
                    'expired_removed': 0,
                    'invalid_removed': 0
                }
                
                current_time = time.time()
                
                # Clean expired disk entries
                expired_keys = []
                for key, info in self._disk_cache_index.items():
                    if info.get('ttl') and current_time > info['created'] + info['ttl']:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self._remove_from_disk_cache(key)
                    stats['expired_removed'] += 1
                
                # Clean invalid disk entries
                invalid_keys = []
                for key, info in self._disk_cache_index.items():
                    cache_file = self.cache_dir / info['filename']
                    if not cache_file.exists():
                        invalid_keys.append(key)
                
                for key in invalid_keys:
                    self._remove_from_disk_index(key)
                    stats['invalid_removed'] += 1
                
                # Evict from memory cache if over limit
                while len(self._memory_cache) > self.max_memory_items:
                    # Remove least recently used
                    key, value = self._memory_cache.popitem(last=False)
                    self._memory_cache_size -= self._calculate_size(value)
                    stats['memory_evicted'] += 1
                    self.stats['evictions'] += 1
                
                # Evict from disk cache if over size limit
                while self._disk_cache_size > self.max_size_bytes:
                    # Find oldest entry
                    oldest_key = min(
                        self._disk_cache_index.keys(),
                        key=lambda k: self._disk_cache_index[k]['created']
                    )
                    self._remove_from_disk_cache(oldest_key)
                    stats['disk_evicted'] += 1
                    self.stats['evictions'] += 1
                
                self._last_cleanup = current_time
                
                if any(stats.values()):
                    self.logger.info(f"Cache cleanup completed: {stats}")
                
                return stats
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Cache cleanup error: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns
        -------
        Dict[str, Any]
            Cache statistics
        """
        with self._lock:
            memory_size_mb = self._memory_cache_size / (1024 * 1024)
            disk_size_mb = self._disk_cache_size / (1024 * 1024)
            
            return {
                **self.stats,
                'memory_items': len(self._memory_cache),
                'memory_size_mb': memory_size_mb,
                'disk_items': len(self._disk_cache_index),
                'disk_size_mb': disk_size_mb,
                'total_size_mb': memory_size_mb + disk_size_mb,
                'hit_rate': self._calculate_hit_rate(),
                'last_cleanup': self._last_cleanup
            }
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        bool
            True if key exists
        """
        with self._lock:
            return (
                (self.enable_memory_cache and key in self._memory_cache) or
                (self.enable_disk_cache and key in self._disk_cache_index)
            )
    
    def keys(self) -> List[str]:
        """Get all cache keys.
        
        Returns
        -------
        List[str]
            List of cache keys
        """
        with self._lock:
            memory_keys = set(self._memory_cache.keys()) if self.enable_memory_cache else set()
            disk_keys = set(self._disk_cache_index.keys()) if self.enable_disk_cache else set()
            return list(memory_keys | disk_keys)
    
    def memoize(self, ttl: Optional[int] = None):
        """Decorator for memoizing function results.
        
        Parameters
        ----------
        ttl : int, optional
            Time to live for cached results
            
        Returns
        -------
        Callable
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': sorted(kwargs.items())
                }
                key = self._create_key(key_data)
                
                # Try to get from cache
                result = self.get(key)
                if result is not None:
                    return result
                
                # Compute and cache result
                result = func(*args, **kwargs)
                self.put(key, result, ttl=ttl)
                return result
            
            return wrapper
        return decorator
    
    def _add_to_memory_cache(self, key: str, value: Any, size: Optional[int] = None) -> None:
        """Add item to memory cache."""
        if size is None:
            size = self._calculate_size(value)
        
        # Remove if already exists
        if key in self._memory_cache:
            old_value = self._memory_cache.pop(key)
            self._memory_cache_size -= self._calculate_size(old_value)
        
        self._memory_cache[key] = value
        self._memory_cache_size += size
        
        # Evict if necessary
        while (len(self._memory_cache) > self.max_memory_items or 
               self._memory_cache_size > self.max_size_bytes * 0.5):
            if not self._memory_cache:
                break
            old_key, old_value = self._memory_cache.popitem(last=False)
            self._memory_cache_size -= self._calculate_size(old_value)
            self.stats['evictions'] += 1
    
    def _save_to_disk(self, key: str, value: Any, ttl: Optional[int], current_time: float) -> None:
        """Save item to disk cache."""
        # Generate filename
        filename = self._create_filename(key)
        cache_file = self.cache_dir / filename
        
        # Serialize and save
        try:
            if isinstance(value, np.ndarray):
                # Use numpy's save for arrays
                np.save(cache_file.with_suffix('.npy'), value)
                serialization = 'numpy'
            else:
                # Use pickle for other objects
                with open(cache_file.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                serialization = 'pickle'
            
            # Update index
            file_size = cache_file.with_suffix(f'.{serialization[0:3]}').stat().st_size
            
            # Remove old entry if exists
            if key in self._disk_cache_index:
                self._remove_from_disk_cache(key)
            
            self._disk_cache_index[key] = {
                'filename': filename,
                'serialization': serialization,
                'size': file_size,
                'created': current_time,
                'ttl': ttl
            }
            
            self._disk_cache_size += file_size
            self._save_disk_index()
            
        except Exception as e:
            self.logger.error(f"Failed to save to disk cache: {e}")
            raise
    
    def _load_from_disk(self, key: str) -> Any:
        """Load item from disk cache."""
        info = self._disk_cache_index[key]
        cache_file = self.cache_dir / info['filename']
        
        try:
            if info['serialization'] == 'numpy':
                return np.load(cache_file.with_suffix('.npy'))
            else:
                with open(cache_file.with_suffix('.pkl'), 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load from disk cache: {e}")
            raise
    
    def _remove_from_disk_cache(self, key: str) -> None:
        """Remove item from disk cache."""
        if key not in self._disk_cache_index:
            return
        
        info = self._disk_cache_index[key]
        cache_file = self.cache_dir / info['filename']
        
        # Remove files
        for ext in ['.npy', '.pkl']:
            file_path = cache_file.with_suffix(ext)
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to remove cache file {file_path}: {e}")
        
        # Update size tracking
        self._disk_cache_size -= info['size']
        
        # Remove from index
        del self._disk_cache_index[key]
        self._save_disk_index()
    
    def _remove_from_disk_index(self, key: str) -> None:
        """Remove key from disk index without removing files."""
        if key in self._disk_cache_index:
            info = self._disk_cache_index[key]
            self._disk_cache_size -= info['size']
            del self._disk_cache_index[key]
            self._save_disk_index()
    
    def _load_disk_index(self) -> None:
        """Load disk cache index."""
        index_file = self.cache_dir / 'cache_index.json'
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self._disk_cache_index = json.load(f)
                
                # Calculate total size
                self._disk_cache_size = sum(
                    info['size'] for info in self._disk_cache_index.values()
                )
                
                self.logger.debug(f"Loaded disk cache index: {len(self._disk_cache_index)} items")
                
            except Exception as e:
                self.logger.warning(f"Failed to load cache index: {e}")
                self._disk_cache_index = {}
                self._disk_cache_size = 0
    
    def _save_disk_index(self) -> None:
        """Save disk cache index."""
        index_file = self.cache_dir / 'cache_index.json'
        
        try:
            with open(index_file, 'w') as f:
                json.dump(self._disk_cache_index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def _cleanup_if_needed(self) -> None:
        """Perform cleanup if interval has passed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            self.cleanup()
    
    def _calculate_size(self, obj: Any) -> int:
        """Calculate approximate size of object in bytes."""
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (str, bytes)):
                return len(obj)
            elif isinstance(obj, (list, tuple)):
                return sum(self._calculate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) for k, v in obj.items())
            else:
                # Fallback: use pickle size
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1024  # Default estimate
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_hits = self.stats['memory_hits'] + self.stats['disk_hits']
        total_requests = total_hits + self.stats['memory_misses'] + self.stats['disk_misses']
        
        if total_requests == 0:
            return 0.0
        
        return total_hits / total_requests
    
    def _create_key(self, data: Any) -> str:
        """Create cache key from data."""
        # Create deterministic string representation
        key_str = json.dumps(data, sort_keys=True, default=str)
        
        # Hash to create fixed-length key
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _create_filename(self, key: str) -> str:
        """Create filename from cache key."""
        return f"cache_{key}"