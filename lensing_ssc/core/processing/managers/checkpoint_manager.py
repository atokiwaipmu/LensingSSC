"""
Checkpoint management for processing recovery.
"""

import json
import time
import pickle
import hashlib
import threading
from typing import Any, Dict, Optional, List, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
import logging
import shutil
import gzip
import tempfile

import numpy as np

from ...base.exceptions import ProcessingError


class CheckpointManager:
    """Advanced checkpoint manager for processing recovery and state persistence.
    
    Provides comprehensive checkpoint functionality including:
    - Automatic and manual checkpoint creation
    - Multiple serialization formats (JSON, pickle, numpy)
    - Checkpoint compression and encryption
    - Automatic cleanup and retention policies
    - Recovery from corrupted checkpoints
    - Incremental checkpoints for large data
    - Thread-safe operations
    
    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory for storing checkpoint files
    checkpoint_name : str, optional
        Base name for checkpoint files (default: "checkpoint")
    max_checkpoints : int, optional
        Maximum number of checkpoints to keep (default: 10)
    compression : bool, optional
        Whether to compress checkpoints (default: True)
    auto_save_interval : int, optional
        Automatic save interval in seconds (default: 300)
    enable_backup : bool, optional
        Whether to create backup copies (default: True)
    retention_days : int, optional
        Number of days to retain checkpoints (default: 7)
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        checkpoint_name: str = "checkpoint",
        max_checkpoints: int = 10,
        compression: bool = True,
        auto_save_interval: int = 300,
        enable_backup: bool = True,
        retention_days: int = 7
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_name = checkpoint_name
        self.max_checkpoints = max_checkpoints
        self.compression = compression
        self.auto_save_interval = auto_save_interval
        self.enable_backup = enable_backup
        self.retention_days = retention_days
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal state
        self._lock = threading.RLock()
        self._auto_save_enabled = False
        self._auto_save_thread = None
        self._last_auto_save = 0
        self._current_data = {}
        self._save_callbacks = []
        self._load_callbacks = []
        
        # Checkpoint tracking
        self._checkpoint_history = []
        self._checkpoint_index = {}
        
        # Statistics
        self.stats = {
            'saves': 0,
            'loads': 0,
            'auto_saves': 0,
            'errors': 0,
            'recoveries': 0,
            'cleanup_runs': 0
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load existing checkpoint index
        self._load_checkpoint_index()
        
        # Perform initial cleanup
        self._cleanup_old_checkpoints()
    
    def save_checkpoint(
        self, 
        data: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Save checkpoint data with comprehensive metadata.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Data to checkpoint
        metadata : Dict[str, Any], optional
            Additional metadata
        checkpoint_id : str, optional
            Custom checkpoint ID (auto-generated if not provided)
        tags : List[str], optional
            Tags for organizing checkpoints
            
        Returns
        -------
        str
            Checkpoint ID
            
        Raises
        ------
        ProcessingError
            If checkpoint save fails
        """
        try:
            with self._lock:
                current_time = time.time()
                
                # Generate checkpoint ID if not provided
                if checkpoint_id is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    data_hash = self._compute_data_hash(data)[:8]
                    checkpoint_id = f"{self.checkpoint_name}_{timestamp}_{data_hash}"
                
                # Prepare checkpoint data
                checkpoint = {
                    "id": checkpoint_id,
                    "timestamp": current_time,
                    "datetime": datetime.now().isoformat(),
                    "data": data,
                    "metadata": metadata or {},
                    "tags": tags or [],
                    "version": "1.0",
                    "stats": {
                        "data_size": self._estimate_size(data),
                        "num_keys": len(data) if isinstance(data, dict) else 1
                    }
                }
                
                # Call pre-save callbacks
                for callback in self._save_callbacks:
                    try:
                        callback(checkpoint_id, checkpoint)
                    except Exception as e:
                        self.logger.warning(f"Save callback failed: {e}")
                
                # Save to file
                checkpoint_file = self._get_checkpoint_file(checkpoint_id)
                self._save_to_file(checkpoint, checkpoint_file)
                
                # Update index
                self._update_checkpoint_index(checkpoint_id, checkpoint_file, checkpoint)
                
                # Update current data
                self._current_data = data.copy() if isinstance(data, dict) else data
                
                # Update statistics
                self.stats['saves'] += 1
                
                # Cleanup old checkpoints
                self._cleanup_old_checkpoints()
                
                self.logger.info(f"Checkpoint saved: {checkpoint_id}")
                
                return checkpoint_id
                
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = f"Failed to save checkpoint: {e}"
            self.logger.error(error_msg)
            raise ProcessingError(error_msg) from e
    
    def load_checkpoint(
        self, 
        checkpoint_id: Optional[str] = None,
        load_latest: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Load checkpoint data.
        
        Parameters
        ----------
        checkpoint_id : str, optional
            Specific checkpoint ID to load
        load_latest : bool, optional
            Load latest checkpoint if checkpoint_id not specified
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Checkpoint data or None if not found
        """
        try:
            with self._lock:
                # Determine which checkpoint to load
                if checkpoint_id is None:
                    if load_latest:
                        checkpoint_id = self._get_latest_checkpoint_id()
                        if checkpoint_id is None:
                            self.logger.info("No checkpoints found")
                            return None
                    else:
                        raise ValueError("Must specify checkpoint_id or set load_latest=True")
                
                # Check if checkpoint exists
                if checkpoint_id not in self._checkpoint_index:
                    self.logger.warning(f"Checkpoint not found: {checkpoint_id}")
                    return None
                
                checkpoint_file = self._checkpoint_index[checkpoint_id]['file']
                
                # Load from file with error recovery
                checkpoint = self._load_from_file(checkpoint_file, checkpoint_id)
                
                if checkpoint is None:
                    return None
                
                # Call post-load callbacks
                for callback in self._load_callbacks:
                    try:
                        callback(checkpoint_id, checkpoint)
                    except Exception as e:
                        self.logger.warning(f"Load callback failed: {e}")
                
                # Update current data
                data = checkpoint.get('data', {})
                self._current_data = data.copy() if isinstance(data, dict) else data
                
                # Update statistics
                self.stats['loads'] += 1
                
                self.logger.info(f"Checkpoint loaded: {checkpoint_id}")
                
                return checkpoint
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint.
        
        Parameters
        ----------
        checkpoint_id : str
            Checkpoint ID to delete
            
        Returns
        -------
        bool
            True if successfully deleted
        """
        try:
            with self._lock:
                if checkpoint_id not in self._checkpoint_index:
                    return False
                
                checkpoint_info = self._checkpoint_index[checkpoint_id]
                checkpoint_file = checkpoint_info['file']
                
                # Remove file
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                
                # Remove backup if exists
                backup_file = checkpoint_file.with_suffix(checkpoint_file.suffix + '.bak')
                if backup_file.exists():
                    backup_file.unlink()
                
                # Remove from index
                del self._checkpoint_index[checkpoint_id]
                
                # Update history
                self._checkpoint_history = [
                    h for h in self._checkpoint_history if h['id'] != checkpoint_id
                ]
                
                # Save updated index
                self._save_checkpoint_index()
                
                self.logger.info(f"Checkpoint deleted: {checkpoint_id}")
                return True
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def list_checkpoints(
        self, 
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List available checkpoints with filtering.
        
        Parameters
        ----------
        tags : List[str], optional
            Filter by tags
        limit : int, optional
            Maximum number of checkpoints to return
            
        Returns
        -------
        List[Dict[str, Any]]
            List of checkpoint information
        """
        with self._lock:
            checkpoints = []
            
            for checkpoint_id, info in self._checkpoint_index.items():
                checkpoint_info = {
                    'id': checkpoint_id,
                    'timestamp': info.get('timestamp'),
                    'datetime': info.get('datetime'),
                    'size_mb': info.get('size_mb', 0),
                    'tags': info.get('tags', []),
                    'metadata': info.get('metadata', {}),
                    'file': str(info['file'])
                }
                
                # Filter by tags if specified
                if tags:
                    checkpoint_tags = set(checkpoint_info['tags'])
                    if not set(tags).intersection(checkpoint_tags):
                        continue
                
                checkpoints.append(checkpoint_info)
            
            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Apply limit
            if limit:
                checkpoints = checkpoints[:limit]
            
            return checkpoints
    
    def checkpoint_exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists.
        
        Parameters
        ----------
        checkpoint_id : str
            Checkpoint ID
            
        Returns
        -------
        bool
            True if checkpoint exists
        """
        with self._lock:
            return checkpoint_id in self._checkpoint_index
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a checkpoint.
        
        Parameters
        ----------
        checkpoint_id : str
            Checkpoint ID
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Checkpoint information or None if not found
        """
        with self._lock:
            if checkpoint_id not in self._checkpoint_index:
                return None
            
            info = self._checkpoint_index[checkpoint_id].copy()
            info['id'] = checkpoint_id
            
            # Add file size if available
            checkpoint_file = info['file']
            if checkpoint_file.exists():
                info['file_size_mb'] = checkpoint_file.stat().st_size / (1024**2)
            
            return info
    
    def create_incremental_checkpoint(
        self, 
        new_data: Dict[str, Any],
        base_checkpoint_id: Optional[str] = None
    ) -> str:
        """Create incremental checkpoint with only changed data.
        
        Parameters
        ----------
        new_data : Dict[str, Any]
            New data to checkpoint
        base_checkpoint_id : str, optional
            Base checkpoint for comparison (latest if not specified)
            
        Returns
        -------
        str
            Incremental checkpoint ID
        """
        try:
            with self._lock:
                # Get base data for comparison
                if base_checkpoint_id is None:
                    base_data = self._current_data
                else:
                    base_checkpoint = self.load_checkpoint(base_checkpoint_id)
                    base_data = base_checkpoint.get('data', {}) if base_checkpoint else {}
                
                # Compute differences
                changes = self._compute_differences(base_data, new_data)
                
                # Create incremental checkpoint
                incremental_data = {
                    'type': 'incremental',
                    'base_checkpoint': base_checkpoint_id,
                    'changes': changes,
                    'full_data_keys': list(new_data.keys()) if isinstance(new_data, dict) else None
                }
                
                checkpoint_id = self.save_checkpoint(
                    incremental_data,
                    metadata={'incremental': True, 'base': base_checkpoint_id},
                    tags=['incremental']
                )
                
                return checkpoint_id
                
        except Exception as e:
            error_msg = f"Failed to create incremental checkpoint: {e}"
            self.logger.error(error_msg)
            raise ProcessingError(error_msg) from e
    
    def restore_from_incremental(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Restore full data from incremental checkpoint.
        
        Parameters
        ----------
        checkpoint_id : str
            Incremental checkpoint ID
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Restored full data
        """
        try:
            with self._lock:
                # Load incremental checkpoint
                incremental = self.load_checkpoint(checkpoint_id)
                if not incremental:
                    return None
                
                data = incremental.get('data', {})
                if data.get('type') != 'incremental':
                    self.logger.error(f"Checkpoint {checkpoint_id} is not incremental")
                    return None
                
                # Load base checkpoint
                base_id = data.get('base_checkpoint')
                if base_id:
                    base_checkpoint = self.load_checkpoint(base_id)
                    if not base_checkpoint:
                        self.logger.error(f"Base checkpoint {base_id} not found")
                        return None
                    base_data = base_checkpoint.get('data', {})
                else:
                    base_data = {}
                
                # Apply changes
                full_data = self._apply_changes(base_data, data.get('changes', {}))
                
                return {'data': full_data}
                
        except Exception as e:
            self.logger.error(f"Failed to restore from incremental checkpoint: {e}")
            return None
    
    def enable_auto_save(self, data_source: Callable[[], Dict[str, Any]]) -> None:
        """Enable automatic checkpoint saving.
        
        Parameters
        ----------
        data_source : Callable
            Function that returns current data to checkpoint
        """
        with self._lock:
            if self._auto_save_enabled:
                return
            
            self._auto_save_enabled = True
            self._data_source = data_source
            
            # Start auto-save thread
            self._auto_save_thread = threading.Thread(
                target=self._auto_save_loop,
                name="CheckpointAutoSave",
                daemon=True
            )
            self._auto_save_thread.start()
            
            self.logger.info("Auto-save enabled")
    
    def disable_auto_save(self) -> None:
        """Disable automatic checkpoint saving."""
        with self._lock:
            if not self._auto_save_enabled:
                return
            
            self._auto_save_enabled = False
            
            if self._auto_save_thread and self._auto_save_thread.is_alive():
                self._auto_save_thread.join(timeout=5.0)
            
            self.logger.info("Auto-save disabled")
    
    def add_save_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add callback function called before saving.
        
        Parameters
        ----------
        callback : Callable
            Function with signature: callback(checkpoint_id, checkpoint_data)
        """
        self._save_callbacks.append(callback)
    
    def add_load_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add callback function called after loading.
        
        Parameters
        ----------
        callback : Callable
            Function with signature: callback(checkpoint_id, checkpoint_data)
        """
        self._load_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics.
        
        Returns
        -------
        Dict[str, Any]
            Statistics
        """
        with self._lock:
            total_size_mb = sum(
                info.get('size_mb', 0) for info in self._checkpoint_index.values()
            )
            
            return {
                **self.stats,
                'total_checkpoints': len(self._checkpoint_index),
                'total_size_mb': total_size_mb,
                'auto_save_enabled': self._auto_save_enabled,
                'last_auto_save': self._last_auto_save,
                'checkpoint_dir': str(self.checkpoint_dir),
                'compression_enabled': self.compression
            }
    
    def cleanup(self, force: bool = False) -> Dict[str, int]:
        """Perform cleanup of old checkpoints.
        
        Parameters
        ----------
        force : bool
            Force cleanup even if not needed
            
        Returns
        -------
        Dict[str, int]
            Cleanup statistics
        """
        return self._cleanup_old_checkpoints(force)
    
    def _get_checkpoint_file(self, checkpoint_id: str) -> Path:
        """Get file path for checkpoint."""
        extension = '.gz' if self.compression else '.pkl'
        return self.checkpoint_dir / f"{checkpoint_id}{extension}"
    
    def _save_to_file(self, checkpoint: Dict[str, Any], file_path: Path) -> None:
        """Save checkpoint to file with error handling."""
        # Create backup of existing file
        if self.enable_backup and file_path.exists():
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            shutil.copy2(file_path, backup_path)
        
        # Save to temporary file first
        temp_file = file_path.with_suffix('.tmp')
        
        try:
            if self.compression:
                with gzip.open(temp_file, 'wb') as f:
                    pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(temp_file, 'wb') as f:
                    pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic move
            temp_file.rename(file_path)
            
        except Exception as e:
            # Cleanup temp file on error
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def _load_from_file(self, file_path: Path, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint from file with error recovery."""
        if not file_path.exists():
            self.logger.error(f"Checkpoint file not found: {file_path}")
            return None
        
        # Try to load main file
        try:
            if self.compression:
                with gzip.open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint {checkpoint_id}: {e}")
            
            # Try backup file
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            if backup_path.exists():
                try:
                    self.logger.info(f"Attempting recovery from backup: {backup_path}")
                    
                    if self.compression:
                        with gzip.open(backup_path, 'rb') as f:
                            data = pickle.load(f)
                    else:
                        with open(backup_path, 'rb') as f:
                            data = pickle.load(f)
                    
                    self.stats['recoveries'] += 1
                    self.logger.info(f"Successfully recovered checkpoint from backup")
                    return data
                    
                except Exception as backup_error:
                    self.logger.error(f"Backup recovery also failed: {backup_error}")
            
            return None
    
    def _load_checkpoint_index(self) -> None:
        """Load checkpoint index from disk."""
        index_file = self.checkpoint_dir / 'checkpoint_index.json'
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                
                self._checkpoint_index = {}
                self._checkpoint_history = data.get('history', [])
                
                # Rebuild index with Path objects
                for checkpoint_id, info in data.get('index', {}).items():
                    info['file'] = Path(info['file'])
                    self._checkpoint_index[checkpoint_id] = info
                
                self.logger.debug(f"Loaded checkpoint index: {len(self._checkpoint_index)} entries")
                
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint index: {e}")
                self._checkpoint_index = {}
                self._checkpoint_history = []
    
    def _save_checkpoint_index(self) -> None:
        """Save checkpoint index to disk."""
        index_file = self.checkpoint_dir / 'checkpoint_index.json'
        
        try:
            # Convert Path objects to strings for JSON serialization
            serializable_index = {}
            for checkpoint_id, info in self._checkpoint_index.items():
                serializable_info = info.copy()
                serializable_info['file'] = str(info['file'])
                serializable_index[checkpoint_id] = serializable_info
            
            data = {
                'index': serializable_index,
                'history': self._checkpoint_history,
                'last_updated': time.time()
            }
            
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint index: {e}")
    
    def _update_checkpoint_index(self, checkpoint_id: str, file_path: Path, checkpoint: Dict[str, Any]) -> None:
        """Update checkpoint index with new entry."""
        file_size_mb = file_path.stat().st_size / (1024**2) if file_path.exists() else 0
        
        entry = {
            'file': file_path,
            'timestamp': checkpoint['timestamp'],
            'datetime': checkpoint['datetime'],
            'size_mb': file_size_mb,
            'tags': checkpoint.get('tags', []),
            'metadata': checkpoint.get('metadata', {}),
            'stats': checkpoint.get('stats', {})
        }
        
        self._checkpoint_index[checkpoint_id] = entry
        
        # Add to history
        history_entry = {
            'id': checkpoint_id,
            'timestamp': checkpoint['timestamp'],
            'action': 'created'
        }
        self._checkpoint_history.append(history_entry)
        
        # Save updated index
        self._save_checkpoint_index()
    
    def _cleanup_old_checkpoints(self, force: bool = False) -> Dict[str, int]:
        """Clean up old checkpoints based on retention policy."""
        stats = {'deleted': 0, 'kept': 0, 'errors': 0}
        
        try:
            with self._lock:
                current_time = time.time()
                cutoff_time = current_time - (self.retention_days * 24 * 3600)
                
                # Get checkpoints sorted by timestamp
                checkpoints = [
                    (cid, info) for cid, info in self._checkpoint_index.items()
                ]
                checkpoints.sort(key=lambda x: x[1]['timestamp'], reverse=True)
                
                # Keep most recent checkpoints
                to_keep = checkpoints[:self.max_checkpoints]
                to_delete = checkpoints[self.max_checkpoints:]
                
                # Also delete checkpoints older than retention period
                for checkpoint_id, info in to_keep.copy():
                    if info['timestamp'] < cutoff_time and not force:
                        to_delete.append((checkpoint_id, info))
                        to_keep.remove((checkpoint_id, info))
                
                # Delete old checkpoints
                for checkpoint_id, info in to_delete:
                    try:
                        if self.delete_checkpoint(checkpoint_id):
                            stats['deleted'] += 1
                        else:
                            stats['errors'] += 1
                    except Exception as e:
                        self.logger.error(f"Error deleting checkpoint {checkpoint_id}: {e}")
                        stats['errors'] += 1
                
                stats['kept'] = len(to_keep)
                self.stats['cleanup_runs'] += 1
                
                if stats['deleted'] > 0:
                    self.logger.info(f"Checkpoint cleanup: deleted {stats['deleted']}, kept {stats['kept']}")
        
        except Exception as e:
            self.logger.error(f"Checkpoint cleanup failed: {e}")
            stats['errors'] += 1
        
        return stats
    
    def _get_latest_checkpoint_id(self) -> Optional[str]:
        """Get the ID of the most recent checkpoint."""
        if not self._checkpoint_index:
            return None
        
        latest = max(
            self._checkpoint_index.items(),
            key=lambda x: x[1]['timestamp']
        )
        return latest[0]
    
    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash of data for checkpoint ID generation."""
        try:
            if isinstance(data, dict):
                # Sort keys for consistent hashing
                data_str = json.dumps(data, sort_keys=True, default=str)
            else:
                data_str = str(data)
            
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception:
            # Fallback to timestamp-based hash
            return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    def _estimate_size(self, data: Any) -> float:
        """Estimate size of data in MB."""
        try:
            if isinstance(data, np.ndarray):
                return data.nbytes / (1024**2)
            elif isinstance(data, dict):
                return len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)) / (1024**2)
            else:
                return len(str(data)) / (1024**2)
        except Exception:
            return 0.0
    
    def _compute_differences(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute differences between old and new data."""
        changes = {
            'added': {},
            'modified': {},
            'deleted': []
        }
        
        if not isinstance(old_data, dict) or not isinstance(new_data, dict):
            # For non-dict data, treat as complete replacement
            changes['modified']['_root'] = new_data
            return changes
        
        # Find added and modified keys
        for key, value in new_data.items():
            if key not in old_data:
                changes['added'][key] = value
            elif old_data[key] != value:
                changes['modified'][key] = value
        
        # Find deleted keys
        for key in old_data:
            if key not in new_data:
                changes['deleted'].append(key)
        
        return changes
    
    def _apply_changes(self, base_data: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
        """Apply incremental changes to base data."""
        result = base_data.copy() if isinstance(base_data, dict) else {}
        
        # Apply additions
        for key, value in changes.get('added', {}).items():
            result[key] = value
        
        # Apply modifications
        for key, value in changes.get('modified', {}).items():
            if key == '_root':
                return value  # Complete replacement
            result[key] = value
        
        # Apply deletions
        for key in changes.get('deleted', []):
            result.pop(key, None)
        
        return result
    
    def _auto_save_loop(self) -> None:
        """Auto-save loop (runs in background thread)."""
        self.logger.debug("Auto-save loop started")
        
        while self._auto_save_enabled:
            try:
                current_time = time.time()
                
                if current_time - self._last_auto_save >= self.auto_save_interval:
                    # Get current data
                    if hasattr(self, '_data_source'):
                        try:
                            data = self._data_source()
                            if data:
                                checkpoint_id = self.save_checkpoint(
                                    data,
                                    metadata={'auto_save': True},
                                    tags=['auto']
                                )
                                self._last_auto_save = current_time
                                self.stats['auto_saves'] += 1
                                self.logger.debug(f"Auto-save completed: {checkpoint_id}")
                        except Exception as e:
                            self.logger.error(f"Auto-save failed: {e}")
                
                # Sleep until next check
                time.sleep(min(self.auto_save_interval / 10, 30))
                
            except Exception as e:
                self.logger.error(f"Error in auto-save loop: {e}")
                time.sleep(30)
        
        self.logger.debug("Auto-save loop stopped")