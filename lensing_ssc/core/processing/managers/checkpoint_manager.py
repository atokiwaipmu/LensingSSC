"""
Checkpoint manager for saving and restoring processing state.

Provides robust checkpoint functionality with automatic recovery, incremental saves,
and state validation for long-running processing pipelines.
"""

import json
import pickle
import time
import hashlib
import shutil
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

from .exceptions import CheckpointError
from ...config.settings import ProcessingConfig


logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files."""
    timestamp: float = field(default_factory=time.time)
    version: str = "1.0"
    checksum: Optional[str] = None
    size_bytes: int = 0
    description: str = ""
    tags: List[str] = field(default_factory=list)
    data_keys: List[str] = field(default_factory=list)
    
    @property
    def datetime_str(self) -> str:
        """Human-readable timestamp."""
        return datetime.fromtimestamp(self.timestamp).isoformat()


class CheckpointManager:
    """Manager for saving and restoring processing state.
    
    Features:
    - Automatic checkpoint creation and recovery
    - Multiple serialization formats (pickle, JSON)
    - Incremental saves and state validation
    - Checkpoint history and cleanup
    - Atomic operations with rollback support
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        checkpoint_name: str = "checkpoint",
        max_checkpoints: int = 10,
        auto_save_interval: Optional[float] = None,
        compression: bool = True,
        validate_on_load: bool = True,
        config: Optional[ProcessingConfig] = None
    ):
        """Initialize checkpoint manager.
        
        Parameters
        ----------
        checkpoint_dir : str or Path
            Directory for checkpoint files
        checkpoint_name : str
            Base name for checkpoint files
        max_checkpoints : int
            Maximum number of checkpoints to keep
        auto_save_interval : float, optional
            Auto-save interval in seconds
        compression : bool
            Enable compression for checkpoint files
        validate_on_load : bool
            Validate checkpoints when loading
        config : ProcessingConfig, optional
            Configuration object
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_name = checkpoint_name
        self.max_checkpoints = max_checkpoints
        self.auto_save_interval = auto_save_interval
        self.compression = compression
        self.validate_on_load = validate_on_load
        
        # Load from config if provided
        if config:
            self.checkpoint_dir = config.cache_dir / "checkpoints"
            
        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self._last_save_time = 0.0
        self._current_checkpoint: Optional[Path] = None
        self._checkpoint_history: List[Path] = []
        self._validators: List[Callable[[Dict[str, Any]], bool]] = []
        
        # Load existing checkpoints
        self._load_checkpoint_history()
        
        logger.debug(f"CheckpointManager initialized: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> Path:
        """Save checkpoint data to file.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Data to checkpoint
        metadata : Dict[str, Any], optional
            Additional metadata
        description : str
            Human-readable description
        tags : List[str], optional
            Tags for categorization
            
        Returns
        -------
        Path
            Path to saved checkpoint file
        """
        timestamp = time.time()
        checkpoint_id = f"{self.checkpoint_name}_{int(timestamp)}"
        
        # Create checkpoint metadata
        checkpoint_meta = CheckpointMetadata(
            timestamp=timestamp,
            description=description,
            tags=tags or [],
            data_keys=list(data.keys())
        )
        
        if metadata:
            for key, value in metadata.items():
                if hasattr(checkpoint_meta, key):
                    setattr(checkpoint_meta, key, value)
        
        # Determine file paths
        data_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        meta_file = self.checkpoint_dir / f"{checkpoint_id}.meta.json"
        
        try:
            # Save data
            self._save_data(data, data_file)
            
            # Update metadata with file info
            checkpoint_meta.size_bytes = data_file.stat().st_size
            checkpoint_meta.checksum = self._calculate_checksum(data_file)
            
            # Save metadata
            self._save_metadata(checkpoint_meta, meta_file)
            
            # Update state
            self._current_checkpoint = data_file
            self._checkpoint_history.append(data_file)
            self._last_save_time = timestamp
            
            # Cleanup old checkpoints
            self._cleanup_checkpoints()
            
            logger.info(f"Checkpoint saved: {data_file}")
            return data_file
            
        except Exception as e:
            # Cleanup partial files
            for file_path in [data_file, meta_file]:
                if file_path.exists():
                    file_path.unlink()
            raise CheckpointError(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        validate: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """Load checkpoint data from file.
        
        Parameters
        ----------
        checkpoint_path : str or Path, optional
            Specific checkpoint to load (default: latest)
        validate : bool, optional
            Override validation setting
            
        Returns
        -------
        Dict[str, Any] or None
            Loaded checkpoint data, None if no checkpoint exists
        """
        if checkpoint_path:
            data_file = Path(checkpoint_path)
        else:
            data_file = self.get_latest_checkpoint()
            
        if not data_file or not data_file.exists():
            logger.info("No checkpoint found to load")
            return None
        
        # Load metadata
        meta_file = data_file.with_suffix('.meta.json')
        metadata = self._load_metadata(meta_file) if meta_file.exists() else None
        
        try:
            # Validate file integrity
            validate_flag = validate if validate is not None else self.validate_on_load
            if validate_flag and metadata:
                if not self._validate_checkpoint(data_file, metadata):
                    raise CheckpointError(f"Checkpoint validation failed: {data_file}")
            
            # Load data
            data = self._load_data(data_file)
            
            # Run custom validators
            if self._validators:
                for validator in self._validators:
                    if not validator(data):
                        raise CheckpointError("Custom validation failed")
            
            self._current_checkpoint = data_file
            logger.info(f"Checkpoint loaded: {data_file}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {data_file}: {e}")
            raise CheckpointError(f"Failed to load checkpoint: {e}")
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint."""
        if not self._checkpoint_history:
            return None
        return max(self._checkpoint_history, key=lambda p: p.stat().st_mtime)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []
        
        for data_file in self._checkpoint_history:
            if not data_file.exists():
                continue
                
            meta_file = data_file.with_suffix('.meta.json')
            metadata = self._load_metadata(meta_file) if meta_file.exists() else None
            
            checkpoint_info = {
                'path': data_file,
                'name': data_file.stem,
                'size_mb': data_file.stat().st_size / (1024 ** 2),
                'modified': datetime.fromtimestamp(data_file.stat().st_mtime),
            }
            
            if metadata:
                checkpoint_info.update({
                    'description': metadata.description,
                    'tags': metadata.tags,
                    'data_keys': metadata.data_keys,
                    'checksum': metadata.checksum,
                })
            
            checkpoints.append(checkpoint_info)
        
        return sorted(checkpoints, key=lambda x: x['modified'], reverse=True)
    
    def delete_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """Delete specific checkpoint."""
        data_file = Path(checkpoint_path)
        meta_file = data_file.with_suffix('.meta.json')
        
        try:
            if data_file.exists():
                data_file.unlink()
            if meta_file.exists():
                meta_file.unlink()
                
            # Remove from history
            if data_file in self._checkpoint_history:
                self._checkpoint_history.remove(data_file)
                
            logger.info(f"Checkpoint deleted: {data_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {data_file}: {e}")
            return False
    
    def cleanup_old_checkpoints(self, keep_count: Optional[int] = None) -> int:
        """Remove old checkpoints beyond limit."""
        keep_count = keep_count or self.max_checkpoints
        
        if len(self._checkpoint_history) <= keep_count:
            return 0
        
        # Sort by modification time
        sorted_checkpoints = sorted(
            self._checkpoint_history,
            key=lambda p: p.stat().st_mtime if p.exists() else 0
        )
        
        # Remove oldest
        to_remove = sorted_checkpoints[:-keep_count]
        removed_count = 0
        
        for checkpoint in to_remove:
            if self.delete_checkpoint(checkpoint):
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} old checkpoints")
        return removed_count
    
    def add_validator(self, validator: Callable[[Dict[str, Any]], bool]) -> None:
        """Add custom checkpoint validator."""
        self._validators.append(validator)
    
    def should_auto_save(self) -> bool:
        """Check if auto-save should be triggered."""
        if not self.auto_save_interval:
            return False
        return (time.time() - self._last_save_time) >= self.auto_save_interval
    
    def _save_data(self, data: Dict[str, Any], file_path: Path) -> None:
        """Save data to file with optional compression."""
        try:
            if self.compression:
                import gzip
                with gzip.open(file_path.with_suffix('.pkl.gz'), 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                # Rename to final path
                file_path.with_suffix('.pkl.gz').rename(file_path)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise CheckpointError(f"Failed to save data: {e}")
    
    def _load_data(self, file_path: Path) -> Dict[str, Any]:
        """Load data from file with compression support."""
        try:
            # Try compressed first
            if self.compression:
                try:
                    import gzip
                    with gzip.open(file_path, 'rb') as f:
                        return pickle.load(f)
                except (gzip.BadGzipFile, OSError):
                    pass
            
            # Try uncompressed
            with open(file_path, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            raise CheckpointError(f"Failed to load data: {e}")
    
    def _save_metadata(self, metadata: CheckpointMetadata, file_path: Path) -> None:
        """Save metadata to JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
    
    def _load_metadata(self, file_path: Path) -> Optional[CheckpointMetadata]:
        """Load metadata from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return CheckpointMetadata(**data)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return None
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
    
    def _validate_checkpoint(self, file_path: Path, metadata: CheckpointMetadata) -> bool:
        """Validate checkpoint file integrity."""
        if not file_path.exists():
            return False
        
        # Check size
        actual_size = file_path.stat().st_size
        if metadata.size_bytes and actual_size != metadata.size_bytes:
            logger.warning(f"Size mismatch: expected {metadata.size_bytes}, got {actual_size}")
            return False
        
        # Check checksum
        if metadata.checksum:
            actual_checksum = self._calculate_checksum(file_path)
            if actual_checksum != metadata.checksum:
                logger.warning("Checksum mismatch")
                return False
        
        return True
    
    def _load_checkpoint_history(self) -> None:
        """Load existing checkpoint files from directory."""
        if not self.checkpoint_dir.exists():
            return
        
        pattern = f"{self.checkpoint_name}_*.pkl"
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        
        # Sort by modification time
        self._checkpoint_history = sorted(
            checkpoint_files,
            key=lambda p: p.stat().st_mtime
        )
        
        logger.debug(f"Found {len(self._checkpoint_history)} existing checkpoints")
    
    def _cleanup_checkpoints(self) -> None:
        """Remove excess checkpoints."""
        if len(self._checkpoint_history) > self.max_checkpoints:
            self.cleanup_old_checkpoints()
    
    def get_status(self) -> Dict[str, Any]:
        """Get checkpoint manager status."""
        return {
            'checkpoint_dir': str(self.checkpoint_dir),
            'checkpoint_count': len(self._checkpoint_history),
            'latest_checkpoint': str(self._current_checkpoint) if self._current_checkpoint else None,
            'last_save_time': self._last_save_time,
            'auto_save_interval': self.auto_save_interval,
            'max_checkpoints': self.max_checkpoints,
            'compression': self.compression,
            'validator_count': len(self._validators),
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass