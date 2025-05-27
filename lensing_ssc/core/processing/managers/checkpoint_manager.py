"""
Checkpoint management for processing recovery.
"""

import json
import time
from typing import Any, Dict, Optional
from pathlib import Path
import logging

from ...core.base.exceptions import ProcessingError


class CheckpointManager:
    """Manage processing checkpoints for recovery."""
    
    def __init__(self, checkpoint_dir: Path, checkpoint_name: str = "checkpoint"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save checkpoint data."""
        checkpoint = {
            "data": data,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "version": "1.0"
        }
        
        try:
            # Write atomically
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
            temp_file.rename(self.checkpoint_file)
            
            self.logger.debug(f"Checkpoint saved to {self.checkpoint_file}")
        except Exception as e:
            raise ProcessingError(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint data."""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            # Validate checkpoint format
            if "data" not in checkpoint or "timestamp" not in checkpoint:
                self.logger.warning("Invalid checkpoint format, ignoring")
                return None
            
            self.logger.info(f"Checkpoint loaded from {self.checkpoint_file}")
            return checkpoint["data"]
            
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self) -> None:
        """Remove checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                self.logger.info("Checkpoint cleared")
            except Exception as e:
                self.logger.warning(f"Failed to clear checkpoint: {e}")
    
    def checkpoint_exists(self) -> bool:
        """Check if checkpoint exists."""
        return self.checkpoint_file.exists()
    
    def get_checkpoint_age(self) -> Optional[float]:
        """Get checkpoint age in seconds."""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            return time.time() - checkpoint.get("timestamp", 0)
        except Exception:
            return None