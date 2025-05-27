# ====================
# lensing_ssc/core/preprocessing/config.py
# ====================
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import yaml
import json


@dataclass
class ProcessingConfig:
    """Configuration for mass sheet preprocessing."""
    
    # Data access optimization
    chunk_size: int = 50000
    cache_size_mb: int = 1024
    mmap_threshold: int = 1000000
    
    # Processing parameters
    sheet_range: Tuple[int, int] = (20, 100)
    extra_index: int = 100
    overwrite: bool = False
    
    # Parallel processing
    num_workers: Optional[int] = None
    batch_size: int = 10
    
    # Memory management
    memory_limit_mb: Optional[int] = None
    cleanup_interval: int = 50
    max_cache_entries: int = 1000
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_progress_bar: bool = True
    checkpoint_interval: int = 10
    
    # Data validation
    validate_input: bool = True
    strict_validation: bool = False
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'ProcessingConfig':
        """Load configuration from YAML or JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return cls(**data)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.sheet_range[0] >= self.sheet_range[1]:
            raise ValueError("sheet_range must be (start, end) with start < end")
        
        if self.num_workers is not None and self.num_workers <= 0:
            raise ValueError("num_workers must be positive or None")
        
        if self.cache_size_mb <= 0:
            raise ValueError("cache_size_mb must be positive")