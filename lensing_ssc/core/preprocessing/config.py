# preprocessing/config.py
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
    mmap_threshold: int = 1000000  # Use memory mapping for arrays larger than this
    
    # Processing parameters
    sheet_range: Tuple[int, int] = (20, 100)
    extra_index: int = 100
    overwrite: bool = False
    
    # Parallel processing
    num_workers: Optional[int] = None
    batch_size: int = 10
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_progress_bar: bool = True
    checkpoint_interval: int = 10  # Save checkpoint every N sheets
    
    # Memory management
    cleanup_interval: int = 50  # Clean cache every N operations
    max_cache_entries: int = 1000
    
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
    
    def save(self, config_path: Path) -> None:
        """Save configuration to file."""
        data = asdict(self)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
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


@dataclass
class KappaConfig:
    """Configuration for kappa map construction."""
    
    # Output parameters
    nside: int = 8192
    zs_list: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)
    dtype: str = "float32"
    
    # Cosmology parameters
    cosmo_params: Dict[str, float] = None
    
    # Processing
    num_workers: Optional[int] = None
    overwrite: bool = False
    
    def __post_init__(self):
        if self.cosmo_params is None:
            self.cosmo_params = {"H0": 67.74, "Om0": 0.309}
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'KappaConfig':
        """Load kappa configuration from file."""
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls(**data)


def get_default_config() -> ProcessingConfig:
    """Get default processing configuration."""
    return ProcessingConfig()


def get_optimized_config(data_size: int) -> ProcessingConfig:
    """Get optimized configuration based on data size."""
    config = ProcessingConfig()
    
    # Adjust parameters based on data size
    if data_size > 50_000_000_000:  # > 50B records
        config.chunk_size = 100000
        config.cache_size_mb = 2048
        config.num_workers = 8
        config.checkpoint_interval = 5
    elif data_size > 10_000_000_000:  # > 10B records
        config.chunk_size = 50000
        config.cache_size_mb = 1024
        config.num_workers = 4
    
    return config