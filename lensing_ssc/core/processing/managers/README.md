# Processing Managers

This directory contains comprehensive manager classes for overseeing various aspects of data processing workflows in the `lensing_ssc` package.

## Overview

The managers provide robust, thread-safe, and configurable solutions for:
- **Resource Management**: System resource monitoring and control
- **Checkpoint Management**: State persistence and recovery
- **Progress Tracking**: Real-time progress reporting and metrics
- **Caching**: Multi-tier data caching with persistence
- **Logging**: Centralized structured logging with performance tracking

## Manager Classes

### ResourceManager (`resource_manager.py`)

Monitors and controls system resources during processing operations.

**Key Features:**
- Real-time monitoring of memory, CPU, disk, and swap usage
- Configurable resource limits with automatic enforcement
- Background monitoring with customizable callbacks
- Context manager support for scoped resource management
- Detailed resource usage reporting and peak tracking

**Usage:**
```python
from lensing_ssc.processing.managers import ResourceManager

# Basic usage with memory limit
with ResourceManager(memory_limit_mb=8000) as rm:
    rm.check_limits()  # Raises exception if exceeded
    usage = rm.get_current_usage()
    print(f"Memory: {usage.memory_mb:.1f}MB")

# With monitoring callbacks
def on_warning(usage, warnings):
    print(f"Resource warning: {warnings}")

rm = ResourceManager(memory_limit_mb=4000)
rm.add_warning_callback(on_warning)
rm.start_monitoring()
```

### CheckpointManager (`checkpoint_manager.py`)

Provides robust checkpoint functionality for long-running processing pipelines.

**Key Features:**
- Automatic checkpoint creation and recovery
- Multiple serialization formats with compression
- Incremental saves and state validation
- Checkpoint history management and cleanup
- Atomic operations with rollback support

**Usage:**
```python
from lensing_ssc.processing.managers import CheckpointManager

# Initialize checkpoint manager
checkpoint_mgr = CheckpointManager("./checkpoints")

# Save processing state
state = {"step": 10, "results": data, "config": params}
checkpoint_mgr.save_checkpoint(state, description="Processing step 10")

# Load latest checkpoint
restored_state = checkpoint_mgr.load_checkpoint()
if restored_state:
    print(f"Resumed from step {restored_state['step']}")

# List available checkpoints
checkpoints = checkpoint_mgr.list_checkpoints()
```

### ProgressManager (`progress_manager.py`)

Tracks and reports processing progress with multiple display formats.

**Key Features:**
- Multi-level progress tracking (main + nested subtasks)
- Real-time console display with progress bars, rates, and ETA
- Thread-safe operations with pause/resume functionality
- Progress persistence for checkpoint integration
- Customizable callbacks and logging integration

**Usage:**
```python
from lensing_ssc.processing.managers import ProgressManager

# Basic progress tracking
with ProgressManager(total=1000, description="Processing items") as pm:
    for i in range(1000):
        # Do work
        pm.update(1)  # Update by 1 item

# Nested progress tracking
with ProgressManager(total=10, description="Main task") as pm:
    for i in range(10):
        with pm.subprogress(f"subtask_{i}", total=100) as sub:
            for j in range(100):
                # Do subtask work
                sub.update(1)
        pm.update(1)
```

### CacheManager (`cache_manager.py`)

Multi-tier caching system with configurable backends and eviction policies.

**Key Features:**
- LRU/LFU/FIFO eviction policies with TTL expiration
- Memory and disk-based caching with compression
- Thread-safe operations with automatic cleanup
- Cache statistics and hit/miss tracking
- Tag-based invalidation and memory pressure handling

**Usage:**
```python
from lensing_ssc.processing.managers import CacheManager

# Initialize cache with size limits
cache = CacheManager(max_size_mb=512, max_entries=1000)

# Cache expensive computation results
def expensive_function(params):
    key = f"computation_{hash(str(params))}"
    result = cache.get(key)
    if result is None:
        result = perform_computation(params)
        cache.put(key, result, ttl=3600)  # Cache for 1 hour
    return result

# Cache with tags for batch invalidation
cache.put("user_data_123", data, tags=["user_123", "recent"])
cache.invalidate_by_tags(["user_123"])  # Remove all user data

# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")
```

### LogManager (`log_manager.py`)

Centralized logging management with structured logging and performance tracking.

**Key Features:**
- Multiple output formats (standard, JSON, detailed)
- Log rotation and archival with size limits
- Context-aware logging with operation tracking
- Performance logging with timing decorators
- Thread-safe operations with comprehensive statistics

**Usage:**
```python
from lensing_ssc.processing.managers import LogManager

# Initialize logging
log_mgr = LogManager(
    level="INFO",
    file_path="./logs/processing.log",
    format_type="json"
)

# Get loggers
logger = log_mgr.get_logger("my_module")
logger.info("Processing started")

# Structured logging with context
with log_mgr.log_context(operation="data_processing", step="validation"):
    logger.info("Validating input data")
    
# Performance logging
@log_mgr.performance.time_function("expensive_operation")
def expensive_operation():
    # Long-running operation
    pass

# Manual timing
with log_mgr.performance.time_operation("batch_processing"):
    # Process batch
    pass
```

## Coordinated Management with ManagerContext

The `ManagerContext` class provides unified management of multiple managers:

```python
from lensing_ssc.processing.managers import ManagerContext

# Coordinate multiple managers
with ManagerContext(
    resource_limit_mb=8000,
    checkpoint_dir="./checkpoints",
    progress_total=1000,
    cache_size_mb=512,
    log_level="INFO"
) as ctx:
    
    # Access individual managers
    ctx.resource.check_limits()
    ctx.progress.update(10)
    ctx.checkpoint.save_checkpoint({"step": 10})
    
    # Use cache
    result = ctx.cache.get("key", default=None)
    
    # Get overall status
    status = ctx.get_manager_status()
```

## Configuration Integration

All managers integrate with the `ProcessingConfig` system:

```python
from lensing_ssc.core.config import ProcessingConfig
from lensing_ssc.processing.managers import ResourceManager

# Load configuration
config = ProcessingConfig.from_file("config.yaml")

# Initialize managers with config
resource_mgr = ResourceManager(config=config)
checkpoint_mgr = CheckpointManager(config=config)
```

## Error Handling

All managers use custom exception classes from `lensing_ssc.core.base.exceptions`:

- `ResourceError`: Resource management failures
- `CheckpointError`: Checkpoint save/load failures  
- `ProgressError`: Progress tracking issues
- `CacheError`: Cache operation failures
- `LoggingError`: Logging configuration problems

## Thread Safety

All managers are designed to be thread-safe and can be used in multi-threaded processing environments. They use appropriate locking mechanisms and atomic operations where necessary.

## Performance Considerations

- **ResourceManager**: Lightweight monitoring with configurable intervals
- **CheckpointManager**: Atomic saves with compression to minimize I/O
- **ProgressManager**: Throttled display updates to reduce overhead
- **CacheManager**: Efficient memory management with background cleanup
- **LogManager**: Asynchronous logging with structured formats

## Integration with Processing Pipelines

The managers are designed to integrate seamlessly with processing pipelines:

```python
def processing_pipeline(data, config):
    with ManagerContext(config=config) as ctx:
        # Setup
        ctx.log_operation_start("data_processing")
        
        # Load previous state if available
        state = ctx.load_checkpoint()
        start_idx = state.get("last_index", 0) if state else 0
        
        # Process with monitoring
        for i, item in enumerate(data[start_idx:], start_idx):
            # Check resources
            ctx.check_resources()
            
            # Process item with caching
            result = process_item_with_cache(item, ctx.cache)
            
            # Update progress
            ctx.update_progress(1)
            
            # Periodic checkpoint
            if i % 100 == 0:
                ctx.save_checkpoint({"last_index": i, "results": results})
        
        # Final checkpoint and cleanup
        ctx.save_checkpoint({"completed": True, "results": results})
        ctx.log_operation_end("data_processing", success=True)
```

## Dependencies

The managers require:
- `psutil` for system resource monitoring
- `threading` for thread-safe operations
- Standard library modules: `logging`, `pickle`, `json`, `time`, `pathlib`

Optional dependencies:
- `gzip` for compression support (included in standard library)

All managers are designed to gracefully handle missing optional dependencies.