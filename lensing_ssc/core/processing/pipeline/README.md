# Processing Pipeline

This directory contains the modular processing pipeline architecture for LensingSSC. The pipeline system provides a flexible, extensible framework for data processing workflows with support for checkpointing, parallel execution, and resource management.

## Overview

The processing pipeline is built around the concept of composable processing steps that can be combined into complete workflows. This design allows for:

- **Modularity**: Individual processing steps can be developed, tested, and maintained independently
- **Reusability**: Processing steps can be reused across different pipelines
- **Flexibility**: Pipelines can be configured dynamically based on requirements
- **Scalability**: Built-in support for parallel processing and resource management
- **Reliability**: Automatic checkpointing and error recovery

## Architecture

### Core Components

#### Base Pipeline (`base_pipeline.py`)
- Abstract base class defining the pipeline interface
- Common functionality for pipeline execution, validation, and monitoring
- Integration with resource management and checkpointing systems

#### Preprocessing Pipeline (`preprocessing.py`)
- Specialized pipeline for mass sheet preprocessing
- Handles conversion from simulation data to processed mass sheets
- Includes validation, chunking, and parallel processing

#### Analysis Pipeline (`analysis.py`)
- Pipeline for statistical analysis of kappa maps
- Manages patch extraction, statistics calculation, and result aggregation
- Supports multiple analysis configurations and output formats

### Pipeline Features

#### Step-based Architecture
Pipelines are composed of individual processing steps that:
- Have well-defined inputs and outputs
- Can be executed independently or as part of a workflow
- Support validation of inputs and outputs
- Provide progress reporting and logging

#### Resource Management
- Automatic memory management and cleanup
- CPU and memory usage monitoring
- Configurable resource limits and throttling
- Support for distributed processing

#### Checkpointing and Recovery
- Automatic checkpointing at configurable intervals
- Recovery from interruptions with minimal data loss
- Incremental processing for large datasets
- State serialization and restoration

#### Parallel Processing
- Multi-process execution with configurable worker pools
- Task distribution and load balancing
- Error handling and retry mechanisms
- Progress aggregation across workers

## Usage Examples

### Basic Pipeline Usage

```python
from lensing_ssc.processing.pipeline import PreprocessingPipeline
from lensing_ssc.core.config import ProcessingConfig

# Create configuration
config = ProcessingConfig(
    data_dir="/path/to/data",
    output_dir="/path/to/output",
    num_workers=8
)

# Initialize pipeline
pipeline = PreprocessingPipeline(config)

# Execute pipeline
result = pipeline.execute()
```

### Custom Pipeline Creation

```python
from lensing_ssc.processing.pipeline import BasePipeline
from lensing_ssc.processing.steps import DataLoadingStep, ValidationStep

class CustomPipeline(BasePipeline):
    def _create_steps(self):
        return [
            DataLoadingStep(self.config),
            ValidationStep(self.config),
            # Add more steps as needed
        ]
    
    def _validate_config(self):
        # Custom configuration validation
        pass

# Use custom pipeline
pipeline = CustomPipeline(config)
result = pipeline.execute()
```

### Step Composition

```python
from lensing_ssc.processing.steps import (
    DataLoadingStep, 
    PatchingStep, 
    StatisticsStep
)

# Create individual steps
loader = DataLoadingStep(config)
patcher = PatchingStep(config) 
analyzer = StatisticsStep(config)

# Execute step by step
data = loader.execute(input_data)
patches = patcher.execute(data)
results = analyzer.execute(patches)
```

## Pipeline Types

### PreprocessingPipeline
Handles the conversion of raw simulation data to processed mass sheets:

**Input**: Raw simulation data (usmesh files)
**Output**: Processed mass sheets (delta-sheet-*.fits files)

**Steps**:
1. Data discovery and validation
2. Coordinate transformation
3. Mass sheet generation
4. Quality control and validation
5. Output writing

### AnalysisPipeline
Performs statistical analysis on kappa maps:

**Input**: Kappa maps (kappa_*.fits files)
**Output**: Statistical results (HDF5 files)

**Steps**:
1. Kappa map loading
2. Patch extraction
3. Statistical analysis (power spectra, PDFs, peak counts)
4. Result aggregation
5. Output writing

## Configuration

Pipelines are configured using the centralized configuration system:

```yaml
# processing.yaml
pipeline:
  type: "preprocessing"
  batch_size: 10
  checkpoint_interval: 50
  max_retries: 3
  
preprocessing:
  sheet_range: [20, 100]
  chunk_size: 50000
  validation_level: "strict"
  
analysis:
  patch_size_deg: 10.0
  statistics: ["power_spectrum", "pdf", "peak_counts"]
  smoothing_lengths: [2.0, 5.0, 8.0, 10.0]
```

## Error Handling

The pipeline system provides robust error handling:

- **Validation Errors**: Input/output validation with detailed error messages
- **Processing Errors**: Automatic retry with exponential backoff
- **Resource Errors**: Graceful handling of memory/disk space issues
- **System Errors**: Recovery from system-level failures

## Monitoring and Logging

Built-in monitoring provides:

- **Progress Tracking**: Real-time progress reporting with ETA estimates
- **Performance Metrics**: CPU, memory, and I/O usage statistics
- **Error Tracking**: Detailed error logs with context information
- **Resource Usage**: Monitoring of system resources and bottlenecks

## Extension Points

The pipeline system is designed for extensibility:

### Custom Steps
Create new processing steps by inheriting from `BaseStep`:

```python
from lensing_ssc.processing.steps import BaseStep

class CustomProcessingStep(BaseStep):
    def execute(self, input_data):
        # Custom processing logic
        return processed_data
    
    def validate_input(self, input_data):
        # Input validation
        return True
    
    def validate_output(self, output_data):
        # Output validation
        return True
```

### Custom Pipelines
Create specialized pipelines for specific workflows:

```python
from lensing_ssc.processing.pipeline import BasePipeline

class SpecializedPipeline(BasePipeline):
    def _create_steps(self):
        # Define custom step sequence
        pass
    
    def _setup_resources(self):
        # Custom resource setup
        pass
```

### Plugin Integration
The pipeline system supports plugins for extended functionality:

```python
from lensing_ssc.plugins import register_pipeline_plugin

@register_pipeline_plugin("custom_analysis")
class CustomAnalysisPlugin:
    def process(self, data, config):
        # Plugin processing logic
        return results
```

## Performance Considerations

### Memory Management
- Automatic memory cleanup between steps
- Configurable memory limits per step
- Lazy loading for large datasets
- Memory-mapped file support for huge files

### Parallel Processing
- Step-level parallelization for independent operations
- Data-level parallelization for batch processing
- NUMA-aware worker allocation
- Load balancing across available resources

### I/O Optimization
- Asynchronous I/O operations
- Batch reading/writing operations
- Compression for intermediate files
- Smart caching strategies

## Testing

The pipeline system includes comprehensive testing:

```bash
# Run pipeline tests
pytest tests/processing/pipeline/

# Run integration tests
pytest tests/integration/test_pipelines.py

# Run performance tests
pytest tests/performance/test_pipeline_performance.py
```

## Troubleshooting

### Common Issues

#### Memory Issues
```python
# Reduce batch size
config.batch_size = 5

# Enable memory monitoring
config.memory_monitoring = True

# Set memory limits
config.memory_limit_mb = 8192
```

#### Performance Issues
```python
# Increase worker count
config.num_workers = 16

# Enable parallel I/O
config.parallel_io = True

# Adjust chunk size
config.chunk_size = 25000
```

#### Checkpoint Issues
```python
# Enable more frequent checkpointing
config.checkpoint_interval = 10

# Set checkpoint directory
config.checkpoint_dir = "/fast/storage/checkpoints"

# Enable checkpoint compression
config.compress_checkpoints = True
```

For more detailed information, see the individual module documentation and the main LensingSSC documentation.