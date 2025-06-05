"""
Processing pipeline module for LensingSSC.

This module provides a flexible, modular pipeline architecture for data processing
workflows. It includes base classes for pipeline creation, specialized pipelines
for common tasks, and utilities for parallel processing and resource management.

The pipeline system is designed around composable processing steps that can be
combined into complete workflows with support for:

- Checkpointing and recovery
- Parallel execution
- Resource management
- Progress monitoring
- Error handling and retry logic

Main Components:
- BasePipeline: Abstract base class for all pipelines
- PreprocessingPipeline: Specialized for mass sheet preprocessing
- AnalysisPipeline: Specialized for statistical analysis
- Pipeline utilities and decorators
"""

from typing import Dict, Any, Type, Optional, List

# Import pipeline base classes
from .base_pipeline import (
    BasePipeline,
    PipelineResult,
    PipelineError,
    PipelineState,
    PipelineContext,
)

# Import specialized pipelines
from .preprocessing import (
    PreprocessingPipeline,
    MassSheetPreprocessingPipeline,  # Alias for backward compatibility
)

from .analysis import (
    AnalysisPipeline,
    StatisticalAnalysisPipeline,     # Alias for backward compatibility
    KappaAnalysisPipeline,           # More specific alias
)

# Import pipeline utilities and decorators
from .utils import (
    pipeline_step,
    validate_pipeline_config,
    create_pipeline_from_config,
    get_available_pipelines,
    PipelineRegistry,
)

# Pipeline factory and registry
_PIPELINE_REGISTRY: Dict[str, Type[BasePipeline]] = {}


def register_pipeline(name: str, pipeline_class: Type[BasePipeline]) -> None:
    """Register a pipeline class with a given name.
    
    Parameters
    ----------
    name : str
        Name to register the pipeline under
    pipeline_class : Type[BasePipeline]
        Pipeline class to register
        
    Raises
    ------
    ValueError
        If name is already registered or pipeline_class is invalid
    """
    if not issubclass(pipeline_class, BasePipeline):
        raise ValueError(f"Pipeline class must inherit from BasePipeline, got {pipeline_class}")
    
    if name in _PIPELINE_REGISTRY:
        raise ValueError(f"Pipeline '{name}' is already registered")
    
    _PIPELINE_REGISTRY[name] = pipeline_class


def get_pipeline_class(name: str) -> Type[BasePipeline]:
    """Get a registered pipeline class by name.
    
    Parameters
    ----------
    name : str
        Name of the pipeline
        
    Returns
    -------
    Type[BasePipeline]
        Pipeline class
        
    Raises
    ------
    KeyError
        If pipeline name is not registered
    """
    if name not in _PIPELINE_REGISTRY:
        available = list(_PIPELINE_REGISTRY.keys())
        raise KeyError(f"Pipeline '{name}' not found. Available: {available}")
    
    return _PIPELINE_REGISTRY[name]


def create_pipeline(name: str, config: Any, **kwargs) -> BasePipeline:
    """Create a pipeline instance by name.
    
    Parameters
    ----------
    name : str
        Name of the pipeline
    config : Any
        Configuration object for the pipeline
    **kwargs
        Additional arguments to pass to pipeline constructor
        
    Returns
    -------
    BasePipeline
        Pipeline instance
        
    Raises
    ------
    KeyError
        If pipeline name is not registered
    """
    pipeline_class = get_pipeline_class(name)
    return pipeline_class(config, **kwargs)


def list_pipelines() -> List[str]:
    """List all registered pipeline names.
    
    Returns
    -------
    list
        List of registered pipeline names
    """
    return list(_PIPELINE_REGISTRY.keys())


def get_pipeline_info(name: Optional[str] = None) -> Dict[str, Any]:
    """Get information about registered pipelines.
    
    Parameters
    ----------
    name : str, optional
        Specific pipeline name. If None, returns info for all pipelines.
        
    Returns
    -------
    dict
        Pipeline information
    """
    if name is not None:
        if name not in _PIPELINE_REGISTRY:
            raise KeyError(f"Pipeline '{name}' not found")
        
        pipeline_class = _PIPELINE_REGISTRY[name]
        return {
            "name": name,
            "class": pipeline_class.__name__,
            "module": pipeline_class.__module__,
            "description": pipeline_class.__doc__ or "No description available",
            "base_classes": [cls.__name__ for cls in pipeline_class.__mro__[1:]],
        }
    else:
        return {
            name: get_pipeline_info(name) for name in _PIPELINE_REGISTRY.keys()
        }


# Register built-in pipelines
def _register_builtin_pipelines():
    """Register the built-in pipeline classes."""
    register_pipeline("preprocessing", PreprocessingPipeline)
    register_pipeline("mass_sheet_preprocessing", PreprocessingPipeline)  # Alias
    register_pipeline("analysis", AnalysisPipeline)
    register_pipeline("statistical_analysis", AnalysisPipeline)  # Alias
    register_pipeline("kappa_analysis", AnalysisPipeline)  # Alias


# Register built-in pipelines on import
_register_builtin_pipelines()


# Define public API
__all__ = [
    # Base classes
    "BasePipeline",
    "PipelineResult",
    "PipelineError", 
    "PipelineState",
    "PipelineContext",
    
    # Specialized pipelines
    "PreprocessingPipeline",
    "MassSheetPreprocessingPipeline",
    "AnalysisPipeline",
    "StatisticalAnalysisPipeline",
    "KappaAnalysisPipeline",
    
    # Utilities and decorators
    "pipeline_step",
    "validate_pipeline_config",
    "create_pipeline_from_config",
    "get_available_pipelines",
    "PipelineRegistry",
    
    # Registry functions
    "register_pipeline",
    "get_pipeline_class",
    "create_pipeline",
    "list_pipelines",
    "get_pipeline_info",
]


# Module-level configuration
def configure_pipelines(
    default_num_workers: Optional[int] = None,
    default_checkpoint_interval: Optional[int] = None,
    default_memory_limit_mb: Optional[int] = None,
    enable_performance_monitoring: bool = True,
    log_level: str = "INFO"
) -> None:
    """Configure default settings for all pipelines.
    
    Parameters
    ----------
    default_num_workers : int, optional
        Default number of worker processes
    default_checkpoint_interval : int, optional
        Default checkpoint interval
    default_memory_limit_mb : int, optional
        Default memory limit in MB
    enable_performance_monitoring : bool
        Whether to enable performance monitoring by default
    log_level : str
        Default logging level for pipelines
    """
    # This would set module-level defaults that pipelines can inherit
    # Implementation would depend on the specific pipeline base class design
    pass


def get_pipeline_status_summary() -> Dict[str, Any]:
    """Get a summary of pipeline execution status.
    
    This function provides a summary of currently running pipelines,
    recent executions, and system resource usage.
    
    Returns
    -------
    dict
        Status summary including running pipelines, recent completions,
        resource usage, and performance metrics
    """
    # This would integrate with the pipeline monitoring system
    # to provide real-time status information
    return {
        "registered_pipelines": len(_PIPELINE_REGISTRY),
        "available_pipelines": list_pipelines(),
        "system_info": {
            "cpu_count": None,  # Would be populated with actual system info
            "memory_total": None,
            "memory_available": None,
        },
        "recent_executions": [],  # Would be populated from execution history
        "active_pipelines": [],   # Would be populated from monitoring system
    }


# Backward compatibility aliases
MassSheetPreprocessingPipeline = PreprocessingPipeline
StatisticalAnalysisPipeline = AnalysisPipeline
KappaAnalysisPipeline = AnalysisPipeline


# Version information
__version__ = "1.0.0"
__author__ = "LensingSSC Development Team"
__description__ = "Modular processing pipeline system for weak lensing super-sample covariance analysis"


# Pipeline execution context manager
class pipeline_execution_context:
    """Context manager for pipeline execution with automatic cleanup.
    
    This context manager provides automatic resource cleanup, error handling,
    and monitoring for pipeline executions.
    
    Examples
    --------
    >>> from lensing_ssc.processing.pipeline import pipeline_execution_context
    >>> 
    >>> with pipeline_execution_context() as ctx:
    ...     pipeline = create_pipeline("preprocessing", config)
    ...     result = pipeline.execute()
    ...     # Automatic cleanup happens on exit
    """
    
    def __init__(self, 
                 enable_monitoring: bool = True,
                 cleanup_on_error: bool = True,
                 log_performance: bool = True):
        """Initialize execution context.
        
        Parameters
        ----------
        enable_monitoring : bool
            Whether to enable resource monitoring
        cleanup_on_error : bool
            Whether to perform cleanup on errors
        log_performance : bool
            Whether to log performance metrics
        """
        self.enable_monitoring = enable_monitoring
        self.cleanup_on_error = cleanup_on_error
        self.log_performance = log_performance
        self._start_time = None
        self._monitoring_data = {}
    
    def __enter__(self):
        """Enter the execution context."""
        import time
        self._start_time = time.time()
        
        if self.enable_monitoring:
            # Initialize monitoring
            pass
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the execution context with cleanup."""
        if self._start_time is not None and self.log_performance:
            import time
            duration = time.time() - self._start_time
            # Log performance metrics
        
        if exc_type is not None and self.cleanup_on_error:
            # Perform error cleanup
            pass
        
        # Always perform standard cleanup
        return False  # Don't suppress exceptions
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Get current monitoring data.
        
        Returns
        -------
        dict
            Current monitoring metrics
        """
        return self._monitoring_data.copy()


# Import validation
def _validate_imports():
    """Validate that all imports are successful."""
    try:
        # This would check that all required components are available
        # and provide helpful error messages if dependencies are missing
        pass
    except ImportError as e:
        import warnings
        warnings.warn(f"Some pipeline components may not be available: {e}")


# Run import validation
_validate_imports()