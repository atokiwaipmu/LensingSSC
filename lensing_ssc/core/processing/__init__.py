"""
Processing pipelines and workflows for LensingSSC.

This module provides a flexible processing framework with:
- Pipeline-based architecture for complex workflows
- Individual processing steps that can be composed
- Resource and checkpoint management
- Error handling and recovery
- Progress tracking and monitoring

The processing system supports both simple and complex workflows:

Simple Usage:
    from lensing_ssc.processing import PreprocessingPipeline
    from lensing_ssc.config import ProcessingConfig
    
    config = ProcessingConfig(data_dir="path/to/data")
    pipeline = PreprocessingPipeline(config)
    results = pipeline.run()

Advanced Usage:
    from lensing_ssc.processing import BasePipeline, DataLoadingStep
    from lensing_ssc.processing.managers import ResourceManager
    
    class CustomPipeline(BasePipeline):
        def setup(self):
            self.add_step(DataLoadingStep("data_loading"))
            # Add more steps...
    
    pipeline = CustomPipeline(config)
    with ResourceManager(memory_limit_mb=8000):
        results = pipeline.run()

Architecture:
    - pipeline/: Base pipeline classes and common pipelines
    - steps/: Individual processing steps
    - managers/: Resource and checkpoint management utilities
"""

import logging
from typing import Dict, Type, Any, Optional, List

# Import pipeline base classes
from .pipeline.base_pipeline import BasePipeline, ProcessingStep

# Import common pipelines
from .pipeline.preprocessing import PreprocessingPipeline
from .pipeline.analysis import AnalysisPipeline

# Import individual steps
from .steps import (
    # Data loading steps
    FileDiscoveryStep,
    DataLoadingStep,
    DataValidationStep,
    
    # Patching steps
    FibonacciGridStep,
    PatchExtractionStep,
    PatchValidationStep,
    
    # Statistics steps
    PowerSpectrumStep,
    BispectrumStep,
    PDFAnalysisStep,
    PeakCountingStep,
    
    # Output steps
    HDF5OutputStep,
    PlotGenerationStep,
    ReportGenerationStep,
)

# Import managers
from .managers import (
    ResourceManager,
    CheckpointManager,
    ProgressManager,
)

# Pipeline registry for dynamic discovery
_PIPELINE_REGISTRY: Dict[str, Type[BasePipeline]] = {
    'preprocessing': PreprocessingPipeline,
    'analysis': AnalysisPipeline,
}

# Step registry for dynamic discovery
_STEP_REGISTRY: Dict[str, Type[ProcessingStep]] = {
    # Data steps
    'file_discovery': FileDiscoveryStep,
    'data_loading': DataLoadingStep,
    'data_validation': DataValidationStep,
    
    # Patching steps
    'fibonacci_grid': FibonacciGridStep,
    'patch_extraction': PatchExtractionStep,
    'patch_validation': PatchValidationStep,
    
    # Statistics steps
    'power_spectrum': PowerSpectrumStep,
    'bispectrum': BispectrumStep,
    'pdf_analysis': PDFAnalysisStep,
    'peak_counting': PeakCountingStep,
    
    # Output steps
    'hdf5_output': HDF5OutputStep,
    'plot_generation': PlotGenerationStep,
    'report_generation': ReportGenerationStep,
}

logger = logging.getLogger(__name__)


def get_pipeline(name: str, config: Any = None, **kwargs) -> BasePipeline:
    """Get a pipeline by name.
    
    Parameters
    ----------
    name : str
        Pipeline name ('preprocessing', 'analysis', etc.)
    config : Any, optional
        Configuration object
    **kwargs
        Additional arguments passed to pipeline constructor
        
    Returns
    -------
    BasePipeline
        Pipeline instance
        
    Raises
    ------
    ValueError
        If pipeline name is not found
    """
    if name not in _PIPELINE_REGISTRY:
        available = list(_PIPELINE_REGISTRY.keys())
        raise ValueError(f"Unknown pipeline '{name}'. Available: {available}")
    
    pipeline_class = _PIPELINE_REGISTRY[name]
    
    if config is not None:
        return pipeline_class(config, **kwargs)
    else:
        return pipeline_class(**kwargs)


def get_step(name: str, step_name: Optional[str] = None, **kwargs) -> ProcessingStep:
    """Get a processing step by name.
    
    Parameters
    ----------
    name : str
        Step type name
    step_name : str, optional
        Instance name for the step (defaults to step type name)
    **kwargs
        Additional arguments passed to step constructor
        
    Returns
    -------
    ProcessingStep
        Step instance
        
    Raises
    ------
    ValueError
        If step name is not found
    """
    if name not in _STEP_REGISTRY:
        available = list(_STEP_REGISTRY.keys())
        raise ValueError(f"Unknown step '{name}'. Available: {available}")
    
    step_class = _STEP_REGISTRY[name]
    instance_name = step_name or name
    
    return step_class(instance_name, **kwargs)


def register_pipeline(name: str, pipeline_class: Type[BasePipeline]) -> None:
    """Register a custom pipeline.
    
    Parameters
    ----------
    name : str
        Pipeline name
    pipeline_class : Type[BasePipeline]
        Pipeline class
    """
    if not issubclass(pipeline_class, BasePipeline):
        raise TypeError("pipeline_class must inherit from BasePipeline")
    
    _PIPELINE_REGISTRY[name] = pipeline_class
    logger.info(f"Registered pipeline: {name}")


def register_step(name: str, step_class: Type[ProcessingStep]) -> None:
    """Register a custom processing step.
    
    Parameters
    ----------
    name : str
        Step name
    step_class : Type[ProcessingStep]
        Step class
    """
    if not issubclass(step_class, ProcessingStep):
        raise TypeError("step_class must inherit from ProcessingStep")
    
    _STEP_REGISTRY[name] = step_class
    logger.info(f"Registered step: {name}")


def list_pipelines() -> List[str]:
    """List available pipeline names.
    
    Returns
    -------
    List[str]
        List of pipeline names
    """
    return list(_PIPELINE_REGISTRY.keys())


def list_steps() -> List[str]:
    """List available step names.
    
    Returns
    -------
    List[str]
        List of step names
    """
    return list(_STEP_REGISTRY.keys())


def create_custom_pipeline(name: str, config: Any, steps: List[str]) -> BasePipeline:
    """Create a custom pipeline from a list of step names.
    
    Parameters
    ----------
    name : str
        Pipeline name
    config : Any
        Configuration object
    steps : List[str]
        List of step names to include
        
    Returns
    -------
    BasePipeline
        Custom pipeline instance
    """
    class CustomPipeline(BasePipeline):
        def __init__(self, config, step_names):
            super().__init__(config)
            self.step_names = step_names
            
        def setup(self) -> None:
            """Setup pipeline with specified steps."""
            for step_name in self.step_names:
                step = get_step(step_name)
                self.add_step(step)
        
        def validate_inputs(self) -> bool:
            """Basic validation - check config exists."""
            return self.config is not None
    
    return CustomPipeline(config, steps)


def run_pipeline(name: str, config: Any, **kwargs) -> Dict[str, Any]:
    """Convenience function to run a pipeline by name.
    
    Parameters
    ----------
    name : str
        Pipeline name
    config : Any
        Configuration object
    **kwargs
        Additional arguments passed to pipeline.run()
        
    Returns
    -------
    Dict[str, Any]
        Pipeline results
    """
    pipeline = get_pipeline(name, config)
    return pipeline.run(**kwargs)


def get_pipeline_info(name: str) -> Dict[str, Any]:
    """Get information about a pipeline.
    
    Parameters
    ----------
    name : str
        Pipeline name
        
    Returns
    -------
    Dict[str, Any]
        Pipeline information
    """
    if name not in _PIPELINE_REGISTRY:
        raise ValueError(f"Unknown pipeline: {name}")
    
    pipeline_class = _PIPELINE_REGISTRY[name]
    
    return {
        'name': name,
        'class_name': pipeline_class.__name__,
        'module': pipeline_class.__module__,
        'docstring': pipeline_class.__doc__,
        'base_classes': [base.__name__ for base in pipeline_class.__bases__],
    }


def get_step_info(name: str) -> Dict[str, Any]:
    """Get information about a processing step.
    
    Parameters
    ----------
    name : str
        Step name
        
    Returns
    -------
    Dict[str, Any]
        Step information
    """
    if name not in _STEP_REGISTRY:
        raise ValueError(f"Unknown step: {name}")
    
    step_class = _STEP_REGISTRY[name]
    
    return {
        'name': name,
        'class_name': step_class.__name__,
        'module': step_class.__module__,
        'docstring': step_class.__doc__,
        'base_classes': [base.__name__ for base in step_class.__bases__],
    }


__all__ = [
    # Base classes
    'BasePipeline',
    'ProcessingStep',
    
    # Common pipelines  
    'PreprocessingPipeline',
    'AnalysisPipeline',
    
    # Individual steps (organized by category)
    # Data loading
    'FileDiscoveryStep',
    'DataLoadingStep', 
    'DataValidationStep',
    
    # Patching
    'FibonacciGridStep',
    'PatchExtractionStep',
    'PatchValidationStep',
    
    # Statistics
    'PowerSpectrumStep',
    'BispectrumStep',
    'PDFAnalysisStep',
    'PeakCountingStep',
    
    # Output
    'HDF5OutputStep',
    'PlotGenerationStep',
    'ReportGenerationStep',
    
    # Managers
    'ResourceManager',
    'CheckpointManager',
    'ProgressManager',
    
    # Registry functions
    'get_pipeline',
    'get_step',
    'register_pipeline',
    'register_step',
    'list_pipelines',
    'list_steps',
    'create_custom_pipeline',
    'run_pipeline',
    'get_pipeline_info',
    'get_step_info',
]