# lensing_ssc/processing/pipeline/__init__.py
"""
Pipeline architecture for LensingSSC processing workflows.

This module provides the base pipeline classes and common pipeline implementations
for orchestrating complex data processing workflows. The pipeline system is designed
to be modular, extensible, and robust with support for:

- Step-based processing with dependency management
- Resource monitoring and management
- Checkpoint and recovery capabilities
- Progress tracking and logging
- Error handling and validation
- Parallel and distributed processing

Base Architecture:
    - BasePipeline: Abstract base class for all pipelines
    - ProcessingStep: Base class for individual processing steps
    - StepResult: Container for step outputs and metadata
    - PipelineContext: Shared context and resources across steps

Pipeline Types:
    - PreprocessingPipeline: Mass sheet preprocessing workflow
    - AnalysisPipeline: Statistical analysis workflow
    - VisualizationPipeline: Plotting and visualization workflow

Usage:
    from lensing_ssc.processing.pipeline import PreprocessingPipeline
    from lensing_ssc.config import ProcessingConfig
    
    config = ProcessingConfig(data_dir="path/to/data")
    pipeline = PreprocessingPipeline(config)
    
    # Run with default settings
    results = pipeline.run()
    
    # Run with custom settings
    results = pipeline.run(
        resume_from_checkpoint=True,
        max_workers=8,
        memory_limit_mb=8000
    )

Advanced Usage:
    from lensing_ssc.processing.pipeline import BasePipeline
    from lensing_ssc.processing.steps import DataLoadingStep, PatchExtractionStep
    
    class CustomPipeline(BasePipeline):
        def setup(self):
            # Add steps in order
            self.add_step(DataLoadingStep("load_data"))
            self.add_step(PatchExtractionStep("extract_patches"))
            
        def validate_inputs(self):
            return self.config.data_dir.exists()
    
    pipeline = CustomPipeline(config)
    results = pipeline.run()
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable, Type
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import time
import traceback
from enum import Enum

# Import base exceptions from core
from lensing_ssc.core.base import (
    LensingSSCError, 
    ProcessingError, 
    ValidationError,
    ConfigurationError
)

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status enumeration for processing steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStatus(Enum):
    """Status enumeration for pipelines."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StepResult:
    """Container for processing step results and metadata.
    
    Attributes
    ----------
    step_name : str
        Name of the processing step
    status : StepStatus
        Step execution status
    data : Any
        Primary output data from the step
    metadata : Dict[str, Any]
        Additional metadata and intermediate results
    execution_time : float
        Time taken to execute the step in seconds
    memory_usage : Dict[str, float]
        Memory usage statistics during step execution
    error : Optional[Exception]
        Exception if step failed
    warnings : List[str]
        List of warning messages
    """
    step_name: str
    status: StepStatus
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)
    error: Optional[Exception] = None
    warnings: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def is_successful(self) -> bool:
        """Check if step completed successfully."""
        return self.status == StepStatus.COMPLETED
    
    def has_data(self) -> bool:
        """Check if step produced data."""
        return self.data is not None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the step result."""
        return {
            'step_name': self.step_name,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'has_data': self.has_data(),
            'has_error': self.error is not None,
            'warnings_count': len(self.warnings),
            'memory_peak_mb': self.memory_usage.get('peak_mb', 0),
        }


@dataclass 
class PipelineContext:
    """Shared context and resources for pipeline execution.
    
    This class maintains shared state and resources that are accessible
    to all steps in a pipeline.
    
    Attributes
    ----------
    config : Any
        Configuration object
    temp_dir : Path
        Temporary directory for intermediate files
    checkpoint_dir : Path
        Directory for checkpoint files
    shared_data : Dict[str, Any]
        Data shared between steps
    resource_limits : Dict[str, Any]
        Resource limits and constraints
    """
    config: Any
    temp_dir: Path = field(default_factory=lambda: Path.cwd() / ".temp")
    checkpoint_dir: Path = field(default_factory=lambda: Path.cwd() / ".checkpoints")
    shared_data: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize directories."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class ProcessingStep(ABC):
    """Abstract base class for individual processing steps.
    
    Each step represents a discrete unit of work within a pipeline.
    Steps can have dependencies, validate inputs, and produce outputs.
    
    Parameters
    ----------
    name : str
        Unique name for this step instance
    dependencies : List[str], optional
        Names of steps that must complete before this step
    skip_on_failure : bool, optional
        Whether to skip this step if dependencies fail
    """
    
    def __init__(
        self, 
        name: str,
        dependencies: Optional[List[str]] = None,
        skip_on_failure: bool = False
    ):
        self.name = name
        self.dependencies = dependencies or []
        self.skip_on_failure = skip_on_failure
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
    @abstractmethod
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute the processing step.
        
        Parameters
        ----------
        context : PipelineContext
            Shared pipeline context and resources
        inputs : Dict[str, StepResult]
            Results from dependency steps
            
        Returns
        -------
        StepResult
            Result of step execution
        """
        pass
    
    def validate_inputs(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> bool:
        """Validate inputs before execution.
        
        Parameters
        ----------
        context : PipelineContext
            Pipeline context
        inputs : Dict[str, StepResult]
            Input data from dependencies
            
        Returns
        -------
        bool
            True if inputs are valid
        """
        # Check that all dependencies are satisfied
        for dep in self.dependencies:
            if dep not in inputs:
                self.logger.error(f"Missing dependency: {dep}")
                return False
            if not inputs[dep].is_successful():
                self.logger.error(f"Dependency {dep} failed")
                return False
        
        return True
    
    def should_skip(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> bool:
        """Determine if this step should be skipped.
        
        Parameters
        ----------
        context : PipelineContext
            Pipeline context
        inputs : Dict[str, StepResult]
            Input data from dependencies
            
        Returns
        -------
        bool
            True if step should be skipped
        """
        if self.skip_on_failure:
            for dep in self.dependencies:
                if dep in inputs and not inputs[dep].is_successful():
                    return True
        return False
    
    def get_step_info(self) -> Dict[str, Any]:
        """Get information about this step.
        
        Returns
        -------
        Dict[str, Any]
            Step information
        """
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'module': self.__class__.__module__,
            'dependencies': self.dependencies,
            'skip_on_failure': self.skip_on_failure,
        }


class BasePipeline(ABC):
    """Abstract base class for processing pipelines.
    
    A pipeline orchestrates the execution of multiple processing steps,
    managing dependencies, resources, and error handling.
    
    Parameters
    ----------
    config : Any
        Configuration object containing pipeline settings
    name : str, optional
        Pipeline name (defaults to class name)
    """
    
    def __init__(self, config: Any, name: Optional[str] = None):
        self.config = config
        self.name = name or self.__class__.__name__
        self.steps: Dict[str, ProcessingStep] = {}
        self.step_order: List[str] = []
        self.status = PipelineStatus.INITIALIZED
        self.results: Dict[str, StepResult] = {}
        self.context: Optional[PipelineContext] = None
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Execution tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.total_execution_time: float = 0.0
        
        # Callbacks
        self.on_step_start: Optional[Callable[[str], None]] = None
        self.on_step_complete: Optional[Callable[[str, StepResult], None]] = None
        self.on_step_error: Optional[Callable[[str, Exception], None]] = None
    
    @abstractmethod
    def setup(self) -> None:
        """Setup the pipeline by adding and configuring steps.
        
        This method should be implemented by subclasses to define the
        specific steps and their order for the pipeline.
        """
        pass
    
    @abstractmethod
    def validate_inputs(self) -> bool:
        """Validate pipeline inputs and configuration.
        
        Returns
        -------
        bool
            True if inputs are valid
        """
        pass
    
    def add_step(self, step: ProcessingStep) -> None:
        """Add a processing step to the pipeline.
        
        Parameters
        ----------
        step : ProcessingStep
            Step to add
            
        Raises
        ------
        ValueError
            If step name already exists
        """
        if step.name in self.steps:
            raise ValueError(f"Step '{step.name}' already exists in pipeline")
        
        self.steps[step.name] = step
        self.step_order.append(step.name)
        self.logger.debug(f"Added step: {step.name}")
    
    def remove_step(self, step_name: str) -> None:
        """Remove a step from the pipeline.
        
        Parameters
        ----------
        step_name : str
            Name of step to remove
        """
        if step_name in self.steps:
            del self.steps[step_name]
            self.step_order.remove(step_name)
            self.logger.debug(f"Removed step: {step_name}")
    
    def get_step(self, step_name: str) -> ProcessingStep:
        """Get a step by name.
        
        Parameters
        ----------
        step_name : str
            Step name
            
        Returns
        -------
        ProcessingStep
            The requested step
            
        Raises
        ------
        KeyError
            If step not found
        """
        if step_name not in self.steps:
            raise KeyError(f"Step '{step_name}' not found")
        return self.steps[step_name]
    
    def validate_dependencies(self) -> bool:
        """Validate that all step dependencies are satisfied.
        
        Returns
        -------
        bool
            True if all dependencies are valid
        """
        for step_name, step in self.steps.items():
            for dep in step.dependencies:
                if dep not in self.steps:
                    self.logger.error(f"Step '{step_name}' depends on unknown step '{dep}'")
                    return False
        return True
    
    def get_execution_order(self) -> List[str]:
        """Get the order of step execution based on dependencies.
        
        Returns
        -------
        List[str]
            Ordered list of step names
            
        Raises
        ------
        ProcessingError
            If circular dependencies are detected
        """
        # Simple topological sort
        ordered = []
        visited = set()
        temp_visited = set()
        
        def visit(step_name: str):
            if step_name in temp_visited:
                raise ProcessingError(f"Circular dependency detected involving step '{step_name}'")
            if step_name in visited:
                return
            
            temp_visited.add(step_name)
            step = self.steps[step_name]
            
            for dep in step.dependencies:
                if dep in self.steps:  # Only visit existing dependencies
                    visit(dep)
            
            temp_visited.remove(step_name)
            visited.add(step_name)
            ordered.append(step_name)
        
        for step_name in self.step_order:
            visit(step_name)
        
        return ordered
    
    def initialize_context(self, **kwargs) -> PipelineContext:
        """Initialize the pipeline context.
        
        Parameters
        ----------
        **kwargs
            Additional context parameters
            
        Returns
        -------
        PipelineContext
            Initialized context
        """
        context_params = {
            'config': self.config,
        }
        
        # Extract common parameters from config
        if hasattr(self.config, 'temp_dir'):
            context_params['temp_dir'] = Path(self.config.temp_dir)
        if hasattr(self.config, 'checkpoint_dir'):
            context_params['checkpoint_dir'] = Path(self.config.checkpoint_dir)
        
        # Override with kwargs
        context_params.update(kwargs)
        
        return PipelineContext(**context_params)
    
    def run(
        self,
        resume_from_checkpoint: bool = False,
        max_workers: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the pipeline.
        
        Parameters
        ----------
        resume_from_checkpoint : bool, optional
            Whether to resume from previous checkpoint
        max_workers : int, optional
            Maximum number of parallel workers
        memory_limit_mb : int, optional
            Memory limit in megabytes
        **kwargs
            Additional execution parameters
            
        Returns
        -------
        Dict[str, Any]
            Pipeline execution results
        """
        try:
            self.start_time = datetime.now()
            self.status = PipelineStatus.RUNNING
            
            self.logger.info(f"Starting pipeline: {self.name}")
            
            # Setup pipeline
            self.setup()
            
            # Validate inputs
            if not self.validate_inputs():
                raise ValidationError("Pipeline input validation failed")
            
            # Validate dependencies
            if not self.validate_dependencies():
                raise ProcessingError("Pipeline dependency validation failed")
            
            # Initialize context
            self.context = self.initialize_context(
                memory_limit_mb=memory_limit_mb,
                max_workers=max_workers,
                **kwargs
            )
            
            # Get execution order
            execution_order = self.get_execution_order()
            self.logger.info(f"Execution order: {execution_order}")
            
            # Execute steps
            for step_name in execution_order:
                try:
                    result = self._execute_step(step_name)
                    self.results[step_name] = result
                    
                    if not result.is_successful() and not self.steps[step_name].skip_on_failure:
                        raise ProcessingError(f"Step '{step_name}' failed: {result.error}")
                        
                except Exception as e:
                    self.logger.error(f"Error executing step '{step_name}': {e}")
                    if self.on_step_error:
                        self.on_step_error(step_name, e)
                    raise
            
            self.status = PipelineStatus.COMPLETED
            self.end_time = datetime.now()
            self.total_execution_time = (self.end_time - self.start_time).total_seconds()
            
            self.logger.info(f"Pipeline completed successfully in {self.total_execution_time:.2f}s")
            
            return self._create_pipeline_result()
            
        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.end_time = datetime.now()
            if self.start_time:
                self.total_execution_time = (self.end_time - self.start_time).total_seconds()
            
            self.logger.error(f"Pipeline failed: {e}")
            raise ProcessingError(f"Pipeline '{self.name}' failed") from e
    
    def _execute_step(self, step_name: str) -> StepResult:
        """Execute a single step.
        
        Parameters
        ----------
        step_name : str
            Name of step to execute
            
        Returns
        -------
        StepResult
            Step execution result
        """
        step = self.steps[step_name]
        start_time = time.time()
        
        self.logger.info(f"Executing step: {step_name}")
        
        if self.on_step_start:
            self.on_step_start(step_name)
        
        # Gather inputs from dependencies
        inputs = {}
        for dep in step.dependencies:
            if dep in self.results:
                inputs[dep] = self.results[dep]
        
        try:
            # Check if step should be skipped
            if step.should_skip(self.context, inputs):
                self.logger.info(f"Skipping step: {step_name}")
                result = StepResult(
                    step_name=step_name,
                    status=StepStatus.SKIPPED,
                    start_time=datetime.now()
                )
                result.end_time = datetime.now()
                return result
            
            # Validate inputs
            if not step.validate_inputs(self.context, inputs):
                raise ValidationError(f"Input validation failed for step '{step_name}'")
            
            # Execute step
            result = step.execute(self.context, inputs)
            result.execution_time = time.time() - start_time
            
            if self.on_step_complete:
                self.on_step_complete(step_name, result)
            
            self.logger.info(f"Step '{step_name}' completed in {result.execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Step '{step_name}' failed after {execution_time:.2f}s: {e}")
            
            result = StepResult(
                step_name=step_name,
                status=StepStatus.FAILED,
                error=e,
                execution_time=execution_time,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.now()
            )
            
            return result
    
    def _create_pipeline_result(self) -> Dict[str, Any]:
        """Create the final pipeline result.
        
        Returns
        -------
        Dict[str, Any]
            Pipeline results and metadata
        """
        successful_steps = [name for name, result in self.results.items() if result.is_successful()]
        failed_steps = [name for name, result in self.results.items() if result.status == StepStatus.FAILED]
        skipped_steps = [name for name, result in self.results.items() if result.status == StepStatus.SKIPPED]
        
        return {
            'pipeline_name': self.name,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_execution_time': self.total_execution_time,
            'step_results': {name: result.get_summary() for name, result in self.results.items()},
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'skipped_steps': skipped_steps,
            'step_count': len(self.steps),
            'success_rate': len(successful_steps) / len(self.steps) if self.steps else 0,
            'data': {name: result.data for name, result in self.results.items() if result.has_data()},
            'metadata': {name: result.metadata for name, result in self.results.items()},
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline.
        
        Returns
        -------
        Dict[str, Any]
            Pipeline information
        """
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'module': self.__class__.__module__,
            'status': self.status.value,
            'step_count': len(self.steps),
            'steps': {name: step.get_step_info() for name, step in self.steps.items()},
            'execution_order': self.get_execution_order() if self.validate_dependencies() else None,
        }


# Factory functions for common pipeline operations
def create_pipeline_from_config(config: Any, pipeline_type: str = "auto") -> BasePipeline:
    """Create a pipeline from configuration.
    
    Parameters
    ----------
    config : Any
        Configuration object
    pipeline_type : str, optional
        Type of pipeline to create ("preprocessing", "analysis", "auto")
        
    Returns
    -------
    BasePipeline
        Pipeline instance
    """
    # This will be implemented when specific pipeline classes are available
    from .preprocessing import PreprocessingPipeline
    from .analysis import AnalysisPipeline
    
    if pipeline_type == "preprocessing":
        return PreprocessingPipeline(config)
    elif pipeline_type == "analysis":
        return AnalysisPipeline(config)
    elif pipeline_type == "auto":
        # Auto-detect based on config attributes
        if hasattr(config, 'mass_sheet_dir') or hasattr(config, 'datadir'):
            return PreprocessingPipeline(config)
        elif hasattr(config, 'stats_output_dir') or hasattr(config, 'patch_output_dir'):
            return AnalysisPipeline(config)
        else:
            raise ValueError("Cannot auto-detect pipeline type from config")
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


__all__ = [
    # Enums
    'StepStatus',
    'PipelineStatus',
    
    # Data structures
    'StepResult',
    'PipelineContext',
    
    # Base classes
    'ProcessingStep',
    'BasePipeline',
    
    # Factory functions
    'create_pipeline_from_config',
]