"""
Base pipeline classes and interfaces for LensingSSC processing workflows.

This module defines the abstract base class and core components for all processing
pipelines in the LensingSSC package. It provides a flexible, extensible framework
with support for checkpointing, parallel execution, resource management, and 
comprehensive error handling.
"""

import os
import time
import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Iterator, Tuple
import json
import pickle
import threading
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np

from ...base.exceptions import (
    ProcessingError, 
    ValidationError, 
    ConfigurationError,
    LensingSSCError
)
from ...base.validation import Validator, DataValidator, validate_not_none
from ...base.data_structures import DataStructure
from ...config.settings import ProcessingConfig


class PipelineState(Enum):
    """Enumeration of possible pipeline states."""
    
    INITIALIZED = "initialized"
    VALIDATING = "validating"
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineResult:
    """Container for pipeline execution results.
    
    This class holds the results of pipeline execution along with metadata
    about the execution process, performance metrics, and any warnings or errors.
    """
    
    success: bool
    output_data: Any = None
    execution_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    steps_completed: int = 0
    total_steps: int = 0
    errors: List[Exception] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.start_time is not None and self.end_time is not None:
            self.execution_time = self.end_time - self.start_time
    
    @property
    def completion_percentage(self) -> float:
        """Get completion percentage.
        
        Returns
        -------
        float
            Completion percentage (0-100)
        """
        if self.total_steps <= 0:
            return 0.0
        return (self.steps_completed / self.total_steps) * 100.0
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors.
        
        Returns
        -------
        bool
            True if there are errors
        """
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings.
        
        Returns
        -------
        bool
            True if there are warnings
        """
        return len(self.warnings) > 0
    
    def add_error(self, error: Exception) -> None:
        """Add an error to the result.
        
        Parameters
        ----------
        error : Exception
            Error to add
        """
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the result.
        
        Parameters
        ----------
        warning : str
            Warning message to add
        """
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.
        
        Returns
        -------
        dict
            Dictionary representation
        """
        return {
            "success": self.success,
            "execution_time": self.execution_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "steps_completed": self.steps_completed,
            "total_steps": self.total_steps,
            "completion_percentage": self.completion_percentage,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics,
            "checkpoint_info": self.checkpoint_info,
        }


class PipelineError(ProcessingError):
    """Specialized exception for pipeline errors.
    
    This exception provides additional context specific to pipeline execution,
    including the pipeline state, step information, and recovery suggestions.
    """
    
    def __init__(self, message: str, pipeline_state: Optional[PipelineState] = None,
                 current_step: Optional[str] = None, step_index: Optional[int] = None,
                 recoverable: bool = False, **kwargs):
        """Initialize pipeline error.
        
        Parameters
        ----------
        message : str
            Error message
        pipeline_state : PipelineState, optional
            Current pipeline state
        current_step : str, optional
            Name of the step where error occurred
        step_index : int, optional
            Index of the step where error occurred
        recoverable : bool
            Whether the error is recoverable
        **kwargs
            Additional arguments for base class
        """
        super().__init__(message, step=current_step, **kwargs)
        self.pipeline_state = pipeline_state
        self.current_step = current_step
        self.step_index = step_index
        self.recoverable = recoverable


@dataclass
class PipelineContext:
    """Execution context for pipeline runs.
    
    This class maintains the execution context including configuration,
    state, progress tracking, and resource management information.
    """
    
    config: ProcessingConfig
    state: PipelineState = PipelineState.INITIALIZED
    current_step: Optional[str] = None
    step_index: int = 0
    total_steps: int = 0
    start_time: Optional[float] = None
    checkpoint_dir: Optional[Path] = None
    temp_dir: Optional[Path] = None
    logger: Optional[logging.Logger] = None
    resource_monitor: Optional[Any] = None
    progress_callback: Optional[Callable[[float, str], None]] = None
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    
    def __post_init__(self):
        """Initialize context after creation."""
        if self.logger is None:
            self.logger = logging.getLogger(f"pipeline.{id(self)}")
        
        # Set up checkpoint directory
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.config.cache_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up temporary directory
        if self.temp_dir is None:
            self.temp_dir = self.config.cache_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def update_progress(self, step_name: str, completion: Optional[float] = None) -> None:
        """Update progress information.
        
        Parameters
        ----------
        step_name : str
            Name of the current step
        completion : float, optional
            Step completion percentage (0-100)
        """
        self.current_step = step_name
        
        if self.progress_callback is not None:
            overall_completion = (self.step_index / max(self.total_steps, 1)) * 100
            if completion is not None:
                # Include step-level completion
                step_contribution = completion / max(self.total_steps, 1)
                overall_completion += step_contribution
            
            self.progress_callback(overall_completion, step_name)
        
        if self.logger:
            self.logger.info(f"Step {self.step_index + 1}/{self.total_steps}: {step_name}")
    
    def is_cancelled(self) -> bool:
        """Check if pipeline execution has been cancelled.
        
        Returns
        -------
        bool
            True if execution should be cancelled
        """
        return self.cancel_flag.is_set()
    
    def cancel(self) -> None:
        """Cancel pipeline execution."""
        self.cancel_flag.set()
        if self.logger:
            self.logger.warning("Pipeline execution cancelled")


class BasePipeline(ABC):
    """Abstract base class for all processing pipelines.
    
    This class defines the interface and common functionality for all pipelines
    in the LensingSSC system. It provides:
    
    - Abstract methods that must be implemented by subclasses
    - Common pipeline execution logic
    - Resource management and cleanup
    - Checkpointing and recovery
    - Progress monitoring and logging
    - Error handling and retry logic
    """
    
    def __init__(self, config: ProcessingConfig, 
                 checkpoint_interval: Optional[int] = None,
                 enable_monitoring: bool = True,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """Initialize the base pipeline.
        
        Parameters
        ----------
        config : ProcessingConfig
            Configuration object for the pipeline
        checkpoint_interval : int, optional
            Number of steps between checkpoints
        enable_monitoring : bool
            Whether to enable resource monitoring
        max_retries : int
            Maximum number of retries for failed steps
        retry_delay : float
            Delay between retries in seconds
        """
        validate_not_none(config, "config")
        
        self.config = config
        self.checkpoint_interval = checkpoint_interval or config.checkpoint_interval
        self.enable_monitoring = enable_monitoring
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize context
        self.context = PipelineContext(config=config)
        
        # Initialize validators
        self.data_validator = DataValidator(strict=config.strict_validation)
        
        # Pipeline state
        self._steps: List[Callable] = []
        self._step_names: List[str] = []
        self._checkpoints: Dict[int, Path] = {}
        
        # Resource management
        self._temp_files: List[Path] = []
        self._resource_locks: List[Any] = []
        
        # Performance monitoring
        self._step_times: List[float] = []
        self._memory_usage: List[float] = []
        
        # Setup pipeline
        self._setup()
    
    def _setup(self) -> None:
        """Setup pipeline components and validate configuration."""
        try:
            # Validate configuration
            self._validate_config()
            
            # Create necessary directories
            self.config.create_directories()
            
            # Initialize steps
            self._create_steps()
            
            # Setup monitoring if enabled
            if self.enable_monitoring:
                self._setup_monitoring()
            
            # Setup logging
            self._setup_logging()
            
            self.context.state = PipelineState.INITIALIZED
            
        except Exception as e:
            raise PipelineError(f"Pipeline setup failed: {e}", 
                              pipeline_state=PipelineState.FAILED) from e
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate pipeline-specific configuration.
        
        This method should be implemented by subclasses to validate
        configuration parameters specific to their functionality.
        
        Raises
        ------
        ConfigurationError
            If configuration is invalid
        """
        pass
    
    @abstractmethod
    def _create_steps(self) -> None:
        """Create the sequence of processing steps.
        
        This method should be implemented by subclasses to define
        the specific sequence of processing steps for their pipeline.
        Steps should be added using the _add_step method.
        """
        pass
    
    @abstractmethod
    def _execute_step(self, step_index: int, input_data: Any) -> Any:
        """Execute a single processing step.
        
        Parameters
        ----------
        step_index : int
            Index of the step to execute
        input_data : Any
            Input data for the step
            
        Returns
        -------
        Any
            Output data from the step
            
        Raises
        ------
        ProcessingError
            If step execution fails
        """
        pass
    
    def _add_step(self, step_func: Callable, name: str) -> None:
        """Add a processing step to the pipeline.
        
        Parameters
        ----------
        step_func : Callable
            Function to execute for this step
        name : str
            Descriptive name for the step
        """
        self._steps.append(step_func)
        self._step_names.append(name)
        self.context.total_steps = len(self._steps)
    
    def execute(self, input_data: Any = None, 
                resume_from_checkpoint: bool = True,
                progress_callback: Optional[Callable[[float, str], None]] = None) -> PipelineResult:
        """Execute the complete pipeline.
        
        Parameters
        ----------
        input_data : Any, optional
            Initial input data for the pipeline
        resume_from_checkpoint : bool
            Whether to resume from an existing checkpoint
        progress_callback : Callable, optional
            Callback function for progress updates
            
        Returns
        -------
        PipelineResult
            Results of pipeline execution
        """
        # Initialize result
        result = PipelineResult(
            success=False,
            start_time=time.time(),
            total_steps=len(self._steps)
        )
        
        try:
            # Setup context
            self.context.start_time = result.start_time
            self.context.progress_callback = progress_callback
            self.context.state = PipelineState.VALIDATING
            
            # Validate input
            if input_data is not None:
                self._validate_input(input_data)
            
            # Check for resumption
            start_step = 0
            current_data = input_data
            
            if resume_from_checkpoint:
                checkpoint_data = self._load_latest_checkpoint()
                if checkpoint_data is not None:
                    start_step = checkpoint_data['step_index'] + 1
                    current_data = checkpoint_data['data']
                    result.steps_completed = start_step
                    self.context.step_index = start_step
                    
                    if self.context.logger:
                        self.context.logger.info(f"Resuming from checkpoint at step {start_step}")
            
            # Execute pipeline steps
            self.context.state = PipelineState.RUNNING
            
            for step_index in range(start_step, len(self._steps)):
                # Check for cancellation
                if self.context.is_cancelled():
                    self.context.state = PipelineState.CANCELLED
                    result.add_warning("Pipeline execution was cancelled")
                    break
                
                # Update context
                self.context.step_index = step_index
                step_name = self._step_names[step_index]
                self.context.update_progress(step_name)
                
                # Execute step with retry logic
                try:
                    step_start_time = time.time()
                    current_data = self._execute_step_with_retry(step_index, current_data)
                    step_time = time.time() - step_start_time
                    self._step_times.append(step_time)
                    
                    result.steps_completed += 1
                    
                    # Create checkpoint if needed
                    if self._should_checkpoint(step_index):
                        self._create_checkpoint(step_index, current_data)
                    
                except Exception as e:
                    error = PipelineError(
                        f"Step '{step_name}' failed: {e}",
                        pipeline_state=self.context.state,
                        current_step=step_name,
                        step_index=step_index
                    )
                    result.add_error(error)
                    self.context.state = PipelineState.FAILED
                    raise error
            
            # Pipeline completed successfully
            if self.context.state != PipelineState.CANCELLED:
                self.context.state = PipelineState.COMPLETED
                result.success = True
                result.output_data = current_data
                
                # Validate output
                self._validate_output(result.output_data)
            
        except Exception as e:
            result.add_error(e)
            if self.context.logger:
                self.context.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        
        finally:
            # Finalize result
            result.end_time = time.time()
            result.performance_metrics = self._get_performance_metrics()
            
            # Cleanup resources
            self._cleanup()
            
            if self.context.logger:
                status = "completed successfully" if result.success else "failed"
                self.context.logger.info(f"Pipeline {status} in {result.execution_time:.2f}s")
        
        return result
    
    def _execute_step_with_retry(self, step_index: int, input_data: Any) -> Any:
        """Execute a step with retry logic.
        
        Parameters
        ----------
        step_index : int
            Index of the step to execute
        input_data : Any
            Input data for the step
            
        Returns
        -------
        Any
            Output data from the step
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return self._execute_step(step_index, input_data)
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries:
                    if self.context.logger:
                        self.context.logger.warning(
                            f"Step {step_index} attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {self.retry_delay}s..."
                        )
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    break
        
        # All retries exhausted
        raise PipelineError(
            f"Step {step_index} failed after {self.max_retries + 1} attempts",
            current_step=self._step_names[step_index],
            step_index=step_index
        ) from last_error
    
    def _validate_input(self, input_data: Any) -> None:
        """Validate input data.
        
        Parameters
        ----------
        input_data : Any
            Input data to validate
            
        Raises
        ------
        ValidationError
            If input validation fails
        """
        if self.config.validate_input:
            if isinstance(input_data, DataStructure):
                if not input_data.validate():
                    raise ValidationError("Input data structure validation failed")
            elif hasattr(input_data, '__array__'):
                if not self.data_validator.validate(input_data):
                    errors = self.data_validator.get_errors()
                    raise ValidationError(f"Input data validation failed: {'; '.join(errors)}")
    
    def _validate_output(self, output_data: Any) -> None:
        """Validate output data.
        
        Parameters
        ----------
        output_data : Any
            Output data to validate
            
        Raises
        ------
        ValidationError
            If output validation fails
        """
        if self.config.validate_input:  # Reuse the same flag for output validation
            if isinstance(output_data, DataStructure):
                if not output_data.validate():
                    raise ValidationError("Output data structure validation failed")
    
    def _should_checkpoint(self, step_index: int) -> bool:
        """Check if a checkpoint should be created.
        
        Parameters
        ----------
        step_index : int
            Current step index
            
        Returns
        -------
        bool
            True if checkpoint should be created
        """
        return (step_index + 1) % self.checkpoint_interval == 0
    
    def _create_checkpoint(self, step_index: int, data: Any) -> None:
        """Create a checkpoint for the current state.
        
        Parameters
        ----------
        step_index : int
            Current step index
        data : Any
            Current data state
        """
        try:
            checkpoint_file = self.context.checkpoint_dir / f"checkpoint_{step_index}.pkl"
            
            checkpoint_data = {
                'step_index': step_index,
                'data': data,
                'timestamp': time.time(),
                'pipeline_class': self.__class__.__name__,
                'config': self.config.to_dict(),
            }
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self._checkpoints[step_index] = checkpoint_file
            
            if self.context.logger:
                self.context.logger.debug(f"Created checkpoint at step {step_index}")
                
        except Exception as e:
            if self.context.logger:
                self.context.logger.warning(f"Failed to create checkpoint: {e}")
    
    def _load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint.
        
        Returns
        -------
        dict or None
            Checkpoint data if available
        """
        try:
            checkpoint_files = list(self.context.checkpoint_dir.glob("checkpoint_*.pkl"))
            if not checkpoint_files:
                return None
            
            # Find the latest checkpoint
            latest_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Validate checkpoint compatibility
            if checkpoint_data.get('pipeline_class') != self.__class__.__name__:
                if self.context.logger:
                    self.context.logger.warning("Checkpoint is from a different pipeline class")
                return None
            
            return checkpoint_data
            
        except Exception as e:
            if self.context.logger:
                self.context.logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def _setup_monitoring(self) -> None:
        """Setup resource monitoring."""
        # This would integrate with system monitoring tools
        # For now, we'll track basic metrics
        pass
    
    def _setup_logging(self) -> None:
        """Setup pipeline-specific logging."""
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self.context.logger = logging.getLogger(logger_name)
        
        # Configure log level from config
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.context.logger.setLevel(level)
        
        # Add file handler if specified
        if self.config.log_file:
            handler = logging.FileHandler(self.config.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.context.logger.addHandler(handler)
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the pipeline execution.
        
        Returns
        -------
        dict
            Performance metrics
        """
        metrics = {
            'total_steps': len(self._steps),
            'completed_steps': len(self._step_times),
            'step_times': self._step_times.copy(),
            'total_step_time': sum(self._step_times),
            'average_step_time': np.mean(self._step_times) if self._step_times else 0,
            'max_step_time': max(self._step_times) if self._step_times else 0,
            'min_step_time': min(self._step_times) if self._step_times else 0,
        }
        
        if self._memory_usage:
            metrics.update({
                'memory_usage': self._memory_usage.copy(),
                'peak_memory': max(self._memory_usage),
                'average_memory': np.mean(self._memory_usage),
            })
        
        return metrics
    
    def _cleanup(self) -> None:
        """Cleanup resources and temporary files."""
        # Remove temporary files
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                if self.context.logger:
                    self.context.logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        # Release resource locks
        for lock in self._resource_locks:
            try:
                if hasattr(lock, 'release'):
                    lock.release()
            except Exception as e:
                if self.context.logger:
                    self.context.logger.warning(f"Failed to release resource lock: {e}")
        
        # Clear lists
        self._temp_files.clear()
        self._resource_locks.clear()
    
    @contextmanager
    def _temporary_file(self, suffix: str = ".tmp", prefix: str = "pipeline_") -> Iterator[Path]:
        """Context manager for temporary files.
        
        Parameters
        ----------
        suffix : str
            File suffix
        prefix : str
            File prefix
            
        Yields
        ------
        Path
            Path to temporary file
        """
        import tempfile
        
        fd, temp_path = tempfile.mkstemp(
            suffix=suffix, 
            prefix=prefix, 
            dir=self.context.temp_dir
        )
        os.close(fd)  # Close the file descriptor
        
        temp_file = Path(temp_path)
        self._temp_files.append(temp_file)
        
        try:
            yield temp_file
        finally:
            # File will be cleaned up in _cleanup()
            pass
    
    def cancel(self) -> None:
        """Cancel pipeline execution."""
        self.context.cancel()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status.
        
        Returns
        -------
        dict
            Status information
        """
        return {
            'state': self.context.state.value,
            'current_step': self.context.current_step,
            'step_index': self.context.step_index,
            'total_steps': self.context.total_steps,
            'completion_percentage': (self.context.step_index / max(self.context.total_steps, 1)) * 100,
            'start_time': self.context.start_time,
            'elapsed_time': time.time() - self.context.start_time if self.context.start_time else 0,
            'is_cancelled': self.context.is_cancelled(),
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup()
        return False  # Don't suppress exceptions


# Utility decorators and functions
def pipeline_step(name: str, validate_input: bool = True, validate_output: bool = True):
    """Decorator for pipeline step functions.
    
    Parameters
    ----------
    name : str
        Step name for logging and monitoring
    validate_input : bool
        Whether to validate step input
    validate_output : bool
        Whether to validate step output
    
    Returns
    -------
    Callable
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(self, input_data: Any) -> Any:
            # Pre-execution validation
            if validate_input and hasattr(self, 'data_validator'):
                if not self.data_validator.validate(input_data):
                    errors = self.data_validator.get_errors()
                    raise ValidationError(f"Step '{name}' input validation failed: {'; '.join(errors)}")
            
            # Execute step
            start_time = time.time()
            try:
                result = func(self, input_data)
            except Exception as e:
                raise ProcessingError(f"Step '{name}' execution failed: {e}") from e
            
            execution_time = time.time() - start_time
            
            # Post-execution validation
            if validate_output and hasattr(self, 'data_validator'):
                if not self.data_validator.validate(result):
                    errors = self.data_validator.get_errors()
                    raise ValidationError(f"Step '{name}' output validation failed: {'; '.join(errors)}")
            
            # Log performance
            if hasattr(self, 'context') and self.context.logger:
                self.context.logger.debug(f"Step '{name}' completed in {execution_time:.3f}s")
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._step_name = name
        return wrapper
    
    return decorator