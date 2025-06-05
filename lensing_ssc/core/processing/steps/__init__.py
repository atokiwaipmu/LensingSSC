# lensing_ssc/processing/steps/__init__.py
"""
Individual processing steps for LensingSSC pipelines.

This module provides discrete, reusable processing steps that can be composed
into pipelines. Each step is designed to be:

- Self-contained with clear inputs and outputs
- Independently testable and debuggable
- Reusable across different pipeline configurations
- Robust with comprehensive error handling
- Monitorable with detailed progress reporting

Step Categories:
    - Data Steps: File discovery, loading, and validation
    - Patching Steps: Fibonacci grid generation and patch extraction
    - Statistics Steps: Power spectrum, bispectrum, PDF, peak counting
    - Output Steps: HDF5 writing, plotting, and report generation

Architecture:
    Each step inherits from ProcessingStep and implements:
    - execute(): Main processing logic
    - validate_inputs(): Input validation
    - should_skip(): Skip conditions
    - get_step_info(): Metadata about the step

Usage:
    from lensing_ssc.processing.steps import DataLoadingStep, PatchExtractionStep
    
    # Create steps
    loader = DataLoadingStep("load_kappa_maps")
    patcher = PatchExtractionStep("extract_patches", dependencies=["load_kappa_maps"])
    
    # Use in pipeline
    pipeline.add_step(loader)
    pipeline.add_step(patcher)

Step Dependencies:
    Steps can declare dependencies on other steps by name. The pipeline
    ensures proper execution order and provides dependency outputs as inputs.
    
    class CustomStep(ProcessingStep):
        def __init__(self, name):
            super().__init__(name, dependencies=["previous_step"])
        
        def execute(self, context, inputs):
            previous_result = inputs["previous_step"]
            # Process using previous_result.data
"""

import logging
from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path
from abc import ABC, abstractmethod

# Import base classes from pipeline module
from ..pipeline import ProcessingStep, StepResult, PipelineContext, StepStatus
from lensing_ssc.core.base import ValidationError, ProcessingError, DataError

# Import individual step implementations
from .data_loading import (
    FileDiscoveryStep,
    DataLoadingStep,
    DataValidationStep,
)

from .patching import (
    FibonacciGridStep,
    PatchExtractionStep,
    PatchValidationStep,
)

from .statistics import (
    PowerSpectrumStep,
    BispectrumStep,
    PDFAnalysisStep,
    PeakCountingStep,
    CorrelationAnalysisStep,
)

from .output import (
    HDF5OutputStep,
    PlotGenerationStep,
    ReportGenerationStep,
    SummaryStatisticsStep,
)

logger = logging.getLogger(__name__)


class BaseDataStep(ProcessingStep):
    """Base class for data-related processing steps.
    
    Provides common functionality for steps that handle data loading,
    validation, and transformation operations.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.data_formats = ['.fits', '.hdf5', '.npy', '.npz']
        self.max_file_size_gb = 10  # Default size limit
    
    def _validate_file_path(self, file_path: Path) -> bool:
        """Validate that a file path exists and is readable."""
        if not file_path.exists():
            self.logger.error(f"File does not exist: {file_path}")
            return False
        
        if not file_path.is_file():
            self.logger.error(f"Path is not a file: {file_path}")
            return False
        
        # Check file size
        size_gb = file_path.stat().st_size / (1024**3)
        if size_gb > self.max_file_size_gb:
            self.logger.warning(f"Large file detected ({size_gb:.1f} GB): {file_path}")
        
        return True
    
    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get basic information about a file."""
        stat = file_path.stat()
        return {
            'path': str(file_path),
            'name': file_path.name,
            'stem': file_path.stem,
            'suffix': file_path.suffix,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024**2),
            'modified_time': stat.st_mtime,
        }
    
    def _discover_files(self, directory: Path, pattern: str) -> List[Path]:
        """Discover files matching a pattern in a directory."""
        if not directory.exists():
            raise DataError(f"Directory does not exist: {directory}")
        
        files = list(directory.glob(pattern))
        self.logger.info(f"Found {len(files)} files matching '{pattern}' in {directory}")
        
        return sorted(files)


class BaseStatisticsStep(ProcessingStep):
    """Base class for statistics-related processing steps.
    
    Provides common functionality for steps that compute statistical
    measures from patch data.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.requires_lenstools = True
        self.parallel_capable = True
    
    def validate_inputs(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> bool:
        """Validate inputs with statistics-specific checks."""
        if not super().validate_inputs(context, inputs):
            return False
        
        # Check for required statistics dependencies
        if self.requires_lenstools:
            try:
                import lenstools
            except ImportError:
                self.logger.error("lenstools is required for statistics calculations")
                return False
        
        return True
    
    def _validate_patch_data(self, patch_data: Any) -> bool:
        """Validate patch data for statistics computation."""
        import numpy as np
        
        if not isinstance(patch_data, np.ndarray):
            self.logger.error("Patch data must be numpy array")
            return False
        
        if patch_data.ndim < 2:
            self.logger.error("Patch data must be at least 2D")
            return False
        
        if not np.isfinite(patch_data).all():
            self.logger.error("Patch data contains non-finite values")
            return False
        
        return True
    
    def _prepare_analysis_bins(self, lmin: int, lmax: int, n_bins: int) -> Dict[str, Any]:
        """Prepare analysis bins for power spectrum/bispectrum."""
        import numpy as np
        
        l_edges = np.logspace(np.log10(lmin), np.log10(lmax), n_bins + 1)
        l_mids = (l_edges[:-1] + l_edges[1:]) / 2
        
        return {
            'l_edges': l_edges,
            'l_mids': l_mids,
            'n_bins': n_bins,
            'lmin': lmin,
            'lmax': lmax
        }


class BaseOutputStep(ProcessingStep):
    """Base class for output-related processing steps.
    
    Provides common functionality for steps that write results to files,
    generate plots, or create reports.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.overwrite = False
        self.create_backup = True
    
    def _ensure_output_dir(self, output_path: Path) -> Path:
        """Ensure output directory exists."""
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Ensured output directory exists: {output_dir}")
        return output_dir
    
    def _check_output_exists(self, output_path: Path) -> bool:
        """Check if output already exists and handle accordingly."""
        if output_path.exists():
            if not self.overwrite:
                self.logger.info(f"Output exists and overwrite=False: {output_path}")
                return True
            elif self.create_backup:
                backup_path = output_path.with_suffix(f"{output_path.suffix}.backup")
                output_path.rename(backup_path)
                self.logger.info(f"Created backup: {backup_path}")
        
        return False
    
    def _validate_output(self, output_path: Path) -> bool:
        """Validate that output was created successfully."""
        if not output_path.exists():
            self.logger.error(f"Output file was not created: {output_path}")
            return False
        
        if output_path.stat().st_size == 0:
            self.logger.error(f"Output file is empty: {output_path}")
            return False
        
        return True


# Step registry for dynamic discovery and creation
STEP_REGISTRY: Dict[str, Type[ProcessingStep]] = {
    # Data loading steps
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
    'correlation_analysis': CorrelationAnalysisStep,
    
    # Output steps
    'hdf5_output': HDF5OutputStep,
    'plot_generation': PlotGenerationStep,
    'report_generation': ReportGenerationStep,
    'summary_statistics': SummaryStatisticsStep,
}


def create_step(step_type: str, name: Optional[str] = None, **kwargs) -> ProcessingStep:
    """Create a processing step by type.
    
    Parameters
    ----------
    step_type : str
        Type of step to create
    name : str, optional
        Instance name (defaults to step_type)
    **kwargs
        Additional arguments for step constructor
        
    Returns
    -------
    ProcessingStep
        Created step instance
        
    Raises
    ------
    ValueError
        If step_type is not registered
    """
    if step_type not in STEP_REGISTRY:
        available = list(STEP_REGISTRY.keys())
        raise ValueError(f"Unknown step type '{step_type}'. Available: {available}")
    
    step_class = STEP_REGISTRY[step_type]
    instance_name = name or step_type
    
    return step_class(instance_name, **kwargs)


def register_step(step_type: str, step_class: Type[ProcessingStep]) -> None:
    """Register a custom step type.
    
    Parameters
    ----------
    step_type : str
        Step type identifier
    step_class : Type[ProcessingStep]
        Step class to register
    """
    if not issubclass(step_class, ProcessingStep):
        raise TypeError("step_class must inherit from ProcessingStep")
    
    STEP_REGISTRY[step_type] = step_class
    logger.info(f"Registered step type: {step_type}")


def list_step_types() -> List[str]:
    """List available step types.
    
    Returns
    -------
    List[str]
        List of registered step types
    """
    return list(STEP_REGISTRY.keys())


def get_step_info(step_type: str) -> Dict[str, Any]:
    """Get information about a step type.
    
    Parameters
    ----------
    step_type : str
        Step type to query
        
    Returns
    -------
    Dict[str, Any]
        Step type information
    """
    if step_type not in STEP_REGISTRY:
        raise ValueError(f"Unknown step type: {step_type}")
    
    step_class = STEP_REGISTRY[step_type]
    
    return {
        'type': step_type,
        'class_name': step_class.__name__,
        'module': step_class.__module__,
        'docstring': step_class.__doc__,
        'base_classes': [base.__name__ for base in step_class.__bases__],
    }


def create_step_sequence(step_configs: List[Dict[str, Any]]) -> List[ProcessingStep]:
    """Create a sequence of steps from configuration.
    
    Parameters
    ----------
    step_configs : List[Dict[str, Any]]
        List of step configurations, each containing:
        - 'type': step type
        - 'name': instance name (optional)
        - 'dependencies': list of dependency names (optional)
        - Additional step-specific parameters
        
    Returns
    -------
    List[ProcessingStep]
        List of created steps
    """
    steps = []
    
    for config in step_configs:
        step_type = config.pop('type')
        step_name = config.pop('name', step_type)
        dependencies = config.pop('dependencies', [])
        
        step = create_step(step_type, step_name, dependencies=dependencies, **config)
        steps.append(step)
    
    return steps


def validate_step_sequence(steps: List[ProcessingStep]) -> bool:
    """Validate that a sequence of steps has valid dependencies.
    
    Parameters
    ----------
    steps : List[ProcessingStep]
        List of steps to validate
        
    Returns
    -------
    bool
        True if sequence is valid
    """
    step_names = {step.name for step in steps}
    
    for step in steps:
        for dep in step.dependencies:
            if dep not in step_names:
                logger.error(f"Step '{step.name}' depends on unknown step '{dep}'")
                return False
    
    return True


class StepFactory:
    """Factory for creating and managing processing steps.
    
    Provides advanced step creation with configuration management,
    dependency validation, and step customization.
    """
    
    def __init__(self):
        self.registry = STEP_REGISTRY.copy()
        self.default_configs = {}
    
    def register_step_type(self, step_type: str, step_class: Type[ProcessingStep]) -> None:
        """Register a step type with the factory."""
        self.registry[step_type] = step_class
    
    def set_default_config(self, step_type: str, config: Dict[str, Any]) -> None:
        """Set default configuration for a step type."""
        self.default_configs[step_type] = config
    
    def create_step(self, step_type: str, name: Optional[str] = None, **kwargs) -> ProcessingStep:
        """Create a step with default configuration applied."""
        if step_type not in self.registry:
            raise ValueError(f"Unknown step type: {step_type}")
        
        # Merge default config with provided kwargs
        config = self.default_configs.get(step_type, {}).copy()
        config.update(kwargs)
        
        step_class = self.registry[step_type]
        instance_name = name or step_type
        
        return step_class(instance_name, **config)
    
    def create_configured_sequence(self, config: Dict[str, Any]) -> List[ProcessingStep]:
        """Create a sequence of steps from structured configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration with 'steps' key containing step definitions
            
        Returns
        -------
        List[ProcessingStep]
            Configured step sequence
        """
        step_configs = config.get('steps', [])
        steps = []
        
        for step_config in step_configs:
            step = self.create_step(**step_config)
            steps.append(step)
        
        return steps


# Default factory instance
default_factory = StepFactory()

# Convenience functions using default factory
def create_step_with_defaults(step_type: str, name: Optional[str] = None, **kwargs) -> ProcessingStep:
    """Create step using default factory with default configurations."""
    return default_factory.create_step(step_type, name, **kwargs)


__all__ = [
    # Base classes
    'BaseDataStep',
    'BaseStatisticsStep', 
    'BaseOutputStep',
    
    # Data loading steps
    'FileDiscoveryStep',
    'DataLoadingStep',
    'DataValidationStep',
    
    # Patching steps
    'FibonacciGridStep',
    'PatchExtractionStep',
    'PatchValidationStep',
    
    # Statistics steps
    'PowerSpectrumStep',
    'BispectrumStep',
    'PDFAnalysisStep',
    'PeakCountingStep',
    'CorrelationAnalysisStep',
    
    # Output steps
    'HDF5OutputStep',
    'PlotGenerationStep',
    'ReportGenerationStep',
    'SummaryStatisticsStep',
    
    # Registry and factory functions
    'create_step',
    'register_step',
    'list_step_types',
    'get_step_info',
    'create_step_sequence',
    'validate_step_sequence',
    'StepFactory',
    'default_factory',
    'create_step_with_defaults',
    
    # Registry
    'STEP_REGISTRY',
]