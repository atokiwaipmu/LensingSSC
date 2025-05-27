"""
Base pipeline architecture for processing workflows.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
from pathlib import Path

from ...core.base.exceptions import ProcessingError
from ...core.config.settings import ProcessingConfig


class BasePipeline(ABC):
    """Abstract base class for processing pipelines."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._steps = []
        self._results = {}
    
    @abstractmethod
    def setup(self) -> None:
        """Setup pipeline steps and dependencies."""
        pass
    
    @abstractmethod
    def validate_inputs(self) -> bool:
        """Validate input data and configuration."""
        pass
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the complete pipeline."""
        try:
            self.logger.info(f"Starting pipeline: {self.__class__.__name__}")
            
            # Setup and validate
            self.setup()
            if not self.validate_inputs():
                raise ProcessingError("Input validation failed")
            
            # Execute steps
            for step in self._steps:
                self.logger.info(f"Executing step: {step.__class__.__name__}")
                step_result = step.execute(self._results, **kwargs)
                self._results[step.name] = step_result
            
            self.logger.info("Pipeline completed successfully")
            return self._results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise ProcessingError(f"Pipeline execution failed: {e}")
    
    def add_step(self, step: 'ProcessingStep') -> None:
        """Add a processing step to the pipeline."""
        self._steps.append(step)
    
    def get_results(self) -> Dict[str, Any]:
        """Get pipeline results."""
        return self._results


class ProcessingStep(ABC):
    """Abstract base class for pipeline steps."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def execute(self, context: Dict[str, Any], **kwargs) -> Any:
        """Execute the processing step."""
        pass
    
    def validate(self, context: Dict[str, Any]) -> bool:
        """Validate step prerequisites."""
        return True