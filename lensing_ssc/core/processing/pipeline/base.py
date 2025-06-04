from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

class ProcessingStep(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Step.{name}")
    
    @abstractmethod
    def execute(self, context: Dict[str, Any], **kwargs) -> Any:
        pass
    
    def __call__(self, context: Dict[str, Any], **kwargs) -> Any:
        self.logger.info(f"Executing {self.name}")
        try:
            result = self.execute(context, **kwargs)
            self.logger.info(f"Completed {self.name}")
            return result
        except Exception as e:
            self.logger.error(f"Failed {self.name}: {e}")
            raise

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.steps: List[ProcessingStep] = []
        self.context: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_step(self, step: ProcessingStep):
        self.steps.append(step)
        return self
    
    def run(self, **kwargs) -> Dict[str, Any]:
        for step in self.steps:
            result = step(self.context, **kwargs)
            self.context[step.name] = result
        return self.context 