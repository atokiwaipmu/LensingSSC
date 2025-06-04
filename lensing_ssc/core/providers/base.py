from abc import ABC, abstractmethod
import logging

class Provider(ABC):
    def __init__(self):
        self._initialized = False
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def initialize(self, **kwargs):
        if not self._initialized:
            self._check_dependencies()
            self._setup(**kwargs)
            self._initialized = True
    
    def _check_dependencies(self):
        pass
    
    def _setup(self, **kwargs):
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass 