"""
Main client interface for LensingSSC.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

from ..core.config.settings import ProcessingConfig, AnalysisConfig, get_config
from ..core.base.exceptions import LensingSSCError
from ..core.providers.factory import get_provider, list_available_providers # Adjusted import path
from ..core.processing.pipeline.preprocessing import PreprocessingPipeline # Adjusted import path
from ..core.processing.pipeline.analysis import AnalysisPipeline # Adjusted import path


class LensingSSCClient:
    """Main client interface for LensingSSC operations."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._providers = {}
    
    def get_provider(self, provider_type: str, **kwargs):
        """Get a provider instance."""
        if provider_type not in self._providers:
            self._providers[provider_type] = get_provider(provider_type, **kwargs)
        return self._providers[provider_type]
    
    def list_providers(self) -> List[str]:
        """List available providers."""
        return list_available_providers()
    
    def preprocess(self, data_dir: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Run preprocessing pipeline."""
        data_dir = Path(data_dir)
        
        # Update config with provided data directory
        processing_config = ProcessingConfig()
        processing_config.data_dir = data_dir
        
        # Override config parameters with kwargs
        for key, value in kwargs.items():
            if hasattr(processing_config, key):
                setattr(processing_config, key, value)
        
        # Create and run pipeline
        pipeline = PreprocessingPipeline(processing_config)
        results = pipeline.run(config=processing_config, **kwargs)
        
        return results
    
    def analyze(self, input_dir: Union[str, Path], 
                analysis_config: Optional[AnalysisConfig] = None, **kwargs) -> Dict[str, Any]:
        """Run analysis pipeline."""
        input_dir = Path(input_dir)
        
        if analysis_config is None:
            analysis_config = AnalysisConfig()
        
        # Create and run analysis pipeline
        pipeline = AnalysisPipeline(analysis_config)
        results = pipeline.run(input_dir=input_dir, **kwargs)
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """Get client information."""
        return {
            "config": self.config.to_dict(),
            "available_providers": self.list_providers(),
            "loaded_providers": list(self._providers.keys())
        }