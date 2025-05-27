"""
Preprocessing pipeline implementation.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np

from .base_pipeline import BasePipeline, ProcessingStep
from ...core.base.exceptions import ProcessingError
from ...providers.factory import get_provider


class PreprocessingPipeline(BasePipeline):
    """Pipeline for mass sheet preprocessing."""
    
    def setup(self) -> None:
        """Setup preprocessing steps."""
        self.add_step(DataValidationStep("data_validation"))
        self.add_step(IndicesCreationStep("indices_creation"))
        self.add_step(MassSheetProcessingStep("mass_sheet_processing"))
    
    def validate_inputs(self) -> bool:
        """Validate preprocessing inputs."""
        data_dir = self.config.data_dir
        
        if not data_dir.exists():
            self.logger.error(f"Data directory not found: {data_dir}")
            return False
        
        usmesh_dir = data_dir / "usmesh"
        if not usmesh_dir.exists():
            self.logger.error(f"usmesh directory not found: {usmesh_dir}")
            return False
        
        return True


class DataValidationStep(ProcessingStep):
    """Step to validate input data structure."""
    
    def execute(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute data validation."""
        from ...core.preprocessing.validation import DataValidator
        
        config = kwargs.get('config')
        validator = DataValidator()
        
        is_valid = validator.validate_data(config.data_dir)
        
        return {
            "is_valid": is_valid,
            "validation_report": validator.get_validation_report()
        }


class IndicesCreationStep(ProcessingStep):
    """Step to create or load processing indices."""
    
    def execute(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute indices creation."""
        from ...core.preprocessing.indices import OptimizedIndicesFinder
        
        config = kwargs.get('config')
        finder = OptimizedIndicesFinder(config.data_dir, config)
        
        # Check if indices exist
        indices_file = config.data_dir / f"preproc_indices.csv"
        if indices_file.exists() and not config.overwrite:
            self.logger.info("Loading existing indices")
            import pandas as pd
            indices_df = pd.read_csv(indices_file)
        else:
            self.logger.info("Creating new indices")
            finder.find_indices(*config.sheet_range)
            indices_df = pd.read_csv(indices_file)
        
        return {
            "indices_df": indices_df,
            "indices_file": indices_file
        }


class MassSheetProcessingStep(ProcessingStep):
    """Step to process mass sheets."""
    
    def execute(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute mass sheet processing."""
        from ...core.preprocessing.processing import MassSheetProcessor
        
        config = kwargs.get('config')
        indices_result = context.get("indices_creation", {})
        
        processor = MassSheetProcessor(
            datadir=config.data_dir,
            config=config
        )
        
        results = processor.preprocess(resume=kwargs.get('resume', False))
        
        return {
            "processing_results": results,
            "output_dir": processor.output_dir
        }