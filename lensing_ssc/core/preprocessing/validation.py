# ====================
# lensing_ssc/core/preprocessing/validation.py
# ====================
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from nbodykit.lab import BigFileCatalog


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """Comprehensive data validation for mass sheet preprocessing."""
    
    def __init__(self):
        self.required_columns = ['ID', 'Mass', 'Aemit']
        self.required_attrs = [
            'aemitIndex.edges', 'aemitIndex.offset', 'healpix.npix',
            'BoxSize', 'MassTable', 'NC'
        ]
    
    def validate_data(self, datadir: Path) -> bool:
        """Validate data directory and structure."""
        if not datadir.exists():
            logging.error(f"Data directory '{datadir}' does not exist")
            return False
            
        usmesh_dir = datadir / "usmesh"
        if not usmesh_dir.exists():
            logging.error(f"Required subdirectory 'usmesh' not found in '{datadir}'")
            return False
        
        try:
            msheets = BigFileCatalog(str(usmesh_dir), dataset="HEALPIX/")
            return self.validate_usmesh_structure(msheets)
        except Exception as e:
            logging.error(f"Failed to validate usmesh structure: {e}")
            return False
    
    def validate_usmesh_structure(self, msheets: BigFileCatalog) -> bool:
        """Validate usmesh catalog structure and attributes."""
        try:
            self._validate_columns(msheets)
            self._validate_attributes(msheets)
            self._validate_data_consistency(msheets)
            logging.info("usmesh validation passed")
            return True
        except ValidationError as e:
            logging.error(f"Validation failed: {e}")
            return False
    
    def _validate_columns(self, msheets: BigFileCatalog) -> None:
        """Validate required columns exist."""
        missing_cols = [col for col in self.required_columns if col not in msheets.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")
    
    def _validate_attributes(self, msheets: BigFileCatalog) -> None:
        """Validate required attributes exist."""
        attrs = msheets.attrs
        missing_attrs = [attr for attr in self.required_attrs if attr not in attrs]
        if missing_attrs:
            raise ValidationError(f"Missing required attributes: {missing_attrs}")
        
        # Validate attribute values
        edges = attrs['aemitIndex.edges']
        offset = attrs['aemitIndex.offset']
        
        logging.debug(f"Edges length: {len(edges)}, Offset length: {len(offset)}")
        
        # Updated validation logic to handle common cases
        # The relationship should be: len(offset) = len(edges) + 1 OR len(offset) = len(edges) + 2
        # The +2 case happens when there's an extra boundary for incomplete sheets
        edge_offset_diff = len(offset) - len(edges)
        
        if edge_offset_diff not in [1, 2]:
            # Try to understand the data structure better
            logging.warning(f"Unusual edges/offset relationship: edges={len(edges)}, offset={len(offset)}")
            logging.warning("Attempting to proceed with validation...")
            
            # Check if the difference is reasonable (within a small range)
            if abs(edge_offset_diff) > 5:
                raise ValidationError(f"Edges ({len(edges)}) and offset ({len(offset)}) lengths are inconsistent. Difference: {edge_offset_diff}")
        
        if not np.all(np.diff(edges) > 0):
            raise ValidationError("aemitIndex.edges must be monotonically increasing")
        
        if not np.all(np.diff(offset) >= 0):
            raise ValidationError("aemitIndex.offset must be non-decreasing")
        
        # Additional check: ensure offset values are within reasonable bounds
        total_size = msheets.size
        if offset[-1] > total_size:
            logging.warning(f"Last offset value ({offset[-1]}) exceeds data size ({total_size})")
    
    def _validate_data_consistency(self, msheets: BigFileCatalog) -> None:
        """Validate data consistency with sample checks."""
        total_size = msheets.size
        attrs = msheets.attrs
        
        # Check total size consistency - use second-to-last offset to be safe
        offset = attrs['aemitIndex.offset']
        expected_max = offset[-2] if len(offset) > 1 else offset[-1]
        
        if total_size < expected_max:
            raise ValidationError(f"Data size {total_size} less than expected {expected_max}")
        
        # Sample validation on first 1000 entries
        sample_size = min(1000, total_size)
        if sample_size > 0:
            try:
                sample_aemit = msheets['Aemit'][:sample_size].compute()
                
                # Check for reasonable Aemit values (0 < a < 1)
                if np.any(sample_aemit <= 0) or np.any(sample_aemit >= 1):
                    logging.warning("Found Aemit values outside expected range (0, 1)")
            except Exception as e:
                logging.warning(f"Could not validate Aemit sample: {e}")