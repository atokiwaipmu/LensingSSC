# preprocessing/validation.py
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
    
    def validate_usmesh_structure(self, msheets: BigFileCatalog) -> bool:
        """Validate usmesh catalog structure and attributes."""
        self._validate_columns(msheets)
        self._validate_attributes(msheets)
        self._validate_data_consistency(msheets)
        logging.info("usmesh validation passed")
        return True
    
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
        
        if len(edges) != len(offset) - 1:
            raise ValidationError(f"Inconsistent edges ({len(edges)}) and offset ({len(offset)}) lengths")
        
        if not np.all(np.diff(edges) > 0):
            raise ValidationError("aemitIndex.edges must be monotonically increasing")
        
        if not np.all(np.diff(offset) >= 0):
            raise ValidationError("aemitIndex.offset must be non-decreasing")
    
    def _validate_data_consistency(self, msheets: BigFileCatalog) -> None:
        """Validate data consistency with sample checks."""
        total_size = msheets.size
        attrs = msheets.attrs
        
        # Check total size consistency
        expected_max = attrs['aemitIndex.offset'][-1]
        if total_size < expected_max:
            raise ValidationError(f"Data size {total_size} less than expected {expected_max}")
        
        # Sample validation on first 1000 entries
        sample_size = min(1000, total_size)
        if sample_size > 0:
            sample_aemit = msheets['Aemit'][:sample_size].compute()
            
            # Check for reasonable Aemit values (0 < a < 1)
            if np.any(sample_aemit <= 0) or np.any(sample_aemit >= 1):
                logging.warning("Found Aemit values outside expected range (0, 1)")
    
    def validate_indices_file(self, indices_path: Path) -> pd.DataFrame:
        """Validate and load indices CSV file."""
        if not indices_path.exists():
            raise ValidationError(f"Indices file not found: {indices_path}")
        
        try:
            df = pd.read_csv(indices_path)
        except Exception as e:
            raise ValidationError(f"Failed to read indices CSV: {e}")
        
        # Validate required columns
        required_cols = {'sheet', 'start', 'end'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValidationError(f"Indices CSV missing columns: {missing}")
        
        # Validate data types and ranges
        for col in ['sheet', 'start', 'end']:
            if not pd.api.types.is_integer_dtype(df[col]):
                raise ValidationError(f"Column '{col}' must be integer type")
        
        # Validate logical consistency
        invalid_ranges = df[df['start'] >= df['end']]
        if not invalid_ranges.empty:
            raise ValidationError(f"Invalid ranges in sheets: {invalid_ranges['sheet'].tolist()}")
        
        # Check for negative indices
        negative_indices = df[(df['start'] < 0) | (df['end'] < 0)]
        if not negative_indices.empty:
            raise ValidationError(f"Negative indices in sheets: {negative_indices['sheet'].tolist()}")
        
        logging.info(f"Validated indices file with {len(df)} entries")
        return df
    
    def validate_output_directory(self, output_dir: Path, create: bool = True) -> None:
        """Validate output directory exists and is writable."""
        if not output_dir.exists():
            if create:
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    logging.info(f"Created output directory: {output_dir}")
                except Exception as e:
                    raise ValidationError(f"Cannot create output directory {output_dir}: {e}")
            else:
                raise ValidationError(f"Output directory does not exist: {output_dir}")
        
        # Test write permissions
        test_file = output_dir / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise ValidationError(f"Output directory not writable: {e}")
    
    def validate_sheet_processing_feasibility(self, msheets: BigFileCatalog, 
                                            indices_df: pd.DataFrame) -> Dict[str, any]:
        """Validate that sheet processing is feasible given data and indices."""
        attrs = msheets.attrs
        aemit_offset = attrs['aemitIndex.offset']
        total_size = msheets.size
        
        validation_results = {
            "feasible": True,
            "warnings": [],
            "errors": [],
            "stats": {}
        }
        
        # Check each sheet's indices against data bounds
        for _, row in indices_df.iterrows():
            sheet = int(row['sheet'])
            start = int(row['start'])
            end = int(row['end'])
            
            # Check bounds
            if end > total_size:
                validation_results["errors"].append(
                    f"Sheet {sheet}: end index {end} exceeds data size {total_size}"
                )
                validation_results["feasible"] = False
            
            # Check against aemit_offset if sheet index is valid
            if sheet + 2 < len(aemit_offset):
                expected_end = aemit_offset[sheet + 2]
                if abs(end - expected_end) > 1000:  # Allow some tolerance
                    validation_results["warnings"].append(
                        f"Sheet {sheet}: end index {end} differs significantly from expected {expected_end}"
                    )
        
        # Calculate processing statistics
        total_elements = indices_df['end'].sum() - indices_df['start'].sum()
        avg_sheet_size = total_elements / len(indices_df) if len(indices_df) > 0 else 0
        
        validation_results["stats"] = {
            "total_sheets": len(indices_df),
            "total_elements_to_process": int(total_elements),
            "average_sheet_size": int(avg_sheet_size),
            "largest_sheet": int(indices_df['end'].max() - indices_df['start'].min()) if len(indices_df) > 0 else 0
        }
        
        # Memory estimation (rough)
        bytes_per_element = 8 + 8 + 8  # ID, Mass, Aemit (assuming 8 bytes each)
        max_memory_mb = (validation_results["stats"]["largest_sheet"] * bytes_per_element) / (1024**2)
        validation_results["stats"]["estimated_max_memory_mb"] = int(max_memory_mb)
        
        if max_memory_mb > 8192:  # 8GB warning
            validation_results["warnings"].append(
                f"Large memory usage estimated: {max_memory_mb:.0f} MB for largest sheet"
            )
        
        return validation_results
    
    def validate_healpix_parameters(self, npix: int) -> Dict[str, any]:
        """Validate HEALPix parameters."""
        import healpy as hp
        
        try:
            nside = hp.npix2nside(npix)
        except:
            raise ValidationError(f"Invalid npix value: {npix}")
        
        # Check if nside is a power of 2
        if nside & (nside - 1) != 0:
            raise ValidationError(f"nside {nside} is not a power of 2")
        
        return {
            "npix": npix,
            "nside": nside,
            "valid": True,
            "resolution_arcmin": hp.nside2resol(nside, arcmin=True)
        }