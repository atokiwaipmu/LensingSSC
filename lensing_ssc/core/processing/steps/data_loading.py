"""
Data loading processing steps.
"""

import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import logging

from ..pipeline.base_pipeline import ProcessingStep
from ...core.base.exceptions import ProcessingError
from ...providers.factory import get_provider


class FileDiscoveryStep(ProcessingStep):
    """Step to discover and catalog input files."""
    
    def execute(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute file discovery."""
        input_dir = kwargs.get('input_dir')
        if not input_dir:
            raise ProcessingError("input_dir required for file discovery")
        
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise ProcessingError(f"Input directory does not exist: {input_dir}")
        
        # Discover different file types
        file_catalog = {
            'kappa_maps': list(input_dir.glob("**/kappa_*.fits")),
            'mass_sheets': list(input_dir.glob("**/delta-sheet-*.fits")),
            'patch_files': list(input_dir.glob("**/*patches*.npy")),
            'config_files': list(input_dir.glob("**/*.yaml")) + list(input_dir.glob("**/*.json")),
        }
        
        # Extract metadata from filenames
        file_metadata = {}
        for file_type, files in file_catalog.items():
            file_metadata[file_type] = []
            for file_path in files:
                metadata = self._extract_file_metadata(file_path)
                file_metadata[file_type].append({
                    'path': file_path,
                    'metadata': metadata
                })
        
        self.logger.info(f"Discovered files: {sum(len(files) for files in file_catalog.values())}")
        
        return {
            'file_catalog': file_catalog,
            'file_metadata': file_metadata,
            'input_dir': input_dir
        }
    
    def _extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from filename."""
        from ...utils.extractors import InfoExtractor
        
        metadata = InfoExtractor.extract_info_from_path(file_path)
        
        # Add file-specific metadata
        metadata.update({
            'file_size': file_path.stat().st_size,
            'modification_time': file_path.stat().st_mtime,
            'file_type': file_path.suffix,
        })
        
        return metadata


class DataLoadingStep(ProcessingStep):
    """Step to load data from discovered files."""
    
    def execute(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute data loading."""
        file_discovery = context.get('file_discovery', {})
        file_catalog = file_discovery.get('file_catalog', {})
        
        loaded_data = {}
        
        # Load different data types
        if file_catalog.get('kappa_maps'):
            loaded_data['kappa_maps'] = self._load_kappa_maps(file_catalog['kappa_maps'])
        
        if file_catalog.get('mass_sheets'):
            loaded_data['mass_sheets'] = self._load_mass_sheets(file_catalog['mass_sheets'])
        
        if file_catalog.get('patch_files'):
            loaded_data['patch_files'] = self._load_patch_files(file_catalog['patch_files'])
        
        return {
            'loaded_data': loaded_data,
            'data_summary': self._generate_data_summary(loaded_data)
        }
    
    def _load_kappa_maps(self, kappa_files: List[Path]) -> Dict[str, Any]:
        """Load kappa maps using appropriate provider."""
        healpix_provider = get_provider('healpix')
        
        kappa_data = {}
        
        for file_path in kappa_files:
            try:
                map_data = healpix_provider.read_map(file_path)
                file_key = file_path.stem
                
                kappa_data[file_key] = {
                    'data': map_data,
                    'nside': healpix_provider.get_nside(map_data),
                    'npix': map_data.size,
                    'path': file_path
                }
                
                self.logger.debug(f"Loaded kappa map: {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load kappa map {file_path}: {e}")
                continue
        
        return kappa_data
    
    def _load_mass_sheets(self, mass_sheet_files: List[Path]) -> Dict[str, Any]:
        """Load mass sheet data."""
        healpix_provider = get_provider('healpix')
        
        mass_sheet_data = {}
        
        for file_path in mass_sheet_files:
            try:
                map_data = healpix_provider.read_map(file_path)
                
                # Extract sheet number from filename
                match = re.search(r'delta-sheet-(\d+)', file_path.name)
                sheet_number = int(match.group(1)) if match else None
                
                mass_sheet_data[sheet_number] = {
                    'data': map_data,
                    'path': file_path
                }
                
                self.logger.debug(f"Loaded mass sheet {sheet_number}: {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load mass sheet {file_path}: {e}")
                continue
        
        return mass_sheet_data
    
    def _load_patch_files(self, patch_files: List[Path]) -> Dict[str, Any]:
        """Load patch data from numpy files."""
        patch_data = {}
        
        for file_path in patch_files:
            try:
                patches = np.load(file_path)
                file_key = file_path.stem
                
                patch_data[file_key] = {
                    'patches': patches,
                    'n_patches': len(patches),
                    'patch_shape': patches.shape[1:] if len(patches) > 0 else None,
                    'path': file_path
                }
                
                self.logger.debug(f"Loaded patches: {file_path.name} ({len(patches)} patches)")
                
            except Exception as e:
                self.logger.error(f"Failed to load patches {file_path}: {e}")
                continue
        
        return patch_data
    
    def _generate_data_summary(self, loaded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of loaded data."""
        summary = {}
        
        for data_type, data_dict in loaded_data.items():
            summary[data_type] = {
                'count': len(data_dict),
                'files': list(data_dict.keys())
            }
            
            if data_type == 'kappa_maps':
                nsides = [info['nside'] for info in data_dict.values()]
                summary[data_type]['nsides'] = list(set(nsides))
            
            elif data_type == 'patch_files':
                total_patches = sum(info['n_patches'] for info in data_dict.values())
                summary[data_type]['total_patches'] = total_patches
        
        return summary


class DataValidationStep(ProcessingStep):
    """Step to validate loaded data."""
    
    def execute(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute data validation."""
        data_loading = context.get('data_loading', {})
        loaded_data = data_loading.get('loaded_data', {})
        
        validation_results = {}
        
        for data_type, data_dict in loaded_data.items():
            validation_results[data_type] = self._validate_data_type(data_type, data_dict)
        
        # Overall validation status
        all_valid = all(
            result['valid'] for result in validation_results.values()
        )
        
        return {
            'validation_results': validation_results,
            'all_valid': all_valid,
            'validation_summary': self._generate_validation_summary(validation_results)
        }
    
    def _validate_data_type(self, data_type: str, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate specific data type."""
        if data_type == 'kappa_maps':
            return self._validate_kappa_maps(data_dict)
        elif data_type == 'mass_sheets':
            return self._validate_mass_sheets(data_dict)
        elif data_type == 'patch_files':
            return self._validate_patch_files(data_dict)
        else:
            return {'valid': True, 'errors': [], 'warnings': []}
    
    def _validate_kappa_maps(self, kappa_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate kappa maps."""
        errors = []
        warnings = []
        
        if not kappa_data:
            errors.append("No kappa maps found")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Check consistency
        nsides = [info['nside'] for info in kappa_data.values()]
        if len(set(nsides)) > 1:
            warnings.append(f"Multiple NSIDE values found: {set(nsides)}")
        
        # Check for NaN/inf values
        for file_key, info in kappa_data.items():
            data = info['data'].data
            if np.any(np.isnan(data)):
                warnings.append(f"NaN values found in {file_key}")
            if np.any(np.isinf(data)):
                warnings.append(f"Infinite values found in {file_key}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_mass_sheets(self, mass_sheet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mass sheets."""
        errors = []
        warnings = []
        
        if not mass_sheet_data:
            errors.append("No mass sheets found")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Check sheet numbering
        sheet_numbers = list(mass_sheet_data.keys())
        if None in sheet_numbers:
            warnings.append("Some mass sheets have invalid numbering")
        
        # Check for gaps in numbering
        valid_numbers = [n for n in sheet_numbers if n is not None]
        if valid_numbers:
            min_sheet, max_sheet = min(valid_numbers), max(valid_numbers)
            expected_range = set(range(min_sheet, max_sheet + 1))
            actual_range = set(valid_numbers)
            missing = expected_range - actual_range
            if missing:
                warnings.append(f"Missing mass sheets: {sorted(missing)}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_patch_files(self, patch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate patch files."""
        errors = []
        warnings = []
        
        if not patch_data:
            warnings.append("No patch files found")
            return {'valid': True, 'errors': errors, 'warnings': warnings}
        
        # Check patch consistency
        shapes = [info['patch_shape'] for info in patch_data.values() if info['patch_shape']]
        if len(set(shapes)) > 1:
            warnings.append(f"Multiple patch shapes found: {set(shapes)}")
        
        # Check for empty patch files
        for file_key, info in patch_data.items():
            if info['n_patches'] == 0:
                warnings.append(f"Empty patch file: {file_key}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary."""
        total_errors = sum(len(result['errors']) for result in validation_results.values())
        total_warnings = sum(len(result['warnings']) for result in validation_results.values())
        
        return {
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'data_types_validated': len(validation_results),
            'overall_status': 'valid' if total_errors == 0 else 'invalid'
        }