# lensing_ssc/processing/steps/patching.py
"""
Patching steps for LensingSSC processing pipelines.

Provides steps for Fibonacci grid generation, patch extraction from full-sky maps,
and patch validation for weak lensing analysis.
"""

import logging
import time
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np

from ..pipeline import ProcessingStep, StepResult, PipelineContext, StepStatus
from lensing_ssc.core.base import ValidationError, ProcessingError, GeometryError


logger = logging.getLogger(__name__)


class FibonacciGridStep(ProcessingStep):
    """Generate optimal Fibonacci grid points for patch extraction."""
    
    def __init__(
        self,
        name: str,
        patch_size_deg: float = 10.0,
        nside: int = 1024,
        ninit: int = 280,
        optimize_count: bool = True,
        save_points: bool = True,
        points_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.patch_size_deg = patch_size_deg
        self.nside = nside
        self.ninit = ninit
        self.optimize_count = optimize_count
        self.save_points = save_points
        self.points_dir = Path(points_dir) if points_dir else None
        
        # Calculate patch radius in radians
        self.radius = np.radians(patch_size_deg) * np.sqrt(2)
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute Fibonacci grid generation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Determine output directory
            if not self.points_dir:
                self.points_dir = self._get_points_dir_from_context(context)
            
            # Check for existing points
            points_file = self.points_dir / f"fibonacci_points_{self.patch_size_deg}.txt"
            
            if points_file.exists():
                self.logger.info(f"Loading existing Fibonacci points from {points_file}")
                points = np.loadtxt(points_file)
                n_opt = len(points)
            else:
                # Generate new points
                if self.optimize_count:
                    n_opt = self._optimize_patch_count()
                else:
                    n_opt = self.ninit
                
                points = self._generate_fibonacci_grid(n_opt)
                
                if self.save_points:
                    self.points_dir.mkdir(parents=True, exist_ok=True)
                    np.savetxt(points_file, points)
                    self.logger.info(f"Saved Fibonacci points to {points_file}")
            
            # Validate and filter points
            valid_points = self._filter_valid_points(points)
            
            result.data = {
                'fibonacci_points': points,
                'valid_points': valid_points,
                'n_total': len(points),
                'n_valid': len(valid_points),
                'n_optimal': n_opt,
                'points_file': str(points_file),
                'patch_radius': self.radius,
                'patch_size_deg': self.patch_size_deg
            }
            
            result.metadata = {
                'n_total_points': len(points),
                'n_valid_points': len(valid_points),
                'n_optimal': n_opt,
                'patch_size_deg': self.patch_size_deg,
                'optimization_used': self.optimize_count,
                'points_saved': self.save_points,
                'validity_rate': len(valid_points) / len(points) if points.size > 0 else 0
            }
            
            self.logger.info(f"Generated Fibonacci grid: {len(valid_points)}/{len(points)} valid points")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _get_points_dir_from_context(self, context: PipelineContext) -> Path:
        """Get points directory from context."""
        config = context.config
        
        # Try various config attributes
        for attr in ['center_points_path', 'points_dir', 'output_dir', 'temp_dir']:
            if hasattr(config, attr):
                path = getattr(config, attr)
                if path:
                    return Path(path)
        
        # Default to context temp directory
        return context.temp_dir / "fibonacci_points"
    
    def _optimize_patch_count(self) -> int:
        """Optimize the number of non-overlapping patches."""
        try:
            from lensing_ssc.core.fibonacci_utils import PatchOptimizer
            
            optimizer = PatchOptimizer(
                nside=self.nside,
                patch_size=self.patch_size_deg,
                Ninit=self.ninit
            )
            
            n_opt = optimizer.optimize(verbose=True)
            self.logger.info(f"Optimized patch count: {n_opt}")
            
            return n_opt
            
        except ImportError:
            self.logger.warning("PatchOptimizer not available, using initial count")
            return self.ninit
        except Exception as e:
            self.logger.warning(f"Patch optimization failed: {e}, using initial count")
            return self.ninit
    
    def _generate_fibonacci_grid(self, n_points: int) -> np.ndarray:
        """Generate Fibonacci grid points."""
        try:
            from lensing_ssc.core.fibonacci_utils import FibonacciGrid
            
            # Ensure odd number of points
            if n_points % 2 == 0:
                n_points += 1
            
            points = FibonacciGrid.fibonacci_grid_on_sphere(n_points)
            self.logger.info(f"Generated {len(points)} Fibonacci points")
            
            return points
            
        except ImportError:
            # Fallback implementation
            self.logger.warning("FibonacciGrid not available, using fallback implementation")
            return self._fallback_fibonacci_grid(n_points)
        except Exception as e:
            self.logger.warning(f"Fibonacci grid generation failed: {e}, using fallback")
            return self._fallback_fibonacci_grid(n_points)
    
    def _fallback_fibonacci_grid(self, n_points: int) -> np.ndarray:
        """Fallback Fibonacci grid implementation."""
        if n_points < 3 or n_points % 2 == 0:
            n_points = max(3, n_points + (n_points % 2))
        
        N = (n_points - 1) // 2
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(-N, N + 1, 1, dtype=int)
        
        theta_i = np.arcsin(2 * indices / (2 * N + 1))
        phi_i = 2 * np.pi * indices / golden_ratio
        
        # Shift theta into [0, π]
        theta_i += np.pi / 2
        # Wrap phi into [0, 2π)
        phi_i = np.mod(phi_i, 2 * np.pi)
        
        return np.column_stack((theta_i, phi_i))
    
    def _filter_valid_points(self, points: np.ndarray) -> np.ndarray:
        """Filter points that are too close to poles."""
        if points.size == 0:
            return points
        
        # Filter points too close to poles
        valid_mask = (
            (points[:, 0] < (np.pi - self.radius)) &
            (points[:, 0] > self.radius)
        )
        
        valid_points = points[valid_mask]
        self.logger.debug(f"Filtered {len(points) - len(valid_points)} points near poles")
        
        return valid_points


class PatchExtractionStep(ProcessingStep):
    """Extract patches from full-sky maps using Fibonacci grid points."""
    
    def __init__(
        self,
        name: str,
        xsize: int = 2048,
        padding: float = 0.1,
        num_processes: Optional[int] = None,
        chunk_size: int = 10,
        save_patches: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.xsize = xsize
        self.padding = padding + np.sqrt(2)  # Additional padding for rotation
        self.num_processes = num_processes or mp.cpu_count()
        self.chunk_size = chunk_size
        self.save_patches = save_patches
        self.output_dir = Path(output_dir) if output_dir else None
        self.overwrite = overwrite
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute patch extraction."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Get Fibonacci grid points
            grid_result = self._get_grid_result(inputs)
            if not grid_result:
                raise ProcessingError("Fibonacci grid step result not found or failed")
            
            # Get map data to process
            map_data = self._get_map_data(inputs, context)
            if map_data is None:
                raise ProcessingError("No map data found for patch extraction")
            
            valid_points = grid_result.data['valid_points']
            patch_size_deg = grid_result.data['patch_size_deg']
            
            self.logger.info(f"Extracting patches from {len(valid_points)} points")
            
            # Calculate resolution
            resolution_arcmin = (patch_size_deg * 60.0) / self.xsize
            
            # Convert points to lon/lat for healpy
            points_lonlat = self._convert_points_to_lonlat(valid_points)
            
            # Extract patches
            patches = self._extract_patches_parallel(
                map_data, points_lonlat, resolution_arcmin
            )
            
            # Save patches if requested
            output_file = None
            if self.save_patches:
                output_file = self._save_patches(patches, context, patch_size_deg)
            
            result.data = {
                'patches': patches,
                'n_patches': len(patches),
                'patch_shape': patches.shape if len(patches) > 0 else None,
                'patch_size_deg': patch_size_deg,
                'resolution_arcmin': resolution_arcmin,
                'output_file': str(output_file) if output_file else None,
                'extraction_points': points_lonlat
            }
            
            result.metadata = {
                'n_patches_extracted': len(patches),
                'patch_shape': patches.shape if len(patches) > 0 else None,
                'patch_size_deg': patch_size_deg,
                'xsize': self.xsize,
                'resolution_arcmin': resolution_arcmin,
                'num_processes': self.num_processes,
                'patches_saved': self.save_patches,
            }
            
            self.logger.info(f"Extracted {len(patches)} patches with shape {patches.shape if len(patches) > 0 else 'N/A'}")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _get_grid_result(self, inputs: Dict[str, StepResult]) -> Optional[StepResult]:
        """Get Fibonacci grid result from inputs."""
        for step_result in inputs.values():
            if (step_result.is_successful() and 
                'valid_points' in step_result.data and
                'patch_size_deg' in step_result.data):
                return step_result
        return None
    
    def _get_map_data(self, inputs: Dict[str, StepResult], context: PipelineContext) -> Optional[np.ndarray]:
        """Get map data from inputs or context."""
        # Try to get from data loading step
        for step_result in inputs.values():
            if (step_result.is_successful() and 'loaded_data' in step_result.data):
                loaded_data = step_result.data['loaded_data']
                for file_path, data_info in loaded_data.items():
                    if data_info['data_type'] == 'healpix_map':
                        return data_info['data']
        
        # Try to get from shared context
        for key, value in context.shared_data.items():
            if 'map' in key.lower() and isinstance(value, np.ndarray):
                return value
        
        # Try to get from config
        config = context.config
        if hasattr(config, 'map_data') and config.map_data is not None:
            return config.map_data
        
        return None
    
    def _convert_points_to_lonlat(self, points: np.ndarray) -> List[Tuple[float, float]]:
        """Convert spherical coordinates to lon/lat for healpy."""
        try:
            import healpy as hp
            points_lonlat = []
            
            for point in points:
                theta, phi = point
                # Convert to healpy's convention and then to lon/lat
                vec = hp.ang2vec(theta, phi)
                lon, lat = hp.rotator.vec2dir(vec, lonlat=True)
                points_lonlat.append((lon, lat))
            
            return points_lonlat
            
        except ImportError:
            raise ProcessingError("healpy is required for coordinate conversion")
    
    def _extract_patches_parallel(
        self, 
        map_data: np.ndarray, 
        points_lonlat: List[Tuple[float, float]], 
        resolution_arcmin: float
    ) -> np.ndarray:
        """Extract patches using parallel processing."""
        try:
            from multiprocessing import shared_memory
            
            # Create shared memory for the map
            shm = shared_memory.SharedMemory(create=True, size=map_data.nbytes)
            shared_map = np.ndarray(map_data.shape, dtype=map_data.dtype, buffer=shm.buf)
            np.copyto(shared_map, map_data)
            
            try:
                # Prepare arguments for workers
                args_list = [
                    (shm.name, map_data.shape, map_data.dtype, point, resolution_arcmin)
                    for point in points_lonlat
                ]
                
                # Process in parallel
                with mp.Pool(processes=self.num_processes) as pool:
                    patches = pool.starmap(self._extract_single_patch, args_list)
                
                # Filter out None results and convert to array
                valid_patches = [p for p in patches if p is not None]
                
                if not valid_patches:
                    return np.array([])
                
                return np.array(valid_patches, dtype=np.float32)
                
            finally:
                shm.close()
                shm.unlink()
                
        except ImportError:
            self.logger.warning("Shared memory not available, using sequential processing")
            return self._extract_patches_sequential(map_data, points_lonlat, resolution_arcmin)
    
    def _extract_single_patch(
        self, 
        shm_name: str, 
        shape: Tuple[int, ...], 
        dtype: np.dtype, 
        point_lonlat: Tuple[float, float], 
        resolution_arcmin: float
    ) -> Optional[np.ndarray]:
        """Extract a single patch (worker function)."""
        try:
            from multiprocessing import shared_memory
            import healpy as hp
            from lensing_ssc.core.fibonacci_utils import FibonacciGrid
            
            # Access shared memory
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            input_map = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
            
            # Extract patch using gnomonic projection
            patch_projected = hp.gnomview(
                input_map,
                rot=point_lonlat,
                xsize=int(self.xsize * self.padding),
                reso=resolution_arcmin,
                return_projected_map=True,
                nest=False,
                no_plot=True,
            )
            
            # Process patch (rotate and extract center)
            patch_processed = FibonacciGrid.get_patch_pixels(patch_projected, self.xsize)
            
            existing_shm.close()
            return patch_processed.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error extracting patch at {point_lonlat}: {e}")
            return None
    
    def _extract_patches_sequential(
        self, 
        map_data: np.ndarray, 
        points_lonlat: List[Tuple[float, float]], 
        resolution_arcmin: float
    ) -> np.ndarray:
        """Extract patches sequentially (fallback)."""
        try:
            import healpy as hp
            from lensing_ssc.core.fibonacci_utils import FibonacciGrid
            
            patches = []
            
            for i, point in enumerate(points_lonlat):
                if i % 50 == 0:
                    self.logger.info(f"Processing patch {i}/{len(points_lonlat)}")
                
                try:
                    patch_projected = hp.gnomview(
                        map_data,
                        rot=point,
                        xsize=int(self.xsize * self.padding),
                        reso=resolution_arcmin,
                        return_projected_map=True,
                        nest=False,
                        no_plot=True,
                    )
                    
                    patch_processed = FibonacciGrid.get_patch_pixels(patch_projected, self.xsize)
                    patches.append(patch_processed.astype(np.float32))
                    
                except Exception as e:
                    self.logger.error(f"Error extracting patch {i} at {point}: {e}")
                    continue
            
            return np.array(patches) if patches else np.array([])
            
        except ImportError:
            raise ProcessingError("healpy and fibonacci_utils are required for patch extraction")
    
    def _save_patches(self, patches: np.ndarray, context: PipelineContext, patch_size_deg: float) -> Path:
        """Save patches to file."""
        if not self.output_dir:
            self.output_dir = self._get_output_dir_from_context(context)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"patches_oa{int(patch_size_deg)}_x{self.xsize}_{timestamp}.npy"
        output_file = self.output_dir / filename
        
        # Check if file exists
        if output_file.exists() and not self.overwrite:
            counter = 1
            while output_file.exists():
                filename = f"patches_oa{int(patch_size_deg)}_x{self.xsize}_{timestamp}_{counter:03d}.npy"
                output_file = self.output_dir / filename
                counter += 1
        
        np.save(output_file, patches)
        self.logger.info(f"Saved {len(patches)} patches to {output_file}")
        
        return output_file
    
    def _get_output_dir_from_context(self, context: PipelineContext) -> Path:
        """Get output directory from context."""
        config = context.config
        
        # Try various config attributes
        for attr in ['patch_output_dir', 'output_dir', 'patches_dir']:
            if hasattr(config, attr):
                path = getattr(config, attr)
                if path:
                    return Path(path)
        
        # Default to context temp directory
        return context.temp_dir / "patches"


class PatchValidationStep(ProcessingStep):
    """Validate extracted patches for quality and consistency."""
    
    def __init__(
        self,
        name: str,
        min_patches: int = 1,
        max_nan_fraction: float = 0.1,
        check_statistics: bool = True,
        strict_mode: bool = False,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.min_patches = min_patches
        self.max_nan_fraction = max_nan_fraction
        self.check_statistics = check_statistics
        self.strict_mode = strict_mode
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute patch validation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Get patch extraction result
            extraction_result = self._get_extraction_result(inputs)
            if not extraction_result:
                raise ProcessingError("Patch extraction step result not found or failed")
            
            patches = extraction_result.data['patches']
            
            if patches.size == 0:
                result.status = StepStatus.SKIPPED
                result.warnings.append("No patches to validate")
                return result
            
            # Perform validation
            validation_results = self._validate_patches(patches)
            
            # Check if validation passed
            validation_passed = all([
                validation_results['basic_checks']['passed'],
                validation_results['shape_checks']['passed'],
                validation_results['data_quality']['passed']
            ])
            
            if self.check_statistics:
                validation_passed = validation_passed and validation_results['statistics']['passed']
            
            result.data = {
                'validation_results': validation_results,
                'validation_passed': validation_passed,
                'patches_shape': patches.shape,
                'n_patches': len(patches)
            }
            
            result.metadata = {
                'validation_passed': validation_passed,
                'n_patches_validated': len(patches),
                'patches_shape': patches.shape,
                'strict_mode': self.strict_mode,
                'checks_performed': list(validation_results.keys())
            }
            
            if not validation_passed:
                if self.strict_mode:
                    raise ValidationError("Patch validation failed in strict mode")
                else:
                    result.warnings.append("Some patch validation checks failed")
            
            self.logger.info(f"Patch validation {'passed' if validation_passed else 'failed'}")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _get_extraction_result(self, inputs: Dict[str, StepResult]) -> Optional[StepResult]:
        """Get patch extraction result from inputs."""
        for step_result in inputs.values():
            if (step_result.is_successful() and 'patches' in step_result.data):
                return step_result
        return None
    
    def _validate_patches(self, patches: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive patch validation."""
        validation_results = {}
        
        # Basic checks
        validation_results['basic_checks'] = self._check_basic_properties(patches)
        
        # Shape checks
        validation_results['shape_checks'] = self._check_patch_shapes(patches)
        
        # Data quality checks
        validation_results['data_quality'] = self._check_data_quality(patches)
        
        # Statistical checks
        if self.check_statistics:
            validation_results['statistics'] = self._check_patch_statistics(patches)
        
        return validation_results
    
    def _check_basic_properties(self, patches: np.ndarray) -> Dict[str, Any]:
        """Check basic patch properties."""
        issues = []
        
        # Check minimum number of patches
        if len(patches) < self.min_patches:
            issues.append(f"Too few patches: {len(patches)}, minimum: {self.min_patches}")
        
        # Check if patches exist
        if patches.size == 0:
            issues.append("Patches array is empty")
        
        # Check dimensions
        if patches.ndim != 3:
            issues.append(f"Expected 3D patch array, got {patches.ndim}D")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'n_patches': len(patches) if patches.size > 0 else 0,
            'array_shape': patches.shape
        }
    
    def _check_patch_shapes(self, patches: np.ndarray) -> Dict[str, Any]:
        """Check patch shape consistency."""
        issues = []
        
        if patches.size == 0:
            return {'passed': False, 'issues': ['No patches to check'], 'consistent_shapes': False}
        
        if patches.ndim == 3:
            n_patches, height, width = patches.shape
            
            # Check for square patches
            if height != width:
                issues.append(f"Non-square patches detected: {height}x{width}")
            
            # Check for reasonable patch sizes
            if height < 16:
                issues.append(f"Patches too small: {height}x{width}")
            elif height > 4096:
                issues.append(f"Patches too large: {height}x{width}")
            
            # Check for consistent shapes across all patches
            shapes_consistent = True
            expected_shape = (height, width)
            
        else:
            issues.append(f"Cannot validate shapes for {patches.ndim}D array")
            shapes_consistent = False
            expected_shape = None
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'consistent_shapes': shapes_consistent,
            'expected_shape': expected_shape
        }
    
    def _check_data_quality(self, patches: np.ndarray) -> Dict[str, Any]:
        """Check patch data quality."""
        issues = []
        warnings = []
        
        if patches.size == 0:
            return {'passed': False, 'issues': ['No patches to check'], 'warnings': []}
        
        # Check for finite values
        total_elements = patches.size
        finite_mask = np.isfinite(patches)
        n_finite = np.sum(finite_mask)
        n_nan = np.sum(np.isnan(patches))
        n_inf = np.sum(np.isinf(patches))
        
        nan_fraction = n_nan / total_elements
        inf_fraction = n_inf / total_elements
        
        if nan_fraction > self.max_nan_fraction:
            issues.append(f"Too many NaN values: {nan_fraction:.3f} > {self.max_nan_fraction}")
        elif nan_fraction > 0:
            warnings.append(f"Found {nan_fraction:.3f} fraction of NaN values")
        
        if inf_fraction > 0:
            issues.append(f"Found infinite values: {inf_fraction:.3f} fraction")
        
        # Check value ranges
        if n_finite > 0:
            finite_data = patches[finite_mask]
            data_min, data_max = np.min(finite_data), np.max(finite_data)
            data_std = np.std(finite_data)
            
            # Check for extremely large values (might indicate processing errors)
            if abs(data_max) > 100 or abs(data_min) > 100:
                warnings.append(f"Large values detected: range [{data_min:.3f}, {data_max:.3f}]")
            
            # Check for zero variance (might indicate empty or constant patches)
            if data_std < 1e-10:
                issues.append(f"Very low variance: {data_std:.3e}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'nan_fraction': nan_fraction,
            'inf_fraction': inf_fraction,
            'finite_fraction': n_finite / total_elements
        }
    
    def _check_patch_statistics(self, patches: np.ndarray) -> Dict[str, Any]:
        """Check statistical properties of patches."""
        issues = []
        warnings = []
        
        if patches.size == 0:
            return {'passed': False, 'issues': ['No patches to check'], 'warnings': []}
        
        # Calculate statistics per patch
        patch_means = []
        patch_stds = []
        
        for i in range(len(patches)):
            patch = patches[i]
            finite_mask = np.isfinite(patch)
            
            if np.sum(finite_mask) > 0:
                patch_finite = patch[finite_mask]
                patch_means.append(np.mean(patch_finite))
                patch_stds.append(np.std(patch_finite))
            else:
                patch_means.append(np.nan)
                patch_stds.append(np.nan)
        
        patch_means = np.array(patch_means)
        patch_stds = np.array(patch_stds)
        
        # Remove NaN values for statistics
        valid_means = patch_means[np.isfinite(patch_means)]
        valid_stds = patch_stds[np.isfinite(patch_stds)]
        
        if len(valid_means) == 0:
            issues.append("No patches with finite statistics")
            return {'passed': False, 'issues': issues, 'warnings': warnings}
        
        # Check mean statistics
        overall_mean = np.mean(valid_means)
        mean_std = np.std(valid_means)
        
        # For kappa maps, expect mean close to zero
        if abs(overall_mean) > 0.1:
            warnings.append(f"Large overall mean: {overall_mean:.6f}")
        
        # Check standard deviation statistics
        std_mean = np.mean(valid_stds)
        std_std = np.std(valid_stds)
        
        if std_mean < 1e-6:
            issues.append(f"Very small patch standard deviations: {std_mean:.6e}")
        
        # Check for outlier patches
        mean_outliers = np.sum(np.abs(valid_means - overall_mean) > 5 * mean_std)
        std_outliers = np.sum(np.abs(valid_stds - std_mean) > 5 * std_std)
        
        outlier_fraction = (mean_outliers + std_outliers) / (2 * len(valid_means))
        
        if outlier_fraction > 0.1:
            warnings.append(f"High fraction of outlier patches: {outlier_fraction:.3f}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'statistics': {
                'overall_mean': overall_mean,
                'mean_std': mean_std,
                'std_mean': std_mean,
                'std_std': std_std,
                'n_valid_patches': len(valid_means),
                'outlier_fraction': outlier_fraction
            }
        }


__all__ = [
    'FibonacciGridStep',
    'PatchExtractionStep',
    'PatchValidationStep',
]