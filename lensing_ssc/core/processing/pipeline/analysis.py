# lensing_ssc/processing/pipeline/analysis.py
"""
Analysis pipeline for statistical analysis of kappa maps and patches.

This module implements the complete statistical analysis workflow that processes
kappa maps to extract patches and compute various statistical measures including
power spectra, bispectra, PDFs, and peak counts. The pipeline handles the
workflow from kappa map loading through final statistical output generation.

Pipeline Steps:
1. Input Discovery: Find and validate kappa maps and existing patches
2. Patch Generation: Extract patches from full-sky kappa maps if needed
3. Statistical Analysis: Compute power spectra, bispectra, PDFs, peak counts
4. Output Generation: Save results to HDF5 files
5. Validation: Verify output completeness and quality

The pipeline supports:
- Multiple source redshifts and noise levels
- Parallel processing for statistical computations
- Resume from checkpoints for long-running jobs
- Comprehensive validation and error recovery
- Memory-efficient processing of large datasets

Usage:
    from lensing_ssc.processing.pipeline import AnalysisPipeline
    from lensing_ssc.config import AnalysisConfig
    
    config = AnalysisConfig(
        kappa_input_dir="/path/to/kappa/maps",
        patch_output_dir="/path/to/patches",
        stats_output_dir="/path/to/stats",
        patch_size_deg=10.0,
        ngal_list=[0, 7, 15, 30, 50],
        sl_list=[2.0, 5.0, 8.0, 10.0]
    )
    
    pipeline = AnalysisPipeline(config)
    results = pipeline.run(
        resume_from_checkpoint=True,
        num_processes=8,
        memory_limit_mb=16000
    )

Advanced Usage:
    # Custom analysis parameters
    config.lmin = 300
    config.lmax = 3000
    config.nbin_ps_bs = 8
    config.epsilon_noise = 0.26
    
    pipeline = AnalysisPipeline(config)
    
    # Add custom callback for progress monitoring
    def on_patch_analyzed(step_name, result):
        if 'patch_count' in result.metadata:
            print(f"Analyzed {result.metadata['patch_count']} patches")
    
    pipeline.on_step_complete = on_patch_analyzed
    results = pipeline.run()
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import json
import re
import time
import multiprocessing as mp
import numpy as np

from . import BasePipeline, ProcessingStep, StepResult, PipelineContext, StepStatus
from lensing_ssc.core.base import ValidationError, ProcessingError, ConfigurationError


logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for analysis pipeline.
    
    This standardizes configuration access for the analysis workflow.
    """
    kappa_input_dir: Path
    patch_output_dir: Path
    stats_output_dir: Path
    
    # Patch parameters
    patch_size_deg: float = 10.0
    patch_xsize: int = 256
    center_points_path: Optional[Path] = None
    
    # Analysis parameters
    ngal_list: List[int] = field(default_factory=lambda: [0, 7, 15, 30, 50])
    sl_list: List[float] = field(default_factory=lambda: [2.0, 5.0, 8.0, 10.0])
    lmin: int = 300
    lmax: int = 3000
    nbin_ps_bs: int = 8
    nbin_pdf_peaks: int = 50
    pdf_peaks_range: Tuple[float, float] = (-5.0, 5.0)
    epsilon_noise: float = 0.26
    
    # Processing control
    overwrite_patches: bool = False
    overwrite_stats: bool = False
    num_processes: Optional[int] = None
    
    @classmethod
    def from_config(cls, config: Any) -> 'AnalysisConfig':
        """Create from various config object types."""
        if isinstance(config, cls):
            return config
        
        # Extract required paths
        kappa_input_dir = getattr(config, 'kappa_input_dir', None)
        patch_output_dir = getattr(config, 'patch_output_dir', None)
        stats_output_dir = getattr(config, 'stats_output_dir', None)
        
        if not all([kappa_input_dir, patch_output_dir, stats_output_dir]):
            raise ConfigurationError(
                "kappa_input_dir, patch_output_dir, and stats_output_dir must be specified"
            )
        
        # Handle JSON strings in arguments
        ngal_list = getattr(config, 'ngal_list', [0, 7, 15, 30, 50])
        sl_list = getattr(config, 'sl_list', [2.0, 5.0, 8.0, 10.0])
        
        # Parse JSON strings if needed
        if isinstance(ngal_list, str):
            ngal_list = json.loads(ngal_list)
        if isinstance(sl_list, str):
            sl_list = json.loads(sl_list)
        
        pdf_peaks_range = getattr(config, 'pdf_peaks_min_max', [-5.0, 5.0])
        if isinstance(pdf_peaks_range, str):
            pdf_peaks_range = tuple(json.loads(pdf_peaks_range))
        
        return cls(
            kappa_input_dir=Path(kappa_input_dir),
            patch_output_dir=Path(patch_output_dir),
            stats_output_dir=Path(stats_output_dir),
            patch_size_deg=getattr(config, 'patch_size_deg', 10.0),
            patch_xsize=getattr(config, 'patch_xsize', 256),
            center_points_path=Path(getattr(config, 'center_points_path', '')) if hasattr(config, 'center_points_path') else None,
            ngal_list=ngal_list,
            sl_list=sl_list,
            lmin=getattr(config, 'lmin', 300),
            lmax=getattr(config, 'lmax', 3000),
            nbin_ps_bs=getattr(config, 'nbin_ps_bs', 8),
            nbin_pdf_peaks=getattr(config, 'nbin_pdf_peaks', 50),
            pdf_peaks_range=tuple(pdf_peaks_range),
            epsilon_noise=getattr(config, 'epsilon_noise', 0.26),
            overwrite_patches=getattr(config, 'overwrite_patches', False),
            overwrite_stats=getattr(config, 'overwrite_stats', False),
            num_processes=getattr(config, 'num_processes', None),
        )


def parse_kappa_filename(filename: str) -> Dict[str, Any]:
    """Parse kappa filename to extract metadata."""
    match = re.match(r"kappa_zs(\d+\.?\d*)_s(\w+)_nside(\d+).fits", Path(filename).name)
    if match:
        return {
            "zs": float(match.group(1)),
            "seed": str(match.group(2)),
            "nside": int(match.group(3)),
        }
    return {}


class InputDiscoveryStep(ProcessingStep):
    """Discover and validate input kappa maps and existing patches."""
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute input discovery."""
        config = AnalysisConfig.from_config(context.config)
        
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Discover kappa maps
            kappa_files = self._discover_kappa_maps(config.kappa_input_dir)
            
            # Discover existing patches
            existing_patches = self._discover_existing_patches(config.patch_output_dir)
            
            # Discover existing statistics
            existing_stats = self._discover_existing_stats(config.stats_output_dir)
            
            result.data = {
                'kappa_files': kappa_files,
                'existing_patches': existing_patches,
                'existing_stats': existing_stats
            }
            
            result.metadata = {
                'n_kappa_files': len(kappa_files),
                'n_existing_patches': len(existing_patches),
                'n_existing_stats': len(existing_stats),
                'kappa_redshifts': list(set(info.get('zs') for info in kappa_files.values() if info.get('zs'))),
                'kappa_seeds': list(set(info.get('seed') for info in kappa_files.values() if info.get('seed')))
            }
            
            self.logger.info(f"Found {len(kappa_files)} kappa maps")
            self.logger.info(f"Found {len(existing_patches)} existing patch files")
            self.logger.info(f"Found {len(existing_stats)} existing statistics files")
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _discover_kappa_maps(self, kappa_dir: Path) -> Dict[Path, Dict[str, Any]]:
        """Discover kappa map files."""
        if not kappa_dir.exists():
            raise ValidationError(f"Kappa input directory does not exist: {kappa_dir}")
        
        kappa_files = {}
        for fits_file in kappa_dir.glob("kappa_*.fits"):
            file_info = parse_kappa_filename(fits_file.name)
            if file_info:
                kappa_files[fits_file] = file_info
            else:
                self.logger.warning(f"Could not parse kappa filename: {fits_file.name}")
        
        return kappa_files
    
    def _discover_existing_patches(self, patch_dir: Path) -> Dict[Path, Dict[str, Any]]:
        """Discover existing patch files."""
        existing_patches = {}
        if patch_dir.exists():
            for patch_file in patch_dir.glob("*_patches_*.npy"):
                # Extract info from filename
                info = {}
                if "_oa" in patch_file.stem:
                    # Extract patch size
                    match = re.search(r"_oa(\d+)", patch_file.stem)
                    if match:
                        info['patch_size'] = int(match.group(1))
                
                if "_x" in patch_file.stem:
                    # Extract xsize
                    match = re.search(r"_x(\d+)", patch_file.stem)
                    if match:
                        info['xsize'] = int(match.group(1))
                
                existing_patches[patch_file] = info
        
        return existing_patches
    
    def _discover_existing_stats(self, stats_dir: Path) -> Dict[Path, Dict[str, Any]]:
        """Discover existing statistics files."""
        existing_stats = {}
        if stats_dir.exists():
            for stats_file in stats_dir.glob("*_stats_*.hdf5"):
                existing_stats[stats_file] = {}
        
        return existing_stats


class PatchGenerationStep(ProcessingStep):
    """Generate patches from kappa maps if needed."""
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute patch generation."""
        config = AnalysisConfig.from_config(context.config)
        
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Get input discovery results
            discovery_result = inputs.get('input_discovery')
            if not discovery_result or not discovery_result.is_successful():
                raise ProcessingError("Input discovery step failed or missing")
            
            kappa_files = discovery_result.data['kappa_files']
            existing_patches = discovery_result.data['existing_patches']
            
            # Create patch processor
            patch_processor = self._create_patch_processor(config)
            
            # Generate patches for each kappa map
            generated_patches = {}
            skipped_patches = {}
            
            for kappa_file, kappa_info in kappa_files.items():
                patch_file_name = self._get_patch_filename(kappa_file, config)
                patch_file_path = config.patch_output_dir / patch_file_name
                
                # Check if patches already exist
                if patch_file_path.exists() and not config.overwrite_patches:
                    skipped_patches[patch_file_path] = kappa_info
                    self.logger.info(f"Skipping existing patches: {patch_file_name}")
                    continue
                
                try:
                    # Generate patches
                    self.logger.info(f"Generating patches for {kappa_file.name}")
                    patches_data = patch_processor.generate_patches(kappa_file, config.num_processes)
                    
                    # Save patches
                    config.patch_output_dir.mkdir(parents=True, exist_ok=True)
                    np.save(patch_file_path, patches_data)
                    
                    generated_patches[patch_file_path] = {
                        'kappa_file': kappa_file,
                        'kappa_info': kappa_info,
                        'n_patches': len(patches_data),
                        'patch_shape': patches_data.shape if len(patches_data) > 0 else None
                    }
                    
                    self.logger.info(f"Generated {len(patches_data)} patches -> {patch_file_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate patches for {kappa_file.name}: {e}")
                    continue
            
            result.data = {
                'generated_patches': generated_patches,
                'skipped_patches': skipped_patches,
                'patch_processor': patch_processor
            }
            
            result.metadata = {
                'n_generated': len(generated_patches),
                'n_skipped': len(skipped_patches),
                'total_patch_files': len(generated_patches) + len(skipped_patches)
            }
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _create_patch_processor(self, config: AnalysisConfig):
        """Create patch processor instance."""
        try:
            # Import here to avoid circular imports
            from lensing_ssc.core.patching_utils import PatchProcessor
            
            center_points_path = config.center_points_path or "lensing_ssc/core/fibonacci/center_points/"
            
            return PatchProcessor(
                patch_size_deg=config.patch_size_deg,
                xsize=config.patch_xsize,
                center_points_path=str(center_points_path)
            )
        except ImportError:
            raise ProcessingError("PatchProcessor not available - check patching_utils import")
    
    def _get_patch_filename(self, kappa_file: Path, config: AnalysisConfig) -> str:
        """Generate patch filename from kappa filename."""
        base_name = kappa_file.stem  # Remove .fits extension
        return f"{base_name}_patches_oa{int(config.patch_size_deg)}_x{config.patch_xsize}.npy"


class StatisticalAnalysisStep(ProcessingStep):
    """Perform statistical analysis on patches."""
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute statistical analysis."""
        config = AnalysisConfig.from_config(context.config)
        
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Get inputs
            discovery_result = inputs.get('input_discovery')
            patch_result = inputs.get('patch_generation')
            
            if not all([discovery_result, patch_result]):
                raise ProcessingError("Missing required input steps")
            
            # Prepare analysis parameters
            analysis_params = self._prepare_analysis_params(config)
            
            # Get all patch files to analyze
            patch_files = self._get_patch_files_to_analyze(
                patch_result.data['generated_patches'],
                patch_result.data['skipped_patches'],
                config
            )
            
            # Analyze each patch file
            analysis_results = {}
            
            for patch_file, patch_info in patch_files.items():
                try:
                    stats_file = self._get_stats_filename(patch_file, config)
                    
                    # Check if stats already exist
                    if stats_file.exists() and not config.overwrite_stats:
                        self.logger.info(f"Skipping existing stats: {stats_file.name}")
                        continue
                    
                    # Load patches
                    self.logger.info(f"Analyzing patches from {patch_file.name}")
                    patches_data = np.load(patch_file)
                    
                    if len(patches_data) == 0:
                        self.logger.warning(f"No patches found in {patch_file.name}")
                        continue
                    
                    # Perform analysis
                    stats_result = self._analyze_patches(
                        patches_data, analysis_params, config
                    )
                    
                    # Save results
                    self._save_analysis_results(stats_result, stats_file, analysis_params, patch_info)
                    
                    analysis_results[stats_file] = {
                        'patch_file': patch_file,
                        'patch_info': patch_info,
                        'n_patches': len(patches_data),
                        'stats_computed': True
                    }
                    
                    self.logger.info(f"Analysis completed -> {stats_file.name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to analyze {patch_file.name}: {e}")
                    continue
            
            result.data = {
                'analysis_results': analysis_results,
                'analysis_params': analysis_params
            }
            
            result.metadata = {
                'n_analyzed': len(analysis_results),
                'n_patch_files': len(patch_files),
                'analysis_params_summary': self._get_params_summary(analysis_params)
            }
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _prepare_analysis_params(self, config: AnalysisConfig) -> Dict[str, Any]:
        """Prepare analysis parameters."""
        # Prepare bins for power spectrum and bispectrum
        ps_bs_l_edges = np.logspace(np.log10(config.lmin), np.log10(config.lmax), config.nbin_ps_bs + 1)
        ps_bs_ell_mids = (ps_bs_l_edges[:-1] + ps_bs_l_edges[1:]) / 2
        
        # Prepare bins for PDF and peak counts
        pdf_peaks_nu_bins = np.linspace(
            config.pdf_peaks_range[0], 
            config.pdf_peaks_range[1], 
            config.nbin_pdf_peaks + 1
        )
        
        return {
            'patch_size_deg': config.patch_size_deg,
            'sl_list': config.sl_list,
            'ngal_list': config.ngal_list,
            'ps_bs_l_edges': ps_bs_l_edges,
            'ps_bs_ell_mids': ps_bs_ell_mids,
            'pdf_peaks_nu_bins': pdf_peaks_nu_bins,
            'epsilon_noise': config.epsilon_noise,
            'xsize': config.patch_xsize,
            'num_processes': config.num_processes or mp.cpu_count()
        }
    
    def _get_patch_files_to_analyze(self, generated_patches, skipped_patches, config) -> Dict[Path, Dict]:
        """Get all patch files that need analysis."""
        patch_files = {}
        
        # Add generated patches
        for patch_file, patch_info in generated_patches.items():
            patch_files[patch_file] = patch_info
        
        # Add skipped patches (already existing)
        for patch_file, patch_info in skipped_patches.items():
            patch_files[patch_file] = {'kappa_info': patch_info}
        
        # Also check patch output directory for any other patch files
        if config.patch_output_dir.exists():
            for patch_file in config.patch_output_dir.glob("*_patches_*.npy"):
                if patch_file not in patch_files:
                    patch_files[patch_file] = {}
        
        return patch_files
    
    def _get_stats_filename(self, patch_file: Path, config: AnalysisConfig) -> Path:
        """Generate statistics filename from patch filename."""
        # Replace _patches_ with _stats_ and .npy with .hdf5
        stats_name = patch_file.stem.replace('_patches_', '_stats_') + '.hdf5'
        return config.stats_output_dir / stats_name
    
    def _analyze_patches(
        self, 
        patches_data: np.ndarray, 
        analysis_params: Dict[str, Any],
        config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Analyze patches using the statistical analysis worker."""
        try:
            # Import analysis functions
            from lensing_ssc.stats.power_spectrum import calculate_power_spectrum
            from lensing_ssc.stats.bispectrum import calculate_bispectrum
            from lensing_ssc.stats.pdf import calculate_pdf
            from lensing_ssc.stats.peak_counts import calculate_peak_counts
            from lenstools import ConvergenceMap
            from astropy import units as u
        except ImportError:
            raise ProcessingError("Required analysis modules not available")
        
        # Initialize results structure
        aggregated_results = {
            ng: {
                "cl": [], "bispec_equ": [], "bispec_iso": [], "bispec_sq": [],
                **{sl: {"pdf": [], "peaks": [], "minima": [], "sigma0": []} for sl in analysis_params['sl_list']}
            } for ng in analysis_params['ngal_list']
        }
        
        # Process each patch
        pixarea_arcmin2 = (analysis_params['patch_size_deg'] * 60.0 / analysis_params['xsize']) ** 2
        
        for i_patch, patch_data in enumerate(patches_data):
            if i_patch % 50 == 0:
                self.logger.info(f"Processing patch {i_patch}/{len(patches_data)}")
            
            for ngal in analysis_params['ngal_list']:
                # Add noise if needed
                current_patch_data = patch_data.copy()
                if ngal > 0:
                    noise_sigma = analysis_params['epsilon_noise'] / np.sqrt(ngal * pixarea_arcmin2)
                    noise = np.random.normal(0, noise_sigma, current_patch_data.shape)
                    current_patch_data += noise
                
                conv_map = ConvergenceMap(current_patch_data, angle=analysis_params['patch_size_deg'] * u.deg)
                
                # Power Spectrum & Bispectrum (on un-smoothed, potentially noisy map)
                cl = calculate_power_spectrum(conv_map, analysis_params['ps_bs_l_edges'], analysis_params['ps_bs_ell_mids'])
                equ, iso, sq = calculate_bispectrum(conv_map, analysis_params['ps_bs_l_edges'], analysis_params['ps_bs_ell_mids'])
                
                aggregated_results[ngal]["cl"].append(cl)
                aggregated_results[ngal]["bispec_equ"].append(equ)
                aggregated_results[ngal]["bispec_iso"].append(iso)
                aggregated_results[ngal]["bispec_sq"].append(sq)
                
                # Smoothed statistics
                for sl in analysis_params['sl_list']:
                    smoothed_map_data = conv_map.smooth(sl * u.arcmin).data
                    sigma0 = np.std(smoothed_map_data)
                    
                    # SNR map for PDF and Peaks
                    if sigma0 > 1e-9:
                        snr_map_data = smoothed_map_data / sigma0
                    else:
                        snr_map_data = smoothed_map_data
                    
                    snr_conv_map = ConvergenceMap(snr_map_data, angle=analysis_params['patch_size_deg'] * u.deg)
                    
                    pdf = calculate_pdf(snr_conv_map, analysis_params['pdf_peaks_nu_bins'])
                    peaks = calculate_peak_counts(snr_conv_map, analysis_params['pdf_peaks_nu_bins'], is_minima=False)
                    minima = calculate_peak_counts(snr_conv_map, analysis_params['pdf_peaks_nu_bins'], is_minima=True)
                    
                    aggregated_results[ngal][sl]["pdf"].append(pdf)
                    aggregated_results[ngal][sl]["peaks"].append(peaks)
                    aggregated_results[ngal][sl]["minima"].append(minima)
                    aggregated_results[ngal][sl]["sigma0"].append(sigma0)
        
        # Convert lists to arrays
        final_results = {}
        for ngal in analysis_params['ngal_list']:
            final_results[ngal] = {}
            final_results[ngal]["cl"] = np.array(aggregated_results[ngal]["cl"])
            final_results[ngal]["bispec_equ"] = np.array(aggregated_results[ngal]["bispec_equ"])
            final_results[ngal]["bispec_iso"] = np.array(aggregated_results[ngal]["bispec_iso"])
            final_results[ngal]["bispec_sq"] = np.array(aggregated_results[ngal]["bispec_sq"])
            
            for sl in analysis_params['sl_list']:
                final_results[ngal][sl] = {
                    "pdf": np.array(aggregated_results[ngal][sl]["pdf"]),
                    "peaks": np.array(aggregated_results[ngal][sl]["peaks"]),
                    "minima": np.array(aggregated_results[ngal][sl]["minima"]),
                    "sigma0": np.array(aggregated_results[ngal][sl]["sigma0"]),
                }
        
        return final_results
    
    def _save_analysis_results(
        self, 
        stats_result: Dict[str, Any], 
        stats_file: Path,
        analysis_params: Dict[str, Any],
        patch_info: Dict[str, Any]
    ) -> None:
        """Save analysis results to HDF5 file."""
        try:
            from lensing_ssc.io.file_handlers import save_results_to_hdf5
            import argparse
            
            # Create metadata object for save function
            metadata = argparse.Namespace(
                patch_size=analysis_params['patch_size_deg'],
                xsize=analysis_params['xsize'],
                pixarea_arcmin2=(analysis_params['patch_size_deg'] * 60.0 / analysis_params['xsize'])**2,
                lmin=np.min(analysis_params['ps_bs_l_edges']),
                lmax=np.max(analysis_params['ps_bs_l_edges']),
                nbin=len(analysis_params['ps_bs_ell_mids']),
                epsilon=analysis_params['epsilon_noise'],
                ngal_list=analysis_params['ngal_list'],
                sl_list=analysis_params['sl_list'],
                bins=analysis_params['pdf_peaks_nu_bins'],
                nu=(analysis_params['pdf_peaks_nu_bins'][:-1] + analysis_params['pdf_peaks_nu_bins'][1:]) / 2,
                l_edges=analysis_params['ps_bs_l_edges'],
                ell=analysis_params['ps_bs_ell_mids'],
                kappa_file_info=patch_info.get('kappa_info', {})
            )
            
            # Ensure output directory exists
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save results
            save_results_to_hdf5(stats_result, stats_file, analyzer=metadata)
            
        except ImportError:
            raise ProcessingError("HDF5 save functionality not available")
    
    def _get_params_summary(self, analysis_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of analysis parameters."""
        return {
            'patch_size_deg': analysis_params['patch_size_deg'],
            'n_ngal_values': len(analysis_params['ngal_list']),
            'n_sl_values': len(analysis_params['sl_list']),
            'n_ell_bins': len(analysis_params['ps_bs_ell_mids']),
            'n_nu_bins': len(analysis_params['pdf_peaks_nu_bins']) - 1,
            'ell_range': (np.min(analysis_params['ps_bs_ell_mids']), np.max(analysis_params['ps_bs_ell_mids'])),
            'nu_range': (np.min(analysis_params['pdf_peaks_nu_bins']), np.max(analysis_params['pdf_peaks_nu_bins']))
        }


class ValidationStep(ProcessingStep):
    """Validate analysis outputs."""
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute validation."""
        config = AnalysisConfig.from_config(context.config)
        
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Get analysis results
            analysis_result = inputs.get('statistical_analysis')
            if not analysis_result or not analysis_result.is_successful():
                raise ProcessingError("Statistical analysis step failed or missing")
            
            analysis_results = analysis_result.data['analysis_results']
            
            # Validate each output file
            validation_results = self._validate_output_files(analysis_results, config)
            
            result.data = validation_results
            result.metadata = {
                'n_validated_files': validation_results['n_valid_files'],
                'n_invalid_files': validation_results['n_invalid_files'],
                'validation_passed': validation_results['n_invalid_files'] == 0
            }
            
            if validation_results['n_invalid_files'] > 0:
                self.logger.warning(f"Validation found {validation_results['n_invalid_files']} invalid files")
            else:
                self.logger.info("All output files passed validation")
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _validate_output_files(self, analysis_results: Dict, config: AnalysisConfig) -> Dict[str, Any]:
        """Validate analysis output files."""
        try:
            import h5py
        except ImportError:
            raise ProcessingError("h5py is required for HDF5 validation")
        
        valid_files = []
        invalid_files = []
        
        for stats_file, file_info in analysis_results.items():
            try:
                if not stats_file.exists():
                    invalid_files.append({'file': stats_file, 'error': 'File does not exist'})
                    continue
                
                # Try to open and validate HDF5 structure
                with h5py.File(stats_file, 'r') as f:
                    # Check for required groups/datasets
                    required_groups = []
                    for ngal in config.ngal_list:
                        required_groups.append(str(ngal))
                    
                    missing_groups = []
                    for group in required_groups:
                        if group not in f:
                            missing_groups.append(group)
                    
                    if missing_groups:
                        invalid_files.append({
                            'file': stats_file, 
                            'error': f'Missing groups: {missing_groups}'
                        })
                        continue
                    
                    # Validate data shapes and types
                    validation_errors = []
                    for ngal in config.ngal_list:
                        ngal_group = f[str(ngal)]
                        
                        # Check for required datasets
                        required_datasets = ['cl', 'bispec_equ', 'bispec_iso', 'bispec_sq']
                        for dataset in required_datasets:
                            if dataset not in ngal_group:
                                validation_errors.append(f'Missing dataset: {ngal}/{dataset}')
                        
                        # Check smoothing length groups
                        for sl in config.sl_list:
                            sl_key = str(sl)
                            if sl_key in ngal_group:
                                sl_group = ngal_group[sl_key]
                                sl_datasets = ['pdf', 'peaks', 'minima', 'sigma0']
                                for dataset in sl_datasets:
                                    if dataset not in sl_group:
                                        validation_errors.append(f'Missing dataset: {ngal}/{sl_key}/{dataset}')
                    
                    if validation_errors:
                        invalid_files.append({
                            'file': stats_file,
                            'error': f'Validation errors: {validation_errors}'
                        })
                        continue
                
                # File passed all validation
                valid_files.append({
                    'file': stats_file,
                    'size_mb': stats_file.stat().st_size / (1024**2),
                    'info': file_info
                })
                
            except Exception as e:
                invalid_files.append({'file': stats_file, 'error': str(e)})
        
        return {
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'n_valid_files': len(valid_files),
            'n_invalid_files': len(invalid_files)
        }


class ReportGenerationStep(ProcessingStep):
    """Generate summary report of analysis results."""
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute report generation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            metadata={}
        )
        
        try:
            # Gather information from all previous steps
            report_data = self._gather_report_data(inputs)
            
            # Generate report
            report_content = self._generate_report_content(report_data)
            
            # Save report
            config = AnalysisConfig.from_config(context.config)
            report_file = config.stats_output_dir / "analysis_report.txt"
            
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            result.data = {
                'report_file': report_file,
                'report_content': report_content,
                'report_data': report_data
            }
            
            result.metadata = {
                'report_generated': True,
                'report_file': str(report_file)
            }
            
            result.status = StepStatus.COMPLETED
            return result
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            return result
    
    def _gather_report_data(self, inputs: Dict[str, StepResult]) -> Dict[str, Any]:
        """Gather data for report from all pipeline steps."""
        report_data = {}
        
        for step_name, step_result in inputs.items():
            if step_result.is_successful():
                report_data[step_name] = {
                    'metadata': step_result.metadata,
                    'execution_time': step_result.execution_time,
                    'warnings': step_result.warnings
                }
        
        return report_data
    
    def _generate_report_content(self, report_data: Dict[str, Any]) -> str:
        """Generate report content."""
        from datetime import datetime
        
        report_lines = [
            "=" * 60,
            "LENSING SSC ANALYSIS PIPELINE REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # Summary section
        report_lines.extend([
            "PIPELINE SUMMARY",
            "-" * 20,
        ])
        
        for step_name, step_data in report_data.items():
            metadata = step_data['metadata']
            execution_time = step_data['execution_time']
            
            report_lines.append(f"{step_name.upper()}:")
            report_lines.append(f"  Execution time: {execution_time:.2f}s")
            
            if step_name == 'input_discovery':
                report_lines.extend([
                    f"  Kappa files found: {metadata.get('n_kappa_files', 0)}",
                    f"  Existing patches: {metadata.get('n_existing_patches', 0)}",
                    f"  Existing stats: {metadata.get('n_existing_stats', 0)}",
                    f"  Redshifts: {metadata.get('kappa_redshifts', [])}",
                ])
            
            elif step_name == 'patch_generation':
                report_lines.extend([
                    f"  Patches generated: {metadata.get('n_generated', 0)}",
                    f"  Patches skipped: {metadata.get('n_skipped', 0)}",
                    f"  Total patch files: {metadata.get('total_patch_files', 0)}",
                ])
            
            elif step_name == 'statistical_analysis':
                report_lines.extend([
                    f"  Files analyzed: {metadata.get('n_analyzed', 0)}",
                    f"  Total patch files: {metadata.get('n_patch_files', 0)}",
                ])
                
                params_summary = metadata.get('analysis_params_summary', {})
                if params_summary:
                    report_lines.extend([
                        f"  Patch size: {params_summary.get('patch_size_deg', 'N/A')} deg",
                        f"  Galaxy density levels: {params_summary.get('n_ngal_values', 'N/A')}",
                        f"  Smoothing lengths: {params_summary.get('n_sl_values', 'N/A')}",
                        f"  Ell bins: {params_summary.get('n_ell_bins', 'N/A')}",
                        f"  Nu bins: {params_summary.get('n_nu_bins', 'N/A')}",
                    ])
            
            elif step_name == 'validation':
                report_lines.extend([
                    f"  Valid files: {metadata.get('n_validated_files', 0)}",
                    f"  Invalid files: {metadata.get('n_invalid_files', 0)}",
                    f"  Validation passed: {metadata.get('validation_passed', False)}",
                ])
            
            # Add warnings if any
            warnings = step_data.get('warnings', [])
            if warnings:
                report_lines.append(f"  Warnings: {len(warnings)}")
                for warning in warnings[:3]:  # Show first 3 warnings
                    report_lines.append(f"    - {warning}")
                if len(warnings) > 3:
                    report_lines.append(f"    ... and {len(warnings) - 3} more")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


class AnalysisPipeline(BasePipeline):
    """Pipeline for statistical analysis of kappa maps and patches.
    
    This pipeline handles the complete workflow from kappa maps to
    statistical analysis results including patch extraction and
    computation of various statistics.
    
    Parameters
    ----------
    config : Any
        Configuration object with analysis settings
    name : str, optional
        Pipeline name
    """
    
    def __init__(self, config: Any, name: str = "AnalysisPipeline"):
        super().__init__(config, name)
        
        # Convert to standardized config
        self._analysis_config = AnalysisConfig.from_config(config)
    
    def setup(self) -> None:
        """Setup the analysis pipeline steps."""
        # Add steps in order
        self.add_step(InputDiscoveryStep("input_discovery"))
        
        self.add_step(PatchGenerationStep(
            "patch_generation",
            dependencies=["input_discovery"]
        ))
        
        self.add_step(StatisticalAnalysisStep(
            "statistical_analysis",
            dependencies=["input_discovery", "patch_generation"]
        ))
        
        self.add_step(ValidationStep(
            "validation",
            dependencies=["statistical_analysis"],
            skip_on_failure=True
        ))
        
        self.add_step(ReportGenerationStep(
            "report_generation",
            dependencies=["input_discovery", "patch_generation", "statistical_analysis"],
            skip_on_failure=True
        ))
    
    def validate_inputs(self) -> bool:
        """Validate pipeline inputs."""
        try:
            config = self._analysis_config
            
            # Check required paths
            if not config.kappa_input_dir.exists():
                self.logger.error(f"Kappa input directory does not exist: {config.kappa_input_dir}")
                return False
            
            # Check for kappa files
            kappa_files = list(config.kappa_input_dir.glob("kappa_*.fits"))
            if not kappa_files:
                self.logger.error(f"No kappa files found in {config.kappa_input_dir}")
                return False
            
            # Validate analysis parameters
            if config.patch_size_deg <= 0:
                self.logger.error(f"Invalid patch size: {config.patch_size_deg}")
                return False
            
            if config.patch_xsize <= 0:
                self.logger.error(f"Invalid patch xsize: {config.patch_xsize}")
                return False
            
            if not config.ngal_list:
                self.logger.error("ngal_list cannot be empty")
                return False
            
            if not config.sl_list:
                self.logger.error("sl_list cannot be empty")
                return False
            
            if config.lmin >= config.lmax:
                self.logger.error(f"Invalid ell range: lmin={config.lmin}, lmax={config.lmax}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def get_analysis_info(self) -> Dict[str, Any]:
        """Get information about the analysis configuration.
        
        Returns
        -------
        Dict[str, Any]
            Analysis information
        """
        config = self._analysis_config
        
        # Count kappa files
        n_kappa_files = 0
        kappa_redshifts = set()
        if config.kappa_input_dir.exists():
            for kappa_file in config.kappa_input_dir.glob("kappa_*.fits"):
                n_kappa_files += 1
                file_info = parse_kappa_filename(kappa_file.name)
                if file_info.get('zs'):
                    kappa_redshifts.add(file_info['zs'])
        
        return {
            'kappa_input_dir': str(config.kappa_input_dir),
            'patch_output_dir': str(config.patch_output_dir),
            'stats_output_dir': str(config.stats_output_dir),
            'n_kappa_files': n_kappa_files,
            'kappa_redshifts': sorted(list(kappa_redshifts)),
            'patch_size_deg': config.patch_size_deg,
            'patch_xsize': config.patch_xsize,
            'ngal_list': config.ngal_list,
            'sl_list': config.sl_list,
            'ell_range': (config.lmin, config.lmax),
            'n_ell_bins': config.nbin_ps_bs,
            'nu_range': config.pdf_peaks_range,
            'n_nu_bins': config.nbin_pdf_peaks,
            'overwrite_patches': config.overwrite_patches,
            'overwrite_stats': config.overwrite_stats,
        }
    
    def estimate_processing_time(self) -> Dict[str, float]:
        """Estimate processing time for the pipeline.
        
        Returns
        -------
        Dict[str, float]
            Time estimates in seconds
        """
        config = self._analysis_config
        
        # Count files to process
        n_kappa_files = len(list(config.kappa_input_dir.glob("kappa_*.fits"))) if config.kappa_input_dir.exists() else 0
        
        # Rough estimates based on typical processing times
        estimates = {
            'input_discovery': 30,  # seconds
            'patch_generation': n_kappa_files * 300,  # ~5 minutes per kappa file
            'statistical_analysis': n_kappa_files * len(config.ngal_list) * len(config.sl_list) * 60,  # ~1 min per config
            'validation': n_kappa_files * 10,  # seconds
            'report_generation': 10,  # seconds
        }
        
        estimates['total'] = sum(estimates.values())
        
        return estimates


__all__ = [
    'AnalysisConfig',
    'AnalysisPipeline',
    'InputDiscoveryStep',
    'PatchGenerationStep',
    'StatisticalAnalysisStep',
    'ValidationStep',
    'ReportGenerationStep',
    'parse_kappa_filename',
]