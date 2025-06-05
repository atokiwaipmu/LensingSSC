plt.suptitle(f'Peak Counts (ngal = {ngal})')
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / f"peak_counts_ngal{ngal}.{self.figure_format}"
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
        
        return plot_files
    
    def _plot_correlations(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot correlation functions."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'correlations' not in stats_data:
            return plot_files
        
        correlations = stats_data['correlations']
        theta_mids = stats_data.get('theta_mids', np.logspace(-2, 1, 20))
        ngal_list = stats_data.get('ngal_list', list(correlations.keys()))
        correlation_types = stats_data.get('correlation_types', ['2pt'])
        
        # Plot for each correlation type
        for corr_type in correlation_types:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            for ngal in ngal_list:
                if ngal not in correlations or corr_type not in correlations[ngal]:
                    continue
                
                corr_data = correlations[ngal][corr_type]
                if corr_data.size > 0:
                    mean_corr = np.mean(corr_data, axis=0)
                    std_corr = np.std(corr_data, axis=0)
                    
                    ax.errorbar(theta_mids, mean_corr, yerr=std_corr,
                              label=f'ngal = {ngal}', capsize=3)
            
            ax.set_xlabel('θ [arcmin]')
            ax.set_ylabel(f'{corr_type} correlation')
            ax.set_title(f'{corr_type.upper()} Correlation Function')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_file = output_dir / f"correlation_{corr_type}.{self.figure_format}"
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
        
        return plot_files
    
    def _plot_summary(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Generate summary plots."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        # Create summary plot with multiple statistics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Power spectrum comparison
        if 'power_spectra' in stats_data:
            ax = axes[0, 0]
            power_spectra = stats_data['power_spectra']
            l_mids = stats_data.get('l_mids', np.arange(8))
            
            for ngal in [0, 15, 30]:  # Selected ngal values
                if ngal in power_spectra and power_spectra[ngal].size > 0:
                    mean_ps = np.mean(power_spectra[ngal], axis=0)
                    ax.plot(l_mids, mean_ps, label=f'ngal = {ngal}')
            
            ax.set_xlabel('$\\ell# lensing_ssc/processing/steps/output.py
"""
Output generation steps for LensingSSC processing pipelines.

Provides steps for saving results to various formats, generating plots,
and creating comprehensive reports from analysis results.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np

from ..pipeline import ProcessingStep, StepResult, PipelineContext, StepStatus
from lensing_ssc.core.base import ValidationError, ProcessingError, IOError


logger = logging.getLogger(__name__)


class BaseOutputStep(ProcessingStep):
    """Base class for output generation steps."""
    
    def __init__(
        self,
        name: str,
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        create_backup: bool = True,
        compress: bool = False,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.output_dir = Path(output_dir) if output_dir else None
        self.overwrite = overwrite
        self.create_backup = create_backup
        self.compress = compress
    
    def _get_output_dir(self, context: PipelineContext) -> Path:
        """Get output directory from configuration or context."""
        if self.output_dir:
            return self.output_dir
        
        # Try to get from context config
        config = context.config
        for attr in ['output_dir', 'stats_output_dir', 'results_dir']:
            if hasattr(config, attr):
                path = getattr(config, attr)
                if path:
                    return Path(path)
        
        # Default to context temp directory
        return context.temp_dir / "output"
    
    def _ensure_output_dir(self, output_path: Path) -> Path:
        """Ensure output directory exists."""
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _handle_existing_file(self, output_path: Path) -> bool:
        """Handle existing output files. Returns True if should skip."""
        if output_path.exists():
            if not self.overwrite:
                self.logger.info(f"Output exists and overwrite=False: {output_path}")
                return True
            elif self.create_backup:
                backup_path = output_path.with_suffix(f"{output_path.suffix}.backup")
                if backup_path.exists():
                    backup_path.unlink()
                output_path.rename(backup_path)
                self.logger.info(f"Created backup: {backup_path}")
        return False
    
    def _validate_output(self, output_path: Path) -> bool:
        """Validate that output was created successfully."""
        if not output_path.exists():
            self.logger.error(f"Output file was not created: {output_path}")
            return False
        
        if output_path.stat().st_size == 0:
            self.logger.error(f"Output file is empty: {output_path}")
            return False
        
        return True


class HDF5OutputStep(BaseOutputStep):
    """Save analysis results to HDF5 format."""
    
    def __init__(
        self,
        name: str,
        include_metadata: bool = True,
        chunk_size: Optional[int] = None,
        compression: str = 'gzip',
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.include_metadata = include_metadata
        self.chunk_size = chunk_size
        self.compression = compression
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute HDF5 output generation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Get statistics results
            stats_data = self._collect_statistics_data(inputs)
            if not stats_data:
                raise ProcessingError("No statistics data found for HDF5 output")
            
            # Generate output filename
            output_dir = self._get_output_dir(context)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"statistics_results_{timestamp}.hdf5"
            
            # Check if file exists
            if self._handle_existing_file(output_file):
                result.status = StepStatus.SKIPPED
                result.warnings.append(f"Output file already exists: {output_file}")
                return result
            
            # Ensure output directory exists
            self._ensure_output_dir(output_file)
            
            # Save to HDF5
            self._save_to_hdf5(stats_data, output_file, context)
            
            # Validate output
            if not self._validate_output(output_file):
                raise ProcessingError("HDF5 output validation failed")
            
            result.data = {
                'output_file': str(output_file),
                'file_size_mb': output_file.stat().st_size / (1024**2),
                'data_structure': self._get_data_structure_summary(stats_data)
            }
            
            result.metadata = {
                'output_file': str(output_file),
                'file_size_mb': output_file.stat().st_size / (1024**2),
                'compression': self.compression,
                'include_metadata': self.include_metadata,
                'n_datasets': self._count_datasets(stats_data)
            }
            
            self.logger.info(f"HDF5 output saved: {output_file} ({result.metadata['file_size_mb']:.2f} MB)")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _collect_statistics_data(self, inputs: Dict[str, StepResult]) -> Optional[Dict[str, Any]]:
        """Collect statistics data from input steps."""
        stats_data = {}
        
        # Look for statistics results in inputs
        for step_name, step_result in inputs.items():
            if not step_result.is_successful():
                continue
            
            data = step_result.data
            
            # Check for different types of statistics data
            if 'power_spectra' in data:
                stats_data['power_spectra'] = data['power_spectra']
                stats_data.setdefault('metadata', {})['power_spectrum'] = {
                    'l_edges': data.get('l_edges'),
                    'l_mids': data.get('l_mids'),
                    'ngal_list': data.get('ngal_list')
                }
            
            if 'bispectra' in data:
                stats_data['bispectra'] = data['bispectra']
                stats_data.setdefault('metadata', {})['bispectrum'] = {
                    'l_edges': data.get('l_edges'),
                    'l_mids': data.get('l_mids'),
                    'bispectrum_types': data.get('bispectrum_types'),
                    'ngal_list': data.get('ngal_list')
                }
            
            if 'pdfs' in data:
                stats_data['pdfs'] = data['pdfs']
                stats_data.setdefault('metadata', {})['pdf'] = {
                    'nu_bins': data.get('nu_bins'),
                    'nu_mids': data.get('nu_mids'),
                    'ngal_list': data.get('ngal_list'),
                    'smoothing_lengths': data.get('smoothing_lengths')
                }
            
            if 'peak_counts' in data:
                stats_data['peak_counts'] = data['peak_counts']
                stats_data.setdefault('metadata', {})['peak_counts'] = {
                    'nu_bins': data.get('nu_bins'),
                    'nu_mids': data.get('nu_mids'),
                    'ngal_list': data.get('ngal_list'),
                    'smoothing_lengths': data.get('smoothing_lengths'),
                    'count_minima': data.get('count_minima')
                }
            
            if 'correlations' in data:
                stats_data['correlations'] = data['correlations']
                stats_data.setdefault('metadata', {})['correlations'] = {
                    'theta_bins': data.get('theta_bins'),
                    'theta_mids': data.get('theta_mids'),
                    'correlation_types': data.get('correlation_types'),
                    'ngal_list': data.get('ngal_list')
                }
            
            if 'statistics' in data:  # Composite statistics
                stats_data['composite_statistics'] = data['statistics']
                stats_data.setdefault('metadata', {})['composite'] = {
                    'l_edges': data.get('l_edges'),
                    'l_mids': data.get('l_mids'),
                    'nu_bins': data.get('nu_bins'),
                    'nu_mids': data.get('nu_mids'),
                    'ngal_list': data.get('ngal_list'),
                    'smoothing_lengths': data.get('smoothing_lengths'),
                    'statistics_types': data.get('statistics_types')
                }
        
        return stats_data if stats_data else None
    
    def _save_to_hdf5(self, stats_data: Dict[str, Any], output_file: Path, context: PipelineContext) -> None:
        """Save statistics data to HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ProcessingError("h5py is required for HDF5 output")
        
        with h5py.File(output_file, 'w') as f:
            # Save each statistic type
            for stat_type, stat_data in stats_data.items():
                if stat_type == 'metadata':
                    continue
                
                group = f.create_group(stat_type)
                self._save_statistic_to_group(group, stat_data)
            
            # Save metadata if requested
            if self.include_metadata and 'metadata' in stats_data:
                meta_group = f.create_group('metadata')
                self._save_metadata_to_group(meta_group, stats_data['metadata'], context)
    
    def _save_statistic_to_group(self, group, stat_data: Any) -> None:
        """Save statistic data to HDF5 group."""
        if isinstance(stat_data, dict):
            for key, value in stat_data.items():
                if isinstance(value, np.ndarray):
                    # Save array with compression
                    if value.size > 0:
                        group.create_dataset(
                            key, 
                            data=value,
                            compression=self.compression,
                            chunks=True if self.chunk_size else None
                        )
                    else:
                        # Handle empty arrays
                        group.create_dataset(key, data=np.array([]))
                elif isinstance(value, dict):
                    subgroup = group.create_group(key)
                    self._save_statistic_to_group(subgroup, value)
                elif isinstance(value, (list, tuple)):
                    # Convert to array if possible
                    try:
                        arr_value = np.array(value)
                        group.create_dataset(key, data=arr_value)
                    except Exception:
                        # Save as string representation if conversion fails
                        group.attrs[key] = str(value)
                else:
                    # Save as attribute
                    group.attrs[key] = value
        elif isinstance(stat_data, np.ndarray):
            if stat_data.size > 0:
                group.create_dataset(
                    'data',
                    data=stat_data,
                    compression=self.compression,
                    chunks=True if self.chunk_size else None
                )
    
    def _save_metadata_to_group(self, group, metadata: Dict[str, Any], context: PipelineContext) -> None:
        """Save metadata to HDF5 group."""
        # Save pipeline metadata
        pipeline_meta = group.create_group('pipeline')
        pipeline_meta.attrs['timestamp'] = datetime.now().isoformat()
        pipeline_meta.attrs['step_name'] = self.name
        
        # Save configuration metadata
        try:
            config_meta = group.create_group('configuration')
            config = context.config
            
            # Save common config attributes
            config_attrs = [
                'patch_size_deg', 'xsize', 'ngal_list', 'smoothing_lengths',
                'lmin', 'lmax', 'nu_min', 'nu_max'
            ]
            
            for attr in config_attrs:
                if hasattr(config, attr):
                    value = getattr(config, attr)
                    if isinstance(value, (list, tuple)):
                        config_meta.create_dataset(attr, data=np.array(value))
                    else:
                        config_meta.attrs[attr] = value
        except Exception as e:
            self.logger.warning(f"Failed to save configuration metadata: {e}")
        
        # Save statistic-specific metadata
        for stat_type, stat_meta in metadata.items():
            if isinstance(stat_meta, dict):
                stat_group = group.create_group(stat_type)
                for key, value in stat_meta.items():
                    if isinstance(value, np.ndarray):
                        stat_group.create_dataset(key, data=value)
                    elif isinstance(value, (list, tuple)):
                        stat_group.create_dataset(key, data=np.array(value))
                    else:
                        stat_group.attrs[key] = value
    
    def _get_data_structure_summary(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of data structure."""
        summary = {}
        
        for stat_type, stat_data in stats_data.items():
            if stat_type == 'metadata':
                continue
            
            summary[stat_type] = self._summarize_data_structure(stat_data)
        
        return summary
    
    def _summarize_data_structure(self, data: Any, max_depth: int = 3) -> Any:
        """Recursively summarize data structure."""
        if max_depth <= 0:
            return "..."
        
        if isinstance(data, np.ndarray):
            return {
                'type': 'array',
                'shape': data.shape,
                'dtype': str(data.dtype)
            }
        elif isinstance(data, dict):
            return {
                key: self._summarize_data_structure(value, max_depth - 1)
                for key, value in list(data.items())[:5]  # Limit to first 5 items
            }
        elif isinstance(data, (list, tuple)):
            return {
                'type': type(data).__name__,
                'length': len(data),
                'sample': self._summarize_data_structure(data[0], max_depth - 1) if data else None
            }
        else:
            return str(type(data).__name__)
    
    def _count_datasets(self, stats_data: Dict[str, Any]) -> int:
        """Count number of datasets in statistics data."""
        count = 0
        
        def count_recursive(data):
            nonlocal count
            if isinstance(data, np.ndarray):
                count += 1
            elif isinstance(data, dict):
                for value in data.values():
                    count_recursive(value)
        
        count_recursive(stats_data)
        return count


class PlotGenerationStep(BaseOutputStep):
    """Generate plots from analysis results."""
    
    def __init__(
        self,
        name: str,
        plot_types: Optional[List[str]] = None,
        figure_format: str = 'pdf',
        dpi: int = 300,
        figure_size: Tuple[float, float] = (10, 8),
        style: str = 'default',
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.plot_types = plot_types or ['power_spectrum', 'pdf', 'peaks', 'summary']
        self.figure_format = figure_format
        self.dpi = dpi
        self.figure_size = figure_size
        self.style = style
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute plot generation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Check matplotlib availability
            try:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                if self.style != 'default':
                    plt.style.use(self.style)
            except ImportError:
                raise ProcessingError("matplotlib is required for plot generation")
            
            # Get statistics data
            stats_data = self._collect_plot_data(inputs)
            if not stats_data:
                raise ProcessingError("No statistics data found for plotting")
            
            # Generate output directory
            output_dir = self._get_output_dir(context) / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate plots
            generated_plots = {}
            
            for plot_type in self.plot_types:
                try:
                    plot_files = self._generate_plot_type(plot_type, stats_data, output_dir)
                    generated_plots[plot_type] = plot_files
                    self.logger.info(f"Generated {len(plot_files)} {plot_type} plots")
                except Exception as e:
                    self.logger.error(f"Failed to generate {plot_type} plots: {e}")
                    generated_plots[plot_type] = []
            
            total_plots = sum(len(plots) for plots in generated_plots.values())
            
            result.data = {
                'generated_plots': generated_plots,
                'output_dir': str(output_dir),
                'total_plots': total_plots,
                'plot_types': self.plot_types
            }
            
            result.metadata = {
                'total_plots': total_plots,
                'plot_types_generated': list(generated_plots.keys()),
                'output_dir': str(output_dir),
                'figure_format': self.figure_format,
                'dpi': self.dpi
            }
            
            self.logger.info(f"Plot generation completed: {total_plots} plots")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _collect_plot_data(self, inputs: Dict[str, StepResult]) -> Dict[str, Any]:
        """Collect data for plotting from input steps."""
        plot_data = {}
        
        for step_name, step_result in inputs.items():
            if not step_result.is_successful():
                continue
            
            data = step_result.data
            
            # Collect different types of data
            for key in ['power_spectra', 'bispectra', 'pdfs', 'peak_counts', 'correlations', 'statistics']:
                if key in data:
                    plot_data[key] = data[key]
            
            # Collect metadata
            for key in ['l_edges', 'l_mids', 'nu_bins', 'nu_mids', 'theta_bins', 'theta_mids']:
                if key in data:
                    plot_data[key] = data[key]
            
            # Collect configuration
            for key in ['ngal_list', 'smoothing_lengths', 'bispectrum_types', 'correlation_types']:
                if key in data:
                    plot_data[key] = data[key]
        
        return plot_data
    
    def _generate_plot_type(self, plot_type: str, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Generate plots for a specific type."""
        if plot_type == 'power_spectrum':
            return self._plot_power_spectra(stats_data, output_dir)
        elif plot_type == 'bispectrum':
            return self._plot_bispectra(stats_data, output_dir)
        elif plot_type == 'pdf':
            return self._plot_pdfs(stats_data, output_dir)
        elif plot_type == 'peaks':
            return self._plot_peak_counts(stats_data, output_dir)
        elif plot_type == 'correlations':
            return self._plot_correlations(stats_data, output_dir)
        elif plot_type == 'summary':
            return self._plot_summary(stats_data, output_dir)
        else:
            self.logger.warning(f"Unknown plot type: {plot_type}")
            return []
    
    def _plot_power_spectra(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot power spectra."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'power_spectra' not in stats_data:
            return plot_files
        
        power_spectra = stats_data['power_spectra']
        l_mids = stats_data.get('l_mids', np.arange(len(list(power_spectra.values())[0][0])))
        ngal_list = stats_data.get('ngal_list', list(power_spectra.keys()))
        
        # Plot for each ngal value
        for ngal in ngal_list:
            if ngal not in power_spectra:
                continue
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            ps_data = power_spectra[ngal]
            if ps_data.size > 0:
                # Calculate mean and std
                mean_ps = np.mean(ps_data, axis=0)
                std_ps = np.std(ps_data, axis=0)
                
                # Plot with error bars
                ax.errorbar(l_mids, mean_ps, yerr=std_ps, 
                           label=f'ngal = {ngal}', capsize=3)
                
                ax.set_xlabel('$\\ell$')
                ax.set_ylabel('$C_\\ell$')
                ax.set_title(f'Power Spectrum (ngal = {ngal})')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_file = output_dir / f"power_spectrum_ngal{ngal}.{self.figure_format}"
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
        
        return plot_files
    
    def _plot_bispectra(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot bispectra."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'bispectra' not in stats_data:
            return plot_files
        
        bispectra = stats_data['bispectra']
        l_mids = stats_data.get('l_mids', np.arange(8))
        ngal_list = stats_data.get('ngal_list', list(bispectra.keys()))
        bispectrum_types = stats_data.get('bispectrum_types', ['equilateral', 'isosceles', 'squeezed'])
        
        # Plot for each ngal and bispectrum type
        for ngal in ngal_list:
            if ngal not in bispectra:
                continue
            
            fig, axes = plt.subplots(1, len(bispectrum_types), figsize=(5*len(bispectrum_types), 4))
            if len(bispectrum_types) == 1:
                axes = [axes]
            
            for i, bs_type in enumerate(bispectrum_types):
                if bs_type not in bispectra[ngal]:
                    continue
                
                bs_data = bispectra[ngal][bs_type]
                if bs_data.size > 0:
                    mean_bs = np.mean(bs_data, axis=0)
                    std_bs = np.std(bs_data, axis=0)
                    
                    axes[i].errorbar(l_mids, mean_bs, yerr=std_bs, capsize=3)
                    axes[i].set_xlabel('$\\ell$')
                    axes[i].set_ylabel(f'$B_\\ell^{{({bs_type[:3]})}}$')
                    axes[i].set_title(f'{bs_type.capitalize()} Bispectrum')
                    axes[i].set_xscale('log')
                    axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f'Bispectra (ngal = {ngal})')
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / f"bispectrum_ngal{ngal}.{self.figure_format}"
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
        
        return plot_files
    
    def _plot_pdfs(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot probability density functions."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'pdfs' not in stats_data:
            return plot_files
        
        pdfs = stats_data['pdfs']
        nu_mids = stats_data.get('nu_mids', np.linspace(-4, 4, 50))
        ngal_list = stats_data.get('ngal_list', list(pdfs.keys()))
        smoothing_lengths = stats_data.get('smoothing_lengths', [2.0, 5.0, 8.0, 10.0])
        
        # Plot for each ngal value
        for ngal in ngal_list:
            if ngal not in pdfs:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
            axes = axes.flatten()
            
            for i, sl in enumerate(smoothing_lengths[:4]):
                if sl not in pdfs[ngal] or i >= len(axes):
                    continue
                
                pdf_data = pdfs[ngal][sl]['pdf']
                if pdf_data.size > 0:
                    mean_pdf = np.mean(pdf_data, axis=0)
                    std_pdf = np.std(pdf_data, axis=0)
                    
                    axes[i].fill_between(nu_mids, mean_pdf - std_pdf, mean_pdf + std_pdf, 
                                       alpha=0.3, label='±1σ')
                    axes[i].plot(nu_mids, mean_pdf, label=f'sl = {sl}′')
                    axes[i].set_xlabel('ν')
                    axes[i].set_ylabel('P(ν)')
                    axes[i].set_title(f'PDF (sl = {sl}′)')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f'PDFs (ngal = {ngal})')
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / f"pdf_ngal{ngal}.{self.figure_format}"
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
        
        return plot_files
    
    def _plot_peak_counts(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot peak counts."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'peak_counts' not in stats_data:
            return plot_files
        
        peak_counts = stats_data['peak_counts']
        nu_mids = stats_data.get('nu_mids', np.linspace(-4, 4, 50))
        ngal_list = stats_data.get('ngal_list', list(peak_counts.keys()))
        smoothing_lengths = stats_data.get('smoothing_lengths', [2.0, 5.0, 8.0, 10.0])
        
        # Plot for each ngal value
        for ngal in ngal_list:
            if ngal not in peak_counts:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
            axes = axes.flatten()
            
            for i, sl in enumerate(smoothing_lengths[:4]):
                if sl not in peak_counts[ngal] or i >= len(axes):
                    continue
                
                peaks_data = peak_counts[ngal][sl]['peaks']
                if peaks_data.size > 0:
                    mean_peaks = np.mean(peaks_data, axis=0)
                    std_peaks = np.std(peaks_data, axis=0)
                    
                    axes[i].errorbar(nu_mids, mean_peaks, yerr=std_peaks, 
                                   capsize=3, label='Peaks')
                    
                    # Plot minima if available
                    if 'minima' in peak_counts[ngal][sl]:
                        minima_data = peak_counts[ngal][sl]['minima']
                        if minima_data.size > 0:
                            mean_minima = np.mean(minima_data, axis=0)
                            std_minima = np.std(minima_data, axis=0)
                            axes[i].errorbar(nu_mids, mean_minima, yerr=std_minima,
                                           capsize=3, label='Minima', alpha=0.7)
                    
                    axes[i].set_xlabel('ν')
                    axes[i].set_ylabel('N(ν)')
                    axes[i].set_title(f'Peak Counts (sl = {sl}′)')
                    axes[i].set_yscale('log')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f'Peak Counts (ngal = {ngal})')
            plt.)
            ax.set_ylabel('$C_\\ell# lensing_ssc/processing/steps/output.py
"""
Output generation steps for LensingSSC processing pipelines.

Provides steps for saving results to various formats, generating plots,
and creating comprehensive reports from analysis results.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np

from ..pipeline import ProcessingStep, StepResult, PipelineContext, StepStatus
from lensing_ssc.core.base import ValidationError, ProcessingError, IOError


logger = logging.getLogger(__name__)


class BaseOutputStep(ProcessingStep):
    """Base class for output generation steps."""
    
    def __init__(
        self,
        name: str,
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        create_backup: bool = True,
        compress: bool = False,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.output_dir = Path(output_dir) if output_dir else None
        self.overwrite = overwrite
        self.create_backup = create_backup
        self.compress = compress
    
    def _get_output_dir(self, context: PipelineContext) -> Path:
        """Get output directory from configuration or context."""
        if self.output_dir:
            return self.output_dir
        
        # Try to get from context config
        config = context.config
        for attr in ['output_dir', 'stats_output_dir', 'results_dir']:
            if hasattr(config, attr):
                path = getattr(config, attr)
                if path:
                    return Path(path)
        
        # Default to context temp directory
        return context.temp_dir / "output"
    
    def _ensure_output_dir(self, output_path: Path) -> Path:
        """Ensure output directory exists."""
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _handle_existing_file(self, output_path: Path) -> bool:
        """Handle existing output files. Returns True if should skip."""
        if output_path.exists():
            if not self.overwrite:
                self.logger.info(f"Output exists and overwrite=False: {output_path}")
                return True
            elif self.create_backup:
                backup_path = output_path.with_suffix(f"{output_path.suffix}.backup")
                if backup_path.exists():
                    backup_path.unlink()
                output_path.rename(backup_path)
                self.logger.info(f"Created backup: {backup_path}")
        return False
    
    def _validate_output(self, output_path: Path) -> bool:
        """Validate that output was created successfully."""
        if not output_path.exists():
            self.logger.error(f"Output file was not created: {output_path}")
            return False
        
        if output_path.stat().st_size == 0:
            self.logger.error(f"Output file is empty: {output_path}")
            return False
        
        return True


class HDF5OutputStep(BaseOutputStep):
    """Save analysis results to HDF5 format."""
    
    def __init__(
        self,
        name: str,
        include_metadata: bool = True,
        chunk_size: Optional[int] = None,
        compression: str = 'gzip',
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.include_metadata = include_metadata
        self.chunk_size = chunk_size
        self.compression = compression
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute HDF5 output generation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Get statistics results
            stats_data = self._collect_statistics_data(inputs)
            if not stats_data:
                raise ProcessingError("No statistics data found for HDF5 output")
            
            # Generate output filename
            output_dir = self._get_output_dir(context)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"statistics_results_{timestamp}.hdf5"
            
            # Check if file exists
            if self._handle_existing_file(output_file):
                result.status = StepStatus.SKIPPED
                result.warnings.append(f"Output file already exists: {output_file}")
                return result
            
            # Ensure output directory exists
            self._ensure_output_dir(output_file)
            
            # Save to HDF5
            self._save_to_hdf5(stats_data, output_file, context)
            
            # Validate output
            if not self._validate_output(output_file):
                raise ProcessingError("HDF5 output validation failed")
            
            result.data = {
                'output_file': str(output_file),
                'file_size_mb': output_file.stat().st_size / (1024**2),
                'data_structure': self._get_data_structure_summary(stats_data)
            }
            
            result.metadata = {
                'output_file': str(output_file),
                'file_size_mb': output_file.stat().st_size / (1024**2),
                'compression': self.compression,
                'include_metadata': self.include_metadata,
                'n_datasets': self._count_datasets(stats_data)
            }
            
            self.logger.info(f"HDF5 output saved: {output_file} ({result.metadata['file_size_mb']:.2f} MB)")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _collect_statistics_data(self, inputs: Dict[str, StepResult]) -> Optional[Dict[str, Any]]:
        """Collect statistics data from input steps."""
        stats_data = {}
        
        # Look for statistics results in inputs
        for step_name, step_result in inputs.items():
            if not step_result.is_successful():
                continue
            
            data = step_result.data
            
            # Check for different types of statistics data
            if 'power_spectra' in data:
                stats_data['power_spectra'] = data['power_spectra']
                stats_data.setdefault('metadata', {})['power_spectrum'] = {
                    'l_edges': data.get('l_edges'),
                    'l_mids': data.get('l_mids'),
                    'ngal_list': data.get('ngal_list')
                }
            
            if 'bispectra' in data:
                stats_data['bispectra'] = data['bispectra']
                stats_data.setdefault('metadata', {})['bispectrum'] = {
                    'l_edges': data.get('l_edges'),
                    'l_mids': data.get('l_mids'),
                    'bispectrum_types': data.get('bispectrum_types'),
                    'ngal_list': data.get('ngal_list')
                }
            
            if 'pdfs' in data:
                stats_data['pdfs'] = data['pdfs']
                stats_data.setdefault('metadata', {})['pdf'] = {
                    'nu_bins': data.get('nu_bins'),
                    'nu_mids': data.get('nu_mids'),
                    'ngal_list': data.get('ngal_list'),
                    'smoothing_lengths': data.get('smoothing_lengths')
                }
            
            if 'peak_counts' in data:
                stats_data['peak_counts'] = data['peak_counts']
                stats_data.setdefault('metadata', {})['peak_counts'] = {
                    'nu_bins': data.get('nu_bins'),
                    'nu_mids': data.get('nu_mids'),
                    'ngal_list': data.get('ngal_list'),
                    'smoothing_lengths': data.get('smoothing_lengths'),
                    'count_minima': data.get('count_minima')
                }
            
            if 'correlations' in data:
                stats_data['correlations'] = data['correlations']
                stats_data.setdefault('metadata', {})['correlations'] = {
                    'theta_bins': data.get('theta_bins'),
                    'theta_mids': data.get('theta_mids'),
                    'correlation_types': data.get('correlation_types'),
                    'ngal_list': data.get('ngal_list')
                }
            
            if 'statistics' in data:  # Composite statistics
                stats_data['composite_statistics'] = data['statistics']
                stats_data.setdefault('metadata', {})['composite'] = {
                    'l_edges': data.get('l_edges'),
                    'l_mids': data.get('l_mids'),
                    'nu_bins': data.get('nu_bins'),
                    'nu_mids': data.get('nu_mids'),
                    'ngal_list': data.get('ngal_list'),
                    'smoothing_lengths': data.get('smoothing_lengths'),
                    'statistics_types': data.get('statistics_types')
                }
        
        return stats_data if stats_data else None
    
    def _save_to_hdf5(self, stats_data: Dict[str, Any], output_file: Path, context: PipelineContext) -> None:
        """Save statistics data to HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ProcessingError("h5py is required for HDF5 output")
        
        with h5py.File(output_file, 'w') as f:
            # Save each statistic type
            for stat_type, stat_data in stats_data.items():
                if stat_type == 'metadata':
                    continue
                
                group = f.create_group(stat_type)
                self._save_statistic_to_group(group, stat_data)
            
            # Save metadata if requested
            if self.include_metadata and 'metadata' in stats_data:
                meta_group = f.create_group('metadata')
                self._save_metadata_to_group(meta_group, stats_data['metadata'], context)
    
    def _save_statistic_to_group(self, group, stat_data: Any) -> None:
        """Save statistic data to HDF5 group."""
        if isinstance(stat_data, dict):
            for key, value in stat_data.items():
                if isinstance(value, np.ndarray):
                    # Save array with compression
                    if value.size > 0:
                        group.create_dataset(
                            key, 
                            data=value,
                            compression=self.compression,
                            chunks=True if self.chunk_size else None
                        )
                    else:
                        # Handle empty arrays
                        group.create_dataset(key, data=np.array([]))
                elif isinstance(value, dict):
                    subgroup = group.create_group(key)
                    self._save_statistic_to_group(subgroup, value)
                elif isinstance(value, (list, tuple)):
                    # Convert to array if possible
                    try:
                        arr_value = np.array(value)
                        group.create_dataset(key, data=arr_value)
                    except Exception:
                        # Save as string representation if conversion fails
                        group.attrs[key] = str(value)
                else:
                    # Save as attribute
                    group.attrs[key] = value
        elif isinstance(stat_data, np.ndarray):
            if stat_data.size > 0:
                group.create_dataset(
                    'data',
                    data=stat_data,
                    compression=self.compression,
                    chunks=True if self.chunk_size else None
                )
    
    def _save_metadata_to_group(self, group, metadata: Dict[str, Any], context: PipelineContext) -> None:
        """Save metadata to HDF5 group."""
        # Save pipeline metadata
        pipeline_meta = group.create_group('pipeline')
        pipeline_meta.attrs['timestamp'] = datetime.now().isoformat()
        pipeline_meta.attrs['step_name'] = self.name
        
        # Save configuration metadata
        try:
            config_meta = group.create_group('configuration')
            config = context.config
            
            # Save common config attributes
            config_attrs = [
                'patch_size_deg', 'xsize', 'ngal_list', 'smoothing_lengths',
                'lmin', 'lmax', 'nu_min', 'nu_max'
            ]
            
            for attr in config_attrs:
                if hasattr(config, attr):
                    value = getattr(config, attr)
                    if isinstance(value, (list, tuple)):
                        config_meta.create_dataset(attr, data=np.array(value))
                    else:
                        config_meta.attrs[attr] = value
        except Exception as e:
            self.logger.warning(f"Failed to save configuration metadata: {e}")
        
        # Save statistic-specific metadata
        for stat_type, stat_meta in metadata.items():
            if isinstance(stat_meta, dict):
                stat_group = group.create_group(stat_type)
                for key, value in stat_meta.items():
                    if isinstance(value, np.ndarray):
                        stat_group.create_dataset(key, data=value)
                    elif isinstance(value, (list, tuple)):
                        stat_group.create_dataset(key, data=np.array(value))
                    else:
                        stat_group.attrs[key] = value
    
    def _get_data_structure_summary(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of data structure."""
        summary = {}
        
        for stat_type, stat_data in stats_data.items():
            if stat_type == 'metadata':
                continue
            
            summary[stat_type] = self._summarize_data_structure(stat_data)
        
        return summary
    
    def _summarize_data_structure(self, data: Any, max_depth: int = 3) -> Any:
        """Recursively summarize data structure."""
        if max_depth <= 0:
            return "..."
        
        if isinstance(data, np.ndarray):
            return {
                'type': 'array',
                'shape': data.shape,
                'dtype': str(data.dtype)
            }
        elif isinstance(data, dict):
            return {
                key: self._summarize_data_structure(value, max_depth - 1)
                for key, value in list(data.items())[:5]  # Limit to first 5 items
            }
        elif isinstance(data, (list, tuple)):
            return {
                'type': type(data).__name__,
                'length': len(data),
                'sample': self._summarize_data_structure(data[0], max_depth - 1) if data else None
            }
        else:
            return str(type(data).__name__)
    
    def _count_datasets(self, stats_data: Dict[str, Any]) -> int:
        """Count number of datasets in statistics data."""
        count = 0
        
        def count_recursive(data):
            nonlocal count
            if isinstance(data, np.ndarray):
                count += 1
            elif isinstance(data, dict):
                for value in data.values():
                    count_recursive(value)
        
        count_recursive(stats_data)
        return count


class PlotGenerationStep(BaseOutputStep):
    """Generate plots from analysis results."""
    
    def __init__(
        self,
        name: str,
        plot_types: Optional[List[str]] = None,
        figure_format: str = 'pdf',
        dpi: int = 300,
        figure_size: Tuple[float, float] = (10, 8),
        style: str = 'default',
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.plot_types = plot_types or ['power_spectrum', 'pdf', 'peaks', 'summary']
        self.figure_format = figure_format
        self.dpi = dpi
        self.figure_size = figure_size
        self.style = style
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute plot generation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Check matplotlib availability
            try:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                if self.style != 'default':
                    plt.style.use(self.style)
            except ImportError:
                raise ProcessingError("matplotlib is required for plot generation")
            
            # Get statistics data
            stats_data = self._collect_plot_data(inputs)
            if not stats_data:
                raise ProcessingError("No statistics data found for plotting")
            
            # Generate output directory
            output_dir = self._get_output_dir(context) / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate plots
            generated_plots = {}
            
            for plot_type in self.plot_types:
                try:
                    plot_files = self._generate_plot_type(plot_type, stats_data, output_dir)
                    generated_plots[plot_type] = plot_files
                    self.logger.info(f"Generated {len(plot_files)} {plot_type} plots")
                except Exception as e:
                    self.logger.error(f"Failed to generate {plot_type} plots: {e}")
                    generated_plots[plot_type] = []
            
            total_plots = sum(len(plots) for plots in generated_plots.values())
            
            result.data = {
                'generated_plots': generated_plots,
                'output_dir': str(output_dir),
                'total_plots': total_plots,
                'plot_types': self.plot_types
            }
            
            result.metadata = {
                'total_plots': total_plots,
                'plot_types_generated': list(generated_plots.keys()),
                'output_dir': str(output_dir),
                'figure_format': self.figure_format,
                'dpi': self.dpi
            }
            
            self.logger.info(f"Plot generation completed: {total_plots} plots")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _collect_plot_data(self, inputs: Dict[str, StepResult]) -> Dict[str, Any]:
        """Collect data for plotting from input steps."""
        plot_data = {}
        
        for step_name, step_result in inputs.items():
            if not step_result.is_successful():
                continue
            
            data = step_result.data
            
            # Collect different types of data
            for key in ['power_spectra', 'bispectra', 'pdfs', 'peak_counts', 'correlations', 'statistics']:
                if key in data:
                    plot_data[key] = data[key]
            
            # Collect metadata
            for key in ['l_edges', 'l_mids', 'nu_bins', 'nu_mids', 'theta_bins', 'theta_mids']:
                if key in data:
                    plot_data[key] = data[key]
            
            # Collect configuration
            for key in ['ngal_list', 'smoothing_lengths', 'bispectrum_types', 'correlation_types']:
                if key in data:
                    plot_data[key] = data[key]
        
        return plot_data
    
    def _generate_plot_type(self, plot_type: str, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Generate plots for a specific type."""
        if plot_type == 'power_spectrum':
            return self._plot_power_spectra(stats_data, output_dir)
        elif plot_type == 'bispectrum':
            return self._plot_bispectra(stats_data, output_dir)
        elif plot_type == 'pdf':
            return self._plot_pdfs(stats_data, output_dir)
        elif plot_type == 'peaks':
            return self._plot_peak_counts(stats_data, output_dir)
        elif plot_type == 'correlations':
            return self._plot_correlations(stats_data, output_dir)
        elif plot_type == 'summary':
            return self._plot_summary(stats_data, output_dir)
        else:
            self.logger.warning(f"Unknown plot type: {plot_type}")
            return []
    
    def _plot_power_spectra(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot power spectra."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'power_spectra' not in stats_data:
            return plot_files
        
        power_spectra = stats_data['power_spectra']
        l_mids = stats_data.get('l_mids', np.arange(len(list(power_spectra.values())[0][0])))
        ngal_list = stats_data.get('ngal_list', list(power_spectra.keys()))
        
        # Plot for each ngal value
        for ngal in ngal_list:
            if ngal not in power_spectra:
                continue
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            ps_data = power_spectra[ngal]
            if ps_data.size > 0:
                # Calculate mean and std
                mean_ps = np.mean(ps_data, axis=0)
                std_ps = np.std(ps_data, axis=0)
                
                # Plot with error bars
                ax.errorbar(l_mids, mean_ps, yerr=std_ps, 
                           label=f'ngal = {ngal}', capsize=3)
                
                ax.set_xlabel('$\\ell$')
                ax.set_ylabel('$C_\\ell$')
                ax.set_title(f'Power Spectrum (ngal = {ngal})')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_file = output_dir / f"power_spectrum_ngal{ngal}.{self.figure_format}"
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
        
        return plot_files
    
    def _plot_bispectra(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot bispectra."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'bispectra' not in stats_data:
            return plot_files
        
        bispectra = stats_data['bispectra']
        l_mids = stats_data.get('l_mids', np.arange(8))
        ngal_list = stats_data.get('ngal_list', list(bispectra.keys()))
        bispectrum_types = stats_data.get('bispectrum_types', ['equilateral', 'isosceles', 'squeezed'])
        
        # Plot for each ngal and bispectrum type
        for ngal in ngal_list:
            if ngal not in bispectra:
                continue
            
            fig, axes = plt.subplots(1, len(bispectrum_types), figsize=(5*len(bispectrum_types), 4))
            if len(bispectrum_types) == 1:
                axes = [axes]
            
            for i, bs_type in enumerate(bispectrum_types):
                if bs_type not in bispectra[ngal]:
                    continue
                
                bs_data = bispectra[ngal][bs_type]
                if bs_data.size > 0:
                    mean_bs = np.mean(bs_data, axis=0)
                    std_bs = np.std(bs_data, axis=0)
                    
                    axes[i].errorbar(l_mids, mean_bs, yerr=std_bs, capsize=3)
                    axes[i].set_xlabel('$\\ell$')
                    axes[i].set_ylabel(f'$B_\\ell^{{({bs_type[:3]})}}$')
                    axes[i].set_title(f'{bs_type.capitalize()} Bispectrum')
                    axes[i].set_xscale('log')
                    axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f'Bispectra (ngal = {ngal})')
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / f"bispectrum_ngal{ngal}.{self.figure_format}"
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
        
        return plot_files
    
    def _plot_pdfs(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot probability density functions."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'pdfs' not in stats_data:
            return plot_files
        
        pdfs = stats_data['pdfs']
        nu_mids = stats_data.get('nu_mids', np.linspace(-4, 4, 50))
        ngal_list = stats_data.get('ngal_list', list(pdfs.keys()))
        smoothing_lengths = stats_data.get('smoothing_lengths', [2.0, 5.0, 8.0, 10.0])
        
        # Plot for each ngal value
        for ngal in ngal_list:
            if ngal not in pdfs:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
            axes = axes.flatten()
            
            for i, sl in enumerate(smoothing_lengths[:4]):
                if sl not in pdfs[ngal] or i >= len(axes):
                    continue
                
                pdf_data = pdfs[ngal][sl]['pdf']
                if pdf_data.size > 0:
                    mean_pdf = np.mean(pdf_data, axis=0)
                    std_pdf = np.std(pdf_data, axis=0)
                    
                    axes[i].fill_between(nu_mids, mean_pdf - std_pdf, mean_pdf + std_pdf, 
                                       alpha=0.3, label='±1σ')
                    axes[i].plot(nu_mids, mean_pdf, label=f'sl = {sl}′')
                    axes[i].set_xlabel('ν')
                    axes[i].set_ylabel('P(ν)')
                    axes[i].set_title(f'PDF (sl = {sl}′)')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f'PDFs (ngal = {ngal})')
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / f"pdf_ngal{ngal}.{self.figure_format}"
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
        
        return plot_files
    
    def _plot_peak_counts(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot peak counts."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'peak_counts' not in stats_data:
            return plot_files
        
        peak_counts = stats_data['peak_counts']
        nu_mids = stats_data.get('nu_mids', np.linspace(-4, 4, 50))
        ngal_list = stats_data.get('ngal_list', list(peak_counts.keys()))
        smoothing_lengths = stats_data.get('smoothing_lengths', [2.0, 5.0, 8.0, 10.0])
        
        # Plot for each ngal value
        for ngal in ngal_list:
            if ngal not in peak_counts:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
            axes = axes.flatten()
            
            for i, sl in enumerate(smoothing_lengths[:4]):
                if sl not in peak_counts[ngal] or i >= len(axes):
                    continue
                
                peaks_data = peak_counts[ngal][sl]['peaks']
                if peaks_data.size > 0:
                    mean_peaks = np.mean(peaks_data, axis=0)
                    std_peaks = np.std(peaks_data, axis=0)
                    
                    axes[i].errorbar(nu_mids, mean_peaks, yerr=std_peaks, 
                                   capsize=3, label='Peaks')
                    
                    # Plot minima if available
                    if 'minima' in peak_counts[ngal][sl]:
                        minima_data = peak_counts[ngal][sl]['minima']
                        if minima_data.size > 0:
                            mean_minima = np.mean(minima_data, axis=0)
                            std_minima = np.std(minima_data, axis=0)
                            axes[i].errorbar(nu_mids, mean_minima, yerr=std_minima,
                                           capsize=3, label='Minima', alpha=0.7)
                    
                    axes[i].set_xlabel('ν')
                    axes[i].set_ylabel('N(ν)')
                    axes[i].set_title(f'Peak Counts (sl = {sl}′)')
                    axes[i].set_yscale('log')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f'Peak Counts (ngal = {ngal})')
            plt.)
            ax.set_title('Power Spectra Comparison')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: PDF comparison
        if 'pdfs' in stats_data:
            ax = axes[0, 1]
            pdfs = stats_data['pdfs']
            nu_mids = stats_data.get('nu_mids', np.linspace(-4, 4, 50))
            
            for ngal in [0, 15, 30]:
                if ngal in pdfs and 5.0 in pdfs[ngal]:
                    pdf_data = pdfs[ngal][5.0]['pdf']
                    if pdf_data.size > 0:
                        mean_pdf = np.mean(pdf_data, axis=0)
                        ax.plot(nu_mids, mean_pdf, label=f'ngal = {ngal}')
            
            ax.set_xlabel('ν')
            ax.set_ylabel('P(ν)')
            ax.set_title('PDF Comparison (sl = 5′)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Peak counts comparison
        if 'peak_counts' in stats_data:
            ax = axes[1, 0]
            peak_counts = stats_data['peak_counts']
            nu_mids = stats_data.get('nu_mids', np.linspace(-4, 4, 50))
            
            for ngal in [0, 15, 30]:
                if ngal in peak_counts and 5.0 in peak_counts[ngal]:
                    peaks_data = peak_counts[ngal][5.0]['peaks']
                    if peaks_data.size > 0:
                        mean_peaks = np.mean(peaks_data, axis=0)
                        ax.plot(nu_mids, mean_peaks, label=f'ngal = {ngal}')
            
            ax.set_xlabel('ν')
            ax.set_ylabel('N(ν)')
            ax.set_title('Peak Counts Comparison (sl = 5′)')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Noise comparison
        ax = axes[1, 1]
        if 'pdfs' in stats_data:
            pdfs = stats_data['pdfs']
            ngal_values = []
            sigma0_means = []
            sigma0_stds = []
            
            for ngal in sorted(pdfs.keys()):
                if 5.0 in pdfs[ngal] and 'sigma0' in pdfs[ngal][5.0]:
                    sigma0_data = pdfs[ngal][5.0]['sigma0']
                    if sigma0_data.size > 0:
                        ngal_values.append(ngal)
                        sigma0_means.append(np.mean(sigma0_data))
                        sigma0_stds.append(np.std(sigma0_data))
            
            if ngal_values:
                ax.errorbar(ngal_values, sigma0_means, yerr=sigma0_stds, 
                           marker='o', capsize=3)
                ax.set_xlabel('ngal [arcmin⁻²]')
                ax.set_ylabel('σ₀')
                ax.set_title('Noise Level vs Galaxy Density')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Analysis Summary')
        plt.tight_layout()
        
        # Save summary plot
        plot_file = output_dir / f"summary.{self.figure_format}"
        fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        plot_files.append(plot_file)
        
        return plot_files


class ReportGenerationStep(BaseOutputStep):
    """Generate comprehensive analysis reports."""
    
    def __init__(
        self,
        name: str,
        report_format: str = 'markdown',
        include_plots: bool = True,
        include_statistics: bool = True,
        template_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.report_format = report_format
        self.include_plots = include_plots
        self.include_statistics = include_statistics
        self.template_path = Path(template_path) if template_path else None
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute report generation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Collect data for report
            report_data = self._collect_report_data(inputs, context)
            
            # Generate output directory
            output_dir = self._get_output_dir(context)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = output_dir / f"analysis_report_{timestamp}.{self.report_format}"
            
            # Check if file exists
            if self._handle_existing_file(report_file):
                result.status = StepStatus.SKIPPED
                result.warnings.append(f"Report file already exists: {report_file}")
                return result
            
            # Generate report content
            report_content = self._generate_report_content(report_data)
            
            # Save report
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Validate output
            if not self._validate_output(report_file):
                raise ProcessingError("Report output validation failed")
            
            result.data = {
                'report_file': str(report_file),
                'file_size_kb': report_file.stat().st_size / 1024,
                'report_format': self.report_format,
                'sections_included': list(report_data.keys())
            }
            
            result.metadata = {
                'report_file': str(report_file),
                'file_size_kb': report_file.stat().st_size / 1024,
                'report_format': self.report_format,
                'include_plots': self.include_plots,
                'include_statistics': self.include_statistics,
            }
            
            self.logger.info(f"Report generated: {report_file} ({result.metadata['file_size_kb']:.1f} KB)")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _collect_report_data(self, inputs: Dict[str, StepResult], context: PipelineContext) -> Dict[str, Any]:
        """Collect data for report generation."""
        report_data = {
            'pipeline_info': self._get_pipeline_info(inputs, context),
            'processing_summary': self._get_processing_summary(inputs),
            'statistics_summary': self._get_statistics_summary(inputs),
            'plots_info': self._get_plots_info(inputs),
            'configuration': self._get_configuration_info(context),
            'execution_details': self._get_execution_details(inputs)
        }
        
        return report_data
    
    def _get_pipeline_info(self, inputs: Dict[str, StepResult], context: PipelineContext) -> Dict[str, Any]:
        """Get pipeline information."""
        return {
            'execution_time': datetime.now(),
            'total_steps': len(inputs),
            'successful_steps': len([r for r in inputs.values() if r.is_successful()]),
            'failed_steps': len([r for r in inputs.values() if r.status == StepStatus.FAILED]),
            'skipped_steps': len([r for r in inputs.values() if r.status == StepStatus.SKIPPED]),
        }
    
    def _get_processing_summary(self, inputs: Dict[str, StepResult]) -> Dict[str, Any]:
        """Get processing summary."""
        summary = {}
        
        for step_name, step_result in inputs.items():
            if step_result.is_successful():
                summary[step_name] = {
                    'status': step_result.status.value,
                    'execution_time': step_result.execution_time,
                    'warnings_count': len(step_result.warnings),
                    'key_outputs': self._summarize_step_outputs(step_result.data)
                }
        
        return summary
    
    def _get_statistics_summary(self, inputs: Dict[str, StepResult]) -> Dict[str, Any]:
        """Get statistics summary."""
        summary = {}
        
        # Collect statistics from all steps
        for step_name, step_result in inputs.items():
            if not step_result.is_successful():
                continue
            
            data = step_result.data
            
            if 'power_spectra' in data:
                summary['power_spectrum'] = self._summarize_power_spectra(data['power_spectra'])
            
            if 'pdfs' in data:
                summary['pdf'] = self._summarize_pdfs(data['pdfs'])
            
            if 'peak_counts' in data:
                summary['peak_counts'] = self._summarize_peak_counts(data['peak_counts'])
            
            if 'statistics' in data:
                summary['composite'] = self._summarize_composite_statistics(data['statistics'])
        
        return summary
    
    def _get_plots_info(self, inputs: Dict[str, StepResult]) -> Dict[str, Any]:
        """Get plots information."""
        plots_info = {}
        
        for step_name, step_result in inputs.items():
            if step_result.is_successful() and 'generated_plots' in step_result.data:
                plots_info[step_name] = {
                    'total_plots': step_result.data.get('total_plots', 0),
                    'plot_types': step_result.data.get('plot_types', []),
                    'output_dir': step_result.data.get('output_dir')
                }
        
        return plots_info
    
    def _get_configuration_info(self, context: PipelineContext) -> Dict[str, Any]:
        """Get configuration information."""
        config_info = {}
        config = context.config
        
        # Extract common configuration parameters
        config_attrs = [
            'patch_size_deg', 'xsize', 'ngal_list', 'smoothing_lengths',
            'lmin', 'lmax', 'nu_min', 'nu_max', 'num_processes'
        ]
        
        for attr in config_attrs:
            if hasattr(config, attr):
                config_info[attr] = getattr(config, attr)
        
        return config_info
    
    def _get_execution_details(self, inputs: Dict[str, StepResult]) -> Dict[str, Any]:
        """Get execution details."""
        total_execution_time = sum(
            step_result.execution_time for step_result in inputs.values()
            if hasattr(step_result, 'execution_time') and step_result.execution_time
        )
        
        return {
            'total_execution_time': total_execution_time,
            'start_time': min(
                step_result.start_time for step_result in inputs.values()
                if step_result.start_time
            ) if inputs else None,
            'end_time': max(
                step_result.end_time for step_result in inputs.values()
                if step_result.end_time
            ) if inputs else None,
        }
    
    def _generate_report_content(self, report_data: Dict[str, Any]) -> str:
        """Generate report content based on format."""
        if self.report_format == 'markdown':
            return self._generate_markdown_report(report_data)
        elif self.report_format == 'html':
            return self._generate_html_report(report_data)
        elif self.report_format == 'txt':
            return self._generate_text_report(report_data)
        else:
            return self._generate_markdown_report(report_data)  # Default to markdown
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate markdown report."""
        lines = []
        
        # Header
        lines.extend([
            "# LensingSSC Analysis Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ])
        
        # Pipeline Overview
        pipeline_info = report_data.get('pipeline_info', {})
        lines.extend([
            "## Pipeline Overview",
            "",
            f"- **Total Steps:** {pipeline_info.get('total_steps', 0)}",
            f"- **Successful:** {pipeline_info.get('successful_steps', 0)}",
            f"- **Failed:** {pipeline_info.get('failed_steps', 0)}",
            f"- **Skipped:** {pipeline_info.get('skipped_steps', 0)}",
            "",
        ])
        
        # Configuration
        config_info = report_data.get('configuration', {})
        if config_info:
            lines.extend([
                "## Configuration",
                "",
            ])
            for key, value in config_info.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")
        
        # Processing Summary
        processing_summary = report_data.get('processing_summary', {})
        if processing_summary:
            lines.extend([
                "## Processing Summary",
                "",
            ])
            for step_name, step_info in processing_summary.items():
                lines.extend([
                    f"### {step_name}",
                    "",
                    f"- **Status:** {step_info.get('status', 'unknown')}",
                    f"- **Execution Time:** {step_info.get('execution_time', 0):.2f}s",
                    f"- **Warnings:** {step_info.get('warnings_count', 0)}",
                    "",
                ])
        
        # Statistics Summary
        stats_summary = report_data.get('statistics_summary', {})
        if stats_summary:
            lines.extend([
                "## Statistics Summary",
                "",
            ])
            for stat_type, stat_info in stats_summary.items():
                lines.extend([
                    f"### {stat_type.replace('_', ' ').title()}",
                    "",
                ])
                if isinstance(stat_info, dict):
                    for key, value in stat_info.items():
                        lines.append(f"- **{key}:** {value}")
                else:
                    lines.append(f"- {stat_info}")
                lines.append("")
        
        # Plots Information
        plots_info = report_data.get('plots_info', {})
        if plots_info and self.include_plots:
            lines.extend([
                "## Generated Plots",
                "",
            ])
            for step_name, plot_info in plots_info.items():
                lines.extend([
                    f"### {step_name}",
                    "",
                    f"- **Total Plots:** {plot_info.get('total_plots', 0)}",
                    f"- **Plot Types:** {', '.join(plot_info.get('plot_types', []))}",
                    f"- **Output Directory:** {plot_info.get('output_dir', 'N/A')}",
                    "",
                ])
        
        # Execution Details
        exec_details = report_data.get('execution_details', {})
        if exec_details:
            lines.extend([
                "## Execution Details",
                "",
                f"- **Total Execution Time:** {exec_details.get('total_execution_time', 0):.2f}s",
            ])
            
            if exec_details.get('start_time'):
                lines.append(f"- **Start Time:** {exec_details['start_time']}")
            if exec_details.get('end_time'):
                lines.append(f"- **End Time:** {exec_details['end_time']}")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        # Convert markdown to HTML (simplified)
        markdown_content = self._generate_markdown_report(report_data)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LensingSSC Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        ul {{ margin: 10px 0; }}
        pre {{ background: #f5f5f5; padding: 10px; }}
    </style>
</head>
<body>
<pre>{markdown_content}</pre>
</body>
</html>
"""
        return html_content
    
    def _generate_text_report(self, report_data: Dict[str, Any]) -> str:
        """Generate plain text report."""
        # Use markdown without formatting
        return self._generate_markdown_report(report_data).replace('#', '').replace('**', '')
    
    def _summarize_step_outputs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize step outputs."""
        summary = {}
        
        if 'patches' in data:
            patches = data['patches']
            summary['patches'] = f"{len(patches)} patches with shape {patches.shape if len(patches) > 0 else 'N/A'}"
        
        if 'power_spectra' in data:
            summary['power_spectra'] = f"{len(data['power_spectra'])} ngal values"
        
        if 'files' in data:
            summary['files'] = f"{len(data['files'])} files discovered"
        
        return summary
    
    def _summarize_power_spectra(self, power_spectra: Dict) -> Dict[str, Any]:
        """Summarize power spectra."""
        return {
            'ngal_values': list(power_spectra.keys()),
            'n_patches_per_ngal': {ngal: len(ps) for ngal, ps in power_spectra.items() if ps.size > 0}
        }
    
    def _summarize_pdfs(self, pdfs: Dict) -> Dict[str, Any]:
        """Summarize PDFs."""
        return {
            'ngal_values': list(pdfs.keys()),
            'smoothing_lengths': list(pdfs[list(pdfs.keys())[0]].keys()) if pdfs else []
        }
    
    def _summarize_peak_counts(self, peak_counts: Dict) -> Dict[str, Any]:
        """Summarize peak counts."""
        return {
            'ngal_values': list(peak_counts.keys()),
            'smoothing_lengths': list(peak_counts[list(peak_counts.keys())[0]].keys()) if peak_counts else []
        }
    
    def _summarize_composite_statistics(self, statistics: Dict) -> Dict[str, Any]:
        """Summarize composite statistics."""
        return {
            'ngal_values': list(statistics.keys()),
            'statistics_computed': list(statistics[list(statistics.keys())[0]].keys()) if statistics else []
        }


class SummaryStatisticsStep(BaseOutputStep):
    """Generate summary statistics and tables."""
    
    def __init__(
        self,
        name: str,
        output_format: str = 'csv',
        include_metadata: bool = True,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.output_format = output_format
        self.include_metadata = include_metadata
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute summary statistics generation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Collect statistics data
            stats_data = self._collect_statistics_for_summary(inputs)
            if not stats_data:
                raise ProcessingError("No statistics data found for summary")
            
            # Generate output directory
            output_dir = self._get_output_dir(context)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate summary tables
            summary_files = self._generate_summary_tables(stats_data, output_dir)
            
            result.data = {
                'summary_files': summary_files,
                'output_dir': str(output_dir),
                'output_format': self.output_format,
                'n_tables': len(summary_files)
            }
            
            result.metadata = {
                'n_tables_generated': len(summary_files),
                'output_format': self.output_format,
                'output_dir': str(output_dir),
            }
            
            self.logger.info(f"Generated {len(summary_files)} summary tables")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _collect_statistics_for_summary(self, inputs: Dict[str, StepResult]) -> Dict[str, Any]:
        """Collect statistics data for summary generation."""
        stats_data = {}
        
        for step_name, step_result in inputs.items():
            if not step_result.is_successful():
                continue
            
            data = step_result.data
            
            # Collect statistics data
            for key in ['power_spectra', 'pdfs', 'peak_counts', 'statistics']:
                if key in data:
                    stats_data[key] = data[key]
            
            # Collect bin information
            for key in ['l_mids', 'nu_mids', 'ngal_list', 'smoothing_lengths']:
                if key in data:
                    stats_data[key] = data[key]
        
        return stats_data
    
    def _generate_summary_tables(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Generate summary tables."""
        summary_files = []
        
        try:
            import pandas as pd
            pandas_available = True
        except ImportError:
            pandas_available = False
            self.logger.warning("pandas not available, using basic CSV output")
        
        # Generate power spectrum summary
        if 'power_spectra' in stats_data:
            ps_file = self._generate_power_spectrum_summary(
                stats_data, output_dir, use_pandas=pandas_available
            )
            if ps_file:
                summary_files.append(ps_file)
        
        # Generate PDF summary
        if 'pdfs' in stats_data:
            pdf_file = self._generate_pdf_summary(
                stats_data, output_dir, use_pandas=pandas_available
            )
            if pdf_file:
                summary_files.append(pdf_file)
        
        # Generate overall summary
        overall_file = self._generate_overall_summary(
            stats_data, output_dir, use_pandas=pandas_available
        )
        if overall_file:
            summary_files.append(overall_file)
        
        return summary_files
    
    def _generate_power_spectrum_summary(
        self, stats_data: Dict[str, Any], output_dir: Path, use_pandas: bool = True
    ) -> Optional[Path]:
        """Generate power spectrum summary table."""
        if 'power_spectra' not in stats_data:
            return None
        
        power_spectra = stats_data['power_spectra']
        l_mids = stats_data.get('l_mids', np.arange(8))
        
        output_file = output_dir / f"power_spectrum_summary.{self.output_format}"
        
        if use_pandas:
            import pandas as pd
            
            # Create summary dataframe
            summary_data = []
            for ngal, ps_data in power_spectra.items():
                if ps_data.size > 0:
                    mean_ps = np.mean(ps_data, axis=0)
                    std_ps = np.std(ps_data, axis=0)
                    
                    for i, (ell, mean_val, std_val) in enumerate(zip(l_mids, mean_ps, std_ps)):
                        summary_data.append({
                            'ngal': ngal,
                            'ell': ell,
                            'mean_cl': mean_val,
                            'std_cl': std_val,
                            'n_patches': len(ps_data)
                        })
            
            df = pd.DataFrame(summary_data)
            
            if self.output_format == 'csv':
                df.to_csv(output_file, index=False)
            elif self.output_format == 'xlsx':
                df.to_excel(output_file, index=False)
        else:
            # Basic CSV output without pandas
            with open(output_file, 'w') as f:
                f.write("ngal,ell,mean_cl,std_cl,n_patches\n")
                
                for ngal, ps_data in power_spectra.items():
                    if ps_data.size > 0:
                        mean_ps = np.mean(ps_data, axis=0)
                        std_ps = np.std(ps_data, axis=0)
                        
                        for ell, mean_val, std_val in zip(l_mids, mean_ps, std_ps):
                            f.write(f"{ngal},{ell},{mean_val},{std_val},{len(ps_data)}\n")
        
        return output_file
    
    def _generate_pdf_summary(
        self, stats_data: Dict[str, Any], output_dir: Path, use_pandas: bool = True
    ) -> Optional[Path]:
        """Generate PDF summary table."""
        if 'pdfs' not in stats_data:
            return None
        
        pdfs = stats_data['pdfs']
        nu_mids = stats_data.get('nu_mids', np.linspace(-4, 4, 50))
        
        output_file = output_dir / f"pdf_summary.{self.output_format}"
        
        if use_pandas:
            import pandas as pd
            
            summary_data = []
            for ngal, ngal_data in pdfs.items():
                for sl, sl_data in ngal_data.items():
                    if 'pdf' in sl_data and sl_data['pdf'].size > 0:
                        mean_pdf = np.mean(sl_data['pdf'], axis=0)
                        std_pdf = np.std(sl_data['pdf'], axis=0)
                        mean_sigma0 = np.mean(sl_data.get('sigma0', [0]))
                        
                        for nu, mean_val, std_val in zip(nu_mids, mean_pdf, std_pdf):
                            summary_data.append({
                                'ngal': ngal,
                                'smoothing_length': sl,
                                'nu': nu,
                                'mean_pdf': mean_val,
                                'std_pdf': std_val,
                                'mean_sigma0': mean_sigma0,
                                'n_patches': len(sl_data['pdf'])
                            })
            
            df = pd.DataFrame(summary_data)
            
            if self.output_format == 'csv':
                df.to_csv(output_file, index=False)
            elif self.output_format == 'xlsx':
                df.to_excel(output_file, index=False)
        else:
            # Basic CSV output
            with open(output_file, 'w') as f:
                f.write("ngal,smoothing_length,nu,mean_pdf,std_pdf,mean_sigma0,n_patches\n")
                
                for ngal, ngal_data in pdfs.items():
                    for sl, sl_data in ngal_data.items():
                        if 'pdf' in sl_data and sl_data['pdf'].size > 0:
                            mean_pdf = np.mean(sl_data['pdf'], axis=0)
                            std_pdf = np.std(sl_data['pdf'], axis=0)
                            mean_sigma0 = np.mean(sl_data.get('sigma0', [0]))
                            
                            for nu, mean_val, std_val in zip(nu_mids, mean_pdf, std_pdf):
                                f.write(f"{ngal},{sl},{nu},{mean_val},{std_val},{mean_sigma0},{len(sl_data['pdf'])}\n")
        
        return output_file
    
    def _generate_overall_summary(
        self, stats_data: Dict[str, Any], output_dir: Path, use_pandas: bool = True
    ) -> Optional[Path]:
        """Generate overall summary table."""
        output_file = output_dir / f"overall_summary.{self.output_format}"
        
        # Collect overall statistics
        summary_info = {}
        
        # Count statistics by type
        for stat_type in ['power_spectra', 'pdfs', 'peak_counts', 'correlations']:
            if stat_type in stats_data:
                stat_data = stats_data[stat_type]
                if isinstance(stat_data, dict):
                    summary_info[f'{stat_type}_ngal_count'] = len(stat_data)
                    
                    # Count total patches
                    total_patches = 0
                    for ngal_data in stat_data.values():
                        if isinstance(ngal_data, np.ndarray) and ngal_data.size > 0:
                            total_patches += len(ngal_data)
                        elif isinstance(ngal_data, dict):
                            for sub_data in ngal_data.values():
                                if isinstance(sub_data, dict) and 'pdf' in sub_data:
                                    if sub_data['pdf'].size > 0:
                                        total_patches += len(sub_data['pdf'])
                                        break
                                elif isinstance(sub_data, np.ndarray) and sub_data.size > 0:
                                    total_patches += len(sub_data)
                                    break
                    
                    summary_info[f'{stat_type}_total_patches'] = total_patches
        
        # Add configuration info
        for key in ['ngal_list', 'smoothing_lengths', 'l_mids', 'nu_mids']:
            if key in stats_data:
                value = stats_data[key]
                if isinstance(value, (list, np.ndarray)):
                    summary_info[f'{key}_count'] = len(value)
                    if key in ['l_mids', 'nu_mids']:
                        summary_info[f'{key}_range'] = f"{np.min(value):.3f} - {np.max(value):.3f}"
        
        if use_pandas:
            import pandas as pd
            
            # Convert to dataframe
            df = pd.DataFrame([summary_info])
            
            if self.output_format == 'csv':
                df.to_csv(output_file, index=False)
            elif self.output_format == 'xlsx':
                df.to_excel(output_file, index=False)
        else:
            # Basic CSV output
            with open(output_file, 'w') as f:
                # Write header
                f.write(','.join(summary_info.keys()) + '\n')
                # Write values
                f.write(','.join(str(v) for v in summary_info.values()) + '\n')
        
        return output_file


__all__ = [
    'BaseOutputStep',
    'HDF5OutputStep',
    'PlotGenerationStep',
    'ReportGenerationStep',
    'SummaryStatisticsStep',
]
                                '# lensing_ssc/processing/steps/output.py
"""
Output generation steps for LensingSSC processing pipelines.

Provides steps for saving results to various formats, generating plots,
and creating comprehensive reports from analysis results.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np

from ..pipeline import ProcessingStep, StepResult, PipelineContext, StepStatus
from lensing_ssc.core.base import ValidationError, ProcessingError, IOError


logger = logging.getLogger(__name__)


class BaseOutputStep(ProcessingStep):
    """Base class for output generation steps."""
    
    def __init__(
        self,
        name: str,
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        create_backup: bool = True,
        compress: bool = False,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.output_dir = Path(output_dir) if output_dir else None
        self.overwrite = overwrite
        self.create_backup = create_backup
        self.compress = compress
    
    def _get_output_dir(self, context: PipelineContext) -> Path:
        """Get output directory from configuration or context."""
        if self.output_dir:
            return self.output_dir
        
        # Try to get from context config
        config = context.config
        for attr in ['output_dir', 'stats_output_dir', 'results_dir']:
            if hasattr(config, attr):
                path = getattr(config, attr)
                if path:
                    return Path(path)
        
        # Default to context temp directory
        return context.temp_dir / "output"
    
    def _ensure_output_dir(self, output_path: Path) -> Path:
        """Ensure output directory exists."""
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _handle_existing_file(self, output_path: Path) -> bool:
        """Handle existing output files. Returns True if should skip."""
        if output_path.exists():
            if not self.overwrite:
                self.logger.info(f"Output exists and overwrite=False: {output_path}")
                return True
            elif self.create_backup:
                backup_path = output_path.with_suffix(f"{output_path.suffix}.backup")
                if backup_path.exists():
                    backup_path.unlink()
                output_path.rename(backup_path)
                self.logger.info(f"Created backup: {backup_path}")
        return False
    
    def _validate_output(self, output_path: Path) -> bool:
        """Validate that output was created successfully."""
        if not output_path.exists():
            self.logger.error(f"Output file was not created: {output_path}")
            return False
        
        if output_path.stat().st_size == 0:
            self.logger.error(f"Output file is empty: {output_path}")
            return False
        
        return True


class HDF5OutputStep(BaseOutputStep):
    """Save analysis results to HDF5 format."""
    
    def __init__(
        self,
        name: str,
        include_metadata: bool = True,
        chunk_size: Optional[int] = None,
        compression: str = 'gzip',
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.include_metadata = include_metadata
        self.chunk_size = chunk_size
        self.compression = compression
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute HDF5 output generation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Get statistics results
            stats_data = self._collect_statistics_data(inputs)
            if not stats_data:
                raise ProcessingError("No statistics data found for HDF5 output")
            
            # Generate output filename
            output_dir = self._get_output_dir(context)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"statistics_results_{timestamp}.hdf5"
            
            # Check if file exists
            if self._handle_existing_file(output_file):
                result.status = StepStatus.SKIPPED
                result.warnings.append(f"Output file already exists: {output_file}")
                return result
            
            # Ensure output directory exists
            self._ensure_output_dir(output_file)
            
            # Save to HDF5
            self._save_to_hdf5(stats_data, output_file, context)
            
            # Validate output
            if not self._validate_output(output_file):
                raise ProcessingError("HDF5 output validation failed")
            
            result.data = {
                'output_file': str(output_file),
                'file_size_mb': output_file.stat().st_size / (1024**2),
                'data_structure': self._get_data_structure_summary(stats_data)
            }
            
            result.metadata = {
                'output_file': str(output_file),
                'file_size_mb': output_file.stat().st_size / (1024**2),
                'compression': self.compression,
                'include_metadata': self.include_metadata,
                'n_datasets': self._count_datasets(stats_data)
            }
            
            self.logger.info(f"HDF5 output saved: {output_file} ({result.metadata['file_size_mb']:.2f} MB)")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _collect_statistics_data(self, inputs: Dict[str, StepResult]) -> Optional[Dict[str, Any]]:
        """Collect statistics data from input steps."""
        stats_data = {}
        
        # Look for statistics results in inputs
        for step_name, step_result in inputs.items():
            if not step_result.is_successful():
                continue
            
            data = step_result.data
            
            # Check for different types of statistics data
            if 'power_spectra' in data:
                stats_data['power_spectra'] = data['power_spectra']
                stats_data.setdefault('metadata', {})['power_spectrum'] = {
                    'l_edges': data.get('l_edges'),
                    'l_mids': data.get('l_mids'),
                    'ngal_list': data.get('ngal_list')
                }
            
            if 'bispectra' in data:
                stats_data['bispectra'] = data['bispectra']
                stats_data.setdefault('metadata', {})['bispectrum'] = {
                    'l_edges': data.get('l_edges'),
                    'l_mids': data.get('l_mids'),
                    'bispectrum_types': data.get('bispectrum_types'),
                    'ngal_list': data.get('ngal_list')
                }
            
            if 'pdfs' in data:
                stats_data['pdfs'] = data['pdfs']
                stats_data.setdefault('metadata', {})['pdf'] = {
                    'nu_bins': data.get('nu_bins'),
                    'nu_mids': data.get('nu_mids'),
                    'ngal_list': data.get('ngal_list'),
                    'smoothing_lengths': data.get('smoothing_lengths')
                }
            
            if 'peak_counts' in data:
                stats_data['peak_counts'] = data['peak_counts']
                stats_data.setdefault('metadata', {})['peak_counts'] = {
                    'nu_bins': data.get('nu_bins'),
                    'nu_mids': data.get('nu_mids'),
                    'ngal_list': data.get('ngal_list'),
                    'smoothing_lengths': data.get('smoothing_lengths'),
                    'count_minima': data.get('count_minima')
                }
            
            if 'correlations' in data:
                stats_data['correlations'] = data['correlations']
                stats_data.setdefault('metadata', {})['correlations'] = {
                    'theta_bins': data.get('theta_bins'),
                    'theta_mids': data.get('theta_mids'),
                    'correlation_types': data.get('correlation_types'),
                    'ngal_list': data.get('ngal_list')
                }
            
            if 'statistics' in data:  # Composite statistics
                stats_data['composite_statistics'] = data['statistics']
                stats_data.setdefault('metadata', {})['composite'] = {
                    'l_edges': data.get('l_edges'),
                    'l_mids': data.get('l_mids'),
                    'nu_bins': data.get('nu_bins'),
                    'nu_mids': data.get('nu_mids'),
                    'ngal_list': data.get('ngal_list'),
                    'smoothing_lengths': data.get('smoothing_lengths'),
                    'statistics_types': data.get('statistics_types')
                }
        
        return stats_data if stats_data else None
    
    def _save_to_hdf5(self, stats_data: Dict[str, Any], output_file: Path, context: PipelineContext) -> None:
        """Save statistics data to HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ProcessingError("h5py is required for HDF5 output")
        
        with h5py.File(output_file, 'w') as f:
            # Save each statistic type
            for stat_type, stat_data in stats_data.items():
                if stat_type == 'metadata':
                    continue
                
                group = f.create_group(stat_type)
                self._save_statistic_to_group(group, stat_data)
            
            # Save metadata if requested
            if self.include_metadata and 'metadata' in stats_data:
                meta_group = f.create_group('metadata')
                self._save_metadata_to_group(meta_group, stats_data['metadata'], context)
    
    def _save_statistic_to_group(self, group, stat_data: Any) -> None:
        """Save statistic data to HDF5 group."""
        if isinstance(stat_data, dict):
            for key, value in stat_data.items():
                if isinstance(value, np.ndarray):
                    # Save array with compression
                    if value.size > 0:
                        group.create_dataset(
                            key, 
                            data=value,
                            compression=self.compression,
                            chunks=True if self.chunk_size else None
                        )
                    else:
                        # Handle empty arrays
                        group.create_dataset(key, data=np.array([]))
                elif isinstance(value, dict):
                    subgroup = group.create_group(key)
                    self._save_statistic_to_group(subgroup, value)
                elif isinstance(value, (list, tuple)):
                    # Convert to array if possible
                    try:
                        arr_value = np.array(value)
                        group.create_dataset(key, data=arr_value)
                    except Exception:
                        # Save as string representation if conversion fails
                        group.attrs[key] = str(value)
                else:
                    # Save as attribute
                    group.attrs[key] = value
        elif isinstance(stat_data, np.ndarray):
            if stat_data.size > 0:
                group.create_dataset(
                    'data',
                    data=stat_data,
                    compression=self.compression,
                    chunks=True if self.chunk_size else None
                )
    
    def _save_metadata_to_group(self, group, metadata: Dict[str, Any], context: PipelineContext) -> None:
        """Save metadata to HDF5 group."""
        # Save pipeline metadata
        pipeline_meta = group.create_group('pipeline')
        pipeline_meta.attrs['timestamp'] = datetime.now().isoformat()
        pipeline_meta.attrs['step_name'] = self.name
        
        # Save configuration metadata
        try:
            config_meta = group.create_group('configuration')
            config = context.config
            
            # Save common config attributes
            config_attrs = [
                'patch_size_deg', 'xsize', 'ngal_list', 'smoothing_lengths',
                'lmin', 'lmax', 'nu_min', 'nu_max'
            ]
            
            for attr in config_attrs:
                if hasattr(config, attr):
                    value = getattr(config, attr)
                    if isinstance(value, (list, tuple)):
                        config_meta.create_dataset(attr, data=np.array(value))
                    else:
                        config_meta.attrs[attr] = value
        except Exception as e:
            self.logger.warning(f"Failed to save configuration metadata: {e}")
        
        # Save statistic-specific metadata
        for stat_type, stat_meta in metadata.items():
            if isinstance(stat_meta, dict):
                stat_group = group.create_group(stat_type)
                for key, value in stat_meta.items():
                    if isinstance(value, np.ndarray):
                        stat_group.create_dataset(key, data=value)
                    elif isinstance(value, (list, tuple)):
                        stat_group.create_dataset(key, data=np.array(value))
                    else:
                        stat_group.attrs[key] = value
    
    def _get_data_structure_summary(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of data structure."""
        summary = {}
        
        for stat_type, stat_data in stats_data.items():
            if stat_type == 'metadata':
                continue
            
            summary[stat_type] = self._summarize_data_structure(stat_data)
        
        return summary
    
    def _summarize_data_structure(self, data: Any, max_depth: int = 3) -> Any:
        """Recursively summarize data structure."""
        if max_depth <= 0:
            return "..."
        
        if isinstance(data, np.ndarray):
            return {
                'type': 'array',
                'shape': data.shape,
                'dtype': str(data.dtype)
            }
        elif isinstance(data, dict):
            return {
                key: self._summarize_data_structure(value, max_depth - 1)
                for key, value in list(data.items())[:5]  # Limit to first 5 items
            }
        elif isinstance(data, (list, tuple)):
            return {
                'type': type(data).__name__,
                'length': len(data),
                'sample': self._summarize_data_structure(data[0], max_depth - 1) if data else None
            }
        else:
            return str(type(data).__name__)
    
    def _count_datasets(self, stats_data: Dict[str, Any]) -> int:
        """Count number of datasets in statistics data."""
        count = 0
        
        def count_recursive(data):
            nonlocal count
            if isinstance(data, np.ndarray):
                count += 1
            elif isinstance(data, dict):
                for value in data.values():
                    count_recursive(value)
        
        count_recursive(stats_data)
        return count


class PlotGenerationStep(BaseOutputStep):
    """Generate plots from analysis results."""
    
    def __init__(
        self,
        name: str,
        plot_types: Optional[List[str]] = None,
        figure_format: str = 'pdf',
        dpi: int = 300,
        figure_size: Tuple[float, float] = (10, 8),
        style: str = 'default',
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.plot_types = plot_types or ['power_spectrum', 'pdf', 'peaks', 'summary']
        self.figure_format = figure_format
        self.dpi = dpi
        self.figure_size = figure_size
        self.style = style
    
    def execute(self, context: PipelineContext, inputs: Dict[str, StepResult]) -> StepResult:
        """Execute plot generation."""
        result = StepResult(
            step_name=self.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Check matplotlib availability
            try:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                if self.style != 'default':
                    plt.style.use(self.style)
            except ImportError:
                raise ProcessingError("matplotlib is required for plot generation")
            
            # Get statistics data
            stats_data = self._collect_plot_data(inputs)
            if not stats_data:
                raise ProcessingError("No statistics data found for plotting")
            
            # Generate output directory
            output_dir = self._get_output_dir(context) / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate plots
            generated_plots = {}
            
            for plot_type in self.plot_types:
                try:
                    plot_files = self._generate_plot_type(plot_type, stats_data, output_dir)
                    generated_plots[plot_type] = plot_files
                    self.logger.info(f"Generated {len(plot_files)} {plot_type} plots")
                except Exception as e:
                    self.logger.error(f"Failed to generate {plot_type} plots: {e}")
                    generated_plots[plot_type] = []
            
            total_plots = sum(len(plots) for plots in generated_plots.values())
            
            result.data = {
                'generated_plots': generated_plots,
                'output_dir': str(output_dir),
                'total_plots': total_plots,
                'plot_types': self.plot_types
            }
            
            result.metadata = {
                'total_plots': total_plots,
                'plot_types_generated': list(generated_plots.keys()),
                'output_dir': str(output_dir),
                'figure_format': self.figure_format,
                'dpi': self.dpi
            }
            
            self.logger.info(f"Plot generation completed: {total_plots} plots")
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def _collect_plot_data(self, inputs: Dict[str, StepResult]) -> Dict[str, Any]:
        """Collect data for plotting from input steps."""
        plot_data = {}
        
        for step_name, step_result in inputs.items():
            if not step_result.is_successful():
                continue
            
            data = step_result.data
            
            # Collect different types of data
            for key in ['power_spectra', 'bispectra', 'pdfs', 'peak_counts', 'correlations', 'statistics']:
                if key in data:
                    plot_data[key] = data[key]
            
            # Collect metadata
            for key in ['l_edges', 'l_mids', 'nu_bins', 'nu_mids', 'theta_bins', 'theta_mids']:
                if key in data:
                    plot_data[key] = data[key]
            
            # Collect configuration
            for key in ['ngal_list', 'smoothing_lengths', 'bispectrum_types', 'correlation_types']:
                if key in data:
                    plot_data[key] = data[key]
        
        return plot_data
    
    def _generate_plot_type(self, plot_type: str, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Generate plots for a specific type."""
        if plot_type == 'power_spectrum':
            return self._plot_power_spectra(stats_data, output_dir)
        elif plot_type == 'bispectrum':
            return self._plot_bispectra(stats_data, output_dir)
        elif plot_type == 'pdf':
            return self._plot_pdfs(stats_data, output_dir)
        elif plot_type == 'peaks':
            return self._plot_peak_counts(stats_data, output_dir)
        elif plot_type == 'correlations':
            return self._plot_correlations(stats_data, output_dir)
        elif plot_type == 'summary':
            return self._plot_summary(stats_data, output_dir)
        else:
            self.logger.warning(f"Unknown plot type: {plot_type}")
            return []
    
    def _plot_power_spectra(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot power spectra."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'power_spectra' not in stats_data:
            return plot_files
        
        power_spectra = stats_data['power_spectra']
        l_mids = stats_data.get('l_mids', np.arange(len(list(power_spectra.values())[0][0])))
        ngal_list = stats_data.get('ngal_list', list(power_spectra.keys()))
        
        # Plot for each ngal value
        for ngal in ngal_list:
            if ngal not in power_spectra:
                continue
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            ps_data = power_spectra[ngal]
            if ps_data.size > 0:
                # Calculate mean and std
                mean_ps = np.mean(ps_data, axis=0)
                std_ps = np.std(ps_data, axis=0)
                
                # Plot with error bars
                ax.errorbar(l_mids, mean_ps, yerr=std_ps, 
                           label=f'ngal = {ngal}', capsize=3)
                
                ax.set_xlabel('$\\ell$')
                ax.set_ylabel('$C_\\ell$')
                ax.set_title(f'Power Spectrum (ngal = {ngal})')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_file = output_dir / f"power_spectrum_ngal{ngal}.{self.figure_format}"
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
        
        return plot_files
    
    def _plot_bispectra(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot bispectra."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'bispectra' not in stats_data:
            return plot_files
        
        bispectra = stats_data['bispectra']
        l_mids = stats_data.get('l_mids', np.arange(8))
        ngal_list = stats_data.get('ngal_list', list(bispectra.keys()))
        bispectrum_types = stats_data.get('bispectrum_types', ['equilateral', 'isosceles', 'squeezed'])
        
        # Plot for each ngal and bispectrum type
        for ngal in ngal_list:
            if ngal not in bispectra:
                continue
            
            fig, axes = plt.subplots(1, len(bispectrum_types), figsize=(5*len(bispectrum_types), 4))
            if len(bispectrum_types) == 1:
                axes = [axes]
            
            for i, bs_type in enumerate(bispectrum_types):
                if bs_type not in bispectra[ngal]:
                    continue
                
                bs_data = bispectra[ngal][bs_type]
                if bs_data.size > 0:
                    mean_bs = np.mean(bs_data, axis=0)
                    std_bs = np.std(bs_data, axis=0)
                    
                    axes[i].errorbar(l_mids, mean_bs, yerr=std_bs, capsize=3)
                    axes[i].set_xlabel('$\\ell$')
                    axes[i].set_ylabel(f'$B_\\ell^{{({bs_type[:3]})}}$')
                    axes[i].set_title(f'{bs_type.capitalize()} Bispectrum')
                    axes[i].set_xscale('log')
                    axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f'Bispectra (ngal = {ngal})')
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / f"bispectrum_ngal{ngal}.{self.figure_format}"
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
        
        return plot_files
    
    def _plot_pdfs(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot probability density functions."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'pdfs' not in stats_data:
            return plot_files
        
        pdfs = stats_data['pdfs']
        nu_mids = stats_data.get('nu_mids', np.linspace(-4, 4, 50))
        ngal_list = stats_data.get('ngal_list', list(pdfs.keys()))
        smoothing_lengths = stats_data.get('smoothing_lengths', [2.0, 5.0, 8.0, 10.0])
        
        # Plot for each ngal value
        for ngal in ngal_list:
            if ngal not in pdfs:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
            axes = axes.flatten()
            
            for i, sl in enumerate(smoothing_lengths[:4]):
                if sl not in pdfs[ngal] or i >= len(axes):
                    continue
                
                pdf_data = pdfs[ngal][sl]['pdf']
                if pdf_data.size > 0:
                    mean_pdf = np.mean(pdf_data, axis=0)
                    std_pdf = np.std(pdf_data, axis=0)
                    
                    axes[i].fill_between(nu_mids, mean_pdf - std_pdf, mean_pdf + std_pdf, 
                                       alpha=0.3, label='±1σ')
                    axes[i].plot(nu_mids, mean_pdf, label=f'sl = {sl}′')
                    axes[i].set_xlabel('ν')
                    axes[i].set_ylabel('P(ν)')
                    axes[i].set_title(f'PDF (sl = {sl}′)')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f'PDFs (ngal = {ngal})')
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / f"pdf_ngal{ngal}.{self.figure_format}"
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
        
        return plot_files
    
    def _plot_peak_counts(self, stats_data: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Plot peak counts."""
        import matplotlib.pyplot as plt
        
        plot_files = []
        
        if 'peak_counts' not in stats_data:
            return plot_files
        
        peak_counts = stats_data['peak_counts']
        nu_mids = stats_data.get('nu_mids', np.linspace(-4, 4, 50))
        ngal_list = stats_data.get('ngal_list', list(peak_counts.keys()))
        smoothing_lengths = stats_data.get('smoothing_lengths', [2.0, 5.0, 8.0, 10.0])
        
        # Plot for each ngal value
        for ngal in ngal_list:
            if ngal not in peak_counts:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
            axes = axes.flatten()
            
            for i, sl in enumerate(smoothing_lengths[:4]):
                if sl not in peak_counts[ngal] or i >= len(axes):
                    continue
                
                peaks_data = peak_counts[ngal][sl]['peaks']
                if peaks_data.size > 0:
                    mean_peaks = np.mean(peaks_data, axis=0)
                    std_peaks = np.std(peaks_data, axis=0)
                    
                    axes[i].errorbar(nu_mids, mean_peaks, yerr=std_peaks, 
                                   capsize=3, label='Peaks')
                    
                    # Plot minima if available
                    if 'minima' in peak_counts[ngal][sl]:
                        minima_data = peak_counts[ngal][sl]['minima']
                        if minima_data.size > 0:
                            mean_minima = np.mean(minima_data, axis=0)
                            std_minima = np.std(minima_data, axis=0)
                            axes[i].errorbar(nu_mids, mean_minima, yerr=std_minima,
                                           capsize=3, label='Minima', alpha=0.7)
                    
                    axes[i].set_xlabel('ν')
                    axes[i].set_ylabel('N(ν)')
                    axes[i].set_title(f'Peak Counts (sl = {sl}′)')
                    axes[i].set_yscale('log')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f'Peak Counts (ngal = {ngal})')
            plt.