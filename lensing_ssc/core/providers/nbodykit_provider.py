"""
NBBodyKit provider implementation.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from ..core.interfaces.data_interface import CatalogProvider
from ..core.base.exceptions import ProviderError
from .base_provider import LazyProvider


class NbodykitProvider(LazyProvider):
    """Provider for N-body simulation data using nbodykit."""
    
    def __init__(self):
        super().__init__()
        self._nbodykit = None
    
    @property
    def name(self) -> str:
        return "NbodykitProvider"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def _check_dependencies(self) -> None:
        """Check if nbodykit is available."""
        try:
            import nbodykit
            self._nbodykit = nbodykit
        except ImportError:
            raise ImportError("nbodykit is required for NbodykitProvider")
    
    def _initialize_backend(self, **kwargs) -> None:
        """Initialize nbodykit backend."""
        self._check_dependencies()
    
    def read_catalog(self, path: Union[str, Path], dataset: str = None, **kwargs) -> Any:
        """Read catalog from file."""
        self.ensure_initialized()
        
        try:
            from nbodykit.lab import BigFileCatalog, CSVCatalog, FITSCatalog
            
            path = Path(path)
            
            if path.is_dir():
                # BigFile format
                if dataset is None:
                    raise ProviderError("dataset parameter required for BigFile catalogs")
                return BigFileCatalog(str(path), dataset=dataset, **kwargs)
            
            elif path.suffix.lower() == '.csv':
                # CSV format
                return CSVCatalog(str(path), **kwargs)
            
            elif path.suffix.lower() in ['.fits', '.fit']:
                # FITS format
                return FITSCatalog(str(path), **kwargs)
            
            else:
                raise ProviderError(f"Unsupported file format: {path.suffix}")
                
        except Exception as e:
            raise ProviderError(f"Failed to read catalog from {path}: {e}")
    
    def get_column(self, catalog: Any, column: str, 
                  start: Optional[int] = None, end: Optional[int] = None) -> np.ndarray:
        """Get column from catalog."""
        self.ensure_initialized()
        
        try:
            if start is not None or end is not None:
                return catalog[column][start:end].compute()
            else:
                return catalog[column].compute()
        except Exception as e:
            raise ProviderError(f"Failed to get column {column}: {e}")
    
    def get_attributes(self, catalog: Any) -> Dict[str, Any]:
        """Get catalog attributes."""
        self.ensure_initialized()
        
        try:
            return dict(catalog.attrs)
        except Exception as e:
            raise ProviderError(f"Failed to get attributes: {e}")
    
    def get_size(self, catalog: Any) -> int:
        """Get catalog size."""
        self.ensure_initialized()
        
        try:
            return catalog.size
        except Exception as e:
            raise ProviderError(f"Failed to get catalog size: {e}")
    
    def create_mesh(self, catalog: Any, boxsize: Union[float, List[float]], 
                   nmesh: Union[int, List[int]], position: str = 'Position',
                   weight: Optional[str] = None, **kwargs) -> Any:
        """Create mesh from catalog."""
        self.ensure_initialized()
        
        try:
            mesh = catalog.to_mesh(boxsize=boxsize, Nmesh=nmesh, 
                                 position=position, weight=weight, **kwargs)
            return mesh
        except Exception as e:
            raise ProviderError(f"Failed to create mesh: {e}")
    
    def mesh_to_real_field(self, mesh: Any) -> np.ndarray:
        """Convert mesh to real field."""
        self.ensure_initialized()
        
        try:
            return mesh.to_real_field()
        except Exception as e:
            raise ProviderError(f"Failed to convert mesh to real field: {e}")
    
    def mesh_to_complex_field(self, mesh: Any) -> Any:
        """Convert mesh to complex field."""
        self.ensure_initialized()
        
        try:
            return mesh.to_complex_field()
        except Exception as e:
            raise ProviderError(f"Failed to convert mesh to complex field: {e}")
    
    def compute_power_spectrum(self, mesh1: Any, mesh2: Any = None, 
                             mode: str = '1d', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectrum from mesh(es)."""
        self.ensure_initialized()
        
        try:
            from nbodykit.algorithms import FFTPower
            
            if mesh2 is None:
                # Auto power spectrum
                fftpower = FFTPower(mesh1, mode=mode, **kwargs)
            else:
                # Cross power spectrum
                fftpower = FFTPower(mesh1, second=mesh2, mode=mode, **kwargs)
            
            return fftpower.power['k'], fftpower.power['power'].real
            
        except Exception as e:
            raise ProviderError(f"Failed to compute power spectrum: {e}")
    
    def compute_correlation_function(self, catalog: Any, boxsize: Union[float, List[float]],
                                   r_bins: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute correlation function."""
        self.ensure_initialized()
        
        try:
            from nbodykit.algorithms import SurveyDataPairCount, RandomCatalog
            
            # Create random catalog
            nrandom = len(catalog) * 10  # 10x random points
            random_cat = RandomCatalog(nrandom, boxsize=boxsize)
            
            # Compute pair counts
            dd = SurveyDataPairCount('1d', catalog, catalog, r_bins, **kwargs)
            dr = SurveyDataPairCount('1d', catalog, random_cat, r_bins, **kwargs)
            rr = SurveyDataPairCount('1d', random_cat, random_cat, r_bins, **kwargs)
            
            # Landy-Szalay estimator
            xi = (dd.pairs['DD'] - 2*dr.pairs['DR'] + rr.pairs['RR']) / rr.pairs['RR']
            
            return dd.pairs['r'], xi
            
        except Exception as e:
            raise ProviderError(f"Failed to compute correlation function: {e}")
    
    def apply_selection(self, catalog: Any, selection_function: str) -> Any:
        """Apply selection to catalog."""
        self.ensure_initialized()
        
        try:
            # Parse selection function and apply
            return catalog[selection_function]
        except Exception as e:
            raise ProviderError(f"Failed to apply selection: {e}")
    
    def get_cosmology(self, catalog: Any) -> Optional[Any]:
        """Get cosmology from catalog attributes."""
        self.ensure_initialized()
        
        try:
            attrs = self.get_attributes(catalog)
            
            # Try to construct cosmology from attributes
            if 'cosmology' in attrs:
                return attrs['cosmology']
            
            # Try to construct from individual parameters
            cosmo_params = {}
            param_mapping = {
                'H0': ['H0', 'h', 'hubble'],
                'Om0': ['Om0', 'OmegaM', 'omega_m'],
                'Ob0': ['Ob0', 'OmegaB', 'omega_b'],
                'sigma8': ['sigma8'],
                'ns': ['ns', 'n_s']
            }
            
            for param, possible_names in param_mapping.items():
                for name in possible_names:
                    if name in attrs:
                        cosmo_params[param] = attrs[name]
                        break
            
            if cosmo_params:
                from nbodykit.cosmology import Planck15
                # Use Planck15 as base and update with found parameters
                cosmo = Planck15.clone(**cosmo_params)
                return cosmo
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get cosmology: {e}")
            return None
    
    def create_halo_catalog(self, catalog: Any, mass_column: str = 'Mass',
                           position_column: str = 'Position',
                           velocity_column: Optional[str] = None) -> Any:
        """Create halo catalog with additional properties."""
        self.ensure_initialized()
        
        try:
            from nbodykit.lab import ArrayCatalog
            
            # Extract basic properties
            data = {}
            data['Position'] = self.get_column(catalog, position_column)
            data['Mass'] = self.get_column(catalog, mass_column)
            
            if velocity_column and velocity_column in catalog.columns:
                data['Velocity'] = self.get_column(catalog, velocity_column)
            
            # Create new catalog
            halo_cat = ArrayCatalog(data)
            
            # Copy attributes
            for key, value in catalog.attrs.items():
                halo_cat.attrs[key] = value
            
            return halo_cat
            
        except Exception as e:
            raise ProviderError(f"Failed to create halo catalog: {e}")