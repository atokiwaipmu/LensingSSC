"""
NBBodyKit provider implementation.

This module provides N-body simulation data access and analysis using nbodykit
with lazy loading, efficient data handling, and comprehensive catalog operations
for cosmological simulations.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import logging
import gc

from ..interfaces.data_interface import CatalogProvider
from ..base.exceptions import ProviderError, DataError
from .base_provider import LazyProvider, CachedProvider


class NbodykitProvider(LazyProvider, CachedProvider, CatalogProvider):
    """Provider for N-body simulation data using nbodykit.
    
    This provider offers:
    - Lazy loading of nbodykit and related packages
    - Efficient catalog data access and manipulation
    - Memory-optimized data processing
    - Cosmological analysis tools
    - Mesh generation and FFT operations
    - Power spectrum and correlation function calculations
    """
    
    def __init__(self, cache_size: int = 10, cache_ttl: Optional[float] = 7200):
        """Initialize nbodykit provider.
        
        Parameters
        ----------
        cache_size : int
            Maximum number of cached catalogs/meshes
        cache_ttl : float, optional
            Cache time-to-live in seconds (default: 2 hours)
        """
        LazyProvider.__init__(self)
        CachedProvider.__init__(self, cache_size=cache_size, cache_ttl=cache_ttl)
        
        self._nbodykit = None
        self._nbodykit_lab = None
        self._algorithms = None
        self._cosmology = None
        self._transform = None
        
        # Catalog type mappings
        self._catalog_types = {
            'bigfile': 'BigFileCatalog',
            'csv': 'CSVCatalog',
            'fits': 'FITSCatalog',
            'hdf5': 'HDFCatalog',
            'array': 'ArrayCatalog',
            'uniform': 'UniformCatalog',
            'random': 'RandomCatalog',
        }
    
    @property
    def name(self) -> str:
        """Provider name."""
        return "NbodykitProvider"
    
    @property
    def version(self) -> str:
        """Provider version."""
        return "1.0.0"
    
    def _check_dependencies(self) -> None:
        """Check if nbodykit and related packages are available."""
        try:
            self._nbodykit = self._lazy_import('nbodykit')
            self._nbodykit_lab = self._lazy_import('nbodykit.lab', 'lab')
            self._algorithms = self._lazy_import('nbodykit.algorithms', 'algorithms')
            self._cosmology = self._lazy_import('nbodykit.cosmology', 'cosmology')
            self._transform = self._lazy_import('nbodykit.transform', 'transform')
        except Exception as e:
            raise ImportError(f"nbodykit is required for NbodykitProvider: {e}")
    
    def _initialize_backend(self, **kwargs) -> None:
        """Initialize nbodykit backend."""
        self._check_dependencies()
        
        # Set default cosmology if provided
        default_cosmo = kwargs.get('default_cosmology', None)
        if default_cosmo:
            self._default_cosmology = default_cosmo
            self._logger.debug(f"Set default cosmology: {type(default_cosmo).__name__}")
        
        # Configure memory management
        chunk_size = kwargs.get('chunk_size', 100000)
        if hasattr(self._nbodykit, 'set_options'):
            try:
                self._nbodykit.set_options(chunk_size=chunk_size)
                self._logger.debug(f"Set nbodykit chunk size: {chunk_size}")
            except Exception as e:
                self._logger.warning(f"Could not set nbodykit options: {e}")
    
    def _get_backend_info(self) -> Dict[str, Any]:
        """Get backend-specific information."""
        info = super()._get_backend_info()
        
        if self._nbodykit is not None:
            info.update({
                'nbodykit_version': getattr(self._nbodykit, '__version__', 'unknown'),
                'available_catalogs': list(self._catalog_types.keys()),
                'available_algorithms': self._get_available_algorithms(),
                'mpi_enabled': self._check_mpi_support(),
            })
        
        return info
    
    def _get_available_algorithms(self) -> List[str]:
        """Get list of available algorithms."""
        if self._algorithms is None:
            return []
        
        algorithms = []
        for attr in dir(self._algorithms):
            if not attr.startswith('_') and hasattr(getattr(self._algorithms, attr), '__call__'):
                algorithms.append(attr)
        return algorithms
    
    def _check_mpi_support(self) -> bool:
        """Check if MPI support is available."""
        try:
            from mpi4py import MPI
            return True
        except ImportError:
            return False
    
    def read_catalog(self, path: Union[str, Path], dataset: str = None, 
                    catalog_type: str = 'auto', **kwargs) -> Any:
        """Read catalog from file.
        
        Parameters
        ----------
        path : str or Path
            Path to catalog file or directory
        dataset : str, optional
            Dataset name within file (required for BigFile)
        catalog_type : str
            Type of catalog ('auto', 'bigfile', 'csv', 'fits', 'hdf5')
        **kwargs
            Additional arguments for catalog creation
            
        Returns
        -------
        nbodykit.base.catalog.CatalogSource
            Loaded catalog
        """
        self.ensure_initialized()
        self._track_usage()
        
        path_obj = Path(path)
        cache_key = f"catalog_{path_obj}_{dataset}_{catalog_type}_{hash(frozenset(kwargs.items()))}"
        
        # Check cache first
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            self._logger.debug(f"Retrieved catalog from cache: {path}")
            return cached_result
        
        try:
            # Auto-detect catalog type if needed
            if catalog_type == 'auto':
                catalog_type = self._detect_catalog_type(path_obj)
            
            # Get catalog class
            catalog_class_name = self._catalog_types.get(catalog_type)
            if not catalog_class_name:
                raise ProviderError(f"Unknown catalog type: {catalog_type}")
            
            catalog_class = getattr(self._nbodykit_lab, catalog_class_name)
            
            # Prepare arguments based on catalog type
            if catalog_type == 'bigfile':
                if dataset is None:
                    raise ProviderError("dataset parameter required for BigFile catalogs")
                catalog = catalog_class(str(path_obj), dataset=dataset, **kwargs)
            
            elif catalog_type in ['csv', 'fits', 'hdf5']:
                catalog = catalog_class(str(path_obj), **kwargs)
            
            else:
                raise ProviderError(f"Unsupported catalog type for reading: {catalog_type}")
            
            # Cache the result
            self._cache_set(cache_key, catalog)
            
            self._logger.debug(f"Successfully loaded {catalog_type} catalog: {path}")
            return catalog
            
        except Exception as e:
            raise ProviderError(f"Failed to read catalog from {path}: {e}")
    
    def _detect_catalog_type(self, path: Path) -> str:
        """Auto-detect catalog type from path."""
        if path.is_dir():
            # Check for BigFile structure
            if (path / 'Header').exists() or any(path.glob('*/Header')):
                return 'bigfile'
        
        suffix = path.suffix.lower()
        if suffix == '.csv':
            return 'csv'
        elif suffix in ['.fits', '.fit']:
            return 'fits'
        elif suffix in ['.h5', '.hdf5']:
            return 'hdf5'
        
        # Default fallback
        return 'bigfile'
    
    def get_column(self, catalog: Any, column: str, 
                  start: Optional[int] = None, end: Optional[int] = None,
                  chunk_size: Optional[int] = None) -> np.ndarray:
        """Get column from catalog with optional slicing.
        
        Parameters
        ----------
        catalog : nbodykit catalog
            Source catalog
        column : str
            Column name
        start, end : int, optional
            Slice indices
        chunk_size : int, optional
            Chunk size for memory management
            
        Returns
        -------
        np.ndarray
            Column data
        """
        self.ensure_initialized()
        self._track_usage()
        
        try:
            if column not in catalog.columns:
                available_columns = list(catalog.columns)
                raise DataError(f"Column '{column}' not found. Available columns: {available_columns}")
            
            # Handle slicing
            if start is not None or end is not None:
                if chunk_size and (end is None or end - (start or 0) > chunk_size):
                    # Process in chunks for large slices
                    return self._get_column_chunked(catalog, column, start, end, chunk_size)
                else:
                    return catalog[column][start:end].compute()
            else:
                # Full column
                if chunk_size:
                    return self._get_column_chunked(catalog, column, None, None, chunk_size)
                else:
                    return catalog[column].compute()
                    
        except Exception as e:
            raise ProviderError(f"Failed to get column {column}: {e}")
    
    def _get_column_chunked(self, catalog: Any, column: str, 
                           start: Optional[int], end: Optional[int],
                           chunk_size: int) -> np.ndarray:
        """Get column data in chunks to manage memory."""
        total_size = catalog.size
        
        if start is None:
            start = 0
        if end is None:
            end = total_size
        
        # Initialize result array
        result_size = end - start
        first_chunk = catalog[column][start:min(start + chunk_size, end)].compute()
        result = np.empty(result_size, dtype=first_chunk.dtype)
        
        # Fill result array chunk by chunk
        current_pos = 0
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end)
            chunk_data = catalog[column][chunk_start:chunk_end].compute()
            
            chunk_size_actual = len(chunk_data)
            result[current_pos:current_pos + chunk_size_actual] = chunk_data
            current_pos += chunk_size_actual
            
            # Periodic garbage collection for large datasets
            if current_pos % (chunk_size * 5) == 0:
                gc.collect()
        
        return result
    
    def get_attributes(self, catalog: Any) -> Dict[str, Any]:
        """Get catalog attributes/metadata.
        
        Parameters
        ----------
        catalog : nbodykit catalog
            Source catalog
            
        Returns
        -------
        Dict[str, Any]
            Catalog attributes
        """
        self.ensure_initialized()
        
        try:
            return dict(catalog.attrs)
        except Exception as e:
            raise ProviderError(f"Failed to get catalog attributes: {e}")
    
    def get_size(self, catalog: Any) -> int:
        """Get catalog size (number of entries).
        
        Parameters
        ----------
        catalog : nbodykit catalog
            Source catalog
            
        Returns
        -------
        int
            Number of entries in catalog
        """
        self.ensure_initialized()
        
        try:
            return catalog.size
        except Exception as e:
            raise ProviderError(f"Failed to get catalog size: {e}")
    
    def create_mesh(self, catalog: Any, boxsize: Union[float, List[float]], 
                   nmesh: Union[int, List[int]], position: str = 'Position',
                   weight: Optional[str] = None, **kwargs) -> Any:
        """Create mesh from catalog.
        
        Parameters
        ----------
        catalog : nbodykit catalog
            Source catalog
        boxsize : float or list
            Box size(s) in Mpc/h
        nmesh : int or list
            Number of mesh cells per dimension
        position : str
            Position column name
        weight : str, optional
            Weight column name
        **kwargs
            Additional mesh parameters
            
        Returns
        -------
        nbodykit.base.mesh.MeshSource
            Created mesh
        """
        self.ensure_initialized()
        self._track_usage()
        
        cache_key = f"mesh_{id(catalog)}_{boxsize}_{nmesh}_{position}_{weight}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            mesh_kwargs = {
                'boxsize': boxsize,
                'Nmesh': nmesh,
                'position': position,
                'dtype': kwargs.get('dtype', 'f4'),
                'compensated': kwargs.get('compensated', True),
                'interlaced': kwargs.get('interlaced', True),
            }
            
            if weight is not None:
                mesh_kwargs['weight'] = weight
            
            # Remove None values
            mesh_kwargs = {k: v for k, v in mesh_kwargs.items() if v is not None}
            
            mesh = catalog.to_mesh(**mesh_kwargs)
            
            # Cache the result
            self._cache_set(cache_key, mesh)
            
            self._logger.debug(f"Created mesh with Nmesh={nmesh}, boxsize={boxsize}")
            return mesh
            
        except Exception as e:
            raise ProviderError(f"Failed to create mesh: {e}")
    
    def mesh_to_real_field(self, mesh: Any) -> np.ndarray:
        """Convert mesh to real field.
        
        Parameters
        ----------
        mesh : nbodykit mesh
            Source mesh
            
        Returns
        -------
        np.ndarray
            Real field data
        """
        self.ensure_initialized()
        
        try:
            real_field = mesh.to_real_field()
            return real_field
        except Exception as e:
            raise ProviderError(f"Failed to convert mesh to real field: {e}")
    
    def mesh_to_complex_field(self, mesh: Any) -> Any:
        """Convert mesh to complex field.
        
        Parameters
        ----------
        mesh : nbodykit mesh
            Source mesh
            
        Returns
        -------
        nbodykit complex field
            Complex field
        """
        self.ensure_initialized()
        
        try:
            complex_field = mesh.to_complex_field()
            return complex_field
        except Exception as e:
            raise ProviderError(f"Failed to convert mesh to complex field: {e}")
    
    def compute_power_spectrum(self, mesh1: Any, mesh2: Any = None, 
                             mode: str = '1d', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectrum from mesh(es).
        
        Parameters
        ----------
        mesh1 : nbodykit mesh
            First mesh (or only mesh for auto-power)
        mesh2 : nbodykit mesh, optional
            Second mesh for cross-power spectrum
        mode : str
            Power spectrum mode ('1d', '2d')
        **kwargs
            Additional FFTPower arguments
            
        Returns
        -------
        tuple
            (k, power) arrays
        """
        self.ensure_initialized()
        self._track_usage()
        
        cache_key = f"power_{id(mesh1)}_{id(mesh2) if mesh2 else None}_{mode}_{hash(frozenset(kwargs.items()))}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            fftpower_class = getattr(self._algorithms, 'FFTPower')
            
            power_kwargs = {
                'mode': mode,
                'dk': kwargs.get('dk', None),
                'kmin': kwargs.get('kmin', 0.0),
                'kmax': kwargs.get('kmax', None),
                'poles': kwargs.get('poles', None),
            }
            
            # Remove None values
            power_kwargs = {k: v for k, v in power_kwargs.items() if v is not None}
            
            if mesh2 is None:
                # Auto power spectrum
                fftpower = fftpower_class(mesh1, **power_kwargs)
            else:
                # Cross power spectrum
                fftpower = fftpower_class(mesh1, second=mesh2, **power_kwargs)
            
            # Extract results
            k = fftpower.power['k']
            power = fftpower.power['power'].real
            
            result = (k, power)
            self._cache_set(cache_key, result)
            
            return result
            
        except Exception as e:
            raise ProviderError(f"Failed to compute power spectrum: {e}")
    
    def compute_correlation_function(self, catalog: Any, edges: np.ndarray,
                                   boxsize: Optional[Union[float, List[float]]] = None,
                                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute correlation function.
        
        Parameters
        ----------
        catalog : nbodykit catalog
            Source catalog
        edges : np.ndarray
            Bin edges for correlation function
        boxsize : float or list, optional
            Box size for periodic boundary conditions
        **kwargs
            Additional correlation function arguments
            
        Returns
        -------
        tuple
            (r, xi) arrays
        """
        self.ensure_initialized()
        self._track_usage()
        
        try:
            # For survey-style correlation function
            if boxsize is not None:
                # Use pair counting approach
                return self._compute_survey_correlation(catalog, edges, boxsize, **kwargs)
            else:
                # Use FFT-based approach for periodic box
                return self._compute_fft_correlation(catalog, edges, **kwargs)
                
        except Exception as e:
            raise ProviderError(f"Failed to compute correlation function: {e}")
    
    def _compute_survey_correlation(self, catalog: Any, edges: np.ndarray,
                                  boxsize: Union[float, List[float]], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute correlation function using pair counting."""
        try:
            # Create random catalog
            nrandom = kwargs.get('nrandom', len(catalog) * 5)
            random_catalog_class = getattr(self._nbodykit_lab, 'RandomCatalog')
            random_cat = random_catalog_class(nrandom, boxsize=boxsize)
            
            # Compute pair counts
            paircount_class = getattr(self._algorithms, 'SurveyDataPairCount')
            
            dd = paircount_class('1d', catalog, catalog, edges, **kwargs)
            dr = paircount_class('1d', catalog, random_cat, edges, **kwargs)
            rr = paircount_class('1d', random_cat, random_cat, edges, **kwargs)
            
            # Landy-Szalay estimator
            xi = (dd.pairs['DD'] - 2*dr.pairs['DR'] + rr.pairs['RR']) / rr.pairs['RR']
            r = dd.pairs['r']
            
            return r, xi
            
        except Exception as e:
            raise ProviderError(f"Failed to compute survey correlation function: {e}")
    
    def _compute_fft_correlation(self, catalog: Any, edges: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute correlation function using FFT method."""
        try:
            # This is a simplified implementation
            # Full implementation would require creating mesh and computing correlation via FFT
            raise NotImplementedError("FFT-based correlation function not yet implemented")
            
        except Exception as e:
            raise ProviderError(f"Failed to compute FFT correlation function: {e}")
    
    def apply_selection(self, catalog: Any, selection: Union[str, Callable]) -> Any:
        """Apply selection to catalog.
        
        Parameters
        ----------
        catalog : nbodykit catalog
            Source catalog
        selection : str or callable
            Selection criteria (string expression or function)
            
        Returns
        -------
        nbodykit catalog
            Filtered catalog
        """
        self.ensure_initialized()
        
        try:
            if isinstance(selection, str):
                # String-based selection
                return catalog[selection]
            elif callable(selection):
                # Function-based selection
                mask = selection(catalog)
                return catalog[mask]
            else:
                raise ValueError("Selection must be string or callable")
                
        except Exception as e:
            raise ProviderError(f"Failed to apply selection: {e}")
    
    def get_cosmology(self, catalog: Any) -> Optional[Any]:
        """Get cosmology from catalog attributes.
        
        Parameters
        ----------
        catalog : nbodykit catalog
            Source catalog
            
        Returns
        -------
        nbodykit cosmology or None
            Cosmology object if available
        """
        self.ensure_initialized()
        
        try:
            attrs = self.get_attributes(catalog)
            
            # Check if cosmology is directly stored
            if 'cosmology' in attrs:
                return attrs['cosmology']
            
            # Try to construct from parameters
            cosmo_params = self._extract_cosmology_params(attrs)
            if cosmo_params:
                return self._create_cosmology(**cosmo_params)
            
            return None
            
        except Exception as e:
            self._logger.warning(f"Failed to get cosmology: {e}")
            return None
    
    def _extract_cosmology_params(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cosmology parameters from attributes."""
        cosmo_params = {}
        
        # Parameter mapping
        param_mapping = {
            'H0': ['H0', 'h', 'hubble'],
            'Om0': ['Om0', 'OmegaM', 'omega_m', 'Omega_m'],
            'Ob0': ['Ob0', 'OmegaB', 'omega_b', 'Omega_b'],
            'sigma8': ['sigma8'],
            'ns': ['ns', 'n_s'],
            'w0_fld': ['w0', 'w'],
            'wa_fld': ['wa'],
        }
        
        for param, possible_names in param_mapping.items():
            for name in possible_names:
                if name in attrs:
                    cosmo_params[param] = attrs[name]
                    break
        
        return cosmo_params
    
    def _create_cosmology(self, **params) -> Any:
        """Create cosmology object from parameters."""
        try:
            # Use Planck15 as base cosmology
            planck15_class = getattr(self._cosmology, 'Planck15')
            base_cosmo = planck15_class
            
            # Clone with new parameters
            if hasattr(base_cosmo, 'clone'):
                return base_cosmo.clone(**params)
            else:
                # Fallback to creating new cosmology
                cosmology_class = getattr(self._cosmology, 'Cosmology')
                return cosmology_class(**params)
                
        except Exception as e:
            self._logger.warning(f"Failed to create cosmology: {e}")
            return None
    
    def create_halo_catalog(self, catalog: Any, mass_column: str = 'Mass',
                           position_column: str = 'Position',
                           velocity_column: Optional[str] = None,
                           **kwargs) -> Any:
        """Create halo catalog with additional properties.
        
        Parameters
        ----------
        catalog : nbodykit catalog
            Source catalog
        mass_column : str
            Mass column name
        position_column : str
            Position column name
        velocity_column : str, optional
            Velocity column name
        **kwargs
            Additional catalog properties
            
        Returns
        -------
        nbodykit catalog
            Halo catalog
        """
        self.ensure_initialized()
        self._track_usage()
        
        try:
            array_catalog_class = getattr(self._nbodykit_lab, 'ArrayCatalog')
            
            # Extract required data
            data = {}
            data['Position'] = self.get_column(catalog, position_column)
            data['Mass'] = self.get_column(catalog, mass_column)
            
            if velocity_column and velocity_column in catalog.columns:
                data['Velocity'] = self.get_column(catalog, velocity_column)
            
            # Add any additional columns specified in kwargs
            for key, column_name in kwargs.items():
                if isinstance(column_name, str) and column_name in catalog.columns:
                    data[key] = self.get_column(catalog, column_name)
            
            # Create new catalog
            halo_cat = array_catalog_class(data)
            
            # Copy attributes
            for key, value in catalog.attrs.items():
                halo_cat.attrs[key] = value
            
            return halo_cat
            
        except Exception as e:
            raise ProviderError(f"Failed to create halo catalog: {e}")
    
    def cleanup_memory(self) -> None:
        """Cleanup memory by clearing caches and forcing garbage collection."""
        self._cache_clear()
        gc.collect()
        self._logger.debug("Cleaned up nbodykit provider memory")