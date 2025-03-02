import numpy as np
import h5py
from pathlib import Path


def save_results_to_hdf5(results, output_path, analyzer=None):
    """
    Save results to HDF5 file with an optimized hierarchical structure.
    
    Parameters
    ----------
    results : dict
        Dictionary with computed statistics
    output_path : str or Path
        Path to the output HDF5 file
    analyzer : PatchAnalyzer, optional
        Analyzer instance to save metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Save metadata if analyzer is provided
        if analyzer:
            meta = f.create_group('metadata/parameters')
            meta.attrs['creation_date'] = str(np.datetime64('now'))
            meta.attrs['patch_size_deg'] = analyzer.patch_size
            meta.attrs['xsize'] = analyzer.xsize
            meta.attrs['pixarea_arcmin2'] = analyzer.pixarea_arcmin2
            meta.attrs['lmin'] = analyzer.lmin
            meta.attrs['lmax'] = analyzer.lmax
            meta.attrs['nbin'] = analyzer.nbin
            meta.attrs['epsilon'] = analyzer.epsilon
            
            meta.create_dataset('ngal_list', data=np.array(analyzer.ngal_list))
            meta.create_dataset('sl_list', data=np.array(analyzer.sl_list))
            meta.create_dataset('bins', data=analyzer.bins)
            meta.create_dataset('nu', data=analyzer.nu)
            meta.create_dataset('l_edges', data=analyzer.l_edges)
            meta.create_dataset('ell', data=analyzer.ell)

        # In the case of processing a single patch file from run_patch_analysis.py
        if not isinstance(next(iter(results.values()), None), dict) or \
           (isinstance(next(iter(results.values()), None), dict) and 
            any(isinstance(v, (int, float)) for v in next(iter(results.values()), {}))):
            
            # Create structure for a single file
            _save_single_result(f, results)
        else:
            # Create structure for multiple files (analyze_batch)
            sim_group = f.create_group('simulations')
            for sim_id, sim_results in results.items():
                sim = sim_group.create_group(sim_id)
                _save_single_result(sim, sim_results)


def _save_single_result(parent, results):
    """
    Helper function to save a single result structure.
    
    Parameters
    ----------
    parent : h5py.Group
        Parent HDF5 group
    results : dict
        Result dictionary
    """
    # Create groups for different statistical measures
    spectra = parent.create_group('power_spectra')
    bispectra = parent.create_group('bispectra') 
    smoothed = parent.create_group('smoothed_statistics')
    
    for ngal, ngal_results in results.items():
        ngal_str = f"ngal_{ngal}"
        
        # Power spectrum data
        if 'cl' in ngal_results:
            ng_ps = spectra.create_group(ngal_str)
            ng_ps.create_dataset('cl', data=ngal_results['cl'])
        
        # Bispectrum data
        if any(k in ngal_results for k in ['equilateral', 'isosceles', 'squeezed']):
            ng_bs = bispectra.create_group(ngal_str)
            for bs_type in ['equilateral', 'isosceles', 'squeezed']:
                if bs_type in ngal_results:
                    ng_bs.create_dataset(bs_type, data=ngal_results[bs_type])
        
        # SL dependent statistics
        for key, val in ngal_results.items():
            if isinstance(key, (int, float)) and isinstance(val, dict):
                sl = key
                sl_str = f"sl_{sl}"
                sl_group = smoothed.create_group(f"{ngal_str}/{sl_str}")
                
                # PDF and Peak statistics
                for stat in ['pdf', 'peaks', 'minima']:
                    if stat in val:
                        sl_group.create_dataset(stat, data=val[stat])
                
                # Minkowski functionals group
                minkowski = sl_group.create_group('minkowski')
                for mf in ['v0', 'v1', 'v2']:
                    if mf in val:
                        minkowski.create_dataset(mf, data=val[mf])
                
                # Sigma values group
                sigma = sl_group.create_group('sigma')
                for s in ['sigma0', 'sigma1']:
                    if s in val:
                        sigma.create_dataset(s.replace('sigma', ''), data=val[s])


def load_results_from_hdf5(file_path: Path) -> dict:
    """
    Load results from HDF5 file.
    
    Parameters
    ----------
    file_path : Path
        Path to HDF5 file
        
    Returns
    -------
    dict
        Loaded results
    """
    def _load_group(group):
        result = {}
        for key, item in group.items():
            # Try to convert numeric keys back to numbers
            try:
                if '.' in key:
                    numeric_key = float(key)
                else:
                    numeric_key = int(key)
                key = numeric_key
            except ValueError:
                pass
                
            if isinstance(item, h5py.Group):
                # Recursively load group
                result[key] = _load_group(item)
            else:
                # Load dataset
                result[key] = item[()]
                
        return result
    
    with h5py.File(file_path, 'r') as f:
        return _load_group(f)