# lensing_ssc/io/file_handlers.py
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
    analyzer : object, optional # Type hint could be more specific if PatchAnalyzer is defined
        Analyzer instance to save metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Save metadata if analyzer is provided
        if analyzer: # Assuming analyzer has these attributes. Consider type checking or hasattr.
            meta = f.create_group('metadata/parameters')
            meta.attrs['creation_date'] = str(np.datetime64('now'))
            meta.attrs['patch_size_deg'] = getattr(analyzer, 'patch_size', None)
            meta.attrs['xsize'] = getattr(analyzer, 'xsize', None)
            meta.attrs['pixarea_arcmin2'] = getattr(analyzer, 'pixarea_arcmin2', None)
            meta.attrs['lmin'] = getattr(analyzer, 'lmin', None)
            meta.attrs['lmax'] = getattr(analyzer, 'lmax', None)
            meta.attrs['nbin'] = getattr(analyzer, 'nbin', None)
            meta.attrs['epsilon'] = getattr(analyzer, 'epsilon', None)
            
            if hasattr(analyzer, 'ngal_list'):
                meta.create_dataset('ngal_list', data=np.array(analyzer.ngal_list))
            if hasattr(analyzer, 'sl_list'):
                meta.create_dataset('sl_list', data=np.array(analyzer.sl_list))
            if hasattr(analyzer, 'bins'):
                meta.create_dataset('bins', data=analyzer.bins)
            if hasattr(analyzer, 'nu'):
                meta.create_dataset('nu', data=analyzer.nu)
            if hasattr(analyzer, 'l_edges'):
                meta.create_dataset('l_edges', data=analyzer.l_edges)
            if hasattr(analyzer, 'ell'):
                meta.create_dataset('ell', data=analyzer.ell)

        # In the case of processing a single patch file from run_patch_analysis.py
        if not isinstance(next(iter(results.values()), None), dict) or \
           (isinstance(next(iter(results.values()), None), dict) and 
            any(isinstance(v, (int, float)) for v in next(iter(results.values()), {}).values())):
            
            # Create structure for a single file
            _save_single_result(f, results)
        else:
            # Create structure for multiple files (analyze_batch)
            sim_group = f.create_group('simulations')
            for sim_id, sim_results in results.items():
                sim = sim_group.create_group(str(sim_id)) # Ensure sim_id is string for group name
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
    
    for ngal_key, ngal_results in results.items():
        # Ensure ngal_key is a string, or can be safely converted for HDF5 group naming
        ngal_str = f"ngal_{ngal_key}" if not str(ngal_key).startswith("ngal_") else str(ngal_key)

        ngal_group_created = False
        current_spectra_group = spectra
        current_bispectra_group = bispectra
        current_smoothed_group = smoothed

        # Power spectrum data
        if 'cl' in ngal_results:
            if not ngal_group_created:
                ng_ps_parent = spectra.create_group(ngal_str)
                ngal_group_created = True
            else:
                ng_ps_parent = spectra[ngal_str]
            ng_ps_parent.create_dataset('cl', data=ngal_results['cl'])
        
        # Bispectrum data
        if any(k in ngal_results for k in ['equilateral', 'isosceles', 'squeezed']):
            if not ngal_group_created or ngal_str not in bispectra:
                ng_bs_parent = bispectra.create_group(ngal_str)
                if not ngal_group_created: ngal_group_created = True # Should be set if not already
            else:
                ng_bs_parent = bispectra[ngal_str]
            for bs_type in ['equilateral', 'isosceles', 'squeezed']:
                if bs_type in ngal_results:
                    ng_bs_parent.create_dataset(bs_type, data=ngal_results[bs_type])
        
        # SL dependent statistics (assuming ngal_results can also directly contain sl_keys)
        sl_data_found_in_ngal_results = False
        for key, val in ngal_results.items():
            if isinstance(key, (int, float)) and isinstance(val, dict): # SL level
                sl_data_found_in_ngal_results = True
                sl_str = f"sl_{key}"
                # Ensure the ngal_str group exists under smoothed before creating sl_group
                if ngal_str not in smoothed:
                     smoothed.create_group(ngal_str)
                sl_group_parent = smoothed[ngal_str]
                sl_group = sl_group_parent.create_group(sl_str)
                
                _save_detailed_stats(sl_group, val)

        # If SL data was not directly in ngal_results, check if ngal_results itself is the SL data for a single SL
        # This handles the case where results might be structured as results[ngal_key][sl_key]
        # or results[ngal_key] (being the data for a single SL)
        if not sl_data_found_in_ngal_results and isinstance(ngal_results, dict):
            # Check if ngal_results contains stat keys directly, implying it's for one SL implicitly
            if any(stat in ngal_results for stat in ['pdf', 'peaks', 'minima', 'v0', 'v1', 'v2', 'sigma0', 'sigma1']):
                if ngal_str not in smoothed:
                    smoothed.create_group(ngal_str)
                # Here, we might not have an explicit SL key. We can use a default or skip SL grouping.
                # For now, let's assume this structure means the data is directly under ngal_str group.
                _save_detailed_stats(smoothed[ngal_str], ngal_results)


def _save_detailed_stats(parent_group, stats_dict):
    """ Helper to save PDF, Peaks, Minima, Minkowski, Sigma within a given parent HDF5 group. """
    for stat in ['pdf', 'peaks', 'minima']:
        if stat in stats_dict:
            parent_group.create_dataset(stat, data=stats_dict[stat])
    
    minkowski_stats = {mf: stats_dict[mf] for mf in ['v0', 'v1', 'v2'] if mf in stats_dict}
    if minkowski_stats:
        minkowski_group = parent_group.create_group('minkowski')
        for mf_key, mf_data in minkowski_stats.items():
            minkowski_group.create_dataset(mf_key, data=mf_data)
            
    sigma_stats = {s.replace('sigma', ''): stats_dict[s] for s in ['sigma0', 'sigma1'] if s in stats_dict}
    if sigma_stats:
        sigma_group = parent_group.create_group('sigma')
        for s_key, s_data in sigma_stats.items():
            sigma_group.create_dataset(s_key, data=s_data)


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
            # Try to convert numeric keys back to numbers if they were stored as strings by HDF5 implicitly
            # or if our saving logic stringified them (e.g. f"ngal_{ngal_key}")
            processed_key = key
            if key.startswith("ngal_"):
                try:
                    processed_key = int(key.split("_")[1])
                except (ValueError, IndexError):
                    pass # Keep original key if parsing fails
            elif key.startswith("sl_"):
                try:
                    processed_key = float(key.split("_")[1]) # SL might be float
                except (ValueError, IndexError):
                    pass # Keep original key
            else: # For other keys that might have been numeric but stored as strings
                try:
                    if '.' in key:
                        num_key = float(key)
                    else:
                        num_key = int(key)
                    processed_key = num_key 
                except ValueError:
                    pass
                
            if isinstance(item, h5py.Group):
                result[processed_key] = _load_group(item)
            else:
                result[processed_key] = item[()] # h5py dataset to numpy array/value
                
        return result
    
    with h5py.File(file_path, 'r') as f:
        return _load_group(f) 