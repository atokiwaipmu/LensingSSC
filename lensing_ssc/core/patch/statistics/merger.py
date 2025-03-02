import os
import numpy as np
import h5py
import logging
from typing import List, Dict, Tuple, Union
from pathlib import Path

from lensing_ssc.utils import PathHandler, InfoExtractor

class StatsMerger:
    def __init__(self, 
                 data_dirs: List[str], 
                 sl: int, 
                 ngal: int,
                 opening_angle: int = 10,
                 zs_list: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5],
                 save_dir: str = "/lustre/work/akira.tokiwa/Projects/LensingSSC/output",
                 overwrite: bool = False):
        self.data_dirs = data_dirs
        self.tiled_dirs, self.bigbox_dirs = PathHandler.categorize_data_dirs_by_box_type(data_dirs)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite

        self.sl = sl
        self.ngal = ngal
        self.oa = opening_angle
        self.zs_list = zs_list
        
        # Define the type and order of statistics
        self.stats_reorder = StatsReorder()

    def run(self) -> None:
        self._run_and_save(is_patch=True, box_type='tiled')
        self._run_and_save(is_patch=True, box_type='bigbox')

    def _run_and_save(self, is_patch: bool = False, box_type: str = 'tiled') -> None:
        suffix = self._generate_suffix(is_patch)
        fname = f"{'patch' if is_patch else 'fullsky'}_stats_{box_type}_{suffix}.h5"
        save_path = self.save_dir / fname
        
        if save_path.exists() and not self.overwrite:
            logging.info(f"Stats file {save_path.name} already exists, skipping")
            return
            
        stats = self.merge_stats(is_patch, box_type)
        self._save_to_hdf5(stats, save_path)
        logging.info(f"Saved stats to {save_path.name}")

    def _save_to_hdf5(self, stats: Dict, save_path: Path) -> None:
        """Save statistics in HDF5 format"""
        with h5py.File(save_path, 'w') as f:
            # Save metadata
            meta = f.create_group('metadata')
            meta.attrs['creation_date'] = str(np.datetime64('now'))
            meta.attrs['sl'] = self.sl
            meta.attrs['ngal'] = self.ngal
            meta.attrs['opening_angle'] = self.oa
            
            meta.create_dataset('zs_list', data=np.array(self.zs_list))
            
            # Information about the type and order of statistics
            stats_meta = meta.create_group('statistics_info')
            for i, stat_name in enumerate(self.stats_reorder.stats_desired_order):
                start, end = self.stats_reorder.stats_name_indices[stat_name]
                stat_info = stats_meta.create_group(f"{i:02d}_{stat_name}")
                stat_info.attrs['start_index'] = start
                stat_info.attrs['end_index'] = end
                stat_info.attrs['length'] = end - start
            
            # Save statistics for each redshift
            stats_group = f.create_group('statistics')
            for zs, zs_stats in stats.items():
                zs_group = stats_group.create_group(f"zs_{zs}")
                
                # Save correlation matrix etc.
                for stat_type, data in zs_stats.items():
                    zs_group.create_dataset(stat_type, data=data)
                
                # Save both original and permuted data
                if 'raw_data' in zs_stats:
                    raw_data = zs_group.create_group('raw_data')
                    
                    # Group by statistic type
                    for stat_name in self.stats_reorder.stats_desired_order:
                        start, end = self.stats_reorder.stats_name_indices[stat_name]
                        stat_data = zs_stats['raw_data'][:, start:end]
                        raw_data.create_dataset(stat_name, data=stat_data)

    def merge_stats(self, is_patch: bool = False, box_type: str = 'tiled') -> Dict:
        work_dirs = self.tiled_dirs if box_type == 'tiled' else self.bigbox_dirs
        stats = {}
        
        for zs in self.zs_list:
            stats_zs = []
            for data_dir in work_dirs:
                try:
                    tmp_stats = self._load_stats(data_dir, zs, is_patch)
                    stats_zs.append(tmp_stats)
                except Exception as e:
                    logging.warning(f"Failed to load stats from {data_dir} for zs={zs}: {e}")
            
            if not stats_zs:
                logging.warning(f"No valid stats found for zs={zs}")
                continue
                
            logging.info(f"Loaded stats for zs={zs}, merged {len(stats_zs)} stats")
            stats_zs_array = np.vstack(stats_zs)
            filtered_stats = self.exclude_outliers(stats_zs_array)  # Remove outliers
            
            diags, corr, stds, means = self.total_stats(filtered_stats)
            
            # Permute statistics
            permuted_stats = filtered_stats[:, self.stats_reorder.permuted_indices]
            p_diags, p_corr, p_stds, p_means = self.total_stats(permuted_stats)
            
            stats[zs] = {
                "diags": diags,
                "corr": corr,
                "stds": stds,
                "means": means,
                "permuted_diags": p_diags,
                "permuted_corr": p_corr,
                "permuted_stds": p_stds,
                "permuted_means": p_means,
                "raw_data": filtered_stats  # Save raw data as well
            }
        return stats

    def total_stats(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        diags = np.diag(np.cov(data, rowvar=False))
        corr = np.corrcoef(data, rowvar=False)
        stds = np.std(data, axis=0)
        means = np.mean(data, axis=0)
        return diags, corr, stds, means

    def exclude_outliers(self, data: np.ndarray, n_sigma: int = 5) -> np.ndarray:
        """Exclude outliers exceeding n_sigma"""
        stds = np.std(data, axis=0)
        means = np.mean(data, axis=0)
        mask = np.all(np.abs(data - means) < n_sigma * stds, axis=1)
        logging.info(f"Excluded {len(data) - np.sum(mask)} outliers out of {len(data)} samples")
        return data[mask]

    def _generate_suffix(self, is_patch: bool = False) -> str:
        suffix = f"oa{self.oa}_" if is_patch else ""
        suffix += f"noiseless_sl{self.sl}"
        if self.ngal != 0:
            suffix = suffix.replace("noiseless", f"ngal{self.ngal}")
        return suffix

    def _generate_fname(self, datadir: str, zs: float, is_patch: bool = False) -> str:
        info = InfoExtractor.extract_info_from_path(datadir)
        suffix = self._generate_suffix(is_patch)
        if is_patch:
            return f"analysis_zs{zs}_s{info['seed']}_{suffix}.h5"  # .npy -> .h5
        return f"fullsky_clpdpm_s{info['seed']}_zs{zs}_{suffix}.h5"  # .npy -> .h5

    def _load_stats(self, data_dir: str, zs: float, is_patch: bool = False) -> np.ndarray:
        fname = self._generate_fname(data_dir, zs, is_patch)
        subdir = "analysis_patch" if is_patch else "analysis_fullsky"
        file_path = Path(data_dir) / subdir / fname
        
        # Load HDF5 file
        if file_path.suffix == '.h5':
            return self._load_from_hdf5(file_path)
        # Load legacy numpy file
        elif file_path.with_suffix('.npy').exists():
            return np.load(file_path.with_suffix('.npy'))
        else:
            raise FileNotFoundError(f"Statistics file not found: {file_path}")

    def _load_from_hdf5(self, file_path: Path) -> np.ndarray:
        """Load statistics from HDF5 file"""
        with h5py.File(file_path, 'r') as f:
            if 'statistics' in f:
                # Load from new structure
                stats_combined = []
                
                # Load data for each statistic type in order
                for stat_name in self.stats_reorder.stats_desired_order:
                    if f'statistics/{stat_name}' in f:
                        stat_data = f[f'statistics/{stat_name}'][:]
                        stats_combined.append(stat_data)
                
                return np.hstack(stats_combined)
            else:
                # Support loading from old structure (legacy support)
                # Needs modification to match the actual structure
                logging.warning(f"Loading from legacy HDF5 format: {file_path}")
                data = []
                
                # Power spectrum
                if 'power_spectra' in f:
                    for ngal_group in f['power_spectra']:
                        if f'power_spectra/{ngal_group}/cl' in f:
                            data.append(f[f'power_spectra/{ngal_group}/cl'][:])
                
                # Bispectrum
                if 'bispectra' in f:
                    for ngal_group in f['bispectra']:
                        for bs_type in ['equilateral', 'isosceles', 'squeezed']:
                            if f'bispectra/{ngal_group}/{bs_type}' in f:
                                data.append(f[f'bispectra/{ngal_group}/{bs_type}'][:])
                
                # Smoothed statistics
                if 'smoothed_statistics' in f:
                    for ngal_group in f['smoothed_statistics']:
                        # PDF
                        if f'smoothed_statistics/{ngal_group}/pdf' in f:
                            data.append(f[f'smoothed_statistics/{ngal_group}/pdf'][:])
                        
                        # Peaks
                        if f'smoothed_statistics/{ngal_group}/peaks' in f:
                            data.append(f[f'smoothed_statistics/{ngal_group}/peaks'][:])
                        
                        # Minima
                        if f'smoothed_statistics/{ngal_group}/minima' in f:
                            data.append(f[f'smoothed_statistics/{ngal_group}/minima'][:])
                        
                        # Minkowski functions
                        if f'smoothed_statistics/{ngal_group}/minkowski' in f:
                            for mf_type in ['v0', 'v1', 'v2']:
                                if f'smoothed_statistics/{ngal_group}/minkowski/{mf_type}' in f:
                                    data.append(f[f'smoothed_statistics/{ngal_group}/minkowski/{mf_type}'][:])
                        
                        # Sigma (variance related)
                        if f'smoothed_statistics/{ngal_group}/sigma' in f:
                            for sigma_level in ['0', '1']:
                                if f'smoothed_statistics/{ngal_group}/sigma/{sigma_level}' in f:
                                    data.append(f[f'smoothed_statistics/{ngal_group}/sigma/{sigma_level}'][:])

                return np.hstack(data)


class StatsReorder:
    def __init__(self):
        self._init_stats_indices()
        self.permuted_indices = self.prepare_permuted_indices()
        
    def _init_stats_indices(self):
        self.stats_name_indices = {
            'angular power spectrum': [45, 60],
            'squeezed bispectrum': [30, 45],
            'isosceles bispectrum': [15, 30],
            'equilateral bispectrum': [0, 15],
            'PDF': [60, 75],
            'Peak': [75, 90],
            'Minima': [90, 105],
            'area(MFs)': [105, 120],
            'perimeter(MFs)': [120, 135],
            'genus(MFs)': [135, 150],
            'Skewness_0': [150, 151],
            'Skewness_1': [151, 152],
            'Skewness_2': [152, 153],
            'Kurtosis_0': [153, 154],
            'Kurtosis_1': [154, 155],
            'Kurtosis_2': [155, 156],
            'Kurtosis_3': [156, 157],
        }

        self.stats_desired_order = [
            'angular power spectrum',
            'squeezed bispectrum',
            'isosceles bispectrum',
            'equilateral bispectrum',
            'Skewness_0', 'Skewness_1', 'Skewness_2',
            'Kurtosis_0', 'Kurtosis_1', 'Kurtosis_2', 'Kurtosis_3',
            'PDF', 'Peak', 'Minima',
            'area(MFs)', 'perimeter(MFs)', 'genus(MFs)'
        ]

    def prepare_permuted_indices(self) -> List[int]:
        permuted_indices = []
        for stat in self.stats_desired_order:
            if stat not in self.stats_name_indices:
                raise ValueError(f"Statistic '{stat}' not found in stats_name_indices mapping.")
            start, end = self.stats_name_indices[stat]
            permuted_indices.extend(list(range(start, end)))
        return permuted_indices


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Merge statistics from multiple simulations')
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--output-dir", type=str, 
                        default="/lustre/work/akira.tokiwa/Projects/LensingSSC/output",
                        help="Directory to save merged statistics")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    data_dirs = PathHandler.find_data_dirs()
    
    for sl in [2, 5, 8, 10]:
        for ngal in [0, 7, 15, 30, 50]:
            logging.info(f"Processing: ngal={ngal}, sl={sl}")
            stats_merger = StatsMerger(
                data_dirs, sl, ngal, 
                save_dir=args.output_dir,
                overwrite=args.overwrite
            )
            try:
                stats_merger.run()
            except Exception as e:
                logging.error(f"Error processing ngal={ngal}, sl={sl}: {str(e)}")
                logging.exception("Detailed error:")
                continue


if __name__ == "__main__":
    main()