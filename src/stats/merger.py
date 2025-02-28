import os
import numpy as np
import logging
from typing import List, Dict, Tuple

from utils.utils import find_data_dirs, separate_dirs
from utils.info_extractor import InfoExtractor

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
        self.tiled_dirs, self.bigbox_dirs = separate_dirs(data_dirs)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.overwrite = overwrite

        self.sl = sl
        self.ngal = ngal
        self.oa = opening_angle
        self.zs_list = zs_list

    def run(self) -> None:
        self._run_and_save(is_patch=True, box_type='tiled')
        self._run_and_save(is_patch=True, box_type='bigbox')

    def _run_and_save(self, is_patch: bool = False, box_type: str = 'tiled') -> None:
        suffix = self._generate_suffix(is_patch)
        fname = f"{'patch' if is_patch else 'fullsky'}_stats_{box_type}_{suffix}.npy"
        save_path = os.path.join(self.save_dir, fname)
        
        if os.path.exists(save_path) and not self.overwrite:
            print(f"Stats file {os.path.basename(save_path)} already exists, skipping")
            return
            
        stats = self.merge_stats(is_patch, box_type)
        np.save(save_path, stats)
        print(f"Saved stats to {os.path.basename(save_path)}")

    def merge_stats(self, is_patch: bool = False, box_type: str = 'tiled') -> Dict:
        work_dirs = self.tiled_dirs if box_type == 'tiled' else self.bigbox_dirs
        stats = {}
        
        for zs in self.zs_list:
            stats_zs = []
            for data_dir in work_dirs:
                tmp_stats = self._load_stats(data_dir, zs, is_patch)
                stats_zs.append(tmp_stats)
            
            logging.info(f"Loaded stats for zs={zs}, merged {len(stats_zs)} stats")
            stats_zs = np.vstack(stats_zs)
            stats_zs = self.exclude_outliers(stats_zs)  # Added outlier removal
            
            diags, corr, stds, means = self.total_stats(stats_zs)
            stats[zs] = {
                "diags": diags,
                "corr": corr,
                "stds": stds,
                "means": means
            }
        return stats

    def total_stats(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        diags = np.diag(np.cov(data, rowvar=False))
        corr = np.corrcoef(data, rowvar=False)
        stds = np.std(data, axis=0)
        means = np.mean(data, axis=0)
        return diags, corr, stds, means

    def exclude_outliers(self, data: np.ndarray, n_sigma: int = 5) -> np.ndarray:
        stds = np.std(data, axis=0)
        means = np.mean(data, axis=0)
        mask = np.all(np.abs(data - means) < n_sigma * stds, axis=1)
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
            return f"analysis_zs{zs}_s{info['seed']}_{suffix}.npy"
        return f"fullsky_clpdpm_s{info['seed']}_zs{zs}_{suffix}.npy"

    def _load_stats(self, data_dir: str, zs: float, is_patch: bool = False) -> np.ndarray:
        fname = self._generate_fname(data_dir, zs, is_patch)
        subdir = "analysis_patch" if is_patch else "analysis_fullsky"
        return np.load(os.path.join(data_dir, subdir, fname))

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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    data_dirs = find_data_dirs()
    
    for sl in [2, 5, 8, 10]:
        for ngal in [0, 7, 15, 30, 50]:
            print(f"Processing: ngal={ngal}, sl={sl}")
            stats_merger = StatsMerger(data_dirs, sl, ngal, overwrite=args.overwrite)
            try:
                stats_merger.run()
            except Exception as e:
                logging.error(f"Error processing ngal={ngal}, sl={sl}: {str(e)}")
                continue

if __name__ == "__main__":
    main()