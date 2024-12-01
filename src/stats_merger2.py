
from ast import arg
import os
import numpy as np
import healpy as hp
import logging

from src.utils import find_data_dirs, separate_dirs
from src.info_extractor import InfoExtractor

class StatsMerger:
    def __init__(self, data_dirs, sl, ngal, opening_angle=10, zs_list=[0.5, 1.0, 1.5, 2.0, 2.5], save_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/output", overwrite=False):
        self.data_dirs = data_dirs
        self.tiled_dirs, self.bigbox_dirs = separate_dirs(data_dirs)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.overwrite = overwrite

        self.sl = sl
        self.ngal = ngal
        self.oa = opening_angle
        self.zs_list = zs_list

    def run(self):
        #self._run_and_save(is_patch=False, box_type='tiled')
        #self._run_and_save(is_patch=False, box_type='bigbox')
        self._run_and_save(is_patch=True, box_type='tiled')
        self._run_and_save(is_patch=True, box_type='bigbox')

    def _run_and_save(self, is_patch=False, box_type='tiled'):
        suffix = self._generate_suffix(is_patch)
        fname = f"fullsky_stats_{box_type}_{suffix}.npy" if not is_patch else f"patch_stats_{box_type}_{suffix}.npy"
        save_path = os.path.join(self.save_dir, fname)
        if os.path.exists(os.path.join(self.save_dir, fname)) and not self.overwrite:
            print(f"Stats file {os.path.basename(save_path)} already exists, skipping")
            return
        stats = self.merge_stats(is_patch, box_type)
        np.save(save_path, stats)
        print(f"Saved stats to {os.path.basename(save_path)}")

    def merge_stats(self, is_patch=False, box_type='tiled'):
        work_dirs = self.tiled_dirs if box_type == 'tiled' else self.bigbox_dirs
        stats = {}
        for zs in self.zs_list:
            stats_zs = []
            for data_dir in work_dirs:
                tmp_stats = self._load_stats(data_dir, zs, is_patch)
                stats_zs.append(tmp_stats)
            logging.info(f"Loaded stats for zs={zs}, merged {len(stats_zs)} stats")
            stats_zs = np.vstack(stats_zs)
            diags, corr, stds, means = self.total_stats(stats_zs)
            stats[zs] = {
                "diags": diags,
                "corr": corr,
                "stds": stds,
                "means": means
            }
        return stats

    def total_stats(self, data):
        diags = np.diag(np.cov(data, rowvar=False))
        corr = np.corrcoef(data, rowvar=False)
        stds = np.std(data, axis=0)
        means = np.mean(data, axis=0)
        return diags, corr, stds, means
    
    def exclude_outliers(self, data, n_sigma=5):
        stds = np.std(data, axis=0)
        means = np.mean(data, axis=0)
        mask = np.all(np.abs(data - means) < n_sigma * stds, axis=1)
        return data[mask]

    def _generate_suffix(self, is_patch=False):
        suffix = f"oa{self.oa}_" if is_patch else ""
        suffix += f"noiseless_sl{self.sl}"
        if self.ngal != 0:
            suffix = suffix.replace("noiseless", f"ngal{self.ngal}")
        return suffix

    def _generate_fname(self, datadir, zs, is_patch=False):
        info = InfoExtractor.extract_info_from_path(datadir)
        suffix = self._generate_suffix(is_patch)
        if is_patch:
            fname = f"analysis_zs{zs}_s{info['seed']}_{suffix}.npy"
        else:
            fname = f"fullsky_clpdpm_s{info['seed']}_zs{zs}_{suffix}.npy"
        return fname

    def _load_stats(self, data_dir, zs, is_patch=False):
        fname = self._generate_fname(data_dir, zs, is_patch)
        data = np.load(os.path.join(data_dir, "analysis_fullsky", fname)) if not is_patch else np.load(os.path.join(data_dir, "analysis_patch", fname))            
        return data
    
class StatsReorder:
    def __init__(self):
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
            'Skewness_0',
            'Skewness_1',
            'Skewness_2',
            'Kurtosis_0',
            'Kurtosis_1',
            'Kurtosis_2',
            'Kurtosis_3',
            'PDF',
            'Peak',
            'Minima',
            'area(MFs)',
            'perimeter(MFs)',
            'genus(MFs)'
        ]
        self.permuted_indices = self.prepare_permuted_indices()
        self.permuted_stats_name_indices = {stat: self.permuted_indices[start:end] for stat, [start, end] in self.stats_name_indices.items()}

    def prepare_permuted_indices(self):
        permuted_indices = []
        for stat in self.stats_desired_order:
            if stat not in self.stats_name_indices:
                raise ValueError(f"Statistic '{stat}' not found in stats_name_indices mapping.")
            
            start, end = self.stats_name_indices[stat]
            stat_indices = list(range(start, end))
            permuted_indices.extend(stat_indices)
        
        return permuted_indices


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--overwrite", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    data_dirs = find_data_dirs()
    sl_list = [2, 5, 8, 10]
    ngal_list = [0, 7, 15, 30, 50]
    for sl in sl_list:
        for ngal in ngal_list:
            print(f"Start merging stats for ngal={ngal}, sl={sl}")
            stats_merger = StatsMerger(data_dirs, sl, ngal, overwrite=args.overwrite)
            try:
                stats_merger.run()
            except Exception as e:
                print(e)
                logging.error(e)
                continue