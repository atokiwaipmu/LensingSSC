
import os
import numpy as np
import healpy as hp
import logging

from src.utils import find_data_dirs
from src.info_extractor import InfoExtractor

class StatsMerger:
    def __init__(self, data_dirs, sl, ngal, oa=10, zs_list=[0.5, 1.0, 2.0, 3.0], lmin=300, lmax=3000, nbin = 15, save_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/output"):
        self.data_dirs = data_dirs
        self._separate_dirs()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.sl = sl
        self.ngal = ngal
        self.oa = oa
        self.zs_list = zs_list

        self.l_edges = np.logspace(np.log10(lmin), np.log10(lmax), nbin + 1, endpoint=True)
        self.bins = np.linspace(-4, 4, nbin + 1, endpoint=True)

        self.ell = (self.l_edges[1:] + self.l_edges[:-1]) / 2
        self.nu = (self.bins[1:] + self.bins[:-1]) / 2

    def _separate_dirs(self):
        self.tiled_dirs = []
        self.bigbox_dirs = []
        for data_dir in self.data_dirs:
            info = InfoExtractor.extract_info_from_path(data_dir)
            if info['box_type'] == 'tiled':
                self.tiled_dirs.append(data_dir)
            elif info['box_type'] == 'bigbox':
                self.bigbox_dirs.append(data_dir)

    def run(self):
        self._run_and_save(is_patch=False, box_type='tiled')
        self._run_and_save(is_patch=False, box_type='bigbox')
        self._run_and_save(is_patch=True, box_type='tiled')
        self._run_and_save(is_patch=True, box_type='bigbox')

    def _run_and_save(self, is_patch=False, box_type='tiled'):
        suffix = self._generate_suffix(is_patch)
        stats = self.merge_stats(is_patch, box_type)
        fname = f"fullsky_stats_{box_type}_{suffix}.npy" if not is_patch else f"patch_stats_{box_type}_{suffix}.npy"
        save_path = os.path.join(self.save_dir, fname)
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

    def _generate_suffix(self, is_patch=False):
        suffix = f"oa{self.oa}_" if is_patch else ""
        suffix += f"sl{self.sl}_noiseless"
        if self.ngal != 0:
            suffix = suffix.replace("noiseless", f"ngal{self.ngal}")
        return suffix

    def _generate_fname(self, datadir, zs, is_patch=False):
        info = InfoExtractor.extract_info_from_path(datadir)
        suffix = self._generate_suffix(is_patch)
        if is_patch:
            fname = f"analysis_sqclpdpm_s{info['seed']}_zs{zs}_{suffix}.npy"
        else:
            fname = f"fullsky_clpdpm_s{info['seed']}_zs{zs}_{suffix}.npy"
        return fname

    def _load_stats(self, data_dir, zs, is_patch=False):
        fname = self._generate_fname(data_dir, zs, is_patch)
        data = np.load(os.path.join(data_dir, "fullsky", fname)) if not is_patch else np.load(os.path.join(data_dir, "flat", fname))

        if is_patch:
            sq, clkk, pdf, peak, minima = np.split(data, 5, axis=1)
            sq = np.abs(sq)
            sq = EllHelper.dimensionless_bispectrum(sq, self.ell)
            clkk = EllHelper.dimensionless_cl(clkk, self.ell)
            data = np.hstack([sq, clkk, pdf, peak, minima])
        else:
            clkk, pdf, peak, minima = np.split(data, 4)
            clkk = EllHelper.dimensionless_cl(clkk, self.ell)
            data = np.hstack([clkk, pdf, peak, minima])
            
        return data

class EllHelper:
    @staticmethod
    def dimensionless_cl(cl, ell):
        return ell * (ell+1) * cl / (2*np.pi)

    @staticmethod
    def dimensionless_bispectrum(bispec, ell):
        return bispec * ell**4 / (2*np.pi)**2

if __name__ == "__main__":
    data_dirs = find_data_dirs()
    sl = 2
    ngal_list = [0, 30]
    for ngal in ngal_list:
        print(f"Start merging stats for ngal={ngal}")
        stats_merger = StatsMerger(data_dirs, sl, ngal)
        try:
            stats_merger.run()
        except Exception as e:
            print(e)
            logging.error(e)
            continue