import sqlite3
import healpy as hp
from astropy import units as u
from lenstools import ConvergenceMap
import matplotlib.pyplot as plt
import numpy as np
import io
from multiprocessing import Pool, cpu_count
import argparse 
import logging

from src.utils.database_query import query_fits_paths

logging.basicConfig(level=logging.INFO)

class WeakLensingAnalyzer:
    def __init__(self, db_path, config_sim, zs, sl, survey, lmin=300, lmax=3000, patch_size=10, xsize=1024, nbin=15):
        self.db_path = db_path
        self.config_sim = config_sim
        self.zs = zs
        self.sl = sl
        self.survey = survey
        self.lmin = lmin
        self.lmax = lmax
        self.patch_size = patch_size
        self.xsize = xsize
        self.nbin = nbin

        self.data = self._read_fits_data()
        self.stddev = np.std(self.data)

        self.l_edges = np.logspace(np.log10(self.lmin), np.log10(self.lmax), num=self.nbin + 1)
        self.kappa_edges = np.linspace(-4 * self.stddev, 4 * self.stddev, self.nbin + 1)

        self._setup_db()
        self.patch_centers()

    def _setup_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lensing_data (
                id INTEGER PRIMARY KEY,
                config_sim TEXT,
                zs REAL,
                sl REAL,
                survey TEXT,
                patch_id INTEGER,
                lmin INTEGER,
                lmax INTEGER,
                ell BLOB,
                cl BLOB,
                bispec_equil BLOB,
                bispec_fold BLOB,
                nu BLOB,
                pdf BLOB,
                peak_counts BLOB,
                minima_counts BLOB
            )
        ''')
        conn.commit()
        conn.close()

    def _read_fits_data(self):
        file_path = query_fits_paths(config_sim=self.config_sim, zs=self.zs, sl=self.sl, survey=self.survey)[0][0]
        data = hp.read_map(file_path)
        if (self.survey == 'noiseless')&(self.sl == 0):
            return hp.reorder(data, n2r=True)
        return data

    def _store_results(self, patches):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for i, patch in enumerate(patches):
            l, pl, bispec_equil, bispec_fold, nu, pdf, peak_counts, minima_counts = patch
            cursor.execute('''
                INSERT INTO lensing_data (config_sim, zs, sl, survey, patch_id, lmin, lmax, ell, cl, bispec_equil, bispec_fold, nu, pdf, peak_counts, minima_counts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (self.config_sim, self.zs, self.sl, self.survey, i, self.lmin, self.lmax, self._np_to_blob(l), self._np_to_blob(pl), self._np_to_blob(bispec_equil), self._np_to_blob(bispec_fold), self._np_to_blob(nu), self._np_to_blob(pdf), self._np_to_blob(peak_counts), self._np_to_blob(minima_counts)))
        
        conn.commit()
        conn.close()

    def _np_to_blob(self, array):
        out = io.BytesIO()
        np.save(out, array)
        return out.getvalue()
    
    def _blob_to_np(self, blob):
        return np.load(io.BytesIO(blob))

    def _patch_map(self, ra_dec):
        return hp.gnomview(self.data, rot=(ra_dec[0], ra_dec[1]), xsize=self.xsize, reso=self.patch_size * 60 / self.xsize, return_projected_map=True, no_plot=True)

    def _exclude_edges(self, heights, positions):
        tmp_positions = positions.value * self.xsize / self.patch_size
        mask = (tmp_positions[:, 0] > 0) & (tmp_positions[:, 0] < self.xsize-1) & (tmp_positions[:, 1] > 0) & (tmp_positions[:, 1] < self.xsize-1)
        return heights[mask], tmp_positions[mask].astype(int)

    def _process_patch(self, patch):
        conv_map = ConvergenceMap(patch, angle=self.patch_size * u.deg)
        conv_map_minus = ConvergenceMap(-patch, angle=self.patch_size * u.deg)

        l, pl = conv_map.powerSpectrum(self.l_edges)
        pl = pl * l * (l + 1) / (2 * np.pi)

        _, bispec_equil = conv_map.bispectrum(l_edges=self.l_edges, configuration='equilateral')
        _, bispec_fold = conv_map.bispectrum(l_edges=self.l_edges, ratio=0.1, configuration='folded')
        bispec_equil = l**4 * np.abs(bispec_equil) / (2 * np.pi)**2
        bispec_fold = l**4 * np.abs(bispec_fold) / (2 * np.pi)**2
        
        nu, pdf = conv_map.pdf(self.kappa_edges)
        peak_height,peak_positions = conv_map.locatePeaks(self.kappa_edges)
        peak_height,peak_positions = self._exclude_edges(peak_height, peak_positions)
        peak_counts, _ = np.histogram(peak_height, bins=self.kappa_edges)

        minima_height,minima_positions = conv_map_minus.locatePeaks(self.kappa_edges)
        minima_height,minima_positions = self._exclude_edges(minima_height, minima_positions)
        minima_counts, _ = np.histogram(minima_height, bins=self.kappa_edges)

        return (l, pl, bispec_equil, bispec_fold, nu, pdf, peak_counts, minima_counts)

    def process_map(self):
        self.patches = [self._patch_map(ra_dec) for ra_dec in self.centers]
        with Pool(cpu_count()) as pool:
            patches = pool.map(self._process_patch, self.patches)
        self._store_results(patches)

    def plot_fullsky_map(self):
        fig = plt.figure(figsize=(10, 5))
        hp.mollview(self.data, fig=fig.number, title='Fullsky convergence map', min=-0.006, max=0.006, cmap='jet')
        plt.show()
        plt.close()

    def patch_centers(self, nside_base=4, nside=8192):
        coarse_num = hp.nside2npix(nside)//hp.nside2npix(nside_base)

        centers = []
        for i in range(hp.nside2npix(nside_base)):
            center = hp.pix2ang(nside=nside_base, ipix=i, nest=True)
            vec = hp.ang2vec(center[0], center[1])
            ipix = hp.query_disc(nside=nside, vec=vec, radius=np.radians(self.patch_size/2)*np.sqrt(2), nest=True)
            if np.min(ipix) >= i*coarse_num and np.max(ipix) < (i+1)*coarse_num:
                centers.append(center)

        self.centers = centers

if __name__ == "__main__":
    # Example usage: python src/flatsky/analysis_database.py --config_sim bigbox --zs 0.5 --sl 0 --survey noiseless
    db_path = '/lustre/work/akira.tokiwa/Projects/LensingSSC/results/kappa_data.db'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_sim', choices=['bigbox', 'tiled'], required=True)
    parser.add_argument('--zs', choices=[0.5, 1.0, 1.5, 2.0], type=float, required=True)
    parser.add_argument('--sl', choices=[0, 2, 5, 8, 10], type=int, required=True)
    parser.add_argument('--survey', choices=['noiseless', 'Euclid-LSST', 'DES-KiDS', 'HSC', 'Roman'], required=True)
    args = parser.parse_args()

    analyzer = WeakLensingAnalyzer(db_path, args.config_sim, args.zs, args.sl, args.survey)
    analyzer.process_map()