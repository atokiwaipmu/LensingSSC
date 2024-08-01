
import os
import argparse
import logging
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy import units as u
from lenstools import ConvergenceMap
import h5py

from src.utils.ConfigData import ConfigData, ConfigAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_file(dir_results, survey, config_sim, zs, sl):
    if sl == 0:
        path = os.path.join(dir_results, config_sim, "data", f"kappa_zs{zs:.1f}.fits")
    elif survey == "noiseless":
        path = os.path.join(dir_results, config_sim, "smoothed", f"sl={sl}", f"kappa_zs{zs:.1f}_smoothed_sl{sl}.fits")
    else:
        path = os.path.join(dir_results, config_sim, "smoothed", f"sl={sl}", f"kappa_zs{zs:.1f}_smoothed_sl{sl}_noise_{survey}_seed100.fits")

    return path

def save_data(data, save_path, args):
    results, bins, centers, std, lmin, lmax, xsize, patch_size = data

    with h5py.File(save_path, 'w') as f:
        # Create datasets to store data and edges arrays
        f.create_dataset('data', data=results, dtype='float64')
        f.create_dataset('bins', data=bins, dtype='float64')
        f.create_dataset('centers', data=centers, dtype='float64')
        
        # Store metadata as attributes
        f.attrs['config_sim'] = args.config_sim
        f.attrs['zs'] = args.zs
        f.attrs['sl'] = args.sl
        f.attrs['survey'] = args.survey
        f.attrs['stddev'] = std
        f.attrs['lmin'] = lmin
        f.attrs['lmax'] = lmax
        f.attrs['xsize'] = xsize
        f.attrs['patch_size'] = patch_size

def patch_map(nside, patch_size = 10, nside_base = 4, coarse_num = hp.nside2npix(8192)//hp.nside2npix(4)):
    centers = []
    for i in range(hp.nside2npix(nside_base)):
        center = hp.pix2ang(nside=nside_base, ipix=i, nest=True)
        vec = hp.ang2vec(center[0], center[1])
        ipix = hp.query_disc(nside=nside, vec=vec, radius=np.radians(patch_size/2)*np.sqrt(2), nest=True)
        if np.min(ipix) >= i*coarse_num and np.max(ipix) < (i+1)*coarse_num:
            centers.append(center)

    return centers

def exclude_edges(heights, positions, xsize=1024, patch_size=10):
    tmp_positions = positions.value * xsize / patch_size
    mask = (tmp_positions[:, 0] > 0) & (tmp_positions[:, 0] < xsize-1) & (tmp_positions[:, 1] > 0) & (tmp_positions[:, 1] < xsize-1)
    return heights[mask], tmp_positions[mask].astype(int)

def process_data(data, nbin=15, lmin=300, lmax=3000, xsize=1024, patch_size=10):
    std = np.std(data)
    logging.info(f"Standard deviation: {std:.4f}")

    kappa_bins = np.linspace(-4, 4, nbin + 1, endpoint=True) * std
    l_edges = np.logspace(np.log10(lmin), np.log10(lmax), num=nbin + 1)

    centers = patch_map(hp.npix2nside(len(data)))

    results = []
    bins = []
    for center in centers:
        logging.info(f"Processing patch at {center}")
        patch = hp.gnomview(data, rot=center, xsize=xsize, reso=patch_size*60/xsize, return_projected_map=True, no_plot=True)
        conv_map = ConvergenceMap(patch, angle=patch_size * u.deg)
        l, pl = conv_map.powerSpectrum(l_edges)
        pl = pl * l * (l + 1) / (2 * np.pi)

        _, bispec_equil = conv_map.bispectrum(l_edges=l_edges, configuration='equilateral')
        _, bispec_fold = conv_map.bispectrum(l_edges=l_edges, ratio=0.1, configuration='folded')
        bispec_equil = l**4 * np.abs(bispec_equil) / (2 * np.pi)**2
        bispec_fold = l**4 * np.abs(bispec_fold) / (2 * np.pi)**2
        
        nu, pdf = conv_map.pdf(kappa_bins)

        peak_height,peak_positions = conv_map.locatePeaks(kappa_bins)
        peak_height,peak_positions = exclude_edges(peak_height, peak_positions, xsize, patch_size)
        peak_counts, _ = np.histogram(peak_height, bins=kappa_bins)

        conv_map_minus = ConvergenceMap(-patch, angle=patch_size * u.deg)
        minima_height,minima_positions = conv_map_minus.locatePeaks(kappa_bins)
        minima_height,minima_positions = exclude_edges(minima_height, minima_positions, xsize, patch_size)
        minima_counts, _ = np.histogram(minima_height, bins=kappa_bins)

        result = np.hstack([pl, bispec_equil, bispec_fold, pdf, peak_counts, minima_counts])
        bin = np.hstack([l, l, l, nu, (kappa_bins[1:] + kappa_bins[:-1])/2, (kappa_bins[1:] + kappa_bins[:-1])/2])

        results.append(result)
        bins.append(bin)

    return np.array(results), np.array(bins), centers, std, lmin, lmax, xsize, patch_size
    

def main(args):
    # Usage: python -m src.pipeline_flatsky --config_sim bigbox --zs 0.5 --sl 2 --survey HSC

    config_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_data.json')
    config_data = ConfigData.from_json(config_file)

    config_analysis_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_analysis.json')
    config_analysis = ConfigAnalysis.from_json(config_analysis_file)

    path = find_file(config_analysis.resultsdir, args.survey, args.config_sim, args.zs, args.sl)
    logging.info(f"Processing {os.path.basename(path)}")

    save_dir = os.path.join(config_analysis.resultsdir, "analysis")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"flatsky_{args.config_sim}_zs{args.zs:.1f}_sl{args.sl}_{args.survey}.h5")
    if os.path.exists(save_path):
        logging.info(f"File already exists, skipping...")
        return

    data = hp.read_map(path)
    if args.sl == 0:
        data = hp.reorder(data, n2r=True)

    results = process_data(data)
    save_data(results, save_path, args)

    logging.info(f"Saved data to {os.path.basename(save_path)}")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_sim', choices=['bigbox', 'tiled'], required=True)
    parser.add_argument('--zs', choices=[0.5, 1.0, 1.5, 2.0], type=float, required=True)
    parser.add_argument('--sl', choices=[0, 2, 5, 8, 10], type=int, required=True)
    parser.add_argument('--survey', choices=['noiseless', 'Euclid-LSST', 'DES-KiDS', 'HSC', 'Roman'], required=True)
    args = parser.parse_args()

    main(args)