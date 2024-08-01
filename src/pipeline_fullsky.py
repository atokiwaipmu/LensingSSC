
import os
import argparse
import logging
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py

from src.utils.ConfigData import ConfigData, ConfigAnalysis
from src.utils.find_extrema import find_extrema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_file(dir_results, survey, config_sim, zs, sl):
    if sl == 0:
        path = os.path.join(dir_results, config_sim, "data", f"kappa_zs{zs:.1f}.fits")
    elif survey == "noiseless":
        path = os.path.join(dir_results, config_sim, "smoothed", f"sl={sl}", f"kappa_zs{zs:.1f}_smoothed_sl{sl}.fits")
    else:
        path = os.path.join(dir_results, config_sim, "smoothed", f"sl={sl}", f"kappa_zs{zs:.1f}_smoothed_sl{sl}_noise_{survey}_seed100.fits")

    return path

def plot_orthview(map, fname):
    fig = plt.figure(figsize=(7, 7))
    hp.orthview(map, half_sky=True, cmap='cividis', fig=fig.number, min=-0.02, max=0.06)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def save_data(data, save_path, args):
    if len(data) == 3:
        clkk, pdf, std = data
    else:
        clkk, pdf, peaks, minima, std = data

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('clkk', data=clkk)
        f.create_dataset('pdf', data=pdf)
        if len(data) == 5:
            f.create_dataset('peaks', data=peaks)
            f.create_dataset('minima', data=minima)
        
        # Store metadata as attributes
        f.attrs['config_sim'] = args.config_sim
        f.attrs['zs'] = args.zs
        f.attrs['sl'] = args.sl
        f.attrs['survey'] = args.survey
        f.attrs['stddev'] = std

def run_PeaksMinima(map_smooth, nside, return_pos=False):
        map_smooth_ma = hp.ma(map_smooth)
        peak_pos, peak_amp, minima_pos, minima_amp = find_extrema(map_smooth_ma, lonlat=True, nside=nside)
        if return_pos:
            peaks = np.vstack([peak_pos.T, peak_amp]).T
            minima = np.vstack([minima_pos.T, minima_amp]).T
        else:
            peaks = peak_amp
            minima = minima_amp
        return peaks, minima


def process_data(data, args, nbin=15, lmax=3000):
    std = np.std(data)
    logging.info(f"Standard deviation: {std:.4f}")

    kappa_bins = np.linspace(-4, 4, nbin + 1, endpoint=True) * std
    
    cl = hp.anafast(data, lmax=lmax)
    logging.info(f"Clkk computed up to ell={lmax}")

    pdf, _ = np.histogram(data, bins=kappa_bins, density=True)
    logging.info(f"PDF computed with {nbin} bins")

    if args.sl == 0:
        return cl, pdf, std
    else:
        peaks, minima = run_PeaksMinima(data, hp.npix2nside(len(data)))
        logging.info(f"Peaks and minima computed")
        return cl, pdf, peaks, minima, std
    

def main(args):
    # Usage: python src.pipeline_fullsky --config_sim bigbox --zs 0.5 --sl 0 --survey noiseless

    config_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_data.json')
    config_data = ConfigData.from_json(config_file)

    config_analysis_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_analysis.json')
    config_analysis = ConfigAnalysis.from_json(config_analysis_file)

    path = find_file(config_analysis.resultsdir, args.survey, args.config_sim, args.zs, args.sl)
    logging.info(f"Processing {os.path.basename(path)}")

    save_dir = os.path.join(config_analysis.resultsdir, "analysis")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"kappa_{args.config_sim}_zs{args.zs:.1f}_sl{args.sl}_{args.survey}.h5")
    if os.path.exists(save_path):
        logging.info(f"File already exists, skipping...")
        return

    data = hp.read_map(path)
    if args.sl == 0:
        data = hp.reorder(data, n2r=True)
    plot_orthview(data, os.path.join(config_analysis.imgdir , f"kappa_{args.config_sim}_zs{args.zs:.1f}_sl{args.sl}_{args.survey}.png"))
    logging.info(f"Plotted map to {os.path.basename(config_analysis.imgdir)}")

    results = process_data(data, args)
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