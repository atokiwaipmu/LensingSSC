
import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import h5py
from glob import glob

from src.utils.ConfigData import ConfigData, ConfigAnalysis

def load_data(load_path, nbin=15, snr=True):
    with h5py.File(load_path, 'r') as f:
        # Load datasets
        results = f['data'][:]
        bins = f['bins'][:]
        centers = f['centers'][:]
        
        # Load metadata attributes
        config_sim = f.attrs['config_sim']
        zs = f.attrs['zs']
        sl = f.attrs['sl']
        survey = f.attrs['survey']
        std = f.attrs['stddev']
        lmin = f.attrs['lmin']
        lmax = f.attrs['lmax']
        xsize = f.attrs['xsize']
        patch_size = f.attrs['patch_size']

        # flip last nbin cols: minima
        results[:, -nbin:] = np.flip(results[:, -nbin:], axis=1)

        bins = bins[0]
        if snr: # replace last 3 nbin cols with np.linspace(-4, 4, nbin+1)
            bins[3*nbin:] = np.tile(np.linspace(-4, 4, nbin), 3)
        
    # Return the loaded data and metadata as a tuple
    return results, bins, centers, std, lmin, lmax, xsize, patch_size, config_sim, zs, sl, survey

def cal_stats(results):
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)
    cov = np.cov(results.T)
    corr = np.corrcoef(results.T)
    diag = np.sqrt(np.diag(cov))
    
    return means, stds, cov, corr, diag

def set_ticks(ax, labels, nbin=15):
    tick_positions = [8 + nbin * i for i in range(len(labels))] 
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=90, ha='right')

    # Add grid lines to separate statistics
    for i in range(1, len(labels)):
        ax.axhline(y=i * nbin - 0.5, color='black', linewidth=2)
        ax.axvline(x=i * nbin - 0.5, color='black', linewidth=2)

def plot_stats(axes, labels, color, bins, diag, linestyle='-', logscale=True, stddev=None, nbin=15):
    for i, label in enumerate(labels):
        ax = axes[i]
        im = ax.plot(bins[i*nbin:i*nbin+nbin], diag[i*nbin:i*nbin+nbin], linestyle=linestyle, color=color)
        if stddev is not None:
            ax.fill_between(bins[i*nbin:i*nbin+nbin], diag[i*nbin:i*nbin+nbin] - stddev[i*nbin:i*nbin+nbin], diag[i*nbin:i*nbin+nbin] + stddev[i*nbin:i*nbin+nbin], color=color, alpha=0.2)
        ax.set_title(label)
        if logscale:
            ax.set_yscale('log')
            ax.set_xscale('log') if i < 3 else None

def set_legends(colors, line_styles, zs_list):
    # Custom legend handles
    legend_handles = []

    # Add handles for line styles
    legend_handles.append(Line2D([0], [0], color='black', linestyle=line_styles["tiled"], label='Tiled'))
    legend_handles.append(Line2D([0], [0], color='black', linestyle=line_styles["bigbox"], label='Bigbox'))

    # Add handles for colors (redshifts)
    for zs in zs_list:
        legend_handles.append(Patch(color=colors[zs_list.index(zs)], label=f"zs={zs}"))

    return legend_handles

def setup_ratio(axes_ratio, labels, bins=None, nbin=15):
    axes_ratio[0].set_ylabel("Bigbox / Tiled")
    axes_ratio[3].set_ylabel("Bigbox / Tiled")

    for i in range(len(labels)):
        tmp_ax = axes_ratio[i]
        tmp_ax.set_ylim(0, 4)
        tmp_ax.hlines(1, np.min(bins[i*nbin:i*nbin+nbin]), np.max(bins[i*nbin:i*nbin+nbin]), linestyle='--', color='black')    

def main(args):

    config_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_data.json')
    config_data = ConfigData.from_json(config_file)

    config_analysis_file = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/configs", 'config_analysis.json')
    config_analysis = ConfigAnalysis.from_json(config_analysis_file)

    img_dir = os.path.join(config_analysis.imgdir, "analysis")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    data_dir = os.path.join(config_analysis.resultsdir, "analysis")
    paths_bigbox = sorted(glob(os.path.join(data_dir, f"flatsky_bigbox_zs*_sl{args.sl}_{args.survey}.h5")))
    paths_tiled = sorted(glob(os.path.join(data_dir, f"flatsky_tiled_zs*_sl{args.sl}_{args.survey}.h5")))

    zs_list = [0.5, 1.0, 1.5, 2.0]

    labels = [r'$\tilde{C}^{\kappa\kappa}_{\ell}$', #r'$C^{\kappa\kappa}_{\ell} \ell (\ell+1) / 2\pi$', 
          r'$\tilde{B}_{eq}$',#r'$B(\ell, \ell, \ell)  \ell^4 / (2\pi)^2 $', 
          r'$\tilde{B}_{sq}$',#r'$B(\ell, \ell, \sim0)  \ell^4 / (2\pi)^2$',
          "PDF",
          "Peaks",
          "Minima"
          ]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # Define line styles
    line_styles = {"tiled": "--", "bigbox": "-"}

    # Set the legend handles
    legend_handles = set_legends(colors, line_styles, zs_list)

    fig, ax = plt.subplots(1, len(zs_list), figsize=(20, 5))
    fig2, ax2 = plt.subplots(1, len(zs_list), figsize=(20, 5))
    fig3, ax3 = plt.subplots(1, len(zs_list), figsize=(20, 5))

    # Create the figure and GridSpec layout
    gs = GridSpec(4, 3, height_ratios=[3, 1, 3, 1], hspace=0.3)

    fig4 = plt.figure(figsize=(20, 10))
    axes4 = [fig4.add_subplot(gs[0, i]) for i in range(3)] + [fig4.add_subplot(gs[2, i]) for i in range(3)]
    axes_ratio4 = [fig4.add_subplot(gs[1, i]) for i in range(3)] + [fig4.add_subplot(gs[3, i]) for i in range(3)]
    axes4[3].legend(handles=legend_handles, loc='lower left')

    fig5 = plt.figure(figsize=(20, 10))
    axes5 = [fig5.add_subplot(gs[0, i]) for i in range(3)] + [fig5.add_subplot(gs[2, i]) for i in range(3)]
    axes_ratio5 = [fig5.add_subplot(gs[1, i]) for i in range(3)] + [fig5.add_subplot(gs[3, i]) for i in range(3)]
    axes5[3].legend(handles=legend_handles, loc='lower left')


    """
    for zs in zs_list:
        halofit = np.load(f"/lustre/work/akira.tokiwa/Projects/LensingSSC/results/halofit/kappa_zs{zs:.1f}_Clkk_ell_0_3000.npz")
        ell = halofit['ell'][300:3000]
        clkk = halofit['clkk'][300:3000]
        axes5[0].plot(ell, clkk * ell * (ell + 1) / 2 / np.pi, label=f'Halofit zs={zs}', color=colors[zs_list.index(zs)], linestyle='-.')
    """
    
    for path, path2 in zip(paths_bigbox, paths_tiled):
        results, bins, centers, std, lmin, lmax, xsize, patch_size, config_sim, zs, sl, survey = load_data(path)
        means, stds, cov, corr, diag = cal_stats(results)

        results2, bins2, centers2, std2, lmin2, lmax2, xsize2, patch_size2, config_sim2, zs2, sl2, survey2 = load_data(path2)
        means2, stds2, cov2, corr2, diag2 = cal_stats(results2)

        # plot the correlation matrix difference
        corr_diff = corr2 - corr
        im = ax[zs_list.index(zs)].imshow(corr_diff, cmap='coolwarm', vmin=-1, vmax=1)
        ax[zs_list.index(zs)].set_title(f'zs={zs}'+r' $r_{Bigbox} - r_{Tiled}$')
        set_ticks(ax[zs_list.index(zs)], labels)

        # plot the correlation matrix of bigbox
        im2 = ax2[zs_list.index(zs)].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax2[zs_list.index(zs)].set_title(f'zs={zs}'+r' $r_{Bigbox}$')
        set_ticks(ax2[zs_list.index(zs)], labels)

        # plot the correlation matrix of tiled
        im3 = ax3[zs_list.index(zs)].imshow(corr2, cmap='coolwarm', vmin=-1, vmax=1)
        ax3[zs_list.index(zs)].set_title(f'zs={zs}'+r' $r_{Tiled}$')
        set_ticks(ax3[zs_list.index(zs)], labels)

        plot_stats(axes4, labels, colors[zs_list.index(zs)], bins, diag, linestyle='-')
        plot_stats(axes4, labels, colors[zs_list.index(zs)], bins2, diag2, linestyle='--')
        plot_stats(axes_ratio4, labels, colors[zs_list.index(zs)], bins, diag2/diag, linestyle='-', logscale=False)

        plot_stats(axes5, labels, colors[zs_list.index(zs)], bins, means, linestyle='-', stddev=stds)
        plot_stats(axes5, labels, colors[zs_list.index(zs)], bins2, means2, linestyle='--', stddev=stds2)
        plot_stats(axes_ratio5, labels, colors[zs_list.index(zs)], bins, means2/means, linestyle='-', logscale=False)

    setup_ratio(axes_ratio4, labels, bins)
    setup_ratio(axes_ratio5, labels, bins)

    fig.savefig(os.path.join(img_dir, f"corr_diff_sl{args.sl}_{args.survey}.png"), bbox_inches='tight')
    fig2.savefig(os.path.join(img_dir, f"corr_bigbox_sl{args.sl}_{args.survey}.png"), bbox_inches='tight')
    fig3.savefig(os.path.join(img_dir, f"corr_tiled_sl{args.sl}_{args.survey}.png"), bbox_inches='tight')
    fig4.savefig(os.path.join(img_dir, f"diagonals_sl{args.sl}_{args.survey}.png"), bbox_inches='tight')
    fig5.savefig(os.path.join(img_dir, f"means_sl{args.sl}_{args.survey}.png"), bbox_inches='tight')

    plt.close('all')

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sl', choices=[0, 2, 5, 8, 10], type=int, required=True)
    parser.add_argument('--survey', choices=['noiseless', 'Euclid-LSST', 'DES-KiDS', 'HSC', 'Roman'], required=True)
    args = parser.parse_args()

    main(args)