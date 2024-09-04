
import os
import argparse
from glob import glob
import yaml
import numpy as np
import datetime
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def plot_cov(fname, cov_tiled, cov_bigbox, title_tiled, title_bigbox, labels):
    nbin = cov_bigbox.shape[0] // len(labels)
    tick_positions = [nbin/2 + nbin * i for i in range(len(labels))]

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    cax = ax[0].imshow(cov_tiled, cmap='viridis')
    fig.colorbar(cax, ax=ax[0], shrink=0.6)
    ax[0].set_title(title_tiled, fontsize=10)

    cax = ax[1].imshow(cov_bigbox, cmap='viridis')
    fig.colorbar(cax, ax=ax[1], shrink=0.6)
    ax[1].set_title(title_bigbox, fontsize=10)

    cax = ax[2].imshow(cov_bigbox - cov_tiled, cmap='bwr')
    fig.colorbar(cax, ax=ax[2], shrink=0.6)
    ax[2].set_title("BigBox - Tiled Covariance", fontsize=10)

    for axes in ax.flatten():
        axes.set_xticks(tick_positions, labels, fontsize=8)
        axes.set_yticks(tick_positions, labels, fontsize=8, rotation=90, va='center')
        axes.invert_yaxis()

    fig.savefig(fname, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.show()
    plt.close(fig)

def plot_corr(fname, corr_tiled, corr_bigbox, title, title_tiled, title_bigbox, labels, vmin=-0.3, vmax=0.3):    
    nbin = corr_bigbox.shape[0] // len(labels)
    tick_positions = [nbin/2 + nbin * i for i in range(len(labels))]

    fig = plt.figure(figsize=(10, 4))
    gs_master = GridSpec(nrows=2, ncols=1, height_ratios=[1, 9])
    gs_plot = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_master[1], wspace=0.2)
    ax = [fig.add_subplot(gs_plot[i]) for i in range(3)]

    cax = ax[0].imshow(corr_tiled, cmap='bwr', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax[0], shrink=0.6)
    ax[0].set_title(title_tiled, fontsize=10)

    cax = ax[1].imshow(corr_bigbox, cmap='bwr', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax[1], shrink=0.6)
    ax[1].set_title(title_bigbox, fontsize=10)

    cax = ax[2].imshow(corr_bigbox - corr_tiled, cmap='bwr', vmin=vmin, vmax=vmax)
    fig.colorbar(cax, ax=ax[2], shrink=0.6)
    ax[2].set_title("BigBox - Tiled Correlation", fontsize=10)

    for axes in ax:
        axes.set_xticks(tick_positions, labels, fontsize=8)
        axes.set_yticks(tick_positions, labels, fontsize=8, rotation=90, va='center')
        axes.invert_yaxis()

    ax_title = fig.add_subplot(gs_master[0])
    ax_title.text(0.5, 0.5, title, fontsize=12, ha='center', va='center')
    ax_title.axis('off')

    fig.savefig(fname, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.show()
    plt.close(fig)

def plot_stats(fname, title, title_tiled, title_bigbox, title_attrs, ell, nu, means_tiled, means_bigbox, labels, stds_tiled=None, stds_bigbox=None, ratio_range=[0.95, 1.05]):
    nbin = means_bigbox.shape[0] // len(labels)

    fig = plt.figure(figsize=(15, 6))

    gs_master = GridSpec(nrows=2, ncols=3, height_ratios=[1, 1])
    gs_ell = [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[0, j], height_ratios=[3, 1], hspace=0.01) for j in range(2)]
    gs_nu = [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[1, j], height_ratios=[3, 1], hspace=0.01) for j in range(3)]

    axes = [fig.add_subplot(gs_ell[i][0]) for i in range(2)] + [fig.add_subplot(gs_nu[i][0]) for i in range(3)]
    ratio_axes = [fig.add_subplot(gs_ell[i][1]) for i in range(2)] + [fig.add_subplot(gs_nu[i][1]) for i in range(3)]

    for i, label in enumerate(labels):
        ax = axes[i]
        ratio_ax = ratio_axes[i]
        
        if i < 2:
            tiled_data = means_tiled[i*nbin:i*nbin+nbin]
            bigbox_data = means_bigbox[i*nbin:i*nbin+nbin]
            if stds_tiled is not None:
                ax.errorbar(ell, tiled_data, yerr=stds_tiled[i*nbin:i*nbin+nbin], label="Tiled", capsize=2)
            else:
                ax.plot(ell, tiled_data, label="Tiled")

            if stds_bigbox is not None:
                ax.errorbar(ell, bigbox_data, yerr=stds_bigbox[i*nbin:i*nbin+nbin], label="BigBox", capsize=2, c="tab:orange")
            else:
                ax.plot(ell, bigbox_data, label="BigBox", c="tab:orange")
            
            ratio = bigbox_data / tiled_data
            ratio_ax.plot(ell, ratio, label="BigBox / Tiled", c="tab:purple")
            ratio_ax.set_xscale('log')
            ratio_ax.set_ylim(ratio_range)
            ratio_ax.set_xticks([300, 500, 1000, 2000, 3000])
            ratio_ax.set_xticklabels([300, 500, 1000, 2000, 3000])
            
            ax.set_title(label)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xticks([300, 500, 1000, 2000, 3000])
            ax.set_xticklabels([300, 500, 1000, 2000, 3000])
        else:
            tiled_data = means_tiled[i*nbin:i*nbin+nbin]
            bigbox_data = means_bigbox[i*nbin:i*nbin+nbin]
            if stds_tiled is not None:
                ax.errorbar(nu, tiled_data, yerr=stds_tiled[i*nbin:i*nbin+nbin], label="Tiled", capsize=2)
            else:
                ax.plot(nu, tiled_data, label="Tiled")

            if stds_bigbox is not None:
                ax.errorbar(nu, bigbox_data, yerr=stds_bigbox[i*nbin:i*nbin+nbin], label="BigBox", capsize=2, c="tab:orange")
            else:
                ax.plot(nu, bigbox_data, label="BigBox", c="tab:orange")
            
            ratio = bigbox_data / tiled_data
            ratio_ax.plot(nu, ratio, label="BigBox / Tiled", c="tab:purple")
            ratio_ax.set_ylim(ratio_range)
            
            ax.set_title(label)
            ax.set_yscale('log')
            
        ratio_ax.set_ylabel("Ratio")
        ratio_ax.axhline(1, color='gray', linestyle='--')  # Add a horizontal line at 1 for reference

        # Hide the x-ticks for the main plots to avoid redundancy
        plt.setp(ax.get_xticklabels(), visible=False)

    # Add text to the gs_master[0, 2] panel
    text_ax = fig.add_subplot(gs_master[0, 2])
    text_ax.text(0.5, 0.9, title, fontsize=16, ha='center', va='center')
    # text the datetime from system
    now = datetime.datetime.date(datetime.datetime.now())
    text_ax.text(0.5, 0.7, f'{now}', fontsize=12, ha='center', va='center')
    text_ax.text(0.5, 0.5, title_tiled, fontsize=12, ha='center', va='center')
    text_ax.text(0.5, 0.3, title_bigbox, fontsize=12, ha='center', va='center')
    text_ax.text(0.5, 0.1, title_attrs, fontsize=12, ha='center', va='center')
    text_ax.axis('off')  # Turn off the axis for the text panel

    # Automatically adjust spacing to avoid overlaps
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(fname, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.show()
    plt.close(fig)

def normalize_peak(peak):
    return (peak.T/np.sum(peak, axis=1)).T

def dimensionless_cl(cl, ell):
    return ell * (ell+1) * cl / (2*np.pi)

def dimiensionless_bispectrum(bispec, ell):
    return bispec * ell**4 / (2*np.pi)**2

def stacking_analysis(data_paths, ell):
    data = np.vstack([np.load(data_path) for data_path in data_paths])
    sq, clkk, pdf, peak, minima = np.split(data, 5, axis=1)
    sq = dimiensionless_bispectrum(sq, ell)
    clkk = dimensionless_cl(clkk, ell)
    peak = normalize_peak(peak)
    minima = normalize_peak(minima)
    data = np.hstack([sq, clkk, pdf, peak, minima])

    patch_per_realizations = data.shape[0] // len(data_paths)
    diag= [np.diag(np.cov(data[i*patch_per_realizations: (i+1)*patch_per_realizations, :], rowvar=False)) for i in range(len(data_paths))]
    diags_std = np.std(diag, axis=0)

    diags = np.diag(np.cov(data, rowvar=False))
    corr = np.corrcoef(data, rowvar=False)
    stds = np.std(data, axis=0)
    means = np.mean(data, axis=0)

    return means, stds, diags, corr, diags_std

def main(imgdir, suffix, data_tiled, data_bigbox, ell, nu, zs, patch_size, sboxsize=625, bboxsize=3750):
    means_tiled, stds_tiled, diags_tiled, corr_tiled, diags_std_tiled = stacking_analysis(data_tiled, ell)
    means_bigbox, stds_bigbox, diags_bigbox, corr_bigbox, diags_std_bigbox = stacking_analysis(data_bigbox, ell)

    fname = os.path.join(imgdir, f"correlation_{suffix}.png")
    title = f"Correlation Matrix, Opening Angle: {patch_size}"+r"$^\circ$,"+f" Scale Angle: {sl}"+r"''"+f", Redshift: {zs}"
    title_tiled = f"Tiled, {len(data_tiled)} realizations"
    title_bigbox = f"BigBox, {len(data_bigbox)} realizations"
    labels = [r'$B_{\ell}^\mathrm{sq}$', r'$C^{\kappa\kappa}_{\ell}$', "PDF", "Peaks", "Minima"]
    plot_corr(fname, corr_tiled, corr_bigbox, title, title_tiled, title_bigbox, labels, vmin=-0.15, vmax=0.15)

    fname = os.path.join(imgdir, f"mean_{suffix}.png")
    title = f"Mean of Stacked Statistics"
    title_tiled = f"Tiled: {len(data_tiled)} realizations, BoxSize: {sboxsize} Mpc/h"
    title_bigbox = f"BigBox: {len(data_bigbox)} realizations, BoxSize: {bboxsize} Mpc/h"
    title_attrs = f"Opening Angle: {patch_size}"+r"$^\circ$,"+f" Scale Angle: {sl}"+r"$''$"+f", Redshift: {zs}"
    plot_stats(fname, title, title_tiled, title_bigbox, title_attrs, ell, nu, means_tiled, means_bigbox, labels, stds_tiled, stds_bigbox)

    fname = os.path.join(imgdir, f"diagonal_{suffix}.png")
    title = f"Diagonals of Covariance Matrix"
    plot_stats(fname, title, title_tiled, title_bigbox, title_attrs, ell, nu, diags_tiled, diags_bigbox, labels, diags_std_tiled, diags_std_bigbox, ratio_range=[0.8, 1.2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting the correlation and statistics of the stacked data")
    parser.add_argument("imgdir", type=str, help="Directory to save the images")
    parser.add_argument("config", type=str, help="Configuration file path")
    parser.add_argument("--zs", type=float, default=2.0, help="Redshift of the data")
    parser.add_argument("--noiseless", action="store_true", help="Use noiseless data")
    args = parser.parse_args()

    # Initialize empty config
    config = {}

    # Load configuration from YAML if provided and exists
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as file:
            try:
                config = yaml.safe_load(file)  # Load the configuration from YAML
            except yaml.YAMLError as exc:
                print("Warning: The config file is empty or invalid. Proceeding with default parameters.")
                print(exc)

    oa = config.get("patch_size")
    sl = config.get("scale_angle")
    ngal = config.get("ngal")

    suffix = f"zs{args.zs:.1f}_oa{oa}_sl{sl}"
    if args.noiseless:
        suffix += "_noiseless"
    else:
        suffix += f"_ngal{ngal}"

    nbin = config["nbin"]
    lmin, lmax = config["lmin"], config["lmax"]
    bins = np.linspace(-4, 4, nbin+1, endpoint=True)
    l_edges = np.linspace(lmin, lmax, nbin+1, endpoint=True)

    ell = (l_edges[1:] + l_edges[:-1]) / 2
    nu = (bins[1:] + bins[:-1]) / 2

    data_tiled = glob(os.path.join(config["storagedir"], "tiled", "*", "flat", f"analysis_sqclpdpm_*_{suffix}.npy"))
    data_bigbox = glob(os.path.join(config["storagedir"], "bigbox", "*", "flat", f"analysis_sqclpdpm_*_{suffix}.npy"))

    print("Tiled data found:", len(data_tiled))
    print("BigBox data found:", len(data_bigbox))

    main(args.imgdir, suffix, data_tiled, data_bigbox, ell, nu, args.zs, oa)