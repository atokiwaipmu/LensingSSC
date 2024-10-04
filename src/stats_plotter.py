
import os
import numpy as np
import healpy as hp
import logging
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from src.info_extractor import InfoExtractor


class StatsPlotter:
    def __init__(self, sl, ngal, data_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/output", oa=10, zs_list=[0.5, 1.0, 2.0], lmin=300, lmax=3000, nbin = 15, save_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/plot"):
        self.data_dir = data_dir
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.sl = sl
        self.ngal = ngal
        self.oa = oa
        self.zs_list = zs_list
        self.colors = {zs: color for zs, color in zip(zs_list, ["tab:blue", "tab:orange", "tab:green", "tab:red"])}
        self.labels = [r'$B_{\ell}^\mathrm{sq}$', 
          r'$C^{\kappa\kappa}_{\ell}$', 
          "PDF",
          "Peaks",
          "Minima"]

        self.l_edges = np.logspace(np.log10(lmin), np.log10(lmax), nbin + 1, endpoint=True)
        self.bins = np.linspace(-4, 4, nbin + 1, endpoint=True)

        self.ell = (self.l_edges[1:] + self.l_edges[:-1]) / 2
        self.nu = (self.bins[1:] + self.bins[:-1]) / 2

    def plot(self):
        #self.plot_stats(is_patch=False)
        self.plot_stats(is_patch=True)

    def plot_stats(self, is_patch=False):
        stats_tiled = self._load_stats(is_patch, box_type='tiled')
        stats_bigbox = self._load_stats(is_patch, box_type='bigbox')

        fig_mean, ax_mean = self._prepare_fig(is_patch, target='mean')
        fig_diag, ax_diag = self._prepare_fig(is_patch, target='diag')
        for zs in self.zs_list:
            self._plot_mean(stats_tiled, stats_bigbox, zs, ax_mean, is_patch)
            self._plot_diag(stats_tiled, stats_bigbox, zs, ax_diag, is_patch)

        suffix = self._generate_suffix(is_patch)
        prefix = "patch" if is_patch else "fullsky"
        fig_mean.savefig(os.path.join(self.save_dir, prefix + "_mean_" + suffix + ".png"), bbox_inches='tight')
        fig_diag.savefig(os.path.join(self.save_dir, prefix + "_diag_" + suffix + ".png"), bbox_inches='tight')
 
    def _load_stats(self, is_patch=False, box_type='tiled'):
        suffix = self._generate_suffix(is_patch)
        fname = f"fullsky_stats_{box_type}_{suffix}.npy" if not is_patch else f"patch_stats_{box_type}_{suffix}.npy"
        load_path = os.path.join(self.data_dir, fname)
        print(f"Loading stats from {os.path.basename(load_path)}")
        return np.load(load_path, allow_pickle=True).item()

    def _generate_suffix(self, is_patch=False):
        suffix = f"oa{self.oa}_" if is_patch else ""
        suffix += f"noiseless_sl{self.sl}"
        if self.ngal != 0:
            suffix = suffix.replace("noiseless", f"ngal{self.ngal}")
        return suffix

    def _prepare_fig(self, is_patch=False, target='mean'):
        fig = plt.figure(figsize=(15, 6))

        gs_master = GridSpec(nrows=2, ncols=3, height_ratios=[1, 1], hspace=0.3)
        gs_ell = [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[0, j], height_ratios=[3, 1], hspace=0.01) for j in range(2)]
        gs_nu = [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[1, j], height_ratios=[3, 1], hspace=0.01) for j in range(3)]

        axes_ell = [plt.subplot(gs_ell[j][0]) for j in range(2)]
        axes_nu = [plt.subplot(gs_nu[j][0]) for j in range(3)]
        axes_ratio_ell = [plt.subplot(gs_ell[j][1]) for j in range(2)]
        axes_ratio_nu = [plt.subplot(gs_nu[j][1]) for j in range(3)]
        axes_legend = plt.subplot(gs_master[0, 2])

        for ax in axes_ell:
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xticks([300, 500, 1000, 2000, 3000])
            ax.set_xticklabels([])

        for ax in axes_nu:
            ax.set_yscale('log')
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_xticklabels([])

        for ax in axes_ratio_ell:
            ax.set_xscale('log')
            ax.set_xticks([300, 500, 1000, 2000, 3000])
            ax.set_xticklabels([300, 500, 1000, 2000, 3000])
            if target == 'mean':
                ax.set_ylim(0.8, 1.2)
            elif target == 'diag' and is_patch:
                ax.set_ylim(0.8, 1.2)
            elif target == 'diag' and not is_patch:
                ax.set_ylim(0, 3)

        for ax in axes_ratio_nu:
            ax.set_xticks([-4, -2, 0, 2, 4])
            if target == 'mean':
                ax.set_ylim(0.8, 1.2)
            elif target == 'diag' and is_patch:
                ax.set_ylim(0.8, 1.2)
            elif target == 'diag' and not is_patch:
                ax.set_ylim(0, 3)

        if not is_patch:
            axes_ell[1].axis('off')
            axes_ratio_ell[1].axis('off')

        ax = [axes_ell, axes_nu, axes_ratio_ell, axes_ratio_nu, axes_legend]

        return fig, ax

    def _plot_mean(self, stats_tiled, stats_bigbox, zs, ax, is_patch=False, box_type='tiled'):
        axes_ell, axes_nu, axes_ratio_ell, axes_ratio_nu, axes_legend = ax
        n_stats = 5 if is_patch else 4
        ell_switch = 2 if is_patch else 1

        means_tiled, stds_tiled = stats_tiled[zs]['means'], stats_tiled[zs]['stds']
        datas_tiled = np.split(means_tiled, n_stats)
        data_stds_tiled = np.split(stds_tiled, n_stats)

        means_bigbox, stds_bigbox = stats_bigbox[zs]['means'], stats_bigbox[zs]['stds']
        datas_bigbox = np.split(means_bigbox, n_stats)
        data_stds_bigbox = np.split(stds_bigbox, n_stats)

        for i in range(len(datas_tiled)):
            if i < ell_switch:
                axes_ell[i].errorbar(self.ell, datas_bigbox[i], yerr=data_stds_bigbox[i], label=f"Bigbox: zs={zs}", color=self.colors[zs])
                axes_ell[i].errorbar(self.ell, datas_tiled[i], yerr=data_stds_tiled[i], label=f"Tiled: zs={zs}", color=self.colors[zs], linestyle='--')

                axes_ratio_ell[i].plot(self.ell, datas_tiled[i] / datas_bigbox[i], color=self.colors[zs])
                axes_ratio_ell[i].hlines(1, 300, 3000, color='k', linestyle='--')

                axes_ell[i].set_title(self.labels[-n_stats + i])
            else:
                k = i - ell_switch
                axes_nu[k].errorbar(self.nu, datas_bigbox[i], yerr=data_stds_bigbox[i], label=f"Bigbox: zs={zs}", color=self.colors[zs])
                axes_nu[k].errorbar(self.nu, datas_tiled[i], yerr=data_stds_tiled[i], label=f"Tiled: zs={zs}", color=self.colors[zs], linestyle='--')

                axes_ratio_nu[k].plot(self.nu, datas_tiled[i] / datas_bigbox[i], color=self.colors[zs])
                axes_ratio_nu[k].hlines(1, -4, 4, color='k', linestyle='--')

                axes_nu[k].set_title(self.labels[-n_stats + i])

        axes_legend.legend(*axes_ell[0].get_legend_handles_labels(), loc='center', fontsize=12)
        axes_legend.axis('off')

    def _plot_diag(self, stats_tiled, stats_bigbox, zs, ax, is_patch=False, box_type='tiled'):
        axes_ell, axes_nu, axes_ratio_ell, axes_ratio_nu, axes_legend = ax
        n_stats = 5 if is_patch else 4
        ell_switch = 2 if is_patch else 1

        diags_tiled = stats_tiled[zs]['diags']
        datas_tiled = np.split(diags_tiled, n_stats)

        diags_bigbox = stats_bigbox[zs]['diags']
        datas_bigbox = np.split(diags_bigbox, n_stats)

        for i in range(len(datas_tiled)):
            if i < ell_switch:
                axes_ell[i].plot(self.ell, datas_bigbox[i], label=f"Bigbox: zs={zs}", color=self.colors[zs])
                axes_ell[i].plot(self.ell, datas_tiled[i], label=f"Tiled: zs={zs}", color=self.colors[zs], linestyle='--')

                axes_ratio_ell[i].plot(self.ell, datas_tiled[i] / datas_bigbox[i], color=self.colors[zs])
                axes_ratio_ell[i].hlines(1, 300, 3000, color='k', linestyle='--')

                axes_ell[i].set_title(self.labels[-n_stats + i])
            else:
                k = i - ell_switch
                axes_nu[k].plot(self.nu, datas_bigbox[i], label=f"Bigbox: zs={zs}", color=self.colors[zs])
                axes_nu[k].plot(self.nu, datas_tiled[i], label=f"Tiled: zs={zs}", color=self.colors[zs], linestyle='--')

                axes_ratio_nu[k].plot(self.nu, datas_tiled[i] / datas_bigbox[i], color=self.colors[zs])
                axes_ratio_nu[k].hlines(1, -4, 4, color='k', linestyle='--')

                axes_nu[k].set_title(self.labels[-n_stats + i])

        axes_legend.legend(*axes_ell[0].get_legend_handles_labels(), loc='center', fontsize=12)
        axes_legend.axis('off')

def plot_kappa(files, datadir):
    zs_order = [0.5, 1, 2]
    noise_order = np.array(["noiseless", "ngal50", "ngal30", "ngal15", "ngal7"])
    smoothing_order = np.array([None, 2, 5, 8, 10])

    # 3 redshift x (4 smoothing scale x 5 noise scale) panels
    fig = plt.figure(figsize=(15, 5))
    gs_master = GridSpec(1, 3, figure=fig, wspace=0.1)
    gs_zs05 = GridSpecFromSubplotSpec(5, 5, subplot_spec=gs_master[0], wspace=0.1, hspace=0)
    gs_zs1 = GridSpecFromSubplotSpec(5, 5, subplot_spec=gs_master[1], wspace=0.1, hspace=0)
    gs_zs2 = GridSpecFromSubplotSpec(5, 5, subplot_spec=gs_master[2], wspace=0.1, hspace=0)

    for file in files:
        data = np.load(file, mmap_mode="r")
        info = InfoExtractor.extract_info_from_path(file)
        zs = info["redshift"]
        sl = info["sl"]
        sl_idx = np.where(smoothing_order == sl)[0][0]
        ngal = info["ngal"]
        noise = f"ngal{ngal}" if ngal != 0 else "noiseless"
        noise_idx = np.where(noise_order == noise)[0][0]

        if zs == 0.5:
            gs = gs_zs05
        elif zs == 1:
            gs = gs_zs1
        else:
            gs = gs_zs2

        ax = fig.add_subplot(gs[sl_idx, noise_idx])
        ax.imshow(data[0], vmin=-0.024, vmax=0.024)
        #ax.set_title(f"{zs}, {sl}, {ngal}", fontsize=8)
        ax.axis("off")

    output_path = os.path.join(datadir, "img", "patches.png")
    fig.savefig(output_path, bbox_inches="tight")

    plt.close()

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

if __name__ == "__main__":
    sl_list = [2, 5, 8, 10]
    ngal_list = [0, 7, 15, 30, 50]
    for sl in sl_list:
        for ngal in ngal_list:
            plotter = StatsPlotter(sl, ngal)
            plotter.plot()

    """sample
    data_tiled_paths = glob.glob("/lustre/work/akira.tokiwa/Projects/LensingSSC/output/patch_stats_tiled*.npy")
    for data_tiled_path in data_tiled_paths:
        data_bigbox_path = data_tiled_path.replace("tiled", "bigbox")

        data_tiled = np.load(data_tiled_path, allow_pickle=True).item()
        data_bigbox = np.load(data_bigbox_path, allow_pickle=True).item()
        info = InfoExtractor.extract_info_from_path(data_tiled_path)
        sl = info["sl"]
        oa = info["oa"]
        ngal = info["ngal"]
        noise = f"ngal{ngal}" if ngal != 0 else "noiseless"

        for zs in [0.5, 1.0, 2.0]:
            fname = os.path.join("/lustre/work/akira.tokiwa/Projects/LensingSSC/plot/corr", f"corr_zs{zs}_oa{oa}_{noise}_sl{sl}.png")
            title = f"Correlation Matrix, Opening Angle: {oa}"+r"$^\circ$,"+f" Scale Angle: {sl}"+r"''"+f", Redshift: {zs}"
            title_tiled = f"Tiled, 20 realizations"
            title_bigbox = f"BigBox, 11 realizations"
            labels = [r'$B_{\ell}^\mathrm{sq}$', r'$C^{\kappa\kappa}_{\ell}$', "PDF", "Peaks", "Minima"]

            corr_tiled = data_tiled[zs]["corr"]
            corr_bigbox = data_bigbox[zs]["corr"]

            plot_corr(fname, corr_tiled, corr_bigbox, title, title_tiled, title_bigbox, labels)
        
    """
