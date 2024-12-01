
import os
import numpy as np
import healpy as hp
import logging
from matplotlib import axes, pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from src.info_extractor import InfoExtractor

def reorder_correlation_matrix(corr_matrix, stats_names, desired_order):
    """
    Reorders a correlation matrix based on a desired order of statistics.

    Parameters:
    - corr_matrix (np.ndarray): The original correlation matrix of shape (157, 157).
    - stats_names (dict): Mapping from statistic names to [start, end] indices.
    - desired_order (list of str): The desired order of statistic names.

    Returns:
    - np.ndarray: The reordered correlation matrix.
    """
    # Initialize list to hold the new order of indices
    permuted_indices = []

    # Iterate over each statistic in the desired order
    for stat in desired_order:
        if stat not in stats_names:
            raise ValueError(f"Statistic '{stat}' not found in stats_names mapping.")
        
        start, end = stats_names[stat]
        # Append all indices for this statistic
        stat_indices = list(range(start, end))
        permuted_indices.extend(stat_indices)
    
    # Verify that all indices are included
    total_indices = corr_matrix.shape[0]
    if len(permuted_indices) != total_indices:
        raise ValueError(
            f"The desired order includes {len(permuted_indices)} indices, "
            f"but the correlation matrix has {total_indices} indices."
        )
    
    # Check for duplicate or missing indices
    if sorted(permuted_indices) != list(range(total_indices)):
        missing = set(range(total_indices)) - set(permuted_indices)
        duplicates = [idx for idx in permuted_indices if permuted_indices.count(idx) > 1]
        error_message = ""
        if missing:
            error_message += f"Missing indices: {missing}. "
        if duplicates:
            error_message += f"Duplicate indices: {set(duplicates)}."
        raise ValueError(f"Permutation indices are invalid. {error_message}")
    
    # Reorder the correlation matrix
    reordered_corr_matrix = corr_matrix[np.ix_(permuted_indices, permuted_indices)]
    
    return reordered_corr_matrix

class FinalPlotter:
    def __init__(self, 
                 sl_main = 2,
                 ngal_main = 0,
                 sl_list = [2, 5, 8, 10],
                 ngal_list = [0, 7, 15, 30, 50], 
                 data_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/output", 
                 oa=10, 
                 zs_list=[0.5, 1.0, 1.5, 2.0, 2.5], 
                 lmin=300, lmax=3000, nbin = 15, 
                 maincmap="viridis", subcmap='cividis',
                 save_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/plot"):
        self.data_dir = data_dir
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.sl_main = sl_main
        self.ngal_main = ngal_main
        self.sl_list = sl_list
        self.ngal_list = ngal_list
        self.oa = oa
        self.zs_list = zs_list
        self.colors = {zs: color for zs, color in zip(zs_list, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])}
        self.labels = [
            r"$C^{\kappa\kappa}_{\ell}$",
            r'$B_{\ell}^\mathrm{sq}$',  
            r'$B_{\ell}^\mathrm{iso}$',
            r'$B_{\ell}^\mathrm{eq}$', 
            "Moment",
            "PDF",
            "Peak",
            "Min",
            r"$Area_\mathrm{MF}$",
            r"$Perim_\mathrm{MF}$",
            r"$Genus_\mathrm{MF}$",
        ]

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

        self.nbin = nbin
        self.l_edges = np.logspace(np.log10(lmin), np.log10(lmax), nbin + 1, endpoint=True)
        self.bins = np.linspace(-4, 4, nbin + 1, endpoint=True)

        self.ell = (self.l_edges[1:] + self.l_edges[:-1]) / 2
        self.nu = (self.bins[1:] + self.bins[:-1]) / 2

        self.maincmap = maincmap
        self.subcmap = subcmap

    def prepare_permuted_indices(self):
        permuted_indices = []
        for stat in self.stats_desired_order:
            if stat not in self.stats_name_indices:
                raise ValueError(f"Statistic '{stat}' not found in stats_name_indices mapping.")
            
            start, end = self.stats_name_indices[stat]
            stat_indices = list(range(start, end))
            permuted_indices.extend(stat_indices)
        
        return permuted_indices

    def plot(self):
        self.plot_mean()

    def _prepare_fig(self):
        fig = plt.figure(figsize=(14, 8))
        gs_master = GridSpec(nrows=2, ncols=5, height_ratios=[1, 1], hspace=0.3)
        gs_ell = [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[0, j], height_ratios=[3, 1], hspace=0.01) for j in range(5)]
        gs_nu = [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[1, j], height_ratios=[3, 1], hspace=0.01) for j in range(5)]

        axes_ell = [plt.subplot(gs_ell[j][0]) for j in range(4)]
        axes_nu = [plt.subplot(gs_ell[4][0])] + [plt.subplot(gs_nu[j][0]) for j in range(5)]
        axes_ratio_ell = [plt.subplot(gs_ell[j][1]) for j in range(4)]
        axes_ratio_nu = [plt.subplot(gs_ell[4][1])] + [plt.subplot(gs_nu[j][1]) for j in range(5)]

        for ax in axes_ell + axes_ratio_ell:
            ax.set_xscale('log')
            ax.set_xticks([300, 500, 1000, 2000, 3000])

        for ax in axes_nu + axes_ratio_nu:
            ax.set_xticks([-4, -2, 0, 2, 4])

        for ax in axes_ratio_ell + axes_ratio_nu:
            ax.set_ylim(0.8, 1.2)

        for ax in axes_ell + axes_nu:
            ax.set_yscale('log')
            ax.set_xticklabels([])
        
        for ax in axes_ratio_nu:
            ax.hlines(1, -4, 4, color='k', linestyle='--')

        for ax in axes_ratio_ell:
            ax.hlines(1, 300, 3000, color='k', linestyle='--')

        ax = [axes_ell, axes_nu, axes_ratio_ell, axes_ratio_nu]

        return fig, ax
    
    def _load_stats(self, sl, ngal, box_type='tiled'):
        suffix = self._generate_suffix(sl, ngal)
        fname = f"patch_stats_{box_type}_{suffix}.npy"
        load_path = os.path.join(self.data_dir, fname)
        print(f"Loading stats from {os.path.basename(load_path)}")
        return np.load(load_path, allow_pickle=True).item()
    
    def _generate_suffix(self, sl, ngal):
        suffix = f"oa{self.oa}_noiseless_sl{sl}" 
        if self.ngal != 0:
            suffix = suffix.replace("noiseless", f"ngal{ngal}")
        return suffix

    def plot_mean(self):
        fig, ax = self._prepare_fig()
        axes_ell, axes_nu, axes_ratio_ell, axes_ratio_nu = ax
        stats_tiled = self._load_stats(self.sl_main, self.ngal_main, box_type='tiled')
        stats_bigbox = self._load_stats(self.sl_main, self.ngal_main, box_type='bigbox')
        for zs in self.zs_list:
            means_tiled, stds_tiled = stats_tiled[zs]['means'][self.permuted_indices], stats_tiled[zs]['stds'][self.permuted_indices]
            means_bigbox, stds_bigbox = stats_bigbox[zs]['means'][self.permuted_indices], stats_bigbox[zs]['stds'][self.permuted_indices]

            for i, label in enumerate(self.stats_desired_order):
                if label in ['angular power spectrum','squeezed bispectrum','isosceles bispectrum','equilateral bispectrum']:
                    axes_ell[i].errorbar(self.ell, means_bigbox[self.permuted_stats_name_indices[label]], yerr=stds_bigbox[self.permuted_stats_name_indices[label]], label=f"Bigbox: zs={zs}", color=self.colors[zs])
                    axes_ell[i].errorbar(self.ell, means_tiled[self.permuted_stats_name_indices[label]], yerr=stds_tiled[self.permuted_stats_name_indices[label]], label=f"Tiled: zs={zs}", color=self.colors[zs], linestyle='--')

                    axes_ratio_ell[i].plot(self.ell, means_tiled[self.permuted_stats_name_indices[label]] / means_bigbox[self.permuted_stats_name_indices[label]], color=self.colors[zs])
                    axes_ell[i].set_title(self.labels[i])                   
                elif label in ['PDF','Peak','Minima','area(MFs)','perimeter(MFs)','genus(MFs)']:
                    k = i - len(self.stats_name_indices)
                    axes_nu[k].errorbar(self.nu, means_bigbox[self.permuted_stats_name_indices[label]], yerr=stds_bigbox[self.permuted_stats_name_indices[label]], label=f"Bigbox: zs={zs}", color=self.colors[zs])
                    axes_nu[k].errorbar(self.nu, means_tiled[self.permuted_stats_name_indices[label]], yerr=stds_tiled[self.permuted_stats_name_indices[label]], label=f"Tiled: zs={zs}", color=self.colors[zs], linestyle='--')

                    axes_ratio_nu[k].plot(self.nu, means_tiled[self.permuted_stats_name_indices[label]] / means_bigbox[self.permuted_stats_name_indices[label]], color=self.colors[zs])
                    axes_nu[k].set_title(self.labels[k])
                            
        fig.savefig(os.path.join(self.save_dir, "mean_final.png"), bbox_inches='tight')
        plt.close(fig)

    def plot_diag(self):
        fig, ax = self._prepare_fig()
        axes_ell, axes_nu, axes_ratio_ell, axes_ratio_nu = ax
        stats_tiled = self._load_stats(self.sl_main, self.ngal_main, box_type='tiled')
        stats_bigbox = self._load_stats(self.sl_main, self.ngal_main, box_type='bigbox')
        for zs in self.zs_list:
            diags_tiled = stats_tiled[zs]['diags'][self.permuted_indices]
            diags_bigbox = stats_bigbox[zs]['diags'][self.permuted_indices]

            for i, label in enumerate(self.stats_desired_order):
                if label in ['angular power spectrum','squeezed bispectrum','isosceles bispectrum','equilateral bispectrum']:
                    axes_ell[i].plot(self.ell, diags_bigbox[self.permuted_stats_name_indices[label]], label=f"Bigbox: zs={zs}", color=self.colors[zs])
                    axes_ell[i].plot(self.ell, diags_tiled[self.permuted_stats_name_indices[label]], label=f"Tiled: zs={zs}", color=self.colors[zs], linestyle='--')

                    axes_ratio_ell[i].plot(self.ell, diags_tiled[self.permuted_stats_name_indices[label]] / diags_bigbox[self.permuted_stats_name_indices[label]], color=self.colors[zs])
                    axes_ell[i].set_title(self.labels[i])                   
                elif label in ['PDF','Peak','Minima','area(MFs)','perimeter(MFs)','genus(MFs)']:
                    k = i - len(self.stats_name_indices)
                    axes_nu[k].plot(self.nu, diags_bigbox[self.permuted_stats_name_indices[label]], label=f"Bigbox: zs={zs}", color=self.colors[zs])
                    axes_nu[k].plot(self.nu, diags_tiled[self.permuted_stats_name_indices[label]], label=f"Tiled: zs={zs}", color=self.colors[zs], linestyle='--')

                    axes_ratio_nu[k].plot(self.nu, diags_tiled[self.permuted_stats_name_indices[label]] / diags_bigbox[self.permuted_stats_name_indices[label]], color=self.colors[zs])
                    axes_nu[k].set_title(self.labels[k])
                            
        fig.savefig(os.path.join(self.save_dir, "diag_final.png"), bbox_inches='tight')
        plt.close(fig)

    def plot_corr_noiseless(self):
        fig = plt.figure(figsize=(35, 14))
        gs_master = GridSpec(nrows=2, ncols=5, height_ratios=[1, 1], hspace=0.1)
        axes_merged = [plt.subplot(gs_master[0, j]) for j in range(5)]
        axes_diff = [plt.subplot(gs_master[1, j]) for j in range(5)]

        for ax in axes_merged + axes_diff:
            ax.axis('off')

        stats_tiled = self._load_stats(self.sl_main, self.ngal_main, box_type='tiled')
        stats_bigbox = self._load_stats(self.sl_main, self.ngal_main, box_type='bigbox')
        for i, zs in enumerate(self.zs_list):
            corr_tiled = stats_tiled[zs]['corr'][np.ix_(self.permuted_indices, self.permuted_indices)]
            corr_bigbox = stats_bigbox[zs]['corr'][np.ix_(self.permuted_indices, self.permuted_indices)]
            corr_merged = merge_corr(corr_tiled, corr_bigbox)
            corr_diff = corr_bigbox - corr_tiled

            cax = axes_merged[i].imshow(corr_merged, cmap=self.maincmap, vmin=-1, vmax=1)
            fig.colorbar(cax, ax=axes_merged[i], shrink=0.6)

            cax = axes_diff[i].imshow(corr_diff, cmap=self.subcmap, vmin=-0.3, vmax=0.3)
            fig.colorbar(cax, ax=axes_diff[i], shrink=0.6)

            axes_merged[i].set_title(f"zs={zs}")
            axes_diff[i].set_title(f"zs={zs}")

        fig.savefig(os.path.join(self.save_dir, "corr_noiseless_final.png"), bbox_inches='tight')
        plt.close(fig)

    def plot_corr_ngal(self):
        fig = plt.figure(figsize=(35, 35))
        gs_master = GridSpec(nrows=5, ncols=5, height_ratios=[1, 1], hspace=0.1)

        for i, ngal in enumerate(self.ngal_list):
            stats_tiled = self._load_stats(self.sl_main, ngal, box_type='tiled')
            stats_bigbox = self._load_stats(self.sl_main, ngal, box_type='bigbox')
            for j, zs in enumerate(self.zs_list):
                corr_tiled = stats_tiled[zs]['corr'][np.ix_(self.permuted_indices, self.permuted_indices)]
                corr_bigbox = stats_bigbox[zs]['corr'][np.ix_(self.permuted_indices, self.permuted_indices)]
                corr_diff = corr_bigbox - corr_tiled

                axes_merged = plt.subplot(gs_master[i, j])
                cax = axes_merged.imshow(merge_corr(corr_tiled, corr_bigbox), cmap=self.maincmap, vmin=-0.3, vmax=0.3)
                fig.colorbar(cax, ax=axes_merged, shrink=0.6)
                axes_merged.set_title(f"ngal={ngal}, zs={zs}")

        fig.savefig(os.path.join(self.save_dir, "corr_ngal_final.png"), bbox_inches='tight')
        plt.close(fig)

class StatsPlotter:
    def __init__(self, sl, ngal, data_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/output", oa=10, zs_list=[0.5, 1.0, 1.5, 2.0, 2.5], lmin=300, lmax=3000, nbin = 15, save_dir="/lustre/work/akira.tokiwa/Projects/LensingSSC/plot"):
        self.data_dir = data_dir
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.sl = sl
        self.ngal = ngal
        self.oa = oa
        self.zs_list = zs_list
        self.colors = {zs: color for zs, color in zip(zs_list, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])}
        self.labels = [
            r"$C^{\kappa\kappa}_{\ell}$",
            r'$B_{\ell}^\mathrm{sq}$',  
            r'$B_{\ell}^\mathrm{iso}$',
            r'$B_{\ell}^\mathrm{eq}$', 
            "Moment",
            "PDF",
            "Peak",
            "Min",
            r"$Area_\mathrm{MF}$",
            r"$Perim_\mathrm{MF}$",
            r"$Genus_\mathrm{MF}$",
        ]

        self.nbin = nbin
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
            self._plot_corr(stats_tiled, stats_bigbox, zs, is_patch=is_patch)

        suffix = self._generate_suffix(is_patch)
        prefix = "patch" if is_patch else "fullsky"
        fig_mean.savefig(os.path.join(self.save_dir, prefix + "_mean_" + suffix + ".png"), bbox_inches='tight')
        fig_diag.savefig(os.path.join(self.save_dir, prefix + "_diag_" + suffix + ".png"), bbox_inches='tight')

        plt.close(fig_mean)
        plt.close(fig_diag)
 
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
        fig = plt.figure(figsize=(20, 10))

        gs_master = GridSpec(nrows=3, ncols=4, height_ratios=[1, 1, 1], hspace=0.3)
        gs_ell = [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[0, j], height_ratios=[3, 1], hspace=0.01) for j in range(4)]
        gs_nu = [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[1, j], height_ratios=[3, 1], hspace=0.01) for j in range(3)] + [GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[2, j], height_ratios=[3, 1], hspace=0.01) for j in range(3)]

        axes_ell = [plt.subplot(gs_ell[j][0]) for j in range(4)]
        axes_nu = [plt.subplot(gs_nu[j][0]) for j in range(6)]
        axes_ratio_ell = [plt.subplot(gs_ell[j][1]) for j in range(4)]
        axes_ratio_nu = [plt.subplot(gs_nu[j][1]) for j in range(6)]
        axes_legend = plt.subplot(gs_master[2, 3])

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
        n_stats = 10 if is_patch else 4
        ell_switch = 4 if is_patch else 1

        means_tiled, stds_tiled = stats_tiled[zs]['means'][:-7], stats_tiled[zs]['stds'][:-7]
        datas_tiled = np.split(means_tiled, n_stats)
        data_stds_tiled = np.split(stds_tiled, n_stats)

        means_bigbox, stds_bigbox = stats_bigbox[zs]['means'][:-7], stats_bigbox[zs]['stds'][:-7]
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
        n_stats = 10 if is_patch else 4
        ell_switch = 4 if is_patch else 1

        diags_tiled = stats_tiled[zs]['diags'][:-7]
        datas_tiled = np.split(diags_tiled, n_stats)

        diags_bigbox = stats_bigbox[zs]['diags'][:-7]
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
    
    def _plot_corr(self, stats_tiled, stats_bigbox, zs, vmin=-0.3, vmax=0.3, is_patch=True):    
        corr_tiled = stats_tiled[zs]['corr']
        corr_bigbox = stats_bigbox[zs]['corr']

        n_stats = 10 if is_patch else 4
        tick_positions = [self.nbin/2 + self.nbin * i for i in range(n_stats)]

        fig = plt.figure(figsize=(18, 6))
        gs_master = GridSpec(nrows=2, ncols=1, height_ratios=[1, 9])
        gs_plot = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_master[1], wspace=0.2)
        ax = [fig.add_subplot(gs_plot[i]) for i in range(3)]

        cax = ax[0].imshow(corr_tiled, cmap='bwr', vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax[0], shrink=0.6)
        ax[0].set_title("Tiled", fontsize=10)

        cax = ax[1].imshow(corr_bigbox, cmap='bwr', vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax[1], shrink=0.6)
        ax[1].set_title("Bigbox", fontsize=10)

        cax = ax[2].imshow(corr_bigbox - corr_tiled, cmap='bwr', vmin=vmin, vmax=vmax)
        fig.colorbar(cax, ax=ax[2], shrink=0.6)
        ax[2].set_title("BigBox - Tiled", fontsize=10)

        for axes in ax:
            axes.set_xticks(tick_positions, self.labels, fontsize=8)
            axes.set_yticks(tick_positions, self.labels, fontsize=8, rotation=90, va='center')
            axes.invert_yaxis()

        title = "Correlation Matrix"+f": Scale Angle: {sl}"+r"''"+f", Redshift: {zs}"
        title += f", ngal={ngal}" if ngal != 0 else ", noiseless"
        ax_title = fig.add_subplot(gs_master[0])
        ax_title.text(0.5, 0.5, title, fontsize=12, ha='center', va='center')
        ax_title.axis('off')

        suffix = self._generate_suffix(is_patch)
        prefix = "patch" if is_patch else "fullsky"
        fname = os.path.join(self.save_dir, prefix + "_corr_" + f"_zs{zs}" + suffix + ".png")

        fig.savefig(fname, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.show()
        plt.close(fig)

    

def plot_kappa(files, datadir):
    zs_order = [0.5, 1, 1.5, 2, 2.5]
    noise_order = np.array(["noiseless", "ngal50", "ngal30", "ngal15", "ngal7"])
    smoothing_order = np.array([None, 2, 5, 8, 10])

    # 5 redshift x (4 smoothing scale x 5 noise scale) panels
    fig = plt.figure(figsize=(5, 25))
    gs_master = GridSpec(5, 1, figure=fig, wspace=0.1)
    gs_zs05 = GridSpecFromSubplotSpec(5, 5, subplot_spec=gs_master[0], wspace=0.1, hspace=0)
    gs_zs1 = GridSpecFromSubplotSpec(5, 5, subplot_spec=gs_master[1], wspace=0.1, hspace=0)
    gs_zs15 = GridSpecFromSubplotSpec(5, 5, subplot_spec=gs_master[2], wspace=0.1, hspace=0)
    gs_zs2 = GridSpecFromSubplotSpec(5, 5, subplot_spec=gs_master[3], wspace=0.1, hspace=0)
    gs_zs25 = GridSpecFromSubplotSpec(5, 5, subplot_spec=gs_master[4], wspace=0.1, hspace=0)

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
        elif zs == 1.5:
            gs = gs_zs15
        elif zs == 2:
            gs = gs_zs2
        else:
            gs = gs_zs25

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

def merge_corr(corr1, corr2):
    if corr1.shape != corr2.shape or corr1.ndim != 2 or corr1.shape[0] != corr1.shape[1]:
        raise ValueError("Both correlation matrices must be square and have the same shape.")
    
    merged_corr = np.zeros_like(corr1)
    upper_indices = np.triu_indices_from(corr1, k=1)
    lower_indices = np.tril_indices_from(corr1, k=-1)
    
    merged_corr[upper_indices] = corr1[upper_indices]
    merged_corr[lower_indices] = corr2[lower_indices]
    np.fill_diagonal(merged_corr, 1.0)
    
    return merged_corr

if __name__ == "__main__":
    
    sl_list = [2, 5, 8, 10]
    ngal_list = [0, 7, 15, 30, 50]
    for sl in sl_list:
        for ngal in ngal_list:
            plotter = StatsPlotter(sl, ngal)
            plotter.plot()
    """
    from src.utils import find_data_dirs
    import glob
    data_dirs = find_data_dirs()
    for datadir in data_dirs:
        patch_snr = glob.glob(os.path.join(datadir, "patch_snr", "*.npy"))
        patch_kappa = glob.glob(os.path.join(datadir, "patch_kappa", "*.npy"))
        files = patch_snr + patch_kappa
        plot_kappa(files, datadir)
    """

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
