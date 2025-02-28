from .base import BasePlotter
from .constants import PlotConstants

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class StatisticsPlotter(BasePlotter):
    def __init__(self, labels, zs_list, colors, config):
        super().__init__(labels, config)
        self.zs_list = zs_list
        self.colors = colors
        self._initialize_axes()

    def _initialize_axes(self):
        self.setup_figure()
        self.setup_grid()
        
        self.ax_upper0 = [self.fig.add_subplot(self.gs_upper[i][0]) 
                         for i in range(len(self.labels))]
        self.ax_upper1 = [self.fig.add_subplot(self.gs_upper[i][1]) 
                         for i in range(len(self.labels))]
        self.ax_lower0 = [self.fig.add_subplot(self.gs_lower[i][0]) 
                         for i in range(len(self.labels))]
        self.ax_lower1 = [self.fig.add_subplot(self.gs_lower[i][1]) 
                         for i in range(len(self.labels))]

    def plot(self, x_data, stats_tiled, stats_bigbox, theory, **kwargs):
        for i, zs_i in enumerate(self.zs_list):
            zs = str(zs_i)
            for j, label in enumerate(self.labels):
                self._plot_statistics(
                    x_data, stats_tiled, stats_bigbox, theory,
                    zs, label, i, j, **kwargs
                )
        
        self._configure_axes(x_data)
        self._add_legend()

    def _plot_statistics(self, x_data, stats_tiled, stats_bigbox, theory,
                        zs, label, i, j, **kwargs):
        reverse = self.config.plot_type == PlotType.NU and label == "minima"
        
        def process_data(data):
            return data[::-1] if reverse else data

        # Plot theory lines if available
        if label in theory:
            self._plot_theory(x_data, theory, zs, label, i, j)

        # Plot statistics
        for data_type, marker, ls in [('TILED', '^', '--'), ('BIGBOX', 'o', '-')]:
            data = stats_tiled if data_type == 'TILED' else stats_bigbox
            means = process_data(data[zs][kwargs.get('ngal')][kwargs.get('sl', '')][label]["means"])
            stds = process_data(data[zs][kwargs.get('ngal')][kwargs.get('sl', '')][label]["stds"])
            diags = process_data(data[zs][kwargs.get('ngal')][kwargs.get('sl', '')][label]["diags"])

            self.ax_upper0[j].errorbar(x_data, means, yerr=stds, 
                                     fmt=marker, color=self.colors[i],
                                     alpha=0.5, linestyle=ls)
            self.ax_upper1[j].plot(x_data, diags, color=self.colors[i],
                                 marker=marker, linestyle=ls)

        # Plot ratios
        means_ratio = process_data(stats_bigbox[zs][kwargs.get('ngal')][kwargs.get('sl', '')][label]["means"]) / \
                     process_data(stats_tiled[zs][kwargs.get('ngal')][kwargs.get('sl', '')][label]["means"])
        diags_ratio = process_data(stats_bigbox[zs][kwargs.get('ngal')][kwargs.get('sl', '')][label]["diags"]) / \
                     process_data(stats_tiled[zs][kwargs.get('ngal')][kwargs.get('sl', '')][label]["diags"])

        self.ax_lower0[j].plot(x_data, means_ratio, color=self.colors[i])
        self.ax_lower1[j].plot(x_data, diags_ratio, color=self.colors[i])

    def save(self, savedir, filename=None):
        if filename is None:
            filename = f"{self.config.plot_type.name.lower()}_main.png"
        self.fig.savefig(os.path.join(savedir, filename), bbox_inches="tight")