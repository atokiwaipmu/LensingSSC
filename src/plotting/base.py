from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from .config import PlotConfig

class BasePlotter(ABC):
    def __init__(self, labels, config: PlotConfig):
        self.labels = labels
        self.config = config
        self.fig = None
        self.axes = {}

    def setup_figure(self):
        height = PlotConstants.FIGURE_HEIGHT_MULTIPLIER * len(self.labels)
        self.fig = plt.figure(figsize=(PlotConstants.FIGURE_WIDTH, height))
        
        title = f"Mean $\mu$ and Variance $\sigma^2$, and their ratio $R_{{\mu}}, R_{{\sigma^2}}$ of ${self.config.title_symbol}$-binned Statistics between TILED and BIGBOX"
        self.fig.suptitle(title, fontsize=PlotConstants.TITLE_FONTSIZE, y=self.config.title_y)

    def setup_grid(self):
        gs_master = GridSpec(
            nrows=2 * len(self.labels),
            ncols=1,
            height_ratios=[3, 1] * len(self.labels),
            hspace=self.config.hspace
        )
        
        self.gs_upper = [GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_master[i*2]) 
                        for i in range(len(self.labels))]
        self.gs_lower = [GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_master[i*2+1]) 
                        for i in range(len(self.labels))]

    @abstractmethod
    def plot(self, data, **kwargs):
        pass