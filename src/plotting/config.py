from .constants import PlotType

class PlotConfig:
    def __init__(self, plot_type: PlotType):
        self.plot_type = plot_type
        self._configure()

    def _configure(self):
        configs = {
            PlotType.ELL: {
                'xlabel': r'$\ell$',
                'xticks': [300, 500, 1000, 2000, 3000],
                'xscale': 'log',
                'yscale': 'log',
                'title_y': 0.93,
                'legend_y': 0.07,
                'hspace': 0.1,
                'title_symbol': r'\ell'
            },
            PlotType.NU: {
                'xlabel': r'$\nu$',
                'xticks': [-4, -2, 0, 2, 4],
                'xscale': 'linear',
                'yscale': 'linear',
                'title_y': 0.91,
                'legend_y': 0.08,
                'hspace': 0.2,
                'title_symbol': r'{\nu}'
            }
        }
        
        for key, value in configs[self.plot_type].items():
            setattr(self, key, value)