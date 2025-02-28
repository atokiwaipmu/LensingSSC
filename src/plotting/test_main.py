import pytest
from unittest.mock import Mock, patch
from src.plotting import PlotType, PlotConfig, StatisticsPlotter

@pytest.fixture
def mock_data():
    return {
        'labels_ell': ['label1', 'label2'],
        'labels_nu': ['nu1', 'nu2'],
        'zs_list': [1.0, 2.0],
        'colors': ['red', 'blue'],
        'ell': [1, 2, 3],
        'nu': [0.1, 0.2, 0.3],
        'stats_tiled': {'data': 'tiled'},
        'stats_bigbox': {'data': 'bigbox'},
        'theory': {'data': 'theory'},
        'ngal': 1000,
        'sl': 0.5,
        'savedir': '/test/path'
    }

@pytest.fixture
def mock_plotter():
    with patch('plotting.StatisticsPlotter') as mock:
        yield mock

class TestPlotting:
    
    def test_ell_plot_creation(self, mock_data, mock_plotter):
        config = PlotConfig(PlotType.ELL)
        
        plotter = StatisticsPlotter(
            labels=mock_data['labels_ell'],
            zs_list=mock_data['zs_list'],
            colors=mock_data['colors'],
            config=config
        )
        
        mock_plotter.assert_called_once_with(
            labels=mock_data['labels_ell'],
            zs_list=mock_data['zs_list'],
            colors=mock_data['colors'],
            config=config
        )

    def test_ell_plot_execution(self, mock_data, mock_plotter):
        plotter_instance = mock_plotter.return_value
        config = PlotConfig(PlotType.ELL)
        
        plotter = StatisticsPlotter(
            labels=mock_data['labels_ell'],
            zs_list=mock_data['zs_list'],
            colors=mock_data['colors'],
            config=config
        )
        
        plotter.plot(
            x_data=mock_data['ell'],
            stats_tiled=mock_data['stats_tiled'],
            stats_bigbox=mock_data['stats_bigbox'],
            theory=mock_data['theory'],
            ngal=mock_data['ngal']
        )
        
        plotter_instance.plot.assert_called_once_with(
            x_data=mock_data['ell'],
            stats_tiled=mock_data['stats_tiled'],
            stats_bigbox=mock_data['stats_bigbox'],
            theory=mock_data['theory'],
            ngal=mock_data['ngal']
        )

    def test_nu_plot_creation(self, mock_data, mock_plotter):
        config = PlotConfig(PlotType.NU)
        
        plotter = StatisticsPlotter(
            labels=mock_data['labels_nu'],
            zs_list=mock_data['zs_list'],
            colors=mock_data['colors'],
            config=config
        )
        
        mock_plotter.assert_called_once_with(
            labels=mock_data['labels_nu'],
            zs_list=mock_data['zs_list'],
            colors=mock_data['colors'],
            config=config
        )

    def test_nu_plot_execution(self, mock_data, mock_plotter):
        plotter_instance = mock_plotter.return_value
        config = PlotConfig(PlotType.NU)
        
        plotter = StatisticsPlotter(
            labels=mock_data['labels_nu'],
            zs_list=mock_data['zs_list'],
            colors=mock_data['colors'],
            config=config
        )
        
        plotter.plot(
            x_data=mock_data['nu'],
            stats_tiled=mock_data['stats_tiled'],
            stats_bigbox=mock_data['stats_bigbox'],
            theory=mock_data['theory'],
            ngal=mock_data['ngal'],
            sl=mock_data['sl']
        )
        
        plotter_instance.plot.assert_called_once_with(
            x_data=mock_data['nu'],
            stats_tiled=mock_data['stats_tiled'],
            stats_bigbox=mock_data['stats_bigbox'],
            theory=mock_data['theory'],
            ngal=mock_data['ngal'],
            sl=mock_data['sl']
        )

    def test_save_functionality(self, mock_data, mock_plotter):
        plotter_instance = mock_plotter.return_value
        config = PlotConfig(PlotType.ELL)
        
        plotter = StatisticsPlotter(
            labels=mock_data['labels_ell'],
            zs_list=mock_data['zs_list'],
            colors=mock_data['colors'],
            config=config
        )
        
        plotter.save(mock_data['savedir'])
        plotter_instance.save.assert_called_once_with(mock_data['savedir'])