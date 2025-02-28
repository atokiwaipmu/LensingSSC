from plotting import PlotType, PlotConfig, StatisticsPlotter

# ELLプロットの例
config_ell = PlotConfig(PlotType.ELL)
plotter_ell = StatisticsPlotter(
    labels=labels_ell,
    zs_list=zs_list,
    colors=colors,
    config=config_ell
)

plotter_ell.plot(
    x_data=ell,
    stats_tiled=stats_tiled,
    stats_bigbox=stats_bigbox,
    theory=theory,
    ngal=ngal
)
plotter_ell.save(savedir)

# NUプロットの例
config_nu = PlotConfig(PlotType.NU)
plotter_nu = StatisticsPlotter(
    labels=labels_nu,
    zs_list=zs_list,
    colors=colors,
    config=config_nu
)

plotter_nu.plot(
    x_data=nu,
    stats_tiled=stats_tiled,
    stats_bigbox=stats_bigbox,
    theory=theory,
    ngal=ngal,
    sl=sl
)
plotter_nu.save(savedir)