# lensing_ssc/plotting/__init__.py

"""Plotting modules for lensing-ssc."""

# Import common utilities
from lensing_ssc.plotting.plot_utils import (
    set_plotting_defaults,
    process_statistics_set,
)

# Import specialized plotting modules
from lensing_ssc.plotting.statistics_plots import plot_binned_statistics_comparison
from lensing_ssc.plotting.correlation_plots import (
    plot_covariance_matrix,
    plot_matrix_ratio,
    plot_correlation_ratio,
)
from lensing_ssc.plotting.rmp_rip_plots import plot_rmp_rip_ratios
from lensing_ssc.plotting.bad_patch_plots import plot_bad_patch_comparison 