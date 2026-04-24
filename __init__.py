"""magno_tracker — analysis library for plume-tracking experiments.

Quick start::

    from magno_tracker import TrackingExperiment, TrackingTrial

    exp = TrackingExperiment(trial_list)
    exp.detect_saccades()
    fig = exp.plot('test_ind', 'body_angle', col_var='condition')
"""

__version__ = "2.0.0"

from .tracking import (
    # Core data classes
    TrackingTrial,
    TrackingExperiment,
    Saccade,
    SummaryDisplay,
    # Phase 3 standalone plot functions
    plot_line,
    plot_hist2d,
    plot_trajectory2d,
    plot_histogram,
    plot_scatter,
    plot_pdf,
    # Phase 1 helpers
    resolve_colors,
    bootstrap_ci,
    get_grid_vals,
    omit_wrapping,
)

__all__ = [
    "__version__",
    # Data classes
    "TrackingTrial",
    "TrackingExperiment",
    "Saccade",
    "SummaryDisplay",
    # Plot functions
    "plot_line",
    "plot_hist2d",
    "plot_trajectory2d",
    "plot_histogram",
    "plot_scatter",
    "plot_pdf",
    # Helpers
    "resolve_colors",
    "bootstrap_ci",
    "get_grid_vals",
    "omit_wrapping",
]
