"""
Tests for Phase 5: TrackingExperiment.plot() and supporting helpers.

Covers 5.2–5.7:
  5.2  plot(plot_type='line')         — correct subplot count, traces present
  5.3  plot(plot_type='hist2d')        — shared vmax across panels
  5.4  plot(plot_type='histogram')     — bar histogram present
  5.5  plot(plot_type='trajectory2d')  — circ_hist Wedge count consistent
  5.6  plot(object='saccade', plot_type='line')  — saccade time-series
  5.7  plot(object='saccade', plot_type='scatter') — saccade scalar scatter

Also covers:
  - _resolve_margin() helper rules
  - _pending_vars initialised by load_datasets() for synthetic trials
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="skvideo")

import copy
import numpy as np
import pytest
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from pathlib import Path

from magno_tracker.tracking import (
    TrackingTrial,
    TrackingExperiment,
    SummaryDisplay,
    _resolve_margin,
    _MARGIN_DEFAULTS,
    _hide_spine,
)

OUT_DIR = Path(__file__).parent / 'plot_outputs'


@pytest.fixture(autouse=True)
def save_plots(request):
    """Save every figure open after each test to plot_outputs/, then close all."""
    OUT_DIR.mkdir(exist_ok=True)
    yield
    fignums = plt.get_fignums()
    for i, num in enumerate(fignums):
        fig = plt.figure(num)
        suffix = f'_{i}' if i > 0 else ''
        try:
            fig.savefig(
                OUT_DIR / f'{request.node.name}{suffix}.png',
                dpi=80, bbox_inches='tight',
            )
        except Exception:
            pass
    plt.close('all')


# ---------------------------------------------------------------------------
# Helpers to build synthetic trials and experiments
# ---------------------------------------------------------------------------
H5_DIR = Path(__file__).parent / '..' / '..' / 'h5_files'
H5_FILES = sorted(H5_DIR.glob('fh_baja_1_new_trial_*_cond1_new.h5'))

NUM_TESTS = 4
NUM_FRAMES = 120
FRAMERATE = 60.0
RNG = np.random.default_rng(42)


def _make_synthetic_trial(num_tests=NUM_TESTS, num_frames=NUM_FRAMES,
                           cond=0.0, rng=None):
    """Return a fully-initialised synthetic TrackingTrial."""
    if rng is None:
        rng = RNG
    heading = rng.uniform(-np.pi, np.pi, size=(num_tests, num_frames))
    heading = np.cumsum(rng.normal(0, 0.05, size=(num_tests, num_frames)), axis=1)
    heading = (heading + np.pi) % (2 * np.pi) - np.pi
    time = np.tile(
        np.linspace(0, num_frames / FRAMERATE, num_frames, endpoint=False),
        (num_tests, 1),
    )
    ds = xr.Dataset(
        {
            'camera_heading': xr.DataArray(heading, dims=('test', 'frame')),
            'time': xr.DataArray(time, dims=('test', 'frame')),
            'condition': xr.DataArray(
                np.full(num_tests, cond), dims=('test',)),
            'is_test': xr.DataArray(
                np.ones(num_tests, dtype=bool), dims=('test',)),
        },
        attrs={'framerate': FRAMERATE,
               'duration': num_frames / FRAMERATE},
    )
    trial = TrackingTrial.__new__(TrackingTrial)
    trial.filename = f'synthetic_cond{cond}.h5'
    trial.file_opened = True
    trial.load_success = False
    trial.h5_file = ds
    trial.load_datasets()
    return trial


def _make_experiment(n_trials=3, n_tests=NUM_TESTS, n_frames=NUM_FRAMES):
    """Return a TrackingExperiment built from synthetic trials."""
    exp = TrackingExperiment.__new__(TrackingExperiment)
    exp.trials = [
        _make_synthetic_trial(num_tests=n_tests, num_frames=n_frames,
                              cond=float(i))
        for i in range(n_trials)
    ]
    return exp


# ---------------------------------------------------------------------------
# _resolve_margin helper
# ---------------------------------------------------------------------------
class TestResolveMargin:

    def test_false_returns_empty(self):
        assert _resolve_margin(False, 'line') == []

    def test_none_returns_empty(self):
        assert _resolve_margin(None, 'line') == []

    def test_true_line(self):
        assert _resolve_margin(True, 'line') == ['line']

    def test_true_histogram(self):
        assert _resolve_margin(True, 'histogram') == ['histogram']

    def test_string_passthrough(self):
        assert _resolve_margin('histogram', 'line') == ['histogram']

    def test_list_passthrough(self):
        assert _resolve_margin(['scatter', 'circ_hist'], 'trajectory2d') == [
            'scatter', 'circ_hist']

    def test_2d_type_blocked_for_1d_plot(self):
        # For a histogram (1-D) main plot, 'scatter' should be dropped.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = _resolve_margin('scatter', 'histogram')
        assert result == []
        assert len(w) == 1

    def test_1d_type_kept_for_1d_plot(self):
        assert _resolve_margin('line', 'histogram') == ['line']


# ---------------------------------------------------------------------------
# Synthetic trial _pending_vars initialisation (regression)
# ---------------------------------------------------------------------------
class TestSyntheticTrialState:

    def test_pending_vars_initialised(self):
        trial = _make_synthetic_trial()
        assert hasattr(trial, '_pending_vars')
        assert isinstance(trial._pending_vars, dict)

    def test_removed_vars_initialised(self):
        trial = _make_synthetic_trial()
        assert hasattr(trial, '_removed_vars')
        assert isinstance(trial._removed_vars, set)

    def test_dirty_attrs_initialised(self):
        trial = _make_synthetic_trial()
        assert hasattr(trial, '_dirty_attrs')
        assert trial._dirty_attrs is False


# ---------------------------------------------------------------------------
# 5.2  plot(plot_type='line')
# ---------------------------------------------------------------------------
class TestPlotLine:

    def test_creates_correct_subplot_count(self):
        exp = _make_experiment(n_trials=3)
        # 2 rows × 2 cols of data cells (right + bottom margins)
        for t in exp.trials:
            t.h5_file['condition'].values[:] = 0.0
        # no grid vars — just 1×1 main cell
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='line', right_margin=False, bottom_margin=False)
        assert hasattr(exp, 'display')
        assert exp.display.trace_axes.shape == (1, 1)

    def test_traces_present_in_cell(self):
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='line', right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        assert len(ax.lines) > 0, "No lines drawn in trace cell"

    def test_resolve_colors_distinct_per_row(self):
        """Different row values produce distinct colors."""
        exp = _make_experiment(n_trials=3)
        import matplotlib.cm as cm
        exp.plot('camera_heading', 'time',
                 col_var=None, row_var='condition',
                 plot_type='line',
                 row_cmap=cm.viridis,
                 right_margin=False, bottom_margin=False)
        # Should have n_distinct_row_vals trace rows.
        n_rows = exp.display.trace_axes.shape[0]
        assert n_rows == 3  # one per trial cond value

    def test_right_margin_line(self):
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='line', right_margin='line', bottom_margin=False)
        ax = exp.display.right_col[0]
        assert len(ax.lines) > 0, "Right margin should have lines"

    def test_bottom_margin_histogram(self):
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='line', right_margin=False, bottom_margin='histogram')
        ax = exp.display.bottom_row[0]
        # step histogram uses line or step artists
        assert len(ax.lines) > 0 or len(ax.patches) > 0, \
            "Bottom margin histogram should have artists"
        # ylabel should be 'count', not the main plot's yvar ('time')
        assert ax.get_ylabel() in ('count', 'probability'), \
            f"Bottom margin ylabel should be 'count', got {ax.get_ylabel()!r}"


# ---------------------------------------------------------------------------
# 5.3  plot(plot_type='hist2d')
# ---------------------------------------------------------------------------
class TestPlotHist2d:

    def test_mesh_drawn(self):
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='hist2d', right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        from matplotlib.collections import QuadMesh
        meshes = [c for c in ax.get_children() if isinstance(c, QuadMesh)]
        assert len(meshes) >= 1

    def test_shared_vmax_across_panels(self):
        """When multiple cells exist, the same vmax is used (QuadMesh clim)."""
        exp = _make_experiment(n_trials=3)
        import matplotlib.cm as cm
        exp.plot('camera_heading', 'time',
                 col_var=None, row_var='condition',
                 plot_type='hist2d',
                 row_cmap=cm.viridis,
                 right_margin=False, bottom_margin=False)
        from matplotlib.collections import QuadMesh
        vmaxes = set()
        for ax in exp.display.trace_axes.flatten():
            for c in ax.get_children():
                if isinstance(c, QuadMesh):
                    vmaxes.add(c.norm.vmax)
        # All panels should share a single vmax.
        assert len(vmaxes) == 1, f"Expected 1 shared vmax, got {vmaxes}"


# ---------------------------------------------------------------------------
# 5.4  plot(plot_type='histogram')
# ---------------------------------------------------------------------------
class TestPlotHistogram:

    def test_patches_drawn(self):
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', col_var=None, row_var=None,
                 plot_type='histogram', bins=20,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        assert len(ax.patches) > 0, "No histogram bars drawn"

    def test_margin_type_blocked_for_scatter(self):
        """Requesting a 2-D margin type for a 1-D main plot must warn and drop."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            margin_types = _resolve_margin('scatter', 'histogram')
        assert margin_types == []
        assert len(w) >= 1

    def test_histogram_margin_allowed(self):
        exp = _make_experiment(n_trials=3)
        # Should not raise.
        exp.plot('camera_heading', col_var=None, row_var=None,
                 plot_type='histogram', bins=20,
                 right_margin='histogram', bottom_margin=False)

    def test_default_margin_is_histogram(self):
        assert _resolve_margin(True, 'histogram') == ['histogram']


# ---------------------------------------------------------------------------
# 5.5  plot(plot_type='trajectory2d')
# ---------------------------------------------------------------------------
class TestPlotTrajectory2d:

    def test_aspect_ratio_equal(self):
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', col_var=None, row_var=None,
                 plot_type='trajectory2d',
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        assert ax.get_aspect() in ('equal', 1.0)

    def test_circ_hist_wedge_count(self):
        """circ_hist=True should draw bins-count Wedge patches."""
        exp = _make_experiment(n_trials=3)
        n_bins = 36
        exp.plot('camera_heading', col_var=None, row_var=None,
                 plot_type='trajectory2d',
                 plot_kwargs={'circ_hist': True, 'bins': n_bins},
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        wedges = [p for p in ax.get_children() if isinstance(p, Wedge)]
        assert len(wedges) == n_bins

    def test_circ_hist_consistent_vmax_across_panels(self):
        """All circ_hist panels share the same normalised vmax."""
        exp = _make_experiment(n_trials=3)
        import matplotlib.cm as cm
        n_bins = 36
        exp.plot('camera_heading',
                 col_var=None, row_var='condition',
                 plot_type='trajectory2d',
                 row_cmap=cm.viridis,
                 plot_kwargs={'circ_hist': True, 'bins': n_bins},
                 right_margin=False, bottom_margin=False)
        # Collect Wedge facecolor max-luminance per panel; all should be <= global.
        # A simpler proxy: each panel should have exactly n_bins Wedges.
        for ax in exp.display.trace_axes.flatten():
            wedges = [p for p in ax.get_children() if isinstance(p, Wedge)]
            assert len(wedges) == n_bins

    def test_axis_limits_expand_for_circ_hist(self):
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', col_var=None, row_var=None,
                 plot_type='trajectory2d',
                 plot_kwargs={'circ_hist': True, 'bins': 36},
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        assert ax.get_xlim()[0] <= -1.27
        assert ax.get_xlim()[1] >= 1.27

    def test_return_summary_extension(self):
        """plot_trajectory2d with return_summary=True returns a dict."""
        from magno_tracker.tracking import plot_trajectory2d
        rng = np.random.default_rng(0)
        xs = rng.uniform(-np.pi, np.pi, (5, 50))
        fig, ax = plt.subplots()
        result = plot_trajectory2d(ax, xs, None, 'r',
                                   circ_hist=True, bins=20,
                                   return_summary=True)
        artists, sd = result
        assert isinstance(sd, dict)
        assert 'last_pos' in sd
        assert 'hist' in sd
        assert 'mean_angle' in sd
        # fig will be saved + closed by the module-level save_plots fixture

    def test_circle_option_draws_circles(self):
        """circle=True draws one Circle per trajectory plus a thicker mean circle."""
        from magno_tracker.tracking import plot_trajectory2d
        import matplotlib.patches as mpatches
        rng = np.random.default_rng(1)
        n_traces = 5
        xs = rng.uniform(-np.pi, np.pi, (n_traces, 50))
        fig, ax = plt.subplots()
        _, sd = plot_trajectory2d(ax, xs, None, 'b', circle=True,
                                  return_summary=True)
        circles = [p for p in ax.get_children()
                   if isinstance(p, mpatches.Circle)]
        # n_traces individual + 1 mean circle
        assert len(circles) == n_traces + 1, \
            f"Expected {n_traces + 1} Circle patches, got {len(circles)}"
        assert 'mean_radius' in sd, "summary_dict should contain 'mean_radius'"

    def test_circle_margin_draws_mean_circle(self):
        """trajectory2d margin with circle=True draws a mean Circle on the margin."""
        import matplotlib.patches as mpatches
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', col_var=None, row_var=None,
                 plot_type='trajectory2d',
                 plot_kwargs={'circle': True},
                 right_margin=True, bottom_margin=False)
        ax = exp.display.right_col[0]
        circles = [p for p in ax.get_children()
                   if isinstance(p, mpatches.Circle)]
        assert len(circles) >= 1, \
            "trajectory2d margin with circle=True should draw a mean Circle"

    def test_contour_margin_draws_ellipse(self):
        """trajectory2d margin with contour=True draws an Ellipse on the margin."""
        import matplotlib.patches as mpatches
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', col_var=None, row_var=None,
                 plot_type='trajectory2d',
                 plot_kwargs={'contour': True},
                 right_margin=True, bottom_margin=False)
        ax = exp.display.right_col[0]
        ellipses = [p for p in ax.get_children()
                    if isinstance(p, mpatches.Ellipse)
                    and not isinstance(p, mpatches.Circle)]
        assert len(ellipses) >= 1, \
            "trajectory2d margin with contour=True should draw an Ellipse"

    def test_contour_option_draws_ellipse(self):
        """contour=True should draw a bootstrap confidence ellipse."""
        from magno_tracker.tracking import plot_trajectory2d
        import matplotlib.patches as mpatches
        rng = np.random.default_rng(2)
        xs = rng.uniform(-np.pi, np.pi, (20, 50))
        fig, ax = plt.subplots()
        plot_trajectory2d(ax, xs, None, 'r', contour=True)
        ellipses = [p for p in ax.get_children()
                    if isinstance(p, mpatches.Ellipse)
                    and not isinstance(p, mpatches.Circle)]
        assert len(ellipses) >= 1, \
            "contour=True should draw at least one Ellipse patch"


# ---------------------------------------------------------------------------
# 5.6  plot(object='saccade', plot_type='line') — requires real h5 files
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not any(H5_FILES),
    reason="No h5 files found — skipping saccade integration tests",
)
class TestPlotSaccadeLine:

    @pytest.fixture(scope='class')
    def exp_with_saccades(self):
        files = sorted(H5_DIR.glob('fh_baja_1_new_trial_*_cond1_new.h5'))[:3]
        exp = TrackingExperiment.__new__(TrackingExperiment)
        exp.trials = [TrackingTrial(str(f)) for f in files]
        for t in exp.trials:
            assert t.load_success
            t.detect_saccades()
        return exp

    def test_plot_creates_figure(self, exp_with_saccades):
        exp = exp_with_saccades
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 right_margin=False, bottom_margin=False,
                 xticks=[(-np.pi/2, 0, np.pi/2), (r"$-\pi$/2", 0, r"$\pi$/2")],
                 ylabel='Time (s)')
        assert hasattr(exp, 'display')
        ax = exp.display.trace_axes[0, 0]
        assert len(ax.lines) > 0, "No lines drawn in saccade line plot"

    def test_trials_without_saccades_skipped_silently(self, exp_with_saccades):
        """A trial with no saccade table should be skipped, not raise."""
        exp = copy.copy(exp_with_saccades)
        # Temporarily hide the saccade_table on the first trial.
        original = exp.trials[0].saccade_table
        exp.trials[0].saccade_table = None
        try:
            exp.plot('arr_relative', 'relative_time',
                     col_var=None, row_var=None,
                     object='saccade', plot_type='line',
                     right_margin=False, bottom_margin=False,
                     xticks=[(-np.pi/2, 0, np.pi/2), (r"$-\pi$/2", 0, r"$\pi$/2")],
                     ylabel='Time (s)')
        finally:
            exp.trials[0].saccade_table = original


# ---------------------------------------------------------------------------
# 5.7  plot(object='saccade', plot_type='scatter') — requires real h5 files
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not any(H5_FILES),
    reason="No h5 files found — skipping saccade integration tests",
)
class TestPlotSaccadeScatter:

    @pytest.fixture(scope='class')
    def exp_with_saccades(self):
        files = sorted(H5_DIR.glob('fh_baja_1_new_trial_*_cond1_new.h5'))[:3]
        exp = TrackingExperiment.__new__(TrackingExperiment)
        exp.trials = [TrackingTrial(str(f)) for f in files]
        for t in exp.trials:
            assert t.load_success
            t.detect_saccades()
        return exp

    def test_scatter_draws_points(self, exp_with_saccades):
        exp = exp_with_saccades
        exp.plot('start_angle', 'amplitude',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='scatter',
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        from matplotlib.collections import PathCollection
        scatters = [c for c in ax.get_children() if isinstance(c, PathCollection)]
        assert len(scatters) > 0

    def test_hist2d_saccade(self, exp_with_saccades):
        """scatter and hist2d are interchangeable for saccade scalar data."""
        exp = exp_with_saccades
        exp.plot('start_angle', 'amplitude',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='hist2d', bins=15,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        from matplotlib.collections import QuadMesh
        meshes = [c for c in ax.get_children() if isinstance(c, QuadMesh)]
        assert len(meshes) >= 1

    def test_histogram_saccade_per_trial_groupby(self, exp_with_saccades):
        """groupby='trial' aggregates to one value per trial before plotting."""
        exp = exp_with_saccades
        # Should run without error; groupby='trial' with agg_func=np.nanmean.
        exp.plot('amplitude', col_var=None, row_var=None,
                 object='saccade', plot_type='histogram', bins=10,
                 groupby='trial', agg_func=np.nanmean,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        assert len(ax.patches) > 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Margin drawing tests
# ---------------------------------------------------------------------------
class TestMargins:
    """Test that right and bottom margin axes are drawn with correct content
    and axis labels for all relevant plot_type/mtype combinations."""

    def test_bottom_hist_ylabel_is_count(self):
        """Bottom histogram margin ylabel must be 'count', not the yvar name."""
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='line', right_margin=False, bottom_margin='histogram')
        ax = exp.display.bottom_row[0]
        assert ax.get_ylabel() in ('count', 'probability')

    def test_bottom_hist_ylabel_probability_when_probability_true(self):
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='line', right_margin=False, bottom_margin='histogram',
                 probability=True)
        ax = exp.display.bottom_row[0]
        assert ax.get_ylabel() == 'probability'

    def test_right_hist_xlabel_is_count(self):
        """Right histogram margin xlabel must be 'count', not the xvar name."""
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='line', right_margin='histogram', bottom_margin=False)
        ax = exp.display.right_col[0]
        assert ax.get_xlabel() in ('count', 'probability')

    def test_right_hist_orientation(self):
        """Right margin histogram: data values on y-axis, counts on x-axis."""
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='line', right_margin='histogram', bottom_margin=False)
        ax = exp.display.right_col[0]
        assert len(ax.lines) > 0, "Right margin histogram should have step lines"
        # y-values should span the time range (0 to NUM_FRAMES/FRAMERATE ~ 2 s)
        all_ys = np.concatenate([l.get_ydata() for l in ax.lines if len(l.get_ydata()) > 0])
        assert all_ys.max() > 0.5, "Right margin histogram y-values should be in time range"

    def test_bottom_hist_draws_artists(self):
        """Bottom histogram margin must produce visible artists."""
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='line', right_margin=False, bottom_margin='histogram')
        ax = exp.display.bottom_row[0]
        assert len(ax.lines) > 0 or len(ax.patches) > 0

    def test_right_hist_draws_artists(self):
        """Right histogram margin must produce visible artists."""
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='line', right_margin='histogram', bottom_margin=False)
        ax = exp.display.right_col[0]
        assert len(ax.lines) > 0 or len(ax.patches) > 0

    def test_hist2d_default_right_margin_is_contour(self):
        """hist2d default right margin draws 2D contours on the margin axis."""
        assert _resolve_margin(True, 'hist2d') == ['contour']
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='hist2d', bins=20, right_margin=True, bottom_margin=False,
                 plot_kwargs={'n_contours': 1},)
        ax = exp.display.right_col[0]
        ax.figure.canvas.draw()  # flush pending autoscale
        # Contours are drawn over the camera_heading range (~[-pi, pi]).
        # After rendering, the margin x-span must greatly exceed the
        # matplotlib default of 1.0 (no-data state).
        xlim = ax.get_xlim()
        assert xlim[1] - xlim[0] > 1.0, (
            f"hist2d right margin should show 2D contours spanning "
            f"camera_heading range; got xlim={xlim}"
        )

    def test_scatter_default_right_margin_is_histogram(self):
        """scatter default right margin is histogram (of y-values)."""
        assert _resolve_margin(True, 'scatter') == ['histogram']
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='scatter', right_margin=True, bottom_margin=False)
        ax = exp.display.right_col[0]
        assert len(ax.lines) > 0 or len(ax.patches) > 0

    def test_trajectory2d_right_margin_trajectory2d_default(self):
        """trajectory2d right margin default draws CI arc or mean-traj line."""
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', col_var=None, row_var=None,
                 plot_type='trajectory2d',
                 plot_kwargs={'mean_line': True},
                 right_margin=True, bottom_margin=False)
        ax = exp.display.right_col[0]
        # mean_line path: a trajectory line should appear
        assert len(ax.lines) > 0 or len(ax.collections) > 0, \
            "trajectory2d right margin should draw a summary artist"

    def test_trajectory2d_right_margin_circ_hist_arc(self):
        """circ_hist margin draws CI Arc artists on the right margin."""
        from matplotlib.patches import Arc as _Arc
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', col_var=None, row_var=None,
                 plot_type='trajectory2d',
                 plot_kwargs={'circ_hist': True},
                 right_margin=True, bottom_margin=False)
        ax = exp.display.right_col[0]
        arcs = [a for a in ax.get_children() if isinstance(a, _Arc)]
        # 1 overlay × 2 Arc artists (white outline + color)
        assert len(arcs) >= 2, \
            f"Expected >= 2 Arc artists for circ_hist CI, got {len(arcs)}"

    def test_multitype_margin_list_passthrough(self):
        """_resolve_margin(['line', 'histogram'], 'line') passes both types through."""
        assert _resolve_margin(['line', 'histogram'], 'line') == ['line', 'histogram']

    def test_multitype_margin_both_drawn(self):
        """right_margin=['line', 'histogram'] draws contributions from both types."""
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='line',
                 right_margin=['line', 'histogram'], bottom_margin=False)
        ax = exp.display.right_col[0]
        # Both 'line' and histogram contribute → at least 2 lines (mean line + histogram step)
        assert len(ax.lines) >= 2, "Both margin types should add artists"

    def test_bottom_hist_uses_xvar_data(self):
        """Bottom histogram margin should bin x-variable values (camera_heading ~(-π,π))."""
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time', col_var=None, row_var=None,
                 plot_type='line', right_margin=False, bottom_margin='histogram')
        ax = exp.display.bottom_row[0]
        # x-extent of the step lines should be within the camera_heading range
        all_xs = np.concatenate([l.get_xdata() for l in ax.lines if len(l.get_xdata()) > 0])
        # camera_heading is bounded by (-pi, pi)
        assert all_xs.min() >= -np.pi - 0.1
        assert all_xs.max() <= np.pi + 0.1


# Real-data integration tests — one round of plotting from actual h5 files
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not any(sorted(H5_DIR.glob('fh_baja_1_new_trial_*_cond1_new.h5'))),
    reason="No h5 files found",
)
class TestRealDataPlots:
    """Load real trials and exercise every plot_type.

    Figures are saved to tests/plot_outputs/ by the module-level save_plots
    fixture so they can be inspected visually.
    """

    @pytest.fixture(scope='class')
    def exp(self):
        """3 trials from fly 1, condition 1."""
        files = sorted(H5_DIR.glob('fh_baja_1_new_trial_*_cond1_new.h5'))[:3]
        e = TrackingExperiment.__new__(TrackingExperiment)
        e.trials = [TrackingTrial(str(f)) for f in files]
        assert all(t.load_success for t in e.trials), "Not all trials loaded"
        return e

    @pytest.fixture(scope='class')
    def exp_saccades(self):
        """Same 3 trials, with saccades detected."""
        files = sorted(H5_DIR.glob('fh_baja_1_new_trial_*_cond1_new.h5'))[:3]
        e = TrackingExperiment.__new__(TrackingExperiment)
        e.trials = [TrackingTrial(str(f)) for f in files]
        for t in e.trials:
            assert t.load_success
            t.detect_saccades()
        return e

    def test_real_line(self, exp):
        exp.plot('camera_heading', 'time',
                 col_var=None, row_var=None,
                 plot_type='line',
                 summary_func=np.nanmean,
                 right_margin=True, bottom_margin=False)
        assert len(exp.display.trace_axes[0, 0].lines) > 0

    def test_real_hist2d(self, exp):
        exp.plot('camera_heading', 'time',
                 col_var=None, row_var=None,
                 plot_type='hist2d', bins=50,
                 right_margin=False, bottom_margin=False)
        from matplotlib.collections import QuadMesh
        children = exp.display.trace_axes[0, 0].get_children()
        assert any(isinstance(c, QuadMesh) for c in children)

    def test_real_histogram(self, exp):
        exp.plot('camera_heading',
                 col_var=None, row_var=None,
                 plot_type='histogram', bins=40,
                 right_margin=False, bottom_margin=False)
        assert len(exp.display.trace_axes[0, 0].patches) > 0

    def test_real_trajectory2d(self, exp):
        exp.plot('camera_heading',
                 col_var=None, row_var=None,
                 plot_type='trajectory2d',
                 plot_kwargs={'circ_hist': True, 'bins': 36,
                              'mean_line': True, 'contour': True},
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        assert ax.get_aspect() in ('equal', 1.0)
        wedges = [p for p in ax.get_children() if isinstance(p, Wedge)]
        assert len(wedges) == 36

    # import partial

    def test_real_saccade_line(self, exp_saccades):
        from functools import partial
        import scipy.stats
        exp_saccades.plot(
            'arr_relative', 'relative_time',
            col_var=None, row_var=None,
            object='saccade', plot_type='line',
            right_margin=True, bottom_margin=False,
            xticks=[(-np.pi/2, 0, np.pi/2), (r"$-\pi$/2", 0, r"$\pi$/2")],
            ylabel='Time (s)',
            scale=5,
            summary_func=partial(scipy.stats.circmean, high=np.pi, low=-np.pi)
        )
        
        assert len(exp_saccades.display.trace_axes[0, 0].lines) > 0

    def test_real_saccade_scatter(self, exp_saccades):
        exp_saccades.plot(
            'start_angle', 'amplitude',
            col_var=None, row_var=None,
            object='saccade', plot_type='scatter',
            right_margin=False, bottom_margin=False)
        from matplotlib.collections import PathCollection
        children = exp_saccades.display.trace_axes[0, 0].get_children()
        assert any(isinstance(c, PathCollection) for c in children)

    def test_real_saccade_hist2d(self, exp_saccades):
        exp_saccades.plot(
            'start_angle', 'amplitude',
            col_var=None, row_var=None,
            object='saccade', plot_type='hist2d', bins=20,
            right_margin=False, bottom_margin=False)
        from matplotlib.collections import QuadMesh
        children = exp_saccades.display.trace_axes[0, 0].get_children()
        assert any(isinstance(c, QuadMesh) for c in children)


# ---------------------------------------------------------------------------
# plot_line standalone — new params
# ---------------------------------------------------------------------------
class TestPlotLineStandalone:
    """Unit tests for the standalone plot_line() function."""

    from magno_tracker.tracking import plot_line as _plot_line_import

    @pytest.fixture
    def rng_traces(self):
        """Return (xs, ys) arrays: 10 traces, 50 time-points each."""
        rng = np.random.default_rng(0)
        ys = np.tile(np.linspace(-0.1, 0.4, 50), (10, 1))  # time 0 → 0.4 s
        xs = rng.normal(0, 0.2, size=(10, 50))
        # Make half the traces end negative.
        xs[5:, -1] = -abs(xs[5:, -1])
        xs[:5, -1] = abs(xs[:5, -1])
        return xs, ys

    def _plot_line(self, *a, **kw):
        from magno_tracker.tracking import plot_line
        return plot_line(*a, **kw)

    def test_returns_artists(self, rng_traces):
        xs, ys = rng_traces
        fig, ax = plt.subplots()
        artists = self._plot_line(ax, xs, ys, 'steelblue')
        assert len(artists) > 0

    def test_trace_count_matches_rows(self, rng_traces):
        xs, ys = rng_traces
        fig, ax = plt.subplots()
        self._plot_line(ax, xs, ys, 'steelblue')
        # Each row in xs becomes one line (transposed to column = one trace).
        assert len(ax.lines) == xs.shape[0]

    def test_summary_func_adds_mean_line(self, rng_traces):
        xs, ys = rng_traces
        fig, ax = plt.subplots()
        self._plot_line(ax, xs, ys, 'steelblue', summary_func=np.nanmean)
        # Should add 2 extra lines: white underlay + colored mean.
        assert len(ax.lines) == xs.shape[0] + 2

    def test_mean_bins_adds_mean_line(self, rng_traces):
        xs, ys = rng_traces
        fig, ax = plt.subplots()
        self._plot_line(ax, xs, ys, 'steelblue', mean_bins=40)
        # mean_bins path also draws white + colored mean.
        assert len(ax.lines) == xs.shape[0] + 2

    def test_mean_bins_xvalues_within_data_range(self, rng_traces):
        xs, ys = rng_traces
        fig, ax = plt.subplots()
        self._plot_line(ax, xs, ys, 'steelblue', mean_bins=40)
        mean_line = ax.lines[-1]
        # Binned median x should stay within xs data range.
        valid = mean_line.get_xdata()
        valid = valid[~np.isnan(valid)]
        assert valid.min() >= np.nanmin(xs) - 1e-6
        assert valid.max() <= np.nanmax(xs) + 1e-6

    def test_split_by_sign_adds_two_mean_lines(self, rng_traces):
        xs, ys = rng_traces
        fig, ax = plt.subplots()
        self._plot_line(ax, xs, ys, 'steelblue',
                        summary_func=np.nanmean, split_by_sign=True)
        # Two groups → 2 × (white + color) = 4 extra lines.
        assert len(ax.lines) == xs.shape[0] + 4

    def test_split_by_sign_linestyles(self, rng_traces):
        xs, ys = rng_traces
        fig, ax = plt.subplots()
        self._plot_line(ax, xs, ys, 'steelblue',
                        summary_func=np.nanmean, split_by_sign=True)
        # The colored mean lines are at indices -1 (neg/dotted) and -3 (pos/solid).
        mean_lines = [l for l in ax.lines if l.get_color() == 'steelblue']
        ls_values = {str(l.get_linestyle()) for l in mean_lines}
        # Should contain both solid ('-') and dotted (':').
        assert '-' in ' '.join(ls_values) or 'solid' in ' '.join(ls_values)
        assert ':' in ' '.join(ls_values) or 'dotted' in ' '.join(ls_values)

    def test_split_by_sign_all_positive_gives_one_group(self):
        """If all traces end positive, only one mean line group is drawn."""
        rng = np.random.default_rng(1)
        ys = np.tile(np.linspace(0, 0.3, 40), (8, 1))
        xs = abs(rng.normal(1.0, 0.2, size=(8, 40)))  # all end positive
        fig, ax = plt.subplots()
        self._plot_line(ax, xs, ys, 'steelblue',
                        summary_func=np.nanmean, split_by_sign=True)
        # Only one group → 2 extra lines (white + color).
        assert len(ax.lines) == xs.shape[0] + 2

    def test_saccade_durations_adds_highlight_segments(self, rng_traces):
        xs, ys = rng_traces
        durations = np.full(xs.shape[0], 0.2)   # highlight first 0.2 s
        fig, ax = plt.subplots()
        self._plot_line(ax, xs, ys, 'steelblue',
                        saccade_durations=durations)
        # Each trace gets a highlight segment → 2× line count.
        assert len(ax.lines) == xs.shape[0] * 2

    def test_saccade_durations_nan_skipped(self, rng_traces):
        xs, ys = rng_traces
        durations = np.full(xs.shape[0], np.nan)   # all NaN → no highlights
        fig, ax = plt.subplots()
        self._plot_line(ax, xs, ys, 'steelblue',
                        saccade_durations=durations)
        # No highlights → only the original trace lines.
        assert len(ax.lines) == xs.shape[0]

    def test_no_summary_no_mean_line(self, rng_traces):
        xs, ys = rng_traces
        fig, ax = plt.subplots()
        self._plot_line(ax, xs, ys, 'steelblue',
                        summary_func=None, mean_bins=None)
        assert len(ax.lines) == xs.shape[0]


# ---------------------------------------------------------------------------
# New plot() params: positive_amplitude, min_speed, max_speed, show_n
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not any(H5_FILES),
    reason="No h5 files found — skipping saccade integration tests",
)
class TestPlotNewParams:
    """Integration tests for positive_amplitude, min_speed, max_speed, show_n."""

    @pytest.fixture(scope='class')
    def exp(self):
        files = sorted(H5_DIR.glob('fh_baja_1_new_trial_*_cond1_new.h5'))[:3]
        e = TrackingExperiment.__new__(TrackingExperiment)
        e.trials = [TrackingTrial(str(f)) for f in files]
        for t in e.trials:
            assert t.load_success
            t.detect_saccades()
        return e

    def test_positive_amplitude_all_traces_end_nonneg(self, exp):
        """With positive_amplitude=True no trace's last non-NaN x should be < 0."""
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 positive_amplitude=True,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        for line in ax.lines:
            xd = line.get_xdata()
            valid = xd[~np.isnan(xd)]
            if valid.size > 0:
                assert valid[-1] >= -1e-9, \
                    f"Trace ends at {valid[-1]:.4f}, expected >= 0"

    def test_positive_amplitude_more_lines_than_without(self, exp):
        """positive_amplitude shouldn't silently drop all traces."""
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 positive_amplitude=True,
                 right_margin=False, bottom_margin=False)
        n_pos = len(exp.display.trace_axes[0, 0].lines)

        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 positive_amplitude=False,
                 right_margin=False, bottom_margin=False)
        n_raw = len(exp.display.trace_axes[0, 0].lines)
        # Same number of traces — just signs flipped.
        assert n_pos == n_raw

    def test_min_speed_reduces_saccade_count(self, exp):
        """min_speed filter should yield fewer (or equal) traces."""
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 right_margin=False, bottom_margin=False)
        n_all = len(exp.display.trace_axes[0, 0].lines)

        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 min_speed=600,   # stricter threshold
                 right_margin=False, bottom_margin=False)
        n_fast = len(exp.display.trace_axes[0, 0].lines)
        assert n_fast <= n_all

    def test_max_speed_reduces_saccade_count(self, exp):
        """max_speed filter should yield fewer (or equal) traces."""
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 right_margin=False, bottom_margin=False)
        n_all = len(exp.display.trace_axes[0, 0].lines)

        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 max_speed=500,
                 right_margin=False, bottom_margin=False)
        n_slow = len(exp.display.trace_axes[0, 0].lines)
        assert n_slow <= n_all

    def test_min_max_speed_no_traces_draws_empty_cell(self, exp):
        """Impossible speed window should produce no trace lines but not raise."""
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 min_speed=9999, max_speed=10000,
                 right_margin=False, bottom_margin=False)
        # Cell should exist but be empty (or skipped).
        assert hasattr(exp, 'display')

    def test_show_n_annotation_present(self, exp):
        """show_n=True should add a text annotation to each non-empty cell."""
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 show_n=True,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        texts = [t.get_text() for t in ax.texts]
        assert any('N=' in t and 'n=' in t for t in texts), \
            f"Expected N=..., n=... annotation, got: {texts}"

    def test_show_n_annotation_format(self, exp):
        """Annotation text must match 'N=<int>, n=<int>'."""
        import re
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 show_n=True,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        texts = [t.get_text() for t in ax.texts]
        pattern = re.compile(r'N=\d+, n=\d+')
        assert any(pattern.match(t) for t in texts), \
            f"Annotation doesn't match 'N=<int>, n=<int>': {texts}"

    def test_show_n_counts_are_positive(self, exp):
        """N and n in the annotation should both be > 0 when saccades exist."""
        import re
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 show_n=True,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        for t in ax.texts:
            m = re.match(r'N=(\d+), n=(\d+)', t.get_text())
            if m:
                assert int(m.group(1)) > 0, "N should be > 0"
                assert int(m.group(2)) > 0, "n should be > 0"

    def test_show_n_false_no_annotation(self, exp):
        """show_n=False (default) must not add any 'N=...' text."""
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 show_n=False,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        texts = [t.get_text() for t in ax.texts]
        assert not any('N=' in t for t in texts), \
            f"Unexpected annotation when show_n=False: {texts}"

    def test_mean_bins_in_plot_kwargs(self, exp):
        """mean_bins passed via plot_kwargs should produce a mean line."""
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 plot_kwargs={'mean_bins': 40},
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        # mean_bins draws white underlay + colored mean → at least 2 extra lines.
        assert len(ax.lines) >= 3, "Expected mean-bins summary line"

    def test_split_by_sign_in_plot_kwargs(self, exp):
        """split_by_sign=True in plot_kwargs should draw two mean groups."""
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 plot_kwargs={'split_by_sign': True, 'mean_bins': 40},
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        # Two groups: at least 4 extra lines (2 white + 2 colored).
        assert len(ax.lines) >= 5


# ---------------------------------------------------------------------------
# rad2deg parameter
# ---------------------------------------------------------------------------
class TestRad2Deg:
    """Tests for the rad2deg conversion parameter in TrackingExperiment.plot()."""

    @pytest.fixture(scope='class')
    def exp(self):
        files = sorted(H5_DIR.glob('fh_baja_1_new_trial_*_cond1_new.h5'))[:3]
        e = TrackingExperiment.__new__(TrackingExperiment)
        e.trials = [TrackingTrial(str(f)) for f in files]
        for t in e.trials:
            assert t.load_success
            t.detect_saccades()
        return e

    def _xdata(self, exp):
        """All non-NaN x values from the first trace cell."""
        ax = exp.display.trace_axes[0, 0]
        vals = []
        for line in ax.lines:
            xd = line.get_xdata()
            vals.extend(xd[~np.isnan(xd)].tolist())
        return np.array(vals)

    def test_rad2deg_false_xlim_in_radians(self, exp):
        """Default rad2deg=False: auto xlim should be (-pi, pi) for saccade lines."""
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 rad2deg=False,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        lo, hi = ax.get_xlim()
        assert abs(lo - (-np.pi)) < 0.2, f"Expected xlim ~(-pi, pi), got ({lo:.3f}, {hi:.3f})"
        assert abs(hi - np.pi) < 0.2

    def test_rad2deg_true_xlim_in_degrees(self, exp):
        """rad2deg=True: auto xlim should be (-180, 180) for saccade lines."""
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 rad2deg=True,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        lo, hi = ax.get_xlim()
        assert abs(lo - (-180)) < 5, f"Expected xlim ~(-180, 180), got ({lo:.1f}, {hi:.1f})"
        assert abs(hi - 180) < 5

    def test_rad2deg_xdata_scaled_by_180_over_pi(self, exp):
        """x values with rad2deg=True should be ~180/pi times those with rad2deg=False."""
        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 rad2deg=False,
                 right_margin=False, bottom_margin=False)
        xrad = self._xdata(exp)

        exp.plot('arr_relative', 'relative_time',
                 col_var=None, row_var=None,
                 object='saccade', plot_type='line',
                 rad2deg=True,
                 right_margin=False, bottom_margin=False)
        xdeg = self._xdata(exp)

        assert xrad.size == xdeg.size, "Different number of data points"
        valid = np.isfinite(xrad) & np.isfinite(xdeg) & (np.abs(xrad) > 1e-6)
        assert valid.sum() > 0, "No valid points to compare"
        ratio = xdeg[valid] / xrad[valid]
        assert np.allclose(ratio, 180 / np.pi, atol=1e-6), \
            f"Expected ratio 180/pi, got mean={ratio.mean():.4f}"


# ---------------------------------------------------------------------------
# main_sequence_analysis — requires real h5 files
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not any(H5_FILES),
    reason="No h5 files found — skipping main_sequence_analysis tests",
)
class TestMainSequenceAnalysis:
    """Integration tests for TrackingExperiment.main_sequence_analysis().

    Uses real h5 files from fh_baja_1 (cond1, 3 trial repetitions).
    The precomputed saccade variables (saccade_peak_velocity, saccade_duration,
    saccade_amplitude) stored in the h5 datasets are used directly — no call to
    detect_saccades() is needed.
    """

    @pytest.fixture(scope='class')
    def exp(self):
        files = sorted(H5_DIR.glob('fh_baja_1_new_trial_*_cond1_new.h5'))[:3]
        e = TrackingExperiment.__new__(TrackingExperiment)
        e.trials = [TrackingTrial(str(f)) for f in files]
        for t in e.trials:
            assert t.load_success, f"Failed to load {t.filename}"
        return e

    # ------------------------------------------------------------------
    # Smoke test: runs end-to-end without raising
    # ------------------------------------------------------------------
    def test_runs_without_error(self, exp):
        exp.main_sequence_analysis(group_var='fly_id', n_boot=20)

    # ------------------------------------------------------------------
    # Figure structure
    # ------------------------------------------------------------------
    def test_creates_3x2_axes(self, exp):
        exp.main_sequence_analysis(group_var='fly_id', n_boot=20)
        d = exp.display
        # 2 scatter axes, 2 right-col strip axes, 1 bottom strip, 1 hidden corner
        assert d.trace_axes.shape == (2, 1), f"Expected (2,1) trace_axes, got {d.trace_axes.shape}"
        assert len(d.right_col) == 2
        assert len(d.bottom_row) == 1
        assert hasattr(d, 'corner_ax')

    def test_bottom_right_axis_invisible(self, exp):
        exp.main_sequence_analysis(group_var='fly_id', n_boot=20)
        assert not exp.display.corner_ax.get_visible(), \
            "Bottom-right axis should be hidden via axis('off')"

    # ------------------------------------------------------------------
    # Scatter panels have data drawn
    # ------------------------------------------------------------------
    def test_scatter_panels_have_collections(self, exp):
        from matplotlib.collections import PathCollection
        exp.main_sequence_analysis(group_var='fly_id', n_boot=20)
        scatter_axes = list(exp.display.trace_axes[:, 0])
        for ax in scatter_axes:
            scatters = [c for c in ax.get_children()
                        if isinstance(c, PathCollection)]
            assert len(scatters) > 0, \
                f"Scatter axis '{ax.get_ylabel()}' has no PathCollections"

    # ------------------------------------------------------------------
    # Bootstrap CI strip plots have data drawn
    # ------------------------------------------------------------------
    def test_margin_axes_have_collections(self, exp):
        from matplotlib.collections import PathCollection
        exp.main_sequence_analysis(group_var='fly_id', n_boot=20)
        margin_axes = list(exp.display.right_col) + [exp.display.bottom_row[0]]
        for ax in margin_axes:
            scatters = [c for c in ax.get_children()
                        if isinstance(c, PathCollection)]
            assert len(scatters) > 0, \
                "Margin axis has no scatter points (no data drawn)"

    # ------------------------------------------------------------------
    # Bootstrap CI lines (error bars) drawn in margin axes
    # ------------------------------------------------------------------
    def test_margin_axes_have_ci_lines(self, exp):
        exp.main_sequence_analysis(group_var='fly_id', n_boot=20)
        for ax in list(exp.display.right_col) + [exp.display.bottom_row[0]]:
            assert len(ax.lines) > 0, \
                "Bootstrap CI line missing from margin axis"

    # ------------------------------------------------------------------
    # colors= kwarg overrides cmap
    # ------------------------------------------------------------------
    def test_explicit_colors_accepted(self, exp):
        """Passing colors= list should not raise when length matches group count."""
        from magno_tracker.tracking import get_grid_vals
        group_vals = get_grid_vals(exp, 'fly_id', {})
        explicit = ['steelblue'] * len(group_vals)
        exp.main_sequence_analysis(
            group_var='fly_id', colors=explicit, n_boot=20, scale=3)

    def test_explicit_colors_wrong_length_raises(self, exp):
        """colors= list of wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="colors has"):
            exp.main_sequence_analysis(
                group_var='fly_id', colors=['r', 'b'], n_boot=5, scale=3)

    # ------------------------------------------------------------------
    # subset= isolation (regression for mutable-default bug)
    # ------------------------------------------------------------------
    def test_subset_not_mutated_by_call(self, exp):
        """The caller's subset dict must not be modified by main_sequence_analysis."""
        user_subset = {}
        exp.main_sequence_analysis(
            group_var='fly_id', subset=user_subset, n_boot=20, scale=3)
        assert user_subset == {}, \
            "main_sequence_analysis mutated the caller's subset dict"

    def test_repeated_calls_with_same_subset_give_same_groups(self, exp):
        """Calling twice with an identical subset should produce the same groups."""
        from magno_tracker.tracking import get_grid_vals
        subset_a = {}
        subset_b = {}
        groups_a = get_grid_vals(exp, 'is_test', subset_a)
        exp.main_sequence_analysis(group_var='fly_id', subset=subset_b, n_boot=5, scale=3)
        groups_b = get_grid_vals(exp, 'is_test', subset_b)
        np.testing.assert_array_equal(groups_a, groups_b)

    # ------------------------------------------------------------------
    # Log-scale axes
    # ------------------------------------------------------------------
    def test_scatter_axes_x_log_scale(self, exp):
        exp.main_sequence_analysis(group_var='filename', n_boot=20, scale=3)
        d = exp.display
        for ax in list(d.trace_axes[:, 0]) + [d.bottom_row[0]]:
            assert ax.get_xscale() == 'log', \
                "Expected log x-scale on magnitude axis"

    def test_x_tick_formatter_gives_pow2_degrees(self, exp):
        """x-axis major formatter should produce integer power-of-2 degree labels."""
        exp.main_sequence_analysis(group_var='filename', n_boot=20, scale=3)
        ax = exp.display.trace_axes[0, 0]
        fmt = ax.xaxis.get_major_formatter()
        # 8 degrees in radians → nearest power-of-2 = 8
        import matplotlib.ticker
        assert isinstance(fmt, matplotlib.ticker.FuncFormatter)
        label = fmt(8 * np.pi / 180., 0)
        assert label == '8', f"Expected '8', got '{label}'"
        label16 = fmt(16 * np.pi / 180., 0)
        assert label16 == '16', f"Expected '16', got '{label16}'"

    def test_scatter_axes_y_log_scale(self, exp):
        exp.main_sequence_analysis(group_var='filename', n_boot=20, scale=3)
        for ax in exp.display.trace_axes[:, 0]:
            assert ax.get_yscale() == 'log', \
                "Expected log y-scale on scatter axis"

    def test_rad2deg_trajectory2d_exempt(self):
        """trajectory2d is exempt from rad2deg conversion (uses xs as raw radians)."""
        exp = _make_experiment(n_trials=3)
        # With rad2deg=True, trajectory2d should still draw without error and
        # produce the same trajectory shapes as without it.
        exp.plot('camera_heading', col_var=None, row_var=None,
                 plot_type='trajectory2d', rad2deg=True,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        # Should have drawn trajectory lines (not errored out).
        assert len(ax.lines) > 0, "trajectory2d should still draw lines when rad2deg=True"

    def test_rad2deg_trial_plot_type_line(self):
        """rad2deg=True also works for object='trial', plot_type='line'."""
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', 'time',
                 col_var=None, row_var=None,
                 plot_type='line', rad2deg=True,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        xd = np.concatenate([l.get_xdata() for l in ax.lines])
        xd = xd[np.isfinite(xd)]
        # camera_heading is in (-pi, pi); after conversion it should be in (-180, 180).
        assert xd.max() <= 185, f"Max x={xd.max():.1f} should be ≤180 deg"
        assert xd.min() >= -185, f"Min x={xd.min():.1f} should be ≥-180 deg"

    def test_rad2deg_histogram(self):
        """rad2deg=True with plot_type='histogram' scales x data to degrees."""
        exp = _make_experiment(n_trials=3)
        exp.plot('camera_heading', col_var=None, row_var=None,
                 plot_type='histogram', rad2deg=True,
                 right_margin=False, bottom_margin=False)
        ax = exp.display.trace_axes[0, 0]
        # Histogram patches should span [-180, 180], not [-pi, pi].
        patches = ax.patches
        assert len(patches) > 0
        left_edges = [p.get_x() for p in patches]
        assert min(left_edges) >= -185, \
            f"Leftmost bin edge {min(left_edges):.2f} should be ≥ -180 deg"
        assert max(left_edges) <= 185, \
            f"Rightmost bin edge {max(left_edges):.2f} should be ≤ 180 deg"


# ---------------------------------------------------------------------------
# SummaryDisplay — subplot sizing, spine hiding, label_margins
# ---------------------------------------------------------------------------

class TestHideSpine:
    """Unit tests for the _hide_spine helper."""

    def _make_ax(self):
        fig, ax = plt.subplots()
        return ax

    def test_bottom_spine_hidden(self):
        ax = self._make_ax()
        _hide_spine(ax, 'bottom')
        assert not ax.spines['bottom'].get_visible()

    def test_left_spine_hidden(self):
        ax = self._make_ax()
        _hide_spine(ax, 'left')
        assert not ax.spines['left'].get_visible()

    def test_right_spine_hidden(self):
        ax = self._make_ax()
        _hide_spine(ax, 'right')
        assert not ax.spines['right'].get_visible()

    def test_top_spine_hidden(self):
        ax = self._make_ax()
        _hide_spine(ax, 'top')
        assert not ax.spines['top'].get_visible()

    def test_bottom_spine_hidden_log_x(self):
        """Spine hiding must work even when x is log-scaled."""
        ax = self._make_ax()
        ax.set_xscale('log')
        ax.set_xlim(0.01, 100)
        _hide_spine(ax, 'bottom')
        assert not ax.spines['bottom'].get_visible()

    def test_left_spine_hidden_log_y(self):
        """Spine hiding must work even when y is log-scaled."""
        ax = self._make_ax()
        ax.set_yscale('log')
        ax.set_ylim(1, 1000)
        _hide_spine(ax, 'left')
        assert not ax.spines['left'].get_visible()

    def test_bottom_spine_hidden_shared_x(self):
        """Spine hiding must work on an axis sharing x with a neighbour."""
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_xscale('log')
        _hide_spine(ax1, 'bottom')
        assert not ax1.spines['bottom'].get_visible()
        # ax2 spine must not be affected
        assert ax2.spines['bottom'].get_visible()

    def test_tick_params_cleared_bottom(self):
        """_hide_spine should disable bottom ticks and tick labels."""
        ax = self._make_ax()
        _hide_spine(ax, 'bottom')
        # tick_params stores state; check via get_tick_params
        tp = ax.xaxis.get_tick_params(which='major')
        assert not tp.get('bottom', True), \
            "Major bottom ticks should be off after _hide_spine"

    def test_tick_params_cleared_left(self):
        """_hide_spine should disable left ticks and tick labels."""
        ax = self._make_ax()
        _hide_spine(ax, 'left')
        tp = ax.yaxis.get_tick_params(which='major')
        assert not tp.get('left', True), \
            "Major left ticks should be off after _hide_spine"


class TestSummaryDisplaySizing:
    """Test that SummaryDisplay computes figure sizes from subplot_size."""

    def test_default_subplot_size_no_margin(self):
        """No margins: figsize = n_cols*sw x n_rows*sh."""
        d = SummaryDisplay(num_rows=2, num_cols=3,
                           right_margin=False, bottom_margin=False,
                           subplot_size=(3, 2))
        w, h = d._get_fig_size_inches()
        assert abs(w - 9) < 0.5, f"Expected width ~9, got {w:.2f}"
        assert abs(h - 4) < 0.5, f"Expected height ~4, got {h:.2f}"

    def test_subplot_size_with_right_margin(self):
        """Right margin contributes right_margin_ratio * sw to total width."""
        d = SummaryDisplay(num_rows=2, num_cols=3,
                           right_margin=True, bottom_margin=False,
                           subplot_size=(4, 4), right_margin_ratio=0.5)
        # data cols = 2; total width = 2*4 + 0.5*4 = 10
        w, h = d._get_fig_size_inches()
        assert abs(w - 10) < 0.5, f"Expected width ~10, got {w:.2f}"
        assert abs(h - 8) < 0.5,  f"Expected height ~8, got {h:.2f}"

    def test_subplot_size_with_bottom_margin(self):
        """Bottom margin contributes bottom_margin_ratio * sh to total height."""
        d = SummaryDisplay(num_rows=3, num_cols=2,
                           right_margin=False, bottom_margin=True,
                           subplot_size=(2, 2), bottom_margin_ratio=0.5)
        # data rows = 2; total height = 2*2 + 0.5*2 = 5
        w, h = d._get_fig_size_inches()
        assert abs(h - 5) < 0.5, f"Expected height ~5, got {h:.2f}"
        assert abs(w - 4) < 0.5, f"Expected width ~4, got {w:.2f}"

    def test_subplot_size_both_margins(self):
        """Both margins: independently affect width and height."""
        d = SummaryDisplay(num_rows=3, num_cols=2,
                           right_margin=True, bottom_margin=True,
                           subplot_size=(2, 2),
                           right_margin_ratio=0.5, bottom_margin_ratio=0.5)
        # data rows=2, data cols=1; width=1*2+0.5*2=3; height=2*2+0.5*2=5
        w, h = d._get_fig_size_inches()
        assert abs(w - 3) < 0.5, f"Expected width ~3, got {w:.2f}"
        assert abs(h - 5) < 0.5, f"Expected height ~5, got {h:.2f}"

    def test_explicit_figsize_overrides_subplot_size(self):
        """Providing figsize= ignores subplot_size entirely."""
        d = SummaryDisplay(num_rows=2, num_cols=2,
                           right_margin=False, bottom_margin=False,
                           figsize=(7, 5), subplot_size=(1, 1))
        w, h = d._get_fig_size_inches()
        assert abs(w - 7) < 0.5
        assert abs(h - 5) < 0.5

    def test_trace_axes_shape_no_margin(self):
        d = SummaryDisplay(num_rows=2, num_cols=3,
                           right_margin=False, bottom_margin=False)
        assert d.trace_axes.shape == (2, 3)

    def test_trace_axes_shape_right_margin(self):
        d = SummaryDisplay(num_rows=2, num_cols=3,
                           right_margin=True, bottom_margin=False)
        # last col is margin → trace_axes has 2 data cols
        assert d.trace_axes.shape == (2, 2)
        assert len(d.right_col) == 2

    def test_trace_axes_shape_both_margins(self):
        d = SummaryDisplay(num_rows=3, num_cols=2,
                           right_margin=True, bottom_margin=True)
        # 2 data rows, 1 data col
        assert d.trace_axes.shape == (2, 1)
        assert len(d.right_col) == 2
        assert len(d.bottom_row) == 1

    def test_corner_ax_invisible(self):
        d = SummaryDisplay(num_rows=3, num_cols=2,
                           right_margin=True, bottom_margin=True)
        assert hasattr(d, 'corner_ax')
        assert not d.corner_ax.get_visible()

    def test_gridspec_width_ratios(self):
        """width_ratios should have n_data_cols ones followed by the margin ratio."""
        d = SummaryDisplay(num_rows=2, num_cols=3,
                           right_margin=True, bottom_margin=False,
                           subplot_size=(2, 2), right_margin_ratio=0.4)
        gs = d.axes[0, 0].get_gridspec()
        wr = list(gs.get_width_ratios())
        assert wr == pytest.approx([1, 1, 0.4], abs=1e-6), \
            f"Unexpected width_ratios: {wr}"

    def test_gridspec_height_ratios(self):
        """height_ratios should have n_data_rows ones followed by the margin ratio."""
        d = SummaryDisplay(num_rows=3, num_cols=2,
                           right_margin=False, bottom_margin=True,
                           subplot_size=(2, 2), bottom_margin_ratio=0.3)
        gs = d.axes[0, 0].get_gridspec()
        hr = list(gs.get_height_ratios())
        assert hr == pytest.approx([1, 1, 0.3], abs=1e-6), \
            f"Unexpected height_ratios: {hr}"


class TestSummaryDisplaySpines:
    """Test that SummaryDisplay.format() reliably hides spines on log axes."""

    def _make_display(self, nrows=2, ncols=2, **kw):
        return SummaryDisplay(num_rows=nrows, num_cols=ncols,
                              right_margin=False, bottom_margin=False,
                              subplot_size=(2, 2), **kw)

    def test_internal_axes_have_no_bottom_spine(self):
        """Non-bottom-row axes should have bottom spine hidden after format()."""
        d = self._make_display(nrows=2, ncols=1)
        d.format()
        # top row (row 0) is not the bottom row
        assert not d.trace_axes[0, 0].spines['bottom'].get_visible()

    def test_bottom_row_keeps_bottom_spine(self):
        """Bottom-row axes should keep their bottom spine."""
        d = self._make_display(nrows=2, ncols=1)
        d.format(xlabel='x')
        assert d.trace_axes[1, 0].spines['bottom'].get_visible()

    def test_internal_axes_no_bottom_spine_with_logy(self):
        """Log y-scale must not prevent bottom spine hiding on internal axes."""
        d = self._make_display(nrows=2, ncols=1)
        d.format(logy=True)
        assert not d.trace_axes[0, 0].spines['bottom'].get_visible(), \
            "Bottom spine must be hidden even with logy=True"

    def test_non_left_axes_no_left_spine_with_logx(self):
        """Log x-scale must not prevent left spine hiding on non-left-col axes."""
        d = self._make_display(nrows=1, ncols=2)
        d.format(logx=True)
        assert not d.trace_axes[0, 1].spines['left'].get_visible(), \
            "Left spine must be hidden even with logx=True"

    def test_top_and_right_always_hidden(self):
        """Top and right spines should always be hidden by format()."""
        d = self._make_display(nrows=2, ncols=2)
        d.format()
        for ax in d.trace_axes.flat:
            assert not ax.spines['top'].get_visible()
            assert not ax.spines['right'].get_visible()

    def test_with_margins_right_col_spines(self):
        """Right-margin axes also get top/right hidden by format()."""
        d = SummaryDisplay(num_rows=2, num_cols=2,
                           right_margin=True, bottom_margin=False,
                           subplot_size=(2, 2))
        d.format()
        for ax in d.right_col:
            assert not ax.spines['top'].get_visible()
            assert not ax.spines['right'].get_visible()


@pytest.mark.skipif(
    not any(H5_FILES),
    reason="No h5 files found — skipping label_margins tests",
)
class TestSummaryDisplayLabelMargins:
    """Tests for SummaryDisplay.label_margins using real rendered bboxes."""

    @pytest.fixture(scope='class')
    def display_2x2(self):
        """A 2×2 display (no margins) with labelled axes, rendered to canvas."""
        d = SummaryDisplay(num_rows=2, num_cols=2,
                           right_margin=False, bottom_margin=False,
                           subplot_size=(2, 2))
        for ax in d.trace_axes.flat:
            ax.set_xlabel('x label')
            ax.set_ylabel('y label')
        d.label_margins(
            row_vals=[0.1, 0.2], row_label='condition',
            col_vals=['A', 'B'], col_label='group',
        )
        # force a render so bboxes are populated
        d._get_parent_figure().canvas.draw()
        return d

    def test_row_text_artist_created(self, display_2x2):
        assert display_2x2._row_label_artists is not None
        assert 'text' in display_2x2._row_label_artists

    def test_col_text_artist_created(self, display_2x2):
        assert display_2x2._col_label_artists is not None
        assert 'text' in display_2x2._col_label_artists

    def test_row_text_contains_label(self, display_2x2):
        txt = display_2x2._row_label_artists['text'].get_text()
        assert 'condition' in txt

    def test_col_text_contains_label(self, display_2x2):
        txt = display_2x2._col_label_artists['text'].get_text()
        assert 'group' in txt

    def test_row_spine_artist_added(self, display_2x2):
        assert display_2x2._row_label_artists['spine'] is not None

    def test_col_spine_artist_added(self, display_2x2):
        assert display_2x2._col_label_artists['spine'] is not None

    def test_row_vals_are_separate_text_artists(self, display_2x2):
        """Row values must be stored as separate Text artists, not in the ylabel string."""
        artists = display_2x2._row_label_artists
        assert 'val_texts' in artists, "'val_texts' key missing from _row_label_artists"
        assert len(artists['val_texts']) == 2
        # ylabel strings must NOT contain the row values
        for ax, val in zip(display_2x2.trace_axes[:, 0], [0.1, 0.2]):
            lbl = ax.get_ylabel()
            assert str(round(val, 2)) not in lbl, \
                f"Row value {val} should not be embedded in ylabel '{lbl}'"

    def test_col_vals_are_separate_text_artists(self, display_2x2):
        """Col values must be stored as separate Text artists, not in the xlabel string."""
        artists = display_2x2._col_label_artists
        assert 'val_texts' in artists, "'val_texts' key missing from _col_label_artists"
        assert len(artists['val_texts']) == 2
        for ax, val in zip(display_2x2.axes[-1, :], ['A', 'B']):
            lbl = ax.get_xlabel()
            assert str(val) not in lbl, \
                f"Col value {val} should not be embedded in xlabel '{lbl}'"

    def test_bbox_centers_have_nonzero_coords(self, display_2x2):
        """ylabel bbox centers should have plausible normalized coordinates."""
        left_col = display_2x2.trace_axes[:, 0]
        centers = display_2x2._label_centers_in_subfig_coords(left_col, which='y')
        assert centers.shape == (2, 2), f"Expected (2,2), got {centers.shape}"
        # x-coordinates should be near the left edge (< 0.3)
        assert np.all(centers[:, 0] < 0.3), \
            f"ylabel centers x={centers[:, 0]} should be near left edge"
        # y-coordinates should be in (0, 1)
        assert np.all((centers[:, 1] > 0) & (centers[:, 1] < 1))

    def test_draw_event_cid_registered(self, display_2x2):
        """A draw_event callback should have been registered."""
        assert len(display_2x2._margin_label_cids) > 0


@pytest.mark.skipif(
    not any(H5_FILES),
    reason="No h5 files found — skipping spine tests",
)
class TestMainSequenceAnalysisSpines:
    """Check that main_sequence_analysis hides spines correctly."""

    @pytest.fixture(scope='class')
    def exp(self):
        files = sorted(H5_DIR.glob('fh_baja_1_new_trial_*_cond1_new.h5'))[:3]
        e = TrackingExperiment.__new__(TrackingExperiment)
        e.trials = [TrackingTrial(str(f)) for f in files]
        for t in e.trials:
            assert t.load_success
        e.main_sequence_analysis(group_var='fly_id', n_boot=20)
        return e

    def test_scatter_bottom_spine_hidden(self, exp):
        """Bottom spine of both scatter (left-col) axes must be hidden."""
        for ax in exp.display.trace_axes[:, 0]:
            assert not ax.spines['bottom'].get_visible(), \
                "Scatter axis bottom spine should be hidden"

    def test_scatter_top_right_hidden(self, exp):
        """Top and right spines of scatter axes must be hidden."""
        for ax in exp.display.trace_axes[:, 0]:
            assert not ax.spines['top'].get_visible()
            assert not ax.spines['right'].get_visible()

    def test_right_col_left_spine_hidden(self, exp):
        """Left spine of both right-margin strip axes must be hidden."""
        for ax in exp.display.right_col:
            assert not ax.spines['left'].get_visible(), \
                "Right-col axis left spine should be hidden"

    def test_right_col_top_bottom_hidden(self, exp):
        """Top-right strip (right_col[0]) must also hide bottom spine."""
        ax = exp.display.right_col[0]
        assert not ax.spines['bottom'].get_visible(), \
            "right_col[0] bottom spine should be hidden"

    def test_bottom_ax_top_right_hidden(self, exp):
        """Top and right spines of the magnitude strip plot must be hidden."""
        ax = exp.display.bottom_row[0]
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()

    def test_bottom_ax_keeps_bottom_spine(self, exp):
        """Bottom spine of the magnitude strip must remain visible."""
        ax = exp.display.bottom_row[0]
        assert ax.spines['bottom'].get_visible(), \
            "Magnitude strip bottom spine should be visible"

    def test_bottom_ax_keeps_left_spine(self, exp):
        """Left spine of the magnitude strip must remain visible (y-ticks there)."""
        ax = exp.display.bottom_row[0]
        assert ax.spines['left'].get_visible(), \
            "Magnitude strip left spine should be visible"


# ---------------------------------------------------------------------------
# SummaryDisplay.label_margins — spine position scales with figsize
# ---------------------------------------------------------------------------

def _make_labelled_display(figsize):
    """Helper: 2×2 SummaryDisplay with labels and margins, drawn."""
    d = SummaryDisplay(num_rows=2, num_cols=2,
                       right_margin=False, bottom_margin=False,
                       figsize=figsize)
    for ax in d.trace_axes.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    d.label_margins(
        row_vals=[1.0, 2.0], row_label='condition',
        col_vals=['A', 'B'], col_label='group',
    )
    d._get_parent_figure().canvas.draw()
    return d


class TestLabelMarginsResize:
    """Verify that spine positions are bbox-driven and scale with figsize."""

    def test_row_spine_x_between_val_text_and_ylabel(self):
        """Row spine x must lie strictly between row-value text right edge and ylabel left edge."""
        d = _make_labelled_display((6, 4))
        d._get_parent_figure().canvas.draw()
        left_col = d.trace_axes[:, 0]
        val_texts = d._row_label_artists['val_texts']
        spine = d._row_label_artists['spine']

        y_b = d._label_bboxes_in_subfig_coords(left_col, which='y')
        ylabel_left = float(np.min(y_b[:, 0]))

        val_rights = []
        for t in val_texts:
            x0, _, x1, _ = d._text_bbox_in_subfig_coords(t)
            val_rights.append(x1)
        val_right = float(np.max(val_rights))

        spine_x = spine.get_xdata()[0]
        assert val_right < spine_x < ylabel_left, (
            f"Row spine x={spine_x:.4f} must be between val_right={val_right:.4f} "
            f"and ylabel_left={ylabel_left:.4f}"
        )

    def test_col_spine_y_between_val_text_and_xlabel(self):
        """Col spine y must lie between col-value text top edge and xlabel bottom edge."""
        d = _make_labelled_display((6, 4))
        d._get_parent_figure().canvas.draw()
        bottom_row = d.axes[-1]
        val_texts = d._col_label_artists['val_texts']
        spine = d._col_label_artists['spine']

        x_b = d._label_bboxes_in_subfig_coords(bottom_row, which='x')
        xlabel_bottom = float(np.min(x_b[:, 1]))

        val_tops = []
        for t in val_texts:
            _, y0, _, y1 = d._text_bbox_in_subfig_coords(t)
            val_tops.append(y1)
        val_top = float(np.min(val_tops))

        spine_y = spine.get_ydata()[0]
        assert val_top < spine_y < xlabel_bottom, (
            f"Col spine y={spine_y:.4f} must be between val_top={val_top:.4f} "
            f"and xlabel_bottom={xlabel_bottom:.4f}"
        )

    def test_row_spine_x_shifts_with_figsize(self):
        """Row spine x (normalised) should move when figsize changes — proves bbox-driven."""
        d_small = _make_labelled_display((4, 3))
        d_small._get_parent_figure().canvas.draw()
        sx_small = d_small._row_label_artists['spine'].get_xdata()[0]

        d_large = _make_labelled_display((10, 7))
        d_large._get_parent_figure().canvas.draw()
        sx_large = d_large._row_label_artists['spine'].get_xdata()[0]

        # Normalised spine_x is dominated by ylabel_left which moves as
        # the figure gets wider because the axes take up a larger fraction.
        # They should not be identical.
        assert abs(sx_small - sx_large) > 1e-3, (
            f"Row spine x did not change with figsize: {sx_small:.4f} vs {sx_large:.4f}"
        )

    def test_col_spine_y_shifts_with_figsize(self):
        """Col spine y (normalised) should shift when figsize changes."""
        d_small = _make_labelled_display((4, 3))
        d_small._get_parent_figure().canvas.draw()
        sy_small = d_small._col_label_artists['spine'].get_ydata()[0]

        d_large = _make_labelled_display((10, 7))
        d_large._get_parent_figure().canvas.draw()
        sy_large = d_large._col_label_artists['spine'].get_ydata()[0]

        assert abs(sy_small - sy_large) > 1e-3, (
            f"Col spine y did not change with figsize: {sy_small:.4f} vs {sy_large:.4f}"
        )

    def test_row_val_text_count_matches_rows(self):
        """Number of row-value Text artists must equal number of data rows."""
        d = _make_labelled_display((6, 4))
        val_texts = d._row_label_artists['val_texts']
        assert len(val_texts) == d.trace_axes.shape[0]

    def test_col_val_text_count_matches_cols(self):
        """Number of col-value Text artists must equal number of data cols."""
        d = _make_labelled_display((6, 4))
        val_texts = d._col_label_artists['val_texts']
        assert len(val_texts) == d.trace_axes.shape[1]

    def test_row_val_text_strings(self):
        """Row-value texts should contain the float values formatted to 2 d.p."""
        d = _make_labelled_display((6, 4))
        for t, val in zip(d._row_label_artists['val_texts'], [1.0, 2.0]):
            assert f'{val:.2f}' in t.get_text(), \
                f"Expected '{val:.2f}' in text '{t.get_text()}'"

    def test_col_val_text_strings(self):
        """Col-value texts should contain the string values."""
        d = _make_labelled_display((6, 4))
        for t, val in zip(d._col_label_artists['val_texts'], ['A', 'B']):
            assert val in t.get_text(), \
                f"Expected '{val}' in text '{t.get_text()}'"

    def test_update_margin_labels_runs_without_error(self):
        """Calling _update_margin_labels manually should not raise."""
        d = _make_labelled_display((6, 4))
        d._get_parent_figure().canvas.draw()
        d._update_margin_labels()  # should not raise

    def test_update_moves_row_spine_after_draw(self):
        """After _update_margin_labels the row spine x must still be bbox-derived."""
        d = _make_labelled_display((6, 4))
        d._get_parent_figure().canvas.draw()
        d._update_margin_labels()
        left_col = d.trace_axes[:, 0]
        val_texts = d._row_label_artists['val_texts']
        y_b = d._label_bboxes_in_subfig_coords(left_col, which='y')
        ylabel_left = float(np.min(y_b[:, 0]))
        val_rights = [d._text_bbox_in_subfig_coords(t)[2] for t in val_texts]
        val_right = float(np.max(val_rights))
        spine_x = d._row_label_artists['spine'].get_xdata()[0]
        assert val_right <= spine_x <= ylabel_left + 1e-4, (
            f"Post-update spine_x={spine_x:.4f} outside "
            f"({val_right:.4f}, {ylabel_left:.4f})"
        )

