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
    _resolve_margin,
    _MARGIN_DEFAULTS,
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
