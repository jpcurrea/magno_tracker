import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="skvideo")
import numpy as np
import pytest
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from magno_tracker.tracking import (
    plot_line, plot_hist2d, plot_trajectory2d, plot_histogram, plot_scatter,
)

OUT_DIR = Path(__file__).parent / 'plot_outputs'


@pytest.fixture(autouse=True)
def save_plots(request, monkeypatch):
    """Save every figure to plot_outputs/<test_name>.png before it is closed."""
    OUT_DIR.mkdir(exist_ok=True)
    counter = {'n': 0}
    original_close = plt.close

    def saving_close(fig=None):
        if fig is not None and plt.fignum_exists(fig.number):
            suffix = f'_{counter["n"]}' if counter['n'] > 0 else ''
            fig.savefig(
                OUT_DIR / f'{request.node.name}{suffix}.png',
                dpi=100, bbox_inches='tight',
            )
            counter['n'] += 1
        original_close(fig)

    monkeypatch.setattr(plt, 'close', saving_close)
    yield


def make_ax():
    fig, ax = plt.subplots()
    return fig, ax


# ---------------------------------------------------------------------------
# plot_line
# ---------------------------------------------------------------------------

class TestPlotLine:
    def test_traces_only(self):
        """Individual trace lines use trace_color (default gray); no mean overlay."""
        rng = np.random.default_rng(0)
        xs = rng.normal(size=(5, 20))
        ys = rng.normal(size=(5, 20))
        fig, ax = make_ax()
        artists = plot_line(ax, xs, ys, color='r')
        lines = [a for a in artists if isinstance(a, matplotlib.lines.Line2D)]
        assert len(lines) == 5
        for ln in lines:
            assert ln.get_color() == 'gray'
        plt.close(fig)

    def test_custom_trace_color(self):
        """trace_color overrides the default gray."""
        rng = np.random.default_rng(14)
        xs = rng.normal(size=(3, 10))
        ys = rng.normal(size=(3, 10))
        fig, ax = make_ax()
        artists = plot_line(ax, xs, ys, color='r', trace_color='b')
        lines = [a for a in artists if isinstance(a, matplotlib.lines.Line2D)]
        for ln in lines:
            assert ln.get_color() == 'b'
        plt.close(fig)

    def test_trace_alpha_via_kw(self):
        """alpha kwarg controls individual trace opacity."""
        rng = np.random.default_rng(15)
        xs = rng.normal(size=(4, 10))
        ys = rng.normal(size=(4, 10))
        fig, ax = make_ax()
        artists = plot_line(ax, xs, ys, color='r', alpha=0.1)
        lines = [a for a in artists if isinstance(a, matplotlib.lines.Line2D)]
        for ln in lines:
            np.testing.assert_allclose(ln.get_alpha(), 0.1)
        plt.close(fig)

    def test_mean_overlay(self):
        """With summary_func, white backing + colored mean line are added."""
        rng = np.random.default_rng(1)
        xs = rng.normal(size=(8, 20))
        ys = rng.normal(size=(8, 20))
        fig, ax = make_ax()
        artists = plot_line(ax, xs, ys, color='b', summary_func=np.nanmean)
        lines = [a for a in artists if isinstance(a, matplotlib.lines.Line2D)]
        # 8 gray + 1 white backing + 1 colored mean
        assert len(lines) == 10
        mean_lines = [ln for ln in lines if ln.get_color() not in ('gray', 'w')]
        assert len(mean_lines) == 1
        assert mean_lines[0].get_color() == 'b'
        plt.close(fig)

    def test_ci_band(self):
        """With ci=True, a fill_betweenx band is also returned."""
        rng = np.random.default_rng(2)
        xs = rng.normal(size=(6, 15))
        ys = rng.normal(size=(6, 15))
        fig, ax = make_ax()
        artists = plot_line(ax, xs, ys, color='g',
                            summary_func=np.nanmean, ci=True, n_boot=50)
        poly = [a for a in artists
                if isinstance(a, matplotlib.collections.PolyCollection)]
        assert len(poly) == 1
        plt.close(fig)

    def test_returns_list(self):
        rng = np.random.default_rng(3)
        xs, ys = rng.normal(size=(3, 10)), rng.normal(size=(3, 10))
        fig, ax = make_ax()
        result = plot_line(ax, xs, ys, color='k')
        assert isinstance(result, list)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_hist2d
# ---------------------------------------------------------------------------

class TestPlotHist2d:
    def test_returns_mesh(self):
        rng = np.random.default_rng(4)
        xs = rng.normal(size=(200,))
        ys = rng.normal(size=(200,))
        fig, ax = make_ax()
        mesh = plot_hist2d(ax, xs, ys, color=(1, 0, 0), bins=20, cbar=False)
        assert isinstance(mesh, matplotlib.collections.QuadMesh)
        plt.close(fig)

    def test_colormap_saturates_at_color(self):
        """The colormap should go from white to the given color."""
        rng = np.random.default_rng(5)
        xs, ys = rng.normal(size=(100,)), rng.normal(size=(100,))
        fig, ax = make_ax()
        color = (0.0, 0.5, 1.0)
        mesh = plot_hist2d(ax, xs, ys, color=color, bins=10, cbar=False)
        cmap = mesh.cmap
        # The colormap's low end should be white
        np.testing.assert_allclose(cmap(0.0)[:3], (1, 1, 1), atol=1e-6)
        # And the high end should match the requested color
        np.testing.assert_allclose(cmap(1.0)[:3], color, atol=1e-6)
        plt.close(fig)

    def test_vmax_respected(self):
        rng = np.random.default_rng(6)
        xs, ys = rng.normal(size=(50,)), rng.normal(size=(50,))
        fig, ax = make_ax()
        mesh = plot_hist2d(ax, xs, ys, color='k', bins=5, vmax=3, cbar=False)
        assert mesh.norm.vmax == 3
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_trajectory2d
# ---------------------------------------------------------------------------

class TestPlotTrajectory2d:
    def _make_circular_xs(self, n_traces=5, n_frames=30, seed=7):
        rng = np.random.default_rng(seed)
        return rng.uniform(-np.pi, np.pi, size=(n_traces, n_frames))

    def test_basic_output(self):
        xs = self._make_circular_xs()
        fig, ax = make_ax()
        artists = plot_trajectory2d(ax, xs, None, color='r')
        assert len(artists) > 0
        plt.close(fig)

    def test_custom_trace_color(self):
        """trace_color overrides the default black for trajectory lines."""
        xs = self._make_circular_xs(n_traces=3)
        fig, ax = make_ax()
        artists = plot_trajectory2d(ax, xs, None, color='r', trace_color='b')
        lines = [a for a in artists if isinstance(a, matplotlib.lines.Line2D)]
        assert len(lines) > 0
        for ln in lines:
            assert ln.get_color() == 'b'
        plt.close(fig)

    def test_trace_alpha_via_kw(self):
        """alpha kwarg controls trajectory line opacity."""
        xs = self._make_circular_xs(n_traces=3)
        fig, ax = make_ax()
        artists = plot_trajectory2d(ax, xs, None, color='r', alpha=0.1)
        lines = [a for a in artists if isinstance(a, matplotlib.lines.Line2D)]
        for ln in lines:
            np.testing.assert_allclose(ln.get_alpha(), 0.1)
        plt.close(fig)

    def test_mean_line_kwarg(self):
        """mean_line=True should add extra line artists."""
        xs = self._make_circular_xs()
        fig, ax = make_ax()
        arts_no_mean = plot_trajectory2d(ax, xs, None, color='b')
        plt.close(fig)
        fig, ax = make_ax()
        arts_mean = plot_trajectory2d(ax, xs, None, color='b', mean_line=True)
        plt.close(fig)
        assert len(arts_mean) > len(arts_no_mean)

    def test_circle_kwarg(self):
        xs = self._make_circular_xs(n_traces=4)
        fig, ax = make_ax()
        arts = plot_trajectory2d(ax, xs, None, color='g', circle=True)
        circles = [a for a in arts if isinstance(a, plt.Circle)]
        assert len(circles) == 4
        plt.close(fig)

    def test_contour_adds_ellipse(self):
        """contour=True should add a matplotlib Ellipse artist."""
        from matplotlib.patches import Ellipse
        xs = self._make_circular_xs(n_traces=20, seed=42)
        fig, ax = make_ax()
        arts = plot_trajectory2d(ax, xs, None, color='r', contour=True)
        ellipses = [a for a in arts if isinstance(a, Ellipse)]
        assert len(ellipses) == 1
        plt.close(fig)

    def test_circ_hist_draws_wedges(self):
        """circ_hist=True should draw Wedge artists forming the ring histogram."""
        from matplotlib.patches import Wedge
        xs = self._make_circular_xs(n_traces=30, seed=3)
        fig, ax = make_ax()
        arts = plot_trajectory2d(ax, xs, None, color='b', circ_hist=True)
        wedges = [a for a in arts if isinstance(a, Wedge)]
        assert len(wedges) > 0
        plt.close(fig)

    def test_circ_hist_bin_count(self):
        """bins kwarg controls the number of Wedges drawn."""
        from matplotlib.patches import Wedge
        n_bins = 36
        xs = self._make_circular_xs(n_traces=30, seed=4)
        fig, ax = make_ax()
        arts = plot_trajectory2d(ax, xs, None, color='b', circ_hist=True, bins=n_bins)
        wedges = [a for a in arts if isinstance(a, Wedge)]
        assert len(wedges) == n_bins
        plt.close(fig)

    def test_circ_hist_adds_ci_arc(self):
        """circ_hist=True should also add Arc artists for the CI."""
        from matplotlib.patches import Arc
        xs = self._make_circular_xs(n_traces=30, seed=5)
        fig, ax = make_ax()
        arts = plot_trajectory2d(ax, xs, None, color='r', circ_hist=True)
        arcs = [a for a in arts if isinstance(a, Arc)]
        # two arcs: white outline + colored
        assert len(arcs) == 2
        plt.close(fig)

    def test_circ_hist_sets_axis_limits(self):
        """circ_hist=True should expand axis limits to ±1.27 to show the ring."""
        xs = self._make_circular_xs(n_traces=30, seed=6)
        fig, ax = make_ax()
        plot_trajectory2d(ax, xs, None, color='g', circ_hist=True)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] <= -1.27 and xlim[1] >= 1.27
        assert ylim[0] <= -1.27 and ylim[1] >= 1.27
        plt.close(fig)

    def test_square_aspect_ratio(self):
        """plot_trajectory2d should force equal aspect ratio."""
        xs = self._make_circular_xs()
        fig, ax = make_ax()
        plot_trajectory2d(ax, xs, None, color='k')
        # matplotlib may return 'equal' or 1.0 depending on version
        assert ax.get_aspect() in ('equal', 1.0)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_histogram
# ---------------------------------------------------------------------------

class TestPlotHistogram:
    def test_basic(self):
        rng = np.random.default_rng(8)
        xs = rng.normal(size=(100,))
        fig, ax = make_ax()
        result = plot_histogram(ax, xs, bins=20, color='r')
        assert len(result) == 3  # n, bins, patches
        plt.close(fig)

    def test_summary_func_adds_vline(self):
        rng = np.random.default_rng(9)
        xs = rng.normal(loc=2.0, size=(200,))
        fig, ax = make_ax()
        result = plot_histogram(ax, xs, bins=20, color='b',
                                summary_func=np.nanmean)
        assert len(result) == 4  # n, bins, patches, vline
        vline = result[3]
        assert isinstance(vline, matplotlib.lines.Line2D)
        # vline x-position should be close to the mean
        np.testing.assert_allclose(
            vline.get_xdata()[0], np.nanmean(xs), atol=0.05)
        plt.close(fig)

    def test_nans_ignored(self):
        xs = np.array([1.0, 2.0, np.nan, 3.0, np.nan])
        fig, ax = make_ax()
        result = plot_histogram(ax, xs, bins=5, color='k')
        assert len(result) == 3
        plt.close(fig)

    def test_probability(self):
        rng = np.random.default_rng(10)
        xs = rng.normal(size=(500,))
        fig, ax = make_ax()
        n, bin_edges, _ = plot_histogram(ax, xs, bins=20, color='k',
                                         probability=True)
        # density=True: integral ≈ 1, not count
        widths = np.diff(bin_edges)
        integral = np.sum(n * widths)
        np.testing.assert_allclose(integral, 1.0, atol=0.01)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_scatter
# ---------------------------------------------------------------------------

class TestPlotScatter:
    def test_returns_pathcollection(self):
        rng = np.random.default_rng(11)
        xs = rng.uniform(-np.pi, np.pi, size=(50,))
        ys = rng.normal(size=(50,))
        fig, ax = make_ax()
        sc = plot_scatter(ax, xs, ys, color='r', s=20, alpha=0.8)
        assert isinstance(sc, matplotlib.collections.PathCollection)
        plt.close(fig)

    def test_jitter_changes_positions(self):
        rng = np.random.default_rng(12)
        xs = np.zeros(30)
        ys = rng.normal(size=(30,))
        fig, ax = make_ax()
        sc = plot_scatter(ax, xs, ys, color='k', jitter_std=0.5, s=20, alpha=0.8)
        offsets = sc.get_offsets()
        # with jitter, x positions should not all be zero
        assert not np.allclose(offsets[:, 0], 0.0)
        plt.close(fig)

    def test_no_jitter_preserves_xs(self):
        rng = np.random.default_rng(13)
        xs = rng.uniform(-1, 1, size=(20,))
        ys = rng.normal(size=(20,))
        fig, ax = make_ax()
        sc = plot_scatter(ax, xs, ys, color='b', jitter_std=0.0, s=20, alpha=0.8)
        offsets = sc.get_offsets()
        np.testing.assert_allclose(offsets[:, 0], xs, atol=1e-10)
        plt.close(fig)

    def test_marker_color_overrides_color(self):
        """marker_color overrides color for the scatter points."""
        rng = np.random.default_rng(16)
        xs = rng.uniform(-1, 1, size=(10,))
        ys = rng.normal(size=(10,))
        fig, ax = make_ax()
        sc = plot_scatter(ax, xs, ys, color='r', marker_color='b', s=20, alpha=0.8)
        rendered = sc.get_facecolor()
        expected = matplotlib.colors.to_rgba('b', alpha=0.8)
        np.testing.assert_allclose(rendered[0], expected, atol=1e-3)
        plt.close(fig)

    def test_marker_color_none_falls_back_to_color(self):
        """When marker_color is None, color is used for the markers."""
        rng = np.random.default_rng(17)
        xs = rng.uniform(-1, 1, size=(10,))
        ys = rng.normal(size=(10,))
        fig, ax = make_ax()
        sc = plot_scatter(ax, xs, ys, color='g', marker_color=None, s=20, alpha=0.8)
        rendered = sc.get_facecolor()
        expected = matplotlib.colors.to_rgba('g', alpha=0.8)
        np.testing.assert_allclose(rendered[0], expected, atol=1e-3)
        plt.close(fig)
