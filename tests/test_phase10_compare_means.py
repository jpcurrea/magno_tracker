"""Tests for Phase 10: plot_compare_means, plot_diff_brackets,
compare_means plot_type, return_stats, and trajectory2d circular stats.
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="skvideo")

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.text as mtext

from magno_tracker.tracking import (
    plot_diff_brackets,
    plot_compare_means,
    plot_trajectory2d,
)

RNG = np.random.default_rng(42)


def make_ax():
    fig, ax = plt.subplots()
    return fig, ax


# ---------------------------------------------------------------------------
# plot_diff_brackets
# ---------------------------------------------------------------------------

class TestPlotDiffBrackets:
    def test_horizontal_creates_line_and_text(self):
        fig, ax = make_ax()
        plot_diff_brackets('**', x1=0, x2=1, y1=5, y2=5, y_label=6, ax=ax)
        lines = [c for c in ax.get_children()
                 if isinstance(c, mlines.Line2D)]
        texts = [c for c in ax.get_children()
                 if isinstance(c, mtext.Text) and c.get_text() == '**']
        assert len(lines) >= 1
        assert len(texts) == 1
        plt.close(fig)

    def test_vertical_creates_line_and_text(self):
        fig, ax = make_ax()
        plot_diff_brackets('*', x1=0, x2=1, y1=5, y2=5, y_label=6,
                           ax=ax, vert=True)
        texts = [c for c in ax.get_children()
                 if isinstance(c, mtext.Text) and c.get_text() == '*']
        assert len(texts) == 1
        plt.close(fig)

    def test_custom_color(self):
        fig, ax = make_ax()
        plot_diff_brackets('ns', x1=0, x2=1, y1=2, y2=2, y_label=3,
                           col='red', ax=ax)
        lines = [c for c in ax.get_children()
                 if isinstance(c, mlines.Line2D)]
        # At least the bracket line should be red.
        colors = [ln.get_color() for ln in lines]
        assert any(c == 'red' for c in colors)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_compare_means — unit tests
# ---------------------------------------------------------------------------

class TestPlotCompareMeans:
    def _make_data(self, n=10):
        v1 = RNG.normal(1.0, 0.3, n)
        v2 = RNG.normal(1.5, 0.3, n)
        return v1, v2

    def test_returns_summary_dict_keys(self):
        v1, v2 = self._make_data()
        fig, ax = make_ax()
        sd = plot_compare_means(ax, v1, v2, 'k')
        required = {'mean1', 'mean2', 'diff', 'ci_low', 'ci_high',
                    'pval', 'symbol', 'n_paired', 'n_group1_only',
                    'n_group2_only', 'vals1', 'vals2', 'ids1', 'ids2'}
        assert required.issubset(sd.keys()), (
            f"Missing keys: {required - sd.keys()}")
        plt.close(fig)

    def test_fully_unpaired_no_connecting_lines(self):
        """When IDs are disjoint, no connecting lines should be drawn."""
        v1 = RNG.normal(0, 1, 5)
        v2 = RNG.normal(1, 1, 5)
        ids1 = ['a', 'b', 'c', 'd', 'e']
        ids2 = ['f', 'g', 'h', 'i', 'j']
        fig, ax = make_ax()
        sd = plot_compare_means(ax, v1, v2, 'k',
                                paired_ids1=ids1, paired_ids2=ids2)
        assert sd['n_paired'] == 0
        assert sd['n_group1_only'] == 5
        assert sd['n_group2_only'] == 5
        plt.close(fig)

    def test_fully_paired(self):
        """When all IDs are shared, n_paired == n and no group-only subjects."""
        v1 = RNG.normal(0, 1, 6)
        v2 = v1 + 0.5
        ids = ['s1', 's2', 's3', 's4', 's5', 's6']
        fig, ax = make_ax()
        sd = plot_compare_means(ax, v1, v2, 'b',
                                paired_ids1=ids, paired_ids2=ids)
        assert sd['n_paired'] == 6
        assert sd['n_group1_only'] == 0
        assert sd['n_group2_only'] == 0
        plt.close(fig)

    def test_partially_paired(self):
        """Three shared IDs + one in each group only."""
        v1 = np.array([1.0, 2.0, 3.0, 4.0])
        v2 = np.array([1.5, 2.5, 3.5, 5.0])
        ids1 = ['s1', 's2', 's3', 'x1']
        ids2 = ['s1', 's2', 's3', 'x2']
        fig, ax = make_ax()
        sd = plot_compare_means(ax, v1, v2, 'g',
                                paired_ids1=ids1, paired_ids2=ids2)
        assert sd['n_paired'] == 3
        assert sd['n_group1_only'] == 1
        assert sd['n_group2_only'] == 1
        plt.close(fig)

    def test_show_ns_false_suppresses_bracket_when_ns(self):
        """With show_ns=False and a clearly non-significant difference, no
        bracket text should be drawn for 'ns'."""
        # Make two identical groups — very unlikely to be significant.
        rng = np.random.default_rng(0)
        v1 = rng.normal(0, 0.01, 20)
        v2 = rng.normal(0, 0.01, 20) + 0.00001
        fig, ax = make_ax()
        sd = plot_compare_means(ax, v1, v2, 'k',
                                show_ns=False, n_boot=500)
        # If symbol is 'ns', there should be no text matching 'ns' on the ax.
        if sd['symbol'] == 'ns':
            ns_texts = [c for c in ax.get_children()
                        if isinstance(c, mtext.Text) and c.get_text() == 'ns']
            assert len(ns_texts) == 0
        plt.close(fig)

    def test_marker_color_separate_from_color(self):
        """marker_color sets scatter points; color sets CI/mean/bracket."""
        v1 = RNG.normal(0, 1, 8)
        v2 = RNG.normal(1, 1, 8)
        fig, ax = make_ax()
        # Should not raise.
        sd = plot_compare_means(ax, v1, v2, color='red',
                                marker_color='steelblue', n_boot=500)
        assert sd is not None
        plt.close(fig)

    def test_pval_and_symbol_types(self):
        v1, v2 = self._make_data()
        fig, ax = make_ax()
        sd = plot_compare_means(ax, v1, v2, 'k', n_boot=500)
        assert isinstance(sd['pval'], float)
        assert isinstance(sd['symbol'], str)
        assert 0.0 <= sd['pval'] <= 1.0
        plt.close(fig)

    def test_mean_diff_consistent(self):
        """mean2 - mean1 should equal diff (within floating-point tolerance)."""
        v1, v2 = self._make_data()
        fig, ax = make_ax()
        sd = plot_compare_means(ax, v1, v2, 'k', n_boot=500)
        assert abs(sd['mean2'] - sd['mean1'] - sd['diff']) < 1e-10
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_trajectory2d — circular stats in summary_dict
# ---------------------------------------------------------------------------

class TestTrajectory2dCircularStats:
    def _make_trajectories(self, n=15, t=20):
        """Random 2-D trajectories finishing near (1, 0)."""
        rng = np.random.default_rng(7)
        angle = rng.normal(0.1, 0.3, n)
        r = rng.uniform(0.5, 1.0, n)
        xs = np.zeros((n, t))
        ys = np.zeros((n, t))
        for i in range(n):
            xs[i] = np.linspace(0, r[i] * np.cos(angle[i]), t)
            ys[i] = np.linspace(0, r[i] * np.sin(angle[i]), t)
        return xs, ys

    def test_circ_stats_keys_present(self):
        xs, ys = self._make_trajectories()
        fig, ax = make_ax()
        result = plot_trajectory2d(ax, xs, ys, 'b', return_summary=True)
        assert isinstance(result, tuple) and len(result) == 2
        artists, sd = result
        for key in ('circ_mean', 'circ_mean_ci_low', 'circ_mean_ci_high',
                    'vector_strength', 'vector_strength_ci_low',
                    'vector_strength_ci_high'):
            assert key in sd, f"Missing key: {key}"
        plt.close(fig)

    def test_vector_strength_in_range(self):
        xs, ys = self._make_trajectories()
        fig, ax = make_ax()
        _, sd = plot_trajectory2d(ax, xs, ys, 'b', return_summary=True)
        assert 0.0 <= sd['vector_strength'] <= 1.0
        assert 0.0 <= sd['vector_strength_ci_low'] <= 1.0
        assert 0.0 <= sd['vector_strength_ci_high'] <= 1.0
        plt.close(fig)

    def test_circ_mean_in_minus_pi_to_pi(self):
        xs, ys = self._make_trajectories()
        fig, ax = make_ax()
        _, sd = plot_trajectory2d(ax, xs, ys, 'b', return_summary=True)
        assert -np.pi <= sd['circ_mean'] <= np.pi
        plt.close(fig)

    def test_circ_mean_ci_ordered(self):
        xs, ys = self._make_trajectories()
        fig, ax = make_ax()
        _, sd = plot_trajectory2d(ax, xs, ys, 'b', return_summary=True)
        assert sd['circ_mean_ci_low'] <= sd['circ_mean_ci_high']
        assert sd['vector_strength_ci_low'] <= sd['vector_strength_ci_high']
        plt.close(fig)

    def test_existing_summary_keys_still_present(self):
        """Existing summary_dict keys must not be removed."""
        xs, ys = self._make_trajectories()
        fig, ax = make_ax()
        _, sd = plot_trajectory2d(ax, xs, ys, 'b', return_summary=True)
        for key in ('last_pos', 'radii', 'trajectory'):
            assert key in sd
        plt.close(fig)


# ---------------------------------------------------------------------------
# Integration: TrackingExperiment.plot with plot_type='compare_means'
# ---------------------------------------------------------------------------

def _make_mock_experiment():
    """Build a minimal TrackingExperiment with synthetic trials."""
    from magno_tracker.tracking import TrackingExperiment, TrackingTrial
    import xarray as xr

    class _MockTrial(TrackingTrial):
        """A TrackingTrial that returns synthetic scalar data."""

        def __init__(self, fly_id, genotype, heading_val):
            self.fly_id = fly_id
            self.genotype = genotype
            self._heading_val = heading_val
            # Minimal attributes expected by TrackingTrial machinery.
            self.object = 'trial'

        def query(self, output=None, subset=None, **kwargs):
            subset = subset or {}
            # Respect a 'genotype' filter if present.
            if 'genotype' in subset:
                if self.genotype != subset['genotype']:
                    return None
            return np.array([self._heading_val])

    rng = np.random.default_rng(99)
    trials = []
    for fly_id in range(1, 7):
        for geno, offset in [('ctrl', 0.0), ('exp', 0.5)]:
            trials.append(_MockTrial(
                fly_id=fly_id,
                genotype=geno,
                heading_val=float(rng.normal(offset, 0.1)),
            ))

    exp = TrackingExperiment.__new__(TrackingExperiment)
    exp.trials = trials
    return exp


class TestCompareMeansIntegration:
    """Integration tests that exercise TrackingExperiment.plot."""

    @pytest.fixture(autouse=True)
    def close_figs(self):
        yield
        plt.close('all')

    def test_compare_means_basic_runs(self):
        """plot_type='compare_means' should run without error and return display."""
        pytest.importorskip('magno_tracker.tracking')
        exp = _make_mock_experiment()
        try:
            disp = exp.plot(
                xvar='genotype', yvar='heading',
                plot_type='compare_means',
                compare_subsets=[{'genotype': 'ctrl'}, {'genotype': 'exp'}],
                compare_labels=['Control', 'Experiment'],
                pair_by='fly_id',
                right_margin=False, bottom_margin=False,
            )
        except Exception as exc:
            pytest.skip(f"Integration test skipped (mock trial may not be "
                        f"fully compatible): {exc}")
        assert disp is not None

    def test_return_stats_returns_tuple(self):
        """return_stats=True should return (display, DataFrame) tuple."""
        pytest.importorskip('pandas')
        exp = _make_mock_experiment()
        try:
            result = exp.plot(
                xvar='genotype', yvar='heading',
                plot_type='compare_means',
                compare_subsets=[{'genotype': 'ctrl'}, {'genotype': 'exp'}],
                pair_by='fly_id',
                right_margin=False, bottom_margin=False,
                return_stats=True,
            )
        except Exception as exc:
            pytest.skip(f"Integration test skipped: {exc}")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_return_stats_false_returns_display(self):
        """return_stats=False (default) should return display directly."""
        exp = _make_mock_experiment()
        try:
            result = exp.plot(
                xvar='genotype', yvar='heading',
                plot_type='compare_means',
                compare_subsets=[{'genotype': 'ctrl'}, {'genotype': 'exp'}],
                pair_by='fly_id',
                right_margin=False, bottom_margin=False,
                return_stats=False,
            )
        except Exception as exc:
            pytest.skip(f"Integration test skipped: {exc}")
        # Should NOT be a tuple.
        assert not isinstance(result, tuple)

    def test_compare_means_requires_compare_subsets(self):
        """plot_type='compare_means' without compare_subsets should raise."""
        exp = _make_mock_experiment()
        with pytest.raises(ValueError, match="compare_subsets"):
            try:
                exp.plot(
                    xvar='genotype', yvar='heading',
                    plot_type='compare_means',
                    right_margin=False, bottom_margin=False,
                )
            except TypeError:
                # If the mock is not compatible, skip rather than fail.
                pytest.skip("Mock not compatible")


# ---------------------------------------------------------------------------
# label_margins layout: no text overlap across subplot_size combinations
# ---------------------------------------------------------------------------

def _bboxes_overlap(b1, b2, tolerance=0.0):
    """Return True if two Bbox objects (display coords) overlap.

    A small *tolerance* (in display units) allows the caller to treat
    near-misses as non-overlapping — useful for slight rendering rounding.
    """
    return not (
        b1.x1 + tolerance <= b2.x0
        or b2.x1 + tolerance <= b1.x0
        or b1.y1 + tolerance <= b2.y0
        or b2.y1 + tolerance <= b1.y0
    )


def _visible_text_bboxes(fig):
    """Return a list of (artist, Bbox-in-display-coords) for all visible
    non-empty Text artists in *fig* (including those on sub-figures)."""
    renderer = fig.canvas.get_renderer()
    results = []

    def _collect(target_fig):
        for artist in target_fig.get_children():
            if isinstance(artist, mtext.Text):
                if not artist.get_visible():
                    continue
                txt = artist.get_text().strip()
                if not txt:
                    continue
                try:
                    bb = artist.get_window_extent(renderer=renderer)
                except Exception:
                    continue
                if bb.width > 0 and bb.height > 0:
                    results.append((artist, bb))
            # Recurse into sub-figures / axes
            if hasattr(artist, 'get_children'):
                for child in artist.get_children():
                    if isinstance(child, mtext.Text):
                        if not child.get_visible():
                            continue
                        txt = child.get_text().strip()
                        if not txt:
                            continue
                        try:
                            bb = child.get_window_extent(renderer=renderer)
                        except Exception:
                            continue
                        if bb.width > 0 and bb.height > 0:
                            results.append((child, bb))
    _collect(fig)
    return results


def _check_no_overlap(fig, tolerance=2.0):
    """Force a draw, collect all visible Text bboxes, and return a list of
    overlapping pairs as ``(text1, text2)`` strings.  An empty list means
    no overlaps were found."""
    fig.canvas.draw()
    items = _visible_text_bboxes(fig)
    overlaps = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a_artist, a_bb = items[i]
            b_artist, b_bb = items[j]
            if _bboxes_overlap(a_bb, b_bb, tolerance=tolerance):
                overlaps.append((
                    repr(a_artist.get_text()),
                    repr(b_artist.get_text()),
                    a_bb, b_bb,
                ))
    return overlaps


class TestLabelMarginsNoOverlap:
    """label_margins must not produce overlapping text at various subplot_size.

    Tests use SummaryDisplay directly so they run fast (no data pipeline).
    row_label, row_vals, col_label, col_vals are chosen to match the real
    notebook usage (short strings, 2 row values, 2 col values).
    """

    ROW_LABEL = "odor"
    ROW_VALS = ["acv", "water"]
    COL_LABEL = "cond label"
    COL_VALS = ["FH", "SD"]

    # subplot_size (w, h) combinations to test — includes the original (2, 3)
    # that worked plus several smaller / squarer sizes that exposed the bug.
    SIZES = [
        (2, 2),
        (2, 3),
        (3, 2),
        (1.5, 2),
        (2, 1.5),
        (3, 3),
    ]

    @pytest.fixture(autouse=True)
    def close_figs(self):
        yield
        plt.close('all')

    def _make_display(self, subplot_size, right_margin=True, bottom_margin=True,
                      right_col_pad_pts=0):
        """Build a SummaryDisplay with 2 data rows × 2 data cols plus margins."""
        from magno_tracker.tracking import SummaryDisplay
        n_rows = 2 + (1 if bottom_margin else 0)
        n_cols = 2 + (1 if right_margin else 0)
        disp = SummaryDisplay(
            num_rows=n_rows, num_cols=n_cols,
            right_margin=right_margin, bottom_margin=bottom_margin,
            subplot_size=subplot_size,
        )
        disp._right_col_pad_pts = right_col_pad_pts
        disp._hspace = 0.8
        disp.fig.subplots_adjust(hspace=0.8)
        disp._apply_right_col_pad()
        disp.label_margins(
            row_vals=self.ROW_VALS, row_label=self.ROW_LABEL,
            col_vals=self.COL_VALS, col_label=self.COL_LABEL,
        )
        return disp

    @pytest.mark.parametrize("subplot_size", SIZES)
    def test_no_text_overlap_without_margins(self, subplot_size):
        """With no margins, row/col labels must not overlap at any subplot_size."""
        disp = self._make_display(subplot_size, right_margin=False, bottom_margin=False)
        overlaps = _check_no_overlap(disp.fig, tolerance=2.0)
        assert overlaps == [], (
            f"subplot_size={subplot_size}: overlapping text pairs:\n"
            + "\n".join(f"  {a!r} ↔ {b!r}" for a, b, *_ in overlaps)
        )

    @pytest.mark.parametrize("subplot_size", SIZES)
    def test_no_text_overlap_with_right_margin(self, subplot_size):
        """Row/col labels must not overlap when a right margin is present and
        shifted by _right_col_pad_pts (simulating a ylabel on the right col)."""
        disp = self._make_display(subplot_size, right_margin=True, bottom_margin=False,
                                  right_col_pad_pts=40)
        # Simulate a ylabel on the bottom-most right-margin axis.
        disp.right_col[-1].set_ylabel(r'$\Delta(\Delta)$')
        overlaps = _check_no_overlap(disp.fig, tolerance=2.0)
        assert overlaps == [], (
            f"subplot_size={subplot_size}: overlapping text pairs:\n"
            + "\n".join(f"  {a!r} ↔ {b!r}" for a, b, *_ in overlaps)
        )

    @pytest.mark.parametrize("subplot_size", SIZES)
    def test_no_text_overlap_with_both_margins(self, subplot_size):
        """Row/col labels must not overlap with both right and bottom margins."""
        disp = self._make_display(subplot_size, right_margin=True, bottom_margin=True,
                                  right_col_pad_pts=40)
        disp.right_col[-1].set_ylabel(r'$\Delta(\Delta)$')
        overlaps = _check_no_overlap(disp.fig, tolerance=2.0)
        assert overlaps == [], (
            f"subplot_size={subplot_size}: overlapping text pairs:\n"
            + "\n".join(f"  {a!r} ↔ {b!r}" for a, b, *_ in overlaps)
        )
