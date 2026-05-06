"""Tools for analyzing tracking data files created by magnocube tracker.

Objects
-------
TrackingTrial
    Loads an h5 dataset representing a TrackingTrial from the magnocube library.
TrialsDataset
    Loads a whole set of TrackingTrial instances, allowing queries and
    statistical analyses along experimental parameters.
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part in the Zarr format 3 specification",
    category=UserWarning,
    module="zarr",
)

import copy
import numpy as np
import numpy
numpy.float = np.float64
numpy.int = numpy.int_

import h5py                             # for handling h5 tracking datasets
import math
from matplotlib import pyplot as plt    # for plotting data
import matplotlib
import os
import pickle
import scipy
from scipy import optimize
import seaborn as sbn
import skimage
from skvideo import io
import sys
from time import time
import pandas as pd
import xarray as xr

from functools import partial

# for the video analysis
# from ring_analysis import *
# for plotting

# use interactive plotting
# matplotlib.use('Qt5Agg')
# set ticks to the inside of the spines
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

blue, green, yellow, orange, red, purple = [
    (0.30, 0.45, 0.69), (0.33, 0.66, 0.41), (0.83, 0.74, 0.37),
    (0.78, 0.50, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]


# === Phase 1 Helper Functions ===
def resolve_colors(row_cmap, col_cmap, row_vals, col_vals, default_color='k'):
    """
    Generate a (num_rows, num_cols, 3) array of RGB colors for a grid.

    Blending rules:
    - Both row_cmap and col_cmap specified → geometric mean blend per cell.
    - Only row_cmap → each row has one color, broadcast across all columns.
    - Only col_cmap → each column has one color, broadcast across all rows.
    - Neither specified → default_color fills every cell.

    Parameters
    ----------
    row_cmap : matplotlib colormap callable, list of colors, or None
        Colormap or list of colors for rows.
    col_cmap : matplotlib colormap callable, list of colors, or None
        Colormap or list of colors for columns.
    row_vals : array-like
        Unique values for grid rows (used only for length).
    col_vals : array-like
        Unique values for grid columns (used only for length).
    default_color : str or tuple, optional
        Fallback color when neither cmap is provided (default 'k').

    Returns
    -------
    color_arr : ndarray, shape (num_rows, num_cols, 3)
        RGB values in [0, 1] for every grid cell.

    Examples
    --------
    >>> import numpy as np
    >>> colors = resolve_colors(None, None, [1, 2], [1, 2, 3], default_color='k')
    >>> colors.shape
    (2, 3, 3)
    >>> np.allclose(colors[0, 0], [0., 0., 0.])
    True
    """
    num_rows = len(row_vals)
    num_cols = len(col_vals)

    def to_rgb(c):
        return matplotlib.colors.to_rgb(c)

    def cmap_to_colors(cmap, n, label, vals=None):
        """Map a cmap spec to an (n, 3) float array of RGB colors.

        Parameters
        ----------
        cmap : str, callable, list/tuple/ndarray
            A named matplotlib colormap string, a callable colormap, or an
            explicit list/array/tuple of ``n`` colors.
        n : int
            Number of colors needed.
        label : str
            'row' or 'col' — used in error messages.
        vals : array-like, optional
            The actual data values corresponding to each color slot.  Required
            when ``cmap`` is a string so the colormap can be normalised against
            the data range.  Ignored for callable and list inputs.
        """
        if isinstance(cmap, str):
            if vals is None:
                raise ValueError(
                    f"string {label}_cmap requires vals to be provided for normalisation.")
            norm_vals = vals
            if len(norm_vals) > 0 and isinstance(norm_vals[0], (str, bytes)):
                norm_vals = np.argsort(norm_vals).astype(float)
            else:
                norm_vals = np.asarray(norm_vals, dtype=float)
            norm = matplotlib.colors.Normalize(norm_vals.min(), norm_vals.max())
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            return sm.to_rgba(norm_vals)[:, :3]
        elif hasattr(cmap, '__call__'):
            return np.array([cmap(i / max(n - 1, 1))[:3] for i in range(n)])
        elif isinstance(cmap, (list, tuple, np.ndarray)):
            colors = np.array([to_rgb(c) for c in cmap])
            if len(colors) != n:
                raise ValueError(
                    f"Length of {label}_cmap ({len(colors)}) does not match "
                    f"number of {label}s ({n}).")
            return colors
        else:
            raise TypeError(
                f"{label}_cmap must be a string colormap name, a callable colormap, "
                f"or a list/tuple/array of colors.")

    row_specified = row_cmap is not None
    col_specified = col_cmap is not None

    if row_specified and col_specified:
        row_colors = cmap_to_colors(row_cmap, num_rows, 'row', vals=row_vals)
        col_colors = cmap_to_colors(col_cmap, num_cols, 'col', vals=col_vals)
        color_arr = np.sqrt(0.5 * (
            row_colors[:, np.newaxis, :] ** 2 +
            col_colors[np.newaxis, :, :] ** 2
        ))
    elif row_specified:
        row_colors = cmap_to_colors(row_cmap, num_rows, 'row', vals=row_vals)
        color_arr = np.broadcast_to(
            row_colors[:, np.newaxis, :], (num_rows, num_cols, 3)).copy()
    elif col_specified:
        col_colors = cmap_to_colors(col_cmap, num_cols, 'col', vals=col_vals)
        color_arr = np.broadcast_to(
            col_colors[np.newaxis, :, :], (num_rows, num_cols, 3)).copy()
    else:
        color_arr = np.full((num_rows, num_cols, 3), to_rgb(default_color), dtype=float)

    return color_arr


def get_grid_vals(experiment, var, subset):
    """
    Return unique, non-NaN values for a grid variable, filtered by subset.

    Parameters
    ----------
    experiment : TrackingExperiment
        The experiment object containing data.
    var : str
        Name of the variable to extract.
    subset : dict
        Dictionary of filters to apply.

    Returns
    -------
    vals : ndarray
        1D array of unique, valid values for the variable.

    Examples
    --------
    Usage with a real experiment is shown in the tutorial notebook.  The
    function filters NaN/b'nan' and returns sorted unique values::

        vals = get_grid_vals(exp, 'condition', subset={'bg_gain': '>0'})
        # → e.g. array([1, 2, 3, 4])
    """
    arr = experiment.query(output=var, subset=subset, skip_empty=True)
    # If arr is a list of arrays, flatten it
    if isinstance(arr, (list, tuple)) and len(arr) > 0 and hasattr(arr[0], '__array__'):
        arr = np.concatenate(arr)
    arr = np.asarray(arr)
    # Remove NaN and b'nan' (bytes)
    if arr.dtype.kind in {'S', 'U'}:
        arr = arr[arr != b'nan'] if arr.dtype.kind == 'S' else arr[arr != 'nan']
    else:
        arr = arr[~np.isnan(arr)]
    vals = np.unique(arr)
    return vals


def omit_wrapping(arr, threshold=np.pi/2, return_mask=False):
    """
    Insert NaNs at discontinuities in a circular array to break plot lines.

    Parameters
    ----------
    arr : ndarray
        Input array of angles (1D or 2D).
    threshold : float, optional
        Discontinuity threshold (default is pi/2).
    return_mask : bool, optional
        If True, also return a boolean mask (True for valid, non-NaN entries).

    Returns
    -------
    arr_out : ndarray
        Copy of arr with NaNs inserted at discontinuities.
    mask : ndarray, optional
        Boolean mask, True for valid (non-NaN) entries. Only if return_mask=True.

    Examples
    --------
    >>> import numpy as np
    >>> omit_wrapping(np.array([0.0, 0.1, 3.0, 3.1]))
    array([0. , 0.1, nan, 3. , 3.1])
    >>> out, mask = omit_wrapping(np.array([0.0, 0.1, 3.0, 3.1]), return_mask=True)
    >>> mask
    array([ True,  True, False,  True,  True])
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        diffs = np.abs(np.diff(arr))
        idx = np.where(diffs > threshold)[0] + 1
        arr_out = arr.astype(float).copy()
        if idx.size > 0:
            arr_out = np.insert(arr_out, idx, np.nan)
        if return_mask:
            mask = ~np.isnan(arr_out)
            return arr_out, mask
        return arr_out
    elif arr.ndim == 2:
        arrs, masks = [], []
        for row in arr:
            if return_mask:
                wrapped, mask = omit_wrapping(row, threshold=threshold, return_mask=True)
                arrs.append(wrapped)
                masks.append(mask)
            else:
                arrs.append(omit_wrapping(row, threshold=threshold))
        # Pad to max length
        maxlen = max(len(a) for a in arrs)
        arr_out = np.full((len(arrs), maxlen), np.nan)
        if return_mask:
            mask_out = np.full((len(arrs), maxlen), False)
        for i, a in enumerate(arrs):
            arr_out[i, :len(a)] = a
            if return_mask:
                mask_out[i, :len(a)] = masks[i]
        if return_mask:
            return arr_out, mask_out
        return arr_out
    else:
        raise ValueError('arr must be 1D or 2D')


def bootstrap_ci(data, stat_func, confidence=0.84, n_boot=1000, axis=0, return_bootstrap=False):
    """
    Compute bootstrap confidence intervals for a statistic.

    Parameters
    ----------
    data : ndarray
        Input data array.
    stat_func : callable
        Function to compute the statistic (e.g., np.nanmean).
    confidence : float, optional
        Confidence level (default is 0.84).
    n_boot : int, optional
        Number of bootstrap samples (default is 1000).
    axis : int, optional
        Axis along which to compute the statistic (default is 0).
    return_bootstrap : bool, optional
        If True, also return the array of bootstrap statistics.

    Returns
    -------
    low : ndarray
        Lower bound(s) of the confidence interval.
    high : ndarray
        Upper bound(s) of the confidence interval.
    boot_stats : ndarray, optional
        Array of bootstrap statistics (only if return_bootstrap is True).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((100, 50))
    >>> low, high = bootstrap_ci(data, np.nanmean, confidence=0.84, n_boot=200)
    >>> low.shape
    (50,)
    >>> bool(np.all(low < high))
    True
    """
    data = np.asarray(data)
    rng = np.random.default_rng()
    n = data.shape[axis]
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = np.take(data, idx, axis=axis)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            stat = stat_func(sample, axis=axis)
        boot_stats.append(stat)
    boot_stats = np.stack(boot_stats, axis=0)
    alpha = (1 - confidence) / 2
    low = np.nanpercentile(boot_stats, 100 * alpha, axis=0)
    high = np.nanpercentile(boot_stats, 100 * (1 - alpha), axis=0)
    if return_bootstrap:
        return low, high, boot_stats
    return low, high


def _hide_spine(ax, side):
    """Unconditionally hide a spine line, its ticks, and tick labels.

    Unlike ``sbn.despine(trim=True)``, this sets visibility directly on the
    spine ``Line2D`` object and updates only tick parameters.  It is therefore
    reliable on log-scaled axes (where ``trim=True`` can position the clipped
    spine incorrectly) and on axes that share a scale with a neighbour (where
    seaborn's internal tick-position inspection can see the wrong state).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    side : {'top', 'bottom', 'left', 'right'}
    """
    ax.spines[side].set_visible(False)
    axis = 'x' if side in ('top', 'bottom') else 'y'
    ax.tick_params(axis=axis, which='both', **{side: False, f'label{side}': False})


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# compare_means shared layout constants (used by plot_compare_means and
# the post-format spine restoration pass in TrackingExperiment.plot()).
# ---------------------------------------------------------------------------
_CM_SPINE_PTS = 18   # points the bottom spine is offset outward (downward)
_CM_TEXT_PTS  = 15   # reference offset (pts) used for annotation anchor (decoupled from spine)
_CM_ANN_PTS   = 3    # gap (pts) between spine and top of mean annotation
_CM_TICK_PAD  = 6    # x tick-label pad from spine (points)

# Phase 3 — Standalone stateless plot-type functions
# Each function takes (ax, xs, ys, color, **kwargs) and returns artist(s).
# They contain no experiment-level logic and are fully testable with synthetic
# data.  The Phase 5 TrackingExperiment.plot() method will call these.
# ---------------------------------------------------------------------------

def plot_line(ax, xs, ys, color, summary_func=None, ci=False,
              confidence=0.84, n_boot=1000, trace_color='gray',
              split_by_sign=False, mean_bins=None, ylim=None,
              saccade_durations=None, saccade_spans=None, **kw):
    """Plot individual traces with an optional colored mean and CI.

    Parameters
    ----------
    ax : matplotlib Axes
    xs : ndarray, shape (n_traces, n_frames)
        X values for each trace.
    ys : ndarray, shape (n_traces, n_frames)
        Y values for each trace.
    color : color spec
        Color for the mean line and CI shading.
    summary_func : callable or None
        If provided, compute and overlay a summary trace (e.g. np.nanmean).
        Ignored when ``mean_bins`` is set.
    ci : bool
        If True (and summary_func is set and mean_bins is None), shade a
        bootstrap CI around the mean.
    confidence : float
        Confidence level for the CI (default 0.84).
    n_boot : int
        Bootstrap resamples for the CI (default 1000).
    trace_color : color spec, default 'gray'
        Color for the individual background traces.
    split_by_sign : bool, default False
        If True, draw separate summary lines for positive-amplitude
        (last non-NaN x ≥ 0) and negative-amplitude traces.  Positive
        group: solid line; negative group: dotted line.
        Has no effect when both ``summary_func`` and ``mean_bins`` are None.
    mean_bins : int or None, default None
        If None, the summary uses ``summary_func(xs, axis=0)``.
        If an int, the y-axis is divided into that many equal bins and the
        summary per bin is ``np.nanmedian`` — useful for variable-length
        NaN-padded traces where a column-wise mean is uneven near the edges.
    saccade_durations : array-like of shape (n_traces,) or None, default None
        If provided, the segment of each trace where
        ``0 ≤ ys[i] ≤ saccade_durations[i]`` is overlaid in ``color`` to
        highlight the saccade body.  Assumes ``ys`` is a time axis with
        t = 0 marking saccade start (e.g. ``Saccade.time``).
    **kw
        Extra kwargs forwarded to ax.plot for the individual traces.
        Use ``alpha`` (default 0.5) and ``lw`` (default 1.0) to control
        trace opacity and line width.

    Returns
    -------
    artists : list
        All artists added to ax.
    """
    artists = []
    alpha = kw.pop('alpha', 0.5)
    lw = kw.pop('lw', 1.0)
    lines = ax.plot(xs.T, ys.T, color=trace_color, lw=lw, alpha=alpha, **kw)
    artists.extend(lines)

    # Highlight the saccade body per trace.
    # saccade_spans: list of (y_start, y_stop) per trace (reference-aware).
    # saccade_durations: legacy — list of durations assuming y=0 at saccade start.
    _spans = None
    if saccade_spans is not None:
        _spans = saccade_spans
    elif saccade_durations is not None:
        dur_arr = np.asarray(saccade_durations)
        _spans = [(0.0, float(d)) for d in dur_arr]
    if _spans is not None:
        for i in range(min(len(_spans), xs.shape[0])):
            y0, y1 = float(_spans[i][0]), float(_spans[i][1])
            if np.isnan(y0) or np.isnan(y1):
                continue
            y_lo, y_hi = min(y0, y1), max(y0, y1)
            xi, yi = xs[i], ys[i]
            in_saccade = (yi >= y_lo) & (yi <= y_hi) & ~np.isnan(xi) & ~np.isnan(yi)
            if in_saccade.any():
                seg_x = np.where(in_saccade, xi, np.nan)
                seg_y = np.where(in_saccade, yi, np.nan)
                ln, = ax.plot(seg_x, seg_y, color=color, lw=lw * 2,
                              alpha=0.5, zorder=2)
                artists.append(ln)

    if summary_func is None and mean_bins is None:
        return artists

    def _bin_mean(xs_sub, ys_sub):
        """Bin-average xs along the y axis using np.nanmedian."""
        if ylim is not None:
            y_min, y_max = min(ylim), max(ylim)
        else:
            y_min = np.nanmin(ys_sub)
            y_max = np.nanmax(ys_sub)
        edges = np.linspace(y_min, y_max, mean_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        mx = np.full(mean_bins, np.nan)
        for b_i in range(mean_bins):
            # Use <= for the last bin so the maximum y value is included.
            hi_op = np.less_equal if b_i == mean_bins - 1 else np.less
            in_b = (ys_sub >= edges[b_i]) & hi_op(ys_sub, edges[b_i + 1])
            vals = xs_sub[in_b]
            valid = vals[~np.isnan(vals)]
            if valid.size > 0:
                mx[b_i] = np.nanmedian(valid)
        return mx, centers

    def _summary_xy(xs_sub, ys_sub):
        """Return (mean_x, y_ref) for the summary line."""
        if mean_bins is not None:
            return _bin_mean(xs_sub, ys_sub)
        mx = summary_func(xs_sub, axis=0)
        y_ref = ys_sub[np.isnan(ys_sub).sum(1).argmin()]
        return mx, y_ref

    def _draw_summary(xs_sub, ys_sub, ls='-'):
        if xs_sub.shape[0] == 0:
            return
        mean_x, y_ref = _summary_xy(xs_sub, ys_sub)
        white, = ax.plot(mean_x, y_ref, color='w', lw=4, zorder=4)
        ml, = ax.plot(mean_x, y_ref, color=color, zorder=5, linestyle=ls)
        artists.extend([white, ml])
        if ci and summary_func is not None and mean_bins is None:
            lows, highs = bootstrap_ci(xs_sub, summary_func,
                                       confidence=confidence, n_boot=n_boot)
            band = ax.fill_betweenx(y_ref, lows, highs,
                                    color=color, alpha=0.3,
                                    zorder=3, linewidth=0)
            artists.append(band)

    if split_by_sign:
        last_valid = np.array([
            xs[i][~np.isnan(xs[i])][-1] if np.any(~np.isnan(xs[i])) else 0.0
            for i in range(xs.shape[0])
        ])
        for mask, ls in [(last_valid >= 0, '-'), (last_valid < 0, ':')]:
            if mask.sum() == 0:
                continue
            _draw_summary(xs[mask], ys[mask], ls=ls)
    else:
        _draw_summary(xs, ys)
    return artists


def plot_hist2d(ax, xs, ys, color, bins=100, density=False, n_contours=0,
                contour_alpha=0.3, return_summary=False, **kw):
    """Plot a 2-D histogram with a white-to-color linear colormap.

    Parameters
    ----------
    ax : matplotlib Axes
    xs : ndarray, shape (n,) — already flattened, NaNs removed by caller
    ys : ndarray, shape (n,)
    color : color spec
        Defines the saturated end of the white→color colormap.
    bins : int or [int, int]
        Passed to np.histogram2d.
    density : bool
        If True, normalise the histogram to a probability density.
    n_contours : int, default 0
        Number of KDE HDR contour levels to draw on margin axes.  These
        are *not* drawn on the main histogram panel.  Pass via
        ``plot_kwargs={'n_contours': N}`` in ``TrackingExperiment.plot()``.
    contour_alpha : float, default 0.3
        Alpha used to fill the HDR contour regions on margin axes.  Set to
        0 to draw lines only (no fill).
    return_summary : bool, default False
        If True, return ``(mesh, summary_dict)`` instead of just ``mesh``.
        ``summary_dict`` contains ``'hist'``, ``'xedges'``, ``'yedges'``,
        ``'n_contours'``, ``'contour_alpha'``, ``'xs'``, and ``'ys'``.
    **kw
        Extra kwargs forwarded to ax.pcolormesh.  Pass ``vmax`` to fix the
        colour scale; pass ``cbar=False`` to suppress the colorbar.

    Returns
    -------
    mesh : QuadMesh
        Or ``(mesh, summary_dict)`` when ``return_summary=True``.
    """
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        '', [(1, 1, 1), color])
    hist, xedges, yedges = np.histogram2d(xs, ys, bins=bins, density=density)
    vmax = kw.pop('vmax', hist.max())
    cbar = kw.pop('cbar', True)
    mesh = ax.pcolormesh(xedges, yedges, hist.T,
                         cmap=cmap, vmin=0, vmax=vmax, **kw)
    if cbar:
        plt.colorbar(mesh, ax=ax)
    if return_summary:
        sd = {
            'hist': hist,
            'xedges': xedges,
            'yedges': yedges,
            'n_contours': n_contours,
            'contour_alpha': contour_alpha,
            'xs': xs,
            'ys': ys,
        }
        return mesh, sd
    return mesh


def _add_polar_ring(ax, has_circ_hist=False, show_labels=True):
    """Overlay a polar direction ring on a trajectory2d Cartesian axis.

    Replaces the Cartesian spines with 8 radial spokes (every 45°) and a
    thin circle at a radius that clears the data / circ_hist ring.

    Parameters
    ----------
    ax : matplotlib Axes
    has_circ_hist : bool
        When True the ring is drawn at R=1.3 (beyond the circ_hist wedges
        at ~1.27).  When False R=1.05 (just above the unit trajectory).
    show_labels : bool, default True
        When False, spoke angle labels are suppressed entirely (used for
        all subplots except the top-left one).
    """
    R = 1.3 if has_circ_hist else 1.05

    # Match visual weight to the surrounding figure aesthetics.
    tick_fs = plt.rcParams.get('xtick.labelsize',
                               plt.rcParams.get('font.size', 8))
    # Spine line-width from rcParams; spokes are half that.
    spine_lw = plt.rcParams.get('axes.linewidth', 0.8)
    spoke_lw = spine_lw * 0.5

    # ±180 label is wider than other labels, so give it more clearance.
    label_r_default = R * 1.11
    label_r_180 = R * 1.17
    pad = label_r_180 * 1.12  # axis limit padding

    # Hide all Cartesian spines and ticks.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Draw 8 radial spokes (half the spine line-width).
    spoke_angles = np.linspace(0, 2 * np.pi, 9)[:-1]  # 0, 45, …, 315 °
    for ang in spoke_angles:
        ax.plot([0, R * np.cos(ang)], [0, R * np.sin(ang)],
                color='lightgray', lw=spoke_lw, zorder=0)

    # Draw the ring circle (spine line-width).
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R * np.cos(theta), R * np.sin(theta),
            color='lightgray', lw=spine_lw, zorder=0)

    # Only draw labels when the axis is physically large enough that they
    # won't collide with the ring.  Estimate axis width in inches; skip
    # labels below ~0.8 in so that small-scale / margin axes stay clean.
    try:
        fig = ax.get_figure()
        # ax.get_figure() may return a SubFigure; walk up to the root Figure
        # so that get_size_inches() is available.
        root_fig = fig
        while hasattr(root_fig, 'figure') and root_fig.figure is not root_fig:
            root_fig = root_fig.figure
        pos = ax.get_position()          # fraction of (sub)figure
        fig_w, fig_h = root_fig.get_size_inches()
        ax_w_in = pos.width * fig_w
    except Exception:
        ax_w_in = 1.0

    if show_labels and ax_w_in >= 2.0:
        for ang in spoke_angles:
            deg = int(round(np.degrees(ang))) % 360
            if deg > 180:
                deg -= 360
            # Only label the four cardinal directions (0, 90, ±180, -90).
            if deg % 90 != 0:
                continue
            is_180 = abs(deg) == 180
            lbl = '±180°' if is_180 else f'{deg}°'
            lr = label_r_180 if is_180 else label_r_default
            ax.text(lr * np.cos(ang), lr * np.sin(ang),
                    lbl, ha='center', va='center',
                    fontsize=tick_fs, color='black', zorder=1,
                    rotation=deg - 90 + (180 if deg < 0 else 0))

    # Expand axis limits to show the ring + labels (or just ring if no labels).
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect('equal', adjustable='box')


def plot_trajectory2d(ax, xs, ys, color, trace_color='k', **kw):
    """Plot cumulative 2-D heading trajectories and optional overlays.

    Parameters
    ----------
    ax : matplotlib Axes
    xs : ndarray, shape (n_traces, n_frames)
        Angular heading in radians for each trace.
    ys : ndarray, shape (n_traces, n_frames)
        Ignored (kept for API consistency); pass ``None`` if unavailable.
    color : color spec
        Used for optional overlays (circle, contour ellipse, circ_hist, mean).
    trace_color : color spec, default 'k'
        Color for the individual trajectory lines and endpoint scatter.
    **kw
        Optional flags and settings:
        ``circle`` — draw per-fly radius circles.
        ``contour`` — draw a bootstrap ellipse for the mean endpoint.
        ``circ_hist`` — draw a Wedge ring histogram of endpoint angles at
            radius 1.01–1.26, plus a bootstrap CI arc and mean-angle dot.
        ``bins`` — number of bins (or bin edges array) for ``circ_hist``
            (default 100).
        ``mean_line`` — overlay the mean trajectory.
        ``confidence`` — CI level for contour/circ_hist (default 0.84).
        ``alpha`` — opacity of individual trajectory lines (default 0.25).
        ``lw`` — line width of individual trajectory lines (default 0.5).

    Returns
    -------
    artists : list
    """
    from matplotlib.patches import Arc, Wedge
    confidence = kw.pop('confidence', 0.84)
    alpha = kw.pop('alpha', 0.25)
    lw = kw.pop('lw', 0.5)
    return_summary = kw.pop('return_summary', False)
    artists = []

    ax.set_aspect('equal', adjustable='box')

    d_vectors = np.array([np.cos(xs), np.sin(xs)]).transpose(1, 0, 2)
    d_vectors[np.isnan(d_vectors)] = 0
    trajectory = np.cumsum(d_vectors, axis=-1)
    trajectory /= trajectory.shape[-1]
    last_pos = trajectory[..., -1]
    # get the radii (vector strength) for all points in the trajectories
    radii = np.linalg.norm(trajectory, axis=1)
    last_radii = radii[..., -1]

    summary_dict = {'last_pos': last_pos, 'radii': radii, 'trajectory': trajectory}

    lines = ax.plot(trajectory[:, 0].T, trajectory[:, 1].T,
                    lw=lw, color=trace_color, alpha=alpha, zorder=2)
    scatter = ax.scatter(last_pos[..., 0], last_pos[..., 1],
                         color=trace_color, marker='.', s=0.5, zorder=2)
    artists.extend(lines)
    artists.append(scatter)

    if kw.get('circle'):
        mean_radius = float(np.nanmean(last_radii))
        for radius in last_radii:
            c = plt.Circle((0, 0), radius=radius,
                           color=color, alpha=0.25, fill=False, lw=0.5)
            ax.add_artist(c)
            artists.append(c)
        # Mean circle: thick outline on top.
        mc = plt.Circle((0, 0), radius=mean_radius,
                        color=color, fill=False, lw=2, zorder=4)
        ax.add_artist(mc)
        artists.append(mc)
        summary_dict['mean_radius'] = mean_radius

    if kw.get('contour'):
        cmap_overlay = matplotlib.colors.LinearSegmentedColormap.from_list(
            '', [(1, 1, 1), color])
        inds = np.arange(len(last_pos))
        rand_inds = np.random.choice(inds, size=(10000, len(inds)), replace=True)
        boot_means = np.nanmean(last_pos[rand_inds], axis=1)
        mean = np.mean(boot_means, axis=0)
        cov = np.cov(boot_means, rowvar=False)
        chi2_val = scipy.stats.chi2.ppf(confidence, 2)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        idx = eigenvals.argsort()[::-1]
        eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]
        a = np.sqrt(chi2_val * eigenvals[0])
        b = np.sqrt(chi2_val * eigenvals[1])
        angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        ellipse = matplotlib.patches.Ellipse(
            mean, 2 * a, 2 * b, angle=np.degrees(angle),
            color=color, alpha=0.25, fill=True, lw=0.5, zorder=2)
        ax.add_artist(ellipse)
        artists.append(ellipse)
        summary_dict.update({
            'ellipse_mean': mean,
            'ellipse_a': a,
            'ellipse_b': b,
            'ellipse_angle': angle,
        })

    if kw.get('circ_hist'):
        angles = np.arctan2(last_pos[..., 1], last_pos[..., 0])
        hist_bins = kw.get('bins', 100)
        if isinstance(hist_bins, int):
            hist_bins = np.linspace(-np.pi, np.pi, hist_bins + 1)
        # Use pre-computed, cross-panel-normalised histogram if provided by plot().
        _precomp_hist = kw.get('_precomputed_hist', None)
        if _precomp_hist is not None:
            hist = _precomp_hist
            hist_bins = kw.get('_precomputed_bins', hist_bins)
        else:
            hist, hist_bins = np.histogram(angles, bins=hist_bins, density=False)
        # draw ring histogram as Wedge tiles
        ring_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', [(1, 1, 1), color])
        max_val = kw.get('_global_vmax', hist.max() if hist.max() > 0 else 1)
        cvals = ring_cmap(hist / max_val)
        for start, stop, cval in zip(hist_bins[:-1], hist_bins[1:], cvals):
            w = Wedge((0, 0), 1.01, start * 180 / np.pi, stop * 180 / np.pi,
                      facecolor=cval, edgecolor=cval, alpha=1, lw=0.25, width=-0.25)
            ax.add_artist(w)
            artists.append(w)
        # adjust axis limits to show the ring (radius ~1.27)
        ax.set_xlim(-1.27, 1.27)
        ax.set_ylim(-1.27, 1.27)
        # bootstrap CI for the mean angle
        inds = np.arange(len(last_pos))
        rand_inds = np.random.choice(inds, size=(10000, len(inds)), replace=True)
        boot_means = np.nanmean(last_pos[rand_inds], axis=1)
        boot_angles = np.arctan2(boot_means[..., 1], boot_means[..., 0])
        mean_angle = np.arctan2(boot_means[..., 1].mean(), boot_means[..., 0].mean())
        lb_diff, ub_diff = np.percentile(
            boot_angles - mean_angle,
            [100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2])
        lb, ub = mean_angle + lb_diff, mean_angle + ub_diff
        radius = 1.125
        for arc_color, lw, zorder in [('w', 4, 4), (color, 2, 5)]:
            arc = Arc((0, 0), width=2 * radius, height=2 * radius, angle=0,
                      theta1=np.degrees(lb), theta2=np.degrees(ub),
                      color=arc_color, lw=lw, zorder=zorder, capstyle='round')
            ax.add_artist(arc)
            artists.append(arc)
        for sc_color, s, z in [('w', 40, 4), (color, 20, 5)]:
            sc = ax.scatter(radius * np.cos(mean_angle), radius * np.sin(mean_angle),
                            color=sc_color, marker='o', s=s, zorder=z)
            artists.append(sc)
        # Store circ_hist results in summary_dict for optional margin drawing.
        summary_dict.update({
            'hist': hist,
            'bin_edges': hist_bins,
            'mean_angle': mean_angle,
            'lb': lb,
            'ub': ub,
        })

    if kw.get('mean_line'):
        mean_traj = np.nanmean(trajectory, axis=0)
        white, = ax.plot(mean_traj[0], mean_traj[1], color='w', lw=4, zorder=4)
        mean_l, = ax.plot(mean_traj[0], mean_traj[1], color=color, lw=1, zorder=5)
        sc_w = ax.scatter(mean_traj[0, -1], mean_traj[1, -1],
                          color='w', marker='o', s=10, zorder=4)
        sc_c = ax.scatter(mean_traj[0, -1], mean_traj[1, -1],
                          color=color, marker='o', s=5, zorder=5)
        artists.extend([white, mean_l, sc_w, sc_c])

    if return_summary:
        # Compute circular mean heading and vector strength from endpoint
        # positions, plus bootstrap CIs.
        _angles_ep = np.arctan2(last_pos[..., 1], last_pos[..., 0])
        _sin_m = float(np.nanmean(np.sin(_angles_ep)))
        _cos_m = float(np.nanmean(np.cos(_angles_ep)))
        _circ_mean = float(np.arctan2(_sin_m, _cos_m))
        _vector_strength = float(np.sqrt(_sin_m ** 2 + _cos_m ** 2))
        _n_ep = len(last_pos)
        if _n_ep > 1:
            _bi = np.random.randint(0, _n_ep, (10000, _n_ep))
            _boot_lp = last_pos[_bi]
            _boot_sin = np.nanmean(
                np.sin(np.arctan2(_boot_lp[..., 1], _boot_lp[..., 0])),
                axis=1)
            _boot_cos = np.nanmean(
                np.cos(np.arctan2(_boot_lp[..., 1], _boot_lp[..., 0])),
                axis=1)
            _boot_cm = np.arctan2(_boot_sin, _boot_cos)
            _boot_vs = np.sqrt(_boot_sin ** 2 + _boot_cos ** 2)
            _conf = kw.get('confidence', confidence if 'confidence' in dir()
                           else 0.84)
            _cm_lo, _cm_hi = np.percentile(
                _boot_cm,
                [100 * (1 - _conf) / 2, 100 * (1 + _conf) / 2])
            _vs_lo, _vs_hi = np.percentile(
                _boot_vs,
                [100 * (1 - _conf) / 2, 100 * (1 + _conf) / 2])
        else:
            _cm_lo = _cm_hi = _circ_mean
            _vs_lo = _vs_hi = _vector_strength
        summary_dict.update({
            'circ_mean': _circ_mean,
            'circ_mean_ci_low': float(_cm_lo),
            'circ_mean_ci_high': float(_cm_hi),
            'vector_strength': _vector_strength,
            'vector_strength_ci_low': float(_vs_lo),
            'vector_strength_ci_high': float(_vs_hi),
        })
        return artists, summary_dict
    return artists


def plot_histogram(ax, xs, bins, color, probability=False, summary_func=None,
                   **kw):
    """Plot a 1-D histogram bar plot with an optional summary line.

    Parameters
    ----------
    ax : matplotlib Axes
    xs : ndarray, shape (n,) — flattened; NaNs silently ignored
    bins : int or array-like
        Bin edges or count, passed to ax.hist.
    color : color spec
        Bar face color.
    probability : bool
        If True, normalise to probability (density=True equivalent).
    summary_func : callable or None
        If provided, draw a vertical line at summary_func(xs[~nan]).
    **kw
        Extra kwargs forwarded to ax.hist (e.g. histtype, alpha).

    Returns
    -------
    artists : tuple  (n, bins, patches[, vline])
    """
    valid = xs[~np.isnan(xs)]
    n, bin_edges, patches = ax.hist(
        valid, bins=bins, color=color,
        density=probability, **kw)
    if summary_func is not None:
        val = summary_func(valid)
        ax.axvline(val, color='w', lw=6, zorder=4)
        vline = ax.axvline(val, color=color, lw=2, zorder=5)
        return n, bin_edges, patches, vline
    return n, bin_edges, patches


def plot_scatter(ax, xs, ys, color, jitter_std=0.0, correlation=False,
                 n_boot=10000, marker_color=None, **kw):
    """Scatter plot with optional jitter and Mardia's circular-linear correlation.

    Parameters
    ----------
    ax : matplotlib Axes
    xs : ndarray, shape (n,)
        X values (circular, in radians, for correlation).
    ys : ndarray, shape (n,)
        Y values.
    color : color spec
        Summary/overlay color (e.g. a future trend line). If ``marker_color``
        is not set, also used for the scatter markers.
    jitter_std : float
        Standard deviation of Gaussian jitter applied to xs.
    correlation : bool
        If True, annotate the axes title with Mardia's r and a bootstrap p-value.
    n_boot : int
        Bootstrap resamples for the correlation p-value (default 10000).
    marker_color : color spec or None, default None
        Color for scatter markers. Overrides ``color`` for the data points
        when set; if None, ``color`` is used.
    **kw
        Extra kwargs forwarded to ax.scatter (e.g. s, alpha, edgecolors).

    Returns
    -------
    sc : PathCollection
    """
    plot_xs = xs.copy()
    if jitter_std > 0:
        plot_xs = plot_xs + np.random.normal(0, jitter_std, size=plot_xs.shape)
    alpha = kw.pop('alpha', 0.25)
    s = kw.pop('s', 1)
    edgecolors = kw.pop('edgecolors', 'none')
    point_color = marker_color if marker_color is not None else color
    sc = ax.scatter(plot_xs, ys, color=point_color,
                    alpha=alpha, s=s, edgecolors=edgecolors, **kw)
    if correlation:
        valid = np.isfinite(xs) & np.isfinite(ys)
        if valid.sum() > 2:
            r = mardia_circ_lin(xs[valid], ys[valid])
            rand_inds = np.random.randint(0, valid.sum(), (n_boot, valid.sum()))
            rand_r = np.array([
                mardia_circ_lin(xs[valid][idx], ys[valid][idx])
                for idx in rand_inds])
            lower, _, upper = np.percentile(rand_r, [2.5, 50, 97.5])
            mid = np.median(rand_r)
            pval = (np.sum(rand_r <= 0) / n_boot
                    if mid > 0 else np.sum(rand_r >= 0) / n_boot)
            ax.set_title(
                f"r={r:.2f} ({lower:.2f}, {upper:.2f}) {sigAsterisk(pval)}",
                fontsize=8)
    return sc


def plot_pdf(ax, xs, bins, color, ci=False, confidence=0.84, n_boot=1000,
             show_traces=True, summary_func=None, **kw):
    """Plot per-fly normalized histograms (PDFs) with a mean line and CI.

    Parameters
    ----------
    ax : matplotlib Axes
    xs : ndarray, shape (n_flies, n_frames)
        Raw data; NaNs are silently dropped per fly.
    bins : int or array-like
        Bin edges or count.  When an int, edges are computed from the union
        of all non-NaN values so that every fly uses the same bins.
    color : color spec
        Color for mean line, CI fill, and (optionally) individual traces.
    ci : bool
        If True, shade a bootstrap CI around the mean.
    confidence : float
        Confidence level for the CI (default 0.84).
    n_boot : int
        Bootstrap resamples (default 1000).
    show_traces : bool
        If True (default), draw a thin line for each fly's PDF.  Set to
        False in margin contexts where only the mean + CI are needed.
    summary_func : callable or None
        Applied across flies to produce the mean line (default np.nanmean).
    **kw
        ``alpha`` (default 0.2) — opacity of individual fly traces.
        ``lw``    (default 0.5) — line width of individual fly traces.

    Returns
    -------
    artists : list
    """
    if summary_func is None:
        summary_func = np.nanmean

    alpha = kw.pop('alpha', 0.2)
    lw = kw.pop('lw', 0.5)

    # Compute global bin edges from all non-NaN data so every fly uses the
    # same bins (required for per-fly PDFs to be meaningfully averaged).
    all_valid = xs[~np.isnan(xs)] if xs.ndim > 1 else xs[~np.isnan(xs)]
    if all_valid.size == 0:
        return []
    if isinstance(bins, int):
        _, edges = np.histogram(all_valid, bins=bins)
    else:
        edges = np.asarray(bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Per-fly normalized histograms.
    hists = []
    for row in (xs if xs.ndim == 2 else xs[np.newaxis]):
        valid = row[~np.isnan(row)]
        if valid.size == 0:
            continue
        h, _ = np.histogram(valid, bins=edges)
        h = h.astype(float)
        s = h.sum()
        if s > 0:
            h /= s
        hists.append(h)

    if not hists:
        return []
    hists = np.array(hists)

    artists = []

    # Thin trace per fly.
    if show_traces and hists.shape[0] > 1:
        for h in hists:
            (line,) = ax.plot(bin_centers, h, color=color, lw=lw, alpha=alpha)
            artists.append(line)

    # Mean (thick line).
    mean_h = summary_func(hists, axis=0)
    (mean_line,) = ax.plot(bin_centers, mean_h, color=color, lw=2, zorder=4)
    artists.append(mean_line)

    # Bootstrap CI fill.
    if ci and hists.shape[0] > 1:
        rand_inds = np.random.randint(0, hists.shape[0], (n_boot, hists.shape[0]))
        boot_means = np.nanmean(hists[rand_inds], axis=1)
        lo, hi = np.percentile(boot_means,
                               [100 * (1 - confidence) / 2,
                                100 * (1 + confidence) / 2],
                               axis=0)
        fill = ax.fill_between(bin_centers, lo, hi,
                               color=color, alpha=0.3,
                               edgecolor='none', zorder=3)
        artists.append(fill)

    return artists


# ---------------------------------------------------------------------------
# compare_means helpers
# ---------------------------------------------------------------------------

def plot_diff_brackets(label, x1, x2, y1, y2, y_label, col='k',
                       vert=False, ax=None, lw=None, size='medium', **plot_kwargs):
    """Draw a significance bracket connecting two x-positions.

    Parameters
    ----------
    label : str
        Text drawn above (or beside) the bracket.
    x1, x2 : float
        x-positions of the two groups.
    y1, y2 : float
        y-positions of the vertical ticks at each end of the bracket.
    y_label : float
        y-position of the horizontal connecting bar.
    col : color spec, default 'k'
    vert : bool, default False
        If True, draw a vertical bracket.
    ax : matplotlib Axes or None
        Defaults to ``plt.gca()``.
    lw : float, default 1
    size : str or float, default 'medium'
        Font size for the label.
    **plot_kwargs
        Forwarded to ``ax.plot``.
    """
    if ax is None:
        ax = plt.gca()
    if lw is None:
        lw = ax.spines['bottom'].get_linewidth()
    if vert:
        ax.plot([y_label, y_label], [x1, x2],
                color=col, clip_on=False, lw=lw, **plot_kwargs)
        ax.text(y_label, (x1 + x2) * 0.5, label,
                ha='left', va='center', color=col,
                rotation='vertical', size=size)
    else:
        ax.plot([x1, x2], [y_label, y_label],
                color=col, clip_on=False, lw=lw, **plot_kwargs)
        ax.text((x1 + x2) * 0.5, y_label, label,
                ha='center', va='bottom', color=col,
                rotation='horizontal', size=size)


def plot_compare_means(ax, vals1, vals2, color,
                       labels=('Group 1', 'Group 2'),
                       paired_ids1=None, paired_ids2=None,
                       marker_color=None, marker_colors=None,
                       jitter_std=0.05, confidence=0.95, n_boot=10000,
                       show_ns=False, ylim=None, **kw):
    """Compare means of two groups with a partially-paired bootstrap test.

    Individual data points are drawn as jitter plots.  Mean ± bootstrap CI
    is overlaid per group.  A significance bracket is drawn above the data.
    Within-subject pairs are connected by lines when subject IDs overlap.

    The **partially-paired bootstrap** is used unconditionally: shared
    subjects (IDs in both groups) are resampled with the same index on each
    iteration (preserving within-subject covariance), while subjects unique
    to one group are resampled independently.  This converges to a standard
    paired-differences bootstrap when all subjects are shared and to a
    two-sample bootstrap when no subjects are shared.

    Parameters
    ----------
    ax : matplotlib Axes
    vals1, vals2 : array-like, shape (n,)
        One value per subject in each group.  NaNs are silently ignored.
    color : color spec
        Controls mean markers, CI bars, and significance bracket.
    labels : sequence of two str, default ('Group 1', 'Group 2')
        x-tick labels for the two groups.
    paired_ids1, paired_ids2 : list or None
        Subject IDs parallel to ``vals1`` / ``vals2``.  Subjects whose ID
        appears in both lists receive connecting lines and are resampled
        together in the bootstrap.  Pass ``None`` to treat all subjects as
        independent.
    marker_color : color spec or None
        Color for individual jitter points and connecting lines.  If
        ``None``, defaults to ``'gray'``.
    jitter_std : float, default 0.05
        Std of Gaussian jitter added to x-positions.
    confidence : float, default 0.95
        Confidence level for the bootstrap CI on each group mean and the
        bracket CI on the difference.
    n_boot : int, default 10000
        Bootstrap resamples.
    show_ns : bool, default True
        If True, draw the bracket even when the test is not significant.
    **kw
        ``s``       — marker size (default 10).
        ``alpha``   — marker and line opacity (default 0.4).
        ``bracket_y`` — y-position of the bracket bar; auto-detected when
                        absent.

    Returns
    -------
    summary_dict : dict
        Keys: ``mean1``, ``mean2``, ``diff``, ``ci_low``, ``ci_high``,
        ``pval``, ``symbol``, ``n_paired``, ``n_group1_only``,
        ``n_group2_only``, ``vals1``, ``vals2``, ``ids1``, ``ids2``.
    """
    vals1 = np.asarray(vals1, dtype=float)
    vals2 = np.asarray(vals2, dtype=float)
    _scatter_color = marker_color if marker_color is not None else color
    s = kw.pop('s', 10)
    alpha = kw.pop('alpha', 0.4)
    clip_on = kw.pop('clip_on', True)
    bracket_y_override = kw.pop('bracket_y', None)

    # ------------------------------------------------------------------ #
    # Partition subjects into shared / group-only                         #
    # ------------------------------------------------------------------ #
    if paired_ids1 is not None and paired_ids2 is not None:
        ids1 = list(paired_ids1)
        ids2 = list(paired_ids2)
    else:
        ids1 = list(range(len(vals1)))
        ids2 = list(range(len(vals2)))

    set2 = set(ids2)
    shared_ids = list(dict.fromkeys(sid for sid in ids1 if sid in set2))
    id1_to_idx = {sid: i for i, sid in enumerate(ids1)}
    id2_to_idx = {sid: i for i, sid in enumerate(ids2)}

    shared_idx1 = np.array([id1_to_idx[sid] for sid in shared_ids], dtype=int)
    shared_idx2 = np.array([id2_to_idx[sid] for sid in shared_ids], dtype=int)
    g1_only_idx = np.array(
        [id1_to_idx[sid] for sid in ids1 if sid not in set2], dtype=int)
    g2_only_idx = np.array(
        [id2_to_idx[sid] for sid in ids2 if sid not in set(ids1)], dtype=int)

    shared_v1 = vals1[shared_idx1] if len(shared_idx1) else np.array([])
    shared_v2 = vals2[shared_idx2] if len(shared_idx2) else np.array([])
    g1_v = vals1[g1_only_idx] if len(g1_only_idx) else np.array([])
    g2_v = vals2[g2_only_idx] if len(g2_only_idx) else np.array([])

    # Drop NaN-pairs for shared subjects (need both values to be finite).
    if len(shared_v1):
        shared_valid = np.isfinite(shared_v1) & np.isfinite(shared_v2)
        shared_v1 = shared_v1[shared_valid]
        shared_v2 = shared_v2[shared_valid]
        # Also update shared_ids list to match filtered arrays.
        shared_ids = [sid for sid, ok in zip(shared_ids, shared_valid) if ok]
    else:
        shared_valid = np.array([], dtype=bool)
    g1_finite = np.isfinite(g1_v)
    g2_finite = np.isfinite(g2_v)
    g1_v = g1_v[g1_finite]
    g2_v = g2_v[g2_finite]

    # Per-point marker colours (used when margin mixes data from different cmaps).
    _per_point = False
    _all_mc1 = _all_mc2 = _shared_mc1 = None
    if marker_colors is not None:
        _mc1_raw = np.asarray(marker_colors[0], dtype=float)  # (N1, 3)
        _mc2_raw = np.asarray(marker_colors[1], dtype=float)  # (N2, 3)
        _nc = _mc1_raw.shape[1] if _mc1_raw.ndim == 2 else 3
        _empty_mc = np.zeros((0, _nc))
        _sh_mc1 = (_mc1_raw[shared_idx1][shared_valid]
                   if len(shared_idx1) else _empty_mc)
        _sh_mc2 = (_mc2_raw[shared_idx2][shared_valid]
                   if len(shared_idx2) else _empty_mc)
        _g1_mc = (_mc1_raw[g1_only_idx][g1_finite]
                  if len(g1_only_idx) else _empty_mc)
        _g2_mc = (_mc2_raw[g2_only_idx][g2_finite]
                  if len(g2_only_idx) else _empty_mc)
        _all_mc1 = (np.concatenate([_sh_mc1, _g1_mc])
                    if (len(_sh_mc1) + len(_g1_mc)) else _empty_mc)
        _all_mc2 = (np.concatenate([_sh_mc2, _g2_mc])
                    if (len(_sh_mc2) + len(_g2_mc)) else _empty_mc)
        _shared_mc1 = _sh_mc1
        _per_point = True

    n_shared = len(shared_v1)
    n_g1_only = len(g1_v)
    n_g2_only = len(g2_v)

    # ------------------------------------------------------------------ #
    # Build per-subject jitter                                            #
    # Shared subjects get the same x-jitter at x=0 and x=1 so that       #
    # connecting lines are exactly vertical (modulo jitter).              #
    # ------------------------------------------------------------------ #
    shared_jitter = (np.random.normal(0, jitter_std, n_shared)
                     if n_shared else np.array([]))
    g1_jitter = (np.random.normal(0, jitter_std, n_g1_only)
                 if n_g1_only else np.array([]))
    g2_jitter = (np.random.normal(0, jitter_std, n_g2_only)
                 if n_g2_only else np.array([]))

    # All x=0 values (shared + group1-only) and their jitter.
    all_v1 = (np.concatenate([shared_v1, g1_v])
              if (n_shared + n_g1_only) else np.array([]))
    all_jitter1 = (np.concatenate([shared_jitter, g1_jitter])
                   if (n_shared + n_g1_only) else np.array([]))
    # All x=1 values (shared + group2-only) and their jitter.
    all_v2 = (np.concatenate([shared_v2, g2_v])
              if (n_shared + n_g2_only) else np.array([]))
    all_jitter2 = (np.concatenate([shared_jitter, g2_jitter])
                   if (n_shared + n_g2_only) else np.array([]))

    # ------------------------------------------------------------------ #
    # Draw scatter                                                        #
    # ------------------------------------------------------------------ #
    if len(all_v1):
        ax.scatter(np.zeros(len(all_v1)) + all_jitter1, all_v1,
                   color='w', s=s, zorder=3, alpha=1, edgecolors='none',
                   clip_on=clip_on)
        if _per_point:
            ax.scatter(np.zeros(len(all_v1)) + all_jitter1, all_v1,
                       c=_all_mc1, s=s, zorder=4, alpha=alpha,
                       edgecolors='none', clip_on=clip_on)
        else:
            ax.scatter(np.zeros(len(all_v1)) + all_jitter1, all_v1,
                       color=_scatter_color, s=s, zorder=4, alpha=alpha,
                       edgecolors='none', clip_on=clip_on)
    if len(all_v2):
        ax.scatter(np.ones(len(all_v2)) + all_jitter2, all_v2,
                   color='w', s=s, zorder=3, alpha=1, edgecolors='none',
                   clip_on=clip_on)
        if _per_point:
            ax.scatter(np.ones(len(all_v2)) + all_jitter2, all_v2,
                       c=_all_mc2, s=s, zorder=4, alpha=alpha,
                       edgecolors='none', clip_on=clip_on)
        else:
            ax.scatter(np.ones(len(all_v2)) + all_jitter2, all_v2,
                       color=_scatter_color, s=s, zorder=4, alpha=alpha,
                       edgecolors='none', clip_on=clip_on)

    # Connecting lines for shared subjects.
    for i in range(n_shared):
        ax.plot([0 + shared_jitter[i], 1 + shared_jitter[i]],
                [shared_v1[i], shared_v2[i]],
                color='w', alpha=1, zorder=2, lw=0.5)
        _lc = _shared_mc1[i] if _per_point else _scatter_color
        ax.plot([0 + shared_jitter[i], 1 + shared_jitter[i]],
                [shared_v1[i], shared_v2[i]],
                color=_lc, alpha=alpha, zorder=2, lw=0.5)

    # ------------------------------------------------------------------ #
    # Per-group marginal CI bars + mean markers                           #
    # ------------------------------------------------------------------ #
    ci_highs = []
    for x_pos, all_vals in [(0, all_v1), (1, all_v2)]:
        valid = all_vals[np.isfinite(all_vals)] if len(all_vals) else np.array([])
        if len(valid) == 0:
            ci_highs.append(np.nan)
            continue
        mean_val = float(np.nanmean(valid))
        if len(valid) > 1:
            _bi = np.random.randint(0, len(valid), (n_boot, len(valid)))
            _bm = np.nanmean(valid[_bi], axis=1)
            lo, hi = np.percentile(
                _bm, [100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2])
        else:
            lo = hi = mean_val
        ci_highs.append(float(hi))
        ax.plot([x_pos, x_pos], [lo, hi], color='w', lw=4, zorder=5)
        ax.plot([x_pos, x_pos], [lo, hi], color='k', lw=2, zorder=6)
        ax.scatter([x_pos], [mean_val], color='w', s=30, zorder=7,
                   edgecolors='none')
        ax.scatter([x_pos], [mean_val], color='k', s=15, zorder=8,
                   edgecolors='none')

    # Mean annotation: anchored at _ymin_now (original data bottom) in data
    # coordinates, offset _CM_ANN_PTS upward so text sits just above the spine
    # (which is also positioned at _ymin_now).
    _ann_fs = plt.rcParams.get('xtick.labelsize',
                               plt.rcParams.get('font.size', 10))
    _ylim_ref = ylim if ylim is not None else ax.get_ylim()
    _ymin_now = min(_ylim_ref)
    # Mirror the xticklabel distance: tick_mark_length + tick_pad.
    _tick_size = plt.rcParams.get('xtick.major.size', 3.5)
    _ann_offset = _tick_size + _CM_TICK_PAD
    _ann_trans = matplotlib.transforms.offset_copy(
        ax.transData, fig=ax.get_figure(), x=0,
        y=_ann_offset, units='points')
    for x_pos, all_vals in [(0, all_v1), (1, all_v2)]:
        valid = all_vals[np.isfinite(all_vals)] if len(all_vals) else np.array([])
        if len(valid) == 0:
            continue
        ax.text(x_pos, _ymin_now,
                f'{float(np.nanmean(valid)):.0f}',
                ha='center', va='bottom', clip_on=False, fontsize=_ann_fs,
                transform=_ann_trans)

    # ------------------------------------------------------------------ #
    # Partially-paired bootstrap for the group difference                 #
    # ------------------------------------------------------------------ #
    # Pre-generate all bootstrap indices (vectorised).
    if n_shared:
        _idx_s = np.random.randint(0, n_shared, (n_boot, n_shared))
        _bs_v1 = shared_v1[_idx_s]   # (n_boot, n_shared)
        _bs_v2 = shared_v2[_idx_s]
    else:
        _bs_v1 = np.zeros((n_boot, 0))
        _bs_v2 = np.zeros((n_boot, 0))

    if n_g1_only:
        _idx_g1 = np.random.randint(0, n_g1_only, (n_boot, n_g1_only))
        _bg1 = g1_v[_idx_g1]         # (n_boot, n_g1_only)
    else:
        _bg1 = np.zeros((n_boot, 0))

    if n_g2_only:
        _idx_g2 = np.random.randint(0, n_g2_only, (n_boot, n_g2_only))
        _bg2 = g2_v[_idx_g2]         # (n_boot, n_g2_only)
    else:
        _bg2 = np.zeros((n_boot, 0))

    _boot_v1 = np.concatenate([_bs_v1, _bg1], axis=1)   # (n_boot, n1_total)
    _boot_v2 = np.concatenate([_bs_v2, _bg2], axis=1)   # (n_boot, n2_total)

    if _boot_v1.shape[1] > 0 and _boot_v2.shape[1] > 0:
        boot_diffs = np.nanmean(_boot_v2, axis=1) - np.nanmean(_boot_v1, axis=1)
    elif _boot_v2.shape[1] > 0:
        boot_diffs = np.nanmean(_boot_v2, axis=1)
    elif _boot_v1.shape[1] > 0:
        boot_diffs = -np.nanmean(_boot_v1, axis=1)
    else:
        boot_diffs = np.full(n_boot, np.nan)

    boot_diffs = boot_diffs[np.isfinite(boot_diffs)]

    if len(boot_diffs):
        mean1 = float(np.nanmean(all_v1)) if len(all_v1) else np.nan
        mean2 = float(np.nanmean(all_v2)) if len(all_v2) else np.nan
        mean_diff = (mean2 - mean1
                     if (np.isfinite(mean1) and np.isfinite(mean2))
                     else np.nan)
        ci_low_d, ci_high_d = np.percentile(
            boot_diffs,
            [100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2])
        if np.isfinite(mean_diff) and mean_diff > 0:
            pval = (np.sum(boot_diffs <= 0) + 1) / (len(boot_diffs) + 1)
        else:
            pval = (np.sum(boot_diffs >= 0) + 1) / (len(boot_diffs) + 1)
    else:
        mean1 = mean2 = mean_diff = ci_low_d = ci_high_d = np.nan
        pval = 1.0

    symbol = sigAsterisk(pval)

    # ------------------------------------------------------------------ #
    # Significance bracket (drawn within axis bounds)                     #
    # ------------------------------------------------------------------ #
    # Expand both ends by 20% of the data range so the bracket fits inside
    # the subplot without leaking, and the empty bottom zone separates the
    # axis from adjacent rows below.
    _ylim_now = ylim if ylim is not None else ax.get_ylim()
    _yrange_now = abs(_ylim_now[1] - _ylim_now[0]) or 1.0
    _ymin_now = min(_ylim_now)
    _ymax_now = max(_ylim_now)
    _headroom = _yrange_now * 0.20
    _new_top = _ymax_now + _headroom
    _new_bot = _ymin_now - _headroom
    if _ylim_now[0] <= _ylim_now[1]:   # normal orientation
        ax.set_ylim(_new_bot, _new_top)
    else:                               # inverted y-axis
        ax.set_ylim(_new_top, _new_bot)
    if show_ns or symbol != 'ns':
        # Place bracket at 40% into the headroom zone (well above data max,
        # well below the new axis top so the text clears).
        _auto_bracket_y = _ymax_now + _headroom * 0.40
        _bracket_y = (bracket_y_override
                      if bracket_y_override is not None else _auto_bracket_y)
        _tick_y = _bracket_y - _yrange_now * 0.02
        plot_diff_brackets(
            symbol, x1=0, x2=1, y1=_tick_y, y2=_tick_y, y_label=_bracket_y,
            col='k', ax=ax, size=_ann_fs)

    # ------------------------------------------------------------------ #
    # Axes formatting                                                     #
    # ------------------------------------------------------------------ #
    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(labels))
    ax.set_xlim(-0.5, 1.5)
    # Spine at original data bottom (inside the expanded ylim buffer zone).
    ax.spines['bottom'].set_position(('data', _ymin_now))
    ax.tick_params(axis='x', pad=_CM_TICK_PAD, bottom=True)
    # Despine is deferred to after format() applies yticks so that
    # trim=True clips to the correct tick range.

    return {
        'vals1': all_v1,
        'vals2': all_v2,
        'ids1': ids1,
        'ids2': ids2,
        'mean1': float(mean1) if np.isfinite(mean1) else np.nan,
        'mean2': float(mean2) if np.isfinite(mean2) else np.nan,
        'diff': float(mean_diff) if np.isfinite(mean_diff) else np.nan,
        'ci_low': float(ci_low_d) if np.isfinite(ci_low_d) else np.nan,
        'ci_high': float(ci_high_d) if np.isfinite(ci_high_d) else np.nan,
        'pval': float(pval),
        'symbol': symbol,
        'n_paired': n_shared,
        'n_group1_only': n_g1_only,
        'n_group2_only': n_g2_only,
        'needed_ymax': float(_new_top),
        'needed_ymin': float(_new_bot),
    }


# ---------------------------------------------------------------------------
# Margin resolution helpers for TrackingExperiment.plot()
# ---------------------------------------------------------------------------

_MARGIN_DEFAULTS = {
    'line': ['line'],
    'hist2d': ['contour'],
    'histogram': ['histogram'],
    'pdf': ['pdf'],
    'trajectory2d': ['trajectory2d'],
    'scatter': ['histogram'],
    'compare_means': ['compare_means'],
}
_1D_PLOT_TYPES = {'histogram', 'pdf'}


def _resolve_margin(margin, plot_type):
    """Return a list of margin plot-type strings, or [] if the margin is disabled.

    Parameters
    ----------
    margin : bool | str | list[str]
        ``False`` → disabled; ``True`` → per-plot_type default;
        ``str`` or ``list[str]`` → explicit override(s).
    plot_type : str
        Main plot type (used to look up defaults and enforce 1-D restriction).

    Returns
    -------
    list[str]
    """
    if margin is False or margin is None:
        return []
    if margin is True:
        return list(_MARGIN_DEFAULTS.get(plot_type, ['line']))
    types = [margin] if isinstance(margin, str) else list(margin)
    if plot_type in _1D_PLOT_TYPES:
        valid = [t for t in types if t in ('line', 'histogram', 'pdf')]
        dropped = [t for t in types if t not in valid]
        if dropped:
            import warnings as _w
            _w.warn(
                f"Margin type(s) {dropped!r} are not valid for "
                f"plot_type={plot_type!r} (1-D output). Dropping.",
                stacklevel=3,
            )
        types = valid
    return types


_MARGIN_LINESTYLES = ['solid', 'dashed', 'dotted', 'dashdot',
                      (0, (3, 1, 1, 1)),        # densely dash-dot
                      (0, (5, 1)),               # densely dashed
                      (0, (1, 1)),               # densely dotted
                      (0, (3, 5, 1, 5, 1, 5))   # dash-dot-dot
                      ]


def _draw_margin_cell(margin_ax, mtype, xs, ys, color, dim,
                      summary_func, bins, probability,
                      confidence_interval, confidence, n_boot,
                      plot_type, summary_dict, n_overlays,
                      overlay_index=0,
                      linestyle='solid', label=None):
    """Draw one cell's contribution into a margin axis.

    Parameters
    ----------
    margin_ax : Axes or None
    mtype : str  — 'line', 'histogram', 'scatter', 'circ_hist', 'contour',
        or 'trajectory2d'
    xs, ys : ndarray or None
        Cell data as cached by TrackingExperiment.plot().
    color : color spec
    dim : 'right' or 'bottom'
    summary_func : callable
    bins : int or array-like
    probability : bool
    confidence_interval : bool
    confidence : float
    n_boot : int
    plot_type : str  — main-panel plot type (used to select trajectory logic)
    summary_dict : dict or None  — from plot_trajectory2d(return_summary=True)
    n_overlays : int  — number of cells sharing this margin (controls alpha)
    overlay_index : int, default 0
        Zero-based index of this cell among all cells sharing the margin.
        Used by the 'trajectory2d' mtype to stack concentric CI arcs.
    linestyle : str or tuple, default 'solid'
        Line style passed to plot/step calls.
    label : str or None
        Label used for the legend entry.  Only applied to the primary
        (mean / summary) line, not to individual traces.
    """
    if margin_ax is None or xs is None:
        return
    alpha_overlay = max(0.2, 1.0 / max(1, n_overlays))

    if mtype == 'line':
        if plot_type == 'trajectory2d':
            # Radial distance vs time (y) for the trajectory margin.
            if summary_dict is None or ys is None or ys.ndim < 2:
                return
            trajectory = summary_dict.get('trajectory')
            if trajectory is None:
                return
            dist = np.linalg.norm(trajectory, axis=1)
            y_axis = ys[np.isnan(ys).sum(1).argmin()]
            margin_ax.plot(dist.T, ys.T, color=color, lw=0.5, alpha=alpha_overlay)
            mean_dist = np.nanmean(dist, axis=0)
            margin_ax.plot(mean_dist, y_axis, color='w', lw=3, zorder=3)
            margin_ax.plot(mean_dist, y_axis, color=color, lw=2, zorder=4)
        elif xs.ndim == 2 and ys is not None and ys.ndim == 2:
            mean_x = summary_func(xs, axis=0)
            y = ys[np.isnan(ys).sum(1).argmin()]
            margin_ax.plot(mean_x, y, color=color, zorder=1,
                           linestyle=linestyle, label=label)
            if confidence_interval:
                lows, highs = bootstrap_ci(xs, summary_func,
                                           confidence=confidence, n_boot=n_boot)
                margin_ax.fill_betweenx(y, lows, highs,
                                        color=color, alpha=0.3,
                                        zorder=2, linewidth=0)

    elif mtype == 'histogram':
        # Right margin: distribution of y-values, count on x-axis.
        # Bottom margin: distribution of x-values, count on y-axis.
        data = (ys if dim == 'right' else xs)
        if data is None:
            return
        flat = data.flatten() if data.ndim > 1 else np.asarray(data)
        flat = flat[~np.isnan(flat)]
        if flat.size == 0:
            return
        counts, edges = np.histogram(flat, bins=bins)
        if probability:
            counts = counts / max(counts.sum(), 1) * 100
        mid_points = (edges[:-1] + edges[1:]) / 2
        if dim == 'right':
            margin_ax.step(counts, mid_points, color=color, where='mid',
                           alpha=alpha_overlay, linestyle=linestyle, label=label)
        else:
            margin_ax.step(mid_points, counts, color=color, where='mid',
                           alpha=alpha_overlay, linestyle=linestyle, label=label)

    elif mtype == 'scatter':
        if summary_dict is None:
            return
        last_pos = summary_dict.get('last_pos')
        if last_pos is None:
            return
        # right: x=endpoint-x (new), y=endpoint-y (shared with main)
        # bottom: x=endpoint-x (shared with main), y=endpoint-y (new)
        margin_ax.scatter(last_pos[:, 0], last_pos[:, 1],
                          color=color, s=1, alpha=alpha_overlay,
                          edgecolors='none')

    elif mtype == 'circ_hist':
        if summary_dict is None:
            return
        hist = summary_dict.get('hist')
        bin_edges = summary_dict.get('bin_edges')
        if hist is None or bin_edges is None:
            return
        mid_points = (bin_edges[:-1] + bin_edges[1:]) / 2
        if dim == 'right':
            margin_ax.step(hist, mid_points, color=color,
                           where='mid', alpha=alpha_overlay)
        else:
            margin_ax.step(mid_points, hist, color=color,
                           where='mid', alpha=alpha_overlay)

    elif mtype == 'trajectory2d':
        # Mirror ALL enabled main-panel overlays onto the margin axis.
        # Each overlay type is drawn independently (not an if/elif chain)
        # so that combinations like circ_hist + mean_line both appear.
        if summary_dict is None:
            return
        from matplotlib.patches import Arc as _Arc
        mean_angle = summary_dict.get('mean_angle')
        lb = summary_dict.get('lb')
        ub = summary_dict.get('ub')
        trajectory = summary_dict.get('trajectory')
        last_pos = summary_dict.get('last_pos')
        arc_gap = 0.14  # radial spacing between concentric overlays
        mean_radius = summary_dict.get('mean_radius')
        ellipse_mean = summary_dict.get('ellipse_mean')
        ellipse_a = summary_dict.get('ellipse_a')
        ellipse_b = summary_dict.get('ellipse_b')
        ellipse_angle = summary_dict.get('ellipse_angle')

        margin_ax.set_aspect('equal', adjustable='box')
        drew_something = False

        if mean_angle is not None and lb is not None and ub is not None:
            # circ_hist — concentric CI arcs stacked by overlay_index.
            base_r = 1.0
            radius = base_r + overlay_index * arc_gap
            for arc_color, lw, zo in [('w', 4, 4), (color, 2, 5)]:
                arc = _Arc((0, 0),
                           width=2 * radius, height=2 * radius,
                           angle=0,
                           theta1=np.degrees(lb), theta2=np.degrees(ub),
                           color=arc_color, lw=lw, zorder=zo,
                           capstyle='round')
                margin_ax.add_artist(arc)
            for sc_color, s, zo in [('w', 40, 4), (color, 20, 5)]:
                margin_ax.scatter(
                    radius * np.cos(mean_angle),
                    radius * np.sin(mean_angle),
                    color=sc_color, marker='o', s=s, zorder=zo)
            max_r = base_r + (n_overlays - 1) * arc_gap + 0.08
            margin_ax.set_xlim(-max_r, max_r)
            margin_ax.set_ylim(-max_r, max_r)
            # Flag this axis so the post-loop sync step skips it.
            margin_ax._traj2d_circ_hist = True
            drew_something = True

        if mean_radius is not None:
            # circle overlay — mean radius circle.
            mc = plt.Circle((0, 0), radius=mean_radius,
                            color=color, fill=False, lw=2,
                            alpha=alpha_overlay, zorder=3)
            margin_ax.add_artist(mc)
            drew_something = True

        if ellipse_mean is not None:
            # contour overlay — confidence ellipse.
            ellipse_alpha = max(0.1, 1.0 / max(1, n_overlays))
            el = matplotlib.patches.Ellipse(
                ellipse_mean,
                2 * ellipse_a, 2 * ellipse_b,
                angle=np.degrees(ellipse_angle),
                color=color, alpha=ellipse_alpha,
                fill=True, lw=0.5, zorder=2)
            margin_ax.add_artist(el)
            drew_something = True

        if trajectory is not None:
            # mean_line overlay — mean 2-D trajectory.
            mean_traj = np.nanmean(trajectory, axis=0)
            margin_ax.plot(mean_traj[0], mean_traj[1],
                           color=color, lw=1.5, alpha=alpha_overlay, zorder=3)
            margin_ax.scatter(mean_traj[0, -1], mean_traj[1, -1],
                              color=color, marker='o', s=8, zorder=4)
            drew_something = True

        if not drew_something and last_pos is not None:
            # Fallback: endpoint scatter when no specific overlay was available.
            margin_ax.scatter(last_pos[:, 0], last_pos[:, 1],
                              color=color, s=1, alpha=alpha_overlay,
                              edgecolors='none')

    elif mtype == 'contour':
        # Draw density contours on the margin axis.
        #
        # When the main plot is 'hist2d', the pre-computed histogram grid
        # (already in summary_dict) is used directly — this is both correct
        # (contours of the binned density) and fast (no KDE computation).
        #
        # For other main plot types there is no pre-computed 2D grid, so we
        # fall back to a subsampled point-cloud gaussian_kde.
        if summary_dict is None:
            return
        hist = summary_dict.get('hist')
        xedges = summary_dict.get('xedges')
        yedges = summary_dict.get('yedges')
        contour_alpha = summary_dict.get('contour_alpha', 0.3)
        n_req = summary_dict.get('n_contours') or 1

        if hist is not None and xedges is not None and yedges is not None:
            # ---- fast path: contours from pre-computed histogram grid ----
            xcen = (xedges[:-1] + xedges[1:]) / 2
            ycen = (yedges[:-1] + yedges[1:]) / 2
            # hist is shape (nx, ny); contourf expects (ny, nx) → transpose
            ZZ = hist.T.astype(float)
            nz = ZZ[ZZ > 0]
            if nz.size < 2:
                return
            if n_req == 1:
                pcts = [50]
            else:
                pcts = np.linspace(100 / (n_req + 1), 50, n_req)
            levels = np.unique(np.percentile(nz, pcts))
            if levels.size == 0:
                return
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                '', [(1, 1, 1), color])
            norm = matplotlib.colors.Normalize(vmin=levels[0], vmax=ZZ.max())
            try:
                if contour_alpha and contour_alpha > 0:
                    fill_levels = np.concatenate([levels, [ZZ.max() * 1.001]])
                    margin_ax.contourf(xcen, ycen, ZZ,
                                       levels=fill_levels,
                                       cmap=cmap, norm=norm,
                                       alpha=contour_alpha)
                margin_ax.contour(xcen, ycen, ZZ,
                                  levels=levels,
                                  colors=[color], linewidths=0.8)
            except Exception:
                pass
        else:
            # ---- fallback: subsampled point-cloud KDE ----
            xs_raw = summary_dict.get('xs')
            ys_raw = summary_dict.get('ys')
            if xs_raw is None or ys_raw is None:
                return
            valid = ~(np.isnan(xs_raw) | np.isnan(ys_raw))
            xv, yv = xs_raw[valid], ys_raw[valid]
            if xv.size < 5:
                return
            _KDE_MAX_PTS = 20_000
            if xv.size > _KDE_MAX_PTS:
                rng = np.random.default_rng(0)
                idx = rng.choice(xv.size, _KDE_MAX_PTS, replace=False)
                xv, yv = xv[idx], yv[idx]
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(np.vstack([xv, yv]))
                _bins = 60
                xcen = np.linspace(xv.min(), xv.max(), _bins)
                ycen = np.linspace(yv.min(), yv.max(), _bins)
                XX, YY = np.meshgrid(xcen, ycen)
                ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
                kde_at_data = kde(np.vstack([xv, yv]))
                if n_req == 1:
                    pcts = [50]
                else:
                    pcts = np.linspace(100 / (n_req + 1), 50, n_req)
                levels = np.unique(np.percentile(kde_at_data, pcts))
                if levels.size == 0:
                    return
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    '', [(1, 1, 1), color])
                norm = matplotlib.colors.Normalize(
                    vmin=levels[0], vmax=ZZ.max())
                if contour_alpha and contour_alpha > 0:
                    fill_levels = np.concatenate([levels, [ZZ.max() * 1.001]])
                    margin_ax.contourf(xcen, ycen, ZZ,
                                       levels=fill_levels,
                                       cmap=cmap, norm=norm,
                                       alpha=contour_alpha)
                margin_ax.contour(xcen, ycen, ZZ,
                                  levels=levels,
                                  colors=[color], linewidths=0.8)
            except Exception:
                pass

    elif mtype == 'pdf':
        # Per-fly normalized histogram mean + CI on the margin axis.
        data = (ys if dim == 'right' else xs)
        if data is None:
            return
        # Guarantee 2-D so per-fly iteration works.
        if data.ndim == 1:
            data = data[np.newaxis]
        all_valid = data[~np.isnan(data)]
        if all_valid.size == 0:
            return
        if isinstance(bins, int):
            _, _edges = np.histogram(all_valid, bins=bins)
        else:
            _edges = np.asarray(bins)
        _centers = (_edges[:-1] + _edges[1:]) / 2
        _hists = []
        for _row in data:
            _v = _row[~np.isnan(_row)]
            if _v.size == 0:
                continue
            _h, _ = np.histogram(_v, bins=_edges)
            _h = _h.astype(float)
            _s = _h.sum()
            if _s > 0:
                _h /= _s
            if probability:
                _h *= 100
            _hists.append(_h)
        if not _hists:
            return
        _hists = np.array(_hists)
        _mean_h = np.nanmean(_hists, axis=0)
        if _hists.shape[0] > 1:
            _rand = np.random.randint(0, _hists.shape[0],
                                      (n_boot, _hists.shape[0]))
            _boot = np.nanmean(_hists[_rand], axis=1)
            _lo, _hi = np.percentile(
                _boot,
                [100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2],
                axis=0)
        else:
            _lo = _hi = _mean_h
        if dim == 'right':
            margin_ax.plot(_mean_h, _centers,
                           color=color, lw=2, zorder=4, alpha=alpha_overlay,
                           linestyle=linestyle, label=label)
            margin_ax.fill_betweenx(_centers, _lo, _hi,
                                    color=color, alpha=0.3,
                                    edgecolor='none', zorder=3)
        else:
            margin_ax.plot(_centers, _mean_h,
                           color=color, lw=2, zorder=4, alpha=alpha_overlay,
                           linestyle=linestyle, label=label)
            margin_ax.fill_between(_centers, _lo, _hi,
                                   color=color, alpha=0.3,
                                   edgecolor='none', zorder=3)


# ---------------------------------------------------------------------------

class Kalman_Filter():
    '''
    2D Kalman filter, assuming constant acceleration.

    Use a 2D Kalman filter to return the estimated position of points given 
    linear prediction of position assuming (1) fixed jerk, (2) gaussian
    jerk noise, and (3) gaussian measurement noise.

    Parameters
    ----------
    num_objects : int
        Number of objects to model as well to expect from detector.
    sampling_interval : float
        Sampling interval in seconds. Should equal (frame rate) ** -1.
    jerk : float
        Jerk is modeled as normally distributed. This is the mean.
    jerk_std : float
        Jerk distribution standard deviation.
    measurement_noise_x : float
        Variance of the x component of measurement noise.
    measurement_noise_y : float
        Variance of the y component of measurement noise.

    '''
    def __init__(self, num_objects, num_frames=None, sampling_interval=30**-1,
                 jerk=0, jerk_std=125,
                 measurement_noise_x=5, measurement_noise_y=5,
                 width=None, height=None):
        self.width = width
        self.height = height if height is not None else 0
        self.num_objects = num_objects
        self.num_frames = num_frames
        self.sampling_interval = sampling_interval
        self.dt = self.sampling_interval
        self.jerk = jerk
        self.jerk_std = jerk_std
        self.measurement_noise_x = measurement_noise_x
        self.measurement_noise_y = measurement_noise_y
        self.tkn_x, self.tkn_y = self.measurement_noise_x, self.measurement_noise_y
        # process error covariance matrix
        self.Ez = np.array(
            [[self.tkn_x, 0         ],
             [0,          self.tkn_y]])
        # measurement error covariance matrix (constant jerk)
        self.Ex = np.array(
            [[self.dt**6/36, 0,             self.dt**5/12, 0,             self.dt**4/6, 0           ],
             [0,             self.dt**6/36, 0,             self.dt**5/12, 0,            self.dt**4/6],
             [self.dt**5/12, 0,             self.dt**4/4,  0,             self.dt**3/2, 0           ],
             [0,             self.dt**5/12, 0,             self.dt**4/4,  0,            self.dt**3/2],
             [self.dt**4/6,  0,             self.dt**3/2,  0,             self.dt**2,   0           ],
             [0,             self.dt**4/6,  0,             self.dt**3/2,  0,            self.dt**2  ]])
        self.Ex *= self.jerk_std**2
        # set initial position variance
        self.P = np.copy(self.Ex)
        ## define update equations in 2D as matrices - a physics based model for predicting
        # object motion
        ## we expect objects to be at:
        # [state update matrix (position + velocity)] + [input control (acceleration)]
        self.state_update_matrix = np.array(
            [[1, 0, self.dt, 0,       self.dt**2/2, 0           ],
             [0, 1, 0,       self.dt, 0,            self.dt**2/2],
             [0, 0, 1,       0,       self.dt,      0           ],
             [0, 0, 0,       1,       0,            self.dt     ],
             [0, 0, 0,       0,       1,            0           ],
             [0, 0, 0,       0,       0,            1           ]])
        self.control_matrix = np.array(
            [self.dt**3/6, self.dt**3/6, self.dt**2/2, self.dt**2/2, self.dt, self.dt])
        # measurement function to predict next measurement
        self.measurement_function = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]])
        self.A = self.state_update_matrix
        self.B = self.control_matrix
        self.C = self.measurement_function
        ## initialize result variables
        self.Q_local_measurement = []  # point detections
        ## initialize estimateion variables for two dimensions
        self.max_tracks = self.num_objects
        dimension = self.state_update_matrix.shape[0]
        self.Q_estimate = np.empty((dimension, self.max_tracks))
        self.Q_estimate.fill(np.nan)
        if self.num_frames is not None:
            self.Q_loc_estimateX = np.empty((self.num_frames, self.max_tracks))
            self.Q_loc_estimateX.fill(np.nan)
            self.Q_loc_estimateY = np.empty((self.num_frames, self.max_tracks))
            self.Q_loc_estimateY.fill(np.nan)
        else:
            self.Q_loc_estimateX = []
            self.Q_loc_estimateY = []
        self.num_tracks = self.num_objects
        self.num_detections = self.num_objects
        self.frame_num = 0

    def get_prediction(self):
        '''
        Get next predicted coordinates using current state and measurement information.

        Returns
        -------
        estimated points : ndarray
            approximated positions with shape.
        '''
        ## kalman filter
        # predict next state with last state and predicted motion
        self.Q_estimate = self.A @ self.Q_estimate + (self.B * self.jerk)[:, None]
        # predict next covariance
        self.P = self.A @ self.P @ self.A.T + self.Ex
        # Kalman Gain
        try:
            self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Ez)
            ## now assign the detections to estimated track positions
            # make the distance (cost) matrix between all pairs; rows = tracks and
            # cols = detections
            self.estimate_points = self.Q_estimate[:2, :self.num_tracks]
            # np.clip(self.estimate_points[0], -self.height/2, self.height/2, out=self.estimate_points[0])
            # np.clip(self.estimate_points[1], -self.width/2, self.width/2, out=self.estimate_points[1])
            return self.estimate_points.T  # shape should be (num_objects, 2)
        except:
            return np.array([np.nan, np.nan])
    

    def add_starting_points(self, points):
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                           "shape (num_objects X 2)")
        self.Q_estimate.fill(0)
        self.Q_estimate[:2] = points.T
        if self.num_frames is not None:
            self.Q_loc_estimateX[self.frame_num] = self.Q_estimate[0]
            self.Q_loc_estimateY[self.frame_num] = self.Q_estimate[1]
        else:
            self.Q_loc_estimateX.append(self.Q_estimate[0])
            self.Q_loc_estimateY.append(self.Q_estimate[1])
        self.frame_num += 1

    def add_measurement(self, points):
        ## detections matrix
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                           "shape (num_objects X 2)")
        self.Q_loc_meas = points
        # find nans, exclude from the distance matrix
        # no_nans_meas = np.isnan(self.Q_loc_meas[:, :self.num_tracks]) == False
        # no_nans_meas = no_nans_meas.max(1)
        # assigned_measurements = np.empty((self.num_tracks, 2))
        # assigned_measurements.fill(np.nan)
        # self.est_dist = scipy.spatial.distance_matrix(
        #     self.estimate_points.T,
        #     self.Q_loc_meas[:self.num_tracks][no_nans_meas])
        # use hungarian algorithm to find best pairings between estimations and measurements
        # if not np.any(np.isnan(self.est_dist)):
        # try:
        #     asgn = scipy.optimize.linear_sum_assignment(self.est_dist)
        # except:
        #     print(self.est_dist)
        # for num, val in zip(asgn[0], asgn[1]):
        #     assigned_measurements[num] = self.Q_loc_meas[no_nans_meas][val]
        # remove problematic cases
        # close_enough = self.est_dist[asgn] < 25
        # no_nans = np.logical_not(np.isnan(assigned_measurements)).max(1)
        # good_cases = np.logical_and(close_enough, no_nans)
        # if self.width is not None:
        #     in_bounds_x = np.logical_and(
        #         assigned_measurements.T[1] > 0, 
        #         assigned_measurements.T[1] < self.width)
        # if self.height is not None:
        #     in_bounds_y = np.logical_and(
        #         assigned_measurements.T[0] > 0, 
        #         assigned_measurements.T[0] < self.height)
        # good_cases = no_nans
        # good_cases = no_nans * in_bounds_x * in_bounds_y
        # apply assignemts to the update
        # for num, (good, val) in enumerate(zip(good_cases, assigned_measurements)):
        #     if good:
        #         self.Q_estimate[:, num] = self.Q_estimate[:, num] + self.K @ (
        #             val.T - self.C @ self.Q_estimate[:, num])
        #         self.track_strikes[num] = 0
        #     else:
        #         self.track_strikes[num] += 1
        #         self.Q_estimate[2:, num] = 0
        self.Q_estimate = self.Q_estimate + self.K @ (self.Q_loc_meas - self.C @ self.Q_estimate)
        # update covariance estimation
        self.P = (np.eye((self.K @ self.C).shape[0]) - self.K @ self.C) @ self.P
        ## store data
        if self.num_frames is not None:
            self.Q_loc_estimateX[self.frame_num] = self.Q_estimate[0]
            self.Q_loc_estimateY[self.frame_num] = self.Q_estimate[1]
        else:
            self.Q_loc_estimateX.append(self.Q_estimate[0])
            self.Q_loc_estimateY.append(self.Q_estimate[1])
        self.frame_num += 1

    def update_vals(self, **kwargs):
        """Allow for replacing parameters like jerk_std and noise estimates."""
        for key, val in kwargs.items():
            self.__setattr__(key, val)


class KalmanAngle(Kalman_Filter):
    """A special instance of the Kalman Filter for single object, 1D data."""
    def __init__(self, **kwargs):
        super().__init__(num_objects=1, measurement_noise_y=0, width=0, **kwargs) 
        self.last_point = 0
        self.revolutions = 0
        self.record = []
        # print key parameters
        # print(f"jerk std={self.jerk_std}, noise std={self.measurement_noise_x}")

    def store(self, point):
        """Converts single point to appropriate shape for the 2D filter.
        
        Note: keep the point and last point variables in wrapped format and 
        unwrap just for when adding the measurement.
        """
        point = np.copy(point)
        if point is np.nan:
            point = self.last_point
        # unwrap
        if (self.last_point < -np.pi/2) and (point > np.pi/2):
            self.revolutions -= 1
        elif (self.last_point > np.pi/2) and (point < -np.pi/2):
            self.revolutions += 1
        self.last_point = np.copy(point)
        point += 2*np.pi*self.revolutions
        self.record += [point]
        # add 0 for second dimension
        point = np.array([point, 0])[np.newaxis]
        if self.frame_num == 0:
            self.add_starting_points(point)
        else:
            self.add_measurement(point)

    def predict(self):
        output = self.get_prediction()[0, 0]
        output %= 2*np.pi
        if output > np.pi:
            output -= 2*np.pi
        elif output < -np.pi:
            output += 2*np.pi
        return output

class TrackingVideo():
    def __init__(self, filename):
        """Load the video file and get metadata.


        Parameters
        ----------
        filename : str
            The path to the video file.
        """
        self.filename = filename
        self.subject = os.path.basename(filename).split(".")[0]
        # load video from matlab video
        if filename.endswith(".mat"):
            # self.video = io.vreader(self.filename)
            self.video = scipy.io.loadmat(self.filename)
            self.times = self.video['t_v'][:, 0]
            self.video = self.video['vidData'][:, :, 0] # height x width x channel x num_frames
            self.video = self.video.transpose(2, 0, 1)
        # or load a frame generator 
        else:
            # self.video = io.vread(self.filename, as_grey=True)[..., 0]
            # self.times = np.arange(len(self.video))
            input_dict = {'-hwaccel': 'cuda', '-hwaccel_output_format': 'cuda'}
            output_dict = {'-c:v': 'h264_nvenc'}
            self.video = io.FFmpegReader(self.filename, inputdict=input_dict, outputdict=output_dict)
            self.video.shape = self.video.getShape()
            self.times = np.arange(self.video.shape[0]) / self.video.inputfps
            # todo: replace the array method for an interative method of getting heading data
        # get video metadata
        self.num_frames, self.height, self.width = self.video.shape[:3]
        # get center and radii from the tracked file
        # tracked_folder = os.path.join(os.path.dirname(self.filename), "tracked_data")
        # tracked_fns = os.listdir(tracked_folder)
        # tracked_fn = [os.path.join(tracked_folder, fn) for fn in tracked_fns if self.subject in fn][0]
        # self.circle_data = np.load(tracked_fn)
        # self.center = np.array([self.circle_data['x'][0], self.circle_data['y'][0]])
        # self.inner_radius, self.outer_radius = self.circle_data[['radius_small', 'radius_large']][0]
        # # get ring coordinates and angles
        # self.set_rings(self.center, self.inner_radius, self.outer_radius)

    def get_background(self):
        """Get average frame of the whole video."""
        self.background = self.video.mean(-1)

    def ring_gui(self):
        """Use a matplotlib GUI to set the ring radii and thickness."""
        # use the camera module to load this video as a dummy video
        # with the 
        import sys
        sys.path.append("..\\holocube")
        from camera import Camera
        cam = Camera(plot_stimulus=False, kalman=True, config_fn="..\\video_player.config", camera=self.filename, video_player_fn="..\\video_player_server.py", com_correction=True)
        # cam.arm()
        cam.displaying = True
        # cam.capture_dummy()
        cam.display_start()
        cam.capture_start()
        resp = input("adjust the ring parameters and press <enter> when you're happy: ")
        cam.capture_stop()
        cam.display_stop()
        # store results from the gui
        for var, var_storage in zip(['thresh', 'inner_r', 'outer_r'], ['threshold', 'inner_radius', 'outer_radius']): 
            self.__setattr__(var_storage, cam.__getattribute__(var))
        # save these new values into the corresponding h5 file
        # extension = "." + self.filename.split(".")[-1]
        # h5_fn = self.filename.replace(extension, ".h5")
        # dataset = h5py.File(h5_fn, mode='r+')

    def set_rings(self, center, inner_radius=10, outer_radius=20, wing_radius=40, 
                  thickness=3):
        """Define two rings for heading detection.


        Parameters
        ----------
        (center_x, center_y) : tuple, len=2
            The 2D coordinate of the center of both rings.
        inner_radius : float, default=10
            The radius of the inner ring, which should intersect both sides 
            of the fly.
        outer_radius : float, default=20
            The radius of the outer ring, which should intersect only the abdomen.
        wing_radius : float, default=40
            The radius of the wing ring, which should intersect only the wings.            
        thickness : int, default=3
            The thickness of the rings.
        """
        if inner_radius > outer_radius:
            outer_r_new = inner_radius
            outer_radius = inner_radius
            inner_radius = outer_r_new
        self.inner_radius, self.outer_radius = inner_radius, outer_radius
        self.wing_radius = wing_radius
        self.center = center
        x, y = center
        # make a mask of the two rings
        xs, ys = np.arange(self.width), np.arange(self.height)
        xgrid, ygrid = np.meshgrid(xs, ys)
        xgrid, ygrid = xgrid.astype(float), ygrid.astype(float)
        xgrid -= x
        ygrid -= y
        dists = np.sqrt(xgrid ** 2 + ygrid ** 2)
        self.dists = dists
        angles = np.arctan2(ygrid, xgrid)
        self.angles = angles
        # get indices of the two ring masks
        # inner ring and angles:
        include_inner = (dists >= inner_radius - thickness/2) * (dists <= inner_radius + thickness/2)
        self.inner_inds = include_inner
        ys_inner, xs_inner = np.where(include_inner)
        self.inner_ring_coords = np.array([xs_inner, ys_inner]).T
        self.inner_ring_angles = angles[include_inner]
        # outer ring and angles:
        include_outer = (dists >= outer_radius - thickness/2) * (dists <= outer_radius + thickness/2)
        self.outer_inds = include_outer
        ys_outer, xs_outer = np.where(include_outer)
        self.outer_ring_coords = np.array([xs_outer, ys_outer]).T
        self.outer_ring_angles = angles[include_outer]
        # wing ring and angles:
        wing_radius = 2 * outer_radius
        # cap the wing radius at the diagonal of the video
        wing_radius = min(wing_radius, min(self.height, self.width)/2)
        include_wing = (dists >= wing_radius - thickness/2) * (dists <= wing_radius + thickness/2)
        self.wing_inds = include_wing
        ys_outer, xs_outer = np.where(include_wing)
        self.wing_ring_coords = np.array([xs_outer, ys_outer]).T
        self.wing_ring_angles = angles[include_wing]


    def get_heading(self, floor=0, ceiling=40, method='rings', wings=False, head=False, gui=False):
        """Threshold the video and get the heading for each frame.


        Parameters
        ----------
        floor : int, default=5
            The pixel value lower bound for the inclusive filter
        ceiling : int, default=np.inf
            The pixel value upper bound for the inclusive filter
        method : str, default='rings'
            The heading can be appprixmated several ways, but only one is
            currently available, 'rings'. TODO: 1) get the svd of the thresholded 
            coordinates; 2) fit an ellipse; 3) use deep lab cut or some other machine
            learning model. 
        wings : bool, default=False
            Whether to also calculate the left and right wingbeat amplitudes.
        head : bool, default=False
            Whether to also calculate the head angle.
        """
        self.heading = []
        self.thrust = []
        self.wing_vals = []
        if 'headings' not in dir(self):
            self.headings = {}
        print('processing individual frames:')
        for num, frame in enumerate(self.video):
            if method == 'combined':
                # get the thresholded frame accounting for inversions
                if floor == 0 and ceiling > 0:
                    thresh = ceiling
                    invert = False
                elif ceiling >= 255 and floor > 0:
                    thresh = floor
                    invert = True
                # get the ring coordinates
                outer_inds, outer_angs = np.where(self.outer_inds), self.outer_ring_angles
                inner_inds, inner_angs = np.where(self.inner_inds), self.inner_ring_angles
                # 1. get outer ring, which should include just the tail
                if frame.ndim > 2:
                    frame = frame.mean(-1).astype('uint8')
                # get thresholded frame                
                if invert:
                    frame_mask = frame < thresh
                else:
                    frame_mask = frame > thresh
                # account for shifts in the center of mass
                com = scipy.ndimage.measurements.center_of_mass(frame_mask)
                diff = np.array(com) - np.array([self.height/2, self.width/2])
                # get the outer ring values
                outer_ring = frame[outer_inds[0], outer_inds[1]]
                heading = np.nan
                # 2. find the tail and head orientation by thresholding the outer ring
                # values and calculate the tail heading as the circular mean
                if invert:
                    tail = outer_ring < thresh
                else:
                    tail = outer_ring > thresh
                tail_angs = outer_angs[tail]
                tail_dir = scipy.stats.circmean(tail_angs.flatten(), low=-np.pi, high=np.pi)
                # the head direction is the polar opposite of the tail
                head_dir = tail_dir + np.pi
                # head_dir = tail_dir
                if head_dir > np.pi:
                    head_dir -= 2 * np.pi
                # 3. get bounds of head angles, ignoring angles within +/- 90 degrees of the tail
                lower_bounds, upper_bounds = [head_dir - np.pi / 2], [head_dir + np.pi / 2]
                # wrap bounds if they go outside of [-pi, pi]
                if lower_bounds[0] < -np.pi:
                    lb = np.copy(lower_bounds[0])
                    lower_bounds[0] = -np.pi
                    lower_bounds += [lb + 2 * np.pi]
                    upper_bounds += [np.pi]
                elif upper_bounds[0] > np.pi:
                    ub = np.copy(upper_bounds[0])
                    upper_bounds[0] = np.pi
                    upper_bounds += [ub - 2 * np.pi]
                    lower_bounds += [-np.pi]
                lower_bounds, upper_bounds = np.array(lower_bounds), np.array(upper_bounds)
                # 4. calculate the heading within the lower and upper bounds
                include = np.zeros(len(inner_angs), dtype=bool)
                for lower, upper in zip(lower_bounds, upper_bounds):
                    include += (inner_angs > lower) * (inner_angs < upper)
                if np.any(include):
                    inner_vals = frame[inner_inds[0], inner_inds[1]][include]
                    if invert:
                        head_pos = inner_vals < thresh
                    else:
                        head_pos = inner_vals > thresh
                    heading = scipy.stats.circmean(
                        inner_angs[include][head_pos],
                        low=-np.pi, high=np.pi)
                # convert from heading angle to head position
                heading_pos = self.inner_radius * np.array([np.sin(heading), np.cos(heading)])
                # calculate direction vector between the center of the fly and the head
                direction = heading_pos - diff
                heading = np.arctan2(direction[0], direction[1])
                # todo: rotate the direction vector by the heading angle
                rot_matrix = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
                direction_subj = rot_matrix @ direction
                thrust_approx = direction_subj[1]
                self.thrust += [thrust_approx]
                # store the shift in the center of mass for plotting
                self.com_shift = np.round(-diff).astype(int)
                # center and wrap the heading
                heading -= np.pi/2
                if heading < -np.pi:
                    heading += 2 * np.pi
                # store
                self.heading += [heading]
            elif method == 'rings':
                if frame.ndim > 2:
                    frame = frame.mean(-1).astype('uint8')
                # 1. get angles of the outside ring corresponding to the fly's tail end
                xs, ys = self.outer_ring_coords.T
                outer_vals = frame[ys, xs]
                tail = (outer_vals > floor) * (outer_vals <= ceiling)
                tail_angs = self.outer_ring_angles[tail]
                # 1.5. check that the distribution
                # 2. get circular mean of tail
                tail_dir = scipy.stats.circmean(tail_angs, low=-np.pi, high=np.pi)
                head_dir = tail_dir + np.pi
                if head_dir > np.pi:
                    head_dir -= 2 * np.pi
                # 3. get bounds of head angles, ignoring angles within +/- 60 degrees of the tail
                lower_bounds, upper_bounds = [head_dir - np.pi/2], [head_dir + np.pi/2]
                # wrap bounds if they go outside of [-pi, pi]
                if lower_bounds[0] < -np.pi:
                    lb = np.copy(lower_bounds[0])
                    lower_bounds[0] = -np.pi
                    lower_bounds += [lb + 2 * np.pi]
                    upper_bounds += [np.pi]
                elif upper_bounds[0] > np.pi:
                    ub = np.copy(upper_bounds[0])
                    upper_bounds[0] = np.pi
                    upper_bounds += [ub - 2 * np.pi]
                    lower_bounds += [-np.pi]
                # 4. get angles of the inside ring corresponding to both head and tail
                head_angs = []
                for lb, ub in zip(lower_bounds, upper_bounds):
                    include = (self.inner_ring_angles > lb) * (self.inner_ring_angles < ub)
                    if np.any(include):
                        xs, ys = self.inner_ring_coords[include].T
                        inner_vals = frame[self.inner_ring_coords[include][:, 1],
                                           self.inner_ring_coords[include][:, 0]]
                        head_pos = (inner_vals > floor) * (inner_vals <= ceiling)
                        head_angs += [self.inner_ring_angles[include][head_pos]]
                if len(head_angs) > 0:
                    head_angs = np.concatenate(head_angs)
                # 5. grab the head angs within those bounds
                head_ang = scipy.stats.circmean(head_angs, low=-np.pi, high=np.pi)
                self.heading += [head_ang]
            elif method == 'svd':
                raise NotImplementedError("SVD heading method is not yet implemented.")
            # extra: measure the wing 
            if wings:
                # get the wing ring values
                wing_vals = frame[self.wing_ring_coords[:, 1], self.wing_ring_coords[:, 0]]
                self.wing_vals += [wing_vals]
                # test: plot the outer ring colored by the values and the predicted head position
                outer_vals = frame[self.outer_ring_coords[:, 1], self.outer_ring_coords[:, 0]]
                plt.imshow(frame, cmap='gray')
                plt.scatter(self.inner_ring_coords[:, 0], self.inner_ring_coords[:, 1], c=self.inner_ring_angles, alpha=.25)
                plt.colorbar()
                # test: use the heading angle to reduce the search for the wings
                plt.imshow(frame, cmap='gray')  
                plt.scatter(self.wing_ring_coords[:, 0], self.wing_ring_coords[:, 1], c=wing_vals)
                # rotate the wing angles based on the heading angle
                new_angs = self.wing_ring_angles - self.heading[-1]
                new_angs %= 2 * np.pi
                # negative angles are the 
                # plot the wing coords, colored by their wing angle
                plt.imshow(frame)
                plt.scatter(self.wing_ring_coords[:, 0], self.wing_ring_coords[:, 1], c=self.wing_ring_angles)
                # plot the heading position
                xc, yc = self.center
                plt.scatter(heading_pos[0] + xc, heading_pos[1] + yc, c='r', s=100)
                plt.gca().set_aspect('equal')
                plt.show()
                # plot a vector using the heading angle
                # plt.figure()
            if head:
                pass  # head tracking not yet implemented
            print_progress(num, self.num_frames)
        if isinstance(self.video, io.ffmpeg.FFmpegReader):
            self.video = io.FFmpegReader(self.filename)
        # convert to ndarray
        self.heading = np.array(self.heading)
        if np.any(np.isnan(self.heading)):
            # if np.isnan(self.heading).mean() > .1:
            #     breakpoint()
            # replace nans with linear interpolation
            inds = np.arange(len(self.heading))
            no_nans = np.isnan(self.heading) == False
            f = scipy.interpolate.interp1d(inds[no_nans], self.heading[no_nans], fill_value='extrapolate')
            self.heading[no_nans == False] = f(inds[no_nans == False])
        # check for absurdly fast movements
        self.headings[method] = self.heading

    def video_preview(self, vid_fn=None, relative=False, marker_size=3):
        """Generate video with headings superimposed.


        Parameters
        ----------
        relative : bool, default=False
            Whether to generate the video after removing the fly's motion.
        marker_size : int, default=3
            The side length in pixels of the square marker used to indicate 
            the head.
        """
        if vid_fn is None:
            vid_fn = ".".join(self.filename.split(".")[:-1])
            vid_fn += "_heading.mp4"
        # get new video parameters
        vid_radius = int(3 * self.outer_radius)
        vid_center = np.array([vid_radius, vid_radius])
        # get the indices of the center pixels
        ylow = max(round(self.center[1]) - vid_radius, 0) 
        yhigh = min(round(self.center[1]) + vid_radius, self.height)
        xlow = max(round(self.center[0]) - vid_radius, 0)
        xhigh = min(round(self.center[0]) + vid_radius, self.width)
        # get the sizes for reindexing
        height, width = yhigh - ylow, xhigh - xlow
        new_height, new_width = vid_radius * 2, vid_radius * 2
        ystart = round((new_height - height)/2)
        ystop = ystart + height
        xstart = round((new_width - width)/2)
        xstop = xstart + width
        # crop the video first, to save time
        cropped_video = np.copy(self.video[:, ylow : yhigh, xlow : xhigh])[..., np.newaxis]
        # make an empty frame for temporary storage
        # frame_centered = np.zeros((2 * vid_radius, 2 * vid_radius, 3), dtype=float)
        # open a video and start storing frames
        # new_vid = io.FFmpegWriter(vid_fn)
        # todo: store into a numpy array and use vwrite instead
        new_vid = np.zeros((self.num_frames, 2 * vid_radius, 2 * vid_radius, 3), dtype='uint8')
        # for each frame:
        # for frame, orientation in zip(cropped_video, self.heading):
        for num, (frame, orientation, frame_centered) in enumerate(zip(
                cropped_video, self.heading, new_vid)):
            if not np.isnan(orientation):
                # get the head position
                d_vector = np.array([np.cos(orientation), np.sin(orientation)])
                pos = np.round(vid_center + self.inner_radius * d_vector).astype(int)
                # make a version of the frame with a red square centered at the mean orientation
                frame_centered[ystart:ystop, xstart:xstop] = frame
                # if specified, rotate the frame to get relative motion
                if relative:
                    frame_centered = scipy.ndimage.rotate(
                        frame_centered, (orientation + np.pi/2) * 180 / np.pi,
                        reshape=False)
                # otherwise, draw a line indicating the heading
                else:
                    try:
                        rr, cc, val = skimage.draw.line_aa(vid_radius, vid_radius, pos[0], pos[1])
                    except Exception:
                        raise
                    # scale down the older values
                    old_vals = frame_centered[cc, rr].astype(float)
                    old_vals *= (1 - val)[..., np.newaxis]
                    frame_centered[cc, rr] = old_vals.astype('uint8')
                    frame_centered[cc, rr, 0] = val * 255
                    # frame_centered[cc, rr, 1:] = 10
                    frame_centered[pos[1] - marker_size : pos[1] + marker_size,
                                   pos[0] - marker_size : pos[0] + marker_size] = [155, 0, 0]
                # store the new frame
                # new_vid.writeFrame(frame_centered)
                # clear the temporary frame
                # frame_centered.fill(0)
            print_progress(num, self.num_frames)
        # new_vid.close()
        # use vwrite to store the array as a video
        io.vwrite(vid_fn, new_vid)
        
        print(f"Heading preview saved in {vid_fn}.")

    def extract_ring(self, ring='inner', bins=np.linspace(-np.pi, np.pi, 361)):
        """Bin frame pixels using one of the defined rings.

        Parameters
        ----------
        ring : str, default='inner'
            The ring variable to use for extracting the pixel values regularly.
        bins : np.linspace
            The list of values for partitioning the pixel orientations.
        """
        # grab the pertinent ring parameters
        angles, coords = self.__getattribute__(f"{ring}_ring_angles"), self.__getattribute__(f"{ring}_ring_coords")
        num_bins = len(bins) - 1
        # make empty array for storing the final graph
        vals = np.zeros((self.num_frames, num_bins), dtype='uint8')
        # group the lists of angles by the defined bins
        bin_groups = np.digitize(angles, bins)
        # get the video pixel values within these bins
        if isinstance(self.video, np.ndarray):
            in_ring = self.video[:, coords[:, 1], coords[:, 0]]
        else:
            in_ring = []
            self.video = io.FFmpegReader(self.filename)
            for frame in self.video: in_ring += [frame[coords[:, 1], coords[:, 0]]]
            in_ring = np.array(in_ring)
        # sort the pixel values using the bin group labels
        order = np.argsort(bin_groups)
        in_ring = in_ring[:, order]
        bin_groups = bin_groups[order]
        changes = np.diff(bin_groups) > 0
        changes = np.where(changes)[0]
        # get binned pixel values
        bin_vals = np.split(in_ring, changes, axis=1)
        bin_vals = np.array([bin_val.mean(1) for bin_val in bin_vals])
        # return the binned values
        return bin_vals

    def graph_preview(self, bins=np.linspace(-np.pi, np.pi, 361), wings=False):
        """A Space X Time graph of inner, outer, and inner-outer rings


        Parameters
        ----------
        bins : array-like, default=np.linspace(-np.pi, np.pi, 361)
            Define the bounds of bins used for flattening the ring values.
        wings : bool, default=False
            Whether to also plot the flattened wing data.
        """
        num_bins = len(bins) - 1
        # store one space X time graph per ring
        graphs = []
        # for each ring:
        rings = ['inner', 'outer']
        if wings:
            rings += ['wing']
        # bin pixel values using the ring extraction function
        for ring in rings:
            graphs += [self.extract_ring(ring, bins=bins)]
        # ring_angles = [self.small_ring_angles, self.large_ring_angles]
        # ring_coords = [self.small_ring_coords, self.large_ring_coords]
        # if wings:
        #     ring_angles += [self.wing_ring_angles]
        #     ring_coords += [self.wing_ring_coords]
        # for angles, coords in zip(ring_angles, ring_coords):
        #     # make empty array for storing the final graph
        #     vals = np.zeros((self.num_frames, num_bins), dtype='uint8')
        #     # group the lists of angles by the defined bins
        #     bin_groups = np.digitize(angles, bins)
        #     # get the video pixel values within these bins
        #     in_ring = self.video[:, coords[:, 1], coords[:, 0]]
        #     # sort the pixel values using the bin group labels
        #     order = np.argsort(bin_groups)
        #     in_ring = in_ring[:, order]
        #     bin_groups = bin_groups[order]
        #     changes = np.diff(bin_groups) > 0
        #     changes = np.where(changes)[0]
        #     # get binned pixel values
        #     bin_vals = np.split(in_ring, changes, axis=1)
        #     bin_vals = np.array([bin_val.mean(1) for bin_val in bin_vals])
        #     # store
        #     graphs += [bin_vals]
        # now graph:
        fig, axes = plt.subplots(ncols=len(graphs), sharey=True, sharex=True)
        # time_bins = np.append([0], self.times)
        time_bins = self.times
        for num, (ax, graph, ring) in enumerate(zip(axes, graphs, rings)):
            if graph.ndim > 2 and graph.shape[-1] > 1:
                ax.pcolormesh(bins, time_bins, graph[..., 0].T)
            else:
                ax.pcolormesh(bins, time_bins, graph.T)
            # line plot excluding steps > np.pi
            for (key, val), color in zip(self.headings.items(), [red, green, blue]):
                headings = np.copy(val)
                speed = np.append([0], np.diff(headings))
                headings[abs(speed) > np.pi] = np.nan
                ax.plot(headings, self.times, color=color, label=key)
                ax.set_title(f"{ring.capitalize()} Ring")
        # 1. the outer ring
        # axes[0].pcolormesh(bins, time_bins, graphs[0].T)
        # axes[0].scatter(self.headings, self.times, color=red, marker='.')
        # line plot excluding steps > np.pi
        # for (key, val), color in zip(self.headings.items(), [red, green, blue]):
        #     headings = np.copy(val)
        #     speed = np.append([0], np.diff(headings))
        #     headings[abs(speed) > np.pi] = np.nan
        #     axes[0].plot(headings, self.times, color=color, label=key)
        # axes[0].invert_yaxis()
        # axes[0].set_title("Inner Ring")
        # 2. the inner ring
        # axes[1].pcolormesh(bins, time_bins, graphs[1].T)
        # axes[1].set_title("Outer Ring")
        # if wings:
            # 3. the wing ring
            # axes[2].pcolormesh(bins, time_bins, graphs[2].T)
            # axes[2].set_title("Wing Ring")
        axes[-1].invert_yaxis()
        axes[-1].set_xticks(
            [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
            [r"-$\pi$", r"-$\pi$/2", r"0", r"$\pi$/2", r"$\pi$"]
        )
        # format
        plt.tight_layout()
        plt.show()

class OfflineTracker():
    def __init__(self, folder="./", vid_extension='.mp4'):
        """Track fly headings for each video and store.

        Parameters
        ----------
        folder : path, default='./'
            The path of the folder to process.
        vid_extension : str, default='.mp4'
            The extensions of the videos we intend to process.
        """
        self.folder = folder
        self.vid_extension = vid_extension
        # get the full path of each element in folder
        self.fns = os.listdir(self.folder)
        self.fns = [os.path.join(self.folder, fn) for fn in self.fns]
        # get all of the relevant files
        self.vid_fns = [fn for fn in self.fns if fn.endswith(self.vid_extension)]
        self.h5_fns = [fn for fn in self.fns if fn.endswith(".h5")]
        # find all videos that have an associated h5 file
        new_vid_fns = []
        new_h5_fns = []
        for vid_fn in self.vid_fns:
            h5_fn = vid_fn.replace(self.vid_extension, '.h5')
            if h5_fn in self.h5_fns:
                new_vid_fns += [vid_fn]
                new_h5_fns += [h5_fn]
        self.vid_fns = new_vid_fns
        self.h5_fns = new_h5_fns

    def process_vids(self, start_over=False, method='combined', wings=False, head=False, gui=False, display=False):
        """For each video and h5 file, track and store the heading data.

        Parameters
        ----------
        start_over : bool, default=False
            Whether to check if the offline dataset has already been saved.
        method : str, options=['combined', 'rings', 'svd', 'fourier_mellin']
            The method of measuring the
        wings : bool, default=False
            Whether to measure the left and right wingbeat amplitudes.
        head : bool, default=False
            Whether to measure the head orientation.
        gui : bool, default=False
            Whether to use a GUI to manually define the ring and threshold parameters.
        display : bool, default=False
            Whether to display the results of each video.
        """
        for vid_fn, h5_fn in zip(self.vid_fns, self.h5_fns):
            # load the dataset, which specifies the inner and outer radii for
            # tracking
            data = TrackingTrial(h5_fn)
            # only continue if 1) this hasn't been run before or start_over
            # is set to True, 2) 'end_test' times were specified, and 3) the
            # dataset loaded successfully
            already_processed = 'camera_heading_offline' in dir(data)
            stop_specified = 'stop_test' in dir(data)
            success = not already_processed and stop_specified and data.load_success
            if success or start_over:
                # load the video and process the fly's heading
                track = TrackingVideo(vid_fn)
                if method == 'combined':
                    # reset the rings and store in the database
                    if gui:
                        track.ring_gui()
                        inner_r, outer_r, threshold = track.inner_radius, track.outer_radius, track.threshold
                        for val, lbl in zip([inner_r, outer_r, threshold], ['inner_r', 'outer_r', 'thresh']):
                            data.add_attr(lbl, val)
        for vid_fn, h5_fn in zip(self.vid_fns, self.h5_fns):
            # load the dataset, which specifies the inner and outer radii for
            # tracking
            data = TrackingTrial(h5_fn)
            # continue if 1) this hasn't been run before or start_over
            # is set to True, 2) 'end_test' times were specified, and 3) the
            # dataset loaded successfully
            already_processed = 'camera_heading_offline' in dir(data)
            stop_specified = 'stop_test' in dir(data)
            success = (not already_processed or start_over) and stop_specified and data.load_success
            if success or start_over:
                # load the video and process the fly's heading
                track = TrackingVideo(vid_fn)
                if method == 'combined':
                    # reset the rings 
                    inner_r, outer_r, threshold = data.inner_r, data.outer_r, data.thresh
                    # set the ring parameters from our dataset
                    center = (track.width/2, track.height/2)
                    track.set_rings(center, inner_radius=inner_r, outer_radius=outer_r, thickness=5)
                    track.get_heading(floor=0, ceiling=threshold, wings=wings, method='combined', head=head)
                    # why is the offline tracking so much longer than the online tracking?
                    # note: I figured out why. the framerate of online tracking is determined by the 
                    # framerate of the stimulus. this must have been limited to 60 Hz, for some reason, and
                    # the camera must have been set to 120 Hz, so there was a 2-fold difference in the time 
                    # course of the responses. Also, NaNs resulted in a lot of missing data due to the unwrap 
                    # function, not actually missing data. The online tracking can be re-calculated almost 
                    # perfectly by placing both on a regular timeline using the start_test and start_exp 
                    # attributes. So, we can pretty safely split up the offline heading data via the 
                    # following algorithm:
                    # 1) apply a timestamp to each frame of the offline heading
                    # 2) figure out the frame ratio between the duration of offline (k) and the online heading (j; k/j should be an integer)
                    # 3) based on the length of the online heading tests, calculate the length of the offline tests, k
                    # 4) generate a new dataset by grabbing the k values before each keyframe
                    # plot the offline tracking with timing based on the start and stop of the experiment
                    # heading_offline
                    # time_offline = np.linspace(data.start_exp, data.stop_exp, len(heading_offline))
                    # note: offline and online headings are off by about pi
                    # plt.plot(time_offline, np.unwrap((heading_offline-np.pi)%2*np.pi - np.pi))
                    # for kf, heading in zip(data.stop_test, data.camera_heading): plt.axvline(kf); ts = np.arange(len(heading)) / 120.; plt.plot(kf-2*ts[::-1], np.unwrap(heading))
                    if display:
                        track.graph_preview()
                    # todo: add tracking for the head
                    # the method below assumes that the recorded framerates for the offline and online
                    # heading are correct, but it looks like sometimes the holocube framerate says 120 
                    # but was actually 60 Hz
                    # get times associated with each frame
                    # time = track.times / data.framerate
                    # the total duration is the same between the two
                    # time_offline = np.linspace(0, data.time.max(), len(track.times))
                    heading_offline = track.headings['combined']
                    # time_online = data.query('time', sort_by='time')
                    # heading_online = data.query('camera_heading', sort_by='time')
                    heading_online = data.camera_heading
                    num_tests, num_frames = heading_online.shape
                    duration = data.stop_exp - data.start_exp
                    fps_offline = heading_offline.size / duration
                    fps_online = heading_online.size / duration
                    frame_ratio = fps_offline / fps_online
                    # frame_ratio = np.round(data.framerate / data.holocube_framerate)
                    num_frames_offline = int(round(frame_ratio * num_frames))
                    heading_offline_arr = np.zeros((num_tests, num_frames_offline), dtype=float)
                    total_frames = num_tests * frame_ratio * num_frames
                    extra_frames = heading_offline.shape[0] - total_frames
                    frame_padding = int(round(extra_frames / num_tests-1))
                    start, stop = data.start_exp, data.stop_exp
                    stops = data.stop_test
                    times_offline = np.linspace(start, stop, len(heading_offline) + 1)[:-1]
                    for num, (storage, stop) in enumerate(
                            zip(heading_offline_arr, stops)):
                        # 1. find the nearest frame to the stop point
                        frame_num = np.argmin(abs(times_offline - stop))
                        # 2. get the clip before this stop point
                        # account for slight differences in the length of each array
                        max_x = min(frame_num, storage.shape[0])
                        storage[- max_x:] = heading_offline[frame_num - max_x:frame_num]
                    heading_offline_arr -= np.pi
                    heading_offline_arr[heading_offline_arr < -np.pi] += 2* np.pi
                    heading_offline_arr[heading_offline_arr > np.pi] -= 2* np.pi
                elif method == 'svd':
                    raise NotImplementedError("SVD heading method is not yet implemented.")
                data.add_dataset('camera_heading_offline', heading_offline_arr)
                # data.add_dataset('com_thrust', self.thrust)
                if wings:
                    data.add_dataset('wing_vals', self.wing_vals)
                # add a time
                time_online = data.query('time', sort_by='test_ind')
                time_offline = np.linspace(
                    0, time_online.max(), heading_offline_arr.size).reshape(
                    heading_offline_arr.shape)
                data.add_dataset('time_offline', time_offline)
                print(f"updated {h5_fn}")
                # check if the online and offline headings are close
                # fig, ax = plt.subplots()
                # times_online = np.linspace(0, data.stop_exp - data.start_exp, heading_online.size)
                # times_offline = np.linspace(0, data.stop_exp - data.start_exp, heading_offline_arr.size)
                # ax.plot(times_online, heading_online.flatten(), color='k', zorder=2)
                # ax.plot(times_offline, heading_offline_arr.flatten(), linestyle="", marker='.', color=red, zorder=3)
                # plt.show()
                # breakpoint()

    def offline_comparison(self, smooth_offline=False):
        """Compare the offline from the online heading measurements.

         Measure the spatial and temporal performance of the online measurement
         in contrast to the more detailed offline measurement.
        """
        fig_summary, axes_summary = plt.subplots(
            nrows=3, ncols=2, gridspec_kw={'width_ratios': [5, 1]})
        # 1. demo individual traces and plot superimpose the bar trajectory
        demo_exp_num = 0
        demo_trial_num = 0
        demo_start_time = 114
        demo_stop_time = 132
        summary_lags = []
        summary_corrs = []
        summary_corr_TS = []
        for num, (vid_fn, h5_fn) in enumerate(zip(self.vid_fns, self.h5_fns)):
            # load the dataset, which specifies the inner and outer radii for
            # tracking
            data = TrackingTrial(h5_fn)
            if 'camera_heading_offline' in dir(data) and 'time' in dir(data):
                # make a figure with two equal rows with one wide and one narrow column
                # fig, axes = plt.subplots(nrows=3, ncols=2, gridspec_kw={'width_ratios':[5, 1]})
                # top row: plot the two time series in the top row
                # top_row, middle_row, bottom_row = axes
                # ts_ax, corr_ax = top_row
                sum_ts_ax, sum_corr_ax = axes_summary[0]
                time_online = data.query('time', sort_by='test_ind')
                # plot the offline
                if smooth_offline:
                    data.butterworth_filter('camera_heading_offline', 0, 10, sample_rate=data.framerate)
                    heading_offline = data.query('camera_heading_offline_smoothed',
                                                sort_by='test_ind')
                    data.butterworth_filter('camera_heading', 0, 10,
                                            sample_rate=data.holocube_framerate)
                    heading_online = data.query('camera_heading_smoothed',
                                                   sort_by='test_ind')

                else:
                    heading_offline = data.query('camera_heading_offline',
                                                    sort_by='test_ind')
                heading_online = data.query('camera_heading',
                                               sort_by='test_ind')
                time_offline = data.query('time_offline',
                                             sort_by='test_ind')
                # diff = heading_offline.flatten()[0] - heading_online.flatten()[0]
                # heading_online = np.unwrap(heading_online.flatten())
                # offset = heading_offline[0] - heading_online[0]
                # offset = diff - offset
                # ts_ax.plot(time_online.flatten(), heading_online.flatten(), color='k', alpha=1, lw=.5)
                # ts_ax.plot(time_offline.flatten(), heading_offline.flatten(), color='lightgray', zorder=1)
                # format
                # ts_ax.set_xticks([])
                # ts_ax.set_yticks([-np.pi, 0, np.pi], ["-$\pi$", "0", "$\pi$"])
                # sbn.despine(ax=ts_ax, bottom=True)
                # right axis: plot the cross-correlation of offline and online heading measurements
                # calculate the cross-correlation of each test, with online signal resampled to match
                # the length of the offline signal
                corrs = []
                peak_lags = []
                peak_corrs = []
                num_frames = heading_offline.shape[-1]
                lags = scipy.signal.correlation_lags(num_frames, num_frames,
                                               mode='same').astype(float)
                lags /= data.framerate
                for test_online, test_offline in zip(heading_online, heading_offline):
                    # interpolate points in online to match offline heading
                    online_resampled = np.repeat(test_online, 4)
                    _, corr = normal_correlate(
                        np.unwrap(test_offline), np.unwrap(online_resampled),
                        mode='same', circular=False)
                    corrs += [corr]
                    no_nans = np.isnan(test_offline)
                    no_nans += np.isnan(online_resampled)
                    no_nans = no_nans == False
                    # corr_ax.plot(lags, corr, lw=.5, color='k', alpha=.25)
                    # get peak maximum
                    pos_lags = lags >= 0
                    max_ind = np.argmax(corr[pos_lags])
                    peak_lags += [lags[pos_lags][max_ind]]
                    peak_corrs += [corr[pos_lags][max_ind]]
                # highlight the range of peaks
                # corr_ax.scatter(peak_lags, peak_corrs, color=red, alpha=.25,
                #                 edgecolors='none')
                # corr_ax.set_xlim(-.5, 1)
                # plot the mean cross-correlation
                mean_corr = np.nanmean(np.array(corrs), axis=0)
                summary_corr_TS += [mean_corr]
                max_ind = np.argmax(mean_corr[pos_lags])
                summary_corrs += [mean_corr[pos_lags][max_ind]]
                summary_lags += [lags[pos_lags][max_ind]]
                # corr_ax.plot(lags, mean_corr, lw=1, alpha=1, color='k')
                # sum_corr_ax.plot(lags, mean_corr, lw=.5, alpha=.25, color='k')
                # add text showing the median +/- IQR of the pearson correlations
                low, mid, high = np.percentile(peak_corrs, [25, 50, 75])
                IQR = high - low
                # corr_ax.text(.5, 1., f"{mid:.2}+/-{IQR:.5}")
                if num == 0:
                    # plot the offline and online headings
                    sum_ts_ax.plot(time_online.flatten(), heading_online.flatten(), color='k', alpha=1, lw=.5)
                    sum_ts_ax.plot(time_offline.flatten(), heading_offline.flatten(), color='lightgray', zorder=1)
                    # zoom into demo region
                    sum_ts_ax.set_xlim(demo_start_time, demo_stop_time)
                    sum_ts_ax.set_ylim(-np.pi/2, np.pi/2)
                    # format
                    sum_ts_ax.set_xticks([])
                    sum_ts_ax.set_yticks(
                        [-np.pi/2, 0, np.pi/2],
                        [r"-$\dfrac{\pi}{2}$", r"0", r"$\dfrac{\pi}{2}$"]
                    )
                    sum_ts_ax.set_ylabel("Heading")
                    sbn.despine(ax=sum_ts_ax, bottom=True, trim=True)
                    # plot the velocity time series below position
                    sum_velo_ax, sum_velo_hist_ax = axes_summary[1]
                    velo_offline = np.diff(np.unwrap(heading_offline).flatten())
                    velo_online = np.diff(np.unwrap(heading_online).flatten())
                    sum_velo_ax.plot(
                        time_online.flatten()[1:], velo_online, color='k',
                        alpha=1, lw=.5)
                    sum_velo_ax.plot(
                        time_offline.flatten()[1:], velo_offline,
                        color='lightgray', zorder=1)
                    # zoom into demo region
                    sum_velo_ax.set_xlim(demo_start_time, demo_stop_time)
                # plot the cross-correlation
                sum_corr_ax.plot(lags, mean_corr, lw=.5, alpha=.25, color='k')

        # format
        sum_corr_ax.set_ylim(0, 1.1)
        sum_corr_ax.set_yticks([0, .5, 1])
        sum_corr_ax.set_ylabel("Correlation")
        sum_corr_ax.set_xlim(-.1, .6)
        sum_corr_ax.set_xticks([0, .5])
        sbn.despine(ax=sum_corr_ax, bottom=False, trim=True)

                # right axis: plot the distribution of velocities
                # bottom row: plot the residual time series in the middle row

                # plot two insets: zoom into the a) highest and b) lowest speed
                # in the right margin, plot the histogram of residuals
        # scatterplot the peak correlations and lags
        sum_corr_ax.scatter(summary_lags, summary_corrs, color=red, alpha=.25,
                            edgecolors='none')
        sum_corr_ax.plot(lags, np.nanmean(summary_corr_TS, axis=0), color='k', alpha=1)
        plt.tight_layout()
        plt.show()

        # 2. plot individual residual traces


class TrackingExperiment():
    def __init__(self, dirname, remove_incompletes=True, **trial_kwargs):
        """Load all H5 TrackingTrial files from the same experiment.

        Parameters
        ----------
        dirname : path, or list
            The directory to load or a list of h5 directories to load.
        remove_incompletes : bool, default=True
            Whether to keep trials with missing tests.
        """
        self.dirname = dirname
        if isinstance(dirname, str):
            self.dirname = [dirname]
        self.files = []
        for dirname in self.dirname:
            fns = os.listdir(dirname)
            self.files += [os.path.join(dirname, fn) for fn in fns]
        self.h5_files = [file for file in self.files if file.endswith(".h5")]
        # load a TrackingTrial for each h5 file
        self.trials = []
        for fn in self.h5_files:
            trial = TrackingTrial(fn, **trial_kwargs)
            if trial.load_success:
                try:
                    self.trials += [trial]
                except:
                    print('here')
        if remove_incompletes:
            self.remove_incompletes()

    def query(self, object='trial', same_size=True, skip_empty=False, **kwargs):
        """Collect data from every trial and return a combined result.

        Parameters
        ----------
        object : {'trial', 'saccade', 'bout'}, default ``'trial'``
            What to query from each trial.  ``'saccade'`` delegates to
            :meth:`TrackingTrial.query` with ``object='saccade'`` and
            requires :meth:`detect_saccades` to have been called first.
        same_size : bool, default True
            Only applies when ``object='trial'``.  When True and trials have
            different lengths, shorter arrays are padded (NaN / 0) to the
            maximum length so the result is a single ``ndarray``.
        skip_empty : bool, default False
            Omit trials that return empty arrays.
        **kwargs
            Forwarded to each trial's ``query()`` call.  Key kwargs:

            output : str
                Variable name to retrieve (``object='trial'``) or saccade
                column / ``'saccade'`` (``object='saccade'``).
            subset : dict, optional
                Filter conditions.  Scalar, membership list, inequality string
                (e.g. ``'>0.5'``), or NaN matching are all supported.
            sort_by : str, default ``'test_ind'``
                Variable to sort results by.
            groupby : {'saccade', 'test', 'trial'}, optional
                Granularity for ``object='saccade'``.
            agg_func : callable, optional
                Reduction applied per group for ``groupby='test'/'trial'``.

        Returns
        -------
        ndarray or list
            For ``object='trial'`` with uniform shapes: ``ndarray`` of shape
            ``(n_trials, n_tests[, n_frames])``.  Otherwise a list.
            For ``object='saccade'``: list of values / :class:`Saccade` objects.

        Examples
        --------
        Get heading time-series for all trials::

            data = exp.query(output='camera_heading')
            # → ndarray shape (n_trials, n_tests, n_frames)

        Filter by condition and group saccades per test::

            amps = exp.query(
                object='saccade', output='amplitude',
                groupby='test',
                subset={'condition': 1, 'peak_velocity': '>5'},
            )
        """
        if 'subset' not in kwargs.keys():
            kwargs['subset'] = {}

        if object == 'saccade':
            groupby = kwargs.get('groupby', 'saccade')
            ret = []
            for trial in self.trials:
                try:
                    res = trial.query(object='saccade', **kwargs)
                except RuntimeError:
                    # No saccade table on this trial — skip silently.
                    continue
                if skip_empty and len(res) == 0:
                    continue
                ret.append(res)
            # Concatenate across trials.
            if groupby in ('saccade', 'test'):
                # Each trial returns a list; flatten into one list.
                flat = []
                for trial_res in ret:
                    flat.extend(trial_res)
                return flat
            else:
                # groupby='trial': one value per trial — return as a list.
                return ret

        # --- trial / bout branch ---
        ret = []
        for trial in self.trials:
            if object == 'bout':
                res = trial.query_bouts(**kwargs)
            else:
                res = trial.query(**kwargs)
            if skip_empty:
                if len(res) > 0:
                    ret += [res]
            else:
                ret += [res]
        if object == 'trial':
            if len(ret) > 0:
                # check if all the results have the same shape
                first_shape = ret[0].shape
                same_shape = [trial_data.shape == first_shape for trial_data in ret]
                # return as an array if they are all the same shape
                if np.all(same_shape):
                    ret = np.array(ret)
                elif same_size:
                    shapes = [arr.shape for arr in ret]
                    sizes = np.array([arr.size for arr in ret])
                    max_shape = shapes[np.argmax(sizes)]
                    new_ret = []
                    dtype = [arr.dtype for arr in ret if len(arr) > 0][0]
                    empty = np.zeros(max_shape, dtype=dtype)
                    for arr in ret:
                        if 'float' in str(dtype):
                            empty.fill(np.nan)
                        elif '<U' in str(dtype):
                            empty.fill('')
                        else:
                            empty.fill(0)
                        if arr.ndim > 1:
                            for vals, storage in zip(arr, empty):
                                storage[:len(vals)] = vals
                        else:
                            if empty.ndim == 2:
                                empty[:, :len(arr)] = arr
                            else:
                                empty[:len(arr)] = arr
                        new_ret += [np.copy(empty)]
                    ret = np.array(new_ret)
                    # find the maximum shape and pad the others with NaNs to match
            else:
                ret = []
        return ret

    def query_bouts(self, **kwargs):
        """Get the bouts from each trial."""
        return self.query(object='bout', **kwargs)
        # ret = []
        # for trial in self.trials:
        #     ret += [trial.query_bouts(**kwargs)]
        # return ret

    def query_saccades(self, **kwargs):
        """Get saccade data from each trial."""
        return self.query(object='saccade', **kwargs)
        # ret = []
        # for trial in self.trials:
        #     ret += [trial.query_saccades(**kwargs)]
        # return ret

    def detect_saccades(self, key='camera_heading', threshold_speed=200, **find_peaks_kwargs):
        """Detect saccades for every trial in the experiment.

        Calls :meth:`TrackingTrial.detect_saccades` on each trial with the
        given arguments.  Does **not** call ``save()``.

        Parameters
        ----------
        key : str, default='camera_heading'
            The heading variable to use for detection.
        threshold_speed : float, default=350
            Minimum peak speed (degrees/s) required to keep a saccade.
        **find_peaks_kwargs
            Forwarded to :func:`_detect_saccades` (distance, width, prominence, wlen).

        Examples
        --------
        ::

            exp.detect_saccades()          # run on every trial
            exp.save()                     # persist all saccade tables

            # Retrieve all amplitudes across the experiment
            amps = exp.query(object='saccade', output='amplitude')
        """
        for trial in self.trials:
            trial.detect_saccades(key=key, threshold_speed=threshold_speed, **find_peaks_kwargs)

    def saccade_table_df(self, extra_cols=None, time_ind='start_frame'):
        """Return saccade data from all trials as a single :class:`pandas.DataFrame`.

        Always includes a ``trial_filename`` column so rows can be traced back
        to their source trial.

        Parameters
        ----------
        extra_cols : list[str] or None
            Passed to each ``TrackingTrial.saccade_table_df()``.
            Scalar / 1-D test-level / 2-D (test × frame) variables are all
            supported — see that method's docstring for details.
        time_ind : {'start_frame', 'stop_frame', 'peak_frame'}, default='start_frame'
            Passed to each ``TrackingTrial.saccade_table_df()``.

        Returns
        -------
        pandas.DataFrame
            One row per saccade across all trials, with a leading
            ``trial_filename`` column.

        Examples
        --------
        ::

            exp.detect_saccades()
            df = exp.saccade_table_df(extra_cols=['condition', 'fly_id'])
            print(df.head())
            # trial_filename  test_ind  amplitude  peak_velocity  ...  condition  fly_id
        """
        import pandas as pd
        frames = []
        for trial in self.trials:
            try:
                df = trial.saccade_table_df(extra_cols=extra_cols, time_ind=time_ind)
            except RuntimeError:
                continue
            if df.empty:
                continue
            df.insert(0, 'trial_filename', os.path.basename(trial.filename))
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def add_dataset(self, name, vals):
        """Add a per-trial dataset to every trial in the experiment.

        Parameters
        ----------
        name : str
            Dataset name.  Must be a valid Python identifier and must not
            shadow an existing trial attribute.
        vals : sequence of array-like, length == len(self.trials)
            One array per trial.  Each element is forwarded to
            :meth:`TrackingTrial.add_dataset` for that trial.

        Examples
        --------
        ::

            smoothed = [butterworth(t.query('camera_heading')) for t in exp.trials]
            exp.add_dataset('camera_heading_smooth', smoothed)
            exp.save()
        """
        for trial, arr in zip(self.trials, vals):
            trial.add_dataset(name, arr)

    def add_attr(self, name, vals):
        """Add a scalar metadata attribute to every trial in the experiment.

        Parameters
        ----------
        name : str
            Attribute name.
        vals : sequence, length == len(self.trials)
            One scalar value per trial, forwarded to
            :meth:`TrackingTrial.add_attr`.

        Examples
        --------
        ::

            exp.add_attr('experimenter', ['Alice'] * len(exp.trials))
            exp.save()
        """
        for trial, arr in zip(self.trials, vals):
            trial.add_attr(name, arr)

    def center_initial(self, **kwargs):
        """Center the initial position for all tests of all trials."""
        for trial in self.trials:
            trial.center_initial(**kwargs)

    def unwrap(self, **kwargs):
        """Center the initial position for all tests of all trials."""
        for trial in self.trials:
            trial.unwrap(**kwargs)

    def remove_incompletes(self):
        """Use only trials with the maximum number of tests."""
        new_trials = []
        # find max number of tests
        try:
            max_tests = max([trial.num_tests for trial in self.trials])
        except:
            print("what's up with the num_tests attr for each trial? Oh, I see. The trials aren't importing at all.")
        for trial in self.trials:
            if trial.num_tests == max_tests:
                new_trials += [trial]
        self.trials = new_trials

    def remove_too_fast(self, variable='camera_heading', speed_limit=np.pi/2, tolerance=.01, replace=True):
        """Use only trials with speeds less than the limit.
        
        This is good for getting rid of trials with errors resulting in head-tail inversions.

        Parameters
        ----------
        variable : str, default='camera_heading'
            The variable used to apply this speed limit.
        speed_limit : float, default=pi/4
            The maximum allowable speed.
        tolerance : float in [0, 1], default=.01
            Keep trials with this proportion of frames that are too fast.
        replace : bool, default=True
            Whether to replace the original trials with the interpolated values.
        """
        new_trials = []
        for trial in self.trials:
            vals = trial.query(variable, sort_by='test_ind')
            # pad with zeros for the first frame
            num_trials = vals.shape[0]
            vals_diffs = np.diff(vals, axis=-1)
            vals_diffs = np.concatenate([np.zeros((num_trials, 1)), vals_diffs], axis=-1)
            speed = np.abs(vals_diffs)
            too_fast = speed > speed_limit
            if replace and np.any(too_fast):
                offset = np.round(vals_diffs[too_fast]/np.pi) * np.pi 
                vals_diffs[too_fast] -= offset
                vals_new = np.cumsum(vals_diffs, axis=-1)
                vals_new[0] += vals[..., 0]
                if vals_new.shape[1] != trial.num_frames:
                    print("why?")
                # replace the dataset
                trial.add_dataset(variable, vals_new)
                # recalculate the too_fast variable
                speed = np.abs(np.diff(vals_new, axis=-1))
                too_fast = speed > speed_limit
            if not np.any(too_fast.mean(-1) > tolerance):
                new_trials += [trial]
        self.trials = new_trials

    def remove_nonrandoms(self, variable, invert=False):
        """Use only trials where variable is in increasing order.

        Parameters
        ----------
        variable: str
            The name of the variable that should be in ascending order.
        invert : bool, default=False
            Whether to instead remove only randomized trials.
        """
        new_trials = []
        for trial in self.trials:
            vals = getattr(trial, variable)
            if np.all(np.diff(vals) >= 0):
                new_trials += [trial]
        self.trials = new_trials

    def remove_subjects_w_nans(self, variable, thresh=0):
        """Use only trials without nans.

        Parameters
        ----------
        variable: str
            The name of the variable that should have no nans.
        thresh: float, default=0
            The maximum proportion of nans allowed. Must be between 0 and 1.
        """
        new_trials = []
        for trial in self.trials:
            vals = getattr(trial, variable)
            if np.mean(np.isnan(vals)) <= thresh:
                new_trials += [trial]
        self.trials = new_trials

    def remove_still_subjects(self, variable, thresh=10):
        """Use only trials where variable is .

        Parameters
        ----------
        variable: str
            The name of the variable that should have no nans.
        thresh: float, default=10
            The minimum tolerable number of unique values.
        """
        new_trials = []
        for trial in self.trials:
            vals = getattr(trial, variable)
            if len(np.unique(vals)) > thresh:
                new_trials += [trial]
        self.trials = new_trials

    def get_saccade_stats(self, **kwargs):
        """Process individual saccades and measure saccade data per trial."""
        print_progress(0, len(self.trials))
        for num, trial in enumerate(self.trials):
            trial.get_saccade_stats(**kwargs)
            print_progress(num, len(self.trials))


    def remove_saccades(self, **saccade_kwargs):
        """Generate a new instance of the """
        for trial in self.trials:
            trial.remove_saccades(**saccade_kwargs)


    def butterworth_filter(self, low=1, high=6, key='camera_heading',
                           sample_rate=60.):
        """Apply a Butterworth filter to the dataset of each trial.

        Parameters
        ----------
        low : float, default=1
            The lower bound for filtering the specified dataset
        high: float, default=6
            The upper bound for filtering the specified dataset.
        key : str, default='camera_heading'
            The variable to filter.
        sample_rate : float, default=60
            The sample rate used for calulating frequencies.
        """
        for trial in self.trials:
            trial.butterworth_filter(key, low, high, sample_rate)
            trial.__setattr__(key, trial.__getattribute__(key+"_smoothed"))

    def plot(self, xvar, yvar=None, col_var=None, row_var=None,
             plot_type='line', object='trial',
             row_cmap=None, col_cmap=None, color='k',
             right_margin=True, bottom_margin=True,
             xlim=None, ylim=None, xticks=None, yticks=None,
             logx=False, logy=False, display=None,
             summary_func=np.nanmean, xlabel=None, ylabel=None,
             scale=1.5, bins=100, margin_bins=None, right_margin_bins=None, bottom_margin_bins=None, density=False, probability=False,
             omit_wrap=True, confidence_interval=False, confidence=0.84,
             n_boot=1000, groupby=None, agg_func=None, margin_groupby=None,
             positive_amplitude=False, min_speed=None, max_speed=None,
             show_n=False, relative_to='start',
             rad2deg=False,
             right_margin_xlim=None, right_margin_ylim=None,
             bottom_margin_xlim=None, bottom_margin_ylim=None,
             right_margin_yticks=None, bottom_margin_yticks=None,
             subplot_size=None, right_margin_ratio=1.0, bottom_margin_ratio=1.0,
             compare_subsets=None, compare_labels=None,
             compare_xlabel=None, right_margin_xlabel=None, bottom_margin_xlabel=None,
             pair_by='fly_id', margin_mode='diff_of_diffs',
             return_stats=False,
             plot_kwargs=None, **query_kwargs):
        """Unified grid-plot entry point for TrackingExperiment.

        Draws a grid of subplots parameterised by ``col_var`` and ``row_var``,
        with optional margin summary axes on the right and bottom.  All querying,
        color resolution, and grid layout happen internally; the caller only
        needs to specify what to plot and how.

        Parameters
        ----------
        xvar : str
            Variable for each panel's x-axis.  For
            ``object='saccade', plot_type='line'`` this must be a time-series
            attribute of :class:`Saccade` (e.g. ``'arr_relative'``).
        yvar : str or None
            Variable for each panel's y-axis.  Not used by
            ``plot_type='trajectory2d'``; ignored by ``plot_type='histogram'``.
        col_var, row_var : str or None
            Variables that parameterise the grid columns and rows.  Pass
            ``None`` for a single column or row.
        plot_type : {'line', 'hist2d', 'trajectory2d', 'histogram', 'scatter', 'pdf'}
            Drawing style per panel.
        object : {'trial', 'saccade'}
            Data source.  ``'saccade'`` requires :meth:`detect_saccades` to
            have been called on every trial first.
        row_cmap, col_cmap : colormap or list, optional
            Colormap for rows / columns.  Forwarded to :func:`resolve_colors`.
            Accepts a named matplotlib colormap string, a callable that takes a
            value in [0, 1], or an explicit list of colors.
        color : color spec, default ``'k'``
            Fallback colour when neither cmap is given.
        right_margin, bottom_margin : bool | str | list[str]
            Margin configuration.  ``True`` → smart default per ``plot_type``;
            ``False`` → disabled; a string or list of strings selects explicit
            margin plot type(s): ``'line'``, ``'histogram'``, ``'scatter'``,
            ``'circ_hist'``.
        xlim, ylim : tuple or None
            Axis limits forwarded to :meth:`SummaryDisplay.format`.
        xticks, yticks : list or None
            Tick positions forwarded to :meth:`SummaryDisplay.format`.
        logx, logy : bool, default False
            Log-scale axes forwarded to :meth:`SummaryDisplay.format`.
        display : SummaryDisplay or None
            Provide an existing display to superimpose data onto.
        summary_func : callable, default ``np.nanmean``
            Applied along axis=0 for mean overlays and margin ``'line'`` plots.
        xlabel, ylabel : str or None
            Custom axis labels; default to ``xvar`` / ``yvar``.
        scale : float, default 1.5
            Multiplier that controls figure size.
        bins : int or array-like, default 100
            Bin count or edges for ``'hist2d'``, ``'histogram'``, and
            ``'circ_hist'``.
        density : bool, default False
            Normalise ``'hist2d'`` counts to density.
        probability : bool, default False
            Normalise ``'histogram'`` counts to probability (0–1).
        omit_wrap : bool, default True
            Insert NaNs at angular discontinuities greater than π/2 when
            ``plot_type='line'``.
        confidence_interval : bool, default False
            Add a bootstrap CI shading band for ``plot_type='line'``.
        confidence : float, default 0.84
            Confidence level for bootstrap CI and ``trajectory2d`` overlays.
        n_boot : int, default 1000
            Number of bootstrap resamples for CI estimation.
        groupby : {'saccade', 'test', 'trial'} or str or None
            Controls grouping at **two levels**:

            *Scalar saccade queries* (``object='saccade'``, non-line plot
            types) — accepts the keyword values ``'saccade'`` (default),
            ``'test'``, or ``'trial'`` with the same semantics as
            :meth:`TrackingTrial.query`.

            *Saccade trajectory plots* (``object='saccade', plot_type='line'``)
            — additionally accepts any :class:`TrackingTrial` attribute name
            (e.g. ``'fly_id'``):

            - ``None`` / ``'saccade'`` — one thin line per individual saccade;
              ``plot_line`` handles mean/CI and optional ``split_by_sign``
              internally across all saccades.
            - Attribute name — saccades sharing the same attribute value are
              averaged into one trace per group.  The thin lines are therefore
              group means and the CI reflects across-group variance.  When
              ``split_by_sign=True`` is passed in ``plot_kwargs``, positive and
              negative-amplitude saccades are averaged **separately** within
              each group (up to 2 × n_groups traces).
        agg_func : callable or None
            Aggregation function applied per group for scalar saccade queries
            when ``groupby`` is ``'test'`` or ``'trial'``.  Default
            ``np.nanmean``.
        margin_groupby : str or None, default None
            :class:`TrackingTrial` attribute used to group data into per-group
            distributions for marginal ``'pdf'`` (and ``'histogram'``) plots.
            Independent of ``groupby``: you can display one thin line per
            saccade (``groupby=None``) while still computing per-fly marginal
            distributions (``margin_groupby='fly_id'``).

            - ``None`` — saccade data: one group per trial object (preserves
              original behaviour).  Trial data: pdf rows = individual trial rows.
            - Attribute name — all saccades (or trial rows) sharing the same
              value are pooled before normalising.  Each group yields one
              normalised histogram; the marginal mean ± CI is then computed
              across groups — statistically correct across biological subjects.
        positive_amplitude : bool, default False
            Only for ``object='saccade', plot_type='line'``.  Flip negative
            saccade traces so all traces end with positive displacement.
        min_speed, max_speed : float or None, default None
            Speed filters for ``object='saccade'`` in degrees/s.  Converted
            internally to rad/s and applied as ``peak_velocity`` subset
            conditions.
        show_n : bool, default False
            Annotate each trace panel with ``N=<groups>, n=<items>`` in the
            bottom-right corner.
        relative_to : {'start', 'peak', 'stop'}, default ``'start'``
            Reference frame alignment for ``object='saccade', plot_type='line'``.
            Each trace is shifted so that the chosen frame sits at x=0.
        rad2deg : bool, default False
            Convert ``xvar`` data from radians to degrees before plotting.
            ``plot_type='trajectory2d'`` is always exempt (it uses raw radians
            for ``cos``/``sin``).  When ``True``, pass ``xlim`` in degrees.
        right_margin_xlim, right_margin_ylim : tuple or None
            Override the auto-synced axis limits on right-margin axes.
        bottom_margin_xlim, bottom_margin_ylim : tuple or None
            Override the auto-synced axis limits on bottom-margin axes.
        plot_kwargs : dict or None
            Extra keyword arguments forwarded to the underlying Phase 3 plot
            function.  Recognised keys depend on ``plot_type``:

            **plot_type='line'**

            ==================  ================================================
            ``trace_color``     Color for individual trace lines (default 'gray').
            ``alpha``           Trace opacity (default 0.5).
            ``mean_bins``       int — bin-average the mean line into N bins over
                                the yvar range instead of a frame-by-frame mean.
            ``split_by_sign``   bool — draw separate solid/dashed mean lines for
                                positive and negative ``xvar`` groups.
            ``saccade_spans``   list of (y_start, y_stop) tuples marking the
                                saccade window on each trace (for
                                ``object='saccade'``).
            ==================  ================================================

            **plot_type='histogram'**

            ==================  ================================================
            (no extra keys)     Use ``bins``, ``probability`` top-level params.
            ==================  ================================================

            **plot_type='hist2d'**

            ==================  ================================================
            ``vmax``            Saturate the colormap at this count value.
            ``cbar``            bool — show a colorbar (default True).
            ``n_contours``      int — overlay KDE contour lines.
            ``contour_alpha``   float — opacity of KDE contours.
            ==================  ================================================

            **plot_type='scatter'**

            ==================  ================================================
            ``jitter_std``      std of Gaussian jitter added to each point.
            ``correlation``     bool — annotate Mardia circular-linear r.
            ``alpha``           point opacity.
            ``s``               marker size.
            ==================  ================================================

            **plot_type='trajectory2d'**

            ==================  ================================================
            ``circle``          bool — draw per-fly radius circles + mean circle.
            ``contour``         bool — draw bootstrap confidence ellipse.
            ``circ_hist``       bool — draw a Wedge ring histogram of endpoint
                                angles at r = 1.01–1.26 with a CI arc.
            ``mean_line``       bool — draw the mean 2D trajectory.
            ``bins``            int or edges for the circ_hist ring.
            ==================  ================================================

        **query_kwargs
            Forwarded to :meth:`TrackingExperiment.query`
            (e.g. ``subset={'condition': 1}``, ``sort_by='test_ind'``).

        Examples
        --------
        Line plot of heading vs time, one column per condition::

            exp.plot('camera_heading', 'time', col_var='condition',
                     row_var=None, col_cmap='viridis')

        Saccade traces aligned to the peak, with binned mean and degree axis::

            exp.plot('arr_relative', 'relative_time',
                     object='saccade', plot_type='line',
                     col_var='condition', row_var=None,
                     relative_to='peak', rad2deg=True,
                     xlim=(-90, 90), ylim=(0.4, -0.1),
                     plot_kwargs={'mean_bins': 20, 'split_by_sign': True},
                     show_n=True)

        Position vs amplitude 2D histogram with degree axes::

            exp.plot('start_angle', 'amplitude',
                     object='saccade', plot_type='hist2d',
                     col_var='condition', row_var=None,
                     rad2deg=True, xlim=(-180, 180), ylim=(-180, 180))

        Trajectory2D with circular histogram overlay::

            exp.plot('camera_heading', 'time',
                     plot_type='trajectory2d',
                     col_var='condition', row_var=None,
                     plot_kwargs={'circ_hist': True, 'contour': True})
        """
        if plot_kwargs is None:
            plot_kwargs = {}
        if 'subset' not in query_kwargs:
            query_kwargs['subset'] = {}
        if 'sort_by' not in query_kwargs:
            query_kwargs['sort_by'] = 'test_ind'

        # ---------------------------------------------------------------- #
        # compare_means validation                                          #
        # ---------------------------------------------------------------- #
        if plot_type == 'compare_means':
            if compare_subsets is None or len(compare_subsets) != 2:
                raise ValueError(
                    "plot_type='compare_means' requires "
                    "compare_subsets=[dict1, dict2]")
            compare_subsets = [dict(compare_subsets[0]),
                               dict(compare_subsets[1])]
            if compare_labels is None:
                compare_labels = [str(compare_subsets[0]),
                                  str(compare_subsets[1])]
            # Auto-enable margins (they will be drawn by the post-loop).
            if right_margin is True:
                right_margin = 'compare_means'
            if bottom_margin is True:
                bottom_margin = 'compare_means'

        subset = copy.copy(query_kwargs['subset'])
        sort_by = query_kwargs['sort_by']
        # For scalar saccade queries, only the keyword values are valid;
        # an attribute name (e.g. 'fly_id') means "no grouping" there.
        _saccade_groupby_keywords = {'saccade', 'test', 'trial'}
        _groupby = (groupby if groupby in _saccade_groupby_keywords
                    else 'saccade')
        if agg_func is not None:
            _agg_func = agg_func
        elif object == 'saccade' and plot_type == 'line':
            import scipy.stats as _scipy_stats
            _agg_func = partial(_scipy_stats.circmean, low=-np.pi, high=np.pi)
        else:
            _agg_func = np.nanmean

        # Build speed filter conditions for saccade object queries.
        _speed_conditions = []
        if object == 'saccade':
            if min_speed is not None:
                _speed_conditions.append(f'>={min_speed * np.pi / 180:.8f}')
            if max_speed is not None:
                _speed_conditions.append(f'<={max_speed * np.pi / 180:.8f}')

        # ---------------------------------------------------------------- #
        # Grid setup                                                        #
        # ---------------------------------------------------------------- #
        row_vals = get_grid_vals(self, row_var, subset) if row_var is not None else [None]
        col_vals = get_grid_vals(self, col_var, subset) if col_var is not None else [None]
        n_data_rows, n_data_cols = len(row_vals), len(col_vals)
        color_arr = resolve_colors(row_cmap, col_cmap, row_vals, col_vals,
                                   default_color=color)

        right_margin_types = _resolve_margin(right_margin, plot_type)
        bottom_margin_types = _resolve_margin(bottom_margin, plot_type)
        _has_right = len(right_margin_types) > 0
        _has_bottom = len(bottom_margin_types) > 0

        num_rows = n_data_rows + (1 if _has_bottom else 0)
        num_cols = n_data_cols + (1 if _has_right else 0)

        if display is None:
            _subplot_size = subplot_size if subplot_size is not None else (scale * 2, scale * 2)
            self.display = SummaryDisplay(
                num_rows=num_rows, num_cols=num_cols,
                right_margin=_has_right, bottom_margin=_has_bottom,
                subplot_size=_subplot_size,
                right_margin_ratio=right_margin_ratio,
                bottom_margin_ratio=bottom_margin_ratio)
        else:
            self.display = display
        trace_axes = self.display.trace_axes

        # ---------------------------------------------------------------- #
        # Data-extraction closure                                           #
        # ---------------------------------------------------------------- #
        def _get_xs_ys(cell_subset):
            """Return (xs, ys, extras) for one cell.

            ``extras`` is a dict with optional keys:
              'durations' : ndarray, shape (n_saccades,) — saccade durations
                            for trace highlighting in plot_line.
            """
            # ---------------------------------------------------------- #
            # compare_means: query yvar for both compare_subsets, build   #
            # per-subject value arrays aligned by pair_by attribute.      #
            # ---------------------------------------------------------- #
            if plot_type == 'compare_means':
                per_subject = [{}, {}]  # {subject_id: [values]} per group
                for group_i, cs in enumerate(compare_subsets):
                    merged = {**cell_subset, **cs}
                    if object == 'saccade':
                        for trial in self.trials:
                            sid = (getattr(trial, pair_by, None)
                                   if pair_by else None)
                            if sid is None:
                                sid = id(trial)
                            try:
                                raw = trial.query(
                                    object='saccade', output=yvar,
                                    subset=merged,
                                    groupby='trial',
                                    agg_func=np.nanmean)
                                if hasattr(raw, '__len__'):
                                    val = float(np.nanmean(raw))
                                else:
                                    val = float(raw)
                                if np.isfinite(val):
                                    per_subject[group_i].setdefault(
                                        sid, []).append(val)
                            except Exception:
                                continue
                    else:
                        for trial in self.trials:
                            sid = (getattr(trial, pair_by, None)
                                   if pair_by else None)
                            if sid is None:
                                sid = id(trial)
                            try:
                                raw = trial.query(
                                    output=yvar, subset=merged,
                                    sort_by=sort_by)
                                if raw is None:
                                    continue
                                arr = np.asarray(raw, dtype=float)
                                val = float(np.nanmean(arr))
                                if np.isfinite(val):
                                    per_subject[group_i].setdefault(
                                        sid, []).append(val)
                            except Exception:
                                continue

                # Collapse multiple trials per subject to a single mean.
                ids1 = list(per_subject[0].keys())
                vals1 = np.array([np.nanmean(per_subject[0][s])
                                  for s in ids1], dtype=float)
                ids2 = list(per_subject[1].keys())
                vals2 = np.array([np.nanmean(per_subject[1][s])
                                  for s in ids2], dtype=float)

                # Apply rad2deg here so both groups are converted together.
                if rad2deg:
                    vals1 = np.degrees(vals1)
                    vals2 = np.degrees(vals2)

                _lbl = (compare_labels if compare_labels is not None
                        else [str(compare_subsets[0]),
                              str(compare_subsets[1])])
                return vals1, vals2, {
                    'ids1': ids1,
                    'ids2': ids2,
                    'labels': _lbl,
                    'vals1': vals1,
                    'vals2': vals2,
                }

            if object == 'saccade':
                # Inject speed conditions into a working copy of the subset.
                eff_subset = dict(cell_subset)
                if _speed_conditions:
                    _ex = eff_subset.get('peak_velocity', [])
                    if isinstance(_ex, str):
                        _ex = [_ex]
                    elif not isinstance(_ex, (list, tuple)):
                        _ex = []
                    eff_subset['peak_velocity'] = list(_ex) + _speed_conditions

                if plot_type == 'line':
                    try:
                        saccades = self.query(
                            object='saccade', output='saccade',
                            subset=eff_subset, groupby='saccade')
                    except RuntimeError:
                        return None, None, {}
                    if not saccades:
                        return None, None, {}
                    xs_list, ys_list, span_list, ref_indices, sign_list = [], [], [], [], []
                    for s in saccades:
                        xv = np.array(getattr(s, xvar))
                        yv = np.array(getattr(s, yvar))
                        # Centre both arrays on the reference frame.
                        if relative_to == 'peak':
                            ref_idx = s.peak_ind
                        elif relative_to == 'stop':
                            ref_idx = s.stop
                        else:  # 'start' (default)
                            ref_idx = s.start
                        ref_idx = min(ref_idx, len(xv) - 1)
                        if ref_idx < len(xv):
                            xv = xv - xv[ref_idx]
                        if ref_idx < len(yv):
                            ref_y = yv[ref_idx]
                            yv = yv - ref_y
                        else:
                            ref_y = 0.0
                        if positive_amplitude and np.any(~np.isnan(xv)):
                            # Track amplitude sign before positive_amplitude flip
                            # (used later when split_by_sign=True with groupby).
                            _ep = xv[~np.isnan(xv)][-1]
                            sign_list.append(1 if _ep >= 0 else -1)
                            if xv[~np.isnan(xv)][-1] < 0:
                                xv = -xv
                        else:
                            _non_nan = xv[~np.isnan(xv)]
                            sign_list.append(
                                1 if (len(_non_nan) == 0 or _non_nan[-1] >= 0) else -1)
                        xs_list.append(xv)
                        ys_list.append(yv)
                        ref_indices.append(ref_idx)
                        # Saccade span in shifted y-coordinates (yv is already shifted).
                        y_start = float(yv[s.start]) if s.start < len(yv) else 0.0
                        y_stop  = float(yv[s.stop])  if s.stop  < len(yv) else float(yv[-1])
                        span_list.append((y_start, y_stop))
                    if not xs_list:
                        return None, None, {}
                    # Align all traces so their reference frame sits at the
                    # same column index. This ensures nanmean(axis=0) passes
                    # through (0, 0) when the traces are centred.
                    max_before = max(ref_indices)
                    max_after = max(len(xs_list[i]) - 1 - ref_indices[i]
                                    for i in range(len(xs_list)))
                    total_len = max_before + max_after + 1
                    if total_len == 0:
                        return None, None, {}
                    xs_pad = np.full((len(xs_list), total_len), np.nan)
                    ys_pad = np.full((len(ys_list), total_len), np.nan)
                    # stop_cols[i]  = padded column index of s.stop for saccade i.
                    # margin_gids[i] = margin_groupby key (for per-group pdf margin).
                    # traj_gids[i]   = groupby key (for trajectory averaging).
                    _groupby_is_attr = (groupby is not None
                                        and groupby not in _saccade_groupby_keywords)
                    _split_by_sign = plot_kwargs.get('split_by_sign', False)
                    stop_cols = []
                    margin_gids = []
                    traj_gids = []
                    for i, (xv, yv) in enumerate(zip(xs_list, ys_list)):
                        offset = max_before - ref_indices[i]
                        xs_pad[i, offset:offset + len(xv)] = xv
                        ys_pad[i, offset:offset + len(yv)] = yv
                        sc = offset + saccades[i].stop
                        stop_cols.append(min(sc, total_len - 1))
                        # Margin grouping key (independent of trajectory grouping).
                        if margin_groupby is not None:
                            key = getattr(saccades[i].trial, margin_groupby, None)
                            margin_gids.append(key if key is not None
                                               else id(saccades[i].trial))
                        else:
                            margin_gids.append(id(saccades[i].trial))
                        # Trajectory grouping key (only when groupby is an attr name).
                        if _groupby_is_attr:
                            key = getattr(saccades[i].trial, groupby, None)
                            traj_gids.append(key if key is not None
                                             else id(saccades[i].trial))

                    # 1D flat endpoints (raw, one value per saccade) for the
                    # histogram margin mtype.  Built from xs_pad before any
                    # per-group averaging so they always represent individual
                    # saccade stopping points.
                    stop_xs_flat = np.array([
                        xs_pad[i, sc] if 0 <= sc < xs_pad.shape[1] else np.nan
                        for i, sc in enumerate(stop_cols)])
                    stop_ys_flat = np.array([
                        ys_pad[i, sc] if 0 <= sc < ys_pad.shape[1] else np.nan
                        for i, sc in enumerate(stop_cols)])

                    # Build per-margin-group endpoint arrays for the pdf margin.
                    # Each row = one margin group (e.g. one fly), values =
                    # that group's saccade stopping points (NaN-padded).
                    unique_margin_gids = list(dict.fromkeys(margin_gids))
                    _stop_xs_by_group, _stop_ys_by_group = [], []
                    for gid in unique_margin_gids:
                        idxs = [i for i, g in enumerate(margin_gids) if g == gid]
                        _stop_xs_by_group.append(
                            [xs_pad[i, stop_cols[i]] for i in idxs])
                        _stop_ys_by_group.append(
                            [ys_pad[i, stop_cols[i]] for i in idxs])
                    _max_s = max(len(r) for r in _stop_xs_by_group)
                    def _pad_rows(rows, width):
                        out = np.full((len(rows), width), np.nan)
                        for ri, r in enumerate(rows):
                            out[ri, :len(r)] = r
                        return out
                    stop_xs_by_group = _pad_rows(_stop_xs_by_group, _max_s)
                    stop_ys_by_group = _pad_rows(_stop_ys_by_group, _max_s)

                    # Trajectory averaging: when groupby names a trial attribute,
                    # average saccade traces within each group so the thin lines
                    # are group means and the CI reflects across-group variance.
                    # When split_by_sign=True, positive and negative-amplitude
                    # saccades are averaged separately within each group.
                    if _groupby_is_attr:
                        unique_traj_gids = list(dict.fromkeys(traj_gids))
                        avg_rows_xs, avg_rows_ys = [], []
                        for gid in unique_traj_gids:
                            idxs = [i for i, g in enumerate(traj_gids) if g == gid]
                            if _split_by_sign:
                                pos_idxs = [i for i in idxs if sign_list[i] >= 0]
                                neg_idxs = [i for i in idxs if sign_list[i] < 0]
                                if pos_idxs:
                                    avg_rows_xs.append(
                                        np.nanmean(xs_pad[pos_idxs], axis=0))
                                    avg_rows_ys.append(
                                        np.nanmean(ys_pad[pos_idxs], axis=0))
                                if neg_idxs:
                                    avg_rows_xs.append(
                                        np.nanmean(xs_pad[neg_idxs], axis=0))
                                    avg_rows_ys.append(
                                        np.nanmean(ys_pad[neg_idxs], axis=0))
                            else:
                                avg_rows_xs.append(
                                    np.nanmean(xs_pad[idxs], axis=0))
                                avg_rows_ys.append(
                                    np.nanmean(ys_pad[idxs], axis=0))
                        if avg_rows_xs:
                            xs_pad = np.array(avg_rows_xs)
                            ys_pad = np.array(avg_rows_ys)
                        span_list = []  # not meaningful for averaged traces

                    return xs_pad, ys_pad, {
                        'spans': span_list,
                        'stop_xs_flat': stop_xs_flat,
                        'stop_ys_flat': stop_ys_flat,
                        'stop_xs_by_group': stop_xs_by_group,
                        'stop_ys_by_group': stop_ys_by_group,
                    }
                else:
                    # Scalar saccade-table column queries.
                    try:
                        xs_raw = self.query(
                            object='saccade', output=xvar,
                            subset=eff_subset, groupby=_groupby,
                            agg_func=_agg_func)
                        xs = np.array(xs_raw, dtype=float)
                    except RuntimeError:
                        return None, None, {}
                    if yvar is None:
                        return xs, None, {}
                    try:
                        ys_raw = self.query(
                            object='saccade', output=yvar,
                            subset=eff_subset, groupby=_groupby,
                            agg_func=_agg_func)
                        ys = np.array(ys_raw, dtype=float)
                    except RuntimeError:
                        ys = None
                    return xs, ys, {}

            # --- object='trial' ---
            xs_raw = self.query(same_size=True, output=xvar,
                                subset=cell_subset, sort_by=sort_by)
            xs = np.array(xs_raw)
            if xs.ndim > 1:
                xs = xs.reshape(-1, xs.shape[-1])
            # Build the row-keep mask *before* filtering xs so that ys can be
            # filtered with the same mask later (rows correspond to the same
            # same_size-padded order: trial_0_rows, trial_1_rows, …).
            xs_row_mask = None
            if xs.ndim == 2 and xs.shape[0] > 0:
                xs_row_mask = np.isnan(xs).mean(1) < 1
                xs = xs[xs_row_mask]
            if xs.size == 0:
                return None, None, {}

            # Query yvar for all plot types that need it (or for margins).
            ys = None
            if yvar is not None:
                ys_raw = self.query(same_size=True, output=yvar,
                                    subset=cell_subset, sort_by=sort_by)
                ys = np.array(ys_raw)
                if ys.ndim > 1:
                    ys = ys.reshape(-1, ys.shape[-1])
                # Apply the same row filter used to drop all-NaN xs rows so
                # that xs[i] and ys[i] always correspond to the same trace.
                if (xs_row_mask is not None and ys.ndim == 2
                        and ys.shape[0] == xs_row_mask.shape[0]):
                    ys = ys[xs_row_mask]
                elif ys.ndim == 2:
                    # Fallback: align by row count.
                    n = xs.shape[0]
                    if ys.shape[0] > n:
                        ys = ys[:n]
                    elif ys.shape[0] < n:
                        ys = np.vstack(
                            [ys, np.full((n - ys.shape[0], ys.shape[1]),
                                         np.nan)])

            if plot_type in ('hist2d', 'scatter'):
                xs_f = xs.flatten()
                ys_f = ys.flatten() if ys is not None else None
                if ys_f is not None:
                    valid = ~np.isnan(xs_f) & ~np.isnan(ys_f)
                    return xs_f[valid], ys_f[valid], {}
                return xs_f[~np.isnan(xs_f)], None, {}

            if plot_type == 'histogram':
                xs_f = xs.flatten()
                return xs_f[~np.isnan(xs_f)], None, {}

            if plot_type == 'trajectory2d':
                return xs, ys, {}

            # For line and pdf plot types, build a per-group xs array when
            # margin_groupby is set.  Each unique attribute value on the trial
            # objects contributes one row (nanmean of its trials' data) so that
            # the pdf margin computes one normalised histogram per group.
            _xs_margin_grouped = None
            if margin_groupby is not None:
                _mg_groups = {}
                for t in self.trials:
                    gv = getattr(t, margin_groupby, None)
                    if gv is None:
                        continue
                    try:
                        _td = np.array(t.query(output=xvar, subset=cell_subset,
                                               sort_by=sort_by))
                    except Exception:
                        continue
                    if _td.size == 0:
                        continue
                    if _td.ndim > 1:
                        _td = _td.reshape(-1, _td.shape[-1])
                    if gv not in _mg_groups:
                        _mg_groups[gv] = []
                    _mg_groups[gv].append(_td)
                if _mg_groups:
                    _mg_rows = [np.nanmean(np.vstack(arrs), axis=0)
                                for arrs in _mg_groups.values()]
                    _xs_margin_grouped = np.array(_mg_rows)

            _trial_extras = ({} if _xs_margin_grouped is None
                             else {'xs_margin_grouped': _xs_margin_grouped})

            if plot_type == 'pdf':
                # Keep 2-D so plot_pdf can compute per-row histograms.
                return xs, None, _trial_extras

            # plot_type == 'line': optionally insert NaN at wrapping discontinuities.
            if omit_wrap and ys is not None and ys.ndim == 2:
                xs_w, mask = omit_wrapping(xs, return_mask=True)
                ys_w = np.full_like(xs_w, np.nan)
                for i in range(min(ys.shape[0], ys_w.shape[0])):
                    vl = min(ys.shape[1], ys_w.shape[1])
                    ys_w[i, :vl] = ys[i, :vl]
                ys_w[~mask] = np.nan
                return xs_w, ys_w, _trial_extras
            return xs, ys, _trial_extras

        # ---------------------------------------------------------------- #
        # Pre-query pass: collect data + compute cross-panel normalisers   #
        # ---------------------------------------------------------------- #
        cell_cache = {}
        hist2d_maxes = []
        _ch_hists, _ch_bins_edges, _ch_keys = [], None, []

        for row_i, row_val in enumerate(row_vals):
            cell_subset = copy.copy(subset)
            if row_var is not None:
                cell_subset[row_var] = row_val
            for col_i, col_val in enumerate(col_vals):
                if col_var is not None:
                    cell_subset[col_var] = col_val
                xs, ys, cell_extra = _get_xs_ys(cell_subset)
                # Convert x data from radians to degrees when requested.
                # trajectory2d is exempt — it feeds xs into cos/sin directly.
                if rad2deg and xs is not None and plot_type not in ('trajectory2d', 'compare_means'):
                    xs = np.degrees(xs)
                cell_cache[(row_i, col_i)] = (xs, ys, cell_extra)
                if xs is None or xs.size == 0:
                    continue

                if plot_type == 'hist2d' and ys is not None and ys.size > 0:
                    h, _, _ = np.histogram2d(xs, ys, bins=bins, density=density)
                    hist2d_maxes.append(float(h.max()))

                elif plot_type == 'trajectory2d' and plot_kwargs.get('circ_hist'):
                    xs2 = xs if xs.ndim == 2 else xs[np.newaxis]
                    d_v = np.array([np.cos(xs2), np.sin(xs2)]).transpose(1, 0, 2)
                    d_v[np.isnan(d_v)] = 0
                    traj = np.cumsum(d_v, axis=-1) / xs2.shape[-1]
                    last_p = traj[..., -1]
                    angles = np.arctan2(last_p[..., 1], last_p[..., 0])
                    n_cbins = plot_kwargs.get('bins', 100)
                    if isinstance(n_cbins, int):
                        b_edges = np.linspace(-np.pi, np.pi, n_cbins + 1)
                    else:
                        b_edges = np.asarray(n_cbins)
                    _ch_bins_edges = b_edges
                    h, _ = np.histogram(angles, bins=b_edges, density=False)
                    _ch_hists.append(h.astype(float))
                    _ch_keys.append((row_i, col_i))

        global_hist2d_vmax = max(hist2d_maxes) if hist2d_maxes else None

        circ_hist_lookup = {}
        circ_global_vmax = None
        if _ch_hists:
            hists_arr = np.array(_ch_hists)
            row_sums = hists_arr.sum(1, keepdims=True).clip(min=1)
            norm_hists = hists_arr / row_sums
            circ_global_vmax = float(norm_hists.max())
            for key, nh in zip(_ch_keys, norm_hists):
                circ_hist_lookup[key] = (nh, _ch_bins_edges)

        # ---------------------------------------------------------------- #
        # Per-cell N / n counts  (only computed when show_n=True)          #
        # ---------------------------------------------------------------- #
        cell_counts = {}
        if show_n:
            for _ri, _rv in enumerate(row_vals):
                _cs = copy.copy(subset)
                if row_var is not None:
                    _cs[row_var] = _rv
                for _ci, _cv in enumerate(col_vals):
                    if col_var is not None:
                        _cs[col_var] = _cv
                    _eff = dict(_cs)
                    if _speed_conditions:
                        _spex = _eff.get('peak_velocity', [])
                        if isinstance(_spex, str):
                            _spex = [_spex]
                        elif not isinstance(_spex, (list, tuple)):
                            _spex = []
                        _eff['peak_velocity'] = list(_spex) + _speed_conditions
                    if object == 'saccade':
                        _N, _n = 0, 0
                        for _trial in self.trials:
                            try:
                                _res = _trial.query(
                                    object='saccade', output='amplitude',
                                    subset=_eff, groupby='saccade')
                                if len(_res) > 0:
                                    _N += 1
                                    _n += len(_res)
                            except RuntimeError:
                                continue
                    else:
                        _xc = cell_cache[(_ri, _ci)][0]
                        _n = int(_xc.shape[0]) if _xc is not None else 0
                        _N = 0
                        for _trial in self.trials:
                            try:
                                _td = _trial.query(output=xvar, subset=_cs)
                                if hasattr(_td, 'size') and _td.size > 0:
                                    _N += 1
                            except Exception:
                                continue
                    cell_counts[(_ri, _ci)] = (_N, _n)

        # ---------------------------------------------------------------- #
        # Draw pass                                                         #
        # ---------------------------------------------------------------- #
        # Effective ylim used to restrict mean_bins binning to the visible
        # range, so 10 bins spans the display window rather than the full
        # data extent (which may be much wider).
        _effective_bin_ylim = ylim
        if _effective_bin_ylim is None and object == 'saccade' and plot_type == 'line':
            _effective_bin_ylim = (0.5, -0.25)

        # Per-cell summary cache and stats rows (used by compare_means post-
        # loop and return_stats).
        cell_summary_cache = {}
        _stats_rows = [] if return_stats else None

        for row_i, (row_ax_row, row_val, row_colors) in enumerate(zip(
                trace_axes, row_vals, color_arr)):
            cell_subset = copy.copy(subset)
            if row_var is not None:
                cell_subset[row_var] = row_val
            row_summ_ax = self.display.right_col[row_i] if _has_right else None

            for col_i, (ax, col_val, cell_color) in enumerate(zip(
                    row_ax_row, col_vals, row_colors)):
                while isinstance(ax, np.ndarray):
                    ax = ax[0]
                if col_var is not None:
                    cell_subset[col_var] = col_val
                col_summ_ax = self.display.bottom_row[col_i] if _has_bottom else None

                xs, ys, cell_extra = cell_cache[(row_i, col_i)]
                if xs is None or xs.size == 0:
                    continue

                summary_dict = None
                kw = dict(plot_kwargs)

                if plot_type == 'line':
                    plot_line(ax, xs, ys, cell_color,
                              summary_func=summary_func,
                              ci=confidence_interval,
                              confidence=confidence,
                              n_boot=n_boot,
                              split_by_sign=kw.pop('split_by_sign', False),
                              mean_bins=kw.pop('mean_bins', None),
                              ylim=_effective_bin_ylim,
                              saccade_spans=cell_extra.get('spans'),
                              **kw)

                elif plot_type == 'hist2d':
                    if ys is None:
                        continue
                    if global_hist2d_vmax is not None and 'vmax' not in kw:
                        kw['vmax'] = global_hist2d_vmax
                    _n_contours = kw.pop('n_contours', 0)
                    _contour_alpha = kw.pop('contour_alpha', 0.3)
                    # Suppress per-panel colorbars by default: the colour
                    # scale is already unified via global_hist2d_vmax, and
                    # plt.colorbar() rewrites the SubFigure GridSpec on each
                    # call which hangs indefinitely for multi-panel grids.
                    kw.setdefault('cbar', False)
                    result = plot_hist2d(ax, xs, ys, cell_color,
                                        bins=bins, density=density,
                                        n_contours=_n_contours,
                                        contour_alpha=_contour_alpha,
                                        return_summary=True, **kw)
                    if isinstance(result, tuple):
                        _, summary_dict = result

                elif plot_type == 'trajectory2d':
                    if (row_i, col_i) in circ_hist_lookup:
                        nh, b_edges = circ_hist_lookup[(row_i, col_i)]
                        kw['_precomputed_hist'] = nh
                        kw['_precomputed_bins'] = b_edges
                        kw['_global_vmax'] = circ_global_vmax
                    kw.setdefault('confidence', confidence)
                    kw['return_summary'] = True
                    result = plot_trajectory2d(ax, xs, None, cell_color, **kw)
                    if isinstance(result, tuple):
                        _, summary_dict = result

                elif plot_type == 'histogram':
                    plot_histogram(ax, xs, bins, cell_color,
                                   probability=probability,
                                   summary_func=(summary_func
                                                 if callable(summary_func)
                                                 else None),
                                   **kw)

                elif plot_type == 'scatter':
                    if ys is None:
                        continue
                    plot_scatter(ax, xs, ys, cell_color, **kw)

                elif plot_type == 'pdf':
                    plot_pdf(ax, xs, bins, cell_color,
                             ci=confidence_interval,
                             confidence=confidence,
                             n_boot=n_boot,
                             show_traces=True,
                             summary_func=(summary_func
                                           if callable(summary_func)
                                           else None),
                             **kw)

                elif plot_type == 'compare_means':
                    # xs = vals1, ys = vals2 from _get_xs_ys compare_means
                    # branch.  Skip silently when either group is empty.
                    if xs is None or ys is None or (len(xs) == 0 and len(ys) == 0):
                        cell_summary_cache[(row_i, col_i)] = {
                            'summary_dict': None,
                            'cell_extra': cell_extra,
                            'cell_color': cell_color,
                        }
                        continue
                    kw_cm = dict(kw)
                    summary_dict = plot_compare_means(
                        ax, xs, ys, cell_color,
                        labels=cell_extra.get('labels', compare_labels),
                        paired_ids1=cell_extra.get('ids1'),
                        paired_ids2=cell_extra.get('ids2'),
                        marker_color=kw_cm.pop('marker_color', None),
                        jitter_std=kw_cm.pop('jitter_std', 0.05),
                        confidence=confidence,
                        n_boot=n_boot,
                        show_ns=kw_cm.pop('show_ns', True),
                        ylim=ylim,
                        **kw_cm)

                # -------------------------------------------------------- #
                # Cache summary_dict for margin post-loop and return_stats. #
                # -------------------------------------------------------- #
                cell_summary_cache[(row_i, col_i)] = {
                    'summary_dict': summary_dict,
                    'cell_extra': cell_extra,
                    'cell_color': cell_color,
                }
                if return_stats and summary_dict is not None:
                    _stats_rows.append({
                        'row_val': row_val,
                        'col_val': col_val,
                        **{k: v for k, v in summary_dict.items()
                           if not isinstance(v, np.ndarray)},
                    })

                # Per-cell N / n annotation.
                if show_n and (row_i, col_i) in cell_counts:
                    _N_val, _n_val = cell_counts[(row_i, col_i)]
                    ax.text(0.98, 0.02, f'N={_N_val}, n={_n_val}',
                            transform=ax.transAxes,
                            fontsize=plt.rcParams.get('xtick.labelsize',
                                                      plt.rcParams.get('font.size', 10)),
                            ha='right', va='bottom', color='black', zorder=10)

                # Margin drawing for this cell.
                # compare_means margins are handled by the post-loop below.
                if plot_type == 'compare_means':
                    continue

                # For saccade line plots, histogram margins show the endpoint
                # (stopping value) of each saccade trace, not all time points.
                _margin_xs, _margin_ys = xs, ys
                _pdf_margin_xs, _pdf_margin_ys = xs, ys  # per-fly grouped, for pdf mtype
                if object == 'saccade' and plot_type == 'line':
                    # Histogram mtype uses the raw 1-D saccade stopping-point
                    # values (precomputed in _get_xs_ys before any averaging).
                    if 'stop_xs_flat' in cell_extra:
                        _margin_xs = cell_extra['stop_xs_flat']
                        _margin_ys = cell_extra['stop_ys_flat']
                        # stop_xs_flat is built in radians; convert if needed.
                        if rad2deg and _margin_xs is not None:
                            _margin_xs = np.degrees(_margin_xs)
                    elif xs is not None and xs.ndim == 2:
                        # Fallback: last non-NaN value per row.
                        _margin_xs = np.array([
                            row[~np.isnan(row)][-1] if np.any(~np.isnan(row))
                            else np.nan for row in xs])
                        if ys is not None and ys.ndim == 2:
                            _margin_ys = np.array([
                                row[~np.isnan(row)][-1] if np.any(~np.isnan(row))
                                else np.nan for row in ys])
                    # Per-margin-group endpoints for pdf mtype (2D: n_groups × n_saccades).
                    # stop_xs_by_group is built in radians; apply rad2deg if needed.
                    _pdf_margin_xs = cell_extra.get('stop_xs_by_group', _margin_xs)
                    if rad2deg and 'stop_xs_by_group' in cell_extra and _pdf_margin_xs is not None:
                        _pdf_margin_xs = np.degrees(_pdf_margin_xs)
                    _pdf_margin_ys = cell_extra.get('stop_ys_by_group', _margin_ys)
                elif 'xs_margin_grouped' in cell_extra:
                    # For trial-level plots, margin_groupby provides a
                    # per-group 2D array for the pdf margin.
                    _pdf_margin_xs = cell_extra['xs_margin_grouped']

                _fallback_bins = margin_bins if margin_bins is not None else bins
                _right_bins = right_margin_bins if right_margin_bins is not None else _fallback_bins
                _bottom_bins = bottom_margin_bins if bottom_margin_bins is not None else _fallback_bins

                # When one cmap is None, cells sharing a margin axis all get
                # the same color.  Use distinct linestyles + labels so they
                # can be told apart.
                _right_ls = _MARGIN_LINESTYLES[col_i % len(_MARGIN_LINESTYLES)] \
                    if col_cmap is None and n_data_cols > 1 else 'solid'
                _right_lbl = (str(col_val) if col_cmap is None and n_data_cols > 1
                              else None)
                _bottom_ls = _MARGIN_LINESTYLES[row_i % len(_MARGIN_LINESTYLES)] \
                    if row_cmap is None and n_data_rows > 1 else 'solid'
                _bottom_lbl = (str(row_val) if row_cmap is None and n_data_rows > 1
                               else None)

                for mtype in right_margin_types:
                    _rx = _pdf_margin_xs if mtype == 'pdf' else _margin_xs
                    _ry = _pdf_margin_ys if mtype == 'pdf' else _margin_ys
                    _draw_margin_cell(
                        row_summ_ax, mtype, _rx, _ry, cell_color, 'right',
                        summary_func, _right_bins, probability,
                        confidence_interval, confidence, n_boot,
                        plot_type, summary_dict, n_data_cols,
                        overlay_index=col_i,
                        linestyle=_right_ls, label=_right_lbl)
                for mtype in bottom_margin_types:
                    _bx = _pdf_margin_xs if mtype == 'pdf' else _margin_xs
                    _by = _pdf_margin_ys if mtype == 'pdf' else _margin_ys
                    _draw_margin_cell(
                        col_summ_ax, mtype, _bx, _by, cell_color, 'bottom',
                        summary_func, _bottom_bins, probability,
                        confidence_interval, confidence, n_boot,
                        plot_type, summary_dict, n_data_rows,
                        overlay_index=row_i,
                        linestyle=_bottom_ls, label=_bottom_lbl)

        # ---------------------------------------------------------------- #
        # compare_means margin post-loop                                    #
        # ---------------------------------------------------------------- #
        # Margin axes for compare_means are populated after all cells have  #
        # been drawn, so we can aggregate per-subject values across cells.  #
        if plot_type == 'compare_means' and (_has_right or _has_bottom):

            def _subject_diffs(sd):
                """Per-subject (vals2 - vals1) for subjects in both groups."""
                if sd is None:
                    return np.array([]), []
                v1 = np.asarray(sd.get('vals1', []), dtype=float)
                v2 = np.asarray(sd.get('vals2', []), dtype=float)
                g1 = list(sd.get('ids1', []))
                g2 = list(sd.get('ids2', []))
                id2_map = {s: i for i, s in enumerate(g2)}
                id1_idx = {s: i for i, s in enumerate(g1)}
                diffs, diff_ids = [], []
                for sid in g1:
                    if sid in id2_map:
                        i = id1_idx[sid]
                        j = id2_map[sid]
                        if (i < len(v1) and j < len(v2)
                                and np.isfinite(v1[i])
                                and np.isfinite(v2[j])):
                            diffs.append(v2[j] - v1[i])
                            diff_ids.append(sid)
                return np.array(diffs), diff_ids

            def _subject_means(sd):
                """Per-subject grand mean across both groups."""
                if sd is None:
                    return np.array([]), []
                v1 = np.asarray(sd.get('vals1', []), dtype=float)
                v2 = np.asarray(sd.get('vals2', []), dtype=float)
                g1 = list(sd.get('ids1', []))
                g2 = list(sd.get('ids2', []))
                id2_map = {s: i for i, s in enumerate(g2)}
                g1_set = set(g1)
                means, mean_ids = [], []
                for i, sid in enumerate(g1):
                    vals = []
                    if i < len(v1) and np.isfinite(v1[i]):
                        vals.append(v1[i])
                    if sid in id2_map:
                        j = id2_map[sid]
                        if j < len(v2) and np.isfinite(v2[j]):
                            vals.append(v2[j])
                    if vals:
                        means.append(np.nanmean(vals))
                        mean_ids.append(sid)
                for j, sid in enumerate(g2):
                    if sid not in g1_set and j < len(v2) and np.isfinite(v2[j]):
                        means.append(v2[j])
                        mean_ids.append(sid)
                return np.array(means), mean_ids

            _margin_labels = (compare_labels if compare_labels is not None
                              else [str(compare_subsets[0]),
                                    str(compare_subsets[1])])
            _cm_kw = {k: v for k, v in plot_kwargs.items()
                      if k in ('marker_color', 'jitter_std', 'show_ns', 's', 'alpha',
                               'clip_on')}
            _cm_kw.setdefault('jitter_std', 0.05)
            _cm_kw.setdefault('show_ns', False)

            if _has_right:
                for row_i, (row_val, row_colors) in enumerate(
                        zip(row_vals, color_arr)):
                    row_summ_ax = self.display.right_col[row_i]
                    if row_summ_ax is None:
                        continue
                    row_color = row_colors[0]
                    m_mc = None
                    if (n_data_cols == 2
                            and margin_mode in ('diff_of_diffs',
                                                'diff_of_means')):
                        _sd0 = ((cell_summary_cache.get((row_i, 0)) or {})
                                .get('summary_dict'))
                        _sd1 = ((cell_summary_cache.get((row_i, 1)) or {})
                                .get('summary_dict'))
                        if margin_mode == 'diff_of_diffs':
                            m_v1, m_i1 = _subject_diffs(_sd0)
                            m_v2, m_i2 = _subject_diffs(_sd1)
                        else:
                            m_v1, m_i1 = _subject_means(_sd0)
                            m_v2, m_i2 = _subject_means(_sd1)
                        # Color x=0 dots with col-0 color, x=1 dots with col-1 color.
                        _cc0 = np.asarray(row_colors[0], dtype=float)
                        _cc1 = np.asarray(row_colors[1], dtype=float)
                        m_mc = (
                            np.tile(_cc0, (len(m_v1), 1)) if len(m_v1) else np.zeros((0, 3)),
                            np.tile(_cc1, (len(m_v2), 1)) if len(m_v2) else np.zeros((0, 3)))
                    else:
                        _parts1, _parts2 = [], []
                        _col_mc1, _col_mc2 = [], []
                        for c in range(n_data_cols):
                            _sd = ((cell_summary_cache.get((row_i, c)) or {})
                                   .get('summary_dict'))
                            _p1 = np.asarray((_sd or {}).get('vals1', []),
                                             dtype=float)
                            _p2 = np.asarray((_sd or {}).get('vals2', []),
                                             dtype=float)
                            _parts1.append(_p1)
                            _parts2.append(_p2)
                            _cc = np.asarray(row_colors[c], dtype=float)
                            _col_mc1.append(np.tile(_cc, (len(_p1), 1)))
                            _col_mc2.append(np.tile(_cc, (len(_p2), 1)))
                        m_v1 = np.concatenate(_parts1)
                        m_v2 = np.concatenate(_parts2)
                        m_mc = (
                            np.concatenate(_col_mc1) if _col_mc1 else np.zeros((0, 3)),
                            np.concatenate(_col_mc2) if _col_mc2 else np.zeros((0, 3)))
                        m_i1 = m_i2 = None
                    if len(m_v1) == 0 and len(m_v2) == 0:
                        continue
                    _right_ylim_ref = (right_margin_ylim if right_margin_ylim is not None
                                       else ylim)
                    plot_compare_means(
                        row_summ_ax, m_v1, m_v2, row_color,
                        labels=_margin_labels,
                        paired_ids1=m_i1 if m_i1 else None,
                        paired_ids2=m_i2 if m_i2 else None,
                        marker_colors=m_mc,
                        confidence=confidence, n_boot=n_boot,
                        ylim=_right_ylim_ref,
                        **_cm_kw)

            if _has_bottom:
                for col_i, col_val in enumerate(col_vals):
                    col_summ_ax = self.display.bottom_row[col_i]
                    if col_summ_ax is None:
                        continue
                    col_color = color_arr[0][col_i]
                    m_mc = None
                    if (n_data_rows == 2
                            and margin_mode in ('diff_of_diffs',
                                                'diff_of_means')):
                        _sd0 = ((cell_summary_cache.get((0, col_i)) or {})
                                .get('summary_dict'))
                        _sd1 = ((cell_summary_cache.get((1, col_i)) or {})
                                .get('summary_dict'))
                        if margin_mode == 'diff_of_diffs':
                            m_v1, m_i1 = _subject_diffs(_sd0)
                            m_v2, m_i2 = _subject_diffs(_sd1)
                        else:
                            m_v1, m_i1 = _subject_means(_sd0)
                            m_v2, m_i2 = _subject_means(_sd1)
                        # Color x=0 dots with row-0 color, x=1 dots with row-1 color.
                        _rc0 = np.asarray(color_arr[0][col_i], dtype=float)
                        _rc1 = np.asarray(color_arr[1][col_i], dtype=float)
                        m_mc = (
                            np.tile(_rc0, (len(m_v1), 1)) if len(m_v1) else np.zeros((0, 3)),
                            np.tile(_rc1, (len(m_v2), 1)) if len(m_v2) else np.zeros((0, 3)))
                    else:
                        _parts1, _parts2 = [], []
                        _row_mc1, _row_mc2 = [], []
                        for r in range(n_data_rows):
                            _sd = ((cell_summary_cache.get((r, col_i)) or {})
                                   .get('summary_dict'))
                            _p1 = np.asarray((_sd or {}).get('vals1', []),
                                             dtype=float)
                            _p2 = np.asarray((_sd or {}).get('vals2', []),
                                             dtype=float)
                            _parts1.append(_p1)
                            _parts2.append(_p2)
                            _rc = np.asarray(color_arr[r][col_i], dtype=float)
                            _row_mc1.append(np.tile(_rc, (len(_p1), 1)))
                            _row_mc2.append(np.tile(_rc, (len(_p2), 1)))
                        m_v1 = np.concatenate(_parts1)
                        m_v2 = np.concatenate(_parts2)
                        m_mc = (
                            np.concatenate(_row_mc1) if _row_mc1 else np.zeros((0, 3)),
                            np.concatenate(_row_mc2) if _row_mc2 else np.zeros((0, 3)))
                        m_i1 = m_i2 = None
                    if len(m_v1) == 0 and len(m_v2) == 0:
                        continue
                    _bottom_ylim_ref = (bottom_margin_ylim if bottom_margin_ylim is not None
                                        else ylim)
                    plot_compare_means(
                        col_summ_ax, m_v1, m_v2, col_color,
                        labels=_margin_labels,
                        paired_ids1=m_i1 if m_i1 else None,
                        paired_ids2=m_i2 if m_i2 else None,
                        marker_colors=m_mc,
                        confidence=confidence, n_boot=n_boot,
                        ylim=_bottom_ylim_ref,
                        **_cm_kw)

        # Add legends to margin axes where linestyles were used to distinguish
        # overlapping same-color lines (i.e. the corresponding cmap was None).
        _leg_fs = plt.rcParams.get('xtick.labelsize',
                                   plt.rcParams.get('font.size', 8))
        if col_cmap is None and n_data_cols > 1 and _has_right:
            _leg_title = col_var if col_var else None
            for _rax in self.display.right_col:
                if _rax is not None:
                    _handles, _labels = _rax.get_legend_handles_labels()
                    if _handles:
                        _rax.legend(_handles, _labels, title=_leg_title,
                                    fontsize=_leg_fs, title_fontsize=_leg_fs,
                                    loc='best', framealpha=0.7)
        if row_cmap is None and n_data_rows > 1 and _has_bottom:
            _leg_title = row_var if row_var else None
            for _bax in self.display.bottom_row:
                if _bax is not None:
                    _handles, _labels = _bax.get_legend_handles_labels()
                    if _handles:
                        _bax.legend(_handles, _labels, title=_leg_title,
                                    fontsize=_leg_fs, title_fontsize=_leg_fs,
                                    loc='best', framealpha=0.7)

        # ---------------------------------------------------------------- #
        # Format and label                                                  #
        # ---------------------------------------------------------------- #
        _xlabel = xlabel if xlabel is not None else xvar
        _ylabel = ylabel if ylabel is not None else (yvar or '')
        if plot_type == 'trajectory2d':
            _xlabel, _ylabel = 'x', 'y'

        # Default axis limits for saccade line plots.
        if object == 'saccade' and plot_type == 'line':
            if xlim is None:
                xlim = (-180, 180) if rad2deg else (-np.pi, np.pi)
            if ylim is None:
                ylim = (0.5, -0.25)

        # For compare_means: expand ylim to match the in-axis bracket headroom
        # that plot_compare_means added on each cell.  All main axes must share
        # the same expanded top so format() sets them consistently.
        # Save the original (unexpanded) ylim so the post-format block can
        # position spines at the original data bottom.
        _cm_original_ylim = ylim
        if plot_type == 'compare_means':
            _needed_tops = [
                v['summary_dict']['needed_ymax']
                for v in cell_summary_cache.values()
                if (v.get('summary_dict') or {}).get('needed_ymax') is not None
            ]
            _needed_bots = [
                v['summary_dict']['needed_ymin']
                for v in cell_summary_cache.values()
                if (v.get('summary_dict') or {}).get('needed_ymin') is not None
            ]
            if (_needed_tops or _needed_bots) and ylim is not None:
                _yl_lo, _yl_hi = min(ylim), max(ylim)
                _global_top = max(_needed_tops) if _needed_tops else _yl_hi
                _global_bot = min(_needed_bots) if _needed_bots else _yl_lo
                _yl_lo = min(_yl_lo, _global_bot)
                _yl_hi = max(_yl_hi, _global_top)
                ylim = (_yl_lo, _yl_hi) if ylim[0] <= ylim[1] \
                       else (_yl_hi, _yl_lo)

        self.display.format(
            xlim=xlim, ylim=ylim, xlabel=_xlabel, ylabel=_ylabel,
            xticks=xticks, yticks=yticks, logx=logx, logy=logy)

        # For trajectory2d margins using the mean_line path, the main panel
        # axes have their limits set internally by plot_trajectory2d (not via
        # format()), so the margin axes need to be synced manually.
        if plot_type == 'trajectory2d' and 'trajectory2d' in (
                right_margin_types + bottom_margin_types):
            _ref_ax = self.display.trace_axes.flat[0]
            _main_xlim = _ref_ax.get_xlim()
            _main_ylim = _ref_ax.get_ylim()
            if _has_right:
                for _max in self.display.right_col:
                    if _max is not None:
                        # circ_hist arcs manage their own limits; skip those.
                        if not getattr(_max, '_traj2d_circ_hist', False):
                            _max.set_xlim(_main_xlim)
                            _max.set_ylim(_main_ylim)
                            _max.set_aspect('equal', adjustable='box')
            if _has_bottom:
                for _bax in self.display.bottom_row:
                    if _bax is not None:
                        if not getattr(_bax, '_traj2d_circ_hist', False):
                            _bax.set_xlim(_main_xlim)
                            _bax.set_ylim(_main_ylim)
                            _bax.set_aspect('equal', adjustable='box')
        # Sync non-trajectory2d margin axes shared axis to match the main panels.
        # Right margin shares the y-axis with the main panels.
        # Bottom margin shares the x-axis with the main panels.
        # When the margin uses the same plot type as the main panels, the
        # independent axis is also synced; otherwise it is left auto-scaled.
        # The user can override any axis with right_margin_xlim / right_margin_ylim
        # / bottom_margin_xlim / bottom_margin_ylim.
        if _has_right or _has_bottom:
            # Union of all main-cell axis limits for consistent reference.
            _all_ref = list(self.display.trace_axes.flat)
            _main_xlim = (min(a.get_xlim()[0] for a in _all_ref),
                          max(a.get_xlim()[1] for a in _all_ref))
            _main_ylim = (min(a.get_ylim()[0] for a in _all_ref),
                          max(a.get_ylim()[1] for a in _all_ref))
            _same_type_right = bool(right_margin_types) and all(
                t == plot_type for t in right_margin_types)
            _same_type_bottom = bool(bottom_margin_types) and all(
                t == plot_type for t in bottom_margin_types)

            if _has_right and 'trajectory2d' not in right_margin_types:
                for _rax in self.display.right_col:
                    if _rax is None:
                        continue
                    # Shared y-axis.  For compare_means, expand the margin
                    # ylim by 20% (same headroom added per-cell inside
                    # plot_compare_means) so the in-axis bracket is visible.
                    _rm_ylim = right_margin_ylim if right_margin_ylim is not None else _main_ylim
                    if (plot_type == 'compare_means'
                            and 'compare_means' in right_margin_types
                            and right_margin_ylim is not None):
                        _rm_r = abs(_rm_ylim[1] - _rm_ylim[0])
                        _rm_lo, _rm_hi = min(_rm_ylim), max(_rm_ylim)
                        _rm_ylim = (_rm_lo - _rm_r * 0.20, _rm_hi + _rm_r * 0.20) if _rm_ylim[0] <= _rm_ylim[1] \
                                   else (_rm_hi + _rm_r * 0.20, _rm_lo - _rm_r * 0.20)
                    _rax.set_ylim(_rm_ylim)
                    # Independent x-axis.
                    if right_margin_xlim is not None:
                        _rax.set_xlim(right_margin_xlim)
                    elif _same_type_right:
                        _rax.set_xlim(_main_xlim)
                    else:
                        _rax.autoscale(enable=True, axis='x')
                        _rax.relim()
                        _rax.autoscale_view(scalex=True, scaley=False)

            if _has_bottom and 'trajectory2d' not in bottom_margin_types:
                for _bax in self.display.bottom_row:
                    if _bax is None:
                        continue
                    # Shared x-axis.
                    _bax.set_xlim(
                        bottom_margin_xlim if bottom_margin_xlim is not None
                        else _main_xlim)
                    # Independent y-axis.  For compare_means, expand the
                    # margin ylim by 20% to accommodate the in-axis bracket.
                    if bottom_margin_ylim is not None:
                        _bm_lim = bottom_margin_ylim
                        if (plot_type == 'compare_means'
                                and 'compare_means' in bottom_margin_types):
                            _bm_r = abs(_bm_lim[1] - _bm_lim[0])
                            _bm_lo, _bm_hi = min(_bm_lim), max(_bm_lim)
                            _bm_lim = (_bm_lo - _bm_r * 0.20, _bm_hi + _bm_r * 0.20) if _bm_lim[0] <= _bm_lim[1] \
                                      else (_bm_hi + _bm_r * 0.20, _bm_lo - _bm_r * 0.20)
                        _bax.set_ylim(_bm_lim)
                    elif _same_type_bottom:
                        _bax.set_ylim(_main_ylim)
                    else:
                        _bax.autoscale(enable=True, axis='y')
                        _bax.relim()
                        _bax.autoscale_view(scalex=False, scaley=True)
                        # relim() ignores PolyCollection (fill_between CI bands),
                        # so expand the ylim manually to cover them.
                        _y0, _y1 = _bax.get_ylim()
                        for _coll in _bax.collections:
                            try:
                                _verts = _coll.get_paths()
                                for _p in _verts:
                                    _ys = _p.vertices[:, 1]
                                    _y0 = min(_y0, float(_ys.min()))
                                    _y1 = max(_y1, float(_ys.max()))
                            except Exception:
                                pass
                        # Normalise: if format() applied an inverted ylim the
                        # autoscaled values may still be ordered (high, low).
                        if _y0 > _y1:
                            _y0, _y1 = _y1, _y0
                        _bax.set_ylim(_y0, _y1)

        # Override the auto-assigned ylabel/xlabel on histogram/pdf margin axes:
        # format() propagates the main plot's y/x label to margin axes, but
        # histogram/pdf margins show counts, not the original variable.
        _count_label = '%' if probability else 'count'
        _count_bottom_types = {'histogram', 'pdf'}
        _count_right_types = {'histogram', 'pdf'}
        if _has_bottom and _count_bottom_types & set(bottom_margin_types):
            # Only label the leftmost bottom-margin axis (the one that has ticks).
            _baxes = [a for a in self.display.bottom_row if a is not None]
            if _baxes:
                _baxes[0].set_ylabel(_count_label)
        if _has_right and _count_right_types & set(right_margin_types):
            # Only label the bottommost right-margin axis (the one that has ticks).
            _raxes = [a for a in self.display.right_col if a is not None]
            if _raxes:
                _raxes[-1].set_xlabel(_count_label)
        self.display.label_margins(row_vals, row_var, col_vals, col_var,
                                   bottom_scale=1.0)

        # Post-format despine for compare_means: trim now that yticks are set.
        # Bottom spines are shown on ALL rows (not just the bottom row) so that
        # the axis line is visible below the mean annotations.
        # Right-margin axes keep their left spine when right_margin_ylim differs
        # from the main ylim (i.e. they show an independent y-scale).
        if plot_type == 'compare_means':
            _n_trace_rows = self.display.trace_axes.shape[0]
            # Derive compare variable name from compare_subsets keys for default xlabel.
            _cs_keys = list(dict.fromkeys(
                k for cs in compare_subsets for k in cs.keys()))
            _compare_var_name = ', '.join(_cs_keys)
            _main_xlabel  = compare_xlabel        if compare_xlabel        is not None else _compare_var_name
            _right_xlabel = right_margin_xlabel   if right_margin_xlabel   is not None else col_var
            _bottom_xlabel= bottom_margin_xlabel  if bottom_margin_xlabel  is not None else row_var
            # Spine y-position: original data minimum (inside expanded ylim buffer).
            _cm_orig_spine_y = (min(_cm_original_ylim)
                                if _cm_original_ylim is not None else None)
            # --- Main grid axes ---
            for _ri, _row in enumerate(self.display.trace_axes):
                for _ci, _tax in enumerate(_row):
                    while isinstance(_tax, np.ndarray):
                        _tax = _tax[0]
                    _is_left = (_ci == 0)
                    # Set xticks BEFORE despine so trim clips to [0, 1] correctly.
                    _tax.set_xticks([0, 1])
                    # sbn.despine: left=True hides left spine, left=False keeps it.
                    sbn.despine(ax=_tax, trim=True, left=not _is_left)
                    # Position bottom spine at original data minimum.
                    _tax.spines['bottom'].set_visible(True)
                    if _cm_orig_spine_y is not None:
                        _tax.spines['bottom'].set_position(('data', _cm_orig_spine_y))
                    else:
                        _tax.spines['bottom'].set_position(('outward', _CM_SPINE_PTS))
                    labelbottom = (_ri == _n_trace_rows - 1)
                    _tax.tick_params(axis='x', which='both', bottom=True,
                                     labelbottom=labelbottom, pad=_CM_TICK_PAD)
                    if labelbottom and _main_xlabel:
                        _tax.set_xlabel(_main_xlabel)
            # Show left spine on margin axes only when they have a different scale.
            _rm_has_own_scale = (right_margin_ylim is not None
                                 or right_margin_yticks is not None)
            _bm_has_own_scale = (bottom_margin_ylim is not None
                                 or bottom_margin_yticks is not None)
            # Ylabel for margin plots (difference-of-differences or difference-of-means).
            _margin_ylabel = (r'$\Delta(\Delta)$' if margin_mode == 'diff_of_diffs'
                              else r'$\Delta$')
            # --- Right-margin axes ---
            if _has_right:
                _right_nonempty = [a for a in self.display.right_col if a is not None]
                for _rax in self.display.right_col:
                    if _rax is not None:
                        _is_bottom_right = (_rax is _right_nonempty[-1])
                        # Set ticks BEFORE despine so trim clips correctly.
                        _rax.set_xticks([0, 1])
                        if right_margin_yticks is not None:
                            _rm_yt, _rm_ytl = right_margin_yticks
                            _rax.set_yticks(_rm_yt)
                            _rax.set_yticklabels([str(v) for v in _rm_ytl])
                        sbn.despine(ax=_rax, trim=True, left=not _rm_has_own_scale)
                        # sbn.despine(trim=True) internally filters yticks to those
                        # within get_ylim(), which may exclude right_margin_yticks
                        # when ylim comes from the main panels.  Re-apply now.
                        if right_margin_yticks is not None:
                            _rax.set_yticks(_rm_yt)
                            _rax.set_yticklabels([str(v) for v in _rm_ytl])
                            _rax.spines['left'].set_bounds(
                                min(_rm_yt), max(_rm_yt))
                        # format() called _hide_spine('left') on non-leftmost axes,
                        # setting tick_params(left=False, labelleft=False). Re-enable
                        # y-ticks now that the left spine has been restored by despine.
                        if _rm_has_own_scale:
                            _rax.tick_params(axis='y', which='both',
                                             left=True, labelleft=True)
                            _rax.set_ylabel(_margin_ylabel)
                        # Re-set xticks in case despine filtered them.
                        _rax.set_xticks([0, 1])
                        _rax.spines['bottom'].set_visible(True)
                        _rm_spine_y = (min(right_margin_ylim)
                                       if right_margin_ylim is not None else None)
                        if _rm_spine_y is not None:
                            _rax.spines['bottom'].set_position(('data', _rm_spine_y))
                        else:
                            _rax.spines['bottom'].set_position(('outward', _CM_SPINE_PTS))
                        # X-tick labels and xlabel: only on the bottom-most right-margin
                        # axis (all share the same x-axis structure, no need to repeat).
                        _rax.set_xticklabels([str(v) for v in list(col_vals)[:2]])
                        _rax.tick_params(axis='x', which='both', bottom=True,
                                         labelbottom=_is_bottom_right, pad=_CM_TICK_PAD)
                        if _is_bottom_right and _right_xlabel:
                            _rax.set_xlabel(_right_xlabel)
            # --- Bottom-margin axes ---
            if _has_bottom:
                for _bi, _bax in enumerate(self.display.bottom_row):
                    if _bax is not None:
                        _bm_is_left = (_bi == 0)
                        # Set ticks BEFORE despine so trim clips correctly.
                        _bax.set_xticks([0, 1])
                        if bottom_margin_yticks is not None:
                            _bm_yt, _bm_ytl = bottom_margin_yticks
                            _bax.set_yticks(_bm_yt)
                            _bax.set_yticklabels([str(v) for v in _bm_ytl])
                        # Left spine: only on first bottom-margin subplot (they
                        # share the same y-axis, so later ones need no spine).
                        _show_bm_left = _bm_is_left and _bm_has_own_scale
                        sbn.despine(ax=_bax, trim=True, left=not _show_bm_left)
                        # Re-apply yticks after despine for the same reason as
                        # the right margin: seaborn may have filtered them.
                        if bottom_margin_yticks is not None:
                            _bax.set_yticks(_bm_yt)
                            _bax.set_yticklabels([str(v) for v in _bm_ytl])
                            _bax.spines['left'].set_bounds(
                                min(_bm_yt), max(_bm_yt))
                        # For the first bottom-margin axis, re-enable ytick visibility
                        # (format() may have hidden them) and set the ylabel.
                        if _bm_is_left and _bm_has_own_scale:
                            _bax.tick_params(axis='y', which='both',
                                             left=True, labelleft=True)
                            _bax.set_ylabel(_margin_ylabel)
                        # Re-set xticks in case despine filtered them.
                        _bax.set_xticks([0, 1])
                        _bax.spines['bottom'].set_visible(True)
                        _bm_spine_y = (min(bottom_margin_ylim)
                                       if bottom_margin_ylim is not None else None)
                        if _bm_spine_y is not None:
                            _bax.spines['bottom'].set_position(('data', _bm_spine_y))
                        else:
                            _bax.spines['bottom'].set_position(('outward', _CM_SPINE_PTS))
                        # X-tick labels: row variable values
                        _bax.set_xticklabels([str(v) for v in list(row_vals)[:2]])
                        _bax.tick_params(axis='x', which='both', bottom=True,
                                         labelbottom=True, pad=_CM_TICK_PAD)
                        if _bottom_xlabel:
                            _bax.set_xlabel(_bottom_xlabel)
            # Brackets are now inside axis bounds, so less hspace is needed.
            _sh = getattr(self.display, '_subplot_height', 2.0)
            _hspace = 0.8 / max(_sh, 0.5)
            self.display._hspace = _hspace
            self.display.fig.subplots_adjust(hspace=_hspace)
            # When right-margin axes carry a ylabel, shift only those axes
            # rightward so the label doesn't overlap the adjacent main column.
            # This avoids touching wspace (which would affect all columns).
            if _has_right and _rm_has_own_scale:
                self.display._right_col_pad_pts = 40
            self.display._apply_right_col_pad()

        # For trajectory2d, replace the Cartesian spines with a polar ring
        # on every trace axis and every margin axis.
        if plot_type == 'trajectory2d':
            _circ_hist = plot_kwargs.get('circ_hist', False)
            # Labels appear only on the first (top-left) trace subplot.
            _first_polar = True
            for _tax in self.display.trace_axes.flat:
                while isinstance(_tax, np.ndarray):
                    _tax = _tax[0]
                _add_polar_ring(_tax, has_circ_hist=bool(_circ_hist),
                                show_labels=_first_polar)
                _first_polar = False
            # Margin axes never show labels.
            if _has_right:
                for _rax in self.display.right_col:
                    if _rax is not None:
                        _add_polar_ring(_rax, has_circ_hist=bool(_circ_hist),
                                        show_labels=False)
            if _has_bottom:
                for _bax in self.display.bottom_row:
                    if _bax is not None:
                        _add_polar_ring(_bax, has_circ_hist=bool(_circ_hist),
                                        show_labels=False)

        # ---------------------------------------------------------------- #
        # Return                                                            #
        # ---------------------------------------------------------------- #
        if return_stats:
            import pandas as pd
            _stats_df = pd.DataFrame(_stats_rows) if _stats_rows else None
            return (self.display, _stats_df)
        return self.display

    def plot_saccades(self, col_var, row_var, output_var='camera_heading', time_var='time', start=0, 
                      stop=.5, row_cmap=None, col_cmap=None, 
                      fig=None, right_margin=True, bottom_margin=True,
                      xlim=(-np.pi, np.pi), ylim=(.5, -.5), xticks=None, yticks=None, 
                      positive_amplitude=False, scale=1.5, reversal_split=False,
                      saccade_var='arr_relative', min_speed=350, max_speed=np.inf,
                      mean_bins=25, bins=100, line_color='k', line_alpha=.25,
                      color='k',
                      **query_kwargs):
        """Plot saccade data in one big grid as in the plot summary below.
        
        The color for each subplot is determined by the average of the column and 
        row colors: color = sqrt(mean([col_color^2, row_color^2])). This is the 
        proper way to average two colors and should generate a unique color for each 
        plot. 

        Parameters
        ----------
        col_var, row_var : str
            The variable names to paramaterize along columns and rows of the grid.
        time_var : str, default='time'
            The time variable to use. So far there are two options: 1) time from the start 
            of the saccade or 2) time from the peak velocity.
        start, stop : float, default=0., np.inf.
            The start and stop times to include in the saccade.
        row_cmap, col_cmap : func or array-like, default=None
            The colormap to use for colorizing along the rows or columns. If both are
            supplied, the product of the two colors is used for each subplot. If a 
            function is supplied, it will be applied to the corresponding col_var or row_var.
            If a list is supplied, it must have as many elements as the corresponding column or
            row.
        right_margin, bottom_margin : bool, default=True
            Whether to plot the means from each subplot in the margin to the right
            or bottom of the axes. A colormap (row_cmap or col_cmap) must be provided
            in order to distinguish individual traces.
        xlim, ylim : tuple=(min, max), default=(-np.pi, np.pi), (-.5, .5)
            The tuple of the minimum and maximum values for that dimension.
        xticks, yticks : list, default=None
            Specify the list of ticks on the x- or y-axis.
        positive_amplitude : bool, default=False
            Whether to normalize the traces so that they all end positive.
        reversal_split : bool, default=False
            Whether to separately plot saccades that are reversing direction from those going in
            the same direction.
        saccade_var : str, default='arr_relative'
            The saccade variable to plot. The default is the main heading relative to the start
            of the saccade. Choose one from the following: arr_relative, velocity, acceleration.
        summary_func : callable, default=np.nanmean
            The function to plot summarizing the data in each subplot.
        mean_bins : int, default=25
            The number of bins to use for bin averaging the saccade time series.
        min_speed, max_speed : float, default=350, np.inf
            The minimum and maximum peak speed to include in the saccades here.
        line_color : str, default='k'
            The color to use for plotting individual saccades.
        **query_kwargs
            These get passed to the query 
        """
        import warnings
        warnings.warn(
            "plot_saccades() is deprecated and will be removed in a future "
            "version.  Use plot(object='saccade', plot_type='line') instead.",
            DeprecationWarning, stacklevel=2,
        )
        if fig is not None:
            warnings.warn(
                "The 'fig' parameter is not forwarded to plot().",
                DeprecationWarning, stacklevel=2,
            )
        if reversal_split:
            warnings.warn(
                "'reversal_split' is not supported by plot() and will be "
                "ignored.",
                DeprecationWarning, stacklevel=2,
            )
        if start != 0 or stop != 0.5:
            warnings.warn(
                "The 'start'/'stop' time-window parameters are not supported "
                "by plot(); the full saccade time series will be shown.",
                DeprecationWarning, stacklevel=2,
            )
        pk = {}
        pk['alpha'] = line_alpha
        pk['trace_color'] = line_color
        pk['mean_bins'] = mean_bins
        eff_max_speed = None if max_speed == np.inf else max_speed
        return self.plot(
            saccade_var, time_var, col_var=col_var, row_var=row_var,
            object='saccade', plot_type='line',
            row_cmap=row_cmap, col_cmap=col_cmap, color=color,
            right_margin=right_margin, bottom_margin=bottom_margin,
            xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks,
            positive_amplitude=positive_amplitude,
            min_speed=min_speed, max_speed=eff_max_speed,
            scale=scale, bins=bins,
            plot_kwargs=pk, **query_kwargs,
        )

    def plot_saccade_dynamics(self, col_var, row_var, row_cmap=None, col_cmap=None,
                              time_var='time',
                              fig=None, right_margin=True, bottom_margin=True, scale=1.5, 
                              reversal_split=False, output='start', 
                              heading_var='camera_heading', reference_var=None, saccade_var='amplitude',
                              bins=21, min_speed=50, max_speed=3000, scatter=False, 
                              xlim=None, ylim=None, xticks=None, yticks=None, display=None,
                              log_cmap=False, jitter_std=0.1, correlation=False,
                              color='k',
                              plot_kwargs={},
                              **query_kwargs):
        """Plot saccade position (x) and amplitude (y) in a grid as in the plot summary below.
        
        These plots will allow us to measure the 

        Parameters
        ----------
        col_var, row_var : str
            The variable names to paramaterize along columns and rows of the grid.
        time_var : str, default='time'
            The time variable to use.
        right_margin, bottom_margin : bool, default=True
            Whether to plot the means from each subplot in the margin to the right
            or bottom of the axes. A colormap (row_cmap or col_cmap) must be provided
            in order to distinguish individual traces.
        output : str, default='start'
            Whether to use the start or stop position of the saccade.
        heading_var : str, default='camera_heading'
            The variable to use for the saccade heading.
        reference_var : str, default=None
            If provided, the saccades will be aligned to the value of this variable.
        saccade_var : str, default='amplitude',
            Which saccade variable to plot. Options include 'amplitude' and 'peak_velocity'.
        bins : int, default=21
            The number of bins to use for the 2D histogram.
        min_speed, max_speed : float, default=350, np.inf
            The minimum and maximum peak speed to include in the saccades here.
        scatter : bool, default=False
            Whether to plot the data as a scatter plot or a 2D histogram.
        jitter_std : float, default=0.05
            The standard deviation of the jitter to add to each point for visualization.
        display : SummaryDisplay, default=None
            If provided, use this display instead of making a new one.
        **query_kwargs
            These get passed to the query 
        """
        import warnings
        warnings.warn(
            "plot_saccade_dynamics() is deprecated and will be removed in a "
            "future version.  Use "
            "plot(object='saccade', plot_type='scatter'/'hist2d') instead.",
            DeprecationWarning, stacklevel=2,
        )
        if fig is not None:
            warnings.warn(
                "The 'fig' parameter is not forwarded to plot().",
                DeprecationWarning, stacklevel=2,
            )
        if reversal_split:
            warnings.warn(
                "'reversal_split' is not supported by plot() and will be "
                "ignored.",
                DeprecationWarning, stacklevel=2,
            )
        if reference_var is not None:
            warnings.warn(
                f"'reference_var' ({reference_var!r}) is not supported by "
                "plot() and will be ignored.",
                DeprecationWarning, stacklevel=2,
            )
        if heading_var != 'camera_heading':
            warnings.warn(
                f"'heading_var={heading_var!r}' is not supported; "
                "saccade start_angle / stop_angle will be used directly.",
                DeprecationWarning, stacklevel=2,
            )
        xvar = 'start_angle' if output == 'start' else 'stop_angle'
        eff_plot_type = 'scatter' if scatter else 'hist2d'
        pk = dict(plot_kwargs)
        pk.setdefault('jitter_std', jitter_std)
        pk.setdefault('correlation', correlation)
        pk.setdefault('log_cmap', log_cmap)
        return self.plot(
            xvar, saccade_var, col_var=col_var, row_var=row_var,
            object='saccade', plot_type=eff_plot_type,
            row_cmap=row_cmap, col_cmap=col_cmap, color=color,
            right_margin=right_margin, bottom_margin=bottom_margin,
            xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks,
            display=display,
            scale=scale, bins=bins,
            min_speed=min_speed, max_speed=max_speed,
            rad2deg=True,
            plot_kwargs=pk, **query_kwargs,
        )

    def main_sequence_analysis(self, group_var='bg_gain', cmap='viridis', scale=1,
                                subset=None, colors=None,
                                confidence=0.84, n_boot=1000,
                                jitter_std=0.1, alpha=0.5):
        """Plot the relation between saccade peak velocity, duration, and magnitude.

        Draws a 3×2 fixed-layout figure:

        - Top-left / middle-left: scatter plots of magnitude vs. duration and
          magnitude vs. peak speed for every saccade, colored by group.
        - Top-right / middle-right: bootstrap CI strip plots of per-trial means
          per group, sharing the y-axis with the corresponding scatter panels.
        - Bottom-left: bootstrap CI strip plot of magnitude per group.
        - Bottom-right: unused (hidden).

        Parameters
        ----------
        group_var : str, default='bg_gain'
            Trial variable used to form groups and color the data.
        cmap : str, callable, or list, default='viridis'
            Colormap for the groups.  Forwarded to :func:`resolve_colors`;
            accepts a named matplotlib colormap string, a callable, or an
            explicit list of colors equal in length to the number of groups.
        scale : float, default=1
            Figure size multiplier (base figure is 4×6 inches).
        subset : dict or None, default=None
            Filter conditions forwarded to every query call.
        colors : list, ndarray, or None, default=None
            Explicit list of ``len(group_vals)`` colors.  Overrides ``cmap``.
        confidence : float, default=0.84
            Confidence level for bootstrap CIs in the strip/margin plots.
        n_boot : int, default=1000
            Number of bootstrap resamples.
        jitter_std : float, default=0.1
            Standard deviation of Gaussian jitter on strip-plot positions.
        alpha : float, default=0.5
            Point opacity for scatter and strip-plot markers.
        """
        if subset is None:
            subset = {}

        # ---------------------------------------------------------------- #
        # Group values and colors                                           #
        # ---------------------------------------------------------------- #
        group_vals = get_grid_vals(self, group_var, subset)

        if colors is not None:
            colors = np.array([matplotlib.colors.to_rgb(c) for c in colors])
            if len(colors) != len(group_vals):
                raise ValueError(
                    f"colors has {len(colors)} entries but there are "
                    f"{len(group_vals)} group values.")
        else:
            color_arr = resolve_colors(cmap, None, group_vals, [None])
            colors = color_arr[:, 0, :]  # (n_groups, 3)

        # ---------------------------------------------------------------- #
        # Figure layout                                                     #
        # ---------------------------------------------------------------- #
        # Use SummaryDisplay: 2 data rows × 1 data col, with right and
        # bottom margins.  Total gridspec = 3 rows × 2 cols.
        display = SummaryDisplay(
            num_rows=3, num_cols=2,
            right_margin=True, bottom_margin=True,
            subplot_size=(2 * scale, 2 * scale),
            right_margin_ratio=0.5,
            bottom_margin_ratio=0.5,
        )
        self.display = display
        scatter_axes = display.trace_axes[:, 0]   # shape (2,): duration row, speed row
        right_col = display.right_col             # shape (2,): CI strip plots
        bottom_ax = display.bottom_row[0]         # magnitude strip plot

        # ---------------------------------------------------------------- #
        # Data-extraction and scatter draw pass                             #
        # ---------------------------------------------------------------- #
        dur_means = []    # per-group: 1-D array of per-trial mean durations
        speed_means = []  # per-group: 1-D array of per-trial mean peak speeds
        mag_means = []    # per-group: 1-D array of per-trial mean amplitudes

        for num, (group_val, color) in enumerate(zip(group_vals, colors)):
            cell_subset = dict(subset)
            cell_subset[group_var] = group_val

            peak_velo = abs(np.array(self.query(
                output='saccade_peak_velocity', sort_by=group_var,
                subset=cell_subset, skip_empty=True,
                same_size=False))) * (180. / np.pi)
            duration = np.array(self.query(
                output='saccade_duration', sort_by=group_var,
                subset=cell_subset, skip_empty=True,
                same_size=False))
            amplitude = abs(np.array(self.query(
                output='saccade_amplitude', sort_by=group_var,
                subset=cell_subset, skip_empty=True,
                same_size=False)))

            sizes = np.unique([arr.size for arr in amplitude])
            same_sizes = len(sizes) == 1

            # Accumulate per-trial means for margin bootstrap CIs.
            for sac_var, storage in zip(
                    [peak_velo, duration, amplitude],
                    [speed_means, dur_means, mag_means]):
                if same_sizes:
                    avg = np.nanmean(sac_var, axis=1)
                else:
                    avg = np.array([np.nanmean(v) for v in sac_var])
                storage.append(avg)

            # Scatter: amplitude (x) vs duration / peak speed (y).
            for ys, ax in zip([duration, peak_velo], scatter_axes):
                if len(sizes) > 1:
                    for yvals, amp in zip(ys, amplitude):
                        ax.scatter(amp, yvals, color=color,
                                   marker='.', edgecolors='none', alpha=alpha)
                else:
                    ys_flat = np.squeeze(ys)
                    amp_flat = np.squeeze(amplitude)
                    non_nan = ~(np.isnan(amp_flat) | np.isnan(ys_flat))
                    try:
                        corr, pval = scipy.stats.pearsonr(
                            amp_flat[non_nan], ys_flat[non_nan])
                        label = f"{corr:.2f} {sigAsterisk(pval)}"
                    except Exception:
                        label = ""
                    ax.scatter(amp_flat, ys_flat, color=color, marker='.',
                               edgecolors='none', alpha=alpha, label=label)

        for ax in scatter_axes:
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=6)

        # ---------------------------------------------------------------- #
        # Bootstrap CI margin draw pass                                     #
        # ---------------------------------------------------------------- #
        rng = np.random.default_rng()

        # Right-column strip plots (shared y-axis with scatter_axes).
        for ax, group_trial_means in zip(right_col[:2], [dur_means, speed_means]):
            for num, (per_trial, color) in enumerate(zip(group_trial_means, colors)):
                low, high = bootstrap_ci(
                    per_trial, np.nanmean,
                    confidence=confidence, n_boot=n_boot)
                mid = float(np.nanmean(per_trial))
                xjitter = rng.standard_normal(len(per_trial)) * jitter_std
                ax.scatter(num + xjitter, per_trial, color=color,
                           edgecolors='none', marker='o', zorder=3, alpha=alpha)
                ax.plot([num, num], [low, high], color='k', zorder=4)
                ax.scatter(num, mid, color='k', marker='o',
                           edgecolors='w', linewidths=2, zorder=5)

        # Bottom strip plot: magnitude (horizontal orientation).
        for num, (per_trial, color) in enumerate(zip(mag_means, colors)):
            low, high = bootstrap_ci(
                per_trial, np.nanmean,
                confidence=confidence, n_boot=n_boot)
            mid = float(np.nanmean(per_trial))
            yjitter = rng.standard_normal(len(per_trial)) * jitter_std
            bottom_ax.scatter(per_trial, num + yjitter, color=color,
                              edgecolors='none', marker='o', zorder=3, alpha=alpha)
            bottom_ax.plot([low, high], [num, num], color='k', zorder=4)
            bottom_ax.scatter(mid, num, color='k', marker='o',
                              edgecolors='w', linewidths=2, zorder=5)

        # ---------------------------------------------------------------- #
        # Formatting                                                        #
        # ---------------------------------------------------------------- #
        # x-axis: log scale, auto-located ticks, labelled as integer degrees
        # rounded to the nearest power of 2.  Auto-location lets matplotlib
        # reduce density when the figure is small.
        def _rad_to_pow2_deg(x, _pos):
            deg = x * 180. / np.pi
            if deg <= 0:
                return ''
            p = round(np.log2(deg))
            return str(int(2 ** p))

        _pow2_fmt = matplotlib.ticker.FuncFormatter(_rad_to_pow2_deg)

        for ax in [scatter_axes[0], scatter_axes[1], bottom_ax]:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(_pow2_fmt)

        for ax, ylim, yticks, ylabels in zip(
                [scatter_axes[0], scatter_axes[1], right_col[0], right_col[1]],
                [(.01, .5), (1, 2000), (.01, .5), (1, 2000)],
                [[.01, .1], [1, 10, 100, 1000], [.01, .1], [1, 10, 100, 1000]],
                [['0.01', '0.1'], ['1', '10', '100', '1000'],
                 ['0.01', '0.1'], ['1', '10', '100', '1000']]):
            ax.set_yscale('log')
            ax.set_ylim(ylim)
            ax.set_yticks(yticks, ylabels)

        for ax in [right_col[0], right_col[1]]:
            ax.minorticks_off()
        bottom_ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

        xmin = 1. * np.pi / 180.
        xmax = 128. * np.pi / 180.
        bottom_ax.set_xlim(xmin, xmax)

        class _BoundedLog2Locator(matplotlib.ticker.LogLocator):
            """LogLocator(base=2) that always includes the axis boundary ticks."""
            def __init__(self, forced):
                super().__init__(base=2)
                self._forced = list(forced)

            def tick_values(self, vmin, vmax):
                ticks = set(super().tick_values(vmin, vmax))
                ticks.update(v for v in self._forced if vmin <= v <= vmax)
                return sorted(ticks)

        for ax in [scatter_axes[0], scatter_axes[1], bottom_ax]:
            ax.xaxis.set_major_locator(_BoundedLog2Locator([xmin, xmax]))

        for ax, lbl in zip(scatter_axes,
                           ["duration (s)", r"peak speed ($\degree$/s)"]):
            for _side in ('top', 'right', 'bottom'):
                _hide_spine(ax, _side)
            ax.set_ylabel(lbl)

        right_col[0].sharey(scatter_axes[0])
        right_col[1].sharey(scatter_axes[1])
        scatter_axes[0].sharex(bottom_ax)
        scatter_axes[1].sharex(bottom_ax)

        _n_grp = len(group_vals)
        _strip_pad = 0.5

        # Shorten path-like string labels (e.g. full filenames) to basename
        # without extension for readable tick labels.
        def _shorten(v):
            s = str(v)
            if '/' in s or os.sep in s or s.endswith('.h5'):
                return os.path.splitext(os.path.basename(s))[0]
            return s
        group_labels = [_shorten(v) for v in group_vals]

        for _side in ('top', 'right', 'left', 'bottom'):
            _hide_spine(right_col[0], _side)
        right_col[0].set_xlim(-_strip_pad, _n_grp - 1 + _strip_pad)

        right_col[1].set_xticks(range(len(group_vals)), group_labels)
        right_col[1].set_xlabel(group_var.replace("_", " "))
        for _side in ('top', 'right', 'left'):
            _hide_spine(right_col[1], _side)
        right_col[1].set_xlim(-_strip_pad, _n_grp - 1 + _strip_pad)

        bottom_ax.set_yticks(range(len(group_vals)), group_labels)
        bottom_ax.set_ylim(-_strip_pad, _n_grp - 1 + _strip_pad)
        bottom_ax.set_xlabel(r"magnitude ($\degree$)")
        bottom_ax.set_ylabel(group_var.replace("_", " "))
        for _side in ('top', 'right'):
            _hide_spine(bottom_ax, _side)

        display._get_parent_figure().tight_layout()

    def plot_summary(self, xvar, yvar, col_var, row_var, 
                     row_cmap=None, col_cmap=None, fig=None,
                     right_margin=True, bottom_margin=True,
                     xlim=None, ylim=None, xticks=None, yticks=None, 
                     logx=False, logy=False, display=None,
                     summary_func=np.nanmean, xlabel=None, ylabel=None,
                     color='k',
                     plot_type='line', bins=100, use_density=False,
                     omit_too_fast=True,
                     confidence_interval=False, confidence=.84,
                     scale=1.5, plot_kwargs={}, **query_kwargs):
        """Plot experimental data in one big grid.
        
        The color for each subplot is determined by the average of the column and 
        row colors: color = sqrt(mean([col_color^2, row_color^2])). This is the 
        proper way to average two colors and should generate a unique color for each 
        plot. 

        todo: add option to define colors in HSV space so that the user can
        align rows and columns with independent channels. For instance, rows can
        correspond to different hues while columns correspond to saturations. 

        Parameters
        ----------
        xvar, yvar : str
            The variable names to plot along the x- and y-axis of each subplot.
        col_var, row_var : str
            The variable names to paramaterize along columns and rows of the grid.
        row_cmap, col_cmap : func or array-like, default=None
            The colormap to use for colorizing along the rows or columns. If both are
            supplied, the product of the two colors is used for each subplot. If a 
            function is supplied, it will be applied to the corresponding col_var or row_var.
            If a list is supplied, it must have as many elements as the corresponding column or
            row.
        right_margin, bottom_margin : bool, default=True
            Whether to plot the means from each subplot in the margin to the right
            or bottom of the axes. A colormap (row_cmap or col_cmap) must be provided
            in order to distinguish individual traces.
        xlim, ylim : tuple=(min, max), default=None
            The tuple of the minimum and maximum values for that dimension.
        xticks, yticks : list, default=None
            Specify the list of ticks on the x- or y-axis.
        logx, logy : bool, default=False
            Whether to format the x- or y-axis on a log scale.
        display : SummaryDisplay, default=None
            Option to provide a SummaryDisplay with the same subplot arrangement
            to allow superimposing different datasets.
        xlabel, ylabel : str, default=None
            Option to provide custom x- and y-labels.
        summary_func : callable, default=np.nanmean
            The function for generating the summary statistic of interest in the right and bottom
            margins. It's applied to the sets of traces plotted in the trace plots.
        plot_type : str, default='line'
            Whether to plot the data as a line plot ('line'), a 2D histogram ('hist2d'),
            or as 2d simulated trajectory ('trajectory2d').
        bins : int or tuple, default=100
            The bins parameter to pass to the 2D histogram.
        use_density : bool, default=False
            Whether to plot the density instead of the count in the 2d histogram.
        scale : float, default=1.5
            Scale parameter for setting the figsize.
        omit_too_fast : bool, default=True
            Assuming xvar represents fly heading data, omit segments that are impossible, replacing with NaNs so that aren't plotted.
        plot_kwargs : dict, default={}
            Additional keyword arguments to pass to the plotting function. These include:
                - 'color': the color of the lines in the trace plots
                - 'alpha': the transparency of the lines in the trace plots
                - 'linewidth': the width of the lines in the trace plots
                - 'marker': the marker style for the scatter plots in the trace plots
                - 'markersize': the size of the markers in the scatter plots in the trace plots
                - 'edgecolor': the edge color of the markers in the scatter plots in the trace plots
                - 'cmap': the colormap to use for the 2D histogram
                - 'bins': the number of bins to use for the 2D histogram
        color : str or tuple, default='k'
            The color to use for the lines in the trace plots. If a colormap is provided
            it will be used instead.
        **query_kwargs
            These get passed to the query 
        """
        import warnings
        warnings.warn(
            "plot_summary() is deprecated and will be removed in a future version. "
            "Use plot() instead.",
            DeprecationWarning, stacklevel=2,
        )
        if fig is not None:
            warnings.warn(
                "The 'fig' parameter is not forwarded to plot(); "
                "a new figure is always created.",
                DeprecationWarning, stacklevel=2,
            )
        if omit_too_fast:
            pass  # omit_too_fast is now handled by omit_wrap=True inside plot()
        return self.plot(
            xvar, yvar, col_var=col_var, row_var=row_var,
            plot_type=plot_type, object='trial',
            row_cmap=row_cmap, col_cmap=col_cmap, color=color,
            right_margin=right_margin, bottom_margin=bottom_margin,
            xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks,
            logx=logx, logy=logy, display=display,
            summary_func=summary_func, xlabel=xlabel, ylabel=ylabel,
            scale=scale, bins=bins, density=use_density,
            confidence_interval=confidence_interval, confidence=confidence,
            plot_kwargs=plot_kwargs or {}, **query_kwargs,
        )

    def plot_histogram_summary(
            self, xvar, col_var, row_var, use_probability=False, bins=None,
            row_cmap=None, col_cmap=None, fig=None,
            right_margin=True, bottom_margin=True,
            xlim=None, xticks=None, logx=False, logy=False, display=None, ylim=None,
            summary_func=partial(scipy.stats.circmean, low=-np.pi, high=np.pi), xlabel=None,
            color='k',
            **query_kwargs):
        """Plot histgrams of data in one big grid arranged by two variables.
        
        The color for each subplot is determined by the average of the column and 
        row colors: color = sqrt(mean([col_color^2, row_color^2])). This is the 
        proper way to average two colors and should generate a unique color for each 
        plot. 

        todo: add option to define colors in HSV space so that the user can
        align rows and columns with independent channels. For instance, rows can
        correspond to different hues while columns correspond to saturations. 

        Parameters
        ----------
        xvar : str
            The variable name to use for the histograms.
        col_var, row_var : str
            The variable names to paramaterize along columns and rows of the grid.
        row_cmap, col_cmap : func or array-like, default=None
            The colormap to use for colorizing along the rows or columns. If both are
            supplied, the product of the two colors is used for each subplot. If a 
            function is supplied, it will be applied to the corresponding col_var or row_var.
            If a list is supplied, it must have as many elements as the corresponding column or
            row.
        right_margin, bottom_margin : bool, default=True
            Whether to plot the means from each subplot in the margin to the right
            or bottom of the axes. A colormap (row_cmap or col_cmap) must be provided
            in order to distinguish individual traces.
        xlim, ylim : tuple=(min, max), default=None
            The tuple of the minimum and maximum values for that dimension.
        xticks, yticks : list, default=None
            Specify the list of ticks on the x- or y-axis.
        logx, logy : bool, default=False
            Whether to format the x- or y-axis on a log scale.
        display : SummaryDisplay, default=None
            Option to provide a SummaryDisplay with the same subplot arrangement
            to allow superimposing different datasets.
        xlabel, ylabel : str, default=None
            Option to provide custom x- and y-labels.
        summary_func : callable, default=np.nanmean
            The function for generating the summary statistic of interest in the right and bottom
            margins. It's applied to the sets of traces plotted in the trace plots.
        **query_kwargs
            These get passed to the query 
        """
        import warnings
        warnings.warn(
            "plot_histogram_summary() is deprecated and will be removed in a "
            "future version.  Use plot(plot_type='histogram') instead.",
            DeprecationWarning, stacklevel=2,
        )
        if fig is not None:
            warnings.warn(
                "The 'fig' parameter is not forwarded to plot(); "
                "a new figure is always created.",
                DeprecationWarning, stacklevel=2,
            )
        return self.plot(
            xvar, yvar=None, col_var=col_var, row_var=row_var,
            plot_type='histogram', object='trial',
            row_cmap=row_cmap, col_cmap=col_cmap, color=color,
            right_margin=right_margin, bottom_margin=bottom_margin,
            xlim=xlim, ylim=ylim, xticks=xticks,
            logx=logx, logy=logy, display=display,
            summary_func=summary_func, xlabel=xlabel,
            bins=bins, probability=use_probability,
            **query_kwargs,
        )

    def save(self):
        """Persist in-memory changes for every trial to disk.

        Calls :meth:`TrackingTrial.save` on each trial.  Only trials with
        pending variable / attribute changes actually write to disk; others
        return immediately.

        Examples
        --------
        ::

            exp.add_dataset('my_var', arrays)   # one array per trial
            exp.save()                          # write all to Zarr stores
        """
        for trial in self.trials:
            trial.save()

    def load(self):
        """Reload each trial from its Zarr store, discarding in-memory state.

        Useful after an external process writes to the store.  Replaces
        ``self.trials`` in-place with freshly opened lazy handles.

        Examples
        --------
        ::

            exp.save()   # flush changes
            exp.load()   # reopen — safe for subsequent lazy access
        """
        loaded_trials = []
        for trial in self.trials:
            loaded_trial = type(trial).load(trial.filename)
            loaded_trials.append(loaded_trial)
        self.trials = loaded_trials

    def close(self):
        """Close all trials and release resources."""
        for trial in getattr(self, 'trials', []):
            if hasattr(trial, 'close'):
                trial.close()

    def __del__(self):
        self.close()



class SummaryDisplay():
    def __init__(self, num_rows=1, num_cols=1, right_margin=True, bottom_margin=True,
                 figsize=None, subplot_size=None,
                 right_margin_ratio=0.3, bottom_margin_ratio=0.3,
                 **fig_kwargs):
        """Setup a figure with a grid of subplots to iteratively populate.

        Parameters
        ----------
        num_rows, num_cols : int, default=1
            Total rows and columns including any margin row/column.
        right_margin, bottom_margin : bool, default=True
            Whether to reserve the last column/row as a margin summary axis.
        figsize : tuple or None
            Explicit ``(width, height)`` in inches.  Overrides ``subplot_size``
            when both are given.  Pass a pre-created ``fig=`` kwarg to skip
            figure creation entirely.
        subplot_size : tuple or None, default=(2, 2)
            ``(width_inches, height_inches)`` for each *data* subplot cell.
            The total figure size is computed automatically from the number of
            data rows/cols and the margin ratios.  Ignored when ``figsize`` or
            ``fig=`` is supplied.  Defaults to ``(2, 2)`` when neither
            ``subplot_size`` nor ``figsize`` is given.
        right_margin_ratio : float, default=0.3
            Width of the right-margin column as a fraction of one data
            subplot's width.  Only used when ``right_margin=True``.
        bottom_margin_ratio : float, default=0.3
            Height of the bottom-margin row as a fraction of one data
            subplot's height.  Only used when ``bottom_margin=True``.
        **fig_kwargs
            Extra keyword arguments forwarded to ``plt.figure()`` (e.g.
            ``dpi``).  Pass ``fig=<Figure>`` to supply an existing figure
            instead of creating one.
        """
        # make the figure and axes
        if num_rows == 0:
            num_rows = 1
        if num_cols == 0:
            num_cols = 1

        # Number of pure-data rows/cols (excluding margin row/col).
        n_data_rows = num_rows - (1 if bottom_margin else 0)
        n_data_cols = num_cols - (1 if right_margin else 0)

        _has_external_fig = 'fig' in fig_kwargs
        if _has_external_fig:
            self.fig = fig_kwargs.pop('fig')
        else:
            # Determine figsize from subplot_size when not given explicitly.
            if figsize is None:
                if subplot_size is None:
                    subplot_size = (2, 2)
                sw, sh = subplot_size
                fig_w = n_data_cols * sw + (right_margin_ratio * sw if right_margin else 0)
                fig_h = n_data_rows * sh + (bottom_margin_ratio * sh if bottom_margin else 0)
                figsize = (fig_w, fig_h)
            self.fig = plt.figure(figsize=figsize, **fig_kwargs)
        # Store subplot height so callers can compute hspace proportionally.
        if subplot_size is not None:
            self._subplot_height = float(subplot_size[1])
        elif figsize is not None and n_data_rows > 0:
            self._subplot_height = float(figsize[1]) / n_data_rows
        else:
            self._subplot_height = 2.0

        if isinstance(self.fig, plt.Figure):
            # make it a subfigure
            self.fig = self.fig.subfigures(1, 1)

        subplot_kwargs = fig_kwargs.copy()
        if 'figsize' in subplot_kwargs:
            subplot_kwargs.pop('figsize')

        # Build gridspec ratios so all data cells are equal and margin
        # cells are proportionally sized.
        wr = [1] * n_data_cols + ([right_margin_ratio] if right_margin else [])
        hr = [1] * n_data_rows + ([bottom_margin_ratio] if bottom_margin else [])
        gskw = subplot_kwargs.pop('gridspec_kw', {})
        gskw.setdefault('width_ratios', wr)
        gskw.setdefault('height_ratios', hr)
        subplot_kwargs['gridspec_kw'] = gskw

        self.axes = self.fig.subplots(num_rows, num_cols,
                                      **subplot_kwargs)
        if num_rows == 1 and num_cols == 1:
            self.axes = np.array([self.axes])[:, np.newaxis]
        elif num_rows == 1:
            self.axes = self.axes[np.newaxis, :]
        elif num_cols == 1:
            self.axes = self.axes[:, np.newaxis]
        # unless using the margins, all axes are for traces
        self.trace_axes = self.axes
        self.left = np.zeros(self.axes.shape, dtype=bool)
        self.bottom = np.copy(self.left)
        self.left[:, 0], self.bottom[-1, :] = True, True
        self.right_margin = right_margin
        # partition the right column from self.trace_axes
        if right_margin:
            if bottom_margin:
                self.right_col = self.axes[:-1, -1]
            else:
                self.right_col = self.axes[:, -1]
            self.trace_axes = self.trace_axes[:, :-1]
        else:
            self.right_row = []
        # partition the bottom row from self.trace_axes
        self.bottom_margin = bottom_margin
        if bottom_margin:
            if right_margin:
                self.bottom_row = self.axes[-1, :-1]
            else:
                self.bottom_row = self.axes[-1, :]
            self.trace_axes = self.trace_axes[:-1, :]
        else:
            self.bottom_row = []
        # if using both margins, avoid the corner margin
        if bottom_margin and right_margin:
            self.left[-1, -1] = False
            self.bottom[-1, -1] = False
            self.bottom[-2, -1] = True
            self.corner_ax = self.axes[-1, -1]
        # completely hide the corner_ax
        if hasattr(self, 'corner_ax'):
            self.corner_ax.set_visible(False)
        # store transform and artist references for dynamic updates
        self._margin_label_cids = []
        self._row_label_artists = None
        self._col_label_artists = None
        # let's add a re-entry guard to the update function
        self._updating = False
        self._bottom_scale = 1.0  # multiplier for bottom_bound in label_margins
        self._hspace = None            # optional hspace override for subplots_adjust
        self._right_col_pad_pts = 0   # pts to shift right-margin axes right (ylabel clearance)

    def _get_fig_size_inches(self):
        """Robustly compute size in inches for Figure or SubFigure."""
        if isinstance(self.fig, plt.Figure):
            # Figure supports get_size_inches
            w, h = self.fig.get_size_inches()
            return float(w), float(h)
        else:
            # SubFigure: derive from bbox and parent figure DPI
            parent = self._get_parent_figure()
            dpi = getattr(parent, 'dpi', 100.0)
            bbox = getattr(self.fig, 'bbox', None)
            if bbox is not None:
                return bbox.width / dpi, bbox.height / dpi
            # fallback to parent figure size
            w, h = parent.get_size_inches()
            return float(w), float(h)

    def _get_parent_figure(self):
        """Return the parent Figure for both Figure and SubFigure inputs."""
        return getattr(self.fig, 'figure', self.fig)

    def _get_coord_transform(self):
        """Return the correct normalized coordinate transform for figure/subfigure."""
        # SubFigure has transSubfigure; Figure has transFigure
        return getattr(self.fig, 'transSubfigure', self._get_parent_figure().transFigure)

    def _get_renderer(self):
        """Get a renderer; force a draw if necessary."""
        parent = self._get_parent_figure()
        canvas = getattr(parent, 'canvas', None)
        if canvas is None:
            return None
        try:
            return canvas.get_renderer()
        except Exception:
            try:
                canvas.draw()
                return canvas.get_renderer()
            except Exception:
                return None

    def _label_bboxes_in_subfig_coords(self, axes_list, which='x'):
        """Return label bboxes (x0, y0, x1, y1) in subfigure-normalized coords.
        which: 'x' for xlabel bboxes; 'y' for ylabel bboxes.
        """
        renderer = self._get_renderer()
        sub_trans = self._get_coord_transform()
        inv_sub = sub_trans.inverted()
        bboxes = []
        for ax in axes_list:
            while isinstance(ax, np.ndarray):
                ax = ax[0]
            label = ax.xaxis.label if which == 'x' else ax.yaxis.label
            try:
                bbox = label.get_window_extent(renderer=renderer)
            except Exception:
                # force a draw and retry once
                renderer = self._get_renderer()
                bbox = label.get_window_extent(renderer=renderer)
            # transform bbox corners from display to subfig coords
            p0 = inv_sub.transform((bbox.x0, bbox.y0))
            p1 = inv_sub.transform((bbox.x1, bbox.y1))
            x0 = min(p0[0], p1[0])
            y0 = min(p0[1], p1[1])
            x1 = max(p0[0], p1[0])
            y1 = max(p0[1], p1[1])
            bboxes.append((x0, y0, x1, y1))
        return np.array(bboxes)

    def _label_centers_in_subfig_coords(self, axes_list, which='x'):
        """Return label centers (x, y) in subfigure-normalized coords."""
        b = self._label_bboxes_in_subfig_coords(axes_list, which=which)
        if len(b) == 0:
            return np.empty((0, 2))
        centers = np.column_stack(((b[:, 0] + b[:, 2]) * 0.5, (b[:, 1] + b[:, 3]) * 0.5))
        return centers

    def _text_bbox_in_subfig_coords(self, text_artist):
        """Return (x0, y0, x1, y1) of a Text artist in subfigure-normalised coords."""
        renderer = self._get_renderer()
        inv_sub = self._get_coord_transform().inverted()
        try:
            bb = text_artist.get_window_extent(renderer=renderer)
        except Exception:
            renderer = self._get_renderer()
            bb = text_artist.get_window_extent(renderer=renderer)
        p0 = inv_sub.transform((bb.x0, bb.y0))
        p1 = inv_sub.transform((bb.x1, bb.y1))
        return min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])

    def _compute_row_spine_x(self, left_col, row_val_texts):
        """Return spine_x as the midpoint between row-value text right edges and ylabel left edges.

        Falls back to a fixed offset from left_bound if bboxes are not yet available.
        """
        y_b = self._label_bboxes_in_subfig_coords(left_col, which='y')
        ylabel_left = float(np.min(y_b[:, 0])) if len(y_b) else self.left_bound
        val_right_list = []
        for t in row_val_texts:
            try:
                x0, _, x1, _ = self._text_bbox_in_subfig_coords(t)
                val_right_list.append(x1)
            except Exception:
                pass
        val_right = float(np.max(val_right_list)) if val_right_list else 0.0
        return (val_right + ylabel_left) * 0.5

    def _compute_col_spine_y(self, bottom_row, col_val_texts):
        """Return spine_y as the midpoint between xlabel bottom edges and col-value text tops.

        Falls back to a fixed offset from bottom_bound if bboxes are not available.
        """
        x_b = self._label_bboxes_in_subfig_coords(bottom_row, which='x')
        xlabel_bottom = float(np.min(x_b[:, 1])) if len(x_b) else self.bottom_bound
        val_top_list = []
        for t in col_val_texts:
            try:
                _, y0, _, y1 = self._text_bbox_in_subfig_coords(t)
                val_top_list.append(y1)
            except Exception:
                pass
        val_top = float(np.min(val_top_list)) if val_top_list else 0.0
        return (val_top + xlabel_bottom) * 0.5

    def _apply_right_col_pad(self):
        """Shift right-margin axes rightward by _right_col_pad_pts points.

        Called after every subplots_adjust so that the automatic layout reset
        does not undo the shift.  Only the right-margin column is moved;
        the main-grid columns are unaffected.
        """
        if not self._right_col_pad_pts or not self.right_margin:
            return
        fig_w, _ = self._get_fig_size_inches()
        if fig_w <= 0:
            return
        pad_frac = self._right_col_pad_pts / (fig_w * 72.0)
        for ax in self.right_col:
            if ax is not None:
                pos = ax.get_position()
                ax.set_position([pos.x0 + pad_frac, pos.y0,
                                 pos.width, pos.height])

    def label_margins(self, row_vals=None, row_label=None, col_vals=None, col_label=None,
                      bottom_scale=1.0):
        """Add values and a label to indicate differences across rows/cols using actual label bboxes."""
        self._bottom_scale = bottom_scale
        left_col = self.trace_axes[:, 0]
        bottom_row = self.axes[-1]
        if self.right_margin:
            bottom_row = bottom_row[:-1]

        coord_trans = self._get_coord_transform()
        self._row_label_artists = None
        self._col_label_artists = None
        self.adjusted_left = False
        self.adjusted_bottom = False

        fig_width, fig_height = self._get_fig_size_inches()
        self.left_bound = 0
        self.bottom_bound = 0
        labelsize = plt.rcParams.get('axes.labelsize', 10)
        labelsize_conv = {'xx-small': 6, 'x-small': 7, 'small': 8, 'medium': 10,
                          'large': 12, 'x-large': 14, 'xx-large': 16}
        if labelsize in labelsize_conv:
            labelsize = labelsize_conv[labelsize]
        if row_label is not None:
            self.left_bound = 1.0 / fig_width   # fixed 1-inch left margin
            self.adjusted_left = True
        if col_label is not None:
            self.bottom_bound = 1.25 * self._bottom_scale / fig_height  # fixed 1.25-inch bottom
            self.adjusted_bottom = True
        if self.adjusted_left or self.adjusted_bottom:
            _adj_kw = {}
            if self._hspace is not None:
                _adj_kw['hspace'] = self._hspace
            self.fig.subplots_adjust(left=self.left_bound, bottom=self.bottom_bound, **_adj_kw)
            self._apply_right_col_pad()

        # ------------------------------------------------------------------ #
        # Row labels: separate Text artists, one per row, placed at x=0       #
        # ------------------------------------------------------------------ #
        if row_label is not None:
            # Force a draw so ylabel bboxes are available before we place anything.
            try:
                self._get_parent_figure().canvas.draw()
            except Exception:
                pass

            y_b = self._label_bboxes_in_subfig_coords(left_col, which='y')
            y_c = self._label_centers_in_subfig_coords(left_col, which='y')
            y_center = float(np.mean(y_c[:, 1])) if len(y_c) else 0.5
            ymins = float(np.min(y_c[:, 1])) if len(y_c) else 0.2
            ymaxs = float(np.max(y_c[:, 1])) if len(y_c) else 0.8

            # Row variable label (vertical, at far left)
            label = row_label.replace("_", " ")
            row_text = self.fig.text(0.0, y_center, label,
                                     va='center', ha='left', rotation='vertical',
                                     transform=coord_trans, fontsize=labelsize)

            # Per-row value texts: centered in the space between 0 and ylabel_left,
            # leaving room for the spine between val_text right edge and ylabel.
            row_val_texts = []
            row_iter = row_vals if row_vals is not None else []
            ylabel_left_min = float(np.min(y_b[:, 0])) if len(y_b) else self.left_bound
            val_x = ylabel_left_min * 0.5  # midpoint between figure left and ylabel
            for ax, val, yc in zip(left_col, row_iter,
                                   y_c[:, 1] if len(y_c) else [y_center]):
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                val_str = f"{val:.2f}" if isinstance(val, (float, int)) else str(val)
                t = self.fig.text(val_x, float(yc), val_str,
                                  va='center', ha='center', rotation='vertical',
                                  transform=coord_trans, fontsize=labelsize)
                row_val_texts.append(t)

            # Force another draw so row_val_texts have real bboxes
            try:
                self._get_parent_figure().canvas.draw()
            except Exception:
                pass

            spine_x = self._compute_row_spine_x(left_col, row_val_texts)
            tick_len = 0.06 / fig_width

            row_spine = matplotlib.lines.Line2D(
                [spine_x, spine_x], [ymins, ymaxs], lw=1, color='k')
            row_spine.set_transform(coord_trans)
            self.fig.add_artist(row_spine)

            row_ticks = []
            for yv in (y_c[:, 1] if len(y_c) else [y_center]):
                t = matplotlib.lines.Line2D(
                    [spine_x, spine_x + tick_len], [float(yv), float(yv)],
                    lw=1, color='k')
                t.set_transform(coord_trans)
                self.fig.add_artist(t)
                row_ticks.append(t)

            self._row_label_artists = {
                'text': row_text,
                'val_texts': row_val_texts,
                'spine': row_spine,
                'ticks': row_ticks,
            }

        # ------------------------------------------------------------------ #
        # Column labels: separate Text artists, one per col, placed at y=0    #
        # ------------------------------------------------------------------ #
        if col_label is not None:
            try:
                self._get_parent_figure().canvas.draw()
            except Exception:
                pass

            x_b = self._label_bboxes_in_subfig_coords(bottom_row, which='x')
            x_c = self._label_centers_in_subfig_coords(bottom_row, which='x')
            x_center = float(np.mean(x_c[:, 0])) if len(x_c) else 0.5
            xmins = float(np.min(x_c[:, 0])) if len(x_c) else 0.2
            xmaxs = float(np.max(x_c[:, 0])) if len(x_c) else 0.8

            label = col_label.replace("_", " ")
            # Pack col labels tightly just below the xlabel of the bottom row.
            # val_y: centre of column-value texts, 2 font-heights below the xlabel;
            # txt_y: bottom of the column-variable label, 2 font-heights below val_y.
            label_h_norm = labelsize / (fig_height * 72.0)
            xlabel_bottom_min = float(np.min(x_b[:, 1])) if len(x_b) else self.bottom_bound
            val_y = max(xlabel_bottom_min - 2.0 * label_h_norm, 3.0 * label_h_norm)
            txt_y = max(val_y - 2.0 * label_h_norm, 0.8 * label_h_norm)
            col_text = self.fig.text(x_center, txt_y, label,
                                     va='bottom', ha='center',
                                     transform=coord_trans, fontsize=labelsize)

            col_val_texts = []
            col_iter = col_vals if col_vals is not None else []
            for ax, val, xc in zip(bottom_row, col_iter,
                                   x_c[:, 0] if len(x_c) else [x_center]):
                while isinstance(ax, np.ndarray):
                    ax = ax[0]
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                t = self.fig.text(float(xc), val_y, val_str,
                                  va='center', ha='center',
                                  transform=coord_trans, fontsize=labelsize)
                col_val_texts.append(t)

            try:
                self._get_parent_figure().canvas.draw()
            except Exception:
                pass

            spine_y = self._compute_col_spine_y(bottom_row, col_val_texts)
            tick_len_y = 0.06 / fig_height

            col_spine = matplotlib.lines.Line2D(
                [xmins, xmaxs], [spine_y, spine_y], lw=1, color='k')
            col_spine.set_transform(coord_trans)
            self.fig.add_artist(col_spine)

            col_ticks = []
            for xv in (x_c[:, 0] if len(x_c) else [x_center]):
                t = matplotlib.lines.Line2D(
                    [float(xv), float(xv)], [spine_y, spine_y + tick_len_y],
                    lw=1, color='k')
                t.set_transform(coord_trans)
                self.fig.add_artist(t)
                col_ticks.append(t)

            self._col_label_artists = {
                'text': col_text,
                'val_texts': col_val_texts,
                'spine': col_spine,
                'ticks': col_ticks,
            }

        # connect dynamic updater to draw events (resize/redraw)
        if len(self._margin_label_cids) == 0:
            try:
                cid = self.fig.canvas.mpl_connect('draw_event', self._update_margin_labels)
                self._margin_label_cids.append(cid)
            except Exception:
                pass

    def _update_margin_labels(self, event=None):
        """Update margin label and tick positions on draw/resize using label bboxes."""
        if self._updating:
            return
        try:
            self._updating = True
            bottom_row = self.axes[-1]
            if self.right_margin:
                bottom_row = bottom_row[:-1]
            left_col = self.trace_axes[:, 0]
            fig_width, fig_height = self._get_fig_size_inches()
            labelsize = plt.rcParams.get('axes.labelsize', 10)
            labelsize_conv = {'xx-small': 6, 'x-small': 7, 'small': 8, 'medium': 10,
                            'large': 12, 'x-large': 14, 'xx-large': 16}
            if labelsize in labelsize_conv:
                labelsize = labelsize_conv[labelsize]
            if self.adjusted_left:
                self.left_bound = 1.0 / fig_width   # fixed 1-inch left margin
            if self.adjusted_bottom:
                self.bottom_bound = 1.25 * self._bottom_scale / fig_height  # fixed 1.25-inch bottom
            if self.adjusted_left or self.adjusted_bottom:
                _adj_kw = {}
                if self._hspace is not None:
                    _adj_kw['hspace'] = self._hspace
                self.fig.subplots_adjust(left=self.left_bound, bottom=self.bottom_bound, **_adj_kw)
                self._apply_right_col_pad()
            coord_trans = self._get_coord_transform()

            # Row updates
            if self._row_label_artists is not None:
                y_c = self._label_centers_in_subfig_coords(left_col, which='y')
                y_center = float(np.mean(y_c[:, 1])) if len(y_c) else 0.5
                ymins = float(np.min(y_c[:, 1])) if len(y_c) else 0.2
                ymaxs = float(np.max(y_c[:, 1])) if len(y_c) else 0.8
                self._row_label_artists['text'].set_position((0.0, y_center))
                self._row_label_artists['text'].set_transform(coord_trans)

                # Reposition row-value texts to match current ylabel centres
                val_texts = self._row_label_artists.get('val_texts', [])
                y_b = self._label_bboxes_in_subfig_coords(left_col, which='y')
                ylabel_left_min = float(np.min(y_b[:, 0])) if len(y_b) else self.left_bound
                val_x = ylabel_left_min * 0.5
                for i, (t, yc) in enumerate(zip(val_texts,
                                                 y_c[:, 1] if len(y_c) else [y_center])):
                    t.set_position((val_x, float(yc)))
                    t.set_transform(coord_trans)

                spine_x = self._compute_row_spine_x(left_col, val_texts)
                tick_len = 0.06 / fig_width
                self._row_label_artists['spine'].set_data([spine_x, spine_x], [ymins, ymaxs])
                self._row_label_artists['spine'].set_transform(coord_trans)
                ticks = self._row_label_artists['ticks']
                if len(ticks) == (len(y_c) if len(y_c) else 1):
                    vals = y_c[:, 1] if len(y_c) else [y_center]
                    for t, yv in zip(ticks, vals):
                        t.set_data([spine_x, spine_x + tick_len], [float(yv), float(yv)])
                        t.set_transform(coord_trans)

            # Column updates
            if self._col_label_artists is not None:
                x_b = self._label_bboxes_in_subfig_coords(bottom_row, which='x')
                x_c = self._label_centers_in_subfig_coords(bottom_row, which='x')
                x_center = float(np.mean(x_c[:, 0])) if len(x_c) else 0.5
                xmins = float(np.min(x_c[:, 0])) if len(x_c) else 0.2
                xmaxs = float(np.max(x_c[:, 0])) if len(x_c) else 0.8
                label_h_norm = float(labelsize) / (fig_height * 72.0)
                xlabel_bottom_min = float(np.min(x_b[:, 1])) if len(x_b) else self.bottom_bound
                val_y = max(xlabel_bottom_min - 2.0 * label_h_norm, 3.0 * label_h_norm)
                txt_y = max(val_y - 2.0 * label_h_norm, 0.8 * label_h_norm)
                self._col_label_artists['text'].set_position((x_center, txt_y))
                self._col_label_artists['text'].set_transform(coord_trans)

                val_texts = self._col_label_artists.get('val_texts', [])
                for i, (t, xc) in enumerate(zip(val_texts,
                                                  x_c[:, 0] if len(x_c) else [x_center])):
                    t.set_position((float(xc), val_y))
                    t.set_transform(coord_trans)

                spine_y = self._compute_col_spine_y(bottom_row, val_texts)
                tick_len_y = 0.06 / fig_height
                self._col_label_artists['spine'].set_data([xmins, xmaxs], [spine_y, spine_y])
                self._col_label_artists['spine'].set_transform(coord_trans)
                ticks = self._col_label_artists['ticks']
                if len(ticks) == (len(x_c) if len(x_c) else 1):
                    vals = x_c[:, 0] if len(x_c) else [x_center]
                    for t, xv in zip(ticks, vals):
                        t.set_data([float(xv), float(xv)], [spine_y, spine_y + tick_len_y])
                        t.set_transform(coord_trans)
        finally:
            self._updating = False

    def format(self, xlim=None, ylim=None, xticks=None, yticks=None, 
               xlabel=None, ylabel=None, special_bottom_left=False, 
               logx=False, logy=False, despine_right=True):
        """Format the subplots and figure."""
        inds = np.arange(len(self.axes.flatten()))
        if self.right_margin:
            inds = inds[:-1]
        if not despine_right:
            if self.bottom_margin:
                self.left[:-1, -1] = True
            else:
                self.left[:, -1] = True
        for row, are_left, are_bottom in zip(self.axes, self.left, self.bottom):
            for ax, is_left, is_bottom in zip(row, are_left, are_bottom):
                while isinstance(ax, np.ndarray): 
                    ax = ax[0]
                if xlim is not None:
                    try:
                        ax.set_xlim(xlim[0], xlim[1])
                    except Exception:
                        raise
                if ylim is not None:
                    if not (special_bottom_left and is_bottom and is_left):
                        ax.set_ylim(ylim[0], ylim[1])
                if logx:
                    ax.set_xscale('log')
                    if not is_bottom:
                        ax.tick_params(axis="x", which="minor", bottom=False)
                if logy:
                    ax.set_yscale('log')
                    if not is_left:
                        ax.tick_params(axis="y", which="minor", left=False)
                if (logx or logy) and not (is_left or is_bottom):
                    ax.minorticks_off()
                if is_bottom:
                    if xlabel is not None:
                        ax.set_xlabel(xlabel)
                    if xticks is not None:
                        ax.set_xticks(xticks[0], xticks[1])
                else:
                    ax.set_xticks([])
                if is_left:
                    if not (special_bottom_left and is_bottom): 
                        if ylabel is not None:
                            ax.set_ylabel(ylabel)
                        if yticks is not None:
                            ax.set_yticks(yticks[0], yticks[1])
                else:
                    ax.set_yticks([])
                # Always hide top and right; hide left/bottom on non-data edges.
                # Use _hide_spine instead of sbn.despine(trim=True) so that this
                # works reliably on log-scaled axes and axes sharing a scale.
                for _side in ('top', 'right'):
                    _hide_spine(ax, _side)
                if not is_left:
                    _hide_spine(ax, 'left')
                if not is_bottom:
                    _hide_spine(ax, 'bottom')

class TrackingTrial():
    def __init__(self, filename, holocube_framerate=120):
        """Load an H5 TrackingTrial file from the magnocube library.

        Parameters
        ----------
        filename : path
            Path to the TrackingTrial H5 file.
        """
        self.filename = filename
        # load the h5 file
        self.file_opened = False
        self.load_success = False
        self.file_opened = self.load(filename)
        if self.file_opened:
            # # store the h5 datasets as attributes
            # self.holocube_framerate = holocube_framerate
            # store whether the datasets were completely loaded
            self.load_datasets()
            # grab default dataset frequently used
            if self.load_success:
                self.data = self.query()
        # tracks which variables/attrs have been changed since the last save.
        # save() uses these to write only the affected arrays rather than
        # rewriting the entire zarr store.
        self._pending_vars = {}    # name → (np.ndarray, dims_tuple)
        self._removed_vars = set()
        self._dirty_attrs = False

    def add_dataset(self, name, arr):
        """Add or overwrite a variable in the in-memory xarray Dataset.

        The change is held in memory only. Call ``self.save()`` explicitly to
        persist it to the Zarr store.

        **Migration note (xarray/Zarr backend)**

        Previously this method wrote directly to the ``.h5`` file via h5py.
        It now assigns an ``xr.DataArray`` into the in-memory Dataset, so
        the new variable is immediately queryable without round-tripping to
        disk::

            trial.add_dataset('my_score', np.array([0.1, 0.9, 0.4]))
            result = trial.query('camera_heading', subset={'my_score': '>0.5'})
            trial.save()  # persist to .zarr if desired

        Dimension assignment rules:

        - 1-D array of length ``num_tests`` → dim ``('test',)``
        - 2-D array of shape ``(num_tests, num_frames)`` → dims ``('test', 'frame')``
        - 2-D array whose second dim ≠ ``num_frames`` → ``ValueError`` (see below)

        Parameters
        ----------
        name : str
            The name of the variable.
        arr : np.ndarray
            The array to store.  Must be 1-D (length ``num_tests``) or 2-D
            (``num_tests × num_frames``).  Raises ``ValueError`` if a 2-D array
            has a different frame count than this trial.

        Raises
        ------
        ValueError
            If ``arr`` is 2-D and ``arr.shape[1] != self.num_frames``.
        """
        arr = np.asarray(arr)
        if arr.ndim == 1:
            dims = ('test',)
        elif arr.ndim == 2:
            if arr.shape[1] == self.num_frames:
                # Use the same dimension names as camera_heading so the new
                # variable aligns with it in xarray. When the zarr was first
                # created from h5 via phony_dims='sort', camera_heading may
                # carry phony dim names (e.g. 'phony_dim_1') while a stale
                # 'frame' dimension of a different size exists from previously
                # saved variables. Using camera_heading's dims avoids the
                # xarray AlignmentError that would otherwise occur.
                dims = ('test', 'frame')
                if 'camera_heading' in self.h5_file.data_vars:
                    ch_dims = self.h5_file['camera_heading'].dims
                    if len(ch_dims) == 2:
                        dims = ch_dims
            else:
                raise ValueError(
                    f"add_dataset('{name}'): array has {arr.shape[1]} frames but "
                    f"this trial has num_frames={self.num_frames}. "
                    f"Ensure the array was derived from this trial's data without "
                    f"changing the frame count, or reshape it before calling add_dataset()."
                )
        else:
            dims = tuple(f'dim_{i}' for i in range(arr.ndim))
        da = xr.DataArray(arr, dims=dims)
        # Drop the existing variable (if any) before assigning so xarray doesn't complain
        # about conflicting coordinate / dimension information.
        if name in self.h5_file.data_vars:
            self.h5_file = self.h5_file.drop_vars(name)
        self.h5_file = self.h5_file.assign({name: da})
        self._pending_vars[name] = (arr, dims)
        self._removed_vars.discard(name)
        _LOAD_DATASETS_SENSITIVE = {'is_test', 'camera_heading_offline'}
        if name in _LOAD_DATASETS_SENSITIVE:
            import warnings
            warnings.warn(
                f"'{name}' affects derived attributes computed by load_datasets() "
                f"(e.g. self.is_test, self.frame_ind_offline). "
                f"Call trial.load_datasets() to update them.",
                UserWarning, stacklevel=2
            )

    def remove_dataset(self, name):
        """Remove a variable from the in-memory xarray Dataset.

        The change is held in memory only. Call ``self.save()`` explicitly to
        persist it to the Zarr store.

        Silently does nothing if *name* is not present in the dataset.

        **Migration note (xarray/Zarr backend)**

        Previously this method deleted a dataset directly from the ``.h5`` file
        via h5py.  It now calls ``xr.Dataset.drop_vars()`` on the in-memory
        Dataset::

            trial.remove_dataset('old_var')  # in-memory only
            trial.save()                     # persist the removal to .zarr

        Parameters
        ----------
        name : str
            The name of the variable to remove.
        """
        if name in self.h5_file.data_vars:
            self.h5_file = self.h5_file.drop_vars(name)
            self._removed_vars.add(name)
            self._pending_vars.pop(name, None)
            _LOAD_DATASETS_SENSITIVE = {'is_test', 'camera_heading_offline'}
            if name in _LOAD_DATASETS_SENSITIVE:
                import warnings
                warnings.warn(
                    f"'{name}' affects derived attributes computed by load_datasets() "
                    f"(e.g. self.is_test, self.frame_ind_offline). "
                    f"Call trial.load_datasets() to update them.",
                    UserWarning, stacklevel=2
                )

    def add_attr(self, name, val):
        """Add or update a global attribute on the in-memory xarray Dataset.

        The change is held in memory only. Call ``self.save()`` explicitly to
        persist it to the Zarr store.

        Sets the value on both ``self.h5_file.attrs`` (so it survives a
        ``save()``/reload round-trip) and directly on ``self`` (so it is
        accessible as ``trial.name`` immediately)::

            trial.add_attr('genotype', 'D. mel')
            print(trial.genotype)           # 'D. mel'
            print(trial.h5_file.attrs['genotype'])  # 'D. mel'
            trial.save()                    # persist to .zarr

        Parameters
        ----------
        name : str
            The attribute name.
        val : scalar or str
            The value to store.
        """
        self.h5_file.attrs[name] = val
        self.__setattr__(name, val)
        self._dirty_attrs = True

    def _get_var(self, key):
        """Retrieve a variable as a numpy array.

        Checks the xarray Dataset data_vars first (file-sourced arrays),
        then falls back to computed attributes set on self (e.g. time, test_ind).
        """
        ds = self.h5_file
        if key in ds.data_vars:
            return ds[key].values
        elif key in self.__dict__:
            val = self.__dict__[key]
            if isinstance(val, np.ndarray):
                return val
            return np.array(val)
        else:
            raise AttributeError(f"Variable '{key}' not found in dataset or computed attributes.")

    def load_datasets(self):
        import os
        self.load_success = False
        ds = self.h5_file  # xr.Dataset

        # Load global (file-level) attributes onto self
        for key, val in ds.attrs.items():
            self.__setattr__(key, val)

        # Get num_tests and num_frames from the renamed dimensions
        if 'test' in ds.dims and 'frame' in ds.dims:
            self.num_tests = ds.sizes['test']
            self.num_frames = ds.sizes['frame']
        elif 'frame' in ds.dims:
            self.num_tests = 1
            self.num_frames = ds.sizes['frame']

        # Reconcile num_frames against the actual shape of camera_heading.
        # When the zarr was first created from h5 via phony_dims='sort', the
        # original variables use phony dimension names (e.g. phony_dim_1) whose
        # sizes may differ by 1 from the 'frame' dimension added by later saves.
        if 'camera_heading' in ds.data_vars:
            ch_shape = ds['camera_heading'].shape
            actual_frames = ch_shape[-1]
            if hasattr(self, 'num_frames') and actual_frames != self.num_frames:
                self.num_frames = actual_frames
            elif not hasattr(self, 'num_frames'):
                self.num_frames = actual_frames
            if len(ch_shape) >= 2 and not hasattr(self, 'num_tests'):
                self.num_tests = ch_shape[0]

        # Handle camera_heading_offline (may have a different frame length)
        if 'camera_heading_offline' in ds.data_vars:
            vals = ds['camera_heading_offline'].values
            if vals.ndim == 2:
                _, self.num_frames_offline = vals.shape
            else:
                self.num_frames_offline = vals.shape[0]
            self.frame_ind_offline = np.arange(self.num_tests * self.num_frames_offline).reshape(
                self.num_tests, self.num_frames_offline)

        # Generate derived attributes from num_tests / num_frames
        if hasattr(self, 'num_frames'):
            self.test_ind = np.arange(self.num_tests)
            self.frame_ind = np.arange(self.num_tests * self.num_frames).reshape(
                self.num_tests, self.num_frames)

            # Estimate duration if not already loaded from attrs
            if 'duration' not in self.__dict__:
                if 'stop_exp' in self.__dict__ and 'start_exp' in self.__dict__:
                    self.duration = self.stop_exp - self.start_exp
                elif 'framerate' in self.__dict__:
                    self.duration = self.num_frames / self.framerate
                else:
                    # Fall back to 60 fps so the trial is still usable.
                    self.framerate = 60.0
                    self.duration = self.num_frames / self.framerate

            # Ensure framerate is always set (derive from duration if needed).
            if 'framerate' not in self.__dict__ and 'duration' in self.__dict__:
                self.framerate = self.num_frames / self.duration

            if 'duration' in self.__dict__:
                self.time = self.frame_ind * (self.duration / self.num_frames)
                self.holocube_framerate = (self.num_tests * self.num_frames) / self.duration
                self.load_success = True
            else:
                print("Could not determine the duration of the trial. Please add a 'duration' or 'framerate' attribute.")

        # Check for pickled bouts
        bouts_fn = str(self.filename).replace(".h5", "_bouts.pkl")
        if os.path.exists(bouts_fn):
            self.bouts = pickle.load(open(bouts_fn, 'rb'))
            for bout in self.bouts:
                bout.trial = self
        else:
            self.bouts = None

        # Ensure is_test is always a direct attribute on self
        if 'is_test' in ds.data_vars:
            self.is_test = ds['is_test'].values.astype(bool)
        elif 'is_test' not in self.__dict__:
            self.is_test = np.ones(self.num_tests, dtype=bool)

        # Reconstruct saccade_table from sidecar .pkl first (fast path), falling
        # back to legacy _saccade_* variables embedded in the zarr Dataset.
        import pickle as _pickle
        sidecar_path = os.path.splitext(self.filename)[0] + '.saccades.pkl'
        if os.path.exists(sidecar_path):
            with open(sidecar_path, 'rb') as _f:
                self.saccade_table = _pickle.load(_f)
        else:
            # Reconstruct saccade_table from _saccade_* variables if stored in this Dataset.
            saccade_vars = [v for v in ds.data_vars if v.startswith('_saccade_')]
            if saccade_vars:
                first_da = ds[saccade_vars[0]]
                # Use dims[0] rather than hardcoding 'saccade': on some zarr round-trips
                # the dimension may have been renamed (e.g. if saccade_count == num_tests).
                saccade_dim = first_da.dims[0]
                n = first_da.sizes[saccade_dim]
                self.saccade_table = [
                    {v[9:]: ds[v].values[i].item() for v in saccade_vars}
                    for i in range(n)
                ]

        # Ensure surgical-save bookkeeping attrs exist (may be absent when the
        # trial was constructed via __new__ + load_datasets() without __init__).
        if not hasattr(self, '_pending_vars'):
            self._pending_vars = {}
        if not hasattr(self, '_removed_vars'):
            self._removed_vars = set()
        if not hasattr(self, '_dirty_attrs'):
            self._dirty_attrs = False

    def get_saccade_stats(self, key='camera_heading', time_var='time', rerun=False, **saccade_kwargs):
        """List saccades for each trial using peak angular velocities.
        
        Parameters
        ----------
        key : str, default='camera_heading'
            The variable to use for saccade data.
        time_var : str, default='time'
            The variable to use for time data.
        rerun : bool, default=False
            Whether to re-analyze the start and stop points of saccades for each bout.
        **saccade_kwargs
        """
        # check if the bouts dataset is provided
        no_bouts = False
        if not isinstance(self.bouts, np.ndarray):
            no_bouts = True
        else:
            if len(self.bouts) != self.num_tests:
                no_bouts = True
        if rerun or no_bouts:
            # get saccades for each trial
            self.bouts = []
            times = self.query(time_var)
            output_vals = self.query(key)
            test_inds = self.query('test_ind')
            for time, test, test_ind in zip(times, output_vals, test_inds):
                bout = Bout(test, time, self, test_ind, self.holocube_framerate)
                bout.process_saccades(**saccade_kwargs)
                bout.get_stats()
                self.bouts += [bout]
            # if len(self.bouts) < 5:
            #     breakpoint()
            self.bouts = np.array(self.bouts)
            # use pickle to save the list of bouts for next time 
            bout_fn = str(self.filename).replace(".h5", "_bouts.pkl")
            if os.path.exists(bout_fn):
                os.remove(bout_fn)
            # we need to ditch the parent trial data before saving the bout
            for bout in self.bouts: 
                bout.trial = None
            pickle.dump(self.bouts, open(bout_fn, 'wb'))
        if 'saccade_duration' not in dir(self) or rerun:
            # store relevant saccade parameters:
            lbls = ['saccade_duration', 'saccade_amplitude', 'saccade_peak_velocity',
                    'saccade_frequency', 'saccade_isi', 'saccading_left', 'saccading_right']
            variables = ['saccade_duration_avg', 'saccade_amps_avg', 'saccade_peak_velo_avg',
                        'saccade_frequency', 'inter_saccade_interval', 
                        'saccading_left', 'saccading_right']
            for lbl, var in zip(lbls, variables):
                vals = []
                for bout in self.bouts:
                    vals += [getattr(bout, var)]
                self.add_dataset(lbl, np.array(vals))

    def query_bouts(self, sort_by='test_ind', subset={'is_test': True}):
        """Return bouts that fit the subset conditions sorted by a specified variable.
    
        Parameters
        ----------
        sort_by : str, default = 'test_ind'
            The parameter to use for sorting the trials.
        subset : dict, default = {'is_test': True}
            The subset of parameters to include in the output.
        """
        # a bout corresponds to each test
        include = np.ones(self.num_tests, dtype=bool)
        if len(subset.keys()) > 0:
            for key, vals in subset.items():
                # convert vals to list if it isn't already
                if not isinstance(vals, (list, tuple, np.ndarray)):
                    vals = [vals]
                for val in vals:
                    logic, thresh = np.equal, val
                    if isinstance(val, str):
                        logic, thresh = interprate_inequality(val)
                    var = self.query(output=key)
                    if not isinstance(var, (list, tuple, np.ndarray)):
                        var = np.repeat(var, self.num_tests)
                    if len(var) == len(include):
                        if isinstance(val, (list, tuple, np.ndarray)):
                            inds = np.array([test in val for test in var])
                        else:
                            inds = logic(var, thresh)
                            if inds.ndim > 1:
                                extra_dims = inds.ndim - 1
                                axes = tuple(np.arange(1, extra_dims+1))
                                # note: if any part satisfies the condition, include it
                                inds = np.any(inds, axis=axes)
                            if isinstance(val, float):
                                if np.isnan(val):
                                    inds = np.isnan(var)
                        while inds.ndim > include.ndim: 
                            inds = np.any(inds, axis=-1)
                        pad = include.ndim - inds.ndim
                        index = [...]
                        index += [np.newaxis for p in range(pad)]
                        include = include * inds[tuple(index)]
                    else:
                        raise ValueError(f"A subset variable, {key}, has length {len(var)} but should be {len(include)}.")
        # grab the indexing variable
        sort_by = self.__getattribute__(sort_by)
        if isinstance(sort_by, (str, bytes)):
            sort_by = np.repeat(sort_by, self.num_tests)
        # select the specified subset
        # todo: fix the order array below
        assert sort_by.ndim == 1, f"the sort_by array must be flat. instead sort_by.ndim={sort_by.ndim}"
        while include.ndim > sort_by.ndim: 
            include = np.any(include, axis=-1).astype(bool)
        sort_by = sort_by[include]
        bouts = self.bouts[include]   # use only the subseted bouts
        # sort using the sort_by
        order = np.argsort(sort_by)
        return bouts[order]

    def query_saccades(self, output='heading', time_var='time', start=0, stop=np.inf, 
                       min_speed=350, max_speed=np.inf, **filter_kwargs):
        """Query the saccade data between the start and stop points.

        Parameters
        ----------
        output : str, default = 'heading'
            The output variable. Defaults to using the heading values but can also be velocity.
        time_var : str, default = 'time'
            The saccade time frame to use for selecting 
        start, stop : float, default=0., np.inf.
            The start and stop times to include in the saccade.
        min_speed, max_speed : float, default=350, np.inf
            The minimum and maximum peak saccade speed to include in the query
        **filter_kwargs
            Conditions for filtering the bouts before querying the saccades.
        """
        bouts = self.query_bouts(**filter_kwargs)
        # collect the saccades
        saccade_arr = []
        time_arr = []
        lengths_arr = []
        min_length = np.inf
        for bout in bouts:
            time, saccades = bout.query_saccades(output=output, start=start, stop=stop, min_speed=min_speed, max_speed=max_speed)
            saccade_arr += [saccades]
            time_arr += [time]
        return time_arr, saccade_arr
    
    def center_initial(self, key='camera_heading'):
        """Center the starting point for all tests.
        
        Parameters
        ----------
        key : str, default='camera_heading'
            The variable to center.
        """
        vals = self.query(key, sort_by='test_ind', subset={})
        if vals.size > 0:
            starts = vals[..., 0]
            vals -= starts[..., np.newaxis]
            self.add_dataset(key + "_centered", vals)

    def unwrap(self, key='camera_heading', lower=-np.pi, upper=np.pi):
        """Unwrap the given time series.
        
        Parameters
        ----------
        key : str, default='camera_heading'
            The variable to unwrap.
        lower, upper : float, default=-np.pi, np.pi
            The bounds used for the unwrap function.
        """
        period = upper - lower
        vals = self.query(key, sort_by='test_ind')
        # todo: interpolate through nans before unwrapping because they will otherwise
        # make the rest of the values NaNs
        # find all nans in the array
        nans = np.isnan(vals)
        # go through each test and interpolate through the nans
        interpolated_vals = []
        for nan, val in zip(nans, vals):
            no_nans = nan == False
            # use scipy interp1d to interpolate through the nans
            interp = scipy.interpolate.interp1d(np.arange(val.size)[no_nans], val[no_nans], kind='nearest', fill_value='extrapolate')
            new_vals = np.copy(val)
            new_vals[nan] = interp(np.arange(val.size)[nan])
            interpolated_vals += [new_vals]
        interpolated_vals = np.array(interpolated_vals)
        # shift up by pi so that the 2pi unwrapping is centered around 0
        vals_unwrapped = np.unwrap(interpolated_vals - lower, axis=-1, period=period)
        vals_unwrapped += lower
        # comment here
        self.add_dataset(key + "_unwrapped", vals_unwrapped)

    def remove_saccades(self, key='camera_heading', invert=False, method='zero_velocity', min_speed=350, max_speed=np.inf):
        """Generate the same dataset but with saccades removed.
        
        To remove saccades, we first identify them in each bout. With this, 
        we can identify which frames are included in a saccade. We generate
        a velocity measurement, set velocity of those frames to 0, and then 
        take the cumulative sum to get the trajectory as if there were no 
        saccades. Then we add that dataset to the database.

        Parameters
        ----------
        key : str, default=camera_heading
            The variable to remove saccades from. This function will produce a new variable,
            called "{key}_no_saccades".
        invert : bool, default=False
            Whether to remove everything that is not a saccade instead.
        method : str, default='zero_velocity'
            Choose how to remove the saccades. There are two options so far:
                'zero_velocity' produces the same length array by finding velocities as the first
                    differences of heading, setting those velocities to 0, and then reverting back to 
                    position using the cumulative sum function. This has the effect of removing the 
                    saccades but maintaining the time course of responses. 
                'remove' actually removes the saccade segments, concatenates across them, and then 
                    pads the array with zeros to match the original array length.
        min_speed, max_speed : float, default=350, np.inf
            The minimum and maximum peak speeds for each saccade that's removed. Defaults to removing 
            saccades based on the Bender threshold of 350 degs/s.
        """
        if 'bouts' not in dir(self):
            # generate bout instances, which generates a list of Saccade instances
            self.get_saccade_stats(key=key)
        new_headings = []
        for bout in self.bouts:
            arr = np.unwrap(bout.arr)
            if method in ['zero_velocity', 'zero_torque']:
                # get the velocity of the bout
                offset = arr[0]
                velo = np.append([0], np.diff(arr))
            # set velo to 0 during saccades
            starts = np.array([saccade.start for saccade in bout.saccades])
            stops = np.array([saccade.stop for saccade in bout.saccades])
            peak_velos = np.array([saccade.peak_velocity for saccade in bout.saccades])
            # test: plot the heading data with the saccades highlighted
            # fig, axes = plt.subplots(nrows=2, sharex=True)
            # axes[0].plot(bout.arr)
            # ax = axes[1]
            # ax.plot(bout.velocity)
            # for start, stop in zip(starts, stops):
            #     ax.axvspan(start, stop, alpha=.2, color='gray')
            # ignore saccades with peak speed above the threshold speed
            fast_enough = abs(peak_velos) >= (min_speed * np.pi / 180.)
            slow_enough = abs(peak_velos) <= (max_speed * np.pi / 180.)
            include = fast_enough * slow_enough
            starts = starts[include]
            stops = stops[include]
            # for start, stop in zip(starts, stops):
            #     for ax in axes:
            #         ax.axvspan(start, stop, alpha=.5, color='gray')
            # plt.show()
            # breakpoint()
            # skip saccade if it starts at the beginning or end of the bout
            if invert:
                # invert the start and stop points
                new_stops = starts[1:]
                new_starts = stops[:-1]
                # replace 
                starts, stops = new_starts, new_stops
                # add the beginning to starts and end to stops
                starts = np.append([0], starts)
                stops = np.append(stops, -1)
            starts, stops = starts.astype(int), stops.astype(int)
            if len(starts) > 0:
                if method == 'zero_velocity':
                    for start, stop in zip(starts, stops):
                        # set velocity during saccade intervals to 0
                        velo[start:stop] = 0
                    # generate a new heading vector by taking the cumulative sum of the new velocity
                    new_heading  = np.cumsum(velo)
                    # new_heading += offset
                    # if new_heading[0] != offset:
                    #     breakpoint()
                    new_headings += [new_heading]
                elif method == 'zero_torque':
                    acc = np.append([0], np.diff(velo))
                    # test: plot acceleration highlighting the patches of saccades
                    # plt.plot(acc, color='k')
                    # for start, stop in zip(starts, stops): plt.axvspan(start, stop, color='gray', alpha=.3)
                    # plt.show()
                    for start, stop in zip(starts, stops):
                        # set acceleration to 0 during saccades
                        acc[start:stop] = acc[start:stop].mean()
                    # generate a new heading vector by taking the cumulative sum of the new velocity
                    new_velo = np.cumsum(acc)
                    new_heading  = np.cumsum(new_velo)
                    # new_heading += offset
                    # if new_heading[0] != offset:
                    #     breakpoint()
                    new_headings += [new_heading]
                elif method == 'remove':
                    # remove all of the 
                    # use the starts + the endpoint as the stopping points for inclusion
                    new_stops = np.append(starts, len(arr))
                    # use the begginging + the stopping points as starting points for inclusion
                    new_starts = np.append([0], stops)
                    # store values between the new starting and stopping points
                    new_vals = []
                    for start, stop in zip(new_starts, new_stops):
                        subset = arr[start:stop]
                        subset -= subset.mean()
                        new_vals += [arr[start:stop]]
                    new_vals = np.concatenate(new_vals)
                    new_heading = np.zeros(arr.shape)
                    new_heading[:len(new_vals)] = new_vals
                    new_headings += [new_heading]
            else:
                new_headings += [arr]
        new_headings = np.array(new_headings)
        if invert:
            lbl = f"{key}_saccades_only_{method}"
        else:
            lbl = f"{key}_no_saccades_{method}"
        self.add_dataset(lbl, new_headings)

    def detect_saccades(self, key='camera_heading', threshold_speed=350, **find_peaks_kwargs):
        """Detect saccades across all tests and store a flat saccade table.

        Iterates over each test, calls ``_detect_saccades()`` to find candidate
        start/stop frames, then instantiates a ``Saccade`` with full baseline
        refinement (``baseline_comparison=True, baseline_test=True``) to refine
        the window and validate the event.  Scalar attributes of each successful
        saccade are harvested into ``self.saccade_table`` (a list of dicts).
        Any existing table is replaced.  Does **not** call ``save()``.

        Parameters
        ----------
        key : str, default='camera_heading'
            The heading variable to use for detection.
        threshold_speed : float, default=350
            Minimum peak speed (degrees/s) required to keep a saccade.
        **find_peaks_kwargs
            Forwarded to ``_detect_saccades`` (distance, width, prominence, wlen).

        Attributes set
        --------------
        saccade_table : list[dict]
            One dict per accepted saccade with keys: ``test_ind``,
            ``test_start_frame``, ``start_frame``, ``stop_frame``,
            ``amplitude``, ``peak_velocity``, ``duration``,
            ``start_angle``, ``stop_angle``.

        Examples
        --------
        ::

            trial.detect_saccades()                # run with defaults
            print(len(trial.saccade_table))        # number of accepted saccades
            trial.save()                           # persist to .zarr

            # Query amplitudes after detection
            amps = trial.query(output='amplitude', object='saccade')
        """
        headings = self.query(key)          # shape (num_tests, num_frames)
        framerate = self.framerate
        rows = []
        for test_i in range(self.num_tests):
            arr = headings[test_i]          # shape (num_frames,)
            test_start_frame = test_i * self.num_frames
            candidates = _detect_saccades(arr, framerate, threshold_speed=threshold_speed,
                                          **find_peaks_kwargs)
            for cand in candidates:
                saccade = Saccade(
                    arr, self, test_i, framerate,
                    start=cand['start'], stop=cand['stop'],
                    baseline_comparison=True, baseline_test=True,
                )
                if not saccade.success:
                    continue
                peak_velo_degs = abs(saccade.peak_velocity) * 180.0 / np.pi
                if peak_velo_degs < threshold_speed:
                    continue
                rows.append({
                    'test_ind':         test_i,
                    'test_start_frame': test_start_frame,
                    'start_frame':      saccade.start,
                    'stop_frame':       saccade.stop,
                    'peak_frame':       int(saccade.peak_ind),
                    'amplitude':        float(saccade.amplitude),
                    'peak_velocity':    float(saccade.peak_velocity),
                    'duration':         float(saccade.duration),
                    'start_angle':      float(saccade.start_angle),
                    'stop_angle':       float(saccade.stop_angle),
                })
        self.saccade_table = rows
        # Record which variable was used for detection so on-demand Saccade
        # reconstruction uses the same array (see query / _row_to_value).
        self.saccade_heading_variable = key
        # Invalidate any cached Saccade objects so the next query rebuilds them.
        self.saccades = None
        self._saccade_id_to_index = None

    def saccade_table_df(self, extra_cols=None, time_ind='start_frame'):
        """Return the saccade table as a :class:`pandas.DataFrame`.

        Parameters
        ----------
        extra_cols : list[str] or None
            Names of trial attributes or datasets to append as extra columns.
            For each name the value is resolved via ``_get_var()``:

            - **Scalar / 0-D / string** → broadcast to every saccade row.
            - **1-D array of length num_tests** → index by ``row['test_ind']``
              so each saccade gets the value for its own test.
            - **2-D array (num_tests × num_frames)** → index by
              ``(row['test_ind'], row[time_ind])`` so each saccade gets the
              value at the chosen frame.

        time_ind : {'start_frame', 'stop_frame', 'peak_frame'}, default='start_frame'
            Which saccade-table frame index to use when looking up 2-D
            (test × frame) variables.

        Returns
        -------
        pandas.DataFrame
            One row per saccade.

        Examples
        --------
        ::

            trial.detect_saccades()
            df = trial.saccade_table_df()
            print(df.columns.tolist())
            # ['test_ind', 'start_frame', 'stop_frame', 'peak_frame',
            #  'amplitude', 'peak_velocity', 'duration',
            #  'start_angle', 'stop_angle', 'test_start_frame']

            # Include a trial-level attribute and a test-level condition variable
            df2 = trial.saccade_table_df(extra_cols=['condition', 'fly_id'])
        """
        import pandas as pd
        if not hasattr(self, 'saccade_table') or not self.saccade_table:
            return pd.DataFrame()
        df = pd.DataFrame(self.saccade_table)
        for col in (extra_cols or []):
            var = self._get_var(col)
            if isinstance(var, np.ndarray):
                if var.ndim == 0:
                    # 0-D numpy scalar — broadcast to all rows.
                    df[col] = var.item()
                elif var.ndim == 1 and len(var) == self.num_tests:
                    df[col] = [var[row['test_ind']] for row in self.saccade_table]
                elif var.ndim == 2 and var.shape[0] == self.num_tests:
                    n_frames = var.shape[1]
                    df[col] = [
                        var[row['test_ind'], min(row[time_ind], n_frames - 1)]
                        for row in self.saccade_table
                    ]
                else:
                    raise ValueError(
                        f"extra_col '{col}' has shape {var.shape}, which cannot be "
                        f"mapped to saccades (expected 1-D of length {self.num_tests} "
                        f"or 2-D of shape ({self.num_tests}, ~{self.num_frames}))."
                    )
            else:
                # Scalar, string, or 0-D — broadcast to all rows.
                df[col] = var
        return df

    def query(self, output='camera_heading', sort_by='test_ind', subset={},
              object='trial', groupby='saccade', agg_func=np.nanmean):
        """Return trial data filtered and sorted by subset conditions.

        Parameters
        ----------
        output : str, default ``'camera_heading'``
            Variable to retrieve.  For ``object='trial'`` this is any dataset
            or attribute stored on the trial.  For ``object='saccade'`` it is
            a scalar saccade-table column (``'amplitude'``, ``'peak_velocity'``,
            ``'duration'``, ``'start_angle'``, ``'stop_angle'``,
            ``'start_frame'``, ``'stop_frame'``, ``'test_ind'``) or
            ``'saccade'`` to get reconstructed :class:`Saccade` objects.
        sort_by : str, default ``'test_ind'``
            Trial-level variable used to sort the returned rows.
        subset : dict, default ``{}``
            Filter conditions.  Each key is a variable name; the value can be:

            - A scalar → exact equality.
            - A list of scalars → membership (OR).
            - An inequality string such as ``'>0.5'`` or ``'<=180'``.
            - ``float('nan')`` → match NaN entries.
            - A list of inequality strings → all conditions ANDed together.

            For ``object='saccade'`` keys that match saccade-table columns
            filter individual saccade rows; all other keys are matched against
            the test-level value for each saccade's ``test_ind``.
        object : {'trial', 'saccade'}, default ``'trial'``
            Data source.  ``'saccade'`` requires :meth:`detect_saccades` to
            have been called first.
        groupby : {'saccade', 'test', 'trial'}, default ``'saccade'``
            Only used when ``object='saccade'``.

            - ``'saccade'`` — flat list, one value per saccade.
            - ``'test'`` — one value per test, reduced via ``agg_func``.
            - ``'trial'`` — one scalar for the whole trial via ``agg_func``.
        agg_func : callable, default ``np.nanmean``
            Reduction applied per group for ``groupby='test'`` or
            ``groupby='trial'``.  Use ``list`` to collect raw values without
            reducing (returns ragged lists when saccade counts differ).

        Returns
        -------
        ndarray or list
            For ``object='trial'``: ``ndarray`` of shape
            ``(n_tests[, n_frames])`` after subsetting and sorting.
            For ``object='saccade'``: list of scalars / :class:`Saccade`
            objects (``groupby='saccade'``), list of scalars
            (``groupby='test'``), or a single scalar (``groupby='trial'``).

        Raises
        ------
        RuntimeError
            If ``object='saccade'`` and :meth:`detect_saccades` has not been
            called yet.

        Examples
        --------
        Full heading time-series for all tests::

            heading = trial.query('camera_heading')
            # \u2192 ndarray shape (n_tests, n_frames)

        Subset to one condition::

            heading = trial.query('camera_heading', subset={'condition': 1})

        Membership filter (conditions 1 and 3)::

            heading = trial.query('camera_heading', subset={'condition': [1, 3]})

        Inequality string filter::

            heading = trial.query('camera_heading', subset={'bg_gain': '>0'})

        Flat list of saccade amplitudes::

            amps = trial.query(output='amplitude', object='saccade')

        Mean amplitude per test::

            amps_per_test = trial.query(
                output='amplitude', object='saccade', groupby='test'
            )

        On-demand Saccade objects filtered by speed::

            saccades = trial.query(
                output='saccade', object='saccade',
                subset={'peak_velocity': '>5'},
            )
        """
        # ------------------------------------------------------------------ #
        # Saccade query branch                                                 #
        # ------------------------------------------------------------------ #
        if object == 'saccade':
            if not hasattr(self, 'saccade_table') or self.saccade_table is None:
                raise RuntimeError(
                    "No saccade table found.  Call detect_saccades() first."
                )
            saccade_cols = set(self.saccade_table[0].keys()) if self.saccade_table else set()

            # --- filter rows ------------------------------------------------
            rows = list(self.saccade_table)
            for key, vals in subset.items():
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                filtered = []
                for row in rows:
                    include = True
                    if key in saccade_cols:
                        # saccade-level filter
                        row_val = row[key]
                        match = False
                        for val in vals:
                            if isinstance(val, str):
                                logic, thresh = interprate_inequality(val)
                                match = match or bool(logic(row_val, thresh))
                            elif isinstance(val, float) and np.isnan(val):
                                match = match or np.isnan(row_val)
                            else:
                                match = match or (row_val == val)
                        include = match
                    else:
                        # trial-level filter: look up value for this saccade's test
                        trial_var = self._get_var(key)
                        if isinstance(trial_var, np.ndarray) and trial_var.ndim == 1:
                            row_val = trial_var[row['test_ind']]
                        else:
                            row_val = trial_var
                        match = False
                        for val in vals:
                            if isinstance(val, str):
                                logic, thresh = interprate_inequality(val)
                                match = match or bool(logic(row_val, thresh))
                            elif isinstance(val, float) and np.isnan(val):
                                match = match or np.isnan(row_val)
                            else:
                                match = match or (row_val == val)
                        include = match
                    if include:
                        filtered.append(row)
                rows = filtered

            # --- extract output values -------------------------------------
            def _row_to_value(row):
                if output == 'saccade':
                    # Build the full Saccade cache on first access.
                    if not getattr(self, 'saccades', None):
                        _saccade_key = getattr(self, 'saccade_heading_variable', 'camera_heading')
                        heading = self.query(_saccade_key)  # (num_tests, num_frames)
                        self.saccades = [
                            Saccade(
                                heading[r['test_ind']], self, r['test_ind'], self.framerate,
                                start=r['start_frame'], stop=r['stop_frame'],
                                baseline_comparison=False, baseline_test=False,
                            )
                            for r in self.saccade_table
                        ]
                        # Map each table row id → list index for O(1) lookup.
                        self._saccade_id_to_index = {id(r): i for i, r in enumerate(self.saccade_table)}
                    return self.saccades[self._saccade_id_to_index[id(row)]]
                return row[output]

            # --- apply groupby -------------------------------------------
            if groupby == 'saccade':
                return [_row_to_value(r) for r in rows]
            elif groupby == 'test':
                test_groups = {}
                for r in rows:
                    test_groups.setdefault(r['test_ind'], []).append(_row_to_value(r))
                return [agg_func(grp) for grp in test_groups.values()]
            elif groupby == 'trial':
                all_vals = [_row_to_value(r) for r in rows]
                return agg_func(all_vals) if all_vals else np.nan
            else:
                raise ValueError(f"groupby must be 'saccade', 'test', or 'trial'; got {groupby!r}")

        # ------------------------------------------------------------------ #
        # Trial query branch (original behaviour)                             #
        # ------------------------------------------------------------------ #
        if output == 'bouts':
            return np.array(self.bouts)

        # --- fetch output as a DataArray (or broadcast scalar to 1-D) ---
        raw = self._get_var(output)
        if isinstance(raw, (str, bytes)) or np.ndim(raw) == 0:
            raw = np.repeat(raw, self.num_tests)
        if raw.ndim == 1:
            da = xr.DataArray(raw, dims=('test',))
        elif raw.ndim == 2:
            da = xr.DataArray(raw, dims=('test', 'frame'))
        else:
            # higher-dim: return as-is without subsetting
            return raw
        # --- build masks from subset ---
        # test_mask: 1-D bool over 'test' — rows to keep
        # frame_mask: 2-D bool over ('test','frame') — frames to NaN-fill
        test_mask = np.ones(self.num_tests, dtype=bool)
        frame_mask = None  # only created if a 2-D subset var is encountered
        for key, vals in subset.items():
            if not isinstance(vals, (list, tuple, np.ndarray)):
                vals = [vals]
            var = self._get_var(key)
            # broadcast scalar/0-D to 1-D
            if isinstance(var, (str, bytes, bool, float, int)) or np.ndim(var) == 0:
                var = np.repeat(var, self.num_tests)

            # Decide whether vals is a membership set or a list of AND conditions.
            # A list of non-string scalars means OR/membership: e.g. {'group': [1, 3]}
            #   → np.isin(var, [1, 3]).
            # A list containing any strings means AND conditions: e.g. {'score': ['>0.4', '<=0.9']}
            #   → each string is a separate inequality ANDed together.
            is_membership = (
                isinstance(vals, (list, tuple, np.ndarray))
                and len(vals) > 0
                and not any(isinstance(v, str) for v in vals)
                and not any(isinstance(v, float) and np.isnan(v) for v in vals)
            )
            if is_membership:
                cond = np.isin(var, vals)
                if var.ndim == 1:
                    test_mask &= cond
                elif var.ndim == 2:
                    if frame_mask is None:
                        frame_mask = np.ones((self.num_tests, self.num_frames), dtype=bool)
                    frame_mask &= cond
            else:
                for val in vals:
                    # compute a boolean array matching var's shape
                    if isinstance(val, str):
                        logic, thresh = interprate_inequality(val)
                        cond = logic(var, thresh)
                    elif isinstance(val, float) and np.isnan(val):
                        cond = np.isnan(var)
                    else:
                        cond = (var == val)

                    if var.ndim == 1:
                        # test-level: AND into test_mask
                        test_mask &= cond
                    elif var.ndim == 2:
                        # frame-level: AND into frame_mask
                        if frame_mask is None:
                            frame_mask = np.ones((self.num_tests, self.num_frames), dtype=bool)
                        frame_mask &= cond

        # --- apply test-level filter: drop non-matching tests ---
        da = da.isel(test=np.where(test_mask)[0])

        # --- apply frame-level filter: NaN-fill non-matching frames ---
        if frame_mask is not None and da.ndim == 2:
            fm = xr.DataArray(frame_mask[test_mask], dims=('test', 'frame'))
            da = da.where(fm)

        # --- sort by sort_by variable (1-D, test-level) ---
        sb = self._get_var(sort_by)
        if isinstance(sb, (str, bytes)) or np.ndim(sb) == 0:
            sb = np.repeat(sb, self.num_tests)
        sb = sb[test_mask]
        sort_order = np.argsort(sb)
        da = da.isel(test=sort_order)

        return da.values

    def butterworth_filter(self, key='camera_heading', low=1, high=6,
                           sample_rate=60):
        """Apply a butterworth to the specified dataset.

        Parameters
        ----------
        key : str, default='camera_heading'
            The dataset to be filtered.
        low : float, default=1
            The lower bound of the bandpass filter in Hz.
        high : float, default=6
            The upper bound of the bandpass filter in Hz.
        sample_rate : float, default=60
            The sample rate used for calculating frequencies for the filter.
        """
        # frequencies must be at least 0
        assert low >= 0 and high > 0, "Frequencies bounds must be non-negative."
        # copy the values to filtered
        vals = np.copy(self._get_var(key))
        # unwrap the vals first and then wrap again
        vals = np.unwrap(vals, axis=1)
        if low > 0 and high < np.inf:
            filter = scipy.signal.butter(5, (low, high),
                                   fs=sample_rate,
                                   btype='bandpass',
                                   output='sos')
        elif np.isinf(high):
            filter = scipy.signal.butter(5, low,
                                   fs=sample_rate,
                                   btype='highpass',
                                   output='sos')
        elif low == 0:
            filter = scipy.signal.butter(5, high,
                                   fs=sample_rate,
                                   btype='lowpass',
                                   output='sos')
        vals_smoothed = scipy.signal.sosfiltfilt(filter, vals, axis=1)
        # persist via add_dataset so the smoothed array survives across sessions
        self.add_dataset(key + "_smoothed", vals_smoothed)

    def save(self):
        """
        Persist in-memory changes to the Zarr store on disk.

        **Migration note (xarray/Zarr backend)**

        Saving is now *opt-in* — ``add_dataset``, ``remove_dataset``, and
        ``add_attr`` only modify the in-memory ``xr.Dataset`` (``self.h5_file``).
        Call ``save()`` explicitly when you want changes to survive the session::

            trial.add_dataset('my_var', arr)
            trial.add_attr('experiment_date', '2026-04-14')
            trial.save()   # writes path/to/trial.zarr atomically

        The save uses a write-to-temp-then-rename pattern so that in-flight lazy
        reads from the existing store are never interrupted::

            # writes to trial.zarr_tmp → closes handle → removes trial.zarr
            # → renames trial.zarr_tmp → trial.zarr → reopens lazily

        Returns
        -------
        str
            Absolute path to the written ``.zarr`` store.
        """
        import os
        import shutil
        import xarray as xr
        import pickle as _pickle
        import zarr as _zarr
        zarr_path = os.path.splitext(self.filename)[0] + ".zarr"
        sidecar_path = os.path.splitext(self.filename)[0] + ".saccades.pkl"
        has_saccades = hasattr(self, 'saccade_table') and self.saccade_table
        any_zarr_changes = self._pending_vars or self._removed_vars or self._dirty_attrs

        if os.path.exists(zarr_path) and any_zarr_changes:
            # Surgical-update path: open the existing store and write only what changed.
            # This is fast regardless of how large the unchanged arrays are.
            store = _zarr.open(zarr_path, mode='r+')
            # Delete removed variables.
            for name in self._removed_vars:
                if name in store:
                    del store[name]
            # Write new/modified variables.
            for name, (arr, dims) in self._pending_vars.items():
                z = store.create_array(name, data=arr, overwrite=True)
                z.attrs['_ARRAY_DIMENSIONS'] = list(dims)
            # Update global attributes (.zattrs).
            if self._dirty_attrs:
                store.attrs.update(dict(self.h5_file.attrs))
            self._pending_vars.clear()
            self._removed_vars.clear()
            self._dirty_attrs = False
        elif not os.path.exists(zarr_path):
            # First-time write: no zarr store exists yet — do a full write via xarray
            # so all dimension coordinates are encoded correctly.
            tmp_path = zarr_path + "_tmp"
            self.h5_file.to_zarr(tmp_path, mode='w')
            self.h5_file.close()
            if os.path.exists(zarr_path):
                shutil.rmtree(zarr_path)
            os.rename(tmp_path, zarr_path)
            self.h5_file = xr.open_dataset(zarr_path, engine='zarr')
            self._pending_vars.clear()
            self._removed_vars.clear()
            self._dirty_attrs = False

        # Saccade table: always saved to a lightweight sidecar .pkl file alongside
        # the zarr store.  This completely avoids zarr overhead on saccade-only saves
        # (the common case after detect_saccades with RERUN=False).
        if has_saccades:
            with open(sidecar_path, 'wb') as _f:
                _pickle.dump(self.saccade_table, _f, protocol=_pickle.HIGHEST_PROTOCOL)

        return zarr_path

    def load(self, filename, trim=False, force_h5=False):
        """
        Load trial data, converting to Zarr on first use.

        **Migration note (xarray/Zarr backend)**

        Trials are now stored as Zarr directory stores rather than NetCDF files.
        On the first load of any ``.h5`` file the data is read via h5netcdf, written
        to a ``.zarr`` store alongside the original file, and reopened lazily::

            trial = TrackingTrial("path/to/trial.h5")
            # → creates path/to/trial.zarr if it does not already exist
            # → self.h5_file is an xr.Dataset opened lazily with engine='zarr'

        Subsequent loads skip the conversion and open the ``.zarr`` store directly.
        All dimension variables are named ``'test'`` (small axis) and ``'frame'``
        (large axis), replacing the opaque ``phony_dim_N`` names from h5netcdf.

        Parameters
        ----------
        filename : str
            Path to the ``.h5`` source file.
        trim : bool, default=False
            If True, trim frame-like dimensions to the same minimum length before
            renaming.  Useful when variables have off-by-one lengths in the source.
        force_h5 : bool, default=False
            If True, skip any existing ``.zarr`` store and re-read from the original
            ``.h5`` file (re-creating the store).  Primarily for testing.
        """
        import os
        import xarray as xr
        base, _ = os.path.splitext(filename)
        new_path = base + ".zarr"
        success = False
        try_nc = False
        # if the NetCDF file exists, load from it; otherwise, load from the original HDF5 file
        if os.path.exists(new_path) and not force_h5:
            try_nc = True
        # xr.open_dataset keeps the file lazily open — data is read on access.
        if try_nc:
            try:
                self.h5_file = xr.open_dataset(new_path, engine='zarr')
                self._source_path = new_path
                success = True
            except:
                success = False
        # If it fails, we can try to load from the original HDF5 file if we haven't already
        if not success:
            try:
                self.h5_file = xr.open_dataset(filename, engine='h5netcdf', phony_dims='sort')
                # store it as a Zarr file using xarray
                self.h5_file.to_zarr(new_path, mode='w')
                self.h5_file.close()
                # and now open the Zarr file
                self.h5_file = xr.open_dataset(new_path, engine='zarr')
                self._source_path = new_path
                success = True
            except:
                success = False
        if success:
            ds = self.h5_file
            # --- Trim frame-like dimensions to the same length before renaming ---
            # Variables like camera_heading and orientation can end up with slightly
            # different lengths (off by 1-2 frames). xarray assigns them separate
            # phony dimensions, so we trim to the minimum length first.
            # Heuristic: frame-like dims have more than 10 values (test dim is small).
            if trim:
                frame_like_dims = [d for d in ds.dims if ds.dims[d] > 10]
                if frame_like_dims:
                    min_frames = min(ds.dims[d] for d in frame_like_dims)
                    for var in list(ds.data_vars):
                        for d in frame_like_dims:
                            if d in ds[var].dims and ds.dims[d] > min_frames:
                                ds[var] = ds[var].isel({d: slice(None, min_frames)})

            # --- Rename phony dimensions to meaningful names ---
            # Only run when the dims are still opaque (e.g. 'phony_dim_0').
            # When loading from a Zarr store that was already saved with named
            # dims ('test', 'frame', 'saccade', ...) the rename is skipped so
            # that extra dims like 'saccade' are never incorrectly overwritten.
            if 'test' not in ds.dims or 'frame' not in ds.dims:
                dim_names_sorted = sorted(ds.sizes.keys(), key=lambda d: ds.sizes[d])
                rename_dict = {}
                if len(dim_names_sorted) >= 1:
                    rename_dict[dim_names_sorted[0]] = 'test'
                if len(dim_names_sorted) >= 2:
                    rename_dict[dim_names_sorted[-1]] = 'frame'
                if rename_dict:
                    ds = ds.rename(rename_dict)
            self.h5_file = ds
        return success

    def close(self):
        """Close the h5 file and delete bouts if present."""
        if hasattr(self, 'h5_file') and self.h5_file:
            try:
                self.h5_file.close()
            except Exception:
                pass
            self.h5_file = None
        if hasattr(self, 'bouts'):
            del self.bouts

    def __del__(self):
        self.close()


# todo: use nonlinear fitting to find the best jerk_std and measurement noise for a Kalman filter 
class KalmanFitter():
    def __init__(self, arr):
        self.arr = arr

    def fit(self):
        self.results = []
        # self.fmin = scipy.optimize.least_squares(self.compare, (100), bounds=(.1, np.inf))
        self.fmin = scipy.optimize.fmin(self.compare, 100)
        # todo: make a plot comparing the results with time on the x-axis
        # note: it seems like this method only really changes the noise variable. we probably have to decide on a jerk_std
        # and then minimize the noise or vice-versa
        fig, axes = plt.subplots(nrows=2)
        axes[0].scatter(range(len(self.arr)), self.arr, color='k', marker='.', s=.1)
        for num, res in enumerate(self.results): axes[0].plot(np.unwrap(res), color='k', alpha= (num+1)/len(self.results))
        axes[1].scatter(range(len(self.arr)), self.arr, color='k', marker='.', s=.1)
        axes[1].plot(np.unwrap(self.results[-1]), color='k')
        plt.tight_layout()
        plt.show()

    def error(self, modelled_vals):
        return sum((self.arr - modelled_vals)**2)

    def generate_vals(self, jerk_std=1, measurement_noise=1):
        # todo: apply KalmanAngle to the array with the given parameters
        self.filter = KalmanAngle(jerk_std=jerk_std, measurement_noise_x=measurement_noise)
        smoothed_vals = []
        for val in self.arr:
            if val != np.nan:
                self.filter.store(val)
            smoothed_vals += [self.filter.predict()]
        return np.array(smoothed_vals)

    def compare(self, jerk_std):
        # jerk_std, measurement_noise = params
        measurement_noise = 1
        self.modelled_vals = self.generate_vals(jerk_std, measurement_noise)
        self.results += [self.modelled_vals]
        return self.error(self.modelled_vals)


def _detect_saccades(arr, framerate, threshold_speed=350, display=False, **find_peaks_kwargs):
    """Detect candidate saccade start/stop frames using the speed-noise method.

    Applies a zero-phase Butterworth filter to the heading array, computes
    velocity, and uses peak-finding to identify candidate saccade windows.
    Returns raw (pre-refinement) start/stop frame indices; the caller is
    responsible for instantiating Saccade with baseline refinement and
    filtering by threshold_speed.

    Parameters
    ----------
    arr : np.ndarray, shape=(n_frames,)
        Heading time series (radians). NaNs are handled internally.
    framerate : float
        Frames per second.
    threshold_speed : float, default=350
        Unused here (kept for API symmetry with detect_saccades). The
        threshold is applied by the caller after Saccade refinement.
    display : bool, default=False
        If True, plot the filtered position, velocity, and acceleration in
        three vertically-stacked subplots with a shared time axis.  Each
        detected saccade window is shaded lightgray and a red vertical line
        marks the peak frame.
    **find_peaks_kwargs
        Optional overrides for scipy.signal.find_peaks: distance, width,
        prominence, wlen. Defaults: distance=framerate/16, width=2,
        prominence=(1, 30), wlen=framerate/4.

    Returns
    -------
    candidates : list[dict]
        Each dict has keys 'start' and 'stop' (int frame indices,
        pre-refinement).
    """
    arr = np.unwrap(np.copy(arr))
    arr_clean, _mask, valid_start, valid_end = _handle_nans_for_filter(arr)
    high = min(15, round(framerate / 2) - 1)
    vals_fwd = butterworth_filter(arr_clean[np.newaxis], low=0, high=high, sample_rate=framerate)
    vals_rev = butterworth_filter(arr_clean[np.newaxis, ::-1], low=0, high=high, sample_rate=framerate)
    vals_filtered = np.full_like(arr, np.nan)
    vals_filtered[valid_start:valid_end] = np.unwrap(
        (vals_fwd + vals_rev[:, ::-1]) / 2, axis=-1
    )
    velocity = np.gradient(vals_filtered) * framerate
    dist = framerate / 4
    distance = find_peaks_kwargs.get('distance', dist / 4)
    width = find_peaks_kwargs.get('width', 2)
    prominence = find_peaks_kwargs.get('prominence', (1, 30))
    wlen = find_peaks_kwargs.get('wlen', dist)
    peaks = scipy.signal.find_peaks(
        np.abs(velocity), distance=distance, width=width,
        prominence=prominence, wlen=wlen,
    )
    starts = peaks[1]['left_bases']
    stops = peaks[1]['right_bases']
    if display:
        import matplotlib.pyplot as plt
        accel = np.gradient(velocity) * framerate
        time = np.arange(len(arr)) / framerate
        peak_frames = peaks[0]
        fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(10, 6))
        axes[0].plot(time, np.degrees(np.squeeze(vals_filtered)), 'k-')
        axes[1].plot(time, np.degrees(velocity), 'k-')
        axes[2].plot(time, np.degrees(accel), 'k-')
        for start, stop, peak in zip(starts, stops, peak_frames):
            for ax in axes:
                ax.axvspan(time[start], time[stop], color='lightgray', alpha=1.0)
                ax.axvline(time[peak], color='r', linewidth=1)
        axes[0].set_ylabel('Position (°)')
        axes[1].set_ylabel('Velocity (°/s)')
        axes[2].set_ylabel('Accel. (°/s²)')
        axes[2].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
    return [{'start': int(s), 'stop': int(t)} for s, t in zip(starts, stops)]


class Bout():
    def __init__(self, arr, time, trial, test_ind, original_times=None):
        """Analyze a flight bout isolating saccades and taking pertinent measurements.
        
        Parameters
        ----------
        arr : np.ndarray, shape=(num_frames)
            The array of heading values.
        time : np.ndarray, shape=(num_frames)
            The array of time values.
        trial : TrackingTrial
            The trial instance that the bout belongs to.
        test_ind : int
            The index of the test that the bout belongs to from its parent Trial.
        original_times : np.ndarray
            Optionally, provide the time values from the video to allow 
        """
        import warnings
        warnings.warn(
            "Bout is deprecated and will be removed in a future version. "
            "Use TrackingTrial.detect_saccades() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # store the parameters
        self.trial = trial
        self.arr = arr
        self.time = time
        self.test_ind = test_ind
        # self.framerate = 1./(self.time[1] - self.time[0])
        self.framerate = 1./np.nanmean(np.diff(self.time))
        self.duration = self.time.max() - self.time.min()

    def get_stats(self, **saccade_kwargs):
        """Calculate a bunch of bout statistics.

        Calculates:
        -----------
        -total angle subtended
        -average velocity
        -inter-saccade interval
        -saccade frequency
        -saccade durations
        -saccade amplitudes
        -peak velocities
        """
        # get all of the saccade data
        if 'saccades' not in dir(self):
            self.process_saccades(**saccade_kwargs)
        # total angle subtended
        self.extreme_frame = np.argmax(abs(self.arr - self.arr[0]))
        self.total_angle = self.arr[self.extreme_frame] - self.arr[0]
        # average velocity
        self.velocity = np.gradient(self.arr)
        self.velocity_avg = self.velocity.mean()
        self.velocity_std = self.velocity.std()
        # inter-saccade interval
        # measure the time from each stop to the next start
        starts = [saccade.stop for saccade in self.saccades][:-1]
        stops = [saccade.start for saccade in self.saccades][1:]
        intervals = []
        for start, stop in zip(starts, stops):
            intervals += [(stop - start)/self.framerate]
        self.inter_saccade_intervals = intervals
        if len(self.saccades) > 0:
            # add the isi to each appropriate saccade
            self.saccades[0].isi = np.nan
        if len(self.saccades) > 1:
            for isi, saccade in zip(intervals, self.saccades[1:]):
                saccade.isi = isi
        self.inter_saccade_interval = np.mean(intervals)
        # saccade frequency
        self.saccade_frequency = len(self.saccades) / self.duration
        # saccade durations
        self.saccade_durations = [saccade.duration for saccade in self.saccades]
        self.saccade_duration_avg = np.mean(self.saccade_durations)
        self.saccade_duration_std = np.std(self.saccade_durations)
        # saccade amplitudes
        self.saccade_amps = [saccade.amplitude for saccade in self.saccades]
        self.saccade_amps_avg = np.mean(self.saccade_amps)
        self.saccade_amps_std = np.std(self.saccade_amps)
        # peak velocities
        self.saccade_peak_velo = [saccade.peak_velocity for saccade in self.saccades]
        self.saccade_peak_velo_avg = np.mean(self.saccade_peak_velo)
        self.saccade_peak_velo_std = np.std(self.saccade_peak_velo)
        # time series indicating when a 1) leftward and 2) rightward saccade is happening
        self.saccade_is_left = np.array(self.saccade_peak_velo) > 0
        self.saccading_left = np.zeros(self.arr.shape, dtype=float)
        self.saccading_right = np.zeros(self.arr.shape, dtype=float)
        for saccade, is_left in zip(self.saccades, self.saccade_is_left):
            storage = [self.saccading_left, self.saccading_right][1 * is_left]
            storage[saccade.start:saccade.stop] = True

    def process_saccades(self, threshold_speed=350, speed_noise_method=True, acceleration_method=False, 
                         maximum_saccade_frequency=3, kalman_method=False, de_lag=True,
                         prominance=(1,30), relative_start_velo=False,
                         **saccade_kwargs):
        """Find saccades in an array of values. 
        
        We used a variant of the procedure from Bender and Dickinson (2006):
        0. apply a median filter (instead of the low-pass Butterworth filter).
        1. threshold the velocity gradient for speeds above 300 or 350 degs/s.
        2. find the peak velocity
        3. find the start and stop points to include velocities > .25*peak velocity
            * optionally, we can also account for baseline velocities by subtracting 
            the initial velocity
    
        Parameters
        ----------
        threshold_speed : float, default=350 degrees per second
            The minimum speed used for detecting saccades.
        speed_noise_method: bool, default=False
            Whether to use the speed distribution time series to detect saccades.
        acceleration_method : bool, default=False
            Whether to use acceleration data for better saccade extraction.
        maximum_saccade_frequency : float, default=3
            The maximum saccade frequency used for finding peak velocities pertaining 
            to saccade torque spikes.
        kalman_method : bool, default=False
            Whether to use a Kalman filter for smoothing the data before finding saccades.
        de_lag : bool, default=True
            If using the acceleration method, whether to use cross-correlation to 
            avoid phase errors due to smoothing.
        prominance : tuple, default=(1, 30)
            The prominence of the peaks used for finding saccades.
        relative_start_velo : bool, default=False
            Whether to use relative start velocity for saccade detection.
        **saccade_kwargs
            Parameters for processing the saccades.
        """
        arr = np.copy(self.arr[np.newaxis])
        arr = np.unwrap(arr, axis=-1)
        if speed_noise_method:
            # todo: try using Kalman Filter instead
            # let's find the best jerk_std and measurement noise for fitting the heading data
            # kfilter = KalmanFitter(arr[0])
            # kalman_filter_params = (100, .3)
            # vals_filtered = np.unwrap(kfilter.generate_vals(kalman_filter_params[0], kalman_filter_params[1]), axis=-1)
            # # use nonlinear optimization to find the best parameters for the Kalman filter
            # from scipy.optimize import minimize
            # def cost_function(params):
            #     jerk_std, measurement_noise = params
            #     vals_filtered = np.unwrap(kfilter.generate_vals(jerk_std, measurement_noise), axis=-1)
            #     # calculate the cost as the sum of squared errors
            #     return np.sum((arr[0] - vals_filtered)**2)
            # res = minimize(cost_function, kalman_filter_params, method='Nelder-Mead', bounds=((0, 1000000), (1, 100)), options={'maxiter': 1000})
            # vals_filtered = np.unwrap(kfilter.generate_vals(res.x[0], res.x[1]*10), axis=-1)
            arr_clean, arr_mask, valid_start, valid_end = _handle_nans_for_filter(arr[0])
            vals_filtered_fwd = butterworth_filter(arr_clean[np.newaxis], low=0, high=min(15, round(self.framerate/2)-1), sample_rate=self.framerate)
            vals_filtered_rev = butterworth_filter(arr_clean[np.newaxis, ::-1], low=0, high=min(15, round(self.framerate/2)-1), sample_rate=self.framerate)
            vals_filtered = np.zeros_like(arr[0])
            vals_filtered.fill(np.nan)
            vals_filtered[valid_start:valid_end] = np.unwrap((vals_filtered_fwd + vals_filtered_rev[:, ::-1]) / 2, axis=-1)
            # vals_filtered = np.unwrap(vals_filtered, axis=-1)
            # plt.plot(range(len(arr[0])), arr[0]) 
            # plt.plot(vals_filtered)
            # plt.show()
            # maybe instead of a kalman filter, I can use the butterworth filter
            self.saccades = []
            # use raw velocity to get the confidence interval based on a rolling window of variance
            # velocity = np.gradient(arr[0])
            # we need to account for NaNs when calculating the velocity, so we can use np.gradient on the filtered values instead
            # and we'll calculate the gradient ourselves
            velocity = np.gradient(vals_filtered)
            velocity *= self.framerate
            # find peak velocities using a peak finding algorithm
            # find the number of frames corresponding to 100 ms, because saccades are 
            # unlikely to occur that frequently
            dist = .25 * self.framerate
            # see if any of the find_peaks parameters were provided via saccade_kwargs
            distance, width, prominance, wlen = saccade_kwargs.get('distance', dist/4), saccade_kwargs.get('width', 2), saccade_kwargs.get('prominence', (1, 30)), saccade_kwargs.get('wlen', dist)
            peaks = scipy.signal.find_peaks(np.abs(velocity), distance=distance, width=width, prominence=prominance, wlen=wlen)
            # test:
            # fig, axes = plt.subplots(nrows=2, sharex=True)
            # axes[0].plot(vals_filtered)
            # # plt.sca(axes[1])
            # plt.plot(velocity, zorder=2)
            # velos = velocity[peaks[0]]
            # plt.scatter(peaks[0], velos, marker='o', color=green, zorder=3)
            # # velos = velocity[neg_peaks[0]]
            # # plt.scatter(neg_peaks[0], velos, marker='o', color=red, zorder=3)
            # for lb, ub in zip(peaks[1]['left_ips'], peaks[1]['right_ips']): axes[0].axvspan(lb, ub, color='gray', alpha=.3, zorder=1); axes[1].axvspan(lb, ub, color='gray', alpha=.3, zorder=1)
            # plt.show()
            # # remove outlier peaks
            starts, stops = peaks[1]['left_bases'], peaks[1]['right_bases']
            # # find the rolling CI of the mean based on the past 5 velocities
            # series = pd.Series(np.abs(velocity))
            # window = 5
            # # rolling_mean = series.rolling(window, center=False).mean()
            # # rolling_median = series.rolling(window, center=True).median()
            # rolling_std = series.rolling(window, center=False).std()
            # lower, upper = - 2*rolling_std, 2*rolling_std
            # # plot the velocity and the lower and upper bounds
            # fig = plt.figure()
            # plt.plot(np.abs(velocity), color='k', label='velocity')
            # plt.plot(lower, color='red', label='lower bound')
            # plt.plot(upper, color='green', label='upper bound')
            # plt.show()

            # lower, upper = lower[:-1], upper[:-1]
            # upward = velocity[1:] > upper
            # downward = velocity[1:] < lower
            # test: can we simply use these cross points to get candidate saccade starts and then apply a duration filter
            # outlier_thresh = 1
            # outlier_up = (velocity - velocity.mean())/velocity.std() > outlier_thresh
            # outlier_down = (velocity - velocity.mean())/velocity.std() < -outlier_thresh
            # todo: find major peak velocities
            # peaks = scipy.signal.find_peaks(abs(velocity), prominence=.9)
            # peaks, lb, ub = peaks[0], peaks[1]['left_bases'], peaks[1]['right_bases']
            # for low, peak, high in zip(lb, peaks, ub): plt.axvline(low, color='g'); plt.axvline(high, color='red'); plt.axvline(peak, color='k'); plt.plot(velocity, color='k')
            # outlier_up = velocity > (np.pi / 180. * threshold_speed)
            # outlier_down = velocity < (np.pi / 180. * -threshold_speed)
            # go through each break in the cw and ccw boolean arrays
            # starts, stops = [], []
            # for dir_arr, direction, thresh, outliers in zip([np.where(upward)[0], np.where(downward)[0]], [np.greater, np.less], [upper, lower], [outlier_up, outlier_down]):
            #     ind = 0
            #     starts_pot = np.array(dir_arr)
            #     while len(starts_pot) > 0:
            #         start = starts_pot[0]
            #         velo = velocity[start+1:]
            #         bound = thresh[start]
            #         # the stop point is the last value within bounds
            #         in_bounds = direction(velo, bound)
            #         stops_pot = np.where((in_bounds[:-1] == True) * (in_bounds[1:] == False))[0]
            #         if len(stops_pot) > 0:
            #             stop = stops_pot[0]
            #             stop += start
            #             # check that the sequence has an outlier velocity
            #             is_fast = outliers[start:stop+1]
            #             # check if there is a peak within this range
            #             is_peak = (peaks[0] > start) * (peaks[0] < stop)
            #             if np.any(is_fast) or np.any(is_peak):
            #             # if np.any(is_peak):
            #                 # check that 
            #                 # plt.plot(velo)
            #                 # plt.axhline(bound)
            #                 # plt.axvspan(0, stop-start, color='gray', alpha=.3)
            #                 starts += [start]
            #                 stops += [stop]
            #             # remove all starts less than stop
            #             starts_pot = starts_pot[starts_pot > start]
            #         else:
            #             starts_pot = starts_pot[1:]
            # store the resulting saccade
            velos = [max(abs(np.gradient(saccade.arr))) * 60 for saccade in self.saccades]
            for start, stop in zip(starts, stops):
                saccade = Saccade(self.arr, self.trial, self.test_ind, self.framerate, start, stop, **saccade_kwargs)
                # velos = np.gradient(saccade.arr) * self.framerate
                # peak_velo = abs(velos).max()
                peak_velo = abs(saccade.peak_velocity) * 180 / np.pi
                if peak_velo < threshold_speed:
                    saccade.success = False
                if saccade.success:
                    self.saccades += [saccade]
            # eliminate any overlapping saccades, keeping the longer one
            # saccade = self.saccades[1]
            # start, stop = saccade.start, saccade.stop
            # saccade = Saccade(self.arr, self.framerate, start, stop, display=True, **saccade_kwargs)
            # plt.show()
            new_saccades = []
            while len(self.saccades) > 0:
                saccade = self.saccades[0]
                self.saccades.remove(saccade)
                # get range of frames included in the other saccades
                spans = [range(sacc.start, sacc.stop+1) for sacc in self.saccades]
                # check if start or stop is within
                overlapping_saccades = [saccade]
                for other_saccade in self.saccades: 
                    other_span = range(other_saccade.start, other_saccade.stop+1)
                    is_contained = (saccade.start in other_span) or (saccade.stop in other_span)
                    span = range(saccade.start, saccade.stop+1)
                    contains_other = (other_saccade.start in span) or (other_saccade.stop in span)
                    same_span = (saccade.start == other_saccade.start) or (saccade.stop == other_saccade.stop)
                    if is_contained or contains_other or same_span:
                        overlapping_saccades += [other_saccade]
                for o_saccade in overlapping_saccades:
                    if o_saccade in self.saccades:
                        self.saccades.remove(o_saccade)
                if len(overlapping_saccades) > 1:
                    # we will remove all of these and replace with the longest duration saccade
                    durs = np.array([saccade.duration for saccade in overlapping_saccades])
                    ind_keep = np.argmax(durs)
                    new_saccades += [overlapping_saccades[ind_keep]]
                else:
                    new_saccades += [saccade]
            self.saccades = new_saccades
            if 'display' in saccade_kwargs.keys():
                if saccade_kwargs['display']:
                    fig, axes = plt.subplots(nrows=3, sharex=True)
                    # for the gradients, we need to account for NaNs

                    axes[0].plot(self.time, np.gradient(velocity), color='k', marker='.')
                    axes[1].plot(self.time, velocity * 180 / np.pi, color='k', marker='.')
                    axes[2].plot(self.time, arr[0])
                    # axes[1].fill_between(self.time + self.time[1], velocity - 2*rolling_std, velocity + 2*rolling_std, color='gray')
                    # axes[0].plot(self.time, rolling_median)
                    for peak in peaks[0]: axes[1].axvline(self.time[peak], color='r')
                    for saccade in self.saccades: axes[2].axvspan(self.time[saccade.start], self.time[saccade.stop], color=blue, alpha=.2)
                    for start, stop in zip(np.round(starts).astype(int), np.round(stops).astype(int)): axes[0].axvspan(self.time[start], self.time[stop], color='gray', alpha=.2)
                    for start, stop in zip(np.round(starts).astype(int), np.round(stops).astype(int)): axes[1].axvspan(self.time[start], self.time[stop], color='gray', alpha=.2)
                    for start, stop in zip(np.round(starts).astype(int), np.round(stops).astype(int)): axes[2].axvspan(self.time[start], self.time[stop], color='gray', alpha=.2)
                    plt.show()
        elif kalman_method:
            kfilter = KalmanFitter(np.unwrap(arr[0]))
            smoothed_arr = np.unwrap(kfilter.generate_vals(100, .01))
            velocity = np.gradient(smoothed_arr)
            acceleration = np.gradient(velocity)
            # test: compare acceleration method of butterworth vs. kalman filter
            fig, axes = plt.subplots(nrows=3, sharex=True)
            plt.sca(axes[0])
            plt.plot(smoothed_arr, label='model')
            plt.scatter(range(len(arr[0])), arr[0], color='k', marker='.', label='data')
            plt.legend()
            plt.sca(axes[1])
            plt.plot(velocity, marker='.')
            plt.plot(np.gradient(arr[0]), '-k', marker='.')
            plt.sca(axes[2])
            plt.plot(acceleration, marker='.')
            plt.plot(np.gradient(np.gradient(arr[0])), '-k', marker='.')
            # plt.show()
        elif acceleration_method:
            # we need a reliable measure of the acceleration data,
            # so let's apply a low-pass filter before taking gradients
            # smoothed_arr = butterworth_filter(np.unwrap(arr[0], axis=-1), 0, 6, 60)
            # smoothed_arr = smoothed_arr
            # breakpoint()
            # try using the kalman filter instead of the the butterworth
            kfilter = KalmanFitter(arr[0])
            smoothed_arr = np.unwrap(kfilter.generate_vals(300, .3))
            velocity = np.gradient(smoothed_arr)
            velocity *= self.framerate
            acceleration = np.gradient(velocity)
            acceleration *= self.framerate
            velocity_og = np.gradient(self.arr)
            # each step is idenfied as a local maximum in the velocity data
            # find peaks by setting the minimum time between saccades
            peaks = scipy.signal.find_peaks(np.abs(velocity), distance=self.framerate / maximum_saccade_frequency, prominence=.3)
            troughs = scipy.signal.find_peaks(-np.abs(acceleration), prominence=.3)[0]
            # add the in and out points of the bout to the troughs
            first_val, last_val = 0, len(smoothed_arr)
            if first_val not in troughs:
                troughs = np.append([first_val], troughs)
            if last_val not in troughs:
                troughs = np.append(troughs, [last_val])
            # for each peak, find the start and stop using the acceleration data
            self.saccades = []
            for peak in peaks[0]:
                # using the absolute value of acceleration, we now have a bimodal function in each direction
                # the start and stop are the previous and proceeding troughs in the acceleration data
                if np.any(troughs < peak) and np.any(troughs > peak):
                    start, stop = max(troughs[troughs < peak]), min(troughs[troughs > peak])
                    # todo: optionally, we can go back to the raw heading data and find the segment closest to the smoothed one here
                    # get the cross correlation of the smoothed segment with the original array
                    arr = smoothed_arr
                    if de_lag:
                        saccade = smoothed_arr[start:stop]
                        saccade_velo = velocity[start:stop]
                        # if max(abs(saccade_velo)) >= threshold_speed * np.pi / 180.:
                        saccade_velo_og = velocity_og[start:stop]
                        saccade_acceleration = np.gradient(saccade_velo)
                        saccade_acceleration_og = np.gradient(saccade_velo_og)
                        # plt.plot(acceleration)
                        # for trough in troughs: plt.axvline(trough, color='r')
                        # for peak in peaks[0]: plt.axvline(peak, color='g')
                        # plt.show() 
                        # corrs = scipy.signal.correlate(velocity_og, saccade_velo, mode='valid')
                        tmin, tmax = max(0, start - 10), min(len(arr), stop + 10)
                        offset = np.nanmean(saccade)
                        lags, corrs = normal_correlate(self.arr[tmin:tmax] - offset, saccade - offset, mode='valid', framerate=1)
                        # lags = scipy.signal.correlation_lags(len(self.arr), len(saccade), mode='valid')
                        # check lags within .5 seconds of the start
                        # included = (lags > start - .5 * self.framerate) * (lags < start + .5 * self.framerate)
                        included = (lags > 0) * (lags < 10)
                        # find the lag with the greatest 
                        if np.any(included):
                            peak_lag = round(lags[included][np.argmax(corrs[included])])
                            # peak_lag *= self.framerate
                            # peak_lag -= (stop - start)/2
                            peak_corr = corrs[peak_lag]
                            # test: check that cross-correlation will work
                            # fig, axes = plt.subplots(nrows=2, sharex=False)
                            # axes[0].plot(np.arange(peak_lag, peak_lag + len(saccade)), saccade - offset)
                            # axes[0].plot(self.arr[tmin:tmax] - offset)
                            # axes[1].scatter(lags * self.framerate, corrs, marker='.')
                            # axes[1].axhline(0, color='k', linestyle='--')
                            # plt.show()
                            # todo: fix the damned lag offset! It's not lining up correctyl!!
                            # only include the saccade if the peak correlation is substantial
                            # replace start and stop with the shifted times
                            # arr = self.arr
                            if peak_corr > .15:
                                dur = stop - start
                                start = round(tmin + peak_lag)
                                stop = round(start + dur)
                            stop = min(len(self.arr)-1, stop)
                            saccade = Saccade(self.arr, self.trial, self.test_ind, self.framerate, start, stop, display=True, **saccade_kwargs)
                            if abs(saccade.peak_velocity) < threshold_speed * np.pi / 180.:
                                saccade.success = False
                            if saccade.success:
                                self.saccades += [saccade]
            fig = plt.figure()
            plt.plot(self.arr)
            for saccade in self.saccades:
                plt.axvspan(saccade.start, saccade.stop, color='gray', alpha=.25)
            plt.show()
        else:
            # from Bender and Dickinson (2006)
            # 0. median filter
            # arr = scipy.signal.medfilt(arr, 101)
            arr = scipy.signal.medfilt(arr, 5)
            # 1. find all angular velocities > 350 degs/s
            velocity = np.gradient(arr)
            self.velocity = np.append([0], velocity)
            # convert to angular velocity
            self.velocity *= self.framerate  # convert to rads/second
            # todo: try using the smoothed velocity and acceleration 
            include = abs(self.velocity) > (threshold_speed * np.pi / 180.)  # Mongeau used 350 degs/second, but that excludes some of our slower saccades
            # plt.plot(original_arr); plt.plot(include); plt.show()     
            diffs = include.astype(int)[1:] - include.astype(int)[:-1]
            diffs = np.append([0], diffs)
            starts = np.where(diffs > 0)[0]
            stops = np.where(diffs < 0)[0]
            # collect the saccades
            self.saccades = []
            if len(starts) > 0 and len(stops) > 0:
                # if the first stop happens before the first start, remove it
                if stops[0] < starts[0]:
                    while stops[0] < starts[0]:
                        stops = stops[1:]
                # if the last start happens after the last stop, remove it
                if starts[-1] > stops[-1]:
                    while starts[-1] < stops[-1]:
                        starts = starts[:-1]
                # if any stops are `adjacent to starts, 
                while len(starts) > 0 and len(stops) > 0:
                    start = starts[0]
                    starts = starts[1:]
                    # find the next stop
                    stop = stops[0]
                    while stop < start and len(stops) > 1:
                        stops = stops[1:]
                        stop = stops[0]
                    if len(stops) > 0 and stop > start and start > 10:
                        inds = np.arange(start, stop)
                        # 2. find the ind that has the maximum velocity
                        if relative_start_velo:
                            start_velocity = self.velocity[start-10:start-5].mean()
                        else:
                            start_velocity = 0
                        relative_velocity = self.velocity - start_velocity
                        segment_velocity = relative_velocity[start:stop]
                        peak_ind = inds[np.argmax(abs(segment_velocity))]
                        peak_velocity = relative_velocity[peak_ind]
                        # 3. find the start and stop points to include velocities > .25*peak velocity
                        threshold_velocity = peak_velocity / 4
                        # test: plot the velocity and threshold. check that this behaves as expected
                        # fig, ax = plt.subplots()
                        # ax.plot(original_arr, 'ok')
                        # ax.plot(velocity)
                        # ax.axhline(threshold_velocity)
                        # plt.show()
                        # threshold depends on velocity sign
                        if peak_velocity > 0:
                            # included = self.velocity > threshold_velocity
                            included = relative_velocity > threshold_velocity
                        else:
                            # included = self.velocity < threshold_velocity
                            included = relative_velocity < threshold_velocity
                        included_inds = np.where(included)[0]
                        # find lowest index with velocity > threshold_velocity that is continuous with the peak
                        # remove any indices with diff > 1
                        diffs = np.diff(included_inds)
                        group_change = np.append([False], diffs > 1)
                        group_lbls = np.cumsum(group_change)
                        included_group = group_lbls[included_inds == peak_ind][0]
                        included_inds = included_inds[group_lbls == included_group]
                        # get the in point as the lowest ind in the included group
                        in_point, out_point = included_inds.min()-1, included_inds.max()+1, 
                        # include if the end point is not the last frame of the test
                        if out_point < 359:
                            # collect Saccades of each slice of the array
                            self.saccades += [Saccade(arr, self.trial, self.test_ind, self.framerate, start=in_point, stop=out_point, **saccade_kwargs)]


    def query_saccades(self, output='heading', time_var='time', reference_time='start', start=0, stop=np.inf, min_speed=350,
                       max_speed=np.inf, **query_kwargs):
        """Grab saccade data between the start and stop times.
        
        Parameters
        ----------
        output : str, default='heading'
            The variable to output. Can also be velocity.
        time_var : str, default = 'time'
            The saccade time frame to use for selecting 
        reference_time : str, default='start'
            The time to use as the reference for subsetting the saccades. 
            For instance, we could look for all saccades that start (vs stop)
            when another variable is at a certain value.
        start, stop : float, default=0., -1.
            The start and stop times to include in the saccade.
        sort_by : str, default = 'time'
            The parameter to use for sorting the trials.
        """
        times, saccades = [], []
        # allow for subseting and sorting just like the other query functions
        # we can only use parameters that are being collected by the Saccade object
        for saccade in self.saccades:
            include = True
            if 'subset' in query_kwargs:
                # check if the subset condition is met by this saccade
                subset = query_kwargs['subset']
                # using the subset conditions, check if this saccade meets the criteria
                for key, vals in subset.items():
                    # convert vals to a list if it is not already
                    if not isinstance(vals, list):
                        vals = [vals]
                    for val in vals:
                    # we need to process saccade parameters differently from bout and trial data
                        key_val = None
                        for obj in [self.trial, self, saccade]:
                            if key in dir(obj):
                                key_vals = obj.__getattribute__(key)
                                key_val = key_vals
                        if key_val is not None:
                            if isinstance(key_vals, h5py.Dataset):
                                # convert to a numpy array
                                key_vals = np.array(key_vals)
                                if key_vals.dtype.type == np.bytes_:
                                    key_vals = np.array([k.decode('utf-8') for k in key_vals])
                            # if key_vals is an array, we need to use the reference time to subset
                            if isinstance(key_vals, np.ndarray):
                                # if obj is a trial, we need to find the data specific to that bout
                                if len(key_vals) == len(self.trial.test_ind):
                                    key_vals = key_vals[self.test_ind]
                                    key_val = key_vals
                                if isinstance(key_val, np.ndarray):
                                    if len(key_vals) == len(saccade.time):
                                        if reference_time == 'start':
                                            key_val = key_vals[saccade.start]
                                        elif reference_time == 'stop':
                                            key_val = key_vals[saccade.stop]
                                        else:
                                            raise ValueError(f'`reference_time` must be either "start" or "stop".')
                            # interpret val for inequalities
                            if isinstance(val, (str, bytes)):
                                starts = [saccade.start for saccade in self.saccades]
                                logic, thresh = interprate_inequality(val)
                                if isinstance(key_val, bytes):
                                    key_val = key_val.decode('utf-8')
                                include *= logic(key_val, thresh)
                            else:
                                include *= key_val == val
            if include:
                peak_speed = abs(saccade.peak_velocity) * 180. / np.pi
                if (peak_speed > min_speed) and (peak_speed < max_speed):
                    time, saccade = saccade.query(output=output, time_var=time_var, start=start, stop=stop)
                    times += [time]
                    saccades += [saccade]
        return times, saccades

class Saccade():
    def __init__(self, arr, trial, test_ind, framerate=1, start=0, stop=-1, interpolate_velocity=True, baseline_comparison=False, baseline_z=2, baseline_padding=10, display=False, baseline_test=True, **kwargs):
        """Wrapper for saccade time series and measurements.

        Parameters
        ----------
        arr : np.ndarray
            The heading time series from the whole trial.
        trial : TrackingTrial or None
            The parent trial containing this saccade. May be None when
            reconstructing on-demand from stored start/stop values.
        test_ind : int or None
            The index of the test within the trial. May be None for
            on-demand reconstruction.
        framerate : float, default=1
            The framerate of the time series.
        start, stop : int, default=0, -1
            The frames of original_arr marking the start and stop of the saccade. 
            Defaults to the whole array if unspecified.
        interpolate_velocity : bool, default=True
            Whether to interpolate the velocity value to find the velocity peak.
        baseline_comparison : bool, default=True
            Whether to use the baseline distribution of velocities to correct the
            start and stop points of the saccade.
        baseline_z : float, default=2
            The z-score threshold for defining the baseline velocity distribution.
        baseline_padding : int, default=10
            The number of frames to include in the baseline velocity calculation.
        baseline_test : bool, default=True
            Whether to use the velocity noise distribution to verify if the saccade is valid.
        display : bool, default=False
            Whether to display the resultant saccade start and stop points.

        Attributes
        ----------
        time : np.ndarray
            The time from the start of the saccade.
        start_angle, stop_angle : float
            The first and last heading angles of the saccade.
        duration : float
            The duration of the time series based on the input framerate.
        velocity : np.ndarray
            The velocity time series assuming no initial motion.
        peak_velocity : float
            The maximum discrete velocity measured in the time series.
        amplitude : float
            The total displacement from start to finish.
        baseline_comparison : bool, default=True
            Whether to re-calculate the start and stop frames using the baseline 
            velocity.
        baseline_z : float, default=2
            The z-score threshold for defining the baseline velocity distribution.
        baseline_test : bool, default=True
            Whether to use the velocity noise distribution to verify if the saccade is valid.
        display : bool, default=False
            Whether to display the resultant saccade start and stop points.
        """
        # store the original time series and time
        self.baseline_z = baseline_z
        self.baseline_padding = baseline_padding
        self.trial = trial
        self.test_ind = test_ind
        arr = np.unwrap(arr)
        self.original_arr = arr
        self.start, self.stop = int(round(start)), int(round(stop))
        self.arr = np.asarray(np.copy(self.original_arr)[self.start:self.stop+1])
        self.framerate = framerate
        self.time = np.arange(len(self.original_arr)).astype(float)
        self.time -= self.start
        self.time /= self.framerate
        # get the start and stop times relative to the start of the bout
        self.start_time, self.stop_time = self.start / self.framerate, self.stop / self.framerate
        # calculate the starting and ending angle
        self.start_angle = self.arr[0]
        self.stop_angle = self.arr[-1]
        self.arr_relative = self.original_arr - self.start_angle
        # calculate the amplitude and duration  
        # self.amplitude = self.arr.ptp()
        self.amplitude = self.stop_angle - self.start_angle
        self.duration = self.stop_time - self.start_time
        # store the velocity time series
        self.velocity = np.gradient(self.original_arr)
        self.velocity *= self.framerate
        # get the peak index and velocity
        peak_ind = np.argmax(abs(self.velocity[self.start: self.stop+1]))
        self.peak_ind = peak_ind + self.start
        self.peak_velocity = self.velocity[self.peak_ind]
        # store the acceleration time series
        self.acceleration = np.gradient(self.velocity)
        self.acceleration *= self.framerate
        if interpolate_velocity:
            # use a spline interpolation of the velocity to find the interpolated maximum
            start_frame, stop_frame = max(self.start - self.baseline_padding, 0), min(self.stop+1+self.baseline_padding, len(self.time))
            # start_frame, stop_frame = max(self.start - 10, 0), min(self.stop+10, len(self.time))
            interp_func = scipy.interpolate.interp1d(self.time[start_frame: stop_frame], self.velocity[start_frame: stop_frame], kind='cubic')
            # new_times = np.linspace(self.time[start_frame], self.time[stop_frame-1], 1000)
            new_times = np.linspace(0, self.duration, 1000)
            new_times = new_times[new_times <= self.time[start_frame:stop_frame].max()]
            new_velocity = interp_func(new_times)
            peak_ind = np.argmax(abs(new_velocity))
            self.peak_velocity = new_velocity[peak_ind]
            self.peak_time = new_times[peak_ind]
            self.relative_time = np.copy(self.time)
            self.relative_time -= self.peak_time
            self.start_relative_time = np.copy(self.time)  # matches arr_relative reference
            # measure the max velocity from the few frames before the start and use as a threshold 
            # for the start of the saccade
            velos_included = self.velocity[start_frame:self.start]
            # assume the velocity is normally distributed
            self.success = False
            if len(velos_included) > 0:
                velo_mean, velo_std = np.nanmean(velos_included), np.nanstd(velos_included)
                velo_floor, velo_ceiling = velo_mean - self.baseline_z * velo_std, velo_mean + self.baseline_z * velo_std
                # test: plot the headings, velocity, and angular acceleration highlighting the saccade interval and peak velocity
                ta, tb = max(0, self.start-10), min(len(self.velocity), self.stop + 11)
                if display:
                    fig, axes = plt.subplots(nrows=3, sharex=True)
                    axes[0].plot(self.time[ta: tb], 180./ np.pi * self.original_arr[ta:tb], 'ko-')
                    axes[1].plot(self.time[ta:tb], 180./ np.pi * self.velocity[ta: tb], 'ko-')
                    axes[1].plot(new_times, 180./ np.pi * new_velocity, 'r-')
                    axes[1].scatter(self.peak_time, 180./ np.pi * self.peak_velocity, marker='o', color='r')
                    for ax in axes: ax.axvline(self.peak_time, color='r')
                    axes[2].plot(self.time[ta: tb], 180./ np.pi * np.gradient(self.velocity[ta: tb]), 'ko-')
                    axes[0].axvspan(0, self.duration, color='gray', alpha=.25)  
                    axes[1].axvspan(0, self.duration, color='gray', alpha=.25)
                    axes[1].axhline(velo_ceiling * 180 / np.pi, color='k', linestyle='--')
                    axes[1].axhline(velo_floor * 180 / np.pi, color='k', linestyle='--')
                    axes[2].axvspan(0, self.duration, color='gray', alpha=.25)
                    # todo: why are the start and stop time points offset from those that I input?
                # todo: get the initial velocity and use this to update the start and stop points
                # the new start is the frame before the first frame with a high speed
                # the new stop is the first frame below threshold
                self.success = True
                if baseline_comparison:
                    offset = -5
                    xmin = max(0, self.start + offset)
                    xmax = min(len(self.velocity), self.stop + 10)
                    if self.peak_velocity > 0:
                        saccading = np.where(self.velocity[xmin: xmax] > velo_ceiling)
                        not_saccading = np.where(self.velocity[xmin: xmax] <= velo_ceiling)
                    else:
                        saccading = np.where(self.velocity[xmin: xmax] < velo_floor)
                        not_saccading = np.where(self.velocity[xmin: xmax] >= velo_floor)
                    saccading, not_saccading = saccading[0], not_saccading[0]
                    not_saccading += xmin
                    saccading += xmin
                    # check if there are any breaks in the saccading frames. We want to use the segment that still includes the peak velocity
                    diffs = np.diff(saccading)
                    if np.any(diffs > 1):
                        # split up saccading into segments including only 
                        splits = np.where(diffs > 1)[0] + 1
                        saccading_split = np.split(saccading, splits)
                        # use just the segment that includes the start index
                        peak_included = [sacc for sacc in saccading_split if self.peak_ind in sacc]
                        if len(peak_included) > 0:
                            saccading = peak_included[0]
                        else:
                            saccading = []
                    if len(saccading) > 0:
                        self.start_original, self.stop_original = self.start, self.stop
                        self.start, self.stop = max(0, saccading.min() - 1), min(saccading.max() + 1, len(self.original_arr) - 1)
                        start_frame, stop_frame = max(self.start - self.baseline_padding, 0), min(self.stop+ self.baseline_padding, len(self.time))
                        if self.start == self.stop:
                            self.success = False
                            # if self.stop == len(self.original_arr) - 1:
                            #     self.start -= 1
                            # else:
                            #     self.stop += 1
                        self.start_angle, self.stop_angle = self.original_arr[self.start], self.original_arr[self.stop]
                        # re-calculate the saccade parameters
                        # subset the original array to the new start and stop points
                        self.arr = np.copy(self.original_arr)[self.start:self.stop+1]
                        # re-calculate the time series
                        self.time = np.arange(len(self.original_arr)).astype(float)
                        self.time -= self.start
                        self.time /= self.framerate
                        # re-calculate the start and stop times
                        self.start_time, self.stop_time = self.start / self.framerate, self.stop / self.framerate
                        # re-calculate the amplitude
                        self.amplitude = self.stop_angle - self.start_angle
                        # re-calculate the relative heading array
                        self.arr_relative = self.original_arr - self.start_angle
                        # re-calculate the duration
                        self.duration = (self.stop - self.start) / self.framerate
                        # re-calculate the peak velocity and relative time
                        if interpolate_velocity:
                            # start_frame, stop_frame = max(self.start - 10, 0), min(self.stop+10, len(self.time))
                            interp_func = scipy.interpolate.interp1d(self.time[start_frame: stop_frame], self.velocity[start_frame: stop_frame], kind='cubic')
                            # new_times = np.linspace(self.time[start_frame], self.time[stop_frame-1], 1000)
                            new_times = np.linspace(0, self.duration, 1000)
                            new_times = new_times[new_times <= self.time[start_frame:stop_frame].max()]
                            new_velocity = interp_func(new_times)
                            try:
                                peak_ind = np.argmax(abs(new_velocity))
                            except:
                                print("ValueError: attempt to get argmax of an empty sequence")
                            self.peak_velocity = new_velocity[peak_ind]
                            self.peak_time = new_times[peak_ind]
                            self.relative_time = np.copy(self.time)
                            self.relative_time -= self.peak_time
                            self.start_relative_time = np.copy(self.time)  # matches arr_relative reference
                        else:
                            # try:    
                            peak_ind = np.argmax(abs(self.velocity[self.start: self.stop]))
                            self.peak_velocity = self.velocity[self.start: self.stop][peak_ind]
                            # except:
                            #     print("peak_ind seems to be larger than self.velocity")
                            # re-calculate the relative time
                            self.peak_time = self.time[self.start: self.stop][peak_ind]
                            self.relative_time = np.copy(self.time)
                            self.relative_time -= self.peak_time
                            self.start_relative_time = np.copy(self.time)  # matches arr_relative reference
                        if self.stop - self.start < 2 or self.duration > 1.5:
                            self.success = False
                        # optionally, check if the new peak velocity is within the bounds
                        if baseline_test:
                            # check if peak velocity is outside of the velocity bounds
                            if self.peak_velocity < 0 and self.peak_velocity > velo_floor:
                                self.success = False
                            elif self.peak_velocity > 0 and self.peak_velocity < velo_ceiling:
                                self.success = False
                        if self.success and display:
                            # plot the new saccade spans
                            offset = self.time[self.start] - self.time[self.start_original]
                            axes[0].axvspan(self.time[self.start] + offset, self.time[self.stop] + offset, color=blue, alpha=.5)
                            axes[1].axvspan(self.time[self.start] + offset, self.time[self.stop] + offset, color=blue, alpha=.5)
                            axes[2].axvspan(self.time[self.start] + offset, self.time[self.stop] + offset, color=blue, alpha=.5)
                    else:
                        self.success = False
        else:
            peak_ind = np.argmax(abs(self.velocity[self.start: self.stop]))
            self.peak_velocity = self.velocity[self.start: self.stop][peak_ind]
            self.peak_time = self.time[self.start: self.stop][peak_ind]
            self.relative_time = np.copy(self.time)
            self.relative_time -= self.peak_time
            self.start_relative_time = np.copy(self.time)  # matches arr_relative reference

    def query(self, output='heading', time_var='time', start=0, stop=0):
        """Grab the saccade data between the start and stop times.

        In order to grab values relative to the start or stop point, which will vary between saccades,
        insert a string for the start or stop +/- the time difference. For example, to grab the values 
        between the start and .2 seconds after the start, set start='start' or 0 and stop='start+.2'. 
        Or to grab values between .1 second before and .3 seconds after the end of the saccade, set 
        start='stop-.1' and stop=.3 or stop='stop+.3'. 

        Parameters
        ----------
        output : str, default='heading'
            The output variable. Can also be velocity.
        time_var : str, default = 'time'
            The saccade time frame to use for selecting 
        start, stop : float or str, default=0.
            The start and stop times to include in the saccade relative to the start and stop of the 
            saccade, respectively. You can explicitly refer to times relative to the start or stop times

        """
        time = self.__getattribute__(time_var)
        # add feature for choosing time points relative to the start and stop of the saccade
        if isinstance(start, str):
            # default to the start
            base = 'start'
            if 'stop' in start:
                base = 'stop'
            start = start.replace(base, '')
            base = self.__getattribute__(base)
            base_val = self.time[base]
            # default to 0
            delta = 0
            if len(start) > 0:
                delta = float(start)
            start = base_val + delta
        if isinstance(stop, str):
            # default to the stop point
            base = 'stop'
            if 'start' in stop:
                base = 'start'
            stop = stop.replace(base, '')
            base = self.__getattribute__(base)
            base_val = self.time[base]
            # default to 0
            delta = 0
            if len(stop) > 0:
                delta = float(stop)
            stop = base_val + delta
        # get frames within the start and stop times
        include = (time >= start) * (time < stop)
        if output == 'heading':
            ret = self.original_arr - self.start_angle
            return time[include], ret[include]
        elif output == 'velocity':
            ret = self.velocity
            return time[include], ret[include]
        elif output == 'saccade':
            return time[include], self            

def angle_rgb(angles, sat=.5, val=.7, period=2*np.pi):
    """Return a color for the given angle for a circular cmap."""
    # if the angle values are strings, just use a range of integers
    if isinstance(angles[0], (str, bytes)):
        angles = np.arange(len(angles))
    hues = (angles % period) / period
    sats, vals = np.repeat(sat, len(hues)), np.repeat(val, len(hues))
    hsv = np.array([hues, sats, vals]).T
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb

def normal_correlate(arr1, arr2, framerate=60., mode='same',
                     circular=True):
    """Normalized cross correlation so that outcome is pearson correlation.

    Parameters
    ----------
    arr1, arr2 : np.ndarray
        The arrays to cross-correlate.
    framerate : int, default=60
        The framerate assumed for calculating lags.
    mode : str, default='same'
        A string indicating the size of the output. See the documentation
        scipy.signal.correlate for more information.
    circular : bool, default=True
        Whether to assume that the data are circular or periodic.
    """
    mean, std = np.nanmean, np.nanstd
    # if circular:
    #     mean, std = stats.circmean, stats.circstd
    arr1_std = (arr1 - mean(arr1)) / (
                std(arr1) * len(arr1))
    arr2_std = (arr2 - mean(arr2)) / (std(arr2))
    corr = scipy.signal.correlate(arr1_std, arr2_std, mode=mode)
    lags = scipy.signal.correlation_lags(len(arr1_std), len(arr2_std), mode=mode).astype('float')
    lags /= framerate
    return lags, corr

def butterworth_filter(vals, low=1, high=6, sample_rate=60):
    """Apply a butterworth to the specified dataset.

    Parameters
    ----------
    vals : np.ndarray
        The array to be filtered.
    low : float, default=1
        The lower bound of the bandpass filter in Hz.
    high : float, default=6
        The upper bound of the bandpass filter in Hz.
    sample_rate : float, default=60
        The sample rate used for calculating frequencies for the filter.
    """
    # frequencies must be at least 0
    assert low >= 0 and high > 0, "Frequency bounds must be non-negative."
    # copy the values to filtered
    vals = np.copy(vals)
    # unwrap the vals first and then wrap again
    vals = np.unwrap(vals, axis=1)
    if low > 0 and high < np.inf:
        filter = scipy.signal.butter(5, (low, high),
                                fs=sample_rate,
                                btype='bandpass',
                                output='sos')
    elif np.isinf(high):
        filter = scipy.signal.butter(5, low,
                                fs=sample_rate,
                                btype='highpass',
                                output='sos')
    elif low == 0:
        filter = scipy.signal.butter(5, high, fs=sample_rate, btype='lowpass', output='sos')
    # todo: remove initial offset and add after smoothing
    offset = vals[0, 0]
    vals_smoothed = scipy.signal.sosfilt(filter, vals - offset) + offset
    # apply the filter and store with a new name
    return vals_smoothed

def print_progress(part, whole):
    prop = float(part) / float(whole)
    sys.stdout.write('\r')
    sys.stdout.write('[%-20s] %d    %%' % ('=' * int(20 * prop), 100 * prop))
    sys.stdout.flush()

def sigAsterisk(p):
    ret = 'ns'
    l = [[.0001, '****'],[.001, '***'],[.01, '**'],[.05, '*']]
    for v in l[::-1]:
        if p <= v[0]:
            ret = v[1]
    return ret

def plot_diff_brackets(label, x1, x2, y1, y2, y_label, col='k',
                       vert=False, ax=None, lw=None, size='medium'):
    if ax is None:
        ax = plt.gca()
    if lw is None:
        lw = ax.spines['bottom'].get_linewidth()
    if vert:
        ax.plot([y_label, y_label], [x1, x2],
                color=col, clip_on=False, lw=lw)
        ax.text(y_label, (x1+x2)*.5, label, ha='center', va='bottom',
                 color=col, rotation='vertical', size=size)
    else:
        ax.plot([x1, x2], [y_label, y_label],
                color=col, clip_on=False, lw=lw)
        ax.text((x1+x2)*.5, y_label, label, ha='center', va='bottom',
                 color=col, rotation='horizontal', size=size)

def moving_average(x, w): return np.convolve(x, np.ones(w), 'valid') / w

def vector_strength(arr, bins=100):
    """Convert array of angles into a time series of vector strength."""
    # convert angles into unit vectors
    xs, ys = np.cos(arr), np.sin(arr)
    # get cumulitive sum of x and y values 
    # xvals, yvals = np.cumsum(xs), np.cumsum(ys)
    # ts = np.arange(len(xvals)) + 1
    # dists = np.sqrt(xvals**2 + yvals**2)
    # normalize by the number of frames
    # test: what should we expect for a random walk???
    # random_pts = np.random.random((10000, 2, len(ts)))
    # norm = np.linalg.norm(random_pts, axis=1, keepdims=True)
    # random_pts /= norm
    # random_walks = np.cumsum(random_pts, axis=-1)
    # # measure the distance traveled, normalized by the length of the line
    # random_dists = np.sqrt(random_walks[:, 0]**2 + random_walks[:, 1]**2)
    # random_dists_normed = random_dists / ts
    # measure the rolling mean of x and y values
    xmeans, ymeans = moving_average(xs, bins), moving_average(ys, bins)
    dists = np.sqrt(xmeans**2 + ymeans**2)
    # bin the distances 
    return dists

def interprate_inequality(string):
    """Interpret a string as an inequality and return its partial function.

    This assumes that inequalities are provided as a string in this order:
        {variable} {inequality} {value}
    
    For example, "x < 5" returns partial(np.less, 5), which will only return
    True if x is less than 5. Note that spaces are ignored.
    """
    if isinstance(string, bytes):
        string = string.decode('utf-8')
    # string = string.replace(' ', '')
    # default to equality logic
    logic = np.equal
    logic_char = '=='
    # first, identify the specific logic
    logic_conv = {'<':np.less, '<=':np.less_equal, '==':np.equal,'>':np.greater,'>=':np.greater_equal}
    logic_included = False
    for key in logic_conv:
        if key in string:
            logic = logic_conv[key]
            logic_char = key
            logic_included = True
    # then, extract the variable and value
    val = string.split(logic_char)[-1]
    # convert to float if possible
    if logic_included:
        try:
            val = float(val)
        except:
            # this sometimes happens when an inequality happens to be in the variable name
            # if converting to float fails, revert to the original parameter values
            logic = np.equal
            val = string
    # return the partial function
    return logic, val

def mardia_circ_lin(circular_data, linear_data):
    """
    Computes Mardia's rank correlation coefficient for circular-linear data.

    Args:
        circular_data (np.ndarray): Circular data in radians, range [0, 2*pi].
        linear_data (np.ndarray): Linear data.

    Returns:
        float: Mardia's rank correlation coefficient.
    """
    if len(circular_data) != len(linear_data):
        raise ValueError("Inputs must have the same length.")
    # Sort the circular data to get ranks
    circular_ranks = circular_data.argsort().argsort() + 1
    # Sort the linear data to get ranks
    linear_ranks = linear_data.argsort().argsort() + 1
    n = len(circular_data)
    # Calculate the numerator
    numerator = np.sum(np.sin(2 * np.pi * circular_ranks / n) * np.sin(2 * np.pi * linear_ranks / n))
    # Calculate the denominator
    denominator = np.sqrt(np.sum(np.sin(2 * np.pi * circular_ranks / n)**2) * np.sum(np.sin(2 * np.pi * linear_ranks / n)**2))
    # Handle the case where denominator is zero
    if denominator == 0:
        return 0.0
    # Calculate and return the correlation coefficient
    corr = numerator / denominator
    return corr

def _handle_nans_for_filter(data):
    """
    Prepare data for filtering by interpolating intermittent NaNs.
    Preserves leading and trailing NaNs.
    
    Parameters
    ----------
    data : np.ndarray
        Input array that may contain NaNs
        
    Returns
    -------
    filtered_data : np.ndarray
        Data with intermittent NaNs interpolated, ready for filtering
    valid_mask : np.ndarray
        Boolean mask of originally valid (non-NaN) values
    valid_start : int
        Index where valid data starts
    valid_end : int
        Index where valid data ends (exclusive)
    """
    valid_mask = ~np.isnan(data)
    
    # Find the range of valid data (excluding leading/trailing NaNs)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        # All NaNs - return as is
        return data.copy(), valid_mask, 0, 0
    
    valid_start = valid_indices[0]
    valid_end = valid_indices[-1] + 1
    
    # Extract the data range that contains valid values
    data_slice = data[valid_start:valid_end].copy()
    
    # Interpolate any intermittent NaNs within this range
    if np.any(np.isnan(data_slice)):
        # Find NaN positions in the slice
        nan_mask = np.isnan(data_slice)
        # Interpolate using linear interpolation
        x = np.arange(len(data_slice))
        data_slice[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], data_slice[~nan_mask])
    
    return data_slice, valid_mask, valid_start, valid_end


def _restore_nans_after_filter(filtered_slice, original_data, valid_start, valid_end):
    """
    Restore the original NaN pattern after filtering.
    
    Parameters
    ----------
    filtered_slice : np.ndarray
        The filtered data slice
    original_data : np.ndarray
        Original data array with NaNs
    valid_start : int
        Start index of valid data
    valid_end : int
        End index of valid data (exclusive)
        
    Returns
    -------
    result : np.ndarray
        Filtered data with original NaN pattern restored
    """
    result = np.full_like(original_data, np.nan)
    result[valid_start:valid_end] = filtered_slice
    
    # Restore intermittent NaNs if they existed in the original
    original_nans = np.isnan(original_data[valid_start:valid_end])
    if np.any(original_nans):
        result[valid_start:valid_end][original_nans] = np.nan
    
    return result

if __name__ == "__main__":
    tracker = OfflineTracker("..\\arena\\fourier feedback")
    # everything was shifted by pi/2, so the bounds are strange
    # let's update the camera_heading and camera_heading_offline datasets
    # experiment = TrackingExperiment("..\\arena\\fourier feedback", remove_incompletes=False)
    # for each trial, shift values to range -pi to pi
    # for trial in experiment.trials:
    #     for var in ['camera_heading', 'camera_heading_offline']:
    #         if var in dir(trial):
    #             arr = trial.__getattribute__(var)
    #             arr[arr < -np.pi] += 2 * np.pi
    #             trial.add_dataset(var, arr)
    tracker.process_vids(start_over=False)
    tracker.offline_comparison(smooth_offline=True)
