# magno_tracker

Analysis library for plume-tracking flight simulator experiments recorded with
the [MagnoCube rig](https://github.com/jpcurrea/magnocube).  It provides:

- **Lazy HDF5 → xarray/Zarr** data loading with a clean query API
- **Automated saccade detection** with a persistent per-trial saccade table
- **A unified `plot()` entry point** covering five plot types, cross-panel
  normalisation, and flexible margin distributions
- **Standalone plot functions** usable outside the experiment context

---

## Installation

### Prerequisites

Python ≥ 3.9.  Install all required dependencies with:

```bash
pip install .
```

or, for development (editable install + test runner):

```bash
pip install -e ".[dev]"
```

### Video capture features

The `ring_gui()` method in `TrackingVideo` requires `camera.py` from the
[magnocube/holocube](https://github.com/jpcurrea/magnocube/blob/main/holocube/camera.py)
sub-package.  This is only needed when running the live rig; offline analysis
does not require it.  To use it, clone the magnocube repository at the same
level as your working directory:

```
magnocube/
  holocube/
    camera.py      ← referenced by sys.path at runtime
magno_tracker/     ← this package
```

---

## Quick start

```python
from magno_tracker import TrackingTrial, TrackingExperiment

# --- load a single trial ---
trial = TrackingTrial("path/to/file.h5")
heading = trial.query(output="body_angle")          # full array
saccades = trial.query(output="amplitude",          # per-test mean amplitudes
                       object="saccade",
                       groupby="test")

# --- load an experiment (multiple trials) ---
import glob
files = sorted(glob.glob("h5_files/*.h5"))
exp = TrackingExperiment(files)
exp.detect_saccades()

df = exp.saccade_table_df()                         # pandas DataFrame

# --- unified plot entry point ---
fig = exp.plot(
    xvar="test_ind",
    yvar="body_angle",
    col_var="condition",
    plot_type="line",
    ci=True,
    margin="histogram",
)
```

---

## Core classes

### `TrackingTrial`

Wraps a single `.h5` recording file.

| Method | Description |
|---|---|
| `query(output, object, subset, groupby, agg_func)` | Flexible data extraction with optional subsetting and aggregation |
| `detect_saccades(**kwargs)` | Run saccade detection; stores result in `saccade_table` |
| `saccade_table_df(extra_cols)` | Export saccade table to a pandas `DataFrame` |
| `add_dataset(name, array)` | Add a derived dataset (lazy; persisted on `save()`) |
| `add_attr(key, value)` | Add or update a trial-level attribute |
| `save()` | Write pending changes to the Zarr store |

**`query()` `groupby` options**

| `groupby=` | Returns |
|---|---|
| `'saccade'` (default) | flat array — one value per saccade |
| `'test'` | one value per test via `agg_func` |
| `'trial'` | one value per trial via `agg_func` |

### `TrackingExperiment`

Aggregates a list of `TrackingTrial` instances.  Mirrors the `TrackingTrial`
API at the experiment level (results concatenated across trials).

### `Saccade`

On-demand reconstructed saccade object.  Key attributes: `arr_relative`,
`velocity`, `relative_time`, `amplitude`, `peak_velocity`, `duration`,
`start_angle`, `stop_angle`.

---

## Plotting

### `TrackingExperiment.plot()`

```python
exp.plot(
    xvar,              # variable on x-axis (or time axis for 'line')
    yvar,              # variable on y-axis
    col_var=None,      # column-facet variable
    row_var=None,      # row-facet variable
    plot_type='line',  # 'line' | 'histogram' | 'hist2d' | 'scatter' | 'trajectory2d'
    object='trial',    # 'trial' | 'saccade'
    row_cmap=None,     # colormap for row levels
    col_cmap=None,     # colormap for column levels
    color='k',         # fallback color
    margin=None,       # margin distribution type: 'histogram' | 'pdf' | 'trajectory2d'
    probability=False, # show percentages instead of counts in margins
    groupby=None,      # trajectory averaging: a TrackingTrial attribute name (e.g. 'fly_id')
    margin_groupby=None, # per-group margin distributions
    split_by_sign=False, # separate positive / negative amplitude saccades
    show_n=False,      # annotate each panel with N=trials, n=saccades
    **plot_kwargs,     # forwarded to the underlying plot function
)
```

### Standalone plot functions

All functions are importable from the top level and accept an `ax` as their
first argument, making them easy to use in custom figure layouts.

| Function | Description |
|---|---|
| `plot_line(ax, xs, ys, color, ...)` | Individual traces + mean ± CI |
| `plot_histogram(ax, xs, bins, color, ...)` | 1-D bar histogram |
| `plot_hist2d(ax, xs, ys, color, ...)` | 2-D histogram with optional KDE contours |
| `plot_scatter(ax, xs, ys, color, ...)` | Scatter with optional jitter and circular-linear correlation |
| `plot_trajectory2d(ax, xs, ys, color, ...)` | Cumulative 2-D trajectories + endpoint ring histogram |
| `plot_pdf(ax, xs, bins, color, ...)` | Smoothed per-group PDF with bootstrap CI |

---

## Saccade detection

```python
trial.detect_saccades(
    threshold_speed=350,   # deg/s velocity threshold
    distance=5,            # minimum frames between saccades (find_peaks)
    prominence=100,        # find_peaks prominence
)

# Retrieve amplitudes for each saccade
amplitudes = trial.query(output="amplitude", object="saccade")

# Filter by speed
fast = trial.query(
    output="amplitude",
    object="saccade",
    subset={"peak_velocity": ">500"},
)

# Export to DataFrame
df = trial.saccade_table_df()
```

---

## Tutorial notebooks

Both notebooks use only the `fh_baja_1_new` trial files and run saccade
detection live, so they work without any pre-processed data.

| Notebook | Description |
|---|---|
| [notebooks/tutorial_query.ipynb](notebooks/tutorial_query.ipynb) | Loading data, basic and experiment-level queries, saccade detection, saving and reloading |
| [notebooks/tutorial_plot.ipynb](notebooks/tutorial_plot.ipynb) | All five plot types, saccade traces, saccade dynamics, main sequence analysis, margin customisation |

---

## Running the tests

From inside the `magno_tracker/` directory:

```bash
pytest                  # all 269 tests
pytest tests/test_phase4_saccades.py   # one module
```

---

## Dependencies

Core: `numpy<2`, `scipy`, `matplotlib`, `pandas`, `xarray`, `zarr`, `h5py`,
`h5netcdf`, `seaborn`, `scikit-image`.

Optional video: `scikit-video` (install with `pip install ".[video]"`).
Live camera capture additionally requires `camera.py` from the
[magnocube/holocube](https://github.com/jpcurrea/magnocube/blob/main/holocube/camera.py) repository.

---

## License

MIT — see [LICENSE](LICENSE).
