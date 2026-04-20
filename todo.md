# NOTE: When working through this todo list, changes should be made step-by-step. Discuss each step and check for conceptual errors before making code changes. This workflow should be followed in future sessions with an LLM to ensure correctness and maintainability.

# CONSTRAINT: Lazy loading (xr.open_dataset) is a hard requirement for most of the library. Do NOT switch to eager loading (xr.load_dataset) as a solution to file-handle conflicts or save/overwrite bugs — lazy loading must be preserved. Any fixes to file-handle issues must work within the lazy-loading model (e.g. explicitly closing handles before writing, writing to a temp path then renaming, etc.).

# tracking.py Refactor & Feature Additions Todo

This document tracks planned changes and improvements for `magno_tracker/tracking.py`,
focusing on plotting modularization, robust discontinuity handling, and hierarchical querying.

---

## Phase 1 — Extract shared helper functions (COMPLETE)

All major blocks of color, grid, wrapping, and bootstrap logic have been extracted from `plot_summary`, `plot_histogram_summary`, `plot_saccades`, `plot_saccade_dynamics`, and `main_sequence_analysis` into standalone, testable helpers:

- [x] **1.1 `resolve_colors(row_cmap, col_cmap, row_vals, col_vals, default_color='k')`**
      → returns `color_arr` (num_rows × num_cols × 3).
      Consolidates the ~30-line cmap-resolution block repeated 5×.
      Also fixes the `isinstance(row_cmap, ...)` bug in `plot_histogram_summary` (L3123).

- [x] **1.2 `get_grid_vals(experiment, var, subset)`**
      → returns filtered, NaN/`b'nan'`-free unique values for a row or column variable.
      Replaces the ~15-line query-unique-filter-NaN block repeated 4×. Now includes flattening to match legacy behavior.

- [x] **1.3 `omit_wrapping(arr, threshold=π/2)`**
      → returns a copy with NaNs inserted at circular discontinuities.
      Replaces the diff→threshold→NaN-insertion pattern used ≥5× inside `plot_summary` alone
      (individual traces, mean trace, row summary, col summary, hist2d overlay).

- [x] **1.4 `bootstrap_ci(data, stat_func, confidence=0.84, n_boot=1000, axis=0)`**
      → returns `(low, high)` arrays.
      Consolidates the bootstrap blocks in `plot_summary` (row margin, col margin)
      and `main_sequence_analysis`.

- [x] **1.5 Write unit tests for all Phase 1 helpers** using small synthetic arrays.

### Implementation summary (March 2026)

- All helper functions were implemented as standalone utilities at the top of `tracking.py`.
- Comprehensive unit tests were written and pass for all helpers.
- All major plotting and analysis functions now use these helpers instead of manual/duplicated logic.
- The codebase is now more maintainable, testable, and consistent, with no remaining manual color, grid, wrapping, or bootstrap logic in the main plotting functions.

**Phase 1 is complete and validated on both synthetic and real data.**

---

## Phase 2 — Migrate to xarray for labeled multidimensional data

Goal: Refactor data handling to use xarray for robust, labeled, multidimensional arrays, improving query/filter logic and reducing shape mismatch bugs.

- [x] **2.1 Prototype xarray Dataset creation**
      - Load HDF5 datasets into xarray, assigning dimension names (e.g., 'test', 'frame') and coordinates.
      - Validate that all variables (e.g., time, body_angle) align on dimensions.
      - HDF5 → Zarr conversion on first load; subsequent loads read from Zarr lazily (`xr.open_dataset(..., engine='zarr')`).
      - `save()` writes to a temp Zarr store then swaps atomically, avoiding h5py file-lock conflicts.
      - Round-trip consistency validated by integration tests (all 9 passing as of April 2026).

- [x] **2.1b Fix `add_dataset` / `add_attr` / `remove_dataset` for zarr**
      - All subtasks complete. Saving is now opt-in (`save()` must be called explicitly).
      - Sensitive variable warnings implemented for `is_test` and `camera_heading_offline`.
      - Derived dataset names standardised to snake_case.
      - Round-trip tests added and passing (26/26 as of April 2026).

- [x] **2.2 Refactor query logic to use xarray**
      - `query()` fully rewritten: 1-D subset vars drop tests via `.isel()`,
        2-D subset vars NaN-fill frames via `.where()`.
      - Membership (`[1, 3]`), inequality strings (`'>0.5'`), NaN matching, and
        AND-conditions all handled correctly.
      - 35 unit + integration tests passing (April 2026), results validated
        empirically against real experimental data.

- [x] **2.3 Update downstream analysis and plotting**
      - `plot_summary` and other experiment-level plotting functions required no
        changes — the new `query()` return type and shape are identical.
      - `butterworth_filter` and `remove_saccades` updated to use `add_dataset()`.

- [x] **2.4 Document migration and add tests**
      - Migration notes and usage examples added to docstrings for `load()`,
        `save()`, `add_dataset()`, `remove_dataset()`, `add_attr()`, and `query()`.
      - 52 tests passing across `TestTrackingTrialFromH5`, `TestTrackingTrialFromZarr`,
        `TestQuery`, `TestAddDataset`, `TestRemoveDataset`, `TestAddAttr`,
        `TestAddDatasetRoundTrip`, `TestQueryIntegration`, `TestButterworthIntegration`,
        and `TestRemoveSaccadesIntegration`.

### Implementation summary (April 2026)

**Phase 2 is complete.** The data layer now uses xarray + Zarr throughout:
lazy loading is preserved, file-lock issues are eliminated, all derived datasets
are stored via `add_dataset()` / `save()`, and the full query/filter API is
backed by xarray dimension-aware operations.

---

## Phase 3 — Standalone plot-type functions

Stateless functions: `(ax, xvals, yvals, color, **kwargs) → artist(s)`.
Easy to test with synthetic data and reusable outside `TrackingExperiment`.

- [x] **3.0 Fix `resolve_colors` / `color` argument handling in plotting functions**
      - One cmap specified → broadcast that cmap across the other axis (no blending).
      - Both cmaps → geometric mean blend: `sqrt(0.5*(r²+c²))`.
      - Neither → fill with `default_color`.
      - Added string cmap support (e.g. `'viridis'`) via `ScalarMappable`.
      - Added `color='k'` parameter to `plot_saccades`, `plot_saccade_dynamics`,
        `plot_histogram_summary`; all call `resolve_colors(..., default_color=color)`.
      - Replaced `plot_histogram_summary`'s ~60-line manual color block with a
        single `resolve_colors()` call.
      - Unit tests: single-color, row-only broadcast, col-only broadcast, both-cmaps.

- [x] **3.1 `plot_line(ax, xs, ys, color, summary_func=None, ci=False, confidence=0.84, n_boot=1000, trace_color='gray', **kw)`**
      Individual traces in `trace_color` + white-backed colored mean + optional bootstrap CI band.
      `alpha`/`lw` via `**kw`.

- [x] **3.2 `plot_hist2d(ax, xs, ys, color, bins=100, density=False, **kw)`**
      2D histogram with white→color colormap. `vmax`/`cbar` extracted from `**kw`.

- [x] **3.3 `plot_trajectory2d(ax, xs, ys, color, trace_color='k', **kw)`**
      Cumulative heading → 2D paths + endpoint scatter. Forced square aspect ratio
      (`ax.set_aspect('equal', adjustable='box')`). Optional overlays via `**kw`:
      - `circle` — per-fly radius circles.
      - `contour` — bootstrap ellipse at `confidence` CI.
      - `circ_hist` — Wedge ring histogram of endpoint angles at r=1.01–1.26,
        normalized within the panel, with bootstrap CI arc + mean-angle dot.
        Bug fix vs. original: ring Wedge tiles were computed but never drawn —
        now drawn immediately after normalization. Axis limits auto-expand to ±1.27.
      - `mean_line` — mean trajectory overlay.
      - `bins` — bin count/edges for `circ_hist` (default 100).

- [x] **3.4 `plot_histogram(ax, xs, bins, color, probability=False, summary_func=None, **kw)`**
      1D bar histogram; NaNs removed silently; optional `axvline` at `summary_func(valid)`.

- [x] **3.5 `plot_scatter(ax, xs, ys, color, jitter_std=0.0, correlation=False, n_boot=10000, marker_color=None, **kw)`**
      Scatter with optional jitter. `marker_color` overrides `color` for data points
      (reserved for future summary overlay). Optional Mardia's circular-linear
      correlation annotation with bootstrap p-value. `alpha`/`s`/`edgecolors` via `**kw`.

- [x] **3.6 Write unit tests for all Phase 3 functions** — 29 tests in
      `tests/test_phase3_plot_functions.py`. PNG outputs saved to `tests/plot_outputs/`
      for visual inspection. Covers: basic output, trace/marker color params, CI,
      `circ_hist` Wedge count, CI arc, axis limits, aspect ratio, `contour` ellipse,
      `circle` count, jitter, probability histogram, correlation annotation.

### Implementation summary (April 2026)

**Phase 3 is complete.** All five standalone functions are implemented and tested.

Key design decisions:
- All functions are module-level (not methods) and accept `ax` as first argument.
- `**kw` absorbs function-specific options so the Phase 5 `plot()` dispatch can
  forward `plot_kwargs` without needing to know each option by name.
- Cross-subplot normalization (e.g. for `circ_hist` ring tiles) is intentionally
  left to the caller: Phase 5's `plot()` should query all data first, compute a
  global `vmax`, then pass it explicitly to each `plot_trajectory2d` call.

---

## Phase 4 — Replace `Bout` intermediary with a flat saccade table per trial

### Motivation

`Bout` is essentially a container: it slices one test's data from a trial, runs
saccade detection, and holds a list of `Saccade` objects. With xarray, per-test
slicing is already handled by `query(subset={'test_ind': X})`. The class adds
ceremony without meaningful logic, and the trial→bout→saccade traversal in
`plot_saccades`/`plot_saccade_dynamics` is the main source of complexity in
Phase 4's query system.

The replacement: a **saccade table** stored as an xarray Dataset on the trial,
one row per detected saccade, with enough columns to (a) quickly filter and plot
without rebuilding `Saccade` objects and (b) reconstruct a full `Saccade`
on demand.

### Saccade table schema

| Column | dtype | Description |
|---|---|---|
| `test_ind` | int | Which test this saccade belongs to |
| `test_start_frame` | int | Frame index in the full trial array where that test begins |
| `start_frame` | int | Saccade start, **test-relative** (matches existing detection algorithm) |
| `stop_frame` | int | Saccade stop, test-relative |
| `amplitude` | float | `stop_angle − start_angle` in radians |
| `peak_velocity` | float | Signed peak velocity in rad/s |
| `duration` | float | Duration in seconds |
| `start_angle` | float | Heading at saccade start (radians) |
| `stop_angle` | float | Heading at saccade stop (radians) |

Trial-relative frame index → `test_start_frame + start_frame`. This preserves
backward compatibility with the detection algorithm (which works on test-relative
arrays) while still enabling fast trial-level reconstruction.

### `Saccade` constructor change

From `Saccade(arr, bout, framerate, start, stop)` → `Saccade(arr, trial, test_ind, framerate, start, stop)`.
`self.bout` references (used only for attribute lookups in `query_saccades`)
are replaced by `self.trial` + `self.test_ind`.

### Lifecycle of the saccade table

- Built eagerly when `detect_saccades()` is called on the trial.
- Replaces any existing table if `detect_saccades()` is called again.
- Stored in memory as a plain dict/xarray Dataset; not persisted until `save()` is called
  (consistent with the `add_dataset()` / `save()` pattern from Phase 2).
- Loaded lazily from Zarr on subsequent trial loads.

### New query interface

`TrackingTrial.query(object='saccade', output, subset={}, groupby='saccade', agg_func=np.nanmean)`

Subsetting uses the existing `query()` logic against the saccade table:
- **Trial-level keys** (`bg_gain`, `condition`, etc.) filter via the trial's xarray data.
- **Saccade-level keys** (`peak_velocity`, `amplitude`, etc.) filter rows of the table directly.

`groupby` controls output granularity:

| `groupby=` | Returns | Use case |
|---|---|---|
| `'saccade'` | flat array / list of `Saccade` objects | histograms, scatter of individual events |
| `'test'` | one value per test via `agg_func` | per-condition traces |
| `'trial'` | one value per trial via `agg_func` | subject-level stats |

`agg_func` accepts any callable and is applied as `agg_func(group_values)` for each group.
Pass `agg_func=list` to collect raw values per group without reducing (returns one list per
group, ragged across groups if saccade counts differ). Default is `np.nanmean`.

| `agg_func=` | `groupby='test'` result |
|---|---|
| `np.nanmean` | one scalar per test |
| `list` | one list of raw values per test (ragged) |
| `np.stack` | one 2D array per test (if all groups have equal length) |

`output` can be:
- A **scalar column name** (`'amplitude'`, `'peak_velocity'`, etc.) → returns ndarray.
- `'saccade'` → returns list of on-demand `Saccade` objects (reconstructed from
  `test_start_frame + start_frame` and the trial's heading array).
- A **time-series attribute** (`'arr_relative'`, `'velocity'`) → returns list of
  variable-length arrays (only meaningful with `groupby='saccade'`).

### Two-phase Saccade construction

`Saccade.__init__` currently does two things: (1) **refine** `start`/`stop` using the
baseline velocity distribution and validate via the noise threshold (`baseline_comparison`,
`baseline_test`), and (2) **compute and store** derived attributes (`amplitude`,
`peak_velocity`, `duration`, etc.).

We exploit this split deliberately:

- **Detection phase** (`detect_saccades()`): instantiate `Saccade` with
  `baseline_comparison=True, baseline_test=True`. If `saccade.success`, read the
  refined `start`, `stop`, and derived scalars into the saccade table. Discard the
  `Saccade` object — its job is done.
- **Reconstruction phase** (`output='saccade'`): instantiate `Saccade` with
  `baseline_comparison=False, baseline_test=False`. The `start`/`stop` stored in
  the table are already the fully refined values; `interpolate_velocity=True` still
  runs so that time-series attributes (`arr_relative`, `velocity`, `relative_time`,
  etc.) are available for plotting. `trial` and `test_ind` can be `None` for
  reconstruction — the caller already has the sliced `arr`.

### `_detect_saccades` simplification

Only `speed_noise_method=True` is kept. The new helper strips:
- The entire commented-out Kalman-filter block inside `speed_noise_method`.
- The large commented-out rolling-std / cross-boundary dead code.
- The `kalman_method` and `acceleration_method` branches entirely.
- The inline `display` / `breakpoint()` block (remove, not port).

Parameters that survive: `threshold_speed`, and the `find_peaks` overrides
(`distance`, `width`, `prominence`, `wlen`).
Parameters dropped: `speed_noise_method` flag (always True now),
`kalman_method`, `acceleration_method`, `de_lag`,
`maximum_saccade_frequency`, `relative_start_velo`.

### Tasks

- [x] **4.1 Update `Saccade.__init__`** to accept `(arr, trial, test_ind, framerate,
      start, stop, ...)` instead of `(arr, bout, ...)`. Replace `self.bout`
      attribute with `self.trial` and `self.test_ind`. `trial` and `test_ind` may
      be `None` when reconstructing on-demand from stored start/stop values.

- [x] **4.2 Add `detect_saccades()` to `TrackingTrial`**
      Iterates over tests, calls `_detect_saccades()` for each test's heading array,
      instantiates `Saccade` with full baseline refinement, harvests scalar columns
      from successful saccades into the saccade table, then discards `Saccade`
      objects. Stores table as `self.saccade_table`. Does not call `save()`.

- [x] **4.3 Persist / load saccade table via the existing Zarr machinery**
      - `save()` writes the saccade table to the Zarr store alongside other datasets.
      - `load()` / `open_dataset()` loads it lazily on the next open.
      - Use `add_dataset()` internally, consistent with Phase 2 conventions.

- [x] **4.4 Implement `TrackingTrial.query(object='saccade', ...)`**
      Uses the saccade table for fast scalar queries. For `output='saccade'`,
      reconstructs `Saccade` on demand with `baseline_comparison=False,
      baseline_test=False` using the stored `start`/`stop` into
      `self.data['body_angle']`.

- [x] **4.5 Implement `TrackingExperiment.query(object='saccade', ...)`**
      Calls each trial's `query(object='saccade', ...)` and concatenates results:
      - `groupby='saccade'` → concatenate flat arrays.
      - `groupby='test'` → concatenate per-test arrays.
      - `groupby='trial'` → one value per trial.

- [x] **4.6 Deprecate `Bout` class**
      - Keep `Bout` in the file but mark it deprecated with a warning on instantiation.
      - Extract the `speed_noise_method` core of `process_saccades()` into a
        module-level helper `_detect_saccades(arr, framerate, threshold_speed=350,
        **find_peaks_kwargs) → list[dict]`. Remove all dead/commented-out code and
        the `kalman_method` / `acceleration_method` branches. Each dict in the
        returned list contains `start`, `stop` (raw, pre-refinement frame indices).

- [x] **4.7 Write integration tests** confirming:
      - `detect_saccades()` populates `saccade_table` with the correct columns.
      - `save()` + reload produces identical saccade table.
      - `query(object='saccade', output='amplitude')` returns a flat ndarray.
      - `query(object='saccade', output='amplitude', groupby='test')` returns
        one value per test.
      - `query(object='saccade', subset={'peak_velocity': '>5'})` correctly
        filters individual saccades.
      - On-demand `Saccade` reconstruction (no baseline re-test) matches scalar
        attributes stored in the table.

### Implementation summary (April 2026)

**Phase 4 is complete.** The saccade detection and querying layer is now fully
independent of `Bout`:

- `_detect_saccades(arr, framerate, ...)` — standalone module-level helper,
  `speed_noise_method` only, all dead code stripped. Parameters reduced to
  `threshold_speed` + `find_peaks` kwargs.
- `Saccade.__init__` signature updated to `(arr, trial, test_ind, framerate,
  start, stop, ...)`. `trial`/`test_ind` may be `None` for on-demand
  reconstruction.
- `TrackingTrial.detect_saccades()` — two-phase: full baseline refinement
  during detection, scalars (including `peak_frame`) harvested into
  `self.saccade_table`, `Saccade` objects discarded.
- Saccade table persisted as columnar `_saccade_*` Zarr variables; the `load()`
  dim-rename guard and robust `dims[0]` lookup ensure safe round-trips even
  when saccade count coincidentally equals `num_tests`.
- `TrackingTrial.query(object='saccade', ...)` — fast scalar queries with
  `groupby='saccade'/'test'/'trial'`, `agg_func=list` support, on-demand
  `Saccade` reconstruction.
- `TrackingExperiment.query(object='saccade', ...)` — flattening concatenation
  across trials.
- `TrackingTrial.saccade_table_df(extra_cols, time_ind)` and
  `TrackingExperiment.saccade_table_df(extra_cols, time_ind)` — export to
  pandas DataFrame; scalar/1-D test-level/2-D (test × frame) extra columns
  all supported; experiment-level df always includes `trial_filename`.
- `Bout` marked deprecated with `DeprecationWarning` on instantiation.
- `load_datasets()` defaults to `framerate=60` when neither `framerate` nor
  `duration` is stored in the file's attrs.
- 27 integration tests, all passing.

**Additional fixes and optimisations (April 2026):**
- **Saccade table persistence** moved from columnar `_saccade_*` Zarr variables to a
  `.saccades.pkl` sidecar file; saccade-only saves now complete in <1 s.
- **`save()` made surgical**: granular `_pending_vars` / `_removed_vars` /
  `_dirty_attrs` tracking so only changed arrays/attrs open the Zarr store;
  threading reverted to sequential (zarr v3 async loops made threads slower).
- **`Saccade` object cache**: `self.saccades` list + `self._saccade_id_to_index`
  dict built on the first `query(object='saccade', output='saccade')` call and
  invalidated when `detect_saccades()` replaces the saccade table. Avoids
  reconstructing `Saccade` objects on every repeated query.
- **`plot_saccades` refactored**: removed `Bout`-based `trial.query(output='bouts')`
  + `bout.query_saccades()` traversal; replaced with a direct
  `trial.query(output='saccade', object='saccade', subset=subset)` call.
  Trials without a saccade table (raising `RuntimeError`) are silently skipped.
- **Bug fix in `detect_saccades` / `saccade_table_df`**: `self._saccade_id_to_index`
  invalidation line was accidentally fused with the `def saccade_table_df` line —
  corrected.


## Phase 5 — Unified `TrackingExperiment.plot()` method

A single entry point that handles grid setup, querying, iteration over row/col
subsets, and margin summaries — delegating actual drawing to Phase 3 functions.
Phase 3 functions are already implemented and unit-tested; Phase 5 wires them
into the experiment-level grid infrastructure.

- [x] **5.1 Implement `TrackingExperiment.plot()`**
      Signature: `plot(xvar, yvar, col_var, row_var, plot_type='line',
      object='trial', row_cmap=None, col_cmap=None, color='k', ...)`
      Internally:
        1. `get_grid_vals()` (Phase 1.2) for row/col values
        2. `resolve_colors()` (Phase 1.1) for the color array
        3. Create `SummaryDisplay` grid
        4. **Pre-query pass**: loop over row × col, call `self.query()` for every
           cell and cache results. This allows computing global normalization
           values (e.g. `circ_hist` Wedge `vmax`, `hist2d` color scale) before
           any drawing happens, so all panels share the same color range.
        5. **Draw pass**: loop over cached (data, color) pairs, dispatch to the
           appropriate Phase 3 function based on `plot_type`, passing the global
           `vmax` (and any other cross-subplot stats) via `**plot_kwargs`.
        6. Populate margin summaries (`_MARGIN_DEFAULTS`, `_resolve_margin()`,
           `_draw_margin_cell()` module-level helpers)
        7. Call `display.format()` and `display.label_margins()`
      Bug fixes (April 2026):
        - Histogram margin ylabel/xlabel correctly labelled as 'count'/'probability'
          instead of inheriting the main plot's variable name.
        - `_draw_margin_cell` histogram uses `ys` data for right margin and `xs`
          for bottom margin; right margin drawn with counts on x-axis.

- [x] **5.2 Test `plot()` with `plot_type='line'`**
- [x] **5.3 Test `plot()` with `plot_type='hist2d'`**
- [x] **5.4 Test `plot()` with `plot_type='histogram'`**
- [x] **5.5 Test `plot()` with `plot_type='trajectory2d'`**
- [x] **5.6 Test `plot()` with `object='saccade', plot_type='line'`**
- [x] **5.7 Test `plot()` with `object='saccade', plot_type='scatter'` or `'hist2d'`**
- [x] **5.8 Test margin axes** — `TestMargins` class in `test_phase5_plot.py`:
      correct labels, orientation, and artist presence for histogram/line margin
      types across plot_type='line', 'hist2d', 'scatter', 'trajectory2d'.

### Implementation summary (April 2026)

**Phase 5 is complete.** `TrackingExperiment.plot()` is implemented and 50+ tests pass.
Key design notes:
- Module-level helpers `_MARGIN_DEFAULTS`, `_resolve_margin()`, `_draw_margin_cell()`.
- `plot_trajectory2d` extended with `return_summary=True` for circ_hist margin passthrough.
- `_pending_vars`, `_removed_vars`, `_dirty_attrs` are now lazily initialised in
  `load_datasets()` to support synthetic trials built via `__new__`.
- `detect_saccades()` now stores `self.saccade_heading_variable = key`; the saccade
  reconstruction path in `query()` uses this instead of the hardcoded `'camera_heading'`.

**Additional features and fixes (April 2026 — session 2):**
- **`saccade_spans`**: `plot_line` accepts `saccade_spans` (list of `(y_start, y_stop)`)
  as a ref-frame-aware replacement for `saccade_durations`. Both params still supported.
- **`relative_to` param**: saccade line traces can be aligned to `'start'` (default),
  `'peak'`, or `'stop'`. All traces are padded with `max_before − ref_idx` offset so
  the reference frame sits at the same column for every trace → `nanmean` passes through
  (0, 0).
- **Saccade line defaults**: `xlim=(−π, π)`, `ylim=(0.5, −0.25)` auto-applied when
  `object='saccade', plot_type='line'`.
- **`mean_bins` binning within ylim**: `plot_line` accepts `ylim` param; `_bin_mean`
  uses the display range instead of the full data extent, so `mean_bins=10` covers the
  visible window. Last-bin precision fixed (`<=` on upper edge). `plot()` computes
  `_effective_bin_ylim` before the draw loop (falls back to saccade defaults when applicable).
- **`split_by_sign`**: `plot_line` draws separate solid/dotted mean lines for
  positive/negative amplitude groups.
- **`show_n` annotation fontsize**: replaced hardcoded `fontsize=6` with
  `plt.rcParams['xtick.labelsize']` so sample-size text matches tick labels.
- **`show_n` param**: added to `plot()` signature; draws `N=<int>, n=<int>` annotation
  in each panel.
- **`mtype='trajectory2d'` for margins**: `_draw_margin_cell` priority chain —
  (1) circ_hist → concentric CI arcs stacked by `overlay_index`;
  (2) circle → mean radius circle;
  (3) contour → confidence ellipse at `1/n_overlays` opacity;
  (4) mean_line → mean 2-D trajectory;
  (5) fallback → endpoint scatter.
  Axes flagged `_traj2d_circ_hist = True` are excluded from the post-loop limit sync.
- **Mean circle in `plot_trajectory2d`**: `circle=True` now also draws a thick mean
  circle on top of the per-fly circles; `mean_radius` stored in `summary_dict`.
- **Contour ellipse params**: `contour=True` stores `ellipse_mean`, `ellipse_a`,
  `ellipse_b`, `ellipse_angle` in `summary_dict` for margin reuse.
- **Margin axis limit sync**: post-loop block syncs non-trajectory2d margin axes to the
  union of main-panel limits; same-type margins sync both axes, different-type margins
  auto-scale the independent axis.
- **`right_margin_xlim` / `right_margin_ylim` / `bottom_margin_xlim` / `bottom_margin_ylim`**
  params added to `plot()` to let the caller override any auto-synced margin limit.
- **`_get_xs_ys` row-alignment bug fixed**: `xs_row_mask` saved before filtering and
  applied to `ys` as well, so `xs[i]` and `ys[i]` always correspond to the same trace
  even when matching-condition trials are not the first in the list. (Previously rows
  with `condition != 0` produced invisible traces because `ys[:n]` grabbed NaN-padded
  rows.)
- **`plot_histogram` mean line**: white underlay (`lw=6, zorder=4`) + colored line
  (`lw=2, zorder=5`) for visibility; both integer lw to avoid sub-pixel misalignment
  at non-native DPI.
- **KDE contour margins for `hist2d`**: `n_contours` / `contour_alpha` in `plot_kwargs`
  passed through to `plot_hist2d`; margin drawing uses the stored `xs`/`ys` for KDE.

---

## Phase 6 — Deprecate old methods and clean up

- [x] **6.1 Rewrite `plot_summary` as a thin wrapper** around `plot()` (for backward compat), mark deprecated.
      Note: `color`, `row_cmap`, `col_cmap` already forward correctly via Phase 3.0
      changes — the wrapper mainly needs to translate old `plot_kwargs` keys.

- [x] **6.2 Rewrite `plot_histogram_summary`** as `plot(..., plot_type='histogram')`, mark deprecated.
      Note: `color='k'` param and `resolve_colors` replacement already done in Phase 3.0.

- [x] **6.3 Rewrite `plot_saccades`** as `plot(..., object='saccade', plot_type='line')`, mark deprecated.
      Note: `color='k'` param already added in Phase 3.0.

- [x] **6.4 Rewrite `plot_saccade_dynamics`** as `plot(..., object='saccade', plot_type='scatter')`, mark deprecated.
      Note: `color='k'` param already added in Phase 3.0.

- [x] **6.5 Remove dead code:** `breakpoint()` calls, large commented-out blocks throughout all methods.

- [x] **6.6 Add/update docstrings** for all new and modified public functions and methods.

### Implementation summary (April 2026)

**6.1–6.4 complete.** Each deprecated method now:
- Emits `DeprecationWarning` pointing callers to `plot()`.
- Warns on unsupported params (`fig`, `reversal_split`, `reference_var`, `start`/`stop`).
- Translates old param names (`use_density`→`density`, `use_probability`→`probability`,
  `saccade_var`/`time_var`/`line_color`/`line_alpha`/`mean_bins`→`plot_kwargs`,
  `output`→`xvar`, `scatter`→`plot_type`).
- `plot_saccade_dynamics` sets `rad2deg=True` so the degree-conversion previously
  done inline is now handled by `plot()`.
- All 84 existing tests still pass.

---

## Notes

- `main_sequence_analysis` has a unique layout (3×2 with shared axes, log scales,
  marginal jitter+CI strips) that doesn't fit the grid pattern. Leave it as-is
  for now but have it use Phase 1 helpers (`resolve_colors`, `bootstrap_ci`).
  **Refactored (April 2026):** now uses `get_grid_vals`, `resolve_colors`,
  `bootstrap_ci`; bugs fixed (mutable default, mutated outer variable, `exec()`
  anti-pattern, missing `same_size=False`). `group_var='filename'` supported,
  full-path filenames auto-shortened to basename in tick labels. Axis padding
  fixed: right-margin x and bottom-margin y both use `[-0.5, N−0.5]` symmetric
  range. Strip scatter uses `color=` kwarg (not `c=`) to avoid RGB-as-colormap
  confusion. Legend only shown when labeled artists exist. `set_visible(False)`
  used instead of `axis('off')` for the hidden bottom-right cell.
  15 integration tests in `TestMainSequenceAnalysis` (all passing).
- The `SummaryDisplay` class and its `format()`/`label_margins()` methods are
  already solid. No changes needed there.
- Each phase is independently mergeable: Phase 1 is pure extraction with no
  behavior change, Phase 2 migrates the data layer to xarray/Zarr, Phase 3
  implements standalone plot functions, Phase 4 fixes the saccade query system,
  Phase 5 wires everything into a unified `plot()` method, Phase 6 cleans up.

---

## Phase 7 — SummaryDisplay improvements

### Motivation

`SummaryDisplay` currently forces all subplots to be the same size and its
row/column tick labels are positioned via a single draw-event callback
(`_update_margin_labels`) that works but does not respond correctly to window
resize events.  Two usability improvements are requested:

1. **Custom per-subplot sizes** — allow the caller to specify relative widths
   and heights for each column/row via `width_ratios` / `height_ratios`.
   For example, a trajectory2d plot could use equal-size square panels while
   a `plot_type='line'` grid uses wider panels.
2. **Dynamic margin label positioning** — row/column tick labels (drawn by
   `label_margins`) should update smoothly when the figure window is resized
   so they never overlap axes content or fall outside the figure boundary.

### Tasks

- [ ] **7.1 Add `width_ratios` / `height_ratios` parameters to `SummaryDisplay.__init__`**
      - Accept `width_ratios: list[float] | None` and `height_ratios: list[float] | None`.
      - Forward to `plt.Figure.subplots()` via `gridspec_kw={'width_ratios': ...,
        'height_ratios': ...}`.
      - If margins are present, auto-append a ratio entry for the margin
        column/row (default 0.25 × largest ratio in the corresponding axis).
      - Validate that the provided list length matches `num_cols`/`num_rows`
        (excluding the margin column/row) and raise `ValueError` otherwise.

- [ ] **7.2 Expose `width_ratios` / `height_ratios` in `TrackingExperiment.plot()`**
      - Add `width_ratios=None, height_ratios=None` parameters forwarded to
        `SummaryDisplay(...)`.

- [ ] **7.3 Improve dynamic label positioning**

      **Root cause of current fragility:** `_update_margin_labels` positions
      the row/column spine and text using hard-coded font-size / figure-size
      fractions (e.g. `spine_x = left_bound - 0.63/fig_width`).  These
      fractions must be re-tuned whenever font size, DPI, or panel count
      changes and they are not updated on window resize because only
      `draw_event` is connected (not `resize_event`).

      **Design goal:** positions should be derived from the **live rendered
      bounding boxes** of the axis labels — exactly how matplotlib positions
      its own tick labels relative to the spine.  Concretely:

      - **Row spine x** = left edge of the tightbbox of the leftmost column's
        axes (`ax.get_tightbbox(renderer).x0`, converted to subfigure-normalized
        coords), shifted left by a small fixed gap (~2 display points).  This
        means the row spine sits just outside the y-axis tick labels regardless
        of tick label width.
      - **Row text x** = left edge of the subfigure (0.0 in normalized coords),
        with `subplots_adjust(left=…)` reserving exactly enough space.
      - **Column spine y** = bottom edge of the tightbbox of the bottom row's
        axes (`ax.get_tightbbox(renderer).y0`, converted to normalized coords),
        shifted down by a small fixed gap.  Sits just outside the x-axis tick
        labels regardless of their height.
      - **Column text y** = bottom edge of subfigure (0.0), with
        `subplots_adjust(bottom=…)` reserving space.
      - **Tick positions** (already computed from ylabel/xlabel center bboxes)
        remain unchanged.

      **Changes required:**
      - Add `_axes_outer_edge_in_subfig_coords(axes_list, which)` helper that
        iterates `get_tightbbox(renderer)` over a list of axes and returns the
        requested edge (`'left'`, `'bottom'`, `'right'`, `'top'`) in
        subfigure-normalized coordinates.
      - Rewrite `_update_margin_labels` to use this helper for `spine_x` and
        `spine_y` instead of the font-size fraction formulas.
      - For `subplots_adjust`: derive `left_bound` from
        `min(tightbbox.x0) - text_width_estimate` so there is always room for
        the row label text without it falling out of the figure.  Similarly for
        `bottom_bound`.
      - Connect to both `draw_event` **and** `resize_event` in `label_margins()`
        (currently only `draw_event` is connected, so labels do not update on
        interactive resize).
      - Ensure callback re-entry guard (`self._updating`) still prevents
        feedback loops on the forced redraw that `subplots_adjust` triggers.
      - Remove the stale hard-coded formulas from both `label_margins` (initial
        placement) and `_update_margin_labels` (update placement) so they share
        a single positioning code-path.

- [ ] **7.4 Write tests for custom sizing**
      - Verify that axes in a 2-col display with `width_ratios=[2, 1]` have
        the correct relative widths (check `ax.get_position().width`).
      - Verify that the margin column width is unaffected by `width_ratios`
        and remains ~0.25 × the largest data column.

---

## Phase 8 — Tutorial notebooks and docstring improvements

### Motivation

The library has a solid, tested API but no user-facing documentation beyond
docstrings.  New collaborators (and LLMs) need worked examples on real data
to understand the query system, saccade extraction, and the full plotting API.
Simultaneously, many docstrings are terse or out-of-date and do not reflect
the Phase 2–6 API.

### Doctest strategy

**Selective doctests — not universal.**

*Good candidates for runnable doctests*: pure, side-effect-free utility
functions with simple deterministic outputs — `omit_wrapping`, `resolve_colors`,
`bootstrap_ci`, `get_grid_vals`.  A 3-line doctest here adds value for both
readers and `pytest --doctest-modules`.

*Bad candidates*: `TrackingTrial`/`TrackingExperiment` methods.  They require
real files on disk, produce matplotlib figures as side-effects, and their repr
is fragile across NumPy/Python versions.  Use an `Examples:` section with a
non-executable code block instead — looks like a tutorial, is not run by pytest.

*Rule of thumb*: if the function can be fully exercised with `np.array([…])` in
two lines, add a doctest.  If it needs fixture data or opens a file, use
`Examples:` prose only.

### Tasks

#### 8.1 Docstring improvements

- [ ] **8.1.1 Add `Examples:` sections to `TrackingTrial` and `TrackingExperiment`**
      Every public method that lacks one should get a minimal, realistic usage
      snippet (non-executable, under an `Examples:` heading).  Priority list:
      `load()`, `save()`, `add_dataset()`, `remove_dataset()`, `add_attr()`,
      `query()`, `detect_saccades()`, `saccade_table_df()`, `plot()`.

- [ ] **8.1.2 Add runnable doctests to Phase 1 helpers**
      `omit_wrapping`, `resolve_colors`, `bootstrap_ci`, `get_grid_vals` —
      each gets 1–2 `>>>` examples with simple `np.array` inputs.
      Verify with `pytest --doctest-modules magno_tracker/tracking.py`.

- [ ] **8.1.3 Update stale parameter descriptions**
      Several docstrings still describe old parameter names (`use_density`,
      `saccade_var`, `output_var`) or omit newer params (`rad2deg`,
      `relative_to`, `show_n`, `groupby`, `agg_func`).  Audit every public
      method and bring parameter lists up to date.

- [ ] **8.1.4 Document `plot()` `plot_kwargs` keys per `plot_type`**
      The `plot_kwargs` parameter accepts different keys depending on
      `plot_type` and `object`.  Add a table or sub-section in the `plot()`
      docstring listing recognised keys for each combination (e.g.
      `plot_type='trajectory2d'`: `circle`, `contour`, `circ_hist`,
      `mean_line`, `bins`; `plot_type='line'`: `mean_bins`, `split_by_sign`,
      `saccade_spans`, `trace_color`, `alpha`; etc.).

#### 8.2 Query tutorial notebook (`notebooks/tutorial_query.ipynb`)

Target reader: a new collaborator who has never used the library.

- [ ] **8.2.1 Loading data**
      - Load one or more real `.h5` files from `h5_files/` via `TrackingTrial`.
      - Show `trial.data` xarray Dataset structure, list available variables.
      - Demonstrate `add_dataset()` and `save()` on a derived variable.

- [ ] **8.2.2 Basic queries**
      - `query(output='camera_heading')` — full time series.
      - `query(output='camera_heading', subset={'condition': 1})` — subset by scalar.
      - `query(output='camera_heading', subset={'condition': [1, 3]})` —
        membership filter.
      - `query(output='camera_heading', subset={'bg_gain': '>0'})` — inequality string.
      - `query(same_size=True, ...)` — explain the padding behaviour.

- [ ] **8.2.3 Experiment-level queries**
      - Combine trials into a `TrackingExperiment`.
      - `exp.query(...)` across all subjects — show shape of returned array.
      - `groupby='trial'` vs `groupby='test'` vs default.

- [ ] **8.2.4 Saccade detection and extraction**
      - `trial.detect_saccades()` — show the resulting `saccade_table`.
      - `trial.query(object='saccade', output='amplitude')`.
      - `trial.query(object='saccade', output='amplitude', groupby='test')`.
      - `trial.query(object='saccade', subset={'peak_velocity': '>5'})`.
      - `trial.saccade_table_df()` — pandas export, useful for seaborn/statsmodels.
      - On-demand `Saccade` object reconstruction: show `s.arr_relative`,
        `s.velocity`, `s.peak_velocity`, `s.amplitude`.

- [ ] **8.2.5 Saving and reloading**
      - Show round-trip: `save()`, close, reload, verify saccade table intact.

#### 8.3 Plotting tutorial notebook (`notebooks/tutorial_plot.ipynb`)

Target reader: a collaborator who understands the data but is new to `plot()`.

- [ ] **8.3.1 Setup**
      - Load a multi-subject, multi-condition dataset into `TrackingExperiment`.
      - Run `detect_saccades()` on every trial (needed for saccade plot sections).

- [ ] **8.3.2 `plot_type='line'` — trial-level traces**
      - `exp.plot('camera_heading', 'time', col_var='condition', row_var=None)`.
      - Add `row_cmap` / `col_cmap` for color.
      - `xlim`, `ylim`, `xticks` formatting.
      - `right_margin=True`, `bottom_margin=True`.

- [ ] **8.3.3 `plot_type='histogram'`**
      - `exp.plot('amplitude', col_var='condition', row_var=None, plot_type='histogram')`.
      - `probability=True`, custom `bins`.

- [ ] **8.3.4 `plot_type='hist2d'` and `plot_type='scatter'`**
      - Show both on the same dataset for comparison.

- [ ] **8.3.5 `plot_type='trajectory2d'`**
      - `exp.plot('camera_heading', 'time', ..., plot_type='trajectory2d')`.
      - `plot_kwargs={'circle': True, 'contour': True, 'circ_hist': True,
        'mean_line': True}`.

- [ ] **8.3.6 Saccade traces — `object='saccade', plot_type='line'`**
      - `exp.plot('arr_relative', 'relative_time', object='saccade',
        plot_type='line', ...)`.
      - `relative_to='peak'` vs `'start'`.
      - `mean_bins`, `split_by_sign`, `show_n`.
      - `rad2deg=True` — show the degree-converted version side-by-side.

- [ ] **8.3.7 Saccade dynamics — `object='saccade', plot_type='hist2d'`**
      - Position vs amplitude 2D histogram.
      - Demonstrate `rad2deg=True` and `xlim=(-180, 180)`.

- [ ] **8.3.8 Margin customisation**
      - `right_margin_xlim`, `bottom_margin_ylim` overrides.
      - `right_margin='histogram'`, `bottom_margin='circ_hist'` explicit types.

#### 8.4 Real-data requirement

Both notebooks must run on the actual files in `h5_files/`.  At least one
subject must have saccades detected during the notebook run (not pre-cached)
so the saccade extraction workflow is demonstrated live.  Use a minimal
hardcoded subset (e.g. `fh_baja_1_new_trial_*.h5`) to keep runtime short.

---
Add new items as needed. Check off items as they are completed.
