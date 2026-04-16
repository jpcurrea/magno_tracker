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

- [ ] **5.1 Implement `TrackingExperiment.plot()`**
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
        6. Populate margin summaries
        7. Call `display.format()` and `display.label_margins()`

- [ ] **5.2 Test `plot()` with `plot_type='line'`**
      Integration test: call on real (or synthetic) experiment data; confirm
      the correct number of subplots, that traces appear on each axis, and
      that `resolve_colors` produces distinct colors per row/col.

- [ ] **5.3 Test `plot()` with `plot_type='hist2d'`**
      Confirm 2D histogram bins appear, shared `vmax` is consistent across panels.

- [ ] **5.4 Test `plot()` with `plot_type='histogram'`**
      Confirm bar histograms appear; verify it produces equivalent output to
      `plot_histogram_summary` on the same data.

- [ ] **5.5 Test `plot()` with `plot_type='trajectory2d'`**
      Confirm `circ_hist` Wedge count is consistent and all panels use the same
      normalized `vmax` (cross-subplot normalization from the pre-query pass).

- [ ] **5.6 Test `plot()` with `object='saccade', plot_type='line'`**
      Requires Phase 4 complete. Confirm it replaces `plot_saccades`.
      Data source switches from `Bout`-based traversal to saccade table query.

- [ ] **5.7 Test `plot()` with `object='saccade', plot_type='scatter'` or `'hist2d'`**
      Requires Phase 4 complete. Confirm it replaces `plot_saccade_dynamics`.
      Data source switches from `Bout`-based traversal to saccade table query.

---

## Phase 6 — Deprecate old methods and clean up

- [ ] **6.1 Rewrite `plot_summary` as a thin wrapper** around `plot()` (for backward compat), mark deprecated.
      Note: `color`, `row_cmap`, `col_cmap` already forward correctly via Phase 3.0
      changes — the wrapper mainly needs to translate old `plot_kwargs` keys.

- [ ] **6.2 Rewrite `plot_histogram_summary`** as `plot(..., plot_type='histogram')`, mark deprecated.
      Note: `color='k'` param and `resolve_colors` replacement already done in Phase 3.0.

- [ ] **6.3 Rewrite `plot_saccades`** as `plot(..., object='saccade', plot_type='line')`, mark deprecated.
      Note: `color='k'` param already added in Phase 3.0.

- [ ] **6.4 Rewrite `plot_saccade_dynamics`** as `plot(..., object='saccade', plot_type='scatter')`, mark deprecated.
      Note: `color='k'` param already added in Phase 3.0.

- [ ] **6.5 Remove dead code:** `breakpoint()` calls, large commented-out blocks throughout all methods.

- [ ] **6.6 Add/update docstrings** for all new and modified public functions and methods.

---

## Notes

- `main_sequence_analysis` has a unique layout (3×2 with shared axes, log scales,
  marginal jitter+CI strips) that doesn't fit the grid pattern. Leave it as-is
  for now but have it use Phase 1 helpers (`resolve_colors`, `bootstrap_ci`).
- The `SummaryDisplay` class and its `format()`/`label_margins()` methods are
  already solid. No changes needed there.
- Each phase is independently mergeable: Phase 1 is pure extraction with no
  behavior change, Phase 2 migrates the data layer to xarray/Zarr, Phase 3
  implements standalone plot functions, Phase 4 fixes the saccade query system,
  Phase 5 wires everything into a unified `plot()` method, Phase 6 cleans up.

---
Add new items as needed. Check off items as they are completed.
