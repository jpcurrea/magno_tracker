# NOTE: When working through this todo list, changes should be made step-by-step. Discuss each step and check for conceptual errors before making code changes. This workflow should be followed in future sessions with an LLM to ensure correctness and maintainability.

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

- [ ] **2.1 Prototype xarray Dataset creation**
      - Load HDF5 datasets into xarray, assigning dimension names (e.g., 'test', 'frame') and coordinates.
      - Validate that all variables (e.g., time, body_angle) align on dimensions.

- [ ] **2.2 Refactor query logic to use xarray**
      - Replace manual boolean mask logic with xarray's `.sel()`, `.where()`, and label-based indexing.
      - Ensure all filtering, subsetting, and aggregation is done via xarray methods.

- [ ] **2.3 Update downstream analysis and plotting**
      - Update plotting and analysis code to use xarray objects.
      - Ensure all code paths are tested and results match previous implementation.

- [ ] **2.4 Document migration and add tests**
      - Add migration notes and examples to the codebase.
      - Write unit tests for xarray-based query and filtering.

---

## Phase 3 — Standalone plot-type functions

Stateless functions: `(ax, xvals, yvals, color, **kwargs) → artist(s)`.
Easy to test with synthetic data and reusable outside `TrackingExperiment`.

- [ ] **3.1 `plot_line(ax, xs, ys, color, summary_func=None, ci=False, **kw)`**
      Individual traces (gray) + colored mean overlay + optional CI shading.

- [ ] **3.2 `plot_hist2d(ax, xs, ys, color, bins=100, density=False, **kw)`**
      2D histogram with a white→color linear colormap, optional mean-scatter overlay.

- [ ] **3.3 `plot_trajectory2d(ax, xs, ys, color, **kw)`**
      Cumulative heading → 2D path; options for contour, circ_hist, mean_line, ellipse CI.

- [ ] **3.4 `plot_histogram(ax, xs, bins, color, probability=False, summary_func=None, **kw)`**
      1D histogram (bar + optional summary line). Covers what `plot_histogram_summary` does per cell.

- [ ] **3.5 `plot_scatter(ax, xs, ys, color, **kw)`**
      Simple scatter with optional jitter and correlation annotation (Mardia's r + bootstrap p).

- [ ] **3.6 Write unit tests for all Phase 3 functions** using synthetic data and `Figure()`/`Axes`.

---

## Phase 4 — Fix hierarchical query so saccade-level data flows through `query()`

Currently `TrackingExperiment.query(object='saccade')` delegates to
`TrackingTrial.query_saccades()`, which returns `(time_arr, saccade_arr)` — a
different signature from trial-level queries. Meanwhile `plot_saccades` and
`plot_saccade_dynamics` bypass the query system entirely and inline their own
trial→bout→saccade loops with ad-hoc filtering.

The goal: make `query(object='saccade', output='amplitude', subset={...})` return
per-saccade values, with subsetting applied hierarchically:

- **Trial/bout-level keys** (`bg_gain`, `condition`, etc.) filter which bouts are included.
- **Saccade-level keys** (`peak_velocity`, `amplitude`, etc.) filter individual saccades.

### Aggregation via `groupby`

Saccade counts are ragged (different bouts have different numbers of saccades),
so we need a `groupby` parameter to control the granularity of the output,
paired with an `agg_func` (default `np.nanmean`) for the reduction:

| `groupby=`    | Returns                        | Use case                                      |
|---------------|--------------------------------|-----------------------------------------------|
| `'saccade'`   | flat array, one val per saccade | scatter plots, histograms of individual events |
| `'bout'`      | one val per bout (via `agg_func`) | per-condition traces where each bout = one trace |
| `'trial'`     | one val per trial/subject (via `agg_func`) | subject-level stats, bootstrapping |

For **time-series** outputs (`arr_relative`, `velocity`, etc.), `groupby='saccade'`
returns a list of variable-length arrays; `'bout'`/`'trial'` require that `agg_func`
can handle ragged inputs (e.g., bin-average to a common time grid first).

For **scalar** outputs (`amplitude`, `peak_velocity`, etc.), all three `groupby`
levels return ndarrays — just at different sizes.

The `plot()` method (Phase 4) passes `groupby` through, so the same query
interface works for all downstream plotting.

### Tasks

- [ ] **4.1 Define which saccade attributes are "scalar" vs "time-series"**
      Scalars: `amplitude`, `duration`, `peak_velocity`, `start_angle`, `stop_angle`,
      `start_time`, `stop_time`. Time-series: `arr_relative`, `velocity`, `time`,
      `relative_time`. Document these in a class-level dict or constant
      (e.g., `Saccade.SCALAR_ATTRS`, `Saccade.TIMESERIES_ATTRS`).

- [ ] **4.2 Refactor `TrackingTrial.query_saccades()`**
      New signature: `query_saccades(output, groupby='saccade', agg_func=np.nanmean,
      subset={}, sort_by='test_ind', min_speed=0, max_speed=np.inf)`.
      - Partition `subset` keys into trial/bout-level vs saccade-level.
      - Apply trial-level filters via `query_bouts()`.
      - Walk filtered bouts → saccades, apply saccade-level filters.
      - Collect the requested `output` attribute from each surviving saccade.
      - If `groupby='saccade'`: return flat array (scalars) or list (time-series).
      - If `groupby='bout'`: apply `agg_func` per bout → one value per bout.
      - If `groupby='trial'`: apply `agg_func` across all saccades → one value.

- [ ] **4.3 Update `Bout.query_saccades()` accordingly**
      Currently walks `self.saccades` and checks subset keys against
      `self.trial`, `self`, and each `saccade`. Refactor to:
      - Accept the same `output`/`groupby`/`agg_func` interface.
      - Use the partitioned saccade-level subset keys only (bout/trial
        filtering already happened upstream).
      - Return the requested attribute per saccade (if `groupby='saccade'`)
        or the aggregated value (if `groupby='bout'`).

- [ ] **4.4 Update `TrackingExperiment.query(object='saccade')`**
      New kwargs: `groupby='saccade'`, `agg_func=np.nanmean`.
      Aggregation at the experiment level:
      - `groupby='saccade'`: concatenate all per-trial flat arrays → one big array.
      - `groupby='bout'`: concatenate per-trial bout-level arrays → one per bout.
      - `groupby='trial'`: collect per-trial scalars → array of length N_trials.

- [ ] **4.5 Write integration tests** for `query(object='saccade')` confirming:
      - Trial-level subset correctly limits included bouts.
      - Saccade-level subset correctly filters by `peak_velocity`, `amplitude`, etc.
      - `groupby='saccade'` + scalar output → flat ndarray, length = total saccades.
      - `groupby='bout'` + scalar output → ndarray, length = total bouts.
      - `groupby='trial'` + scalar output → ndarray, length = total trials.
      - `groupby='saccade'` + time-series output → list of variable-length arrays.
      - `agg_func` is correctly applied (test with `np.nanmean`, `len`, etc.).

---

## Phase 4 — Unified `TrackingExperiment.plot()` method

A single entry point that handles grid setup, querying, iteration over row/col
subsets, and margin summaries — delegating actual drawing to Phase 2 functions.

- [ ] **4.1 Implement `TrackingExperiment.plot()`**
      Signature: `plot(xvar, yvar, col_var, row_var, plot_type='line',
      object='trial', row_cmap=None, col_cmap=None, ...)`
      Internally:
        1. `get_grid_vals()` (Phase 1.2) for row/col values
        2. `resolve_colors()` (Phase 1.1) for the color array
        3. Create `SummaryDisplay` grid
        4. Loop over row × col, calling `self.query(object=object, output=xvar, subset=...)`
        5. Dispatch to the appropriate Phase 2 function based on `plot_type`
        6. Populate margin summaries
        7. Call `display.format()` and `display.label_margins()`

- [ ] **4.2 Test `plot()` with `plot_type='line'`** — verify identical output to current `plot_summary` default

- [ ] **4.3 Test `plot()` with `plot_type='hist2d'`** — verify identical output to `plot_summary(..., plot_type='hist2d')`

- [ ] **4.4 Test `plot()` with `plot_type='hist'`** — verify it replaces `plot_histogram_summary`

- [ ] **4.5 Test `plot()` with `plot_type='trajectory2d'`** — verify identical output to `plot_summary(..., plot_type='trajectory2d')`

- [ ] **4.6 Test `plot()` with `object='saccade', plot_type='line'`** — verify it replaces `plot_saccades`

- [ ] **4.7 Test `plot()` with `object='saccade', plot_type='scatter'` or `'hist2d'`** — verify it replaces `plot_saccade_dynamics`

---

## Phase 5 — Deprecate old methods and clean up


- [ ] **5.1 Rewrite `plot_summary` as a thin wrapper** around `plot()` (for backward compat), mark deprecated

- [ ] **5.2 Rewrite `plot_histogram_summary`** as `plot(..., plot_type='hist')`, mark deprecated

- [ ] **5.3 Rewrite `plot_saccades`** as `plot(..., object='saccade', plot_type='line')`, mark deprecated

- [ ] **5.4 Rewrite `plot_saccade_dynamics`** as `plot(..., object='saccade', plot_type='scatter')`, mark deprecated

- [ ] **5.5 Remove dead code:** `breakpoint()` calls, large commented-out blocks throughout all methods

- [ ] **5.6 Add/update docstrings** for all new and modified public functions and methods

---

## Notes

- `main_sequence_analysis` has a unique layout (3×2 with shared axes, log scales,
  marginal jitter+CI strips) that doesn't fit the grid pattern. Leave it as-is
  for now but have it use Phase 1 helpers (`resolve_colors`, `bootstrap_ci`).
- The `SummaryDisplay` class and its `format()`/`label_margins()` methods are
  already solid. No changes needed there.
- Each phase is independently mergeable: Phase 1 is pure extraction with no
  behavior change, Phase 2 adds new code, Phase 3 fixes the query system,
  Phase 4 wires everything together, Phase 5 cleans up.

---
Add new items as needed. Check off items as they are completed.
