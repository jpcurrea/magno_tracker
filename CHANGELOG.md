# Changelog

All notable changes to `magno_tracker` are documented here.
See [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) for detailed phase-by-phase implementation notes.

---

## [2.0.0] — 2026-04-24

### Breaking changes

- **`Bout` deprecated.** Raises `DeprecationWarning` on instantiation. Replaced by
  `TrackingTrial.detect_saccades()` + `TrackingTrial.query(object='saccade', ...)`.
- **Data layer migrated to xarray + Zarr.** Existing `.h5` files are still readable
  on first open; call `trial.save()` to convert.  Direct HDF5 attribute access is
  replaced by the `query()` / `add_attr()` API.
- **`plot_summary`, `plot_histogram_summary`, `plot_saccades`, `plot_saccade_dynamics`
  are deprecated.** All four raise `DeprecationWarning` and forward to `plot()`.

### New features

- **`TrackingTrial.detect_saccades()` / `TrackingExperiment.detect_saccades()`** —
  two-phase saccade detection; table persisted to a `.saccades.pkl` sidecar file.
- **`TrackingTrial.query()` / `TrackingExperiment.query()`** — flexible querying with
  `subset`, `groupby` (`'saccade'` / `'test'` / `'trial'`), and `agg_func`.
- **`TrackingExperiment.plot()`** — unified plotting entry point supporting `line`,
  `histogram`, `hist2d`, `scatter`, and `trajectory2d` plot types, per-panel margins,
  `groupby` trajectory averaging, `margin_groupby` distribution grouping, and
  `split_by_sign`.
- **Standalone plot functions** — `plot_line`, `plot_hist2d`, `plot_trajectory2d`,
  `plot_histogram`, `plot_scatter`, `plot_pdf` (Phase 3).
- **`SummaryDisplay`** — custom subplot sizing (`width_ratios` / `height_ratios`),
  dynamic label positioning on window resize (Phase 7).
- **`main_sequence_analysis()`** — rewritten with log2-scale tick formatting and
  `resolve_colors` (Phase 7).
- **Tutorial notebooks** — `notebooks/tutorial_query.ipynb` and
  `notebooks/tutorial_plot.ipynb` (Phase 8).
- **`__version__`** and public `__all__` exposed from the package root.

### Other improvements (Phases 1–9)

- `resolve_colors`, `bootstrap_ci`, `get_grid_vals`, `omit_wrapping` extracted as
  module-level helpers usable independently.
- `SummaryDisplay.format()` / `label_margins()` fully rewritten for DPI-independent
  label placement.
- `.saccades.pkl` sidecar reduces saccade-only save time to < 1 s.
- `_detect_saccades` internal helper stripped of Kalman / acceleration dead code.
- 269 tests across 6 test modules (all passing).

---

## [1.0.0] — 2023 (initial release)

Original `Bout`-based tracking and saccade analysis, HDF5 storage.
