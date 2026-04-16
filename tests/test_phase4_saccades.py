"""
Integration tests for Phase 4 saccade detection and querying.

Covers:
  - detect_saccades() populates saccade_table with the correct columns
  - save() + reload produces identical saccade table
  - query(object='saccade', output='amplitude') returns a flat list
  - query(object='saccade', output='amplitude', groupby='test') returns one
    value per test
  - query(object='saccade', subset={'peak_velocity': '>5'}) correctly filters
  - On-demand Saccade reconstruction (no baseline re-test) has matching scalar
    attributes to those stored in the table
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="skvideo")

import os
import shutil
import pytest
import numpy as np

from magno_tracker.tracking import TrackingTrial, Saccade

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
H5_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'h5_files')

H5_FILES = [
    os.path.join(H5_DIR, 'fh_baja_1_new_trial_1_cond1_new.h5'),
]

SACCADE_COLS = {
    'test_ind', 'test_start_frame', 'start_frame', 'stop_frame', 'peak_frame',
    'amplitude', 'peak_velocity', 'duration', 'start_angle', 'stop_angle',
}


def zarr_path(h5):
    return os.path.splitext(h5)[0] + '.zarr'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def trial():
    """Load one trial and run detect_saccades() once."""
    h5 = H5_FILES[0]
    t = TrackingTrial(h5)
    assert t.load_success, "Trial failed to load"
    t.detect_saccades()
    return t


# ---------------------------------------------------------------------------
# detect_saccades() populates saccade_table
# ---------------------------------------------------------------------------
class TestDetectSaccades:

    def test_saccade_table_exists(self, trial):
        assert hasattr(trial, 'saccade_table'), "saccade_table attribute missing"
        assert trial.saccade_table is not None

    def test_saccade_table_is_nonempty(self, trial):
        assert len(trial.saccade_table) > 0, "Expected at least one saccade"

    def test_saccade_table_columns(self, trial):
        row = trial.saccade_table[0]
        assert set(row.keys()) == SACCADE_COLS, (
            f"Column mismatch: got {set(row.keys())}"
        )

    def test_saccade_table_types(self, trial):
        for row in trial.saccade_table:
            assert isinstance(row['test_ind'], (int, np.integer))
            assert isinstance(row['start_frame'], (int, np.integer))
            assert isinstance(row['stop_frame'], (int, np.integer))
            assert isinstance(row['amplitude'], float)
            assert isinstance(row['peak_velocity'], float)
            assert isinstance(row['duration'], float)

    def test_start_before_stop(self, trial):
        for row in trial.saccade_table:
            assert row['start_frame'] < row['stop_frame'], (
                f"start_frame {row['start_frame']} >= stop_frame {row['stop_frame']}"
            )

    def test_test_ind_in_range(self, trial):
        for row in trial.saccade_table:
            assert 0 <= row['test_ind'] < trial.num_tests

    def test_duration_positive(self, trial):
        for row in trial.saccade_table:
            assert row['duration'] > 0

    def test_replaces_existing_table(self, trial):
        old_len = len(trial.saccade_table)
        trial.detect_saccades()
        assert len(trial.saccade_table) == old_len, (
            "Re-running detect_saccades() should produce the same table"
        )


# ---------------------------------------------------------------------------
# save() / reload round-trip
# ---------------------------------------------------------------------------
class TestSaccadeTableRoundTrip:

    def test_save_and_reload(self):
        h5 = H5_FILES[0]
        t = TrackingTrial(h5)
        t.detect_saccades()
        original_table = list(t.saccade_table)
        t.save()

        # Reload from the saved Zarr store.
        t2 = TrackingTrial(h5)
        assert hasattr(t2, 'saccade_table'), "saccade_table not restored after reload"
        assert len(t2.saccade_table) == len(original_table), (
            "Row count differs after round-trip"
        )
        for orig, reloaded in zip(original_table, t2.saccade_table):
            for col in SACCADE_COLS:
                assert col in reloaded, f"Column '{col}' missing after reload"
                orig_val = orig[col]
                reload_val = reloaded[col]
                if isinstance(orig_val, float):
                    assert abs(orig_val - reload_val) < 1e-9, (
                        f"Column '{col}': {orig_val} != {reload_val}"
                    )
                else:
                    assert orig_val == reload_val, (
                        f"Column '{col}': {orig_val} != {reload_val}"
                    )


# ---------------------------------------------------------------------------
# TrackingTrial.query(object='saccade', ...)
# ---------------------------------------------------------------------------
class TestTrialQuerySaccade:

    def test_flat_amplitude_returns_list(self, trial):
        result = trial.query(output='amplitude', object='saccade', groupby='saccade')
        assert isinstance(result, list)
        assert len(result) == len(trial.saccade_table)
        assert all(isinstance(v, float) for v in result)

    def test_groupby_test_returns_one_per_test(self, trial):
        result = trial.query(output='amplitude', object='saccade', groupby='test',
                             agg_func=np.nanmean)
        # Number of groups == number of distinct test_inds that have saccades.
        test_inds_with_saccades = len({r['test_ind'] for r in trial.saccade_table})
        assert len(result) == test_inds_with_saccades

    def test_groupby_test_list_agg(self, trial):
        result = trial.query(output='amplitude', object='saccade', groupby='test',
                             agg_func=list)
        for grp in result:
            assert isinstance(grp, list)

    def test_groupby_trial_returns_scalar(self, trial):
        result = trial.query(output='amplitude', object='saccade', groupby='trial',
                             agg_func=np.nanmean)
        assert isinstance(result, float) or isinstance(result, np.floating)

    def test_subset_peak_velocity_inequality(self, trial):
        all_velos = [abs(r['peak_velocity']) for r in trial.saccade_table]
        # Express threshold in rad/s — the table stores rad/s.
        threshold = np.percentile(all_velos, 50)
        thresh_str = f'>{threshold}'
        result = trial.query(output='peak_velocity', object='saccade', groupby='saccade',
                             subset={'peak_velocity': thresh_str})
        assert len(result) < len(trial.saccade_table), (
            "Filtering by peak_velocity should reduce the count"
        )
        for v in result:
            assert abs(v) > threshold

    def test_subset_returns_empty_list_when_no_match(self, trial):
        result = trial.query(output='amplitude', object='saccade', groupby='saccade',
                             subset={'peak_velocity': '>9999999'})
        assert result == []

    def test_raises_without_saccade_table(self):
        h5 = H5_FILES[0]
        t = TrackingTrial(h5)
        # Do NOT call detect_saccades()
        if hasattr(t, 'saccade_table'):
            del t.saccade_table
        with pytest.raises(RuntimeError, match="detect_saccades"):
            t.query(output='amplitude', object='saccade')


# ---------------------------------------------------------------------------
# saccade_table_df
# ---------------------------------------------------------------------------
class TestSaccadeTableDf:

    def test_returns_dataframe(self, trial):
        import pandas as pd
        df = trial.saccade_table_df()
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, trial):
        df = trial.saccade_table_df()
        assert len(df) == len(trial.saccade_table)

    def test_columns_match_schema(self, trial):
        df = trial.saccade_table_df()
        assert SACCADE_COLS.issubset(set(df.columns))

    def test_extra_col_scalar_attr(self, trial):
        # framerate is a scalar attr on every trial
        df = trial.saccade_table_df(extra_cols=['framerate'])
        assert 'framerate' in df.columns
        assert (df['framerate'] == trial.framerate).all()

    def test_extra_col_1d_test_level(self, trial):
        # is_test is a 1-D test-level bool array — each saccade gets its test's value
        df = trial.saccade_table_df(extra_cols=['is_test'])
        assert 'is_test' in df.columns
        for _, row in df.iterrows():
            assert row['is_test'] == trial.is_test[int(row['test_ind'])]

    def test_extra_col_2d_with_time_ind(self, trial):
        # time is 2-D (num_tests x num_frames); index by start_frame
        df = trial.saccade_table_df(extra_cols=['time'], time_ind='start_frame')
        assert 'time' in df.columns
        assert len(df) == len(trial.saccade_table)
        # Values should be non-negative (time starts at 0)
        assert (df['time'] >= 0).all()

    def test_extra_col_2d_peak_frame(self, trial):
        df = trial.saccade_table_df(extra_cols=['time'], time_ind='peak_frame')
        assert 'time' in df.columns
        assert (df['time'] >= 0).all()

    def test_empty_when_no_saccade_table(self):
        import pandas as pd
        h5 = H5_FILES[0]
        t = TrackingTrial(h5)
        if hasattr(t, 'saccade_table'):
            del t.saccade_table
        df = t.saccade_table_df()
        assert isinstance(df, pd.DataFrame)
        assert df.empty
class TestOnDemandSaccadeReconstruction:

    def test_output_saccade_returns_saccade_objects(self, trial):
        result = trial.query(output='saccade', object='saccade', groupby='saccade')
        assert len(result) == len(trial.saccade_table)
        assert all(isinstance(s, Saccade) for s in result)

    def test_reconstructed_amplitude_matches_table(self, trial):
        saccades = trial.query(output='saccade', object='saccade', groupby='saccade')
        for s, row in zip(saccades, trial.saccade_table):
            # Amplitude may differ slightly due to interpolation, but should
            # have the same sign and be within 10%.
            assert np.sign(s.amplitude) == np.sign(row['amplitude']), (
                "Reconstructed amplitude has opposite sign to stored value"
            )
            if abs(row['amplitude']) > 1e-6:
                rel_err = abs(s.amplitude - row['amplitude']) / abs(row['amplitude'])
                assert rel_err < 0.1, (
                    f"Amplitude relative error {rel_err:.3f} > 10%: "
                    f"stored={row['amplitude']:.4f}, reconstructed={s.amplitude:.4f}"
                )

    def test_reconstructed_start_stop_match_table(self, trial):
        saccades = trial.query(output='saccade', object='saccade', groupby='saccade')
        for s, row in zip(saccades, trial.saccade_table):
            assert s.start == row['start_frame']
            assert s.stop == row['stop_frame']
