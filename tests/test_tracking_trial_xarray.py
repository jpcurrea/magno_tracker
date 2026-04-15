"""
Integration tests for TrackingTrial xarray-based load, load_datasets, query, and save.

Tests run against three real h5 files, then save .zarr stores and re-run the
same assertions from the Zarr versions to confirm round-trip consistency.
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="skvideo")

import os
import pytest
import numpy as np
import xarray as xr

from magno_tracker.tracking import TrackingTrial

# ---------------------------------------------------------------------------
# Paths to three representative h5 files
# ---------------------------------------------------------------------------
H5_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'h5_files')
# H5_DIR = 'h5_files'  # Assuming h5_files is in the current working directory

H5_FILES = [
    os.path.join(H5_DIR, 'fh_baja_1_new_trial_1_cond1_new.h5'),
    os.path.join(H5_DIR, 'fh_baja_1_new_trial_1_cond2_new.h5'),
    os.path.join(H5_DIR, 'fh_baja_1_new_trial_1_cond3_new.h5'),
]

H5_FILES = H5_FILES[:1]


def zarr_path(h5):
    return os.path.splitext(h5)[0] + '.zarr'


# @pytest.fixture(scope='module', autouse=True)
# def cleanup_zarr_stores():
#     """Remove any .zarr stores created during the test session."""
#     yield
#     import gc
#     import shutil
#     gc.collect()
#     for h5 in H5_FILES:
#         zp = zarr_path(h5)
#         if os.path.exists(zp):
#             try:
#                 shutil.rmtree(zp)
#             except Exception as e:
#                 import warnings
#                 warnings.warn(f"Could not remove {zp} — {e}. Delete manually between runs.")


# ---------------------------------------------------------------------------
# Helper: assert a loaded trial looks correct
# ---------------------------------------------------------------------------
def assert_trial_valid(trial, source_label=''):
    assert trial.load_success, f"[{source_label}] load_success should be True"

    # Dimensions
    assert hasattr(trial, 'num_tests'), f"[{source_label}] num_tests missing"
    assert hasattr(trial, 'num_frames'), f"[{source_label}] num_frames missing"
    assert trial.num_tests > 0, f"[{source_label}] num_tests must be > 0"
    assert trial.num_frames > 0, f"[{source_label}] num_frames must be > 0"

    # Derived arrays on self
    assert hasattr(trial, 'test_ind'), f"[{source_label}] test_ind missing"
    assert hasattr(trial, 'frame_ind'), f"[{source_label}] frame_ind missing"
    assert hasattr(trial, 'time'), f"[{source_label}] time missing"
    assert trial.test_ind.shape == (trial.num_tests,)
    assert trial.frame_ind.shape == (trial.num_tests, trial.num_frames)
    assert trial.time.shape == (trial.num_tests, trial.num_frames)

    # xarray Dataset has named dims
    assert 'test' in trial.h5_file.dims, f"[{source_label}] 'test' dim missing"
    assert 'frame' in trial.h5_file.dims, f"[{source_label}] 'frame' dim missing"
    assert trial.h5_file.sizes['test'] == trial.num_tests
    assert trial.h5_file.sizes['frame'] == trial.num_frames

    # is_test present
    assert hasattr(trial, 'is_test'), f"[{source_label}] is_test missing"


def assert_query_valid(trial, source_label=''):
    # Default query returns camera_heading shaped (num_tests, num_frames)
    data = trial.query('camera_heading')
    assert isinstance(data, np.ndarray), f"[{source_label}] query must return ndarray"
    assert data.ndim == 2, f"[{source_label}] camera_heading query should be 2D"
    assert data.shape == (trial.num_tests, trial.num_frames), (
        f"[{source_label}] shape mismatch: {data.shape} vs "
        f"({trial.num_tests}, {trial.num_frames})"
    )

    # Scalar query (test_ind) returns 1D array of length num_tests
    test_inds = trial.query('test_ind')
    assert test_inds.ndim == 1
    assert len(test_inds) == trial.num_tests

    # Time query
    time_data = trial.query('time')
    assert time_data.shape == (trial.num_tests, trial.num_frames)


# ---------------------------------------------------------------------------
# Parametrised tests over the 3 h5 files
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('h5_file', H5_FILES)
class TestTrackingTrialFromH5:

    def test_load_succeeds(self, h5_file):
        trial = TrackingTrial(h5_file)
        assert trial.file_opened, "file_opened should be True"
        assert_trial_valid(trial, source_label=os.path.basename(h5_file))

    def test_load_datasets(self, h5_file):
        trial = TrackingTrial(h5_file)
        assert trial.load_success
        # camera_heading accessible via _get_var
        ch = trial._get_var('camera_heading')
        assert isinstance(ch, np.ndarray)
        assert ch.ndim == 2

    def test_query_default(self, h5_file):
        trial = TrackingTrial(h5_file)
        assert_query_valid(trial, source_label=os.path.basename(h5_file))

    def test_save_creates_zarr(self, h5_file):
        trial = TrackingTrial(h5_file)
        zp = trial.save()
        assert os.path.exists(zp), f".zarr store was not created at {zp}"
        assert zp.endswith('.zarr')

    def test_save_zarr_is_valid_xarray(self, h5_file):
        trial = TrackingTrial(h5_file)
        trial.save()
        zp = zarr_path(h5_file)
        ds = xr.open_dataset(zp, engine='zarr')
        assert ds is not None
        ds.close()


@pytest.mark.parametrize('h5_file', H5_FILES)
class TestTrackingTrialFromZarr:
    """
    Same assertions, but loading from the saved .zarr store.
    Requires TestTrackingTrialFromH5.test_save_creates_zarr to have run first.
    """

    @pytest.fixture(autouse=True)
    def ensure_zarr_exists(self, h5_file):
        """Save the .zarr store if it doesn't already exist."""
        zp = zarr_path(h5_file)
        if not os.path.exists(zp):
            trial = TrackingTrial(h5_file)
            trial.save()

    def test_load_from_zarr_succeeds(self, h5_file):
        trial = TrackingTrial(h5_file)
        assert trial.file_opened
        assert_trial_valid(trial, source_label='Zarr:' + os.path.basename(h5_file))

    def test_load_datasets_from_zarr(self, h5_file):
        trial = TrackingTrial(h5_file)
        ch = trial._get_var('camera_heading')
        assert isinstance(ch, np.ndarray)
        assert ch.ndim == 2

    def test_query_from_zarr(self, h5_file):
        trial = TrackingTrial(h5_file)
        assert_query_valid(trial, source_label='Zarr:' + os.path.basename(h5_file))

    def test_zarr_matches_h5_camera_heading(self, h5_file):
        """Values loaded from .zarr should match those loaded directly from .h5."""
        # Force load from h5 using force_h5=True to avoid renaming the open .nc file
        trial_h5 = TrackingTrial.__new__(TrackingTrial)
        trial_h5.filename = h5_file
        trial_h5.file_opened = trial_h5.load(h5_file, force_h5=True)
        trial_h5.load_datasets()
        ch_h5 = trial_h5._get_var('camera_heading')

        trial_nc = TrackingTrial(h5_file)
        ch_nc = trial_nc._get_var('camera_heading')

        np.testing.assert_allclose(
            ch_h5, ch_nc, rtol=1e-5,
            err_msg=f"camera_heading mismatch between h5 and zarr for {os.path.basename(h5_file)}"
        )


# ---------------------------------------------------------------------------
# Synthetic unit tests for add_dataset / remove_dataset / add_attr
# No real files required — trial is built from a minimal in-memory Dataset.
# ---------------------------------------------------------------------------

def _make_synthetic_trial(num_tests=3, num_frames=10):
    """Return a TrackingTrial populated from a small in-memory xr.Dataset."""
    ds = xr.Dataset(
        {
            'camera_heading': xr.DataArray(
                np.random.rand(num_tests, num_frames),
                dims=('test', 'frame'),
            ),
            'is_test': xr.DataArray(
                np.ones(num_tests, dtype=bool),
                dims=('test',),
            ),
        },
        attrs={'framerate': 60.0, 'duration': num_frames / 60.0},
    )
    trial = TrackingTrial.__new__(TrackingTrial)
    trial.filename = 'synthetic.h5'
    trial.file_opened = True
    trial.load_success = False
    trial.h5_file = ds
    trial.load_datasets()
    return trial


class TestQuery:

    def test_no_subset_returns_all_tests(self):
        trial = _make_synthetic_trial(num_tests=4, num_frames=5)
        result = trial.query('camera_heading')
        assert result.shape == (4, 5)

    def test_no_subset_sorted_by_test_ind(self):
        trial = _make_synthetic_trial(num_tests=4, num_frames=5)
        # test_ind is 0,1,2,3 — result should already be in that order
        result = trial.query('test_ind')
        np.testing.assert_array_equal(result, np.arange(4))

    def test_1d_subset_equality_drops_tests(self):
        trial = _make_synthetic_trial(num_tests=4, num_frames=5)
        # Add a 1-D grouping variable: two tests with val=1, two with val=2
        group = np.array([1, 2, 1, 2])
        trial.add_dataset('group', group)
        result = trial.query('camera_heading', subset={'group': 1})
        assert result.shape[0] == 2  # only 2 tests have group==1
        assert result.shape[1] == 5

    def test_1d_subset_list_membership(self):
        trial = _make_synthetic_trial(num_tests=4, num_frames=5)
        group = np.array([1, 2, 3, 4])
        trial.add_dataset('group', group)
        result = trial.query('camera_heading', subset={'group': [1, 3]})
        assert result.shape[0] == 2

    def test_1d_subset_string_inequality(self):
        trial = _make_synthetic_trial(num_tests=4, num_frames=5)
        score = np.array([0.1, 0.5, 0.8, 0.3])
        trial.add_dataset('score', score)
        result = trial.query('camera_heading', subset={'score': '>0.4'})
        assert result.shape[0] == 2  # 0.5 and 0.8 pass

    def test_1d_subset_nan_match(self):
        trial = _make_synthetic_trial(num_tests=4, num_frames=5)
        score = np.array([1.0, np.nan, 2.0, np.nan])
        trial.add_dataset('score', score)
        result = trial.query('camera_heading', subset={'score': float('nan')})
        assert result.shape[0] == 2

    def test_2d_subset_nan_fills_frames(self):
        trial = _make_synthetic_trial(num_tests=3, num_frames=10)
        # time goes 0..9 for each test
        time = np.tile(np.arange(10, dtype=float), (3, 1))
        trial.add_dataset('time', time)
        result = trial.query('camera_heading', subset={'time': '<=4'})
        # shape is preserved (all 3 tests kept)
        assert result.shape == (3, 10)
        # frames 0-4 should be real values (not NaN), frames 5-9 should be NaN
        assert not np.any(np.isnan(result[:, :5]))
        assert np.all(np.isnan(result[:, 5:]))

    def test_combined_1d_and_2d_subset(self):
        trial = _make_synthetic_trial(num_tests=4, num_frames=10)
        group = np.array([1, 2, 1, 2])
        trial.add_dataset('group', group)
        time = np.tile(np.arange(10, dtype=float), (4, 1))
        trial.add_dataset('time', time)
        result = trial.query('camera_heading', subset={'group': 1, 'time': '<=4'})
        # 2 tests pass the group filter; frames 5-9 are NaN
        assert result.shape == (2, 10)
        assert np.all(np.isnan(result[:, 5:]))
        assert not np.any(np.isnan(result[:, :5]))

    def test_returns_numpy_array(self):
        trial = _make_synthetic_trial()
        result = trial.query('camera_heading')
        assert isinstance(result, np.ndarray)


class TestAddDataset:

    def test_add_1d_array(self):
        trial = _make_synthetic_trial()
        arr = np.arange(trial.num_tests, dtype=float)
        trial.add_dataset('my_var', arr)
        assert 'my_var' in trial.h5_file.data_vars
        np.testing.assert_array_equal(trial._get_var('my_var'), arr)

    def test_add_2d_array(self):
        trial = _make_synthetic_trial()
        arr = np.ones((trial.num_tests, trial.num_frames))
        trial.add_dataset('my_2d_var', arr)
        assert trial.h5_file['my_2d_var'].dims == ('test', 'frame')
        np.testing.assert_array_equal(trial._get_var('my_2d_var'), arr)

    def test_overwrite_existing_variable(self):
        trial = _make_synthetic_trial()
        arr1 = np.zeros(trial.num_tests)
        arr2 = np.ones(trial.num_tests)
        trial.add_dataset('my_var', arr1)
        trial.add_dataset('my_var', arr2)
        np.testing.assert_array_equal(trial._get_var('my_var'), arr2)

    def test_accessible_via_query(self):
        trial = _make_synthetic_trial()
        arr = np.arange(trial.num_tests, dtype=float)
        trial.add_dataset('my_var', arr)
        result = trial.query('my_var')
        assert isinstance(result, np.ndarray)
        assert len(result) == trial.num_tests

    def test_sensitive_variable_warns(self):
        trial = _make_synthetic_trial()
        new_is_test = np.zeros(trial.num_tests, dtype=bool)
        with pytest.warns(UserWarning, match='load_datasets'):
            trial.add_dataset('is_test', new_is_test)

    def test_non_sensitive_variable_no_warning(self):
        trial = _make_synthetic_trial()
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            trial.add_dataset('safe_var', np.zeros(trial.num_tests))  # should not raise


class TestRemoveDataset:

    def test_remove_existing_variable(self):
        trial = _make_synthetic_trial()
        trial.add_dataset('to_remove', np.zeros(trial.num_tests))
        assert 'to_remove' in trial.h5_file.data_vars
        trial.remove_dataset('to_remove')
        assert 'to_remove' not in trial.h5_file.data_vars

    def test_remove_nonexistent_is_silent(self):
        trial = _make_synthetic_trial()
        trial.remove_dataset('does_not_exist')  # should not raise

    def test_sensitive_variable_warns(self):
        trial = _make_synthetic_trial()
        with pytest.warns(UserWarning, match='load_datasets'):
            trial.remove_dataset('is_test')

    def test_removed_variable_not_queryable(self):
        trial = _make_synthetic_trial()
        trial.add_dataset('to_remove', np.zeros(trial.num_tests))
        trial.remove_dataset('to_remove')
        with pytest.raises((AttributeError, KeyError)):
            trial._get_var('to_remove')


class TestAddAttr:

    def test_attr_set_on_dataset(self):
        trial = _make_synthetic_trial()
        trial.add_attr('my_attr', 42)
        assert trial.h5_file.attrs['my_attr'] == 42

    def test_attr_set_on_self(self):
        trial = _make_synthetic_trial()
        trial.add_attr('my_attr', 'hello')
        assert trial.my_attr == 'hello'

    def test_attr_overwrite(self):
        trial = _make_synthetic_trial()
        trial.add_attr('my_attr', 1)
        trial.add_attr('my_attr', 2)
        assert trial.h5_file.attrs['my_attr'] == 2
        assert trial.my_attr == 2


# ---------------------------------------------------------------------------
# Round-trip test: add_dataset → save → reload from Zarr → query
# Uses a real h5 file so save() has a valid path to write to.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('h5_file', H5_FILES)
class TestAddDatasetRoundTrip:

    @pytest.fixture(autouse=True)
    def ensure_zarr_and_cleanup(self, h5_file, tmp_path, monkeypatch):
        """
        Give each trial a unique zarr path inside tmp_path so tests are
        isolated and the store is automatically deleted after the test.
        """
        import shutil
        # Redirect self.filename so save() writes into pytest's tmp_path
        unique_h5 = str(tmp_path / 'trial.h5')
        # We need the trial to think its filename lives in tmp_path, but we
        # still load from the real h5.  Create the trial normally then patch.
        self._h5_file = h5_file
        self._tmp_h5 = unique_h5
        yield
        zp = os.path.splitext(unique_h5)[0] + '.zarr'
        if os.path.exists(zp):
            shutil.rmtree(zp)

    def _make_trial(self):
        trial = TrackingTrial(self._h5_file)
        # Redirect filename so save() writes to the isolated tmp location
        trial.filename = self._tmp_h5
        return trial

    def test_add_1d_persists_after_save_reload(self):
        trial = self._make_trial()
        arr = np.arange(trial.num_tests, dtype=float) * 99
        trial.add_dataset('rt_var_1d', arr)
        trial.save()

        # Reload from the saved zarr
        trial2 = TrackingTrial(self._tmp_h5)
        result = trial2._get_var('rt_var_1d')
        np.testing.assert_array_equal(result, arr)

    def test_add_2d_persists_after_save_reload(self):
        trial = self._make_trial()
        arr = np.random.rand(trial.num_tests, trial.num_frames)
        trial.add_dataset('rt_var_2d', arr)
        trial.save()

        trial2 = TrackingTrial(self._tmp_h5)
        result = trial2._get_var('rt_var_2d')
        np.testing.assert_allclose(result, arr)

    def test_remove_persists_after_save_reload(self):
        trial = self._make_trial()
        trial.add_dataset('to_remove', np.zeros(trial.num_tests))
        trial.save()

        # Confirm it's there after reload
        trial2 = TrackingTrial(self._tmp_h5)
        assert 'to_remove' in trial2.h5_file.data_vars

        # Now remove and save again
        trial2.remove_dataset('to_remove')
        trial2.save()

        trial3 = TrackingTrial(self._tmp_h5)
        assert 'to_remove' not in trial3.h5_file.data_vars

    def test_add_attr_persists_after_save_reload(self):
        trial = self._make_trial()
        trial.add_attr('rt_attr', 123)
        trial.save()

        trial2 = TrackingTrial(self._tmp_h5)
        assert trial2.h5_file.attrs.get('rt_attr') == 123


# ---------------------------------------------------------------------------
# Integration tests for query() using real h5 data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('h5_file', H5_FILES)
class TestQueryIntegration:
    """Tests for query() with real h5 data — subsets, 2-D NaN fill, sort."""

    def test_query_is_test_subset(self, h5_file):
        """Subsetting by is_test=True should return only test trials."""
        trial = TrackingTrial(h5_file)
        is_test = trial._get_var('is_test').astype(bool)
        expected_n = int(np.sum(is_test))
        result = trial.query('camera_heading', subset={'is_test': True})
        assert result.shape[0] == expected_n
        assert result.shape[1] == trial.num_frames

    def test_query_default_sort_is_ascending(self, h5_file):
        """Default sort_by='test_ind' should return indices in ascending order."""
        trial = TrackingTrial(h5_file)
        test_inds = trial.query('test_ind')
        assert np.all(test_inds[:-1] <= test_inds[1:])

    def test_query_1d_inequality_drops_tests(self, h5_file):
        """A string inequality on a 1-D variable should drop non-matching tests."""
        trial = TrackingTrial(h5_file)
        midpoint = max(1, trial.num_tests // 2)
        result = trial.query('camera_heading', subset={'test_ind': f'<{midpoint}'})
        assert result.shape[0] == midpoint
        assert result.shape[1] == trial.num_frames

    def test_query_2d_subset_nan_fills_frames(self, h5_file):
        """A 2-D subset var should NaN-fill non-matching frames, not drop tests."""
        trial = TrackingTrial(h5_file)
        # Synthetic per-test time: each row goes 0.0 → 1.0 regardless of test index
        norm_time = np.tile(
            np.linspace(0.0, 1.0, trial.num_frames), (trial.num_tests, 1)
        )
        trial.add_dataset('norm_time', norm_time)
        result = trial.query('camera_heading', subset={'norm_time': '<=0.5'})
        # All tests kept (2-D subset never drops tests)
        assert result.shape[0] == trial.num_tests
        assert result.shape[1] == trial.num_frames
        # Some frames must be NaN-filled (those where norm_time > 0.5)
        assert np.any(np.isnan(result))
        # The first frame (norm_time == 0.0) must not be NaN in any test
        assert not np.any(np.isnan(result[:, 0]))

    def test_query_combined_1d_and_2d_subset(self, h5_file):
        """Combining a 1-D and a 2-D subset should both drop tests and NaN fill."""
        trial = TrackingTrial(h5_file)
        midpoint = max(1, trial.num_tests // 2)
        norm_time = np.tile(
            np.linspace(0.0, 1.0, trial.num_frames), (trial.num_tests, 1)
        )
        trial.add_dataset('norm_time', norm_time)
        result = trial.query(
            'camera_heading',
            subset={'test_ind': f'<{midpoint}', 'norm_time': '<=0.5'},
        )
        assert result.shape[0] == midpoint
        assert result.shape[1] == trial.num_frames
        assert np.any(np.isnan(result))
        assert not np.any(np.isnan(result[:, 0]))


# ---------------------------------------------------------------------------
# Integration tests for butterworth_filter() using real h5 data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('h5_file', H5_FILES)
class TestButterworthIntegration:
    """Tests for butterworth_filter() against real h5 data."""

    def test_smoothed_variable_in_data_vars(self, h5_file):
        """butterworth_filter should add '{key}_smoothed' to h5_file.data_vars."""
        trial = TrackingTrial(h5_file)
        trial.butterworth_filter(key='camera_heading')
        assert 'camera_heading_smoothed' in trial.h5_file.data_vars

    def test_smoothed_queryable_correct_shape(self, h5_file):
        """The smoothed variable should be queryable with shape (num_tests, num_frames)."""
        trial = TrackingTrial(h5_file)
        trial.butterworth_filter(key='camera_heading')
        result = trial.query('camera_heading_smoothed')
        assert isinstance(result, np.ndarray)
        assert result.shape == (trial.num_tests, trial.num_frames)

    def test_smoothed_differs_from_raw(self, h5_file):
        """The filtered signal should not be identical to the raw signal."""
        trial = TrackingTrial(h5_file)
        raw = trial.query('camera_heading')
        trial.butterworth_filter(key='camera_heading')
        smoothed = trial.query('camera_heading_smoothed')
        assert not np.allclose(raw, smoothed)

    def test_smoothed_persists_after_save_reload(self, h5_file, tmp_path):
        """Saving after butterworth_filter should persist the smoothed data."""
        import shutil
        trial = TrackingTrial(h5_file)
        trial.filename = str(tmp_path / 'trial.h5')
        trial.butterworth_filter(key='camera_heading')
        expected = trial.query('camera_heading_smoothed')
        trial.save()

        trial2 = TrackingTrial(trial.filename)
        result = trial2.query('camera_heading_smoothed')
        np.testing.assert_allclose(result, expected)

        zp = os.path.splitext(trial.filename)[0] + '.zarr'
        if os.path.exists(zp):
            shutil.rmtree(zp)


# ---------------------------------------------------------------------------
# Integration tests for unwrap → get_saccade_stats → remove_saccades pipeline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('h5_file', H5_FILES)
class TestRemoveSaccadesIntegration:
    """Tests for the unwrap → get_saccade_stats → remove_saccades pipeline."""

    def test_unwrap_adds_dataset(self, h5_file):
        """unwrap() should add '{key}_unwrapped' to h5_file.data_vars."""
        trial = TrackingTrial(h5_file)
        trial.unwrap(key='camera_heading')
        assert 'camera_heading_unwrapped' in trial.h5_file.data_vars

    def test_unwrapped_correct_shape(self, h5_file):
        """Unwrapped array should have the same shape as the original."""
        trial = TrackingTrial(h5_file)
        trial.unwrap(key='camera_heading')
        raw = trial.query('camera_heading')
        unwrapped = trial.query('camera_heading_unwrapped')
        assert unwrapped.shape == raw.shape

    def test_get_saccade_stats_populates_bouts(self, h5_file):
        """get_saccade_stats() should populate trial.bouts with one Bout per test."""
        trial = TrackingTrial(h5_file)
        trial.unwrap(key='camera_heading')
        trial.get_saccade_stats(key='camera_heading_unwrapped')
        assert hasattr(trial, 'bouts')
        assert len(trial.bouts) == trial.num_tests

    def test_get_saccade_stats_adds_saccade_datasets(self, h5_file):
        """get_saccade_stats() should add saccade summary variables to data_vars."""
        trial = TrackingTrial(h5_file)
        trial.unwrap(key='camera_heading')
        trial.get_saccade_stats(key='camera_heading_unwrapped')
        for var in ['saccade_duration', 'saccade_amplitude', 'saccade_peak_velocity',
                    'saccade_frequency', 'saccading_left', 'saccading_right']:
            assert var in trial.h5_file.data_vars, f"'{var}' missing from data_vars"

    def test_remove_saccades_adds_dataset(self, h5_file):
        """remove_saccades() should add '{key}_no_saccades_{method}' to data_vars."""
        trial = TrackingTrial(h5_file)
        trial.unwrap(key='camera_heading')
        trial.get_saccade_stats(key='camera_heading_unwrapped')
        trial.remove_saccades(key='camera_heading_unwrapped')
        assert 'camera_heading_unwrapped_no_saccades_zero_velocity' in trial.h5_file.data_vars

    def test_remove_saccades_correct_shape(self, h5_file):
        """The saccade-removed array should have the same shape as the input."""
        trial = TrackingTrial(h5_file)
        trial.unwrap(key='camera_heading')
        trial.get_saccade_stats(key='camera_heading_unwrapped')
        trial.remove_saccades(key='camera_heading_unwrapped')
        unwrapped = trial.query('camera_heading_unwrapped')
        no_saccades = trial.query('camera_heading_unwrapped_no_saccades_zero_velocity')
        assert no_saccades.shape == unwrapped.shape

    def test_remove_saccades_differs_from_original(self, h5_file):
        """The saccade-removed signal should differ from the unwrapped signal."""
        trial = TrackingTrial(h5_file)
        trial.unwrap(key='camera_heading')
        trial.get_saccade_stats(key='camera_heading_unwrapped')
        trial.remove_saccades(key='camera_heading_unwrapped')
        unwrapped = trial.query('camera_heading_unwrapped')
        no_saccades = trial.query('camera_heading_unwrapped_no_saccades_zero_velocity')
        assert not np.allclose(unwrapped, no_saccades), \
            "Expected saccade removal to change the signal, but arrays are identical"

    def test_plot_unwrapped_vs_no_saccades(self, h5_file, tmp_path):
        """Plot camera_heading_unwrapped and the saccade-removed version side-by-side."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        trial = TrackingTrial(h5_file)
        trial.unwrap(key='camera_heading')
        trial.get_saccade_stats(key='camera_heading_unwrapped')
        trial.remove_saccades(key='camera_heading_unwrapped')

        unwrapped = trial.query('camera_heading_unwrapped')
        no_saccades = trial.query('camera_heading_unwrapped_no_saccades_zero_velocity')
        time = trial.time  # shape (num_tests, num_frames)

        fig, axes = plt.subplots(
            nrows=trial.num_tests, ncols=1,
            figsize=(12, 3 * trial.num_tests),
            squeeze=False,
        )
        for i, ax in enumerate(axes[:, 0]):
            t = time[i]
            ax.plot(t, np.degrees(unwrapped[i]), color='steelblue', alpha=0.7,
                    label='unwrapped')
            ax.plot(t, np.degrees(no_saccades[i]), color='tomato', alpha=0.9,
                    label='no saccades')
            ax.set_ylabel('heading (°)')
            ax.set_title(f'Test {i}')
            if i == 0:
                ax.legend(loc='upper right')
        axes[-1, 0].set_xlabel('time (s)')
        fig.suptitle(
            f'camera_heading_unwrapped vs no_saccades\n{os.path.basename(h5_file)}',
            fontsize=10,
        )
        fig.tight_layout()

        out = tmp_path / 'unwrapped_vs_no_saccades.png'
        fig.savefig(str(out), dpi=100)
        plt.close(fig)
        assert out.exists(), "Plot file was not created"
