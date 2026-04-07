import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="skvideo")
import numpy as np
import pytest
from magno_tracker.tracking import resolve_colors, get_grid_vals, omit_wrapping, bootstrap_ci
import matplotlib


def test_resolve_colors_default():
    # Default color fill
    arr = resolve_colors(None, None, [1, 2], [3, 4], default_color='r')
    assert arr.shape == (2, 2, 3)
    expected = np.full((2, 2, 3), matplotlib.colors.to_rgb('r'))
    np.testing.assert_allclose(arr, expected)


def test_resolve_colors_colormap():
    # Use matplotlib colormaps
    # row_cmap = matplotlib.cm.get_cmap('Blues')
    # col_cmap = matplotlib.cm.get_cmap('Reds')
    row_cmap = matplotlib.colormaps['Blues']
    col_cmap = matplotlib.colormaps['Reds']
    arr = resolve_colors(row_cmap, col_cmap, [0, 1], [0, 1])
    assert arr.shape == (2, 2, 3)
    # Should blend row and col colors
    assert np.all(arr >= 0) and np.all(arr <= 1)


def test_resolve_colors_list_length_error():
    # Should raise ValueError if list length mismatches
    with pytest.raises(ValueError):
        resolve_colors(['r'], ['b', 'g'], [1, 2], [3, 4])


def test_get_grid_vals_numeric():
    class DummyExp:
        def query(self, output, subset=None):
            return np.array([1, 2, np.nan, 2, 1])
    vals = get_grid_vals(DummyExp(), 'foo', {})
    assert set(vals) == {1, 2}


def test_get_grid_vals_string():
    class DummyExp:
        def query(self, output, subset=None):
            return np.array(['a', 'b', 'nan', 'a'])
    vals = get_grid_vals(DummyExp(), 'foo', {})
    assert set(vals) == {'a', 'b'}


def test_omit_wrapping_1d():
    arr = np.array([0, 0.1, 3.5, 0.2])
    out = omit_wrapping(arr, threshold=1.0)
    # Should insert NaN at discontinuity (index 2)
    assert np.isnan(out[2])


def test_omit_wrapping_2d():
    arr = np.array([[0, 0.1, 3.5, 0.2], [1, 1.1, 1.2, 1.3]])
    out = omit_wrapping(arr, threshold=1.0)
    assert out.shape[0] == 2
    # Should insert NaN at discontinuity (index 2 in first row)
    assert np.isnan(out[0, 2])


def test_bootstrap_ci_mean():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100,))
    low, high = bootstrap_ci(data, np.nanmean, confidence=0.68, n_boot=200)
    assert low < np.mean(data) < high


def test_bootstrap_ci_return_bootstrap():
    data = np.arange(10)
    low, high, boot_stats = bootstrap_ci(data, np.nanmean, n_boot=100, return_bootstrap=True)
    assert boot_stats.shape[0] == 100
    assert np.all((boot_stats >= data.min()) & (boot_stats <= data.max()))
