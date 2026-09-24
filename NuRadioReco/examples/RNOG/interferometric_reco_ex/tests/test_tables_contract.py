"""Format and physics contract of the travel-time tables (numpy only, fast)."""

import os

import numpy as np
import pytest

from synthetic import HPOL_CHANNELS, VPOL_CHANNELS
from conftest import STATION

C_NS = 0.299792458
N_SURFACE = 1.27
N_DEEP = 1.78
RAY_TYPES = ['direct', 'refracted', 'reflected']


def _load(table_dir, ch, suffix=''):
    path = os.path.join(table_dir, f'station{STATION}', f'st{STATION}_ch{ch}_rz_table{suffix}.npz')
    if not os.path.isfile(path):
        pytest.skip(f'missing {path}')
    return np.load(path)


@pytest.mark.parametrize('ch', VPOL_CHANNELS + HPOL_CHANNELS)
def test_grid_and_keys(table_dir, ch):
    """Every table has the three keys, a uniform ascending grid, and the matching data shape."""
    d = _load(table_dir, ch)
    assert set(d.files) == {'r_range_vals', 'z_range_vals', 'data'}
    r, z, data = d['r_range_vals'], d['z_range_vals'], d['data']
    assert data.shape == (len(r), len(z))
    for axis in (r, z):
        steps = np.diff(axis)
        assert np.all(steps > 0)
        assert np.allclose(steps, steps[0])
    assert r[0] == 0
    assert z[-1] >= 0


@pytest.mark.parametrize('ch', VPOL_CHANNELS)
def test_values_are_physical(table_dir, ch, ant_locs):
    """Finite entries are positive and bounded by straight-line propagation limits."""
    d = _load(table_dir, ch)
    r, z, data = d['r_range_vals'], d['z_range_vals'], d['data']
    rr, zz = np.meshgrid(r, z, indexing='ij')
    dist = np.hypot(rr, zz - ant_locs[ch][2])
    finite = np.isfinite(data)
    assert finite.any()
    assert np.all(data[finite] > 0)
    lower = dist * N_SURFACE / C_NS * 0.999
    upper = dist * N_DEEP / C_NS * 1.5
    assert np.all(data[finite] >= lower[finite])
    assert np.all(data[finite] <= upper[finite])


@pytest.mark.parametrize('ch', VPOL_CHANNELS)
def test_in_ice_tables_have_no_solution_above_surface(table_dir, ch):
    """In-ice tables leave the z >= 0 rows NaN by construction."""
    d = _load(table_dir, ch)
    above = d['z_range_vals'] >= 0
    assert np.all(np.isnan(d['data'][:, above]))


@pytest.mark.parametrize('ch', VPOL_CHANNELS)
def test_combined_is_min_over_ray_types(table_dir, ch):
    """The combined table equals the NaN-aware minimum of the per-ray-type tables, bit for bit."""
    combined = _load(table_dir, ch)['data']
    stack = np.stack([_load(table_dir, ch, f'_{t}')['data'] for t in RAY_TYPES])
    with np.errstate(all='ignore'):
        expected = np.nanmin(np.where(np.isnan(stack), np.inf, stack), axis=0)
    expected[np.isinf(expected)] = np.nan
    assert np.array_equal(np.isnan(combined), np.isnan(expected))
    m = np.isfinite(combined)
    assert np.array_equal(combined[m], expected[m])


@pytest.mark.parametrize('ch', VPOL_CHANNELS)
def test_solution_ordered_tables(table_dir, ch):
    """solution_0 is the fastest arrival (equals the combined table) and solution_1 the next one."""
    combined = _load(table_dir, ch)['data']
    sol0 = _load(table_dir, ch, '_solution_0')['data']
    sol1 = _load(table_dir, ch, '_solution_1')['data']
    m = np.isfinite(combined)
    assert np.array_equal(np.isnan(sol0), np.isnan(combined))
    assert np.array_equal(sol0[m], combined[m])
    both = np.isfinite(sol1)
    assert np.all(sol1[both] >= sol0[both])
    assert both.sum() < m.sum()


def test_search_volume_coverage(table_dir):
    """Deep channels have solutions on at least 90 percent of the reference search volume (rho 1 to 250 m, z -100 to 0 m).

    Documents the known shadow-zone gaps: the shallow VPols (6, 7) are allowed a
    larger NaN fraction. A regression here means a table regeneration lost coverage.
    """
    for ch in VPOL_CHANNELS:
        d = _load(table_dir, ch)
        r, z, data = d['r_range_vals'], d['z_range_vals'], d['data']
        sel = data[np.ix_((r >= 1) & (r <= 250), (z >= -100) & (z <= 0))]
        nan_frac = np.isnan(sel).mean()
        limit = 0.25 if ch in (6, 7) else 0.10
        assert nan_frac < limit, f'ch{ch}: NaN fraction {nan_frac:.3f} exceeds {limit}'
