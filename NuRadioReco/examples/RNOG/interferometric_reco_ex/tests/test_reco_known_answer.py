"""Known-answer reconstruction of synthetic in-ice sources built from the tables.

Two kinds of assertion. The exact-recovery tests use geometries on which the reference
configuration was measured to recover the source to better than 0.05 degree at SNR 50;
they prove the search machinery, tables and geometry are consistent. The
accuracy-summary test gates the distribution over the 16-source characterisation set
at SNR 20 at the measured level with margin (measured: median 1.05 degree, 68th
percentile 1.69, 88 percent within 3 degrees), so a regression fails while an
improvement passes. On this set the reference search has known limitations: several
geometries settle 1 to 3.5 degrees off at any SNR, one deep source 90 m out lands
about 30 degrees off, and one source at the phased-array depth 200 m out flips to a
mirror azimuth at SNR 50; those enter the summary, not the exact tests.
"""

import numpy as np
import pytest

from conftest import STATION, reference_config, rng_sources
from synthetic import (HPOL_CHANNELS, VPOL_CHANNELS, TravelTimeTables, angular_separation,
                       cylindrical_to_enu, make_event)
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

EXACT_TOL_DEG = 0.2
EXACT_SNR = 50.0
EXACT_CORR_MIN = 0.3
EXACT_SOURCES = [
    pytest.param((30.0, 300.0, -5.0), id='near_shallow'),
    pytest.param((80.0, 120.0, -40.0), id='mid_depth'),
    pytest.param((66.5, 246.2, -32.8), id='mid_west'),
    pytest.param((43.6, 332.2, -30.2), id='near_north'),
    pytest.param((113.3, 235.5, -49.9), id='far_southwest'),
]
SUMMARY_SOURCES = [(50.0, 30.0, -20.0), (120.0, 200.0, -60.0), (200.0, 100.0, -90.0), (30.0, 300.0, -5.0),
                   (60.0, 45.0, -2.0), (100.0, 60.0, -45.0), (90.0, 10.0, -50.0), (80.0, 120.0, -40.0)]
SUMMARY_SNR = 20.0
SUMMARY_GATES = dict(median_deg=1.4, p68_deg=2.2, frac_lt_3deg=0.75, frac_lt_1deg=0.35)


def _run(reco, det, config, src, tables, pa, snr, seed=1):
    src_enu = cylindrical_to_enu(*src, pa)
    evt, stn, _ = make_event(det, STATION, src_enu, config['channels'], tables, snr=snr, seed=seed)
    res = reco.run(evt, stn, det, config)
    return res, angular_separation((res['rho'], res['phi'], res['z']), src, pa)


@pytest.mark.slow
@pytest.mark.parametrize('src', EXACT_SOURCES)
def test_exact_recovery(reco, det, base_config, tables, pa, src):
    """Geometries the reference configuration resolves exactly stay exact."""
    res, sep = _run(reco, det, base_config, src, tables, pa, EXACT_SNR)
    assert sep < EXACT_TOL_DEG, (src, sep, {k: res[k] for k in ('rho', 'phi', 'z', 'max_corr')})
    assert abs(res['rho'] - src[0]) < 2.0 and abs(res['z'] - src[2]) < 2.0, res
    assert res['max_corr'] > EXACT_CORR_MIN, res['max_corr']


@pytest.mark.slow
def test_accuracy_summary_at_snr20(reco, det, base_config, tables, pa):
    """Distribution over the characterisation set is no worse than the measured level."""
    sources = SUMMARY_SOURCES + rng_sources(8, 20260916)
    seps = np.array([_run(reco, det, base_config, src, tables, pa, SUMMARY_SNR, seed=5)[1] for src in sources])
    summary = dict(median_deg=float(np.median(seps)), p68_deg=float(np.percentile(seps, 68)),
                   frac_lt_3deg=float(np.mean(seps < 3)), frac_lt_1deg=float(np.mean(seps < 1)))
    print('accuracy summary at SNR 20:', summary)
    assert summary['median_deg'] <= SUMMARY_GATES['median_deg'], summary
    assert summary['p68_deg'] <= SUMMARY_GATES['p68_deg'], summary
    assert summary['frac_lt_3deg'] >= SUMMARY_GATES['frac_lt_3deg'], summary
    assert summary['frac_lt_1deg'] >= SUMMARY_GATES['frac_lt_1deg'], summary


@pytest.mark.slow
def test_multiray_solution_ordered_recovers_source(det, table_dir, tables, pa):
    """The two-table multiray scheme also recovers an exact-recovery source."""
    cfg = reference_config(table_dir, multi_ray_types=True, table_scheme='solution_ordered')
    reco = InterferometricReco3D()
    reco.begin(STATION, cfg, det)
    src = (80.0, 120.0, -40.0)
    res, sep = _run(reco, det, cfg, src, tables, pa, EXACT_SNR)
    assert sep < 1.0, (res, sep)
    assert res['max_corr'] > EXACT_CORR_MIN, res


@pytest.mark.slow
def test_polarization_groups_produce_hpol_result(det, table_dir, pa):
    """With polarization groups the VPol result is primary and an HPol result is attached."""
    channels = VPOL_CHANNELS + HPOL_CHANNELS
    cfg = reference_config(table_dir, channels=channels,
                           polarization_groups={'vpol': VPOL_CHANNELS, 'hpol': HPOL_CHANNELS},
                           hpol_weight_scale=1.0)
    all_tables = TravelTimeTables(table_dir, STATION, channels)
    reco = InterferometricReco3D()
    reco.begin(STATION, cfg, det)
    src = (80.0, 120.0, -40.0)
    src_enu = cylindrical_to_enu(*src, pa)
    evt, stn, _ = make_event(det, STATION, src_enu, channels, all_tables, snr=EXACT_SNR, seed=3)
    res = reco.run(evt, stn, det, cfg)
    assert 'rho_hpol' in res and 'max_corr_hpol' in res
    assert angular_separation((res['rho'], res['phi'], res['z']), src, pa) < 1.0
    assert angular_separation((res['rho_hpol'], res['phi_hpol'], res['z_hpol']), src, pa) < 5.0


@pytest.mark.slow
def test_saved_peaks_are_ordered_and_primary_matches(reco, det, base_config, tables, pa):
    """peak_0 equals the primary result and saved peak correlations are non-increasing."""
    res, _ = _run(reco, det, base_config, (80.0, 120.0, -40.0), tables, pa, EXACT_SNR)
    assert res['peak_0_rho'] == res['rho']
    assert res['peak_0_corr'] == res['max_corr']
    corrs = [res[f'peak_{i}_corr'] for i in range(3) if f'peak_{i}_corr' in res]
    assert corrs == sorted(corrs, reverse=True)
