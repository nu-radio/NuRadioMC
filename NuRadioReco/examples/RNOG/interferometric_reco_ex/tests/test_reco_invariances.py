"""Invariances the reconstruction must satisfy on synthetic events."""

import pytest

from conftest import STATION
from synthetic import (VPOL_CHANNELS, angular_separation, cylindrical_to_enu, make_event,
                       make_noise_event)

SRC = (80.0, 120.0, -40.0)
SNR = 50.0
TIGHT = dict(rho=1e-4, phi=1e-4, z=1e-4, max_corr=1e-6)
LOOSE = dict(rho=0.5, phi=0.05, z=0.5, max_corr=1e-3)


def _reco_result(reco, det, config, evt, stn):
    res = reco.run(evt, stn, det, config)
    return {k: res[k] for k in ('rho', 'phi', 'z', 'max_corr')}


def _assert_same(a, b, tol):
    for k, t in tol.items():
        assert abs(a[k] - b[k]) <= t, (k, a[k], b[k])


@pytest.mark.slow
def test_global_time_shift_invariance(reco, det, base_config, tables, pa):
    """Shifting every trace start time by the same amount leaves the result unchanged."""
    src_enu = cylindrical_to_enu(*SRC, pa)
    e0, s0, _ = make_event(det, STATION, src_enu, VPOL_CHANNELS, tables, seed=7, snr=SNR)
    e1, s1, _ = make_event(det, STATION, src_enu, VPOL_CHANNELS, tables, seed=7, snr=SNR,
                           trace_start_time=123.456)
    _assert_same(_reco_result(reco, det, base_config, e0, s0),
                 _reco_result(reco, det, base_config, e1, s1), TIGHT)


@pytest.mark.slow
def test_amplitude_scale_invariance(reco, det, base_config, tables, pa):
    """Scaling all traces by a constant leaves the normalised correlation result unchanged."""
    src_enu = cylindrical_to_enu(*SRC, pa)
    e0, s0, _ = make_event(det, STATION, src_enu, VPOL_CHANNELS, tables, seed=7, snr=SNR)
    e1, s1, _ = make_event(det, STATION, src_enu, VPOL_CHANNELS, tables, seed=7, snr=SNR, amplitude=25.0)
    _assert_same(_reco_result(reco, det, base_config, e0, s0),
                 _reco_result(reco, det, base_config, e1, s1), LOOSE)


@pytest.mark.slow
def test_channel_insertion_order_invariance(reco, det, base_config, tables, pa):
    """The order in which channels are added to the station does not change the result."""
    src_enu = cylindrical_to_enu(*SRC, pa)
    e0, s0, _ = make_event(det, STATION, src_enu, VPOL_CHANNELS, tables, seed=7, snr=SNR)
    e1, s1, _ = make_event(det, STATION, src_enu, VPOL_CHANNELS, tables, seed=7, snr=SNR,
                           channel_order=list(reversed(VPOL_CHANNELS)))
    _assert_same(_reco_result(reco, det, base_config, e0, s0),
                 _reco_result(reco, det, base_config, e1, s1), LOOSE)


@pytest.mark.slow
def test_azimuth_rotation_covariance(reco, det, base_config, tables, pa):
    """Rotating the source in azimuth rotates the reconstructed azimuth by the same angle."""
    results = []
    for dphi in (0.0, 90.0):
        src = (SRC[0], (SRC[1] + dphi) % 360.0, SRC[2])
        e, s, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS, tables, seed=7, snr=SNR)
        res = _reco_result(reco, det, base_config, e, s)
        assert angular_separation((res['rho'], res['phi'], res['z']), src, pa) < 1.5, (src, res)
        results.append(res)
    dphi_reco = (results[1]['phi'] - results[0]['phi'] + 180.0) % 360.0 - 180.0
    assert abs(dphi_reco - 90.0) < 1.5, results


@pytest.mark.slow
def test_snr_degrades_gracefully(reco, det, base_config, tables, pa):
    """Accuracy degrades gracefully with SNR and the correlation drops with it."""
    src_enu = cylindrical_to_enu(*SRC, pa)
    corr = {}
    for snr in (50.0, 8.0):
        e, s, _ = make_event(det, STATION, src_enu, VPOL_CHANNELS, tables, seed=11, snr=snr)
        res = _reco_result(reco, det, base_config, e, s)
        corr[snr] = res['max_corr']
        tol = 0.5 if snr > 10 else 5.0
        assert angular_separation((res['rho'], res['phi'], res['z']), SRC, pa) < tol, (snr, res)
    assert corr[50.0] > corr[8.0]


@pytest.mark.slow
def test_noise_only_event_has_low_correlation(reco, det, base_config):
    """Pure noise must not produce a confident peak."""
    evt, stn = make_noise_event(STATION, VPOL_CHANNELS, seed=5)
    res = _reco_result(reco, det, base_config, evt, stn)
    assert res['max_corr'] < 0.3, res
