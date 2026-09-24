"""Golden-master regression on fixed synthetic events.

The reference file `golden/st23_synthetic_golden.json` holds the reconstruction of
eight seeded synthetic events at SNR 30 (stable maxima), produced by the reconstruction
code before the air-ice changes. Two checks run against it:

* bit-tight: rho, phi, z and max_corr agree to 1e-6 (any numerical change shows up);
* accuracy: the angular separation to truth is no worse than the reference by more
  than 0.1 degree per event (a change that is numerically different but not worse
  fails only the first check, which is the signal to inspect and regenerate).

Set RECO3D_REGEN_GOLDEN=1 to rewrite the reference from the currently imported code.
"""

import json
import os

import numpy as np
import pytest

from conftest import STATION, rng_sources
from synthetic import VPOL_CHANNELS, angular_separation, cylindrical_to_enu, make_event

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden', 'st23_synthetic_golden.json')
SEED = 20260916
N_EVENTS = 8
SNR = 30.0
TIGHT = 1e-6
ACCURACY_MARGIN_DEG = 0.1


def _reconstruct_all(reco, det, config, tables, pa):
    rows = []
    for i, src in enumerate(rng_sources(N_EVENTS, SEED)):
        evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*src, pa), VPOL_CHANNELS,
                                 tables, snr=SNR, seed=SEED + i)
        res = reco.run(evt, stn, det, config)
        rows.append({
            'source': [float(v) for v in src],
            'rho': float(res['rho']), 'phi': float(res['phi']), 'z': float(res['z']),
            'max_corr': float(res['max_corr']),
            'angular_separation': angular_separation((res['rho'], res['phi'], res['z']), src, pa),
        })
    return rows


@pytest.mark.slow
def test_golden_master(reco, det, base_config, tables, pa):
    """Reproduce the stored reconstruction of the seeded synthetic events."""
    rows = _reconstruct_all(reco, det, base_config, tables, pa)
    if os.environ.get('RECO3D_REGEN_GOLDEN') == '1':
        payload = {'seed': SEED, 'n_events': N_EVENTS, 'snr': SNR, 'station': STATION,
                   'events': rows}
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, 'w') as f:
            json.dump(payload, f, indent=1)
            f.write('\n')
        pytest.skip(f'golden reference written to {GOLDEN}')
    if not os.path.isfile(GOLDEN):
        pytest.skip('no golden reference; run once with RECO3D_REGEN_GOLDEN=1 on the reference version of the code')
    with open(GOLDEN) as f:
        ref = json.load(f)
    assert ref['seed'] == SEED and ref['n_events'] == N_EVENTS and ref['snr'] == SNR
    worse = []
    drift = []
    for i, (row, exp) in enumerate(zip(rows, ref['events'])):
        assert row['source'] == exp['source']
        for k in ('rho', 'phi', 'z', 'max_corr'):
            if abs(row[k] - exp[k]) > TIGHT:
                drift.append((i, k, exp[k], row[k]))
        if row['angular_separation'] > exp['angular_separation'] + ACCURACY_MARGIN_DEG:
            worse.append((i, exp['angular_separation'], row['angular_separation']))
    assert not worse, f'accuracy regressed on events (index, reference deg, now deg): {worse}'
    assert not drift, f'numerical drift from the golden reference (index, key, reference, now): {drift}'


@pytest.mark.slow
def test_golden_reference_accuracy_summary(reco, det, base_config, tables, pa):
    """The seeded set stays at the measured level (median 1.14 degree, 7 of 8 within 3 degrees)."""
    seps = np.array([r['angular_separation'] for r in _reconstruct_all(reco, det, base_config, tables, pa)])
    assert np.median(seps) < 1.5, seps
    assert np.mean(seps < 3.0) >= 0.75, seps
