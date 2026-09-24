"""Shared fixtures for the 3D interferometric reconstruction test suite.

Environment variables:
    RECO3D_TEST_TABLES: root of the in-ice travel-time tables (default: NURADIO_TABLE_DIR).
    RECO3D_TEST_DETECTOR_FILE: exported detector description to use instead of the RNO-G
        database, e.g. on nodes without network access.
    RECO3D_TEST_AIRICE_TABLES: root of the air-ice tables (no default).
"""

import datetime
import logging
import os

import numpy as np
import pytest

from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D

from synthetic import VPOL_CHANNELS, TravelTimeTables, antenna_locations, pa_center

STATION = 23
DETECTOR_DATE = datetime.datetime(2022, 10, 1)

logging.getLogger().setLevel(logging.WARNING)


def pytest_configure(config):
    """Register the markers used by this suite."""
    config.addinivalue_line('markers', 'slow: needs the numba kernels (minutes on first run)')
    config.addinivalue_line('markers', 'airice: needs the air-ice tracer and propagator')


def reference_config(table_dir, channels=None, **overrides):
    """Reference reconstruction parameters: the settings of the cosmic-ray search reconstruction.

    Only keys the module consumes are included; pipeline-owned keys are left out.
    """
    cfg = dict(
        channels=list(channels or VPOL_CHANNELS),
        hierarchical=True,
        multi_ray_types=False,
        multiray_combo_mode='grouped',
        coarse_limits=[1, 250, 0, 360, -100, 0],
        coarse_n_rho=30,
        coarse_n_z=100,
        coarse_step_sizes=[0, 3, 0],
        coarse_n_peaks=7,
        coarse_peak_separation=[50, 15, 50],
        refine_window=[15, 3, 15],
        refine_step_sizes=[1, 0.3, 1],
        limits=[1, 250, 0, 360, -100, 0],
        time_delay_tables=table_dir,
        multiray_table_name_pattern='st{station_id}_ch{ch}_rz_table_{ray_type}.npz',
        hilbert_envelope_mode=None,
        apply_hann_window=True,
        snr_pair_weighting=True,
        correlation_normalization='energy',
        interp_method='linear',
        n_peaks_save=3,
    )
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope='session')
def table_dir():
    """Root of the in-ice travel-time tables, or skip when absent."""
    path = os.environ.get('RECO3D_TEST_TABLES', os.environ.get('NURADIO_TABLE_DIR', ''))
    if not path or not os.path.isdir(os.path.join(path, f'station{STATION}')):
        pytest.skip(f'tables not found under {path}')
    return path


@pytest.fixture(scope='session')
def det():
    """Station 23 detector description at the 2022 epoch, from a file if given, else the database."""
    path = os.environ.get('RECO3D_TEST_DETECTOR_FILE')
    if path:
        if not os.path.isfile(path):
            pytest.skip(f'detector file not found: {path}')
        d = rnog_detector.Detector(detector_file=path, select_stations=STATION,
                                   log_level=logging.WARNING)
    else:
        d = rnog_detector.Detector(select_stations=STATION, log_level=logging.WARNING)
    d.update(DETECTOR_DATE)
    return d


@pytest.fixture(scope='session')
def ant_locs(det):
    """Channel -> [x_rel, y_rel, z_abs] for station 23."""
    return antenna_locations(det, STATION)


@pytest.fixture(scope='session')
def pa(ant_locs):
    """PA reference point of station 23."""
    return pa_center(ant_locs)


@pytest.fixture(scope='session')
def tables(table_dir):
    """Combined travel-time tables for the VPol channels."""
    return TravelTimeTables(table_dir, STATION, VPOL_CHANNELS)


@pytest.fixture(scope='session')
def reco(det, table_dir):
    """Reconstruction object initialised with the reference VPol configuration."""
    r = InterferometricReco3D()
    r.begin(STATION, reference_config(table_dir), det)
    return r


@pytest.fixture(scope='session')
def base_config(table_dir):
    """Reference VPol configuration dictionary."""
    return reference_config(table_dir)


def rng_sources(n, seed):
    """Draw n (rho, phi, z) sources inside the reference search volume, away from its edges."""
    rng = np.random.default_rng(seed)
    rho = rng.uniform(20.0, 160.0, n)
    phi = rng.uniform(0.0, 360.0, n)
    z = rng.uniform(-90.0, -3.0, n)
    return list(zip(rho, phi, z))
