"""
Shared helpers for the regression tests that compare freshly simulated output against checked-in
reference files (e.g. ``NuRadioMC/test/SingleEvents/T04validate_allmost_equal.py``,
``NuRadioMC/test/examples/validate_radio_emitter_allmost_equal.py``).

A validation is a list of `check` specifications passed to `validate_hdf5_files`, which runs all
of them, reports every mismatch and exits non-zero if any failed.
"""
import sys
import h5py
import numpy as np
from numpy import testing

from NuRadioReco.utilities import units

#: Absolute tolerance for timing quantities (travel times, trigger times, trace start times).
#: They inherit last-bit differences from the GSL numerical integration in the C++ raytracer
#: (``get_travel_time()`` in ``NuRadioMC/SignalProp/CPPAnalyticRayTracing/analytic_raytracing.cpp``,
#: epsrel=1e-6), which vary across GSL versions/platforms (observed up to ~9e-5 ns) independent of
#: any code change. This noise floor is absolute, i.e., it does not scale with the (arbitrarily
#: large) time, hence an absolute instead of a relative tolerance.
TIME_ATOL = 1e-3 * units.ns

#: Default relative tolerance for floating point quantities.
DEFAULT_RTOL = 5e-4


def check(keys, location='top', rtol=0, atol=0):
    """
    Specify a comparison for `validate_hdf5_files`.

    Parameters
    ----------
    keys: list of str
        Names of the datasets (or attributes) to compare.
    location: str (default: 'top')
        Where the keys live: 'attrs' (file attributes), 'top' (top-level datasets) or
        'station' (datasets of the station group).
    rtol, atol: float (default: 0)
        Tolerances as in `numpy.testing.assert_allclose`. If both are 0, the values have to be
        exactly equal (this also works for non-numeric values).
    """
    if location not in ('attrs', 'top', 'station'):
        raise ValueError(f"Unknown location '{location}'")
    return dict(keys=keys, location=location, rtol=rtol, atol=atol)


def _get(fin, key, location, station):
    if location == 'attrs':
        return np.array(fin.attrs[key])
    if location == 'station':
        return np.array(fin[station][key])
    return np.array(fin[key])


def _get_event_group_ids(fin, station):
    # different NuRadioMC output formats store event_group_ids either at the top level or
    # nested under the station group
    try:
        return np.array(fin['event_group_ids'])
    except KeyError:
        return np.array(fin[station]['event_group_ids'])


def compare_key(fin1, fin2, key, location='top', rtol=0, atol=0, station='station_101'):
    """
    Compare one key of two open HDF5 files (see `check` for the parameters).

    For tolerance comparisons, NaN is treated as 0 since both are used as "no solution"
    sentinels, depending on code version.

    Returns
    -------
    error: str or None
        Description of the mismatch, None if the values agree.
    """
    try:
        arr1 = _get(fin1, key, location, station)
        arr2 = _get(fin2, key, location, station)
    except KeyError as e:
        return f"{location} '{key}' missing: {e}"

    exact = rtol == 0 and atol == 0
    try:
        if exact:
            testing.assert_array_equal(arr1, arr2)
        else:
            arr1 = np.nan_to_num(arr1.astype(float), nan=0)
            arr2 = np.nan_to_num(arr2.astype(float), nan=0)
            testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol)
    except AssertionError as e:
        msg = f"{location} '{key}' differs (rtol={rtol}, atol={atol}):{e}"
        if location != 'attrs' and arr1.shape == arr2.shape and arr1.ndim:
            # report which events (first axis) differ
            bad = (arr1 != arr2) if exact else ~np.isclose(arr1, arr2, rtol=rtol, atol=atol)
            bad_events = np.nonzero(bad.reshape(len(bad), -1).any(axis=1))[0]
            msg += f"\nDiffering event indices: {bad_events}"
            try:
                gids = _get_event_group_ids(fin1, station)
                if len(gids) == len(arr1):
                    msg += f"\nCorresponding event group ids: {gids[bad_events]}"
            except KeyError:
                pass
        return msg

    return None


def validate_hdf5_files(file1, file2, checks, station='station_101', dump_on_failure=False):
    """
    Run all `checks` comparing the HDF5 output `file1` with the reference `file2`.

    Prints every mismatch and exits with code 1 if any check failed.

    Parameters
    ----------
    file1, file2: str
        Paths to the output and the reference file.
    checks: list of dict
        Specifications created with `check`.
    station: str (default: 'station_101')
        Name of the station group for checks with ``location='station'``.
    dump_on_failure: bool (default: False)
        If True, print the content of both files on failure.
    """
    print(f"Testing the files {file1} and {file2} for (almost) equality")
    errors = []
    with h5py.File(file1, 'r') as fin1, h5py.File(file2, 'r') as fin2:
        for spec in checks:
            for key in spec['keys']:
                error = compare_key(fin1, fin2, key, spec['location'], spec['rtol'], spec['atol'], station)
                if error is not None:
                    print(f"\n{error}")
                    errors.append(key)

    if not errors:
        print(f"The files {file1} and {file2} are (almost) identical.")
        return

    print(f"\n{len(errors)} key(s) differ: {errors}")
    if dump_on_failure:
        from NuRadioMC.utilities.dump_hdf5 import dump
        for fn in (file1, file2):
            print(f"file {fn}")
            dump(fn)
    sys.exit(1)
