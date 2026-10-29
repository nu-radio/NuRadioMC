"""
Shared helpers for the regression tests that compare freshly simulated output against checked-in
reference files (e.g. ``NuRadioMC/test/SingleEvents/T04validate_allmost_equal.py``,
``NuRadioMC/test/examples/validate_radio_emitter_allmost_equal.py``).

A validation is a list of `check` specifications passed to `validate_hdf5_files`, which runs all
of them, reports every mismatch and exits non-zero if any failed. `validate_nur_files` does the
same for the traces and channel/station parameters of ``.nur`` files.
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

#: Default relative tolerance for channel/station parameters in ``.nur`` files (e.g. from the
#: ``channelSignalReconstructor``): tiny platform-dependent trace differences are amplified for
#: low-amplitude channels (observed up to 7.5e-4).
PARAMETER_RTOL = 1e-3

#: Channel/station parameters (enum names) that depend on the noise in a trace. In noiseless
#: simulations, this "noise" is the numerical floor of the trace, which differs by orders of
#: magnitude between platforms and simulation paths (observed: ``noise_rms`` up to a factor 6).
#: Hence, these are compared separately from parameters that depend only on the signal.
NOISE_SENSITIVE_PARAMETERS = ('noise_rms', 'SNR', 'root_power_ratio', 'entropy')


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


def _get_event_group_ids(fin, location, station, n_rows):
    # event group id of each row: station keys have one row per event or per shower
    candidates = [fin['event_group_ids']]
    if location == 'station':
        group = fin[station]
        candidates = [group[k] for k in ('event_group_ids', 'event_group_id_per_shower') if k in group] + candidates
    for ids in candidates:
        if len(ids) == n_rows:
            return np.array(ids)
    return None


def _isnan(arr):
    return np.isnan(arr) if arr.dtype.kind in 'fc' else np.zeros(arr.shape, dtype=bool)


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
            # report which events (first axis) differ; NaN in both files is equal (as in assert_array_equal)
            if exact:
                bad = (arr1 != arr2) & ~(_isnan(arr1) & _isnan(arr2))
            else:
                bad = ~np.isclose(arr1, arr2, rtol=rtol, atol=atol)
            bad_events = np.nonzero(bad.reshape(len(bad), -1).any(axis=1))[0]
            msg += f"\nDiffering event indices: {bad_events}"
            try:
                gids = _get_event_group_ids(fin1, location, station, len(arr1))
                if gids is not None:
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


def _read_nur_file(file):
    """
    Returns all channel traces (concatenated), their trace start times and a dict
    {(run, event, station, channel or None): {parameter: value}} of all station/channel parameters.
    """
    from NuRadioReco.modules.io.eventReader import eventReader  # only needed here, not for the HDF5 validators

    reader = eventReader()
    reader.begin(file)
    traces, start_times, parameters = [], [], {}
    for event in reader.run():
        for station in event.get_stations():
            key = (event.get_run_number(), event.get_id(), station.get_id())
            parameters[key + (None,)] = dict(station._parameters)
            for channel in station.iter_channels(sorted=True):
                traces.append(channel.get_trace())
                start_times.append(channel.get_trace_start_time())
                parameters[key + (channel.get_id(),)] = dict(channel._parameters)
    return np.concatenate(traces), np.array(start_times), parameters


def _relative_difference(value1, value2):
    """ Maximum relative difference |value1 - value2| / |value2| (0 for identical values, including inf and nan) """
    value1, value2 = np.asarray(value1, dtype=float), np.asarray(value2, dtype=float)
    identical = (value1 == value2) | (np.isnan(value1) & np.isnan(value2))
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(identical, 0, np.abs(value1 - value2) / np.abs(value2))
    return float(np.max(np.nan_to_num(rel, nan=np.inf), initial=0))


def _compare_parameter(value1, value2, rtol, name, parameter, largest_differences):
    """
    Returns a list of error messages for all (nested) parameter values which differ (only keys are
    compared if ``rtol`` is None). The largest relative difference of each parameter (and where it
    occurs) is stored in ``largest_differences``.
    """
    if isinstance(value1, dict):
        if value1.keys() != value2.keys():
            return [f"{name}: keys differ ({sorted(value1)} vs. {sorted(value2)})"]
        return [error for key in value1 for error in _compare_parameter(
            value1[key], value2[key], rtol, f"{name}/{key}", f"{parameter}/{key}", largest_differences)]
    rel = _relative_difference(value1, value2)
    if rel > largest_differences.get(parameter, (-1, None))[0]:
        largest_differences[parameter] = (rel, name)
    if rtol is None:
        return []
    try:
        testing.assert_allclose(value1, value2, rtol=rtol, atol=0, err_msg=f"Parameter {name} differs")
    except AssertionError as e:
        return [str(e)]
    return []


def validate_nur_files(file1, file2, precision=7, parameter_rtol=PARAMETER_RTOL, noise_parameter_rtol=None):
    """
    Compare the traces, trace start times and all channel/station parameters of the ``.nur`` file
    `file1` with the reference `file2`.

    Prints every mismatch and the largest relative difference of each parameter, and exits with
    code 1 if any check failed.

    Parameters
    ----------
    file1, file2: str
        Paths to the output and the reference file.
    precision: int (default: 7)
        Number of decimals to which the traces have to agree (as in `numpy.testing.assert_almost_equal`).
    parameter_rtol: float (default: `PARAMETER_RTOL`)
        Relative tolerance for parameters which depend only on the signal.
    noise_parameter_rtol: float or None (default: None)
        Relative tolerance for `NOISE_SENSITIVE_PARAMETERS`. If None (e.g. for noiseless
        simulations), only their presence is checked.
    """
    print(f"Testing the files {file1} and {file2} for equality")
    traces_1, start_times_1, parameters_1 = _read_nur_file(file1)
    traces_2, start_times_2, parameters_2 = _read_nur_file(file2)
    # collect all differences (instead of stopping at the first one) to see the full picture in a single run
    errors = []

    if traces_1.shape != traces_2.shape:
        errors.append(f"Traces have different shapes: {traces_1.shape} vs. {traces_2.shape}")
    else:
        print(f"Maximum difference between traces [mV]: {np.max(np.abs(traces_1 - traces_2)) / units.mV}")
        try:
            testing.assert_almost_equal(traces_1, traces_2, decimal=precision)
        except AssertionError as e:
            errors.append(f"Traces differ:{e}")

    # the trace start times depend on the raytraced travel times
    try:
        testing.assert_allclose(start_times_1, start_times_2, rtol=0, atol=TIME_ATOL)
    except (AssertionError, ValueError) as e:
        errors.append(f"Trace start times differ:{e}")

    largest_differences = {}
    if parameters_1.keys() != parameters_2.keys():
        errors.append("Files contain different events/stations/channels")
    for key in [key for key in parameters_1 if key in parameters_2]:
        if parameters_1[key].keys() != parameters_2[key].keys():
            errors.append(f"{key}: parameters differ ({sorted(map(str, parameters_1[key]))} vs. "
                          f"{sorted(map(str, parameters_2[key]))})")
            continue
        for parameter in parameters_1[key]:
            noise_sensitive = parameter.name in NOISE_SENSITIVE_PARAMETERS
            errors += _compare_parameter(
                parameters_1[key][parameter], parameters_2[key][parameter],
                noise_parameter_rtol if noise_sensitive else parameter_rtol,
                f"{key}/{parameter}", ('* ' if noise_sensitive else '  ') + str(parameter), largest_differences)

    # the largest relative difference of each parameter shows how tight the tolerance can be
    print("Largest relative difference per parameter (* = noise sensitive):")
    for parameter, (rel, name) in sorted(largest_differences.items(), key=lambda item: -item[1][0]):
        print(f"  {rel:9.2e} {parameter}  (at {name})")

    noise_tolerance = 'only presence checked' if noise_parameter_rtol is None else f'rtol={noise_parameter_rtol}'
    tolerances = (f"traces: {precision} decimals, parameters: rtol={parameter_rtol}, "
                  f"noise sensitive parameters: {noise_tolerance}")
    if not errors:
        print(f"The files {file1} and {file2} agree ({tolerances}).")
        return

    print("\n" + "\n\n".join(errors))
    print(f"\n{len(errors)} difference(s) found ({tolerances})")
    sys.exit(1)
