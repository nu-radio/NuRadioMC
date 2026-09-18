"""
Shared helper functions for comparing two HDF5 simulation output files for (almost) equality.

Used by the various validate*.py regression tests that check a freshly generated output file
against a checked-in reference file (e.g. NuRadioMC/test/SingleEvents/T04validate_allmost_equal.py,
NuRadioMC/test/examples/validate_radio_emitter_allmost_equal.py).

Each function returns the (possibly incremented) running error count rather than raising, so a
script can run all its checks and report every mismatch before deciding whether to exit non-zero.
"""
import numpy as np
from numpy import testing


def _get_event_group_ids(fin, station):
    # different NuRadioMC output formats store event_group_ids either at the top level or
    # nested under the station group
    try:
        return np.array(fin['event_group_ids'])
    except KeyError:
        return np.array(fin[station]['event_group_ids'])


def assert_equal_attributes(fin1, fin2, keys, error=0):
    """Check that the given file-level attributes are exactly equal."""
    for key in keys:
        try:
            testing.assert_equal(fin1.attrs[key], fin2.attrs[key])
        except AssertionError as e:
            print("\n attribute {} not equal".format(key))
            print(e)
            error += 1
    return error


def assert_equal_keys(fin1, fin2, keys, error=0):
    """Check that the given top-level datasets are exactly equal."""
    for key in keys:
        try:
            testing.assert_equal(np.array(fin1[key]), np.array(fin2[key]))
        except AssertionError as e:
            print("\narray {} not equal".format(key))
            print("Reference: {}, reconstruction: {}".format(fin2[key], fin1[key]))
            print(e)
            error += 1
    return error


def assert_equal_station_keys(fin1, fin2, keys, station='station_101', error=0):
    """Check that the given per-station datasets are exactly equal."""
    for key in keys:
        try:
            testing.assert_equal(np.array(fin1[station][key]), np.array(fin2[station][key]))
        except AssertionError as e:
            print("\narray {} not equal".format(key))
            print("Reference: {}, reconstruction: {}".format(fin2[station][key], fin1[station][key]))
            print(e)
            error += 1
    return error


def assert_almost_equal_attributes(fin1, fin2, keys, accuracy=0.0005, error=0):
    """Check that the given file-level attributes agree within a relative tolerance."""
    for key in keys:
        arr1 = np.array(fin1.attrs[key])
        arr2 = np.array(fin2.attrs[key])
        max_diff = np.max(np.abs((arr1 - arr2) / arr2))
        if max_diff > accuracy:
            print('Reconstruction of {} does not agree with reference (error: {})'.format(key, max_diff))
            print("\n attribute {} not almost equal".format(key))
            error += 1
    return error


def assert_almost_equal_keys(fin1, fin2, keys, accuracy=0.0005, error=0):
    """Check that the given top-level, per-event datasets agree within a relative tolerance."""
    for key in keys:
        arr1 = np.array(fin1[key])
        arr2 = np.array(fin2[key])
        for i in range(arr1.shape[0]):
            max_diff = np.max(np.abs((arr1[i] - arr2[i]) / arr2[i]))
            if max_diff > accuracy:
                print('Reconstruction of {} of event {} does not agree with reference (error: {})'.format(key, i, max_diff))
                print("\n attribute {} not almost equal".format(key))
                error += 1
    return error


def assert_almost_equal_keys_absolute(fin1, fin2, keys, atol, error=0):
    """
    Check that the given top-level, per-event datasets agree within an absolute tolerance.

    Use for quantities (e.g. timing) where an absolute tolerance is physically appropriate,
    rather than a tolerance relative to the (arbitrary) magnitude of the value.
    """
    for key in keys:
        arr1 = np.array(fin1[key])
        arr2 = np.array(fin2[key])
        for i in range(arr1.shape[0]):
            max_diff = np.max(np.abs(arr1[i] - arr2[i]))
            if max_diff > atol:
                print('Reconstruction of {} of event {} does not agree with reference (absolute error: {})'.format(key, i, max_diff))
                print("\n attribute {} not almost equal".format(key))
                error += 1
    return error


def assert_almost_equal_station_keys(fin1, fin2, keys, station='station_101', accuracy=0.0005, error=0):
    """
    Check that the given per-station, per-event datasets agree within a relative tolerance.

    Entries that are exactly zero in the reference (`fin2`) are compared for exact equality
    instead, since a relative error is undefined at zero; NaN and zero are treated as equivalent
    "no solution" sentinels there, since different code versions have used either convention.
    """
    gids = _get_event_group_ids(fin1, station)
    for key in keys:
        arr1 = np.array(fin1[station][key])
        arr2 = np.array(fin2[station][key])
        for i in range(arr1.shape[0]):
            zero_mask = arr2[i] == 0
            max_diff = np.max(np.abs((arr1[i][~zero_mask] - arr2[i][~zero_mask]) / arr2[i][~zero_mask]))
            if max_diff > accuracy:
                print(f'Reconstruction of {key} of event index {i} = group event id {gids[i]} does not agree with reference (relative error: {max_diff})')
                print("\n attribute {} not almost equal".format(key))
                print(np.abs((arr1[i] - arr2[i]) / arr2[i]))
                print(arr1[i])
                print(arr2[i])
                error += 1
            # zero entries can't be compared with a relative tolerance; check for exact equality
            # instead, treating NaN as equivalent to zero since both are used as "no solution"
            # sentinels depending on code version
            tmp1 = np.nan_to_num(arr1[i][zero_mask])
            tmp2 = np.nan_to_num(arr2[i][zero_mask])
            if not np.all(tmp1 == tmp2):
                max_diff = np.max(np.abs(tmp1 - tmp2))
                print('Reconstruction of {} of event {} does not agree with reference (absolute error: {})'.format(key, i, max_diff))
                print("\n attribute {} not almost equal".format(key))
                print(arr1[i])
                print(arr2[i])
                error += 1
    return error


def assert_almost_equal_station_keys_absolute(fin1, fin2, keys, station='station_101', atol=1e-3, error=0):
    """
    Check that the given per-station, per-event datasets agree within an absolute tolerance.

    Use for quantities (e.g. timing) where an absolute tolerance is physically appropriate,
    rather than a tolerance relative to the (arbitrary) magnitude of the value.
    """
    gids = _get_event_group_ids(fin1, station)
    for key in keys:
        arr1 = np.array(fin1[station][key])
        arr2 = np.array(fin2[station][key])
        for i in range(arr1.shape[0]):
            max_diff = np.max(np.abs(arr1[i] - arr2[i]))
            if max_diff > atol:
                print(f'Reconstruction of {key} of event index {i} = group event id {gids[i]} does not agree with reference (absolute error: {max_diff})')
                print("\n attribute {} not almost equal".format(key))
                print(np.abs(arr1[i] - arr2[i]))
                print(arr1[i])
                print(arr2[i])
                error += 1
    return error
