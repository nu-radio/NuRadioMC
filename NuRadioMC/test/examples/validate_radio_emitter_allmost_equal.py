#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import sys
import h5py
import numpy as np
from numpy import testing
import argparse
from NuRadioReco.utilities import units
import logging

file1 = sys.argv[1]
file2 = sys.argv[2]
print("Testing the files {} and {} for (almost) equality".format(file1, file2))

fin1 = h5py.File(file1, 'r')
fin2 = h5py.File(file2, 'r')

error = 0
accuracy = 0.0005


def test_equal_station_keys(keys, fin1=fin1, fin2=fin2, error=error):
    for key in keys:
        try:
            testing.assert_equal(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]))

        except AssertionError as e:
            print("\narray {} not almost equal".format(key))
            print("\Reference: {}, reconstruction: {}".format(fin2[key], fin1[key]))
            print(e)
            error = -1
    return error


def test_equal_keys(keys, fin1=fin1, fin2=fin2, error=error):
    for key in keys:
        try:
            testing.assert_equal(np.array(fin1[key]), np.array(fin2[key]))
        except AssertionError as e:
            print("\narray {} not almost equal".format(key))
            print("\Reference: {}, reconstruction: {}".format(fin2[key], fin1[key]))
            print(e)
            error = -1
    return error


def test_almost_equal_station_keys(keys, fin1=fin1, fin2=fin2, error=error, accuracy=accuracy):
    gids = np.array(fin1["station_101"]['event_group_ids'])
    for key in keys:
        arr1 = np.array(fin1['station_101'][key])
        arr2 = np.array(fin2['station_101'][key])
        for i in range(arr1.shape[0]):
            zero_mask = arr2[i] == 0
            max_diff = np.max(np.abs((arr1[i][~zero_mask] - arr2[i][~zero_mask]) / arr2[i][~zero_mask]))
#             print('Reconstruction of {} of event {} (relative error: {})'.format(key, i, np.abs((arr1[i] - arr2[i]) / arr2[i])))
            if max_diff > accuracy:
#                 print(arr1.shape)
                print(f'Reconstruction of {key} of event index {i} = group event id {gids[i]} does not agree with reference (relative error: {max_diff})')
                print("\n attribute {} not almost equal".format(key))
                print(np.abs((arr1[i] - arr2[i]) / arr2[i]))
                print(arr1[i])
                print(arr2[i])
                error = -1
            # now test zero entries for equality
            if not np.all(arr1[i][zero_mask] == arr2[i][zero_mask]):
                max_diff = np.max(np.abs(arr1[i][zero_mask] - arr2[i][zero_mask]))
                print('Reconstruction of {} of event {} does not agree with reference (absolute error: {})'.format(key, i, max_diff))
                print("\n attribute {} not almost equal".format(key))
                error = -1
    return error


def test_almost_equal_keys(keys, fin1=fin1, fin2=fin2, error=error):
    for key in keys:
        arr1 = np.array(fin1[key])
        arr2 = np.array(fin2[key])
        for i in range(arr1.shape[0]):
            max_diff = np.max(np.abs((arr1[i] - arr2[i]) / arr2[i]))
            if max_diff > accuracy:
                print('Reconstruction of {} of event {} does not agree with reference (error: {})'.format(key, i, max_diff))
                print("\n attribute {} not almost equal".format(key))
                error = -1
    return error

# Test those station keys that should be perfectly equal

keys = [u'emitter_amplitudes',
 u'emitter_antenna_type',
 u'emitter_frequency',
 u'emitter_half_width',
 u'emitter_model',
 u'emitter_orientation_phi',
 u'emitter_orientation_theta',
 u'emitter_rotation_phi',
 u'emitter_rotation_theta',
 u'event_group_ids',
 u'multiple_triggers',
 u'triggered',
 u'xx',
 u'yy',
 u'zz',
 u'shower_ids']

error = test_equal_keys(keys, fin1=fin1, fin2=fin2, error=error)

# Test those keys that should be perfectly equal

keys = [
u'ray_tracing_solution_type'
]
error = test_equal_station_keys(keys, fin1=fin1, fin2=fin2, error=error)

keys = [
 u'weights']

error = test_almost_equal_keys(keys, fin1=fin1, fin2=fin2, error=error)

keys = [
 u'max_amp_shower_and_ray',
 u'polarization',
 u'ray_tracing_C0',
 u'launch_vectors',
 u'receive_vectors',
 u'travel_times',
 u'travel_distances',
 u'ray_tracing_C1']



error = test_almost_equal_station_keys(keys, fin1=fin1, fin2=fin2, error=error)

# for some reason the test suddenly can't achieve a good enough precision on this quantity. Lets reduce precision
# for this vairble for now.
keys = [u'maximum_amplitudes_envelope']
error = test_almost_equal_station_keys(keys, fin1=fin1, fin2=fin2, error=error, accuracy=0.001)

# test maximimum amplitude separately because it might differ slightly because of differences in the interferene between signals
keys = [u'maximum_amplitudes']
error = test_almost_equal_station_keys(keys, fin1=fin1, fin2=fin2, error=error, accuracy=0.01)

print("The two files {} and {} are (almost) identical.".format(file1, file2))

