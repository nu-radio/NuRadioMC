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


def test_equal_attributes(keys, fin1=fin1, fin2=fin2, error=error):
    for key in keys:
        try:
            testing.assert_equal(fin1.attrs[key], fin2.attrs[key])
        except AssertionError as e:
            print("\n attribute {} not almost equal".format(key))
            print(e)
            error = -1
    return error


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


def test_almost_equal_attributes(keys, fin1=fin1, fin2=fin2, error=error):
    for key in keys:
        arr1 = np.array(fin1.attrs[key])
        arr2 = np.array(fin2.attrs[key])
        max_diff = np.max(np.abs((arr1 - arr2) / arr2))
        if max_diff > accuracy:
            print('Reconstruction of {} does not agree with reference (error: {})'.format(key, max_diff))
            print("\n attribute {} not almost equal".format(key))
            error = -1
    return error


def test_almost_equal_station_keys(keys, fin1=fin1, fin2=fin2, error=error):
    for key in keys:
        arr1 = np.array(fin1['station_101'][key])
        arr2 = np.array(fin2['station_101'][key])
        for i in range(arr1.shape[0]):
            max_diff = np.max(np.abs((arr1[i] - arr2[i]) / arr2[i]))
#             print('Reconstruction of {} of event {} (relative error: {})'.format(key, i, np.abs((arr1[i] - arr2[i]) / arr2[i])))
            if max_diff > accuracy:
#                 print(arr1.shape)
                print('Reconstruction of {} of event {} does not agree with reference (relative error: {})'.format(key, i, max_diff))
                print("\n attribute {} not almost equal".format(key))
#                 print(np.abs((arr1[i] - arr2[i]) / arr2[i]))
#                 print(arr1[i])
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

# Test those attributes that should be perfectly equal


attributes = [u'trigger_names',
 u'Tnoise',
 u'dt',
 u'bandwidth',
 u'n_samples',
 u'thetamin',
 u'zmax',
 u'zmin',
 u'thetamax',
 u'header',
 u'fiducial_zmax',
 u'fiducial_zmin',
 u'flavors',
 u'rmin',
 u'total_number_of_events',
 u'deposited',
 u'phimax',
 u'phimin',
 u'Emin',
 u'rmax',
 u'fiducial_rmax',
 u'Emax',
 u'fiducial_rmin',
 u'n_events']

error = test_equal_attributes(attributes, fin1=fin1, fin2=fin2, error=error)

# Test those attributes that should be numerically equal

attributes = [
 u'Vrms']

error = test_almost_equal_attributes(attributes, fin1=fin1, fin2=fin2, error=error)

# Test those station keys that should be perfectly equal

keys = [u'azimuths',
 u'energies',
 u'event_ids',
 u'flavors',
 u'inelasticity',
 u'interaction_type',
 u'multiple_triggers',
 u'n_interaction',
 u'triggered',
 u'xx',
 u'yy',
 u'zeniths',
 u'multiple_triggers',
 u'zz']
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
 u'SNRs',
 u'maximum_amplitudes',
 u'maximum_amplitudes_envelope',
u'polarization',
 u'ray_tracing_C0',
 u'launch_vectors',
 u'receive_vectors',
 u'travel_times',
 u'travel_distances',
 u'ray_tracing_C1',
 ]

error = test_almost_equal_station_keys(keys, fin1=fin1, fin2=fin2, error=error)

if error == -1:
    sys.exit(error)
else:
    print("The two files are (almost) identical.")
