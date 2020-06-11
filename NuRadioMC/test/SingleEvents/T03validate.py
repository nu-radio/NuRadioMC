#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import sys
import h5py
import numpy as np
from numpy import testing
import argparse
from NuRadioReco.utilities import units
import logging

error = 0

file1 = sys.argv[1]
file2 = sys.argv[2]
print("Testing the files {} and {} for equality".format(file1, file2))

fin1 = h5py.File(file1, 'r')
fin2 = h5py.File(file2, 'r')

attributes = [
 u'trigger_names',
 u'Tnoise',
 u'Vrms',
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
for key in attributes:
    try:
        testing.assert_equal(fin1.attrs[key], fin2.attrs[key])
    except AssertionError as e:
        print("\n attribute {} not equal".format(key))
        print(e)
        error = -1

keys = [u'azimuths',
 u'energies',
 u'event_group_ids',
 u'flavors',
 u'inelasticity',
 u'interaction_type',
 u'multiple_triggers',
 u'n_interaction',
 u'triggered',
 u'xx',
 u'yy',
 u'zeniths',
 u'zz']
for key in keys:
    try:
        testing.assert_equal(np.array(fin1[key]), np.array(fin2[key]))
    except AssertionError as e:
        print("\narray {} not equal".format(key))
        print(e)
        error = -1

keys = [
 u'weights']
for key in keys:
    try:
        testing.assert_allclose(np.array(fin1[key]), np.array(fin2[key]), rtol=1e-12)
    except AssertionError as e:
        print("\narray {} not equal".format(key))
        print(e)
        error = -1

keys2 = [
 u'maximum_amplitudes',
 u'maximum_amplitudes_envelope']
for key in keys2:
    try:
        testing.assert_allclose(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]), rtol=1e-3)
#         testing.assert_almost_equal(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]))
    except AssertionError as e:
        print("\narray {} of group station_101 not equal".format(key))
        print(e)
        error = -1

keys2 = [
 u'multiple_triggers',
 u'ray_tracing_solution_type',
 u'triggered']
for key in keys2:
    try:
        testing.assert_allclose(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]), rtol=1e-9)
#         testing.assert_almost_equal(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]))
    except AssertionError as e:
        print("\narray {} of group station_101 not equal".format(key))
        print(e)
        error = -1

keys2 = [
    u'travel_distances',
    u'ray_tracing_C1', ]
for key in keys2:
    try:
        testing.assert_allclose(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]), rtol=1e-9, atol=2 * units.mm)
#         testing.assert_almost_equal(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]))
    except AssertionError as e:
        print("\narray {} of group station_101 not equal".format(key))
        print(e)
        error = -1

keys2 = [
    u'travel_times']
for key in keys2:
    try:
        testing.assert_allclose(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]), rtol=1e-9, atol=10 * units.ps)
#         testing.assert_almost_equal(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]))
    except AssertionError as e:
        print("\narray {} of group station_101 not equal".format(key))
        print(e)
        error = -1

keys2 = [
 u'polarization',
 u'launch_vectors',
 u'receive_vectors',
 ]
for key in keys2:
    try:
        testing.assert_allclose(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]), rtol=1e-9, atol=1e-6)
#         testing.assert_almost_equal(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]))
    except AssertionError as e:
        print("\narray {} of group station_101 not equal".format(key))
        print(e)
        error = -1

keys2 = [
 u'ray_tracing_C0'
 ]
for key in keys2:
    try:
        testing.assert_allclose(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]), rtol=1e-6, atol=1e-9)
#         testing.assert_almost_equal(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]))
    except AssertionError as e:
        print("\narray {} of group station_101 not equal".format(key))
        print(e)
        error = -1

if(error == -1):
    sys.exit(error)
else:
    print("The two hdf5 files are identical.")

