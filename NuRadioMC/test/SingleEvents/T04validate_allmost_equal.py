#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import sys
import h5py
from NuRadioReco.utilities import units
from NuRadioMC.utilities._test_helpers import (
    assert_equal_attributes, assert_equal_keys, assert_equal_station_keys,
    assert_almost_equal_attributes, assert_almost_equal_keys, assert_almost_equal_keys_absolute,
    assert_almost_equal_station_keys, assert_almost_equal_station_keys_absolute)


file1 = sys.argv[1]
file2 = sys.argv[2]
print("Testing the files {} and {} for (almost) equality".format(file1, file2))

fin1 = h5py.File(file1, 'r')
fin2 = h5py.File(file2, 'r')

error = 0

# Timing quantities (travel_times, trigger_times, trigger_times_per_event) are subject to
# last-bit differences in the C++ raytracer's GSL numerical integration that vary across GSL
# versions/platforms (see get_travel_time() in
# NuRadioMC/SignalProp/CPPAnalyticRayTracing/analytic_raytracing.cpp), independent of any actual
# code change. A relative tolerance is physically inappropriate here since this noise floor is an
# absolute quantity that does not scale with the (arbitrarily large) travel/trigger time, so these
# are checked against an absolute tolerance instead.
time_accuracy = 1e-3 * units.ns

# Test those attributes that should be perfectly equal

attributes = [u'trigger_names',
 u'Tnoise',
 u'dt',
#  u'n_samples',
 u'thetamin',
 u'zmax',
 u'zmin',
 u'thetamax',
#  u'header',
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

error = assert_equal_attributes(fin1, fin2, attributes, error=error)

# Test those attributes that should be numerically equal

attributes = [
 u'bandwidth',
 u'Vrms']

error = assert_almost_equal_attributes(fin1, fin2, attributes, error=error)

# Test those station keys that should be perfectly equal

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
 u'multiple_triggers',
 u'zz']
error = assert_equal_keys(fin1, fin2, keys, error=error)

# Test those keys that should be perfectly equal

keys = [
u'ray_tracing_solution_type'
]
error = assert_equal_station_keys(fin1, fin2, keys, error=error)

keys = [
 u'weights']

error = assert_almost_equal_keys(fin1, fin2, keys, error=error)

keys = [
 u'trigger_times']

error = assert_almost_equal_keys_absolute(fin1, fin2, keys, atol=time_accuracy, error=error)

keys = [
 u'ray_tracing_C0',
 u'ray_tracing_C1',
 u'launch_vectors',
 u'receive_vectors',
 u'travel_distances',
 u'polarization',
 u'max_amp_shower_and_ray',
 ]

error = assert_almost_equal_station_keys(fin1, fin2, keys, error=error)

keys = [
 u'travel_times',
 u'trigger_times_per_event',
 u'trigger_times',
 ]

error = assert_almost_equal_station_keys_absolute(fin1, fin2, keys, atol=time_accuracy, error=error)

# for some reason the test suddenly can't achieve a good enough precision on this quantity. Lets reduce precision
# for this vairble for now.
keys = [u'maximum_amplitudes_envelope']
error = assert_almost_equal_station_keys(fin1, fin2, keys, accuracy=0.002, error=error)

# test maximimum amplitude separately because it might differ slightly because of differences in the interferene between signals
keys = [u'maximum_amplitudes']
error = assert_almost_equal_station_keys(fin1, fin2, keys, accuracy=0.01, error=error)

if error == 0:
    print("The two files are (almost) identical.")
else:
    from NuRadioMC.utilities.dump_hdf5 import dump
    print(f"file 1 {file1}")
    dump(file1)
    print(f"file 2 {file2}")
    dump(file2)
    sys.exit(error)
