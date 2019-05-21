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

fin1 = h5py.File(file1, 'r')
fin2 = h5py.File(file2, 'r')

# Test all attributes in file

attributes = [u'trigger_names',
 u'antenna_positions',
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
    testing.assert_equal(fin1.attrs[key], fin2.attrs[key])

#Test all keys in file

keys = [u'SNRs',
    u'azimuths',
    u'energies',
    u'event_ids',
    u'flavors',
    u'inelasticity',
    u'interaction_type',
    u'launch_vectors',
    u'maximum_amplitudes',
    u'maximum_amplitudes_envelope',
    u'multiple_triggers',
    u'n_interaction',
    u'polarization',
    u'ray_tracing_C0',
    u'ray_tracing_C1',
    u'ray_tracing_solution_type',
    u'receive_vectors',
    u'travel_distances',
    u'travel_times',
    u'triggered',
    u'weights',
    u'xx',
    u'yy',
    u'zeniths',
    u'zz']

for key in keys:
    testing.assert_equal(np.array(fin1[key]), np.array(fin2[key]))


# keys2 = [u'SNRs',
#  u'launch_vectors',
#  u'maximum_amplitudes',
#  u'maximum_amplitudes_envelope',
#  u'multiple_triggers',
#  u'polarization',
#  u'ray_tracing_C0',
#  u'ray_tracing_C1',
#  u'ray_tracing_solution_type',
#  u'receive_vectors',
#  u'travel_distances',
#  u'travel_times',
#  u'triggered']
#
# for key in keys2:
#     testing.assert_equal(np.array(fin1['station_101'][key]), np.array(fin2['station_101'][key]))


print("if no error occured, previous results are exactly reproduced.")



