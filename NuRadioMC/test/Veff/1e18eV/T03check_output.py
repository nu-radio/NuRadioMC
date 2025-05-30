#!/usr/bin/env python3
import numpy as np
import h5py
from NuRadioReco.utilities import units
import sys
import os


###########################
# Reference values from previous run, have to be updated, if code changes
###########################

# the event generation has a fixed seed and I switched to Alvarez2000 (also no randomness)
# thus, the Veff has no statistical scatter

trigger_per_file = {
    "output.hdf5": 'highlow_2sigma',
    "output_noise.hdf5": 'PA_4channel_100Hz',
}

Veff_mean_per_file = {
    "output.hdf5": 4.62703,
    "output_noise.hdf5": 7.3716
}
Veff_sigma = 0.0001

file_name = sys.argv[1]
Veff_mean = Veff_mean_per_file[file_name]
trigger = trigger_per_file[file_name]

path = os.path.dirname(os.path.abspath(__file__))
fin = h5py.File(os.path.join(path, file_name), 'r')


def calculate_veff(fin):
    n_events = fin.attrs['n_events']
    uids, unique_mask = np.unique(np.array(fin['event_group_ids']), return_index=True)
    weights = np.array(fin['weights'])[unique_mask]

    n_triggered = np.sum(weights)

    V = fin.attrs['volume']
    Veff = V * 4 * np.pi * np.sum(weights) / n_events
    return n_triggered, n_events, Veff

###########################
# calculate effective volume
###########################


n_triggered, n_events, Veff = calculate_veff(fin)

print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f} -> {:.6g} km^3 sr'.format(
    n_triggered, n_events, n_triggered / n_events, Veff / units.km ** 3))

delta = (Veff / units.km ** 3 - Veff_mean) / Veff_sigma
rdelta = (Veff / units.km ** 3 - Veff_mean) / Veff_mean
if np.abs(delta) > 3:
    print("Effective volume deviates by {:.0f}% from the expectation (this is not expected!)".format(100 * rdelta))
    sys.exit(-1)

# calculate Veff using veff utility
import NuRadioMC.utilities.Veff
data = NuRadioMC.utilities.Veff.get_Veff_Aeff(os.path.join(path, file_name))[0]
Veff_utl, Veff_utl_error, utl_weighed_sum, t1, t2 = data['veff']['all_triggers']
Veff_utl = Veff_utl * 4 * np.pi
np.testing.assert_almost_equal(Veff_utl, Veff, decimal=3)
Veff_utl, Veff_utl_error, utl_weighed_sum, t1, t2 = data['veff'][trigger]
Veff_utl = Veff_utl * 4 * np.pi
np.testing.assert_almost_equal(Veff_utl, Veff, decimal=3)

###########################
# Code to generate new average values for this test
###########################

Veffs = []

if False:
    files = os.listdir('.')
    for file in files:
        if (".hdf5" in file) and ("output" in file):
            fin = h5py.File(file, 'r')
            n_triggered, n_events, Veff = calculate_veff(fin)
            Veffs.append(Veff / units.km ** 3)
            print(Veff / units.km ** 3)

    print("New mean Veff {}".format(np.mean(Veffs)))
    print("New sigma Veff {}".format(np.std(Veffs)))
