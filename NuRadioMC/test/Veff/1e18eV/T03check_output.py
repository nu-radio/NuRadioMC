#!/usr/bin/env python
import numpy as np
import h5py
from NuRadioReco.utilities import units
import sys
import os

###########################
# Reference values from previous run, have to be updated, if code changes
###########################

Veff_mean = 3.53
Veff_sigma = 0.05

path = os.path.dirname(os.path.abspath(__file__))
fin = h5py.File(os.path.join(path,"output.hdf5"), 'r')

def calculate_veff(fin):
    weights = np.array(fin['weights'])
    n_events = fin.attrs['n_events']
    density_ice = 0.9167 * units.g / units.cm ** 3
    density_water = 997 * units.kg / units.m ** 3

    n_triggered = np.sum(weights)

    V = None
    rmin = fin.attrs['rmin']
    rmax = fin.attrs['rmax']
    dZ = fin.attrs['zmax'] - fin.attrs['zmin']
    V = np.pi * (rmax ** 2 - rmin ** 2) * dZ
    Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights) / n_events
    return n_triggered, n_events, Veff

###########################
# calculate effective volume
###########################

n_triggered, n_events, Veff = calculate_veff(fin)

print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f} -> {:.6g} km^3 sr'.format(n_triggered, n_events, n_triggered / n_events, Veff / units.km ** 3))

delta = (Veff / units.km ** 3 - Veff_mean) / Veff_sigma
rdelta = (Veff / units.km ** 3 - Veff_mean) / Veff_mean
print("effective volume deviates {:.1f} sigma ({:.0f}%) from the mean".format(delta, 100 * rdelta))

if(np.abs(Veff / units.km ** 3 - Veff_mean) > 2 * Veff_sigma):
    print("deviation is more than 2 sigma -> this should only happen in 5\% of the tests. Rerun the test and see if the error persists.")
    sys.exit(-1)

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
            Veffs.append(Veff/ units.km ** 3)
            print(Veff/ units.km ** 3)

    print("New mean Veff {}".format(np.mean(Veffs)))
    print("New sigma Veff {}".format(np.std(Veffs)))


