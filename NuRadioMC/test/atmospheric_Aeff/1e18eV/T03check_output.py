#!/usr/bin/env python
import numpy as np
import h5py
from NuRadioReco.utilities import units
from NuRadioMC.utilities.Veff import get_triggered
import sys
import os

###########################
# Reference values from previous run, have to be updated, if code changes
###########################

# the event generation has a fixed seed and the Askaryan model is Alvarez2000,
# so as a result the effective area is no statistical scatter.
Aeff_mean = 0.175822
Aeff_sigma = 0.0001

path = os.path.dirname(os.path.abspath(__file__))
fin = h5py.File(os.path.join(path, "output.hdf5"), 'r')


def calculate_aeff(fin):
    triggered = get_triggered(fin)

    weights = np.array(fin['weights'])
    n_events = fin.attrs['n_events']

    n_triggered = np.sum(weights[triggered])

    V = None
    rmin = fin.attrs['rmin']
    rmax = fin.attrs['rmax']
    geometrical_area = np.pi * ( rmax - rmin ) ** 2

    thetamin = fin.attrs['thetamin']
    thetamax = fin.attrs['thetamax']
    projected_area = geometrical_area * 0.5 * ( np.cos(thetamin) + np.cos(thetamax) )

    Aeff = projected_area * n_triggered / n_events

    return n_triggered, n_events, Aeff

###########################
# calculate effective area
###########################


n_triggered, n_events, Aeff = calculate_aeff(fin)

print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f} -> Effective area is ~ {:.6g} km^2'.format(n_triggered, n_events, n_triggered / n_events, Aeff / units.km ** 2))

delta = (Aeff / units.km ** 2 - Aeff_mean) / Aeff_sigma
rdelta = (Aeff / units.km ** 2 - Aeff_mean) / Aeff_mean
print("effective area deviates {:.1f} sigma ({:.0f}%) from the mean".format(delta, 100 * rdelta))

if(np.abs(Aeff / units.km ** 2 - Aeff_mean) > 3 * Aeff_sigma):
    print("deviation is more than 3 sigma -> this should only happen in less than 1\% of the tests. Rerun the test and see if the error persists.")
    sys.exit(-1)

###########################
# Code to generate new average values for this test
###########################

Aeffs = []

if False:
    files = os.listdir('.')
    for file in files:
        if (".hdf5" in file) and ("output" in file):
            fin = h5py.File(file, 'r')
            n_triggered, n_events, Aeff = calculate_aeff(fin)
            Aeffs.append(Aeff / units.km ** 2)
            print(Aeff / units.km ** 2)

    print("New mean Aeff {}".format(np.mean(Aeffs)))
    print("New sigma Aeff {}".format(np.std(Aeffs)))
