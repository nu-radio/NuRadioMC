#!/usr/bin/env python
import numpy as np
import h5py
from NuRadioMC.utilities import units
import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
print path
fin = h5py.File(os.path.join(path,"output.hdf5"), 'r')
weights = np.array(fin['weights'])
n_events = fin.attrs['n_events']

###########################
# calculate effective volume
###########################
density_ice = 0.9167 * units.g / units.cm ** 3
density_water = 997 * units.kg / units.m ** 3

n_triggered = np.sum(weights)

V = None
rmin = fin.attrs['rmin']
rmax = fin.attrs['rmax']
dZ = fin.attrs['zmax'] - fin.attrs['zmin']
V = np.pi * (rmax ** 2 - rmin ** 2) * dZ
Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights) / n_events
print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f} -> {:.6g} km^3 sr'.format(n_triggered, n_events, n_triggered / n_events, Veff / units.km ** 3))

Veff_mean = 4.74
Veff_sigma = 0.43

delta = (Veff / units.km ** 3 - Veff_mean) / Veff_sigma
rdelta = (Veff / units.km ** 3 - Veff_mean) / Veff_mean
print("effective volume deviates {:.1f} sigma ({:.0f}%) from the mean".format(delta, 100 * rdelta))

if(np.abs(Veff / units.km ** 3 - Veff_mean) > 2 * Veff_sigma):
    print("deviation is more than 2 sigma -> this should only happen in 5\% of the tests. Rerun the test and see if the error persists.")
    sys.exit(-1)
