#!/usr/bin/env python
import numpy as np
import h5py
from NuRadioReco.utilities import units
import sys
import os
from NuRadioMC.utilities.Veff import remove_duplicate_triggers

###########################
# Reference values from previous run, have to be updated, if code changes
###########################

# the event generation used to have a fixed seed. The Askaryan model is Alvarez2000,
# so as a result the effective area should have no statistical scatter. However,
# Proposal gives a different result for the same seed depending on the system,
# so now we don't fix the seed and check that the calculated effective area is
# within a number of standard of the previous calculation.
# The following numbers have been obtained with 5e3 input events. The mean
# and uncertainty have been calculated using the function calculate_aeff.
Aeff_mean = 0.406902 * units.km2
Aeff_sigma = 0.045211 * units.km2

path = os.path.dirname(os.path.abspath(__file__))
fin = h5py.File(sys.argv[1], 'r')


def calculate_aeff(fin):
    triggered = np.array(fin['triggered'])
    triggered = remove_duplicate_triggers(triggered, fin['event_group_ids'])

    weights = np.array(fin['weights'])
    n_events = fin.attrs['n_events']

    n_triggered = np.sum(weights[triggered])

    geometrical_area = fin.attrs['area']

    thetamin = fin.attrs['thetamin']
    thetamax = fin.attrs['thetamax']
    projected_area = geometrical_area * 0.5 * (np.cos(thetamin) + np.cos(thetamax))

    Aeff = projected_area * n_triggered / n_events
    Aeff_unc = Aeff / np.sqrt(n_triggered)

    return n_triggered, n_events, Aeff, Aeff_unc

###########################
# calculate effective area
###########################


n_triggered, n_events, Aeff, Aeff_unc = calculate_aeff(fin)

print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f} -> Effective area is ~ {:.6g} km^2'.format(n_triggered, n_events, n_triggered / n_events, Aeff / units.km ** 2))

delta = (Aeff - Aeff_mean) / Aeff_sigma
rdelta = (Aeff - Aeff_mean) / Aeff_mean
print("effective area deviates {:.1f} sigma ({:.0f}%) from the previous result".format(delta, 100 * rdelta))

if(np.abs(Aeff - Aeff_mean) > 4 * Aeff_sigma):
    print("Deviation is more than 4 sigma with respect to the previous result.")
    print("This should only happen in a low number of tests. Rerun the test and see if the error persists.")
    sys.exit(-1)

###########################
# Code to generate new average values for this test
###########################

print("New Aeff {} km^2".format(np.mean(Aeff / units.km2)))
print("New sigma Aeff (poissonian) {} km^2".format(Aeff_unc / units.km2))
