#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import os
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import numpy as np
np.random.seed(0)

# define simulation volume
zmin = -2.7 * units.km  # the ice sheet at South Pole is 2.7km deep
zmax = 0 * units.km
rmin = 0 * units.km
rmax = 4 * units.km

path = os.path.dirname(os.path.abspath(__file__))

# generate one event list at 1e18 eV with 10000 neutrinos
generate_eventlist_cylinder(os.path.join(path,'1e18_full.hdf5'),
                            5e4, 1e18 * units.eV, 1e18 * units.eV,
                            rmin, rmax, zmin, zmax,
                            full_rmin=rmin, full_rmax=rmax,
                            full_zmin=zmin, full_zmax=zmax)
