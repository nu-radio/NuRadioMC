#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import os
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import numpy as np

# define simulation volume
volume = {
'fiducial_rmin':0 * units.km,
'fiducial_rmax': 3 * units.km,
'fiducial_zmin':-2 * units.km,  # the ice sheet at South Pole is 2.7km deep
'fiducial_zmax': 0 * units.km}

path = os.path.dirname(os.path.abspath(__file__))

# generate one event list at 1e18 eV with 10000 neutrinos
generate_eventlist_cylinder(os.path.join(path, '1e18_full_noise.hdf5'),
                            1e4, 1e18 * units.eV, 1e18 * units.eV,
                            volume, seed=10)
