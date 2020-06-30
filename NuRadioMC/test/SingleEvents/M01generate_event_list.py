#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

# define simulation volume
volume = {
'fiducial_rmin':0 * units.km,
'fiducial_rmax': 2 * units.km,
'fiducial_zmin':-0.55 * units.km,  # the ice sheet at South Pole is 2.7km deep
'fiducial_zmax': 0 * units.km}

# generate one event list at 1e18 eV with 10000 neutrinos
generate_eventlist_cylinder('MB_1e18_full.hdf5', 1e3, 1e18 * units.eV, 1e18 * units.eV, volume)
