#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

# define simulation volume
zmin = -2.7 * units.km  # the ice sheet at South Pole is 2.7km deep
zmax = 0 * units.km
rmin = 0 * units.km
rmax = 2 * units.km

# generate one event list at 1e18 eV with 10000 neutrinos
generate_eventlist_cylinder('1e18_full.hdf5', 2000, 1e18 * units.eV, 1e18 * units.eV, rmin, rmax, zmin, zmax,
                            full_rmin=rmin, full_rmax=rmax, full_zmin=zmin, full_zmax=zmax)
