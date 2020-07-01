#!/usr/bin/env python
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

# define simulation volume
volume = {
'fiducial_rmin':0 * units.km,
'fiducial_rmax': 4 * units.km,
'fiducial_zmin':-2.7 * units.km,  # the ice sheet at South Pole is 2.7km deep
'fiducial_zmax': 0 * units.km}

generate_eventlist_cylinder("trigger_test_eventlist.hdf5", 1000, Emin=1e19 * units.eV, Emax=1e19 * units.eV, volume)

