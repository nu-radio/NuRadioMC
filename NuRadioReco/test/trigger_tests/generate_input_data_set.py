#!/usr/bin/env python
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

generate_eventlist_cylinder("trigger_test_eventlist.hdf5", 1000, Emin=1e19 * units.eV, Emax=1e19 * units.eV, fiducial_rmin=0,
                            fiducial_rmax=4 * units.km, fiducial_zmin=-2.7 * units.km, fiducial_zmax=0)

