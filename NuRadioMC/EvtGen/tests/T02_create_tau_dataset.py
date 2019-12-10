import numpy as np
import h5py

from NuRadioMC.EvtGen import generator
from NuRadioReco.utilities import units
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("testtaueventgen")


# define simulation volume
fiducial_zmin = -2.7 * units.km
fiducial_zmax = 0 * units.km
generator.generate_eventlist_cylinder('tau.hdf5', 1e4, 1e18 * units.eV, 1e19 * units.eV,
                            0, 3*units.km, fiducial_zmin, fiducial_zmax, add_tau_second_bang=True)


print("writing many subfiles")
generator.generate_eventlist_cylinder('tau2.hdf5', 1e4, 1e16 * units.eV, 1e19 * units.eV,
                            0, 3*units.km, fiducial_zmin, fiducial_zmax, add_tau_second_bang=True, n_events_per_file=10)
