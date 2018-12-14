import numpy as np
import h5py

from NuRadioMC.EvtGen import generator
from NuRadioMC.utilities import units


# define simulation volume
xmin = -3 * units.km
xmax = 3 * units.km
ymin = -3 * units.km
ymax = 3 * units.km
zmin = -2.7 * units.km
zmax = 0 * units.km
generator.generate_eventlist_cylinder('tau.hdf5', 1e3, 1e19 * units.eV, 1e19 * units.eV,
                            0, 3*units.km, zmin, zmax, addTauSecondBang=True)


generator.generate_eventlist_cylinder('tau2.hdf5', 1e2, 1e19 * units.eV, 1e19 * units.eV,
                            0, 3*units.km, zmin, zmax, addTauSecondBang=True, n_events_per_file=10)

