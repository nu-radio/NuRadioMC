from __future__ import absolute_import, division, print_function
from NuRadioMC.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist

# define simulation volume
xmin = -3 * units.km
xmax = 3 * units.km
ymin = -3 * units.km
ymax = 3 * units.km
zmin = -2.7 * units.km
zmax = 0 * units.km
generate_eventlist('event_input/1e19_n1e5.hdf5', 1e5, 1e19 * units.eV, 1e19 * units.eV,
                   xmin, xmax, ymin, ymax, zmin, zmax)