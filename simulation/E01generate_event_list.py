from __future__ import absolute_import, division, print_function
from NuRadioMC.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cuboid
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

# define simulation volume
xmin = -5 * units.km
xmax = 5 * units.km
ymin = -5 * units.km
ymax = 5 * units.km
zmin = -3 * units.km
zmax = 0 * units.km
rmin = 0 * units.km
rmax = 7 * units.km
#generate_eventlist_cuboid('event_input/1e19_n1e5.hdf5', 1e5, 1e19 * units.eV, 1e19 * units.eV,
#                   xmin, xmax, ymin, ymax, zmin, zmax)
generate_eventlist_cylinder('event_input/1e19_n1e4comparison.hdf5', 1e4, 1e19 * units.eV, 1e19 * units.eV,
                   rmin, rmax, zmin, zmax)
