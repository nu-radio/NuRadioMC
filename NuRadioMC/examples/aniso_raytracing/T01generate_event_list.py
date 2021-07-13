from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

# define simulation volume
volume = {
'fiducial_zmin':-2.7 * units.km,  # the ice sheet at South Pole is 2.7km deep
'fiducial_zmax': 0 * units.km,
'fiducial_rmin': 0 * units.km,
'fiducial_rmax': 4 * units.km}

# generate one event list at 1e19 eV with 1000 neutrinos
generate_eventlist_cylinder('1e19_n10.hdf5', 10, 1e19 * units.eV, 1e19 * units.eV, volume)

