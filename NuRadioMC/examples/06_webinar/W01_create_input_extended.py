"""
This file illustrates how to create an input file for NuRadioMC and introduces
the most basic function and arguments to do so.
"""


"""
One of the first lines in every code that uses NuRadioMC or NuRadioReco should
be an import of the units module in NuRadioReco.utilities. NuRadioMC and NuRadioReco
have a system of base units that is defined in NuRadioReco.utilities.units, and
every dimensional magnitude used with these two codes must be multiplied by a
corresponding unit. All the available units can be consulted in the corresponding
units.py module.
"""
from NuRadioReco.utilities import units
"""
We are going to import the function generate_eventlist_cylinder, which creates
(forced) neutrino events from an isotropic flux in a cylinder. This setup is
appropriate to study effective volumes, for instance. The generator module also
has a generate_surface_muons functions to generate muons on a horizontal surface
to study the effective area for atmospheric neutrino events and estimate the
atmospheric background. See documentation.
"""
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

n_events = 1000
Emin = 5e18 * units.eV
Emax = 1e19 * units.eV
fiducial_rmin = 0 * units.km
fiducial_rmax = 4 * units.km
fiducial_zmin = -3 * units.km
fiducial_zmax = 0 * units.km

thetamin = 0 * units.deg
thetamax = 180 * units.deg

flavor=[12, -12, 14, -14, 16, -16]

n_events_per_file=None
spectrum='log_uniform'
add_tau_second_bang=False

proposal=False
proposal_config='SouthPole'

filename = 'input_{:.1e}_{:.1e}.hdf5'.format(Emin, Emax)

generate_eventlist_cylinder(filename, n_events, Emin, Emax,
                            fiducial_rmin, fiducial_rmax,
                            fiducial_zmin, fiducial_zmax,
                            thetamin=thetamin, thetamax=thetamax,
                            flavor=flavor)
