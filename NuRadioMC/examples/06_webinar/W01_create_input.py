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

The use of this module keeps the units consistent, and it also provides an easy
way of converting units. Let's say we have a variable called length that has
units of length. Let us define it as equal to 5 kilometres:

length = 5 * units.km

NuRadioReco will store this length unit in metres and perform all the calculations
in metres. If we want to print it in centimeters, we can do:

print(length/units.cm)
"""
from NuRadioReco.utilities import units
"""
We are going to import the function generate_eventlist_cylinder, which creates
(forced) neutrino events from an isotropic flux in a cylinder. This setup is
appropriate to study effective volumes, for instance.
"""
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

# Choose the number of events for this file
n_events = 1000
# Choose the minimum energy for the simulated bin.
Emin = 10 ** 19.5 * units.eV
# Choose the maximum energy for the simulated bin.
Emax = 1e20 * units.eV
"""
We have chosen a half-decade bin with a fairly high energy. We will only
use 1000 events for this simple example, so high energy increases our
chances of trigger.
"""

"""
Now we can choose the parameters for our fiducial cylinder. Every saved event
will have an interaction vertex in the region defined by fiducial_rmin,
fiducial_rmax (cylinder radii), fiducial_zmin, and fiducial_zmax (z coordinates
for the lids of the cylinder).

In this case we will create a cylinder with 4 km of radius and 3 km of height,
stretching from z = 0 km to z = -3 km. One should always make sure that the
cylinder is large enough so that every possible trigger interaction is contained
inside the volume, or, equivalently, the probability of trigger at the edges of
the cylinder is negligible.
"""
volume = {
'fiducial_rmin':0 * units.km,
'fiducial_rmax': 4 * units.km,
'fiducial_zmin':-3 * units.km,
'fiducial_zmax': 0 * units.km}

"""
The generator module allows the user to narrow the zenith band for the incoming
events. The incoming flux will still be isotropic, but only events in a zenith
band will be generated. This is useful for studying the angular response of our
detector in horizontal coordinates, which can then be converted into equatorial
or galactic to estimate the sensitivity of our detector to a given source.

We recommend the zenith bands to be constant in solid angle so that the integration
on that variable is easier to perform. To have constant solid angle bands, the
sky should be divided in intervals of constant cos(theta), where theta is the zenith angle.

The variables that control the width of the zenith band are thetamin and thetamax.
Most of the time, we assume that our detector has azimuthal symmetry. However, if
that is not the case, the azimuthal range can be restricted with phimin and phimax.

The thetamin and thetamax we have chosen here are the ones by default (whole sky).
"""
thetamin = 0 * units.deg
thetamax = 180 * units.deg

"""
The generator lets us specify the flavour of the initial neutrinos by using the
PDG particle codes. 12 is for electron neutrino, 14 is for muon neutrino, and
16 is for tau neutrino. The negative signs represent the antineutrinos. We must
pass the flavour choice as a list of particle codes, and the generator will randomly
choose flavours contained in this list. For instance, to get a 1:1:1 flux, which
is also the one by default, we can write:
"""
flavor = [12, -12, 14, -14, 16, -16]

"""
We choose a name for the file to be generated.
"""
filename = 'input_{:.1e}_{:.1e}.hdf5'.format(Emin, Emax)

"""
And we call the function to generate the events.
"""
generate_eventlist_cylinder(filename, n_events, Emin, Emax,
                            volume,
                            thetamin=thetamin, thetamax=thetamax,
                            flavor=flavor)
