"""
This file illustrates how to create an input file for NuRadioMC and introduces
the most basic function and arguments, with expanded comments.
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
appropriate to study effective volumes, for instance. The generator module also
has a generate_surface_muons functions to generate muons on a horizontal surface
to study the effective area for atmospheric neutrino events and estimate the
atmospheric background. See documentation.
"""
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

# Choose the number of events for this file
n_events = 1000
# Choose the minimum energy for the simulated bin.
Emin = 10 ** 18.5 * units.eV
# Choose the maximum energy for the simulated bin.
Emax = 1e19 * units.eV

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
Simulations of effective volumes should contain around hundreds of thousands or
millions of events to have large enough statistics. We recommend the use of
clusters to simulate each energy bin separately, and the compilation of the
C++ version of the ray tracer (see SignalProp/CPPAnalyticRayTracing/README.md).
But sometimes, if the array is too large or the detector is so complex that its
simulation takes more time than what is desirable, we can use the argument
n_events_per_file to split the files. Let us choose 250 events per file:
"""
n_events_per_file = 250

"""
The energy of each event is drawn from a random distribution. By default, this
distribution is flat with the logarithm of the energy ('log_uniform'), although
the user can choose a variety of fluxes: a power law (E-X, with X a float),
the IceCube-measured neutrino flux ('IceCube-nu-2017'), or a cosmogenic neutrino
model by van Vliet et al. 2019 ('GZK-1'). See documentation and generator.py
for more info.
"""
spectrum = 'log_uniform'

"""
If the user wants to specify the full volume manually, these keyword arguments can
be used. We recommend increasing the full_zmin manually and then setting
add_tau_second_bang=True to increase the radius of the cylinder. This also works
when dealing with PROPOSAL simulations.
"""
volume['full_rmax'] = 6 * units.km
volume['full_rmin'] = 0 * units.km
volume['full_zmax'] = 0 * units.km
volume['full_zmin'] = -4 * units.km

"""
NuRadioMC has a more rigorous way of dealing with showers created by the secondary
interactions of muons and taus created by neutrinos. We can use the PROPOSAL lepton
propagation code, provided the user has installed it and the python module is
operative. See https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/INSTALL.md
or try 'pip install proposal'
Setting proposal=True suffices to properly simulate all the lepton-derived interactions,
but be aware that this might slow down the generation and it is advisable
to use a cluster to generate input files.
"""
proposal = True

"""
proposal_config defines the medium used for the lepton propagation by PROPOSAL.
We have four media by default: 'SouthPole', 'MooresBay', 'Greenland' and 'InfIce',
or the user can specify a path to their own files. If the user wants one of the
default media, they have to take the corresponding config*.json.sample file,
modify the paths to the PROPOSAL tables to a valid path in their environment
where they want to save the tables used by PROPOSAL, and then rename the config
file by removing the final .sample ending. So, for instance, the file
config_PROPOSAL_greeenland.json.sample should be renamed to config_PROPOSAL_greeenland.json.

In this case we are going to use the Proposal config file in the present folder
"""
proposal_config = 'config_PROPOSAL_greenland.json'

"""
start_event_id is the event id of the first event generated by the generator file.
In principle, it can be anything. However, when using PROPOSAL, if several jobs
are used to create events for the same energy bin and zenith band, be sure to
change the start_event_id so that different files referring to the same bin.
"""
start_event_id = 0

"""
We choose a name for the file to be generated.
"""
filename = 'input_{:.1e}_{:.1e}.hdf5'.format(Emin, Emax)

generate_eventlist_cylinder(filename, n_events, Emin, Emax,
                            volume,
                            thetamin=thetamin,
                            thetamax=thetamax,
                            flavor=flavor,
                            n_events_per_file=n_events_per_file,
                            spectrum=spectrum,
                            proposal=proposal, proposal_config=proposal_config,
                            start_event_id=start_event_id)
