"""
This file generates events for calculating the efficiency of the alias phased
array trigger as a function of the SNR. The events are localised by default at
a distance of 300 m from the phased array, which is supposed to lie at a depth
of 100 m.

The number of events and the filename can be specified as input parameters.

The neutrino events are muon neutrinos chosen to have an inelasticity of 1 and
to interact via CC current, since we're only interested in having realistic signals
with a definite amplitude for our analysis.

The energy is 1 EeV and the vertex position and arrival directions are taken to be
in the region where most triggers are expected, so as to minimise the computation
time and so that a sizeable portion of the events trigger. These parameters can be
changed by the user in the present file.
"""

from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import write_events_to_hdf5
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generates a simple event list for alias SNR studies')
parser.add_argument('--filename', type=str, help='Output filename', default='input_alias_SNR.hdf5')
parser.add_argument('--n_events', type=int, help='Number of events', default=2500)
args = parser.parse_args()

filename = args.filename
n_events = args.n_events

attributes = {}
data_sets = {}

attributes["n_events"] = n_events
energy = 1e18 * units.eV
attributes["fixed_energy"] = energy
inelasticity = 1
flavor = 14
phimin = 0 * units.deg
phimax = 0.01 * units.deg

delta_zenith = 2 * units.deg
z_ant = -100 * units.m
distance = 1 * units.km
vertex_angle_min = 90 * units.deg
vertex_angle_max = 145 * units.deg
vertex_angles = np.random.uniform(vertex_angle_min, vertex_angle_max, n_events)

data_sets["yy"] = np.zeros(n_events, dtype=np.float)
data_sets["zz"] = z_ant + distance * np.cos(vertex_angles)
data_sets["xx"] = distance * np.sin(vertex_angles)

thetamin = 65 * units.deg
thetamax = 100 * units.deg
data_sets["zeniths"] = np.arccos(np.random.uniform(np.cos(thetamax), np.cos(thetamin), n_events))

data_sets["azimuths"] = np.random.uniform(phimin, phimax, n_events)
data_sets["event_group_ids"] = np.arange(n_events)
data_sets["shower_ids"] = np.arange(n_events)
data_sets["n_interaction"] = np.ones(n_events, dtype=int)
data_sets["vertex_times"] = np.zeros(n_events, dtype=np.float)
data_sets["flavors"] = np.ones(n_events, dtype=int) * flavor
data_sets["energies"] = np.ones(n_events, dtype=int) * energy
data_sets["interaction_type"] = ['cc'] * n_events
data_sets["inelasticity"] = np.ones(n_events, dtype=int) * inelasticity
data_sets["shower_type"] = ['had'] * n_events
data_sets["shower_energies"] = data_sets['energies'] * data_sets["inelasticity"]

write_events_to_hdf5(filename, data_sets, attributes,
                     n_events_per_file=n_events,
                     start_file_id=0)
