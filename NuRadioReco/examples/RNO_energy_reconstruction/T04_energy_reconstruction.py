import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.framework.parameters import showerParameters as shp
import NuRadioReco.modules.neutrinoEnergyReconstructor
import NuRadioMC.utilities.medium
import NuRadioReco.modules.io.eventReader
import NuRadioReco.detector.generic_detector
import argparse

parser = argparse.ArgumentParser('Use the reconstructed vertex position and electric field to determine the shower energy.')
parser.add_argument(
    '--input_file',
    type=str,
    default='reconstructed_efield.nur',
    help='Name of the input file. Should be output file of T03_electric_field_reco.py'
)
parser.add_argument(
    '--detector_file',
    type=str,
    default='../../detector/RNO_G/RNO_single_station.json',
    help='JSON file containing the detector description. Here, we assume it is written for the GenericDetector class.'

)

args = parser.parse_args()

event_reader = NuRadioReco.modules.io.eventReader.eventReader()
event_reader.begin([args.input_file])

ice = NuRadioMC.utilities.medium.get_ice_model('greenland_simple')
energy_reconstructor = NuRadioReco.modules.neutrinoEnergyReconstructor.neutrinoEnergyReconstructor()
energy_reconstructor.begin([[0, 1, 2, 3, 4, 5]], ice)
det = NuRadioReco.detector.generic_detector.GenericDetector(
    json_filename=args.detector_file,
    antenna_by_depth=False
)

sim_energies = []
rec_energies = []
nu_flavors = []
interaction_types = []

for event in event_reader.run():
    for station in event.get_stations():
        energy_reconstructor.run(event, station, det)

    sim_energy = 0
    interaction_type = None
    nu_flavor = None
    for sim_shower in event.get_sim_showers():
        sim_energy += sim_shower.get_parameter(shp.energy)
        nu_flavor = sim_shower.get_parameter(shp.flavor)
        interaction_type = sim_shower.get_parameter(shp.interaction_type)
    for rec_shower in event.get_showers():
        if rec_shower.has_parameter(shp.energy):
            sim_energies.append(sim_energy)
            nu_flavors.append(nu_flavor)
            interaction_types.append(interaction_type)
            rec_energies.append(rec_shower.get_parameter(shp.energy))

sim_energies = np.array(sim_energies)
rec_energies = np.array(rec_energies)
nu_flavors = np.array(nu_flavors)
interaction_types = np.array(interaction_types)
em_shower_filter = (np.abs(nu_flavors) == 12) & (interaction_types == 'cc')

fig1 = plt.figure(figsize=(5, 8))
ax1_1 = fig1.add_subplot(211)
ax1_1.grid()
ax1_1.set_xscale('log')
ax1_1.set_yscale('log')
ax1_1.scatter(
    sim_energies[~em_shower_filter],
    rec_energies[~em_shower_filter],
    label='hadronic shower'
)
ax1_1.scatter(
    sim_energies[em_shower_filter],
    rec_energies[em_shower_filter],
    label='EM shower'
)
ax1_1.legend()
ax1_1.set_aspect('equal')
ax1_1.set_xlim([1.e16, 1.e20])
ax1_1.set_ylim([1.e16, 1.e20])
ax1_1.set_xlabel('$E_{sim}$ [eV]')
ax1_1.set_ylabel('$E_{rec}$ [eV]')
ax1_2 = fig1.add_subplot(212)
ax1_2.hist(
    [
        (rec_energies / sim_energies)[~em_shower_filter],
        (rec_energies / sim_energies)[em_shower_filter]
    ],
    label=['hadronic shower', 'EM shower'],
    bins=np.power(10, np.arange(-2., 2.1, .1)),
    edgecolor='k',
    stacked=True
)
ax1_2.legend()
ax1_2.set_xscale('log')
ax1_2.set_ylabel('# of events')
ax1_2.set_xlabel('$E_{rec} / E_{sim}$')
ax1_2.grid()
fig1.tight_layout()
fig1.savefig('plots/energy_reconstruction.png')
