import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.neutrinoEnergyReconstructor
import NuRadioMC.utilities.medium
import NuRadioReco.modules.io.eventReader
import NuRadioReco.detector.generic_detector
import argparse

parser = argparse.ArgumentParser()
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
parser.add_argument('--det_default_station', type=int, default=11, help='Default station ID for the GenericDetector')
parser.add_argument('--det_default_channel', type=int, default=0, help='Default channel ID for the GenericDetector')

args = parser.parse_args()

event_reader = NuRadioReco.modules.io.eventReader.eventReader()
event_reader.begin([args.input_file])

ice = NuRadioMC.utilities.medium.get_ice_model('greenland_simple')
energy_reconstructor = NuRadioReco.modules.neutrinoEnergyReconstructor.neutrinoEnergyReconstructor()
energy_reconstructor.begin([[0, 1, 2, 3, 4, 5]], ice)
det = NuRadioReco.detector.generic_detector.GenericDetector(
    json_filename=args.detector_file,
    default_station=args.det_default_station,
    default_channel=args.det_default_channel,
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
    for sim_shower in event.get_sim_showers():
        print(sim_shower.get_parameter(shp.flavor))
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
# em_shower_filter = (np.abs(nu_flavors) == 12) & (interaction_types == 'c')

fig1 = plt.figure(figsize=(8, 8))
ax1_1 = fig1.add_subplot(111)
ax1_1.grid()
ax1_1.set_xscale('log')
ax1_1.set_yscale('log')
ax1_1.scatter(
    sim_energies,
    rec_energies,
    label='hadronic shower'
)
ax1_1.set_xlim([1.e16, 1.e20])
ax1_1.set_ylim([1.e16, 1.e20])
fig1.tight_layout()
fig1.savefig('plots/energy_reconstruction.png')
