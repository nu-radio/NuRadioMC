import numpy as np
import os
import glob
import helper_cr_eff as hcr
import json
import NuRadioReco.modules.io.eventReader as eventReader
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.utilities import units
import argparse
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='Nurfile analyser')
parser.add_argument('--directory', type=str, nargs='?',
                    default='output_air_shower_reco/',
                    help='path were results from air shower analysis are stored')
parser.add_argument('--condition', type=str, nargs='?', default='',
                    help='string which should be in dict name')
parser.add_argument('--energy_binning', type=list, nargs='?',
                    default=[16.5, 20, 8], help='energy bins as log() with start, stop, number of bins (np.logspace)')
parser.add_argument('--zenith_binning', type=list, nargs='?',
                    default=[0, 100, 10], help='zenith bins in deg with start, stop, step (np.arange)')

args = parser.parse_args()
energy_binning = args.energy_binning
zenith_binning = args.zenith_binning
distance_binning = args.distance_binning
number_of_sta_in_evt = args.number_of_sta_in_evt

# the entries of this list are defined in the input argument energy_bins.
# [0] is the start value, [1] is the stop value, [2] is the number of samples generated
energy_bins = np.logspace(energy_binning[0], energy_binning[1], energy_binning[2])
energy_bins_low = energy_bins[0:-1]
energy_bins_high = energy_bins[1:]
logger.info(f"Use energy bins {energy_bins} eV")

# the entries of this list are defined in the input argument zenith_bins.
# [0] is the start value, [1] is the stop value, [2] is step size
zenith_bins = np.arange(zenith_binning[0], zenith_binning[1], zenith_binning[2]) * units.deg
zenith_bins_low = zenith_bins[0:-1]
zenith_bins_high = zenith_bins[1:]
logger.info(f"Use zenith bins {zenith_bins/units.deg} deg")

nur_file_list = []  # get non corrupted input files with specified passband
i = 0
for nur_file in glob.glob('{}*.nur'.format(args.directory)):
    if os.path.isfile(nur_file) and str(args.condition) in nur_file:
        i += 1
        if os.path.getsize(nur_file) > 0:
            nur_file_list.append(nur_file)

n_files = len(nur_file_list)

energy = []
zenith = []
azimuth = []
distance = []
events = []
trigger_status = []  # trigger status per event station and threshold
trigger_in_station = []  # name of trigger

num = 0

evtReader = eventReader.eventReader()
evtReader.begin(filename=nur_file, read_detector=True)
det = evtReader.get_detector()
default_station = det.get_station_ids()[0]
for evt in evtReader.run():  # loop over all events, one event is one station
    num += 1
    event_id = evt.get_id()
    events.append(evt)
    det_position = det.get_absolute_position(station_id=default_station)
    sta = evt.get_station(station_id=default_station)
    sim_station = sta.get_sim_station()
    energy.append(sim_station.get_parameter(stnp.cr_energy))
    zenith.append(sim_station.get_parameter(stnp.zenith))  # get zenith for each station
    azimuth.append(sim_station.get_parameter(stnp.azimuth))
    trigger_in_station.append(sta.get_triggers())  # get trigger for each station
    trigger_status.append(sta.has_triggered())

    for sho in evt.get_sim_showers():
        core = sho.get_parameter(shp.core)
        distance.append(np.sqrt(
            ((core[0] - det_position[0])**2)
            + (core[1] - det_position[1])**2))

trigger_status = np.array(trigger_status)
zenith = np.array(zenith)
zenith_deg = zenith / units.deg  # this is necessary to avoid mistakes due to decimals
distance = np.array(distance)
energy = np.array(energy)


# here we create empty array which will be filled in the following loop. The first axis contains all parameters in
# the energy bin and the second axis the zenthis bins

triggered_trigger_e = np.zeros((len(energy_bins_low), len(zenith_bins_low)))
masked_events_e = np.zeros((len(energy_bins_low), len(zenith_bins_low)))
trigger_efficiency_e = np.zeros((len(energy_bins_low), len(zenith_bins_low)))

for dim_0, energy_bin_low, energy_bin_high in zip(range(len(energy_bins_low)), energy_bins_low, energy_bins_high):
    mask_e = (energy >= energy_bin_low) & (energy < energy_bin_high)  # choose one energy bin
    n_events_masked_e = np.sum(mask_e)  # number of events in that energy bin

    for dim_1, zenith_bin_low, zenith_bin_high in zip(range(len(zenith_bins_low)), zenith_bins_low, zenith_bins_high):
        # choose zenith bin
        mask_z = (zenith_deg >= zenith_bin_low/units.deg) & (zenith_deg < zenith_bin_high/units.deg)
        # trigger in in one energy bin and one zenith bin (ez) (values depend on the loop)
        mask_ez = mask_e & mask_z
        masked_trigger_status_ez = trigger_status[mask_ez]
        n_events_masked_ez = np.sum(mask_ez)  # number of events in that energy and zenith bin

        # number of triggered true trigger in energy and zentih bin
        triggered_trigger_ez = np.sum(masked_trigger_status_ez, axis=0)
        # fraction of events in this energy and zeniths bin that triggered true
        trigger_efficiency_ez = triggered_trigger_ez / n_events_masked_ez


dic = {}
dic['default_station'] = default_station
dic['energy_bins_low'] = energy_bins_low
dic['energy_bins_high'] = energy_bins_high
dic['zenith_bins_low'] = zenith_bins_low
dic['zenith_bins_high'] = zenith_bins_high
dic['trigger_masked_events'] = np.nan_to_num(masked_events_e)
dic['triggered_trigger'] = np.nan_to_num(triggered_trigger_e)
dic['trigger_efficiency'] = np.nan_to_num(trigger_efficiency_e)


out_dir = 'results'
os.makedirs(out_dir, exist_ok=True)

output_file = 'dict_air_shower_e{}_z{}.json'.format(
        len(energy_bins_low),
        len(zenith_bins_low))

with open(os.path.join(out_dir, output_file), 'w') as outfile:
    json.dump(dic, outfile, cls=hcr.NumpyEncoder, indent=4)
