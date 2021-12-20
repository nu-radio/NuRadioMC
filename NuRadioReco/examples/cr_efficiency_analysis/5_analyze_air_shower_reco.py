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

'''
This script sorts the triggered events into different energy, zenith, and distance bins. 
Energy means shower energy, 
zenith bin means inclination of the shower arrival direction and 
distance means distance between shower core and station.
These are important parameters to determine the trigger efficiency.
'''

parser = argparse.ArgumentParser(description='Nurfile analyser')
parser.add_argument('--config_file', type=str, nargs='?',
                    default='config/air_shower/final_config_envelope_trigger_1Hz_2of3_22.46mV.json',
                    help='settings from threshold analysis')
parser.add_argument('--directory', type=str, nargs='?',
                    default='output_air_shower_reco/',
                    help='path were results from air shower analysis are stored')
parser.add_argument('--condition', type=str, nargs='?', default='envelope_trigger_0Hz_3of4',
                    help='string which should be in dict name')
parser.add_argument('--energy_binning', type=list, nargs='?',
                    default=[16.5, 20, 8], help='energy bins as log() with start, stop, number of bins (np.logspace)')
parser.add_argument('--zenith_binning', type=list, nargs='?',
                    default=[0, 100, 10], help='zenith bins in deg with start, stop, step (np.arange)')
parser.add_argument('--distance_binning', type=int, nargs='?',
                    default=[0, 700, 4000], help='distance bin edges as list')
parser.add_argument('--number_of_sta_in_evt', type=int, nargs='?',
                    default=72, help='number of stations in one event')

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

# the entries of this list are defined in the input argument distance_bins.
distance_bins_low = np.array(distance_binning[0:-1])
distance_bins_high = np.array(distance_binning[1:])
logger.info(f"Use distance bins low:{distance_bins_low} to high:{distance_bins_high} m")

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
trigger_status_weight = []  # trigger status per event station and threshold with weighting
trigger_in_station = []  # name of trigger

weight = []
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
    current_weight = sim_station.get_simulation_weight() / (units.m**2)
    weight.append(current_weight)
    trigger_in_station.append(sta.get_triggers())  # get trigger for each station
    trigger_status.append(sta.has_triggered())
    trigger_status_weight.append(sta.has_triggered() * current_weight)

    for sho in evt.get_sim_showers():
        core = sho.get_parameter(shp.core)
        distance.append(np.sqrt(
            ((core[0] - det_position[0])**2)
            + (core[1] - det_position[1])**2))

trigger_status = np.array(trigger_status)
trigger_status_weight = np.array(trigger_status_weight)
zenith = np.array(zenith)
zenith_deg = zenith / units.deg  # this is necessary to avoid mistakes due to decimals
distance = np.array(distance)
energy = np.array(energy)

# here we reshape the array in a form that the shower parameter are stored once instead one entry for each station.
# Energy and Zenith are shower parameters.
energy_shower = np.array(energy).reshape(int(len(energy)/number_of_sta_in_evt), number_of_sta_in_evt)[:, 0]
zenith_shower = np.array(zenith).reshape(int(len(zenith)/number_of_sta_in_evt), number_of_sta_in_evt)[:, 0]

# here we calculate the footprint of the shower, e.g. the area which is covered by the shower
footprint_shower = np.sum(np.array(weight).reshape(int(len(weight)/number_of_sta_in_evt), number_of_sta_in_evt), axis=1)

# here we calculate the area of the footprint where the shower triggers a station
footprint_triggered_area_shower = np.sum(np.array(trigger_status_weight).
                                         reshape(int(len(trigger_status_weight)/number_of_sta_in_evt),
                                                 number_of_sta_in_evt), axis=1)

# here is the trigger status sorted by shower
trigger_status_shower = np.sum(np.array(trigger_status).
                               reshape(int(len(trigger_status)/number_of_sta_in_evt),
                                       number_of_sta_in_evt), axis=1)

# here we create empty array which will be filled in the following loop. The first axis contains all parameters in
# the energy bin, the second axis the zenthis bins and the third axis the distance bins

# number of true triggered trigger in each bin
triggered_trigger_e = np.zeros((len(energy_bins_low), len(zenith_bins_low), len(distance_bins_low)))
# the weight (in this case this is the area) of true triggered for each bin
triggered_trigger_weight_e = np.zeros((len(energy_bins_low), len(zenith_bins_low)))
# events within a bin
masked_events_e = np.zeros((len(energy_bins_low), len(zenith_bins_low), len(distance_bins_low)))
# trigger efficiency in each bin
trigger_efficiency_e = np.zeros((len(energy_bins_low), len(zenith_bins_low), len(distance_bins_low)))
# effective area of each bin (= area inwhich a event within that bin triggers)
trigger_effective_area_e = np.zeros((len(energy_bins_low), len(zenith_bins_low)))
# error of the effective area of each energy bin
trigger_effective_area_err_e = np.zeros((len(energy_bins_low), len(zenith_bins_low)))


for dim_0, energy_bin_low, energy_bin_high in zip(range(len(energy_bins_low)), energy_bins_low, energy_bins_high):
    mask_e = (energy >= energy_bin_low) & (energy < energy_bin_high)  # choose one energy bin
    n_events_masked_e = np.sum(mask_e)  # number of events in that energy bin

    for dim_1, zenith_bin_low, zenith_bin_high in zip(range(len(zenith_bins_low)), zenith_bins_low, zenith_bins_high):
        # choose zenith bin
        mask_z = (zenith_deg >= zenith_bin_low/units.deg) & (zenith_deg < zenith_bin_high/units.deg)
        # trigger in in one energy bin and one zenith bin (ez) (values depend on the loop)
        mask_ez = mask_e & mask_z
        masked_trigger_status_ez = trigger_status[mask_ez]
        masked_trigger_status_weight_ez = trigger_status_weight[mask_ez]
        n_events_masked_ez = np.sum(mask_ez)  # number of events in that energy and zenith bin

        # mask ez
        # number of triggered true trigger in energy and zentih bin
        triggered_trigger_ez = np.sum(masked_trigger_status_ez, axis=0)
        triggered_trigger_weight = np.sum(masked_trigger_status_weight_ez, axis=0)
        # fraction of events in this energy and zeniths bin that triggered true
        trigger_efficiency_ez = triggered_trigger_ez / n_events_masked_ez

        # reshape the array. zenith and energy are the same for all stations in the shower
        triggered_trigger_weight_shower = masked_trigger_status_weight_ez.reshape(
            int(len(masked_trigger_status_weight_ez) / number_of_sta_in_evt), number_of_sta_in_evt)

        # array with the effective area of all showers in this energy and zenith bin
        triggered_trigger_weight_shower_sum = np.sum(triggered_trigger_weight_shower, axis=1)
        trigger_effective_area = np.mean(triggered_trigger_weight_shower_sum)
        trigger_effective_area_err = np.std(triggered_trigger_weight_shower_sum)

        # set the values of this energy bin in an array for all bins
        triggered_trigger_weight_e[dim_0, dim_1] = triggered_trigger_weight
        trigger_effective_area_e[dim_0, dim_1] = trigger_effective_area
        trigger_effective_area_err_e[dim_0, dim_1] = trigger_effective_area_err

        for dim_2, distance_bin_low, distance_bin_high in zip(range(len(distance_bins_low)),
                                                              distance_bins_low, distance_bins_high):
            # bin for each event, since distance between shower core and station differs for each station
            # choose here, if distances should be in circles or rings,
            # so if it should include everything with in or only an interval
            mask_d = (distance < distance_bin_high)
            mask_ezd = mask_ez & mask_d
            masked_trigger_status_ezd = trigger_status[mask_ezd]
            masked_trigger_status_weight_ezd = trigger_status_weight[mask_ezd]
            n_events_masked_ezd = np.sum(mask_ezd)

            # mask ezd, no a eff
            triggered_trigger_ezd = np.sum(masked_trigger_status_ezd, axis=0)
            trigger_efficiency_ezd = triggered_trigger_ezd / n_events_masked_ezd

            masked_events_e[dim_0, dim_1, dim_2] = n_events_masked_ezd
            triggered_trigger_e[dim_0, dim_1, dim_2] = triggered_trigger_ezd
            trigger_efficiency_e[dim_0, dim_1, dim_2] = trigger_efficiency_ezd

with open(args.config_file, 'r') as fp:
    cfg = json.load(fp)

dic = {}
for key in cfg:
    dic[key] = cfg[key]
dic['detector_file'] = []
dic['default_station'] = default_station
dic['energy_bins_low'] = energy_bins_low
dic['energy_bins_high'] = energy_bins_high
dic['zenith_bins_low'] = zenith_bins_low
dic['zenith_bins_high'] = zenith_bins_high
dic['distance_bins_high'] = distance_bins_high
dic['distance_bins_low'] = distance_bins_low
dic['trigger_effective_area'] = np.nan_to_num(trigger_effective_area_e)
dic['trigger_effective_area_err'] = np.nan_to_num(trigger_effective_area_err_e)
dic['trigger_masked_events'] = np.nan_to_num(masked_events_e)
dic['triggered_trigger'] = np.nan_to_num(triggered_trigger_e)
dic['trigger_efficiency'] = np.nan_to_num(trigger_efficiency_e)
dic['triggered_trigger_weight'] = triggered_trigger_weight_e

if cfg['hardware_response']:
    out_dir = 'results/air_shower_{}_trigger_{:.0f}Hz_{}of{}_{:.2f}mV'.format(
        cfg['trigger_name'],
        cfg['target_global_trigger_rate'] / units.Hz,
        cfg['number_coincidences'],
        cfg['total_number_triggered_channels'],
        cfg['final_threshold'] / units.millivolt
        )

else:
    out_dir = 'results/air_shower_{}_trigger_{:.0f}Hz_{}of{}_{:.2f}V'.format(
        cfg['trigger_name'],
        cfg['target_global_trigger_rate'] / units.Hz,
        cfg['number_coincidences'],
        cfg['total_number_triggered_channels'],
        cfg['final_threshold']
        )

os.makedirs(out_dir, exist_ok=True)

output_file = 'dict_air_shower_e{}_z{}_d{}_{}.json'.format(
        len(energy_bins_low),
        len(zenith_bins_low),
        len(distance_bins_low),
        max(distance_binning))

with open(os.path.join(out_dir, output_file), 'w') as outfile:
    json.dump(dic, outfile, cls=hcr.NumpyEncoder, indent=4)
