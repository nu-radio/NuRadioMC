import numpy as np
import datetime
import os, scipy, sys
import glob
import bz2
import _pickle as cPickle
import yaml
import pickle
import matplotlib.pyplot as plt
import NuRadioReco.modules.io.eventReader as eventReader
from NuRadioReco.framework.event import Event
from NuRadioReco.detector.generic_detector import GenericDetector
from NuRadioReco.framework.base_station import BaseStation
from NuRadioReco.framework.base_shower import BaseShower
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.utilities import units, io_utilities
import argparse
import sys

'''This script sorts the triggered events into different energy, zenith, and distance bins. Energy means shower energy, 
 zenith bin means inclination of the shower arrival direction and distance means distance between shower core and station.
  These are important parameters to determine the trigger efficiency.'''


parser = argparse.ArgumentParser(description='Nurfile analyser')
parser.add_argument('result_dict', type=str, nargs='?', default = 'results/ntr/example_dict_ntr_high_low_pb_80_180.pbz2', help = 'settings from the results from threshold analysis')
parser.add_argument('input_filepath', type=str, nargs='?', default = 'output_air_shower_reco/', help = 'input path were results from air shower analysis are stored')
parser.add_argument('energy_bins', type=list, nargs='?', default = [16.5, 20, 6], help = 'energy bins as log() with start, stop, number of bins (np.logspace)')
parser.add_argument('zenith_bins', type=list, nargs='?', default = [0, 100, 10], help = 'zenith bins in deg with start, stop, step (np.arange)')
parser.add_argument('distance_bins', type=int, nargs='?', default = [0, 700, 4000], help = 'distance bins as list')

#please set number of stations within one event here
number_of_sta_in_evt = 72

args = parser.parse_args()
result_dict = args.result_dict
input_filepath = args.input_filepath
energy_bins = args.energy_bins
zenith_bins = args.zenith_bins
distance_bins = args.distance_bins

energy_bins = np.logspace(energy_bins[0], energy_bins[1], energy_bins[2])
energy_bins_low = energy_bins[0:-2]
energy_bins_high = energy_bins[1:-1]

zenith_bins = np.arange(zenith_bins[0], zenith_bins[1], zenith_bins[2]) * units.deg
zenith_bins_low = zenith_bins[0:-2]
zenith_bins_high = zenith_bins[1:-1]

distance_bins_low = np.array(distance_bins[0:-2])
distance_bins_high = np.array(distance_bins[1:-1])

bz2 = bz2.BZ2File(result_dict, 'rb')
data = cPickle.load(bz2)

detector_file = data['detector_file']
default_station = data['default_station']
sampling_rate = data['sampling_rate'] * units.gigahertz
station_time = data['station_time']
station_time_random = data['station_time_random']

Vrms_thermal_noise = data['Vrms_thermal_noise'] * units.volt
T_noise = data['T_noise'] * units.kelvin
T_noise_min_freq = data['T_noise_min_freq'] * units.megahertz
T_noise_max_freq = data['T_noise_max_freq '] * units.megahertz

galactic_noise_n_side = data['galactic_noise_n_side']
galactic_noise_interpolation_frequencies_step = data['galactic_noise_interpolation_frequencies_step']

passband_trigger = data['passband_trigger']
number_coincidences = data['number_coincidences']
coinc_window = data['coinc_window'] * units.ns
order_trigger = data['order_trigger']
trigger_thresholds = data['threshold']
n_iterations = data['iteration']
hardware_response = data['hardware_response']

trigger_rate = data['trigger_rate']
threshold_tested = data['threshold']

zeros = np.where(trigger_rate == 0)[0]
first_zero = zeros[0]
trigger_threshold = threshold_tested[first_zero] * units.volt

nur_file_list = []  # get non corrupted input files with specified passband
i = 0
for nur_file in glob.glob('{}*.nur'.input_filepath):
    if os.path.isfile(nur_file) and str(int(passband_trigger[0]/units.MHz)) + '_' + str(int(passband_trigger[1]/units.MHz)) in nur_file:
        i = i+1
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

det = GenericDetector(json_filename=detector_file, default_station=default_station) # detector file
det.update(datetime.datetime(2019, 10, 1))

evtReader = eventReader.eventReader()
evtReader.begin(filename=nur_file, read_detector=False)
for evt in evtReader.run(): # loop over all events, one event is one station
    num += 1
    event_id = evt.get_id()
    events.append(evt)
    det_position = det.get_absolute_position(det, station_id=default_station)
    sta = evt.get_station(station_id=default_station)
    sim_station = sta.get_sim_station()
    energy.append(sim_station.get_parameter(stnp.cr_energy))
    zenith.append(sim_station.get_parameter(stnp.zenith))  # get zenith for each station
    azimuth.append(sim_station.get_parameter(stnp.azimuth))
    current_weight = sim_station.get_simulation_weight() /(units.m**2)
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
zenith_deg = zenith / units.deg # this is necessary to avoid mistakes due to decimals
distance = np.array(distance)
energy = np.array(energy)
n_events = len(events)

#here we reshape the array in a form that the shower parameter are stored once instead one entry for each station.
# Energy and Zenith are shower parameters.
energy_shower = np.array(energy).reshape(int(len(energy)/number_of_sta_in_evt), number_of_sta_in_evt)[:,0]
zenith_shower = np.array(zenith).reshape(int(len(zenith)/number_of_sta_in_evt), number_of_sta_in_evt)[:,0]

# here we calculate the footprint of the shower, e.g. the area which is covered by the shower
footprint_shower = np.sum(np.array(weight).reshape(int(len(weight)/number_of_sta_in_evt), number_of_sta_in_evt), axis=1)

# here we calculate the area of the footprint where the shower triggers a station
footprint_triggered_area_shower = np.sum(np.array(trigger_status_weight).reshape(int(len(trigger_status_weight)/number_of_sta_in_evt), number_of_sta_in_evt), axis=1)

# here is the trigger status sorted by shower
trigger_status_shower = np.sum(np.array(trigger_status).reshape(int(len(trigger_status)/number_of_sta_in_evt), number_of_sta_in_evt), axis=1)

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
        mask_z = (zenith_deg >= zenith_bin_low/units.deg) & (zenith_deg < zenith_bin_high/units.deg) # choose zenith bin
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

        for dim_2, distance_bin_low, distance_bin_high in zip(range(len(distance_bins_low)), distance_bins_low, distance_bins_high):
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

dic = {}
dic['T_noise'] = T_noise
dic['threshold'] = trigger_threshold
dic['passband_trigger'] = passband_trigger
dic['coinc_window'] = coinc_window
dic['order_trigger'] = order_trigger
dic['number_coincidences'] = number_coincidences
dic['T_noise_min_freq'] = T_noise_min_freq
dic['T_noise_max_freq'] = T_noise_max_freq
dic['max_distance'] = mask_d

dic['trigger_effective_area'] = trigger_effective_area_e
dic['trigger_effective_area_err'] = trigger_effective_area_err_e
dic['trigger_masked_events'] = np.nan_to_num(masked_events_e)
dic['triggered_trigger'] = np.nan_to_num(triggered_trigger_e)
dic['trigger_efficiency'] = np.nan_to_num(trigger_efficiency_e)
dic['triggered_trigger_weight'] = triggered_trigger_weight_e
dic['distance_bins_low'] = distance_bins_low
dic['energy_bins_low'] = energy_bins_low
dic['zenith_bins_low'] = zenith_bins_low
dic['distance_bins_high'] = distance_bins_high
dic['energy_bins_high'] = energy_bins_high
dic['zenith_bins_high'] = zenith_bins_high

if os.path.isdir('results/air_shower/') == False:
    os.mkdir('results/air_shower/')

with open('results/air_shower/dict_air_shower_pb_{:.0f}_{:.0f}_e{}_z{}_d{}_{}.pickle'.format(passband_trigger[0]/units.megahertz, passband_trigger[1]/units.megahertz, len(energy_bins_low), len(zenith_bins_low), len(distance_bins_low), max(distance_bins)), 'wb') as pickle_out:
    pickle.dump(dic, pickle_out)
