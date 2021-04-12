import numpy as np
import pickle
import bz2
import _pickle as cPickle
import os
from NuRadioReco.utilities import units, io_utilities

filename = 'results/ntr/example_dict_ntr_high_low_pb_80_120.pbz2'
print('filename', filename)
data = []
bz2_file = bz2.BZ2File(filename, 'rb')
data = cPickle.load(bz2_file)

detector_file = data['detector_file']
default_station = data['default_station']
sampling_rate = data['sampling_rate']
station_time = data['station_time']
station_time_random = data['station_time_random']
hardware_response = data['hardware_response']
trigger_name = data['trigger_name']
Vrms_thermal_noise = data['Vrms_thermal_noise']
T_noise = data['T_noise']
T_noise_min_freq = data['T_noise_min_freq']
T_noise_max_freq = data['T_noise_max_freq ']
galactic_noise_n_side = data['galactic_noise_n_side']
galactic_noise_interpolation_frequencies_step = data['galactic_noise_interpolation_frequencies_step']
passband_trigger = data['passband_trigger']
number_coincidences = data['number_coincidences']
coinc_window = data['coinc_window']
order_trigger = data['order_trigger']
trigger_thresholds = data['threshold']
n_iterations = data['iteration']
efficiency = data['efficiency']
trigger_rate = data['trigger_rate']

dic = {}
dic['detector_file'] = detector_file
dic['default_station'] = default_station
dic['sampling_rate'] = sampling_rate
dic['T_noise'] = T_noise
dic['T_noise_min_freq'] = T_noise_min_freq
dic['T_noise_max_freq '] = T_noise_max_freq
dic['Vrms_thermal_noise'] = Vrms_thermal_noise
dic['galactic_noise_n_side'] = galactic_noise_n_side
dic['galactic_noise_interpolation_frequencies_step'] = galactic_noise_interpolation_frequencies_step
dic['station_time'] = station_time
dic['station_time_random'] = station_time_random
dic['passband_trigger'] = passband_trigger
print(passband_trigger)
dic['coinc_window'] = coinc_window
dic['order_trigger'] = order_trigger
dic['number_coincidences'] = number_coincidences
dic['iteration'] = n_iterations
dic['threshold'] = trigger_thresholds
dic['efficiency'] = efficiency
dic['trigger_rate'] = trigger_rate
dic['hardware_response'] = hardware_response
dic['trigger_name'] = trigger_name


filename = 'results/ntr/1_example_dict_ntr_{}_pb_{:.0f}_{:.0f}.pbz2'.format(trigger_name, passband_trigger[0]/units.megahertz, passband_trigger[1]/units.megahertz)
with bz2.BZ2File(filename, 'w') as f:
    cPickle.dump(dic, f)