import numpy as np
import matplotlib.pyplot as plt
import os, sys
from radiotools import helper as hp
from NuRadioReco.utilities import units, io_utilities
import pickle
import argparse

'''This script reads all dicts for one passbands and writes one new dict which contains then the information for 2e6 interations.
The last part creates a plot of the final dict. If you just want to plot the dict, comment the first part. 
It can be used for results from 2_.. and 2a_...'''

parser = argparse.ArgumentParser(description='Noise Trigger Rate')
parser.add_argument('passband_low', type=int, nargs='?', default = 80, help = 'passband low to check')
parser.add_argument('passband_high', type=int, nargs='?', default = 180, help = 'passband high to check')
parser.add_argument('number_of_files', type=int, nargs='?', default = 100, help = 'number of n_files to loop over')

args = parser.parse_args()
passband_low = args.passband_low
passband_high = args.passband_high
number_of_files = args.number_of_files

n_files = number_of_files

input_filename = 'output_threshold_final/final_threshold_high_low_pb_{:.0f}_{:.0f}_i20000_1.pickle'.format(passband_low,
                                                                                                      passband_high)

data = []
data = io_utilities.read_pickle(input_filename, encoding='latin1')

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

trigger_status = np.zeros((n_files, 20000, len(trigger_thresholds)))
triggered_true = np.zeros((n_files, len(trigger_thresholds)))
triggered_all = np.zeros_like(triggered_true)
trigger_efficiency = np.zeros_like(triggered_true)
trigger_rate = np.zeros_like(triggered_true)


for i_file in range(number_of_files):
    input_filename = 'output_threshold_final/final_threshold_{}_pb_{:.0f}_{:.0f}_i20000_{}.pickle'.format(trigger_name, passband_low,
                                                                                                      passband_high,
                                                                                            i_file)
    data = []
    data = io_utilities.read_pickle(input_filename, encoding='latin1')
    #print(data)
    trigger_efficiency[i_file] = data['efficiency']
    trigger_rate[i_file] = data['trigger_rate']
    triggered_true[i_file]= data['triggered_true']
    triggered_all[i_file] = data['triggered_all']
    trigger_status[i_file] = data['trigger_status']


triggered_true_all = np.sum(triggered_true, axis=0)
trigger_efficiency_all = np.sum(trigger_efficiency, axis=0) / n_files
trigger_rate_all = np.sum(trigger_rate, axis=0) / n_files
iterations = n_iterations * n_files

print('triggered_trigger', triggered_true_all)
print('trigger efficiency all', trigger_efficiency_all)
print('trigger rate all [Hz]', trigger_rate_all/units.Hz)
print(trigger_status.shape)

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
dic['coinc_window'] = coinc_window
dic['order_trigger'] = order_trigger
dic['number_coincidences'] = number_coincidences
dic['iteration'] = iterations
dic['threshold'] = trigger_thresholds
dic['trigger_status'] = trigger_status
dic['triggered_true'] = triggered_true_all
dic['efficiency'] = trigger_efficiency_all
dic['trigger_rate'] = trigger_rate_all
dic['hardware_response'] = hardware_response
dic['trigger_name'] = trigger_name

#print(dic)

with open('results/ntr/dict_ntr_{}_pb_{:.0f}_{:.0f}.pickle'.format(trigger_name, passband_trigger[0]/units.megahertz, passband_trigger[1]/units.megahertz),
          'wb') as pickle_out:
    pickle.dump(dic, pickle_out)


filename = 'results/ntr/dict_ntr_{}_pb_{:.0f}_{:.0f}.pickle'.format(trigger_name, passband_low, passband_high)

data = io_utilities.read_pickle(filename, encoding='latin1')

efficiency= data['efficiency']
trigger_rate = data['trigger_rate']
trigger_thresholds = np.array(data['threshold'])
passband_trigger = data['passband_trigger']
n_iterations = data['iteration']
T_noise = data['T_noise']
coinc_window = data['coinc_window']
order_trigger = data['order_trigger']
iterations = data['iteration']
trigger_name = data['trigger_name']


print('threshold', trigger_thresholds)
print('efficiency', efficiency)
print('trigger rate', trigger_rate/units.Hz)
print('passband', passband_trigger/units.megahertz)

from scipy.interpolate import interp1d


x = trigger_thresholds/units.microvolt
y = trigger_rate/units.Hz
f1 = interp1d(x, y, 'cubic')

print(x)
thresh = 25.273
print('f1',f1(thresh))
xnew = np.linspace(trigger_thresholds[0]/units.microvolt, trigger_thresholds[12]/units.microvolt)
print(xnew)

plt.plot(trigger_thresholds/units.microvolt, trigger_rate/units.Hz, marker='x', label= 'Noise trigger rate', linestyle='none')
plt.plot(xnew, f1(xnew), '--', label='interp1d f({:.4f}) = {:.4f} Hz'.format(thresh, f1(thresh)))
plt.title('{}, passband = {} MHz, iterations = {:.1e}'.format(trigger_name ,passband_trigger/units.megahertz, iterations))
plt.xlabel(r'Threshold [$\mu$V]', fontsize=18)
#plt.hlines(0, trigger_thresholds[0]/units.microvolt, trigger_thresholds[5]/units.microvolt, label='0 Hz')
plt.ylabel('Trigger rate [Hz]', fontsize=18)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.yscale('log')
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('results/fig_ntr_{}_passband_{:.0f}_{:.0f}.png'.format(trigger_name, passband_trigger[0]/units.megahertz, passband_trigger[1]/units.megahertz))
plt.close()
