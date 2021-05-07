import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, io_utilities
import argparse
import sys

'''This script plots the results from the analysis in step 5-6. It makes only sense if you want to compare 
different trigger settings such as passband which are stored in different dict. You can adjust the binning accordingly.'''

parser = argparse.ArgumentParser(description='plot')
parser.add_argument('n_energy_bins', type=int, nargs='?', default = 4, help = 'number of energy bins in the dict')
parser.add_argument('n_zenith_bins', type=int, nargs='?', default = 8, help = 'number of zenith bins in the dict')
parser.add_argument('n_distance_bins', type=int, nargs='?', default = 1, help = 'number of zenith bins in the dict')
parser.add_argument('n_passbands', type=int, nargs='?', default = 1, help = 'number of different passbands or dict files')

args = parser.parse_args()
n_energy_bins = args.n_energy_bins
n_zenith_bins = args.n_zenith_bins
n_distance_bins = args.n_distance_bins
n_passbands = args.n_passbands

number_coincidences_list = []
coinc_windows = []
passband_trigger = np.zeros((n_passbands, 2))
trigger_threshold = np.zeros((n_passbands))
triggered_trigger = np.zeros((n_passbands, n_energy_bins, n_zenith_bins, n_distance_bins)) # number of passbands, energy bins, zenith angle bins, distance bins
trigger_weight = np.zeros((n_passbands, n_energy_bins, n_zenith_bins, n_distance_bins))
masked_events = np.zeros((n_passbands, n_energy_bins, n_zenith_bins, n_distance_bins))
trigger_efficiency = np.zeros((n_passbands, n_energy_bins, n_zenith_bins, n_distance_bins))
trigger_effective_area = np.zeros((n_passbands, n_energy_bins, n_zenith_bins))
trigger_effective_area_err = np.zeros((n_passbands, n_energy_bins, n_zenith_bins))

for pos, file in enumerate(os.listdir('results/air_shower/')):
    filename = os.path.join('results/air_shower/', file)
    if os.path.getsize(filename) > 0:
        data = []
        data = io_utilities.read_pickle(filename, encoding='latin1')
        T_noise = data['T_noise']
        T_noise_min_freq = data['T_noise_min_freq']
        T_noise_max_freq = data['T_noise_max_freq']
        order_trigger = data['order_trigger']
        coinc_window = data['coinc_window']
        number_coincidences = data['number_coincidences']

        trigger_threshold[pos] = data['threshold']
        passband_trigger[pos] = data['passband_trigger']
        trigger_efficiency[pos] = data['trigger_efficiency']
        triggered_trigger[pos] = data['triggered_trigger']
        masked_events[pos] = data['trigger_masked_events']
        trigger_effective_area[pos] = data['trigger_effective_area']
        trigger_effective_area_err[pos] = data['trigger_effective_area_err']
        trigger_weight = data['triggered_trigger_weight']

#choose zenith bins etc here
zenith_start = int(0 /10)
zenith_stop = int(90 /10)

trigger_efficiency = trigger_efficiency[:,:,zenith_start:zenith_stop]
triggered_trigger = triggered_trigger[:,:,zenith_start:zenith_stop]
masked_events = masked_events[:,:,zenith_start:zenith_stop]
trigger_effective_area = trigger_effective_area[:,:,zenith_start:zenith_stop]
trigger_effective_area_err = trigger_effective_area_err[:,:,zenith_start:zenith_stop]
zenith_bin = np.arange(zenith_start,zenith_stop,10)

weight_area = (-(np.cos((zenith_bin+10)*units.deg)**2)/2) - (-(np.cos(zenith_bin*units.deg)**2)/2) #integral of sin*cos is the weight
aeff_zenith = trigger_effective_area * weight_area
total_aeff = np.sum(aeff_zenith, axis=2)  # sum over all zenith bins to get number of cr for each energy bin

x_min = np.min(passband_trigger[:,0])/units.megahertz
x_max = np.max(passband_trigger[:,0])/units.megahertz
y_min = np.min(passband_trigger[:,1])/units.megahertz
y_max = np.max(passband_trigger[:,1])/units.megahertz

binWidth = 10
binLength = 10
x_edges = np.arange(x_min-5, x_max+15, binWidth)
y_edges = np.arange(y_min-5, y_max+15, binLength)

for count_energy, energy_bin in enumerate([1e16, 1e17, 1e18]):
    x = passband_trigger[:,0] / units.megahertz
    y = passband_trigger[:,1] / units.megahertz
    #for count_zenith, zenith_b in enumerate(zenith_bin):
    #z = aeff_zenith
    z = total_aeff[:,count_energy]/units.km**2

    counts, xbins, ybins, image = plt.hist2d(x, y, bins=[x_edges, y_edges], weights=z, vmin=min(z), vmax=max(z), cmap=plt.cm.jet, cmin = 1e-9)
    plt.colorbar(label=r'Effective area [$km^2$]')
    plt.title(r'Energy {}, zenith {}$^\circ$ $-$ {}$^\circ$  '.format(energy_bin, zenith_start*10, zenith_stop*10))
    plt.xlim(x_min - 5, x_max + 5)
    plt.ylim(y_min - 5, y_max + 5)
    plt.xlabel('Lower cutoff frequency [MHz]')
    plt.ylabel('Upper cutoff frequency [MHz]')
    # plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('results/air_shower/fig_pb_thresh_signal_area_energy_{}_{}_{}.png'.format(energy_bin, zenith_start*10, zenith_stop*10))
    plt.close()
