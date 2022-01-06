import numpy as np
import json
import os
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, io_utilities
import argparse

'''This script plots the results from the analysis for different passbands. It makes only sense if you want to compare
different trigger settings such as passband which are stored in different dict. You can adjust the binning accordingly.'''

parser = argparse.ArgumentParser(description='plot')
parser.add_argument('n_energy_bins', type=int, nargs='?', default=7, help='number of energy bins in the dict')
parser.add_argument('n_zenith_bins', type=int, nargs='?', default=9, help='number of zenith bins in the dict')
parser.add_argument('n_distance_bins', type=int, nargs='?', default=2, help='number of zenith bins in the dict')
parser.add_argument('n_passbands', type=int, nargs='?', default=1, help='number of different passbands or dict files')

# choose index of zenith bins for plotting here
zenith_start = int(0 /10)
zenith_stop = int(90 /10)

args = parser.parse_args()
n_energy_bins = args.n_energy_bins
n_zenith_bins = args.n_zenith_bins
n_distance_bins = args.n_distance_bins
n_passbands = args.n_passbands

number_coincidences_list = []
coinc_windows = []
passband_trigger = np.zeros((n_passbands, 2))
trigger_threshold = np.zeros((n_passbands))

# number of passbands, energy bins, zenith angle bins, distance bins
triggered_trigger = np.zeros((n_passbands, n_energy_bins, n_zenith_bins, n_distance_bins))
trigger_weight = np.zeros((n_passbands, n_energy_bins, n_zenith_bins, n_distance_bins))
masked_events = np.zeros((n_passbands, n_energy_bins, n_zenith_bins, n_distance_bins))
trigger_efficiency = np.zeros((n_passbands, n_energy_bins, n_zenith_bins, n_distance_bins))
trigger_effective_area = np.zeros((n_passbands, n_energy_bins, n_zenith_bins))
trigger_effective_area_err = np.zeros((n_passbands, n_energy_bins, n_zenith_bins))

pos = 0
for file in os.listdir('results/air_shower/'):
    filename = os.path.join('results/air_shower/', file)
    if filename.endswith('.json'):
        if os.path.getsize(filename) > 0:
            with open(filename, 'r') as fp:
                data = json.load(fp)
            T_noise = data['T_noise']
            T_noise_min_freq = data['T_noise_min_freq']
            T_noise_max_freq = data['T_noise_max_freq']
            order_trigger = data['order_trigger']
            coinc_window = data['coinc_window']
            number_coincidences = data['number_coincidences']

            energy_bins_low = data['energy_bins_low']
            energy_bins_high = data['energy_bins_high']
            zenith_bins_low = data['zenith_bins_low']
            zenith_bins_high = data['zenith_bins_high']

            trigger_threshold[pos] = data['threshold']
            passband_trigger[pos] = data['passband_trigger']
            trigger_efficiency[pos] = data['trigger_efficiency']
            triggered_trigger[pos] = data['triggered_trigger']
            masked_events[pos] = data['trigger_masked_events']
            trigger_effective_area[pos] = data['trigger_effective_area']
            trigger_effective_area_err[pos] = data['trigger_effective_area_err']
            trigger_weight = data['triggered_trigger_weight']

            pos += 1

trigger_efficiency = trigger_efficiency[:, :, zenith_start:zenith_stop]
triggered_trigger = triggered_trigger[:, :, zenith_start:zenith_stop]
masked_events = masked_events[:, :, zenith_start:zenith_stop]
trigger_effective_area = trigger_effective_area[:, :, zenith_start:zenith_stop]
trigger_effective_area_err = trigger_effective_area_err[:, :, zenith_start:zenith_stop]
zenith_bin = np.arange(zenith_start, zenith_stop, 10)

weight_area = (-(np.cos((zenith_bin+10)*units.deg)**2)/2) - (-(np.cos(zenith_bin*units.deg)**2)/2) #integral of sin*cos is the weight
aeff_zenith = trigger_effective_area * weight_area
total_aeff = np.nansum(aeff_zenith, axis=2)  # sum over all zenith bins to get number of cr for each energy bin

print(aeff_zenith,aeff_zenith.shape, total_aeff)
x_min = np.min(passband_trigger[:, 0])/units.megahertz
x_max = np.max(passband_trigger[:, 0])/units.megahertz
y_min = np.min(passband_trigger[:, 1])/units.megahertz
y_max = np.max(passband_trigger[:, 1])/units.megahertz

binWidth = 10
binLength = 10
x_edges = np.arange(x_min-5, x_max+15, binWidth)
y_edges = np.arange(y_min-5, y_max+15, binLength)

for count_energy, energy_bin in enumerate(energy_bins_low):
    x = passband_trigger[:, 0] / units.megahertz
    y = passband_trigger[:, 1] / units.megahertz
    z = total_aeff[:,count_energy]/units.km**2

    counts, xbins, ybins, image = plt.hist2d(x, y,
                                             bins=[x_edges, y_edges], weights=z,
                                             vmin=min(z), vmax=max(z), cmap=plt.cm.jet, cmin = 1e-9)
    plt.colorbar(label=r'Effective area [$km^2$]')
    plt.title(r'Energy {:.2e}, zenith {}$^\circ$ $-$ {}$^\circ$  '.format(energy_bin, zenith_start*10, zenith_stop*10))
    plt.xlim(x_min - 5, x_max + 5)
    plt.ylim(y_min - 5, y_max + 5)
    plt.xlabel('Lower cutoff frequency [MHz]')
    plt.ylabel('Upper cutoff frequency [MHz]')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('results/air_shower/fig_pb_thresh_signal_area_energy_{:.2e}_{}_{}.png'.format(
        energy_bin, zenith_start*10, zenith_stop*10))
    plt.close()
