import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, io_utilities

number_coincidences_list = []
coinc_windows = []
passband_trigger = np.zeros((201,2))
trigger_threshold = np.zeros((201))
triggered_trigger = np.zeros((201, 3, 9)) # number of passbands, energy bins, zenith angle bins
trigger_weight = np.zeros((201, 3, 9))
masked_events = np.zeros((201, 3, 9))
trigger_efficiency = np.zeros((201, 3, 9))
trigger_effective_area = np.zeros((201, 3, 9))
trigger_effective_area_err = np.zeros((201, 3, 9))

for pos, file in enumerate(os.listdir('results/air_shower/')):
    filename = os.path.join('results/air_shower/', file)
    print('file used', filename)
    if os.path.getsize(filename) > 0:
        data = []
        data = io_utilities.read_pickle(filename, encoding='latin1')
        #print(data)

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

#print('passband', passbands)
#print('threshold', trigger_threshold)
print('trigger_effective_area', trigger_effective_area)
print('trigger_effective_area', trigger_effective_area.shape)

#choose zenith bins etc here
zenith_start = int(0 /10)
zenith_stop = int(90 /10)

trigger_efficiency = trigger_efficiency[:,:,zenith_start:zenith_stop]
triggered_trigger = triggered_trigger[:,:,zenith_start:zenith_stop]
masked_events = masked_events[:,:,zenith_start:zenith_stop]
trigger_effective_area = trigger_effective_area[:,:,zenith_start:zenith_stop]
trigger_effective_area_err = trigger_effective_area_err[:,:,zenith_start:zenith_stop]
zenith_bin = np.arange(zenith_start,zenith_stop,10)

weight_area = (-(np.cos((zenith_bin+10)*units.deg)**2)/2) - (-(np.cos(zenith_bin*units.deg)**2)/2) #integral of sin*cos
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
    #print('x, y', x,y)
    print('z', z)
    print('z', z.shape)

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
