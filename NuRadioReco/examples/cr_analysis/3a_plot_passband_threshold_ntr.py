import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, io_utilities

trigger_rate = []
trigger_thresholds = []
final_thresholds = []
passband_low = []
passband_high = []
passbands = []

for file in os.listdir('results/ntr/'):
    filename = os.path.join('results/ntr/', file)
    print('file used', filename)
    data = []
    data = io_utilities.read_pickle(filename, encoding='latin1')

    trigger_thresholds = data['thresholds']
    print('trigger thresholds', trigger_thresholds)

    passband_trigger = data['passband_trigger']

    print('passband', passband_trigger/units.megahertz)
    passband_low.append(passband_trigger[0])
    print(passband_low)
    passband_high.append(passband_trigger[1])
    print(passband_high)
    passbands.append(passband_trigger)

    trigger_rate = data['trigger_rate']
    print('trigger rate', trigger_rate)
    ######for dict
    zeros = np.where(trigger_rate == 0)[0]
    print(zeros)
    first_zero = zeros[0]
    print(first_zero)
    final_threshold = trigger_thresholds[first_zero]
    print('final threshold', final_threshold/units.mV)
    final_thresholds.append(final_threshold)

    ###for check ntr
    #final_threshold = trigger_thresholds[-1]
    #print('final threshold', final_threshold/units.mV)
    #final_thresholds.append(final_threshold)


passband_low = np.array(passband_low)
passband_high = np.array(passband_high)
final_thresholds = np.array(final_thresholds)
passbands = np.array(passbands)

print(passbands/units.megahertz)
print(final_thresholds/units.mV)

x_min = min(passband_low)/units.megahertz
x_max = max(passband_low)/units.megahertz
y_min = min(passband_high)/units.megahertz
y_max = max(passband_high)/units.megahertz

x = passband_low/units.megahertz
y = passband_high/units.megahertz
z = final_thresholds/units.mV

binWidth = 10
binLength = 10
x_edges = np.arange(x_min-5, x_max+15, binWidth)
y_edges = np.arange(y_min-5, y_max+15, binLength)


counts,xbins,ybins,image = plt.hist2d(x, y, bins=[x_edges, y_edges], weights=z, vmin=min(z), vmax= max(z), cmap=plt.cm.jet, cmin = 1e-9)
plt.colorbar(label='Threshold for NTR < 0.5 Hz [mV]')

CS = plt.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
    linewidths=2, colors= 'black', vmin=min(z), vmax= max(z), levels = [20, 25, 30, 35, 40, 45, 50, 55])

plt.clabel(CS, CS.levels, inline=True, fmt='%1.0f', fontsize=10)
plt.xlim(x_min - 5, x_max + 5)
plt.ylim(y_min - 5, y_max + 5)
plt.xlabel('Lower cutoff frequency [MHz]')
plt.ylabel('Upper cutoff frequency [MHz]')
#plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('results/ntr/fig_passband_threshold_ntr_2e6.png')

