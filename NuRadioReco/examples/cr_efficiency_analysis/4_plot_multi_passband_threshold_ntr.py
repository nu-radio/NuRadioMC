import numpy as np
import json
import os
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units

''' This script is only necessary if you want to compare the thresholds of different passbands. 
First you have to run 3_... to have the final results in one dict. Make sure that the dicts for all the passbands 
you want to compare are in the directory ntr.'''

trigger_rate = []
trigger_thresholds = []
final_thresholds = []
passband_low = []
passband_high = []
passbands = []

for file in os.listdir('results/ntr/'):
    filename = os.path.join('results/ntr/', file)

    with open(filename, 'r') as fp:
        data = json.load(fp)

    trigger_thresholds = data['threshold']
    passband_trigger = data['passband_trigger']
    passband_low.append(passband_trigger[0])
    passband_high.append(passband_trigger[1])
    passbands.append(passband_trigger)

    trigger_rate = np.array(data['trigger_rate'])

    zeros = np.where(trigger_rate == 0)[0]
    first_zero = zeros[0]
    final_threshold = trigger_thresholds[first_zero]
    final_thresholds.append(final_threshold)

passband_low = np.array(passband_low)
passband_high = np.array(passband_high)
final_thresholds = np.array(final_thresholds)
passbands = np.array(passbands)

x_min = min(passband_low)/units.megahertz
x_max = max(passband_low)/units.megahertz
y_min = min(passband_high)/units.megahertz
y_max = max(passband_high)/units.megahertz

x = passband_low/units.megahertz
y = passband_high/units.megahertz
z = final_thresholds/units.mV

binWidth = 10  # steps between lower limit of the passbands
binLength = 10  # steps between higher limit of the passbands
x_edges = np.arange(x_min-5, x_max+15, binWidth)  # shift marker to the center of a bin
y_edges = np.arange(y_min-5, y_max+15, binLength)

counts,xbins,ybins,image = plt.hist2d(x, y, bins=[x_edges, y_edges], weights=z, vmin=min(z), vmax= max(z), cmap=plt.cm.jet, cmin = 1e-9)
plt.colorbar(label='Threshold for NTR < 0.5 Hz [mV]')  # the 0.5 Hz depends on the resolution/number of iterations

# use the following lines to plot iso voltage lines
#CS = plt.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
 #   linewidths=2, colors= 'black', vmin=min(z), vmax= max(z), levels = [20, 25, 30, 35, 40, 45, 50, 55])

#plt.clabel(CS, CS.levels, inline=True, fmt='%1.0f', fontsize=10)
plt.xlim(x_min - 5, x_max + 5)
plt.ylim(y_min - 5, y_max + 5)
plt.xlabel('Lower cutoff frequency [MHz]')
plt.ylabel('Upper cutoff frequency [MHz]')
#plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('results/fig_passband_threshold_ntr.png')

