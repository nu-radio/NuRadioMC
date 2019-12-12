from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
import h5py
import argparse
import json
import time
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
plt.switch_backend('agg')


parser = argparse.ArgumentParser(description='Plot NuRadioMC event list output.')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC hdf5 simulation output')
args = parser.parse_args()

fin = h5py.File(args.inputfilename, 'r')

weights = np.array(fin['weights'])
n_events = fin.attrs['n_events']
Vrms = fin.attrs['Vrms']

depths = np.array(fin['station_101'].attrs['antenna_positions'])[:, 2]
print("simulated depths ", depths)

eff1 = np.zeros_like(depths)
eff2 = np.zeros_like(depths)
eff3 = np.zeros_like(depths)

for iD, depth in enumerate(depths):
    As = np.array(fin['station_101']['max_amp_ray_solution'][:,iD])
    As = np.nan_to_num(As)  # this sets all cases with empty ray tracing solution to an amplitude of zero
    mask = np.isnan(As[:, 0]) | np.isnan(As[:, 1])
    As = As[~mask]
    Ts = np.array(fin['station_101']['travel_times'][:, iD])[~mask]
    Ts = np.nan_to_num(Ts)
    dTs = np.abs(Ts[:, 1] - Ts[:, 0])
    C3 = (As[:, 0] >= 3 * Vrms) | (As[:, 1] >= 3 * Vrms)

    C1 = ((As[:, 0] >= 3 * Vrms) | (As[:, 1] >= 3 * Vrms)) & ((As[:, 0] >= 2 * Vrms) & (As[:, 1] >= 2 * Vrms)) & (dTs < 430 * units.ns)
    C2 = ((As[:, 0] >= 3 * Vrms) & (As[:, 1] >= 3 * Vrms)) & (dTs < 430 * units.ns)
    C4 = ((As[:, 0] >= 4 * Vrms) & (As[:, 1] >= 4 * Vrms)) & (dTs < 430 * units.ns)

    eff1[iD] = np.sum(weights[~mask][C1])/ np.sum(weights[~mask][C3])
    eff2[iD] = np.sum(weights[~mask][C2])/ np.sum(weights[~mask][C3])
    eff3[iD] = np.sum(weights[~mask][C4])/ np.sum(weights[~mask][C3])

sort_mask = np.argsort(depths)
np.savetxt("DnRefficiency_{:.0g}eV.txt".format(np.array(fin['energies']).mean()), [depths[sort_mask], eff1[sort_mask]])

fig, (ax) = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(depths[sort_mask], eff1[sort_mask], 'o-', label='one pulse > 3 sigma, one pulse > 2 sigma')
ax.plot(depths[sort_mask], eff2[sort_mask], "d--", label="both pulses > 3 sigma")
ax.plot(depths[sort_mask], eff3[sort_mask], "^:", label="both pulses > 4 sigma")
ax.set_xlim(0, -100)
ax.set_ylim(0, 1)
ax.set_xlabel("depth [m]")
ax.set_ylabel("efficiency")
ax.set_title("neutrino energy {:.2g}eV".format(np.array(fin['energies']).mean()))
ax.legend(fontsize='x-small', numpoints=1)
fig.tight_layout()
fig.savefig("{}.pdf".format(os.path.basename(args.inputfilename)))
plt.show()
