from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from NuRadioReco.utilities import units
import h5py
import argparse
import json
import time
import os

parser = argparse.ArgumentParser(description='Plot NuRadioMC event list input')
parser.add_argument('inputfilename', type=str, nargs='+',
                    help='path to NuRadioMC hdf5 simulation input')
args = parser.parse_args()

# data for vertex plot
xx = []
yy = []
zz = []
zeniths = []
azimuths = []
inelasticity = []
flavors = []
interaction_type = []

filename = os.path.splitext(os.path.basename(args.inputfilename[0]))[0]
dirname = os.path.dirname(args.inputfilename[0])
plot_folder = os.path.join(dirname, 'plots', 'input', filename)
if(not os.path.exists(plot_folder)):
    os.makedirs(plot_folder)

for input_filename in args.inputfilename:
    print(f"parsing file {input_filename}")
    fin = h5py.File(input_filename, 'r')
    xx.extend(np.array(fin['xx']))
    yy.extend(np.array(fin['yy']))
    zz.extend(np.array(fin['zz']))
    zeniths.extend(np.array(fin['zeniths']))
    azimuths.extend(np.array(fin['azimuths']))
    inelasticity.extend(np.array(fin['inelasticity']))
    flavors.extend(np.array(fin['flavors']))
    interaction_type.extend(np.array(fin['interaction_type']))

print(f"starting plotting")
###########################
# plot flavor ratios
###########################
flavor_labels = ['e cc', r'$\bar{e}$ cc', 'e nc', r'$\bar{e}$ nc',
           '$\mu$ cc', r'$\bar{\mu}$ cc', '$\mu$ nc', r'$\bar{\mu}$ nc',
           r'$\tau$ cc', r'$\bar{\tau}$ cc', r'$\tau$ nc', r'$\bar{\tau}$ nc']
flavors = np.array(flavors)
interaction_type = np.array(interaction_type)
flavor_sum = np.zeros(len(flavor_labels))
flavor_sum[0] = np.sum((flavors == 12) & (interaction_type == b'cc'))
flavor_sum[1] = np.sum((flavors == -12) & (interaction_type == b'cc'))
flavor_sum[2] = np.sum((flavors == 12) & (interaction_type == b'nc'))
flavor_sum[3] = np.sum((flavors == -12) & (interaction_type == b'nc'))

flavor_sum[4] = np.sum((flavors == 14) & (interaction_type == b'cc'))
flavor_sum[5] = np.sum((flavors == -14) & (interaction_type == b'cc'))
flavor_sum[6] = np.sum((flavors == 14) & (interaction_type == b'nc'))
flavor_sum[7] = np.sum((flavors == -14) & (interaction_type == b'nc'))

flavor_sum[8] = np.sum((flavors == 16) & (interaction_type == b'cc'))
flavor_sum[9] = np.sum((flavors == -16) & (interaction_type == b'cc'))
flavor_sum[10] = np.sum((flavors == 16) & (interaction_type == b'nc'))
flavor_sum[11] = np.sum((flavors == -16) & (interaction_type == b'nc'))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.bar(range(len(flavor_labels)), flavor_sum)
ax.set_xticks(range(len(flavor_labels)))
ax.set_xticklabels(flavor_labels, fontsize='large', rotation=45)
ax.set_ylabel('weighted number of triggers', fontsize='large')
fig.tight_layout()
fig.savefig(os.path.join(plot_folder, 'flavor.png'))

# plot vertex distribution
fig, ax = plt.subplots(1, 1)
xx = np.array(xx)
yy = np.array(yy)
rr = (xx ** 2 + yy ** 2) ** 0.5
zz = np.array(zz)
h = ax.hist2d(rr / units.m, zz / units.m, bins=[np.arange(0, 4000, 100), np.arange(-3000, 0, 100)],
              cmap=plt.get_cmap('Blues'))
cb = plt.colorbar(h[3], ax=ax)
cb.set_label("number of events")
ax.set_aspect('equal')
ax.set_xlabel("r [m]")
ax.set_ylabel("z [m]")
fig.tight_layout()
plt.title('vertex distribution')
plt.savefig(os.path.join(plot_folder, "simInputVertex.png"))

# plot incoming direction
zeniths = np.array(zeniths)
azimuths = np.array(azimuths)
fig, axs = php.get_histograms([zeniths / units.deg, azimuths / units.deg],
                              bins=[np.arange(0, 181, 2), np.arange(0, 361, 5)],
                              xlabels=['zenith [deg]', 'azimuth [deg]'],
                              stats=False)
fig.suptitle('neutrino direction')
fig.subplots_adjust(top=0.9)

plt.savefig(os.path.join(plot_folder, "simInputIncoming.png"))

fig, ax = php.get_histogram(np.cos(zeniths), bins=np.arange(-1, 1.01, 0.1),
                            xlabel="cos zenith")
plt.savefig(os.path.join(plot_folder, "simInputIncoming_coszenith.png"))
# plot inelasticity
inelasticity = np.array(inelasticity)
fig, axs = php.get_histogram(inelasticity,
                             bins=np.logspace(np.log10(0.0001), np.log10(1.0), 50),
                             xlabel='inelasticity', figsize=(6, 6),
                             stats=True)
axs.semilogx(True)
plt.title('inelasticity')
plt.savefig(os.path.join(plot_folder, "simInputInelasticity.png"))
