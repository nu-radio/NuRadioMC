from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from NuRadioMC.utilities import units
import h5py
import argparse
import json
import time
import os

parser = argparse.ArgumentParser(description='Plot NuRadioMC event list input')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC hdf5 simulation input')
# parser.add_argument('outputfilename', type=str,
#                     help='name of output file storing the electric field traces at detector positions')
args = parser.parse_args()

fin = h5py.File(args.inputfilename, 'r')
# plot vertex distribution
fig, ax = plt.subplots(1, 1)
xx = np.array(fin['xx'])
yy = np.array(fin['yy'])
rr = (xx ** 2 + yy ** 2) ** 0.5
zz = np.array(fin['zz'])
h = ax.hist2d(rr / units.m, zz / units.m, bins=[np.arange(0, 4000, 100), np.arange(-3000, 0, 100)],
              cmap=plt.get_cmap('Blues'))
cb = plt.colorbar(h[3], ax=ax)
cb.set_label("number of events")
ax.set_aspect('equal')
ax.set_xlabel("r [m]")
ax.set_ylabel("z [m]")
fig.tight_layout()
plt.title('vertex distribution')
plt.savefig("output/simInputVertex.pdf")

# plot incoming direction
zeniths = np.array(fin['zeniths'])
azimuths = np.array(fin['azimuths'])
fig, axs = php.get_histograms([zeniths / units.deg, azimuths / units.deg],
                              bins=[np.arange(0, 181, 10), np.arange(0, 361, 45)],
                              xlabels=['zenith [deg]', 'azimuth [deg]'],
                              stats=False)
fig.suptitle('neutrino direction')
plt.title('incoming direction')
plt.savefig("output/simInputIncoming.pdf")

# plot inelasticity
inelasticity = np.array(fin['inelasticity'])
fig, axs = php.get_histogram(inelasticity,
                             bins=np.logspace(np.log10(0.0001), np.log10(1.0), 50),
                             xlabel='inelasticity', figsize=(6, 6),
                             stats=True)
axs.semilogx(True)
plt.title('inelasticity')
plt.savefig("output/simInputInelasticity.pdf")
