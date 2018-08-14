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

filename = os.path.splitext(os.path.basename(args.inputfilename))[0]
dirname = os.path.dirname(args.inputfilename)
plot_folder = os.path.join(dirname, 'plots', filename)
if(not os.path.exists(plot_folder)):
    os.makedirs(plot_folder)
fin = h5py.File(args.inputfilename, 'r')

# plot vertex distribution
fig, ax = plt.subplots(1, 1)
xx = np.array(fin['xx'])
yy = np.array(fin['yy'])
rr = (xx ** 2 + yy ** 2) ** 0.5
zz = np.array(fin['zz'])
h = ax.hist2d(rr / units.m, zz / units.m, bins=[np.arange(0, 9001, 100), np.arange(-3501, 0, 100)],
              cmap=plt.get_cmap('Blues'))
cb = plt.colorbar(h[3], ax=ax)
cb.set_label("number of events")
ax.set_aspect('equal')
ax.set_xlabel("r [m]")
ax.set_ylabel("z [m]")
fig.tight_layout()
plt.title('vertex distribution')
plt.savefig(os.path.join(plot_folder, 'simInputVertex.pdf'), bbox_inches="tight")
plt.clf()

# plot incoming direction
zeniths = np.array(fin['zeniths'])
azimuths = np.array(fin['azimuths'])
plt.subplot(1, 2, 1)
#plt.hist(zeniths / units.deg, bins = np.arange(0, 181, 10))
plt.hist(np.cos(zeniths / units.radian), bins = np.arange(-1.01, 1.01, 0.2))
plt.xlabel('cos(zenith)')
plt.subplot(1, 2, 2)
plt.hist(azimuths / units.deg, bins = np.arange(0, 361, 45))
plt.xlabel('azimuth [deg]')
plt.suptitle('neutrino direction')
plt.savefig(os.path.join(plot_folder, 'simInputIncoming.pdf'), bbox_inches="tight")
plt.clf()

# plot direction distribution
fig, ax = plt.subplots(1, 1)
h = ax.hist2d(azimuths / units.deg, zeniths / units.deg, bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap=plt.get_cmap('Blues'))
cb = plt.colorbar(h[3], ax=ax)
cb.set_label("number of events")
ax.set_aspect('equal')
ax.set_xlabel("azimuth [deg]")
ax.set_ylabel("zenith [deg]")
fig.tight_layout()
plt.title('direction distribution')
plt.savefig(os.path.join(plot_folder, 'simInputDirection.pdf'), bbox_inches="tight")
plt.clf()

# plot inelasticity
inelasticity = np.array(fin['inelasticity'])
plt.hist(inelasticity, bins=np.logspace(np.log10(0.0001), np.log10(1.0), 50))
plt.xlabel('inelasticity')
plt.semilogx(True)
plt.title('inelasticity')
plt.figtext(1.0, 0.5, "N: " + str(len(inelasticity)) + "\nmean: " + str(np.mean(inelasticity)) + "\nmedian: " + str(np.median(inelasticity)) + "\nstd: " + str(np.std(inelasticity)))
plt.savefig(os.path.join(plot_folder, 'simInputInelasticity.pdf'), bbox_inches="tight")
