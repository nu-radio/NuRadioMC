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
parser.add_argument('inputfilename', type=str, nargs = '+', help='path to NuRadioMC hdf5 simulation input')
# parser.add_argument('outputfilename', type=str,
#                     help='name of output file storing the electric field traces at detector positions')
args = parser.parse_args()

filename = os.path.splitext(os.path.basename(args.inputfilename[0]))[0]
dirname = os.path.dirname(args.inputfilename[0])
plot_folder = os.path.join(dirname, 'plots', filename)
if(not os.path.exists(plot_folder)):
    os.makedirs(plot_folder)

fin = h5py.File(args.inputfilename[0], 'r')
xx = np.array(fin['xx'])
yy = np.array(fin['yy'])
zz = np.array(fin['zz'])
zeniths = np.array(fin['zeniths'])
azimuths = np.array(fin['azimuths'])
inelasticity = np.array(fin['inelasticity'])

for i in range(len((args.inputfilename)) - 1):
    fin = h5py.File(args.inputfilename[i + 1], 'r')
    xx = np.append(xx, np.array(fin['xx']))
    yy = np.append(yy, np.array(fin['yy']))
    zz = np.append(zz, np.array(fin['zz']))
    zeniths = np.append(zeniths, np.array(fin['zeniths']))
    azimuths = np.append(azimuths, np.array(fin['azimuths']))
    inelasticity = np.append(inelasticity, np.array(fin['inelasticity']))

# plot vertex distribution
fig, ax = plt.subplots(1, 1)
rr = (xx ** 2 + yy ** 2) ** 0.5
plt.hist2d(rr / units.m, zz / units.m, bins=[np.arange(0, 9001, 100), np.arange(-3501, 0, 100)], cmap=plt.get_cmap('Blues'))
cb = plt.colorbar()
cb.set_label("number of events")
ax.set_aspect('equal')
plt.xlabel("r [m]")
plt.ylabel("z [m]")
plt.grid(True)
fig.tight_layout()
plt.title('vertex distribution')
plt.savefig(os.path.join(plot_folder, 'simInputVertex.pdf'), bbox_inches="tight")
plt.clf()

# plot incoming direction
plt.subplot(1, 2, 1)
#plt.hist(zeniths / units.deg, bins = np.arange(0, 181, 10))
plt.hist(np.cos(zeniths / units.radian), bins = np.arange(-1.0, 1.001, 0.2))
plt.xlabel('cos(zenith)')
plt.subplot(1, 2, 2)
plt.hist(azimuths / units.deg, bins = np.arange(0, 361, 45))
plt.xlabel('azimuth [deg]')
plt.suptitle('neutrino direction')
plt.savefig(os.path.join(plot_folder, 'simInputIncoming.pdf'), bbox_inches="tight")
plt.clf()

# plot direction distribution
plt.hist2d(azimuths / units.deg, np.cos(zeniths / units.radian), bins=[np.arange(0, 361, 5), np.arange(-1.0, 1.001, 0.05)], cmap=plt.get_cmap('Blues'))
cb = plt.colorbar()
cb.set_label("number of events")
ax.set_aspect('equal')
plt.xlabel("azimuth [deg]")
plt.ylabel("cos(zenith)")
plt.grid(True)
plt.title('direction distribution')
plt.savefig(os.path.join(plot_folder, 'simInputDirection.pdf'), bbox_inches="tight")
plt.clf()

# plot inelasticity
plt.hist(inelasticity, bins=np.logspace(np.log10(0.0001), np.log10(1.0), 50))
plt.xlabel('inelasticity')
plt.semilogx(True)
plt.title('inelasticity')
plt.figtext(1.0, 0.5, "N: " + str(len(inelasticity)) + "\nmean: " + str(np.mean(inelasticity)) + "\nmedian: " + str(np.median(inelasticity)) + "\nstd: " + str(np.std(inelasticity)))
plt.savefig(os.path.join(plot_folder, 'simInputInelasticity.pdf'), bbox_inches="tight")
