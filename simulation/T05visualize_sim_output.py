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

parser = argparse.ArgumentParser(description='Parse ARA event list.')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC hdf5 simulation output')
# parser.add_argument('outputfilename', type=str,
#                     help='name of output file storing the electric field traces at detector positions')
args = parser.parse_args()

fin = h5py.File(args.inputfilename, 'r')

weights = np.array(fin['weights'])

# plot vertex distribution
fig, ax = plt.subplots(1, 1)
xx = np.array(fin['xx'])
yy = np.array(fin['yy'])
rr = (xx ** 2 + yy ** 2) ** 0.5
zz = np.array(fin['zz'])
h = ax.hist2d(rr / units.m, zz / units.m, bins=[np.arange(0, 4000, 100), np.arange(-3000, 0, 100)],
              cmap=plt.get_cmap('Blues'), weights=weights)
cb = plt.colorbar(h[3], ax=ax)
cb.set_label("weighted number of events")
ax.set_aspect('equal')
ax.set_xlabel("r [m]")
ax.set_ylabel("z [m]")
fig.tight_layout()

# plot incoming direction
receive_vectors = np.array(fin['receive_vectors'])
# for all events, antennas and ray tracing solutions
zeniths, azimuths = hp.cartesian_to_spherical_vectorized(receive_vectors[:, :, :, 0].flatten(),
                                                         receive_vectors[:, :, :, 1].flatten(),
                                                         receive_vectors[:, :, :, 2].flatten())
weights_matrix = np.outer(weights, np.ones(np.prod(receive_vectors.shape[1:-1]))).flatten()
mask = ~np.isnan(azimuths)  # exclude antennas with not ray tracing solution (or with just one ray tracing solution)
fig, axs = php.get_histograms([zeniths[mask] / units.deg, azimuths[mask] / units.deg],
                              bins=[np.arange(0, 181, 10), np.arange(0, 361, 45)],
                              xlabels=['zenith [deg]', 'azimuth [deg]'],
                              weights=weights_matrix[mask], stats=False)
fig.suptitle('incoming signal direction')

# plot polarization
polarization = np.array(fin['polarization']).flatten()
# for all events, antennas and ray tracing solutions
mask = zeniths > 90 * units.deg  # select rays coming from below
fig, ax = php.get_histogram(polarization[mask] / units.deg,
                            bins=np.arange(-180, 181, 45),
                            xlabel='polarization [deg]',
                            weights=weights_matrix[mask], stats=False,
                             kwargs={'facecolor':'C1', 'alpha':1, 'edgecolor':"k"})
php.get_histogram(polarization[~mask] / units.deg,
                            bins=np.arange(-180, 181, 45),
                            xlabel='polarization [deg]',
                            weights=weights_matrix[~mask], stats=False,
                            ax=ax)

# fig.suptitle('incoming signal direction')

plt.show()
