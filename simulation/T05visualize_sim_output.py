from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from NuRadioMC.utilities import units
from NuRadioMC.utilities import medium
import h5py
import argparse
import json
import time
import os

parser = argparse.ArgumentParser(description='Plot NuRadioMC event list output.')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC hdf5 simulation output')
parser.add_argument('--Veff', type=str,
                    help='specify json file where effective volume is saved as a function of energy')
args = parser.parse_args()

filename = os.path.splitext(os.path.basename(args.inputfilename))[0]
dirname = os.path.dirname(args.inputfilename)
plot_folder = os.path.join(dirname, 'plots', filename)
if(not os.path.exists(plot_folder)):
    os.makedirs(plot_folder)

fin = h5py.File(args.inputfilename, 'r')

weights = np.array(fin['weights'])
triggered = np.array(fin['triggered'])
n_events = fin.attrs['n_events']

# calculate effective
density_ice = 0.9167 * units.g / units.cm ** 3
density_water = 997 * units.kg / units.m ** 3

n_triggered = np.sum(weights[triggered])
print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f}'.format(n_triggered, n_events, n_triggered / n_events))

dX = fin.attrs['xmax'] - fin.attrs['xmin']
dY = fin.attrs['ymax'] - fin.attrs['ymin']
dZ = fin.attrs['zmax'] - fin.attrs['zmin']
V = dX * dY * dZ
Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights[triggered]) / n_events

print("Veff = {:.6g} km^3 sr".format(Veff / units.km ** 3))

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
fig.savefig(os.path.join(plot_folder, 'vertex_distribution.png'))

# plot incoming direction
receive_vectors = np.array(fin['receive_vectors'])
# for all events, antennas and ray tracing solutions
zeniths, azimuths = hp.cartesian_to_spherical_vectorized(receive_vectors[:, :, :, 0].flatten(),
                                                         receive_vectors[:, :, :, 1].flatten(),
                                                         receive_vectors[:, :, :, 2].flatten())
for i in range(len(azimuths)):
    azimuths[i] = hp.get_normalized_angle(azimuths[i])
weights_matrix = np.outer(weights, np.ones(np.prod(receive_vectors.shape[1:-1]))).flatten()
mask = ~np.isnan(azimuths)  # exclude antennas with not ray tracing solution (or with just one ray tracing solution)
fig, axs = php.get_histograms([zeniths[mask] / units.deg, azimuths[mask] / units.deg],
                              bins=[np.arange(0, 181, 10), np.arange(0, 361, 45)],
                              xlabels=['zenith [deg]', 'azimuth [deg]'],
                              weights=weights_matrix[mask], stats=False)
fig.suptitle('incoming signal direction')
fig.savefig(os.path.join(plot_folder, 'incoming_signal.png'))

# plot polarization
polarization = np.array(fin['polarization']).flatten()
polarization = np.abs(polarization)
polarization[polarization > 90 * units.deg] = 180 * units.deg - polarization[polarization > 90 * units.deg]
bins = np.linspace(0, 90, 50)

# for all events, antennas and ray tracing solutions
mask = zeniths > 90 * units.deg  # select rays coming from below
fig, ax = php.get_histogram(polarization / units.deg,
                            bins=bins,
                            xlabel='polarization [deg]',
                            weights=weights_matrix, stats=False,
                            figsize=(6, 6))
maxy = ax.get_ylim()
php.get_histogram(polarization[mask] / units.deg,
                  bins=bins,
                  xlabel='polarization [deg]',
                  weights=weights_matrix[mask], stats=False,
                  ax=ax, kwargs={'facecolor': 'C0', 'alpha': 1, 'edgecolor': "k"})
# ax.set_xticks(bins)
ax.set_ylim(maxy)
fig.tight_layout()
fig.savefig(os.path.join(plot_folder, 'polarization.png'))

fig, ax = php.get_histogram(np.array(fin['zeniths']) / units.deg, weights=weights,
                            ylabel='weighted entries', xlabel='zenith angle [deg]',
                            bins=np.arange(0, 181, 5), figsize=(6, 6))
ax.set_xticks(np.arange(0, 181, 45))
fig.tight_layout()
fig.savefig(os.path.join(plot_folder, 'neutrino_direction.png'))

shower_axis = hp.spherical_to_cartesian(np.array(fin['zeniths']), np.array(fin['azimuths']))
launch_vectors = np.array(fin['launch_vectors'])
viewing_angles = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 0, 0])])

# calculate correct chereknov angle for ice density at vertex position
ice = medium.southpole_simple()
n_indexs = np.array([ice.get_index_of_refraction(x) for x in np.array([np.array(fin['xx']), np.array(fin['yy']), np.array(fin['zz'])]).T])
rho = np.arccos(1. / n_indexs)

mask = ~np.isnan(viewing_angles)
fig, ax = php.get_histogram((viewing_angles[mask] - rho[mask]) / units.deg, weights=weights[mask],
                            bins=np.arange(-20, 20, 1), xlabel='viewing - cherenkov angle [deg]', figsize=(6, 6))
fig.savefig(os.path.join(plot_folder, 'dCherenkov.png'))

# SNR

# solution type

# plot C0 parameter
# C0s = np.array(fin['ray_tracing_C0'])
# php.get_histogram(C0s.flatten())
# fig.suptitle('incoming signal direction')

plt.show()
