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
# parser.add_argument('outputfilename', type=str,
#                     help='name of output file storing the electric field traces at detector positions')
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

# calculate effective volume
density_ice = 0.9167 * units.g / units.cm ** 3
density_water = 997 * units.kg / units.m ** 3
n_triggered = np.sum(weights[triggered])
print('total number of triggered events = ' + str(len(fin['weights'])))
print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f}'.format(n_triggered, n_events, n_triggered / n_events))
#if generate_eventlist_cuboid used
#dX = fin.attrs['xmax'] - fin.attrs['xmin']
#dY = fin.attrs['ymax'] - fin.attrs['ymin']
#dZ = fin.attrs['zmax'] - fin.attrs['zmin']
#V = dX * dY * dZ
#if generate_eventlist_cylinder used
RMax = fin.attrs['ymax']
RMin = fin.attrs['ymin']
dZ = fin.attrs['zmax'] - fin.attrs['zmin']
V = np.pi * (RMax ** 2 - RMin ** 2) * dZ
Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights[triggered]) / n_events
print("Veff = {:.2g} km^3 sr".format(Veff / units.km ** 3))

# plot vertex distribution rz
plt.subplot(2, 1, 1)
xx = np.array(fin['xx'])
yy = np.array(fin['yy'])
rr = (xx ** 2 + yy ** 2) ** 0.5
zz = np.array(fin['zz'])
#h = ax.hist2d(rr / units.m, zz / units.m, bins=[np.arange(0, 4000, 100), np.arange(-3000, 0, 100)],
#              cmap=plt.get_cmap('Blues'), weights=weights)
plt.hist2d(rr / units.m, zz / units.m, bins=[np.arange(0, 9001, 100), np.arange(-3501, 0, 100)], cmap=plt.get_cmap('Blues'), weights=weights/n_events)
cb = plt.colorbar()
cb.set_label("weighted number of events")
plt.xlabel("r [m]")
plt.ylabel("z [m]")
plt.subplot(2, 1, 2)
plt.hist2d(rr / units.m, zz / units.m, bins=[np.arange(0, 9001, 100), np.arange(-3501, 0, 100)], cmap=plt.get_cmap('Blues'), weights = np.ones(len(rr))/n_events)
cb = plt.colorbar()
cb.set_label("number of events")
plt.xlabel("r [m]")
plt.ylabel("z [m]")
plt.suptitle("vertex distribution")
plt.savefig(os.path.join(plot_folder, 'vertex_distribution.pdf'), bbox_inches="tight")
plt.clf()

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
plt.subplot(1, 2, 1)
plt.hist(zeniths[mask] / units.deg, bins = np.arange(0, 181, 10), weights=weights_matrix[mask])
plt.xlabel('zenith [deg]')
plt.subplot(1, 2, 2)
plt.hist(azimuths[mask] / units.deg, bins = np.arange(0, 361, 45), weights=weights_matrix[mask])
plt.xlabel('azimuth [deg]')
plt.suptitle('weighted incoming signal direction')
plt.savefig(os.path.join(plot_folder, 'incoming_signal.pdf'), bbox_inches="tight")
plt.clf()

# plot direction distribution
theta = np.array(fin['zeniths'])
phi = np.array(fin['azimuths'])
plt.subplot(2, 1, 1)
plt.hist2d(phi / units.deg, theta / units.deg, bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap=plt.get_cmap('Blues'), weights=weights/n_events)
cb = plt.colorbar()
cb.set_label("weighted number of events")
plt.xlabel("azimuth [deg]")
plt.ylabel("zenith [deg]")
plt.subplot(2, 1, 2)
plt.hist2d(phi / units.deg, theta / units.deg, bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap=plt.get_cmap('Blues'), weights = np.ones(len(phi))/n_events)
cb = plt.colorbar()
cb.set_label("number of events")
plt.xlabel("azimuth [deg]")
plt.ylabel("zenith [deg]")
plt.suptitle("direction distribution")
plt.savefig(os.path.join(plot_folder, 'direction_distribution.pdf'), bbox_inches="tight")
plt.clf()

# plot polarization
polarization = np.array(fin['polarization']).flatten()
polarization = np.abs(polarization)
polarization[polarization > 90 * units.deg] = 180 * units.deg - polarization[polarization > 90 * units.deg]
bins = np.linspace(0, 90, 50)
# for all events, antennas and ray tracing solutions
mask = zeniths > 90 * units.deg  # select rays coming from below
plt.hist(polarization / units.deg, bins=bins, weights=weights_matrix)
plt.xlabel('weighted polarization [deg]')
plt.hist(polarization[mask] / units.deg, bins=bins, weights=weights_matrix[mask], color = 'r')
plt.figtext(1.0, 0.5, "red: rays coming from below; N: " + str(len(polarization[mask])) + "\nblue: all; N: " + str(len(polarization)))
plt.savefig(os.path.join(plot_folder, 'polarization.pdf'), bbox_inches="tight")
plt.clf()

#plot neutrino direction
zeniths = np.array(fin['zeniths']) / units.deg
plt.subplot(2, 1, 1)
plt.hist(zeniths, weights=weights, bins=np.arange(0, 181, 5))
plt.xlabel('zenith angle [deg]')
plt.ylabel('weighted entries')
avgZen = np.average(zeniths, weights=weights)
varZen = np.average((zeniths-avgZen)**2, weights=weights)
plt.figtext(1.0, 0.8, "N: " + str(len(zeniths)) + "\nmean: " + str(np.average(zeniths, weights=weights)) + "\nstd: " + str(varZen**0.5))
plt.subplot(2, 1, 2)
plt.hist(zeniths, bins=np.arange(0, 181, 5))
plt.xlabel('zenith angle [deg]')
plt.ylabel('unweighted entries')
plt.figtext(1.0, 0.2, "N: " + str(len(zeniths)) + "\nmean: " + str(np.average(zeniths)) + "\nstd: " + str(np.std(zeniths)))
plt.suptitle("neutrino direction")
plt.savefig(os.path.join(plot_folder, 'neutrino_direction.pdf'), bbox_inches="tight")
plt.clf()

#plot difference between cherenkov angle and viewing angle
# i.e., opposite to the direction of propagation. We need the propagation direction here, so we multiply the shower axis with '-1'
shower_axis = -1.0 * hp.spherical_to_cartesian(np.array(fin['zeniths']), np.array(fin['azimuths']))
launch_vectors = np.array(fin['launch_vectors'])
viewing_angles = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 0, 0])])
# calculate correct chereknov angle for ice density at vertex position
ice = medium.southpole_simple()
n_indexs = np.array([ice.get_index_of_refraction(x) for x in np.array([np.array(fin['xx']), np.array(fin['yy']), np.array(fin['zz'])]).T])
rho = np.arccos(1. / n_indexs)
mask = ~np.isnan(viewing_angles)
dCherenkov = (viewing_angles - rho) / units.deg
plt.subplot(2, 1, 1)
plt.hist(dCherenkov[mask], weights=weights[mask], bins=np.arange(-20, 20, 1))
plt.xlabel('weighted viewing - cherenkov angle [deg]')
avgChe = np.average(dCherenkov[mask], weights=weights[mask])
varChe = np.average((dCherenkov[mask]-avgChe)**2, weights=weights[mask])
plt.figtext(1.0, 0.8, "N: " + str(len(dCherenkov[mask])) + "\nmean: " + str(np.average(dCherenkov[mask], weights=weights[mask])) + "\nstd: " + str(varChe**0.5))
plt.subplot(2, 1, 2)
plt.hist(dCherenkov[mask], bins=np.arange(-20, 20, 1))
plt.xlabel('unweighted viewing - cherenkov angle [deg]')
plt.figtext(1.0, 0.2, "N: " + str(len(dCherenkov[mask])) + "\nmean: " + str(np.average(dCherenkov[mask])) + "\nstd: " + str(np.std(dCherenkov[mask])))
plt.savefig(os.path.join(plot_folder, 'dCherenkov.pdf'), bbox_inches="tight")

# plot C0 parameter
# C0s = np.array(fin['ray_tracing_C0'])
# php.get_histogram(C0s.flatten())
# fig.suptitle('incoming signal direction')
#plt.show()
