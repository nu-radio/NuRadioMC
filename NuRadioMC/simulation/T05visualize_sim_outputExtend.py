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

parser = argparse.ArgumentParser(description='Plot NuRadioMC event list output.')
parser.add_argument('inputfilename', type=str, nargs = '+', help='path to NuRadioMC hdf5 simulation output')
# parser.add_argument('outputfilename', type=str,
#                     help='name of output file storing the electric field traces at detector positions')
args = parser.parse_args()

filename = os.path.splitext(os.path.basename(args.inputfilename[0]))[0]
dirname = os.path.dirname(args.inputfilename[0])
plot_folder = os.path.join(dirname, 'plots', filename)
if(not os.path.exists(plot_folder)):
    os.makedirs(plot_folder)

fin = h5py.File(args.inputfilename[0], 'r')
#weights = np.array(fin['weights'])[np.array(fin['weights']) > 1e-10]
weights = np.array(fin['weights'])
triggered = np.array(fin['triggered'])
xx = np.array(fin['xx'])
yy = np.array(fin['yy'])
zz = np.array(fin['zz'])
receive_vectors = np.array(fin['receive_vectors'])
theta = np.array(fin['zeniths'])
phi = np.array(fin['azimuths'])
polarization = np.array(fin['polarization']).flatten()
launch_vectors = np.array(fin['launch_vectors'])
n_events = fin.attrs['n_events']

for i in range(len(args.inputfilename) - 1):
    fin = h5py.File(args.inputfilename[i + 1], 'r')
    weights = np.append(weights, np.array(fin['weights']))
    triggered = np.append(triggered, np.array(fin['triggered']))
    xx = np.append(xx, np.array(fin['xx']))
    yy = np.append(yy, np.array(fin['yy']))
    zz = np.append(zz, np.array(fin['zz']))
    receive_vectors = np.append(receive_vectors, np.array(fin['receive_vectors']), axis=0)
    theta = np.append(theta, np.array(fin['zeniths']))
    phi = np.append(phi, np.array(fin['azimuths']))
    polarization = np.append(polarization, np.array(fin['polarization']).flatten())
    launch_vectors = np.append(launch_vectors, np.array(fin['launch_vectors']), axis=0)
    n_events += fin.attrs['n_events']

# calculate effective volume
density_ice = 0.9167 * units.g / units.cm ** 3
density_water = 997 * units.kg / units.m ** 3
n_triggered = np.sum(weights[triggered])
print('total number of triggered events = ' + str(len(weights)))
print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f}'.format(n_triggered, n_events, n_triggered / n_events))
V = None
if('xmax' in fin.attrs):
    dX = fin.attrs['xmax'] - fin.attrs['xmin']
    dY = fin.attrs['ymax'] - fin.attrs['ymin']
    dZ = fin.attrs['zmax'] - fin.attrs['zmin']
    V = dX * dY * dZ
elif('rmin' in fin.attrs):
    rmin = fin.attrs['rmin']
    rmax = fin.attrs['rmax']
    dZ = fin.attrs['zmax'] - fin.attrs['zmin']
    V = np.pi * (rmax**2 - rmin**2) * dZ
Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights[triggered]) / n_events
print("Veff = {:.6g} km^3 sr".format(Veff / units.km ** 3))

# plot vertex distribution rz
plt.subplot(2, 1, 1)
rr = (xx ** 2 + yy ** 2) ** 0.5
#h = ax.hist2d(rr / units.m, zz / units.m, bins=[np.arange(0, 4000, 100), np.arange(-3000, 0, 100)],
#              cmap=plt.get_cmap('Blues'), weights=weights)
plt.hist2d(rr / units.m, zz / units.m, bins=[np.arange(0, 9001, 100), np.arange(-3501, 0, 100)], cmap=plt.get_cmap('Blues'), weights=weights/n_events)
cb = plt.colorbar()
cb.set_label("weighted number of events")
plt.xlabel("r [m]")
plt.ylabel("z [m]")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.hist2d(rr / units.m, zz / units.m, bins=[np.arange(0, 9001, 100), np.arange(-3501, 0, 100)], cmap=plt.get_cmap('Blues'), weights = np.ones(len(rr))/n_events)
cb = plt.colorbar()
cb.set_label("number of events")
plt.xlabel("r [m]")
plt.ylabel("z [m]")
plt.grid(True)
plt.suptitle("vertex distribution")
plt.savefig(os.path.join(plot_folder, 'vertex_distribution.pdf'), bbox_inches="tight")
plt.clf()

# plot incoming direction
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
plt.subplot(2, 1, 1)
plt.hist2d(phi / units.deg, theta / units.deg, bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap=plt.get_cmap('Blues'), weights=weights/n_events)
#plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.set_label("weighted number of events")
plt.xlabel("azimuth [deg]")
plt.ylabel("zenith [deg]")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.hist2d(phi / units.deg, theta / units.deg, bins=[np.arange(0, 361, 5), np.arange(0, 181, 5)], cmap=plt.get_cmap('Blues'), weights = np.ones(len(phi))/n_events)
#plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.set_label("number of events")
plt.xlabel("azimuth [deg]")
plt.ylabel("zenith [deg]")
plt.grid(True)
plt.suptitle("direction distribution")
plt.savefig(os.path.join(plot_folder, 'direction_distribution.pdf'), bbox_inches="tight")
plt.clf()

# plot polarization
mask_pol = ~np.isnan(polarization)
polarization = np.abs(polarization)
polarization[polarization > 90 * units.deg] = 180 * units.deg - polarization[polarization > 90 * units.deg]
bins = np.linspace(0, 90, 45)
# for all events, antennas and ray tracing solutions
mask = zeniths > 90 * units.deg  # select rays coming from below
plt.hist(polarization[mask_pol] / units.deg, bins=bins, weights=weights_matrix[mask_pol], color = 'b')
plt.xlabel('weighted polarization [deg]')
plt.hist(polarization[mask] / units.deg, bins=bins, weights=weights_matrix[mask], color = 'r')
plt.figtext(1.0, 0.5, "red: rays coming from below; N: " + str(len(polarization[mask])) + "\nblue: all; N: " + str(len(polarization)))
plt.savefig(os.path.join(plot_folder, 'polarization.pdf'), bbox_inches="tight")
plt.clf()

#plot neutrino direction
zeniths = theta / units.deg
plt.subplot(2, 1, 1)
plt.hist(zeniths, weights=weights, bins=np.arange(0, 181, 5))
plt.xlabel('zenith angle [deg]')
plt.ylabel('weighted entries')
avgZen = np.average(zeniths, weights=weights)
varZen = np.average((zeniths-avgZen)**2, weights=weights)
plt.figtext(1.0, 0.6, "N: " + str(len(zeniths)) + "\nmean: " + str(np.average(zeniths, weights=weights)) + "\nstd: " + str(varZen**0.5))
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
shower_axis = -1.0 * hp.spherical_to_cartesian(theta, phi)
viewing_angles_d = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 0, 0])])
viewing_angles_r = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 0, 1])])
# calculate correct chereknov angle for ice density at vertex position
ice = medium.southpole_simple()
n_indexs = np.array([ice.get_index_of_refraction(x) for x in np.array([xx, yy, zz]).T])
rho = np.arccos(1. / n_indexs)
weightsExt = weights
for chan in range(1, len(launch_vectors[0])):
    viewing_angles_d = np.append(viewing_angles_d, np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, chan, 0])]))
    viewing_angles_r = np.append(viewing_angles_r, np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, chan, 1])]))
    rho = np.append(rho, np.arccos(1. / n_indexs))
    weightsExt = np.append(weightsExt, weights)
dCherenkov_d = (viewing_angles_d - rho) / units.deg
dCherenkov_r = (viewing_angles_r - rho) / units.deg
mask_d = ~np.isnan(viewing_angles_d)
mask_r = ~np.isnan(viewing_angles_r)
plt.subplot(2, 1, 1)
plt.hist([dCherenkov_d[mask_d], dCherenkov_r[mask_r]], weights = [weightsExt[mask_d],  weightsExt[mask_r]], bins = np.arange(-30, 30, 1))
plt.xlabel('weighted viewing - cherenkov angle [deg]')
avgChe_d = np.average(dCherenkov_d[mask_d], weights = weightsExt[mask_d])
varChe_d = np.average((dCherenkov_d[mask_d] - avgChe_d)**2, weights = weightsExt[mask_d])
avgChe_r = np.average(dCherenkov_r[mask_r], weights = weightsExt[mask_r])
varChe_r = np.average((dCherenkov_r[mask_r] - avgChe_r)**2, weights = weightsExt[mask_r])
plt.figtext(1.0, 0.6, "N: " + str(len(dCherenkov_d[mask_d])) + "; Blue: direct ray\nmean: " + str(np.average(dCherenkov_d[mask_d], weights=weightsExt[mask_d])) + "\nstd: " + str(varChe_d**0.5) + "\nN: " + str(len(dCherenkov_r[mask_r])) + "; Red: refracted ray\nmean: " + str(np.average(dCherenkov_r[mask_r], weights=weightsExt[mask_r])) + "\nstd: " + str(varChe_r**0.5))
plt.subplot(2, 1, 2)
plt.hist([dCherenkov_d[mask_d], dCherenkov_r[mask_r]], bins = np.arange(-30, 30, 1))
plt.xlabel('unweighted viewing - cherenkov angle [deg]')
plt.figtext(1.0, 0.2, "N: " + str(len(dCherenkov_d[mask_d])) + "; Blue: direct ray\nmean: " + str(np.average(dCherenkov_d[mask_d])) + "\nstd: " + str(np.std(dCherenkov_d[mask_d])) + "\nN: " + str(len(dCherenkov_r[mask_r])) + "; Red: refracted ray\nmean: " + str(np.average(dCherenkov_r[mask_r])) + "\nstd: " + str(np.std(dCherenkov_r[mask_r])))
plt.savefig(os.path.join(plot_folder, 'dCherenkov.pdf'), bbox_inches="tight")
plt.clf()

#plot viewing_refracted vs viewing_direct
plt.scatter(viewing_angles_d / units.deg, viewing_angles_r / units.deg, 10)
plt.xlabel("direct viewing angle")
plt.ylabel("refracted viewing angle")
plt.axis([20, 110, 20, 110])
plt.grid(True)
plt.savefig(os.path.join(plot_folder, 'viewDvsviewR.pdf'), bbox_inches="tight")

# plot C0 parameter
# C0s = np.array(fin['ray_tracing_C0'])
# php.get_histogram(C0s.flatten())
# fig.suptitle('incoming signal direction')
#plt.show()
