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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

parser = argparse.ArgumentParser(description='Plot NuRadioMC event list output.')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC hdf5 simulation output')
parser.add_argument('--trigger_name', type=str, default=None, nargs='+',
                    help='the name of the trigger that should be used for the plots')
parser.add_argument('--Veff', type=str,
                    help='specify json file where effective volume is saved as a function of energy')
args = parser.parse_args()

filename = os.path.splitext(os.path.basename(args.inputfilename))[0]
dirname = os.path.dirname(args.inputfilename)
plot_folder = os.path.join(dirname, 'plots', filename)
if(not os.path.exists(plot_folder)):
    os.makedirs(plot_folder)

fin = h5py.File(args.inputfilename, 'r')
print('the following triggeres where simulated: {}'.format(fin.attrs['trigger_names']))
if(args.trigger_name is None):
    triggered = np.array(fin['triggered'])
    print("you selected any trigger")
    trigger_name = 'all'
else:
    if(len(args.trigger_name) > 1):
        print("trigger {} selected which is a combination of {}".format(args.trigger_name[0], args.trigger_name[1:]))
        trigger_name = args.trigger_name[0]
        plot_folder = os.path.join(dirname, 'plots', filename, args.trigger_name[0])
        if(not os.path.exists(plot_folder)):
            os.makedirs(plot_folder)
        triggered = np.zeros(len(fin['multiple_triggers'][:, 0]), dtype=np.bool)
        for trigger in args.trigger_name[1:]:
            iTrigger = np.squeeze(np.argwhere(fin.attrs['trigger_names'] == trigger))
            triggered = triggered | np.array(fin['multiple_triggers'][:, iTrigger], dtype=np.bool)
    else:
        trigger_name = args.trigger_name[0]
        iTrigger = np.argwhere(fin.attrs['trigger_names'] == trigger_name)
        triggered = np.array(fin['multiple_triggers'][:, iTrigger], dtype=np.bool)
        print("\tyou selected '{}'".format(trigger_name))
        plot_folder = os.path.join(dirname, 'plots', filename, trigger_name)
        if(not os.path.exists(plot_folder)):
            os.makedirs(plot_folder)

weights = np.array(fin['weights'])[triggered]
n_events = fin.attrs['n_events']

# calculate effective
density_ice = 0.9167 * units.g / units.cm ** 3
density_water = 997 * units.kg / units.m ** 3

n_triggered = np.sum(weights)
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
Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights) / n_events

print("Veff = {:.6g} km^3 sr".format(Veff / units.km ** 3))

fig, ax = php.get_histogram(np.array(fin['zeniths'])[triggered] / units.deg, weights=weights,
                            ylabel='weighted entries', xlabel='zenith angle [deg]',
                            bins=np.arange(0, 181, 5), figsize=(6, 6))
ax.set_xticks(np.arange(0, 181, 45))
ax.set_title(trigger_name)
fig.tight_layout()
fig.savefig(os.path.join(plot_folder, 'neutrino_direction.png'))

# calculate sky coverage of 90% quantile
from radiotools import stats
q2 =stats.quantile_1d(np.array(fin['zeniths'])[triggered], weights, 0.95)
q1 =stats.quantile_1d(np.array(fin['zeniths'])[triggered], weights, 0.05)
from scipy import integrate
def a(theta):
    return np.sin(theta)
b = integrate.quad(a, q1, q2)
print("90% quantile sky coverage {:.2f} sr".format(b[0] * 2 * np.pi))

# plot vertex distribution
fig, ax = plt.subplots(1, 1)
xx = np.array(fin['xx'])[triggered]
yy = np.array(fin['yy'])[triggered]
rr = (xx ** 2 + yy ** 2) ** 0.5
zz = np.array(fin['zz'])[triggered]

mask_weight = weights > 1e-2
max_r = max(np.abs(xx[mask_weight]).max(), np.abs(yy[mask_weight]).max())
max_z = np.abs(zz[mask_weight]).max()
h = ax.hist2d(rr / units.m, zz / units.m, bins=[np.linspace(0, max_r, 50), np.linspace(-max_z, 0, 50)],
              cmap=plt.get_cmap('Blues'), weights=weights)
cb = plt.colorbar(h[3], ax=ax)
cb.set_label("weighted number of events")
ax.set_aspect('equal')
ax.set_xlabel("r [m]")
ax.set_ylabel("z [m]")
ax.set_xlim(0, rmax)
ax.set_ylim(fin.attrs['zmin'], 0)
ax.set_title(trigger_name)
fig.tight_layout()
fig.savefig(os.path.join(plot_folder, 'vertex_distribution.png'))
# plot incoming direction
receive_vectors = np.array(fin['receive_vectors'])[triggered]
# for all events, antennas and ray tracing solutions
zeniths, azimuths = hp.cartesian_to_spherical(receive_vectors[:, :, :, 0].flatten(),
                                                         receive_vectors[:, :, :, 1].flatten(),
                                                         receive_vectors[:, :, :, 2].flatten())
for i in range(len(azimuths)):
    azimuths[i] = hp.get_normalized_angle(azimuths[i])
weights_matrix = np.outer(weights, np.ones(np.prod(receive_vectors.shape[1:-1]))).flatten()
mask = ~np.isnan(azimuths)  # exclude antennas with not ray tracing solution (or with just one ray tracing solution)
fig, axs = php.get_histograms([zeniths[mask] / units.deg, azimuths[mask] / units.deg],
                              bins=[np.arange(0, 181, 5), np.arange(0, 361, 45)],
                              xlabels=['zenith [deg]', 'azimuth [deg]'],
                              weights=weights_matrix[mask], stats=False)
# axs[0].xaxis.set_ticks(np.arange(0, 181, 45))
majorLocator = MultipleLocator(45)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(5)
axs[0].xaxis.set_major_locator(majorLocator)
axs[0].xaxis.set_major_formatter(majorFormatter)
# for the minor ticks, use no labels; default NullFormatter
axs[0].xaxis.set_minor_locator(minorLocator)

fig.suptitle('incoming signal direction')
fig.savefig(os.path.join(plot_folder, 'incoming_signal.png'))

# plot polarization
polarization = np.array(fin['polarization'])[triggered].flatten()
polarization = np.abs(polarization)
polarization[polarization > 90 * units.deg] = 180 * units.deg - polarization[polarization > 90 * units.deg]
bins = np.linspace(0, 90, 50)

# for all events, antennas and ray tracing solutions
# mask = zeniths > 90 * units.deg  # select rays coming from below
# fig, ax = php.get_histogram(polarization / units.deg,
#                             bins=bins,
#                             xlabel='polarization [deg]',
#                             weights=weights_matrix, stats=False,
#                             figsize=(6, 6))
# maxy = ax.get_ylim()
# php.get_histogram(polarization[mask] / units.deg,
#                   bins=bins,
#                   xlabel='polarization [deg]',
#                   weights=weights_matrix[mask], stats=False,
#                   ax=ax, kwargs={'facecolor': 'C0', 'alpha': 1, 'edgecolor': "k"})
# # ax.set_xticks(bins)
# ax.set_ylim(maxy)
# fig.tight_layout()
# fig.savefig(os.path.join(plot_folder, 'polarization.png'))


# fig, ax = php.get_histogram(polarization / units.deg,
#                             bins=bins,
#                             xlabel='polarization [deg]',
#                             weights=weights_matrix, stats=False,
#                             figsize=(6, 6))
# maxy = ax.get_ylim()
# php.get_histogram(polarization[mask] / units.deg,
#                   bins=bins,
#                   xlabel='polarization [deg]',
#                   stats=False,
#                   ax=ax, kwargs={'facecolor': 'C0', 'alpha': 1, 'edgecolor': "k"})
# # ax.set_xticks(bins)
# ax.set_ylim(max(ax.get_ylim(), maxy))
# fig.tight_layout()
# fig.savefig(os.path.join(plot_folder, 'polarization_unweighted.png'))



shower_axis = -1 * hp.spherical_to_cartesian(np.array(fin['zeniths'])[triggered], np.array(fin['azimuths'])[triggered])
launch_vectors = np.array(fin['launch_vectors'])[triggered]
viewing_angles = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 0, 0])])

# calculate correct chereknov angle for ice density at vertex position
ice = medium.southpole_simple()
n_indexs = np.array([ice.get_index_of_refraction(x) for x in np.array([np.array(fin['xx'])[triggered], np.array(fin['yy'])[triggered], np.array(fin['zz'])[triggered]]).T])
rho = np.arccos(1. / n_indexs)

mask = ~np.isnan(viewing_angles)
fig, ax = php.get_histogram((viewing_angles[mask] - rho[mask]) / units.deg, weights=weights[mask],
                            bins=np.arange(-30, 30, 1), xlabel='viewing - cherenkov angle [deg]', figsize=(6, 6))
fig.savefig(os.path.join(plot_folder, 'dCherenkov.png'))

# SNR
flavor_labels = ['e cc', r'$\bar{e}$ cc', 'e nc', r'$\bar{e}$ nc', 
           '$\mu$ cc', r'$\bar{\mu}$ cc', '$\mu$ nc', r'$\bar{\mu}$ nc',
           r'$\tau$ cc', r'$\bar{\tau}$ cc', r'$\tau$ nc', r'$\bar{\tau}$ nc']
yy = np.zeros(len(flavor_labels))
yy[0] = np.sum(weights[(fin['flavors'][triggered] == 12) & (fin['ccncs'][triggered] == 'cc')])
yy[1] = np.sum(weights[(fin['flavors'][triggered] == -12) & (fin['ccncs'][triggered] == 'cc')])
yy[2] = np.sum(weights[(fin['flavors'][triggered] == 12) & (fin['ccncs'][triggered] == 'nc')])
yy[3] = np.sum(weights[(fin['flavors'][triggered] == -12) & (fin['ccncs'][triggered] == 'nc')])

yy[4] = np.sum(weights[(fin['flavors'][triggered] == 14) & (fin['ccncs'][triggered] == 'cc')])
yy[5] = np.sum(weights[(fin['flavors'][triggered] == -14) & (fin['ccncs'][triggered] == 'cc')])
yy[6] = np.sum(weights[(fin['flavors'][triggered] == 14) & (fin['ccncs'][triggered] == 'nc')])
yy[7] = np.sum(weights[(fin['flavors'][triggered] == -14) & (fin['ccncs'][triggered] == 'nc')])

yy[8] = np.sum(weights[(fin['flavors'][triggered] == 16) & (fin['ccncs'][triggered] == 'cc')])
yy[9] = np.sum(weights[(fin['flavors'][triggered] == -16) & (fin['ccncs'][triggered] == 'cc')])
yy[10] = np.sum(weights[(fin['flavors'][triggered] == 16) & (fin['ccncs'][triggered] == 'nc')])
yy[11] = np.sum(weights[(fin['flavors'][triggered] == -16) & (fin['ccncs'][triggered] == 'nc')])

fig, ax = plt.subplots(1, 1)
ax.bar(range(len(flavor_labels)), yy)
ax.set_xticks(range(len(flavor_labels)))
ax.set_xticklabels(flavor_labels)
ax.set_title(trigger_name)
ax.set_ylabel('weighted number of triggers')
fig.tight_layout()
fig.savefig(os.path.join(plot_folder, 'flavor.png'))
plt.show()

# flavor

# plot C0 parameter
# C0s = np.array(fin['ray_tracing_C0'])
# php.get_histogram(C0s.flatten())
# fig.suptitle('incoming signal direction')

# plt.show()
