import numpy as np
import h5py
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units
from scipy import constants
from matplotlib import pyplot as plt
import glob
from radiotools import plthelpers as php
import sys
import os

Tnoise = 300
Vrms = (Tnoise * 50 * constants.k * 500 * units.MHz / units.Hz) ** 0.5

counter = -1
fig, ax = plt.subplots(1, 1, figsize=(7,7))
for iF, filename in enumerate(sorted(glob.glob(os.path.join(sys.argv[1], "*.hdf5")))):

    fin = h5py.File(filename)
    print(np.log10(fin.attrs['Emin']))
    if(np.log10(fin.attrs['Emin']) not in [17.0, 18.0, 19.0, 20.0]):
        continue
    counter += 1

    with open('det.json', 'w') as fout:
        fout.write(fin.attrs['detector'])
        fout.close()
    det = detector.Detector(json_filename="det.json")
    max_amps_env = np.array(fin['station_101']['maximum_amplitudes_envelope'])
    weights = fin['weights']


    n_stations = det.get_number_of_channels(101) / 4  # (we had 4 antennas per station)
    xs = np.zeros(n_stations)
    ys = np.zeros(n_stations)

    triggered_near_surface = np.any(max_amps_env[:, 0:3] > (3 * Vrms), axis=1)  # triggered any LPDA or dipole of the center station
    triggered_near_deep = max_amps_env[:, 3] > (3 * Vrms) # triggered deep dipole
    triggered_surface = np.zeros((max_amps_env.shape[0], n_stations)) # create empy array of shape (n events, n stations)
    triggered_deep = np.zeros((max_amps_env.shape[0], n_stations)) # create empy array of shape (n events, n stations)
    # loop through all stations with different distances
    for i in range(n_stations):
        # select the 2 LPDA + 1 dipole channel and check if they fulfill the trigger condition
        triggered_surface[:, i] = np.any(max_amps_env[:, i * 4:(i * 4 + 3)] > (3 * Vrms), axis=1)
        triggered_deep[:, i] = max_amps_env[:, i * 4 + 3] > (3 * Vrms)
        # get their position
        xs[i] = np.abs(det.get_relative_position(101, i * 4)[0])
        ys[i] = np.abs(det.get_relative_position(101, i * 4)[1])

    # loop through all simulated distances
    coincidence_fractions_surface = np.zeros(len(np.unique(xs)))
    coincidence_fractions_deep = np.zeros(len(np.unique(xs)))
    for i, x in enumerate(np.unique(xs)):
        mask = ((np.abs(xs) == x) & (np.abs(ys) == x))  \
            | ((np.abs(xs) == x) & (ys == 0)) \
            | ((np.abs(ys) == x) & (xs == 0))
        # calculate coincidence fraction
        coincidence_fractions_surface[i] = 1. * np.sum(weights[np.any(triggered_surface[:, mask], axis=1) & triggered_near_surface]) / np.sum(weights[triggered_near_surface])
        coincidence_fractions_deep[i] = 1. * np.sum(weights[np.any(triggered_deep[:, mask], axis=1) & triggered_near_deep]) / np.sum(weights[triggered_near_deep])

    ax.plot(np.unique(xs), coincidence_fractions_surface, php.get_marker2(counter)+'-', label="E = {:.2g}eV".format(fin.attrs['Emin']))
    ax.plot(np.unique(xs), coincidence_fractions_deep, php.get_marker2(counter)+'--') #, label="E = {:.2g}, 50m deep".format(fin.attrs['Emin']))
ax.set_xlabel("distance [m]")
ax.set_ylabel("coincidence fraction")
ax.semilogy(True)
ax.set_aspect('equal', 'box')
ax.set_xlim(0, 3000)
ax.set_ylim(0.5e-2, 1.1)
ax.legend(loc='lower left', fontsize='medium', numpoints=1)
fig.tight_layout()
fig.savefig("coincidences.png")
fig.savefig("coincidences.pdf", bbox='tight')
plt.show()
