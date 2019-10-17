import numpy as np
import h5py
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units
from scipy import constants
from matplotlib import pyplot as plt
import glob
from radiotools import plthelpers as php
import sys
from radiotools import helper as hp
import os
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from mpl_toolkits import mplot3d
import matplotlib
import logging

Tnoise = 300
Vrms = (Tnoise * 50 * constants.k * 500 * units.MHz / units.Hz) ** 0.5

plot=False
counter = -1
filename = sys.argv[1]
x = float(sys.argv[2]) * units.m
print("using file {} and distance {:.0f}m".format(filename, x/units.m))

fin = h5py.File(filename)
print(np.log10(fin.attrs['Emin']))

with open('det.json', 'w') as fout:
    fout.write(fin.attrs['detector'])
    fout.close()
det = detector.Detector(json_filename="det.json")
max_amps_env = np.array(fin['station_101']['maximum_amplitudes_envelope'])
weights = fin['weights']

n_channels = det.get_number_of_channels(101)
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

med = medium.ARAsim_southpole()


mask = ((np.abs(xs) == x) & (np.abs(ys) == x))  \
    | ((np.abs(xs) == x) & (ys == 0)) \
    | ((np.abs(ys) == x) & (xs == 0))

theta  = np.arccos(1. / 1.78)

n_events = len(triggered_near_surface)
dTs = []
for iE in np.arange(n_events, dtype=np.int)[np.any(triggered_deep[:, mask], axis=1) & triggered_near_deep]:
    if(iE % 100 == 0):
        print(iE)
    if(plot):
        ax = plt.axes(projection='3d')
#     x1 = det.get_relative_position(101, 3)
#     ax.plot([x1[0]], [x1[1]], [x1[2]], 'ko')
    for j in np.append([0], np.arange(n_stations)[mask]):
        iC = j * 4 + 3
        x2 = det.get_relative_position(101, iC)
        if(plot):
            ax.plot([x2[0]], [x2[1]], [x2[2]], 'ko')
        if(j != 0 and (~(np.array(triggered_deep[iE], dtype=np.bool) & mask))[j]):
            continue
        vertex = np.array([fin['xx'][iE], fin['yy'][iE], fin['zz'][iE]])
#         print(fin.keys())
#         l1 = fin['launch_vectors'][iE][3] / units.deg
        l2 = fin['station_101']['launch_vectors'][iE][j*4 + 3]

#         r1 = ray.ray_tracing(vertex, x1, med, log_level=logging.DEBUG)
# #         r1.find_solutions()
#         r1.set_solution(fin['ray_tracing_C0'][iE][3], fin['ray_tracing_C1'][iE][3], fin['ray_tracing_solution_type'][iE][3])
#         path1 = r1.get_path(0)

        zen, az = fin['zeniths'][iE], fin['azimuths'][iE]
        v = hp.spherical_to_cartesian(zen, az)

        if(plot):
            r2 = ray.ray_tracing(vertex, x2, med, log_level=logging.INFO)
            r2.set_solution(fin['station_101']['ray_tracing_C0'][iE][iC], fin['station_101']['ray_tracing_C1'][iE][iC], fin['station_101']['ray_tracing_solution_type'][iE][iC])
            path2 = r2.get_path(0)
    #         ax.plot3D(path1.T[0], path1.T[1], path1.T[2], label='path 1')
            ax.plot3D(path2.T[0], path2.T[1], path2.T[2], label='path {}'.format(j))
            ax.plot3D([vertex[0], vertex[0] + 500*v[0]], [vertex[1], vertex[1] + 500*v[1]], [vertex[2], vertex[2] + 500*v[2]],
                      '--', label='shower direction')

        dT = []
        for l in l2:
#             print("{:.1f}".format((theta - hp.get_angle(-v, l))/units.deg))
            dT.append((theta - hp.get_angle(-v, l))/units.deg)
        dTs.append(np.min(np.abs(np.array(dT))))
        if(plot):
            R3 = hp.get_rotation(np.array([0, 0, 1]), -v)

            for phi in np.linspace(0, 2 * np.pi, 200):

                l = hp.spherical_to_cartesian(theta, phi)

    #             zen, az = hp.cartesian_to_spherical(v[0], v[1], v[2])
                R1 = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
                R2 = np.array([[np.cos(zen), 0, -np.sin(zen)], [0, 1, 0], [np.sin(zen), 0, np.cos(zen)]])
    #             l2 = np.matmul(R1, np.matmul(R2,l))
                l2 = np.matmul(R3, l)
    #             l2 = np.matmul(R2, np.matmul(R1, l))
                l2 *= 400
#                 print(hp.get_angle(v, l2)/units.deg)
                l2 += vertex
                ax.plot3D([vertex[0], l2[0]], [vertex[1], l2[1]], [vertex[2], l2[2]], 'C3-', alpha=0.2)
    if(plot):
        ax.legend()
    #     ax.set_aspect('equal')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
    #     plt.ion()
        plt.show()
#     a =     1/0
php.get_histogram(np.array(dTs))
plt.show()
