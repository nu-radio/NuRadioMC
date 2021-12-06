from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import analyticraytracing as ray
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
if __name__ == "__main__":
    args = parser.parse_args()

    fin = h5py.File(args.inputfilename, 'r')

    weights = np.array(fin['weights'])
    triggered = np.array(fin['triggered'])
    n_events = fin.attrs['n_events']

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

    mask = (zz < -2000 * units.m) & (rr < 5000 * units.m)
    for i in np.array(range(len(xx)))[mask]:
    #     C0 = fin['ray_tracing_C0'][i][0][0]
    #     C1 = fin['ray_tracing_C1'][i][0][0]
        print('weight = {:.2f}'.format(weights[i]))
        x1 = np.array([xx[i], yy[i], zz[i]])
        x2 = np.array([0, 0, -5])
        r = ray.ray_tracing(x1, x2, medium.southpole_simple())
        r.find_solutions()
        C0 = r.get_results()[0]['C0']
        x1 = np.array([-rr[i], zz[i]])
        x2 = np.array([0, -5])
        r2 = ray.ray_tracing_2D(medium.southpole_simple())
        yyy, zzz = r2.get_path(x1, x2, C0)

        launch_vector = fin['launch_vectors'][i][0][0]
        print(launch_vector)
        zenith_nu = fin['zeniths'][i]
        print(zenith_nu / units.deg)

        fig, ax = plt.subplots(1, 1)
        ax.plot(-rr[i], zz[i], 'o')
        ax.plot([-rr[i], -rr[i] + 100 * np.cos(zenith_nu)], [zz[i], zz[i] + 100 * np.sin(zenith_nu)], '-C0')
        ax.plot(0, -5, 'd')
        ax.plot(yyy, zzz)
        ax.set_aspect('equal')
        plt.show()
