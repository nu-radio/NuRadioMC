from __future__ import print_function, division
import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.utilities import io_utilities
import os
from scipy import interpolate as intp
import glob
import pickle
import sys
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from NuRadioMC.SignalGen.ARZ import ARZ
import logging
from multiprocessing import Process, Pool
from scipy import signal
import matplotlib.gridspec as gridspec

logger = logging.getLogger("test")
logging.basicConfig(level=logging.WARNING)
rho = 0.924 * units.g / units.cm ** 3  # density g cm^-3

sampling_rate = 5 * units.GHz
dt = 1. / sampling_rate
T2 = 50 * units.ns
N2 = int(np.round(T2 / dt))
n_index = 1.78
theta = np.arccos(1. / n_index)
R = 5 * units.km
T = 600 * units.ns
N = int(np.round(T / dt))
tt = np.arange(0, dt * N, dt)

dCs = np.append(np.arange(-20, -5, 1), np.append(np.arange(-5,-1,0.2), np.arange(-1, 0, .05)))
dCs = np.append(np.append(dCs, [0]), -1 * dCs)
dCs = np.sort(dCs) * units.deg

plot=False

def calculate(shower_type, E, nE):
    lib_path = "/Users/cglaser/work/ARIANNA/data/ARZ/v1.1/library_v1.1.pkl"
    a = ARZ.ARZ(library=lib_path, interp_factor=100)
    showers['meta'] = {'dt': dt,
                       'n_index': n_index,
                       'R': R}
    showers[shower_type] = {}
    if(E not in showers[shower_type]):
        showers[shower_type][E] = {}
    for iN in range(nE):  # loop through shower realizations
        print("{} {:.2g} {:d}".format(shower_type, E, iN))
        if(iN not in showers[shower_type][E]):
            showers[shower_type][E][iN] = {}
        if plot:
            fig, ax = plt.subplots(1, 1)
            fig2, ax2 = plt.subplots(1, 1)
        for dC in dCs:  # loop through different cherenkov angles
            theta1 = theta + dC
            trace, Lmax = a.get_time_trace(E, theta1, N, dt, shower_type, n_index, R, iN=iN, output_mode='Xmax')
            iMax = np.argmax(np.abs(trace[1]))
            i1 = iMax - N2 // 2
            if((i1 + N2) >= N) or (i1 < 0):
                print("tracelength to short, i1 = {}".format(i1))
#                             raise IndexError("tracelength to short, i1 = {}".format(i1))
            t0 = tt[i1] - tt.mean()
            showers[shower_type][E][iN][theta1] = {}
            showers[shower_type][E][iN][theta1]['t0'] = t0
            showers[shower_type][E][iN][theta1]['Lmax'] = Lmax
            showers[shower_type][E][iN][theta1]['trace'] = trace[1][i1:(i1 + N2)]
            if(plot):
                ax.plot(tt[i1:(i1 + N2)], trace[1][i1:(i1 + N2)] / np.abs(trace[1]).max())
                ax2.plot(tt[i1:(i1 + N2)], np.abs(signal.hilbert(trace[1][i1:(i1 + N2)])))
        if(plot):
            ax.set_title("{} E = {:.2g}eV i = {:d}".format(shower_type, E / units.eV, iN))
            ax2.set_title("{} E = {:.2g}eV i = {:d}".format(shower_type, E / units.eV, iN))
            ax.set_xlabel('time [ns]')
            ax2.set_xlabel('time [ns]')
            ax.set_ylabel('normalized amplitude')
            ax2.semilogy(True)
            fig.tight_layout()
            fig2.tight_layout()
#                         plt.show()
            fig.savefig("plots/{}_{:.2g}eV_{:03d}.png".format(shower_type, E / units.eV, iN))
            fig2.savefig("plots/{}_{:.2g}eV_{:03d}_log.png".format(shower_type, E / units.eV, iN))
            plt.close("all")
    print("finished with  {} {:.3g}".format(shower_type, E))
    with open("ARZ_library_v1.1_{}_{:.1f}.pkl".format(shower_type, np.log10(E)), 'wb') as fout:
        pickle.dump(showers, fout, protocol=2)

if __name__ == "__main__":
    print('generating Askaryan pulses for the following viewing angles')
    print(dCs/units.deg)

    ps = []
    showers = {}
    lib_path = "/Users/cglaser/work/ARIANNA/data/ARZ/v1.1/library_v1.1.pkl"

    pool = Pool(processes=4)


    lib = io_utilities.read_pickle(lib_path)
    for iS, shower_type in enumerate(lib):  # loop through shower types
        if(shower_type not in showers):
            showers[shower_type] = {}
        print("shower type {}".format(shower_type))
        b = []
        nb = []
        for iE, E in enumerate(lib[shower_type]):  # loop through energies
            print('E = {:.2g}eV'.format(E))

#                 p = Process(target=calculate, args=(shower_type, E, len(lib[shower_type][E]['charge_excess'])))
#                 p.start()
#                 ps.append(p)
            r = pool.apply_async(calculate, args=(shower_type, E, len(lib[shower_type][E]['charge_excess'])))
            ps.append(r)
    for res in ps:
        print(res.get())
