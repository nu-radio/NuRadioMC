from __future__ import print_function, division
import numpy as np
from NuRadioMC.utilities import units
import os
from scipy import interpolate as intp
import glob
import pickle
import sys
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from NuRadioMC.SignalGen.ARZ import ARZ
import logging
from scipy import signal
import matplotlib.gridspec as gridspec

logger = logging.getLogger("test")
logging.basicConfig(level=logging.INFO)
rho = 0.924 * units.g / units.cm ** 3  # density g cm^-3

plot=False

if __name__ == "__main__":
    sampling_rate = 5 * units.GHz
    dt = 1. / sampling_rate
    T = 500 * units.ns
    T2 = 50 * units.ns
    N = np.int(np.round(T / dt))
    N2 = np.int(np.round(T2 / dt))
    tt = np.arange(0, dt * N, dt)
    n_index = 1.78
    theta = np.arccos(1. / n_index)
    R = 5 * units.km
    
    dCs = np.arange(-15, 15, .1) * units.deg
    
    lib_path = "/Users/cglaser/work/ARIANNA/data/ARZ/v1.1/library_v1.1.pkl"
    a = ARZ.ARZ(library=lib_path, interp_factor=100)
    
    showers = {}
    showers['meta'] = {'dt': dt,
                       'n_index': n_index,
                       'R': R}
    
    with open(lib_path, 'rb') as fin:
        lib = pickle.load(fin)
        for iS, shower_type in enumerate(lib):  # loop through shower types
            if(shower_type not in showers):
                showers[shower_type] = {}
            b = []
            nb = []
            for iE, E in enumerate(lib[shower_type]):  # loop through energies
                if(E < 1e19 * units.eV):
                    continue
                
                nE = len(lib[shower_type][E]['charge_excess'])
                if(E not in showers[shower_type]):
                    showers[shower_type][E] = {}
                for iN in range(nE):  # loop through shower realizations
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
                        t0 = tt[i1] - tt.mean()
                        print("t0 = {:.2f}".format(t0))
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
                            
        with open("ARZ_library_v1.1.pkl", 'wb') as fout:
            pickle.dump(showers, fout, protocol=2)
