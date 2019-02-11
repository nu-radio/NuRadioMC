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

if __name__ == "__main__":
    sampling_rate = 5 * units.GHz
    dt = 1./sampling_rate
    T=50 * units.ns
    N = np.int(np.round(T/dt))
    tt = np.arange(0, dt*N, dt)
    n_index = 1.78
    theta = np.arccos(1./n_index)
    R = 5 * units.km
    
    dCs = np.arange(-15, 15, 2.) * units.deg
    
    lib_path = "/Users/cglaser/work/ARIANNA/data/ARZ/v1.1/library_v1.1.pkl"
    a = ARZ.ARZ(library=lib_path, interp_factor=100)
    
    showers = {}
    showers['meta'] = {'dt': dt,
                       'N': N, 
                       'n_index': n_index,
                       'R': 5/units.km}
    
    with open(lib_path, 'rb') as fin:
        lib = pickle.load(fin)
        for iS, shower_type in enumerate(lib):  # loop through shower types
            if(shower_type not in showers):
                showers[shower_type] = {}
            b = []
            nb = []
            for iE, E in enumerate(lib[shower_type]):  # loop through energies
                if(E < 1e19*units.eV):
                    continue
                if(shower_type == "EM" and E >= 1e17*units.eV):
                    T = 500 * units.ns
                    N = np.int(np.round(T/dt))
                    tt = np.arange(0, dt*N, dt)
                else:
                    T=50 * units.ns
                    N = np.int(np.round(T/dt))
                    tt = np.arange(0, dt*N, dt)
                
                
                nE = len(lib[shower_type][E]['charge_excess'])
                if(E not in showers[shower_type]):
                    showers[shower_type][E] = {}
                for iN in range(nE):  # loop through shower realizations
                    if(iN not in showers[shower_type][E]):
                        showers[shower_type][E][iN] = {}
                    fig, ax = plt.subplots(1, 1)
                    fig2, ax2 = plt.subplots(1, 1)
                    for dC in dCs:  # loop through different cherenkov angles
                        theta1 = theta + dC
                        trace = a.get_time_trace(E, theta1, N, dt, shower_type, n_index, R, iN=iN)
                        showers[shower_type][E][iN][theta1] = trace[1]
                        ax.plot(tt, trace[1]/np.abs(trace[1]).max())
                        ax2.plot(tt, np.abs(signal.hilbert(trace[1])))
                    ax.set_title("{} E = {:.2g}eV i = {:d}".format(shower_type, E/units.eV, iN))
                    ax2.set_title("{} E = {:.2g}eV i = {:d}".format(shower_type, E/units.eV, iN))
                    ax.set_xlabel('time [ns]')
                    ax2.set_xlabel('time [ns]')
                    ax.set_ylabel('normalized amplitude')
                    ax2.semilogy(True)
                    fig.tight_layout()
                    fig2.tight_layout()
                    fig.savefig("plots/{}_{:.2g}eV_{:03d}.png".format(shower_type, E/units.eV, iN))
                    fig2.savefig("plots/{}_{:.2g}eV_{:03d}_log.png".format(shower_type, E/units.eV, iN))
                    plt.close("all")
                            
        with open("ARZ_library_v1.1.pkl", 'wb') as fout:
            pickle.dump(showers, fout, protocol=2)