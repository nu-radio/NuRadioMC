#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from NuRadioReco.utilities import units
from scipy import interpolate as intp
from scipy import constants
from matplotlib import pyplot as plt
import os
import pickle
from time import time
from NuRadioMC.SignalGen.ARZ.ARZ import get_vector_potential_fast
import json

plot = True

energy = 1.e6 * units.TeV
N = 1000
dt = 0.1 * units.ns
times = np.arange(0, N * dt, dt)
times = times + 0.5 * dt - times.mean()
limits = ( int(N/2-25*units.ns/dt), int(N/2+25*units.ns/dt) )
n_index = 1.78
y = 0
ccnc = 'cc'
flavor = 12  # e = 12, mu = 14, tau = 16

cdir = os.path.dirname(__file__)
bins, depth_e, N_e = np.loadtxt(os.path.join(cdir, "../shower_library/nue_1EeV_CC_1_s0001.t1005"), unpack=True)
bins, depth_p, N_p = np.loadtxt(os.path.join(cdir, "../shower_library/nue_1EeV_CC_1_s0001.t1006"), unpack=True)
depth_e *= units.g / units.cm**2
depth_p *= units.g / units.cm**2
depth_e -= 1000 * units.g/units.cm**2  # all simulations have an artificial offset of 1000 g/cm^2
depth_p -= 1000 * units.g/units.cm**2
# sanity check if files electron and positron profiles are compatible
if (not np.all(depth_e == depth_p)):
    raise ImportError("electron and positron profile have different depths")

max_interp = 200
Rs = np.linspace(100, 1000, 10)
Rs = [100, 200, 500, 1000]
thetas = np.linspace(45, 65, 10)*units.deg
interp_factors_2 = np.linspace(1,100,100)
interp_factors_1 = np.linspace(1,100,100)
max_prod = 500
tol = 1.e-1

res_dict = {}

for R in Rs:

    interp_results = {}

    for theta in thetas:

        fields = []
        diffs = []

        vp_prec = get_vector_potential_fast(energy, theta, N, dt, depth_e, N_e-N_p, "EM", n_index, R, \
        interp_factor=max_interp, interp_factor2=max_interp, shift_for_xmax=True)
        field_prec = -np.array([np.diff(vp_prec[:,0]),np.diff(vp_prec[:,1]),np.diff(vp_prec[:,2])])/dt

        for interp_factor_1 in interp_factors_1:

            found = False

            tol1 = 0
            tol2 = 1

            for interp_factor_2 in interp_factors_2:

                if (interp_factor_1*interp_factor_2 > max_prod):
                    break

                prev_tol1 = tol1
                prev_tol2 = tol2

                print("R: ", R, ", theta: ", theta/units.deg, ", interpolation", interp_factor_1, interp_factor_2)

                vp = get_vector_potential_fast(energy, theta, N, dt, depth_e, N_e-N_p, "EM", n_index, R, \
                interp_factor=interp_factor_1, interp_factor2=interp_factor_2, shift_for_xmax=True)
                field = -np.array([np.diff(vp[:,0]),np.diff(vp[:,1]),np.diff(vp[:,2])])/dt
                #fields.append(field)

                diff = np.abs(field_prec-field)/np.abs(field_prec)
                diff = np.nan_to_num(diff)
                #diffs.append(diff)

                #print(np.mean(diff[0]),np.mean(diff[2]))
                #plt.yscale('log')

                tol1 = np.mean(diff[0][limits[0]:limits[1]])
                tol2 = np.mean(diff[2][limits[0]:limits[1]])
                print(tol1, tol2)

                if ( np.abs(prev_tol1-tol1) < tol/1e2 and np.abs(prev_tol2-tol2) < tol/1e2):
                     #and np.abs(tol1-tol) > 0.1*tol and np.abs(tol2-tol) > 0.1*tol ):
                    break

                #if (np.mean(diff[0][diff[0]>0]) < tol and np.mean(diff[2][diff[2]>0]) < tol ):
                #if (np.min(diff[0][diff[0]>0]) < tol and np.min(diff[2][diff[2]>0]) < tol ):
                if ( np.mean(diff[0][limits[0]:limits[1]]) < tol and
                     np.mean(diff[2][limits[0]:limits[1]]) < tol ):
                    interp_results[theta/units.deg] = (interp_factor_1, interp_factor_2)
                    found = True
                    print('found!')
                    if plot:
                        plt.plot(times,field_prec[0])
                        plt.plot(times,field[0])
                        plt.show()
                        plt.plot(times, diff[0])
                        plt.yscale('log')
                        plt.show()
                    break
                plt.clf()

            if found:
                break
            else:
                print("Tolerance not achieved! More interp_factor_1!")
    print("R: ", R)
    print("interp results", interp_results)
    res_dict[R] = dict(interp_results)

with open('interp.json', 'w') as fout:
    json.dump(res_dict, fout, sort_keys=True, indent=4)
