"""

This file is meant to give a guide for creating spline tables that can be used for the birefringence propagation functions.
The script is not used during the propagation.

"""

import numpy as np
import csv
from scipy import constants
from scipy import optimize
from scipy import interpolate
#import scipy.optimize
import matplotlib.pyplot as plt
import time
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from NuRadioMC.utilities.medium import southpole_2015
from NuRadioReco.utilities import units
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('raytracing')

model = 'X'

c = constants.c
e_p = 3.157                 # given in Jordan
e_d = 0.034                 # given in Jordan

m = southpole_2015()
comp = m.get_index_of_refraction(np.array([0, 0, -2500]))


def get_index_of_refraction(z):

    n1 = m.get_index_of_refraction(z) + f1_rec(-z[2]) - comp
    n2 = m.get_index_of_refraction(z) + f2_rec(-z[2]) - comp
    n3 = m.get_index_of_refraction(z) + f3_rec(-z[2]) - comp

    return n1, n2, n3


def get_index_of_refraction_all(z):

    """
    Returns the three indices of refraction for every meter down to the specified depth.

    Parameters
    ---------
    z:    numpy.array (int as entries), position of the interaction (only the z-coordinate matters)

    n:    numpy.array, every entry is a list with the three indices of refrection at the specific depth
    """

    n = []
    znew = np.arange(z[2], 0, 1)
    pos = np.zeros((len(znew), 3))
    pos[:, 2] = znew

    for i in znew:

        n.append(get_index_of_refraction(pos[i]))

    return(np.array(n), znew)




raw_ice_southpole = np.load('raw_ice_greenland.npy')

dep1 = raw_ice_southpole[0]
dep2 = raw_ice_southpole[1]
dep3 = raw_ice_southpole[2]
e1 = raw_ice_southpole[3]
e2 = raw_ice_southpole[4]
e3 = raw_ice_southpole[5]

n1 = np.sqrt(e_p + e_d * e1)
n2 = np.sqrt(e_p + e_d * e2)
n3 = np.sqrt(e_p + e_d * e3)

long = 1000
low = 120
high = 2910

filler_n1 = np.full((1, long), n1[-1])
filler_n2 = np.full((1, long), n2[-1])
filler_n3 = np.full((1, long), n3[-1])
filler_d = np.linspace(high, 3000, long)

n1 = np.concatenate((n1, filler_n1), axis=None)
n2 = np.concatenate((n2, filler_n2), axis=None)
n3 = np.concatenate((n3, filler_n3), axis=None)

depth1 = np.concatenate((dep1, filler_d), axis=None)
depth2 = np.concatenate((dep2, filler_d), axis=None)
depth3 = np.concatenate((dep3, filler_d), axis=None)


filler_n1 = np.full((1, long), n1[0])
filler_n2 = np.full((1, long), n2[0])
filler_n3 = np.full((1, long), n3[0])
filler_d = np.linspace(0, low, long)

n1 = np.concatenate((filler_n1, n1), axis=None)
n2 = np.concatenate((filler_n2, n2), axis=None)
n3 = np.concatenate((filler_n3, n3), axis=None)


depth1 = np.concatenate((filler_d, depth1), axis=None)
depth2 = np.concatenate((filler_d, depth2), axis=None)
depth3 = np.concatenate((filler_d, depth3), axis=None)


xnew = np.arange(0, 3000, 1)

# wrinkly interpolation
node_cond1 = 0.0000007
node_cond2 = 0.0000015
node_cond3 = 0.000001


# smooth interpolation

node_cond1 = 0.000005 
#node_cond1 = 0.00001 #model B
node_cond2 = 0.00001 
node_cond3 = 0.000015

#used for model B
#n_shared = (n1 + n2) / 2
#f1 = interpolate.UnivariateSpline(depth1, n_shared, s=node_cond1)
#f2 = interpolate.UnivariateSpline(depth2, n_shared, s=node_cond2)

#used for model C
n1_new = - n1 * 1/5000 + n1
n2_new = n2 * 1/5000 + n2
#f1 = interpolate.UnivariateSpline(depth1, n1_new, s=node_cond1) #model C
#f2 = interpolate.UnivariateSpline(depth2, n2_new, s=node_cond2) #model C

f1 = interpolate.UnivariateSpline(depth1, n1, s=node_cond1)
f2 = interpolate.UnivariateSpline(depth2, n2, s=node_cond2)
f3 = interpolate.UnivariateSpline(depth3, n3, s=node_cond3)

tck1 = f1._eval_args
f1_rec = interpolate.UnivariateSpline._from_tck(tck1)

tck2 = f2._eval_args
f2_rec = interpolate.UnivariateSpline._from_tck(tck2)

tck3 = f3._eval_args
f3_rec = interpolate.UnivariateSpline._from_tck(tck3)

interpolation = np.array([tck1, tck2, tck3])


# ------------- save the new IceModel
if 0:
    np.save('birefringence_greenland_' + model + '.npy', interpolation)


# --------------Jordan interpolation

if 1:
    n1 = n1[(depth1 < high) & (depth1 > low)]
    n2 = n2[(depth2 < high) & (depth2 > low)]
    n3 = n3[(depth3 < high) & (depth3 > low)]
    depth1 = depth1[(depth1 < high) & (depth1 > low)]
    depth2 = depth2[(depth2 < high) & (depth2 > low)]
    depth3 = depth3[(depth3 < high) & (depth3 > low)]


    plt.plot(-depth1, n1, 'b.', label='nx - data')
    plt.plot(-xnew, f1(xnew), label='nx - interpolation')

    plt.plot(-depth2, n2, 'r.', label='ny - data')
    plt.plot(-xnew, f2(xnew), label='ny - interpolation')

    plt.plot(-depth3, n3, 'g.', label='nz - data')
    plt.plot(-xnew, f3(xnew), label='nz - interpolation')

    plt.xlabel('depth [m]')
    plt.ylabel('refractive index')
    plt.legend()
    # plt.xlim(-1750, -140)
    # plt.ylim(1.778, 1.782)
    plt.grid()
    plt.show()

# -------------Jordan + Greenland
if 0:
    p = [0, 0, -2500]

    i_all = get_index_of_refraction_all(p)

    plt.plot(i_all[1], i_all[0][:, 0], label='nx - model')
    plt.plot(i_all[1], i_all[0][:, 1], label='ny - model')
    plt.plot(i_all[1], i_all[0][:, 2], label='nz - model')

    plt.title('refractive index for birefringence')
    plt.xlabel('depth [m]')
    plt.ylabel('refractive index')
    plt.legend()
    # plt.xlim(-1750, -140)
    # plt.ylim(1.778, 1.782)
    plt.grid()
    plt.show()
