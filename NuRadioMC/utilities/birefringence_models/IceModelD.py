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
from sklearn.feature_selection.tests.test_from_model import test_allow_nan_tag_comes_from_estimator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('raytracing')


model = 'D'


c = constants.c
e_p = 3.157                 #given in Jordan
e_d = 0.034                 #given in Jordan

m = southpole_2015()  
comp = m.get_index_of_refraction(np.array([0, 0, -2500]))



def get_index_of_refraction(z):
    

    
    n1 = m.get_index_of_refraction(z) + interpolation[0] - comp
    n2 = m.get_index_of_refraction(z) + interpolation[1] - comp
    n3 = m.get_index_of_refraction(z) + interpolation[2] - comp 
            
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
    
    #print(znew)
    
    pos = np.zeros((len(znew), 3))       
    pos[:,2] = znew
    
    #print(pos)
                         
    for i in znew:

        n.append(get_index_of_refraction(pos[i]))
    
    return(np.array(n), znew)



depthL = []
e1 = []
e2 = []
e3 = []

with open("eps.csv", 'r') as inputfile:
    for i in range(84):
        line = inputfile.readline()
        if i > 1:
            line_s = line.split(',')
            depthL.append(float(line_s[0]))
            e3.append(float(line_s[3]))
            e2.append(float(line_s[4]))
            e1.append(float(line_s[5])) 


dep = np.array(depthL) 
e1 = np.array(e1)
e2 = np.array(e2)
e3 = np.array(e3)

n1 = np.sqrt(e_p + e_d * e1)
n2 = np.sqrt(e_p + e_d * e2)
n3 = np.sqrt(e_p + e_d * e3)
depth = dep


print(np.mean(n1))
print(np.mean(n2))
print(np.mean(n3))

interpolation = np.array([np.mean(n1), np.mean(n2), np.mean(n3)])

if 0:
    np.save('index_model' + model + '.npy', interpolation)


#--------------Jordan interpolation

if 1:

    
    plt.plot(-depth, n1, 'b.', label = 'nx - data')
    
    plt.plot(-depth, n2, 'r.', label = 'ny - data')
    
    plt.plot(-depth, n3, 'g.', label = 'nz - data')
    
    plt.hlines(np.mean(n1), -depth[0], -depth[-1], 'b', label = 'nx - average')
    plt.hlines(np.mean(n2), -depth[0], -depth[-1], 'r', label = 'ny - average')
    plt.hlines(np.mean(n3), -depth[0], -depth[-1], 'g', label = 'nz - average')
    
    plt.title('Principle refractive index at SPICE')
    plt.xlabel('depth [m]')
    plt.ylabel('refractive index')
    plt.legend()
    #plt.xlim(-1750, -140)
    #plt.ylim(1.778, 1.782)
    plt.grid()
    plt.show()



#-------------Jordan + Southpole
if 0:
    p = [0, 0, -2500]
    
    i_all = get_index_of_refraction_all(p)
    
    
    
    plt.plot(i_all[1], i_all[0][:,0], label = 'nx - model')
    plt.plot(i_all[1], i_all[0][:,1], label = 'ny - model')
    plt.plot(i_all[1], i_all[0][:,2], label = 'nz - model')
    
    plt.title('refractive index for birefringence')
    plt.xlabel('depth [m]')
    plt.ylabel('refractive index')
    plt.legend()
    #plt.xlim(-1750, -140)
    #plt.ylim(1.778, 1.782)
    plt.grid()
    plt.show()
    
    







































