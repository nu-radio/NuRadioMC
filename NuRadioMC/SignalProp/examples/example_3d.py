import matplotlib.pyplot as plt
import numpy as np
import time
from NuRadioMC.SignalProp import analyticraytraycing as ray
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
import logging
from radiotools import helper as hp
from radiotools import plthelpers as php
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('raytracing')
# ray.cpp_available=False

x1 = np.array([478., 0., -149.])*units.m
x2 = np.array([635., 0., -5.])*units.m  # direct ray solution
x3 = np.array([1000., 0., -90.])*units.m  # refracted/reflected ray solution
x4 = np.array([700., 0., -149.])*units.m  # refracted/reflected ray solution
x5 = np.array([1000., 0., -5.])*units.m  # no solution

receive_vectors = np.zeros((4, 2, 3)) * np.nan
ray_tracing_C0 = np.zeros((4, 2)) * np.nan
ray_tracing_C1 = np.zeros((4, 2)) * np.nan
ray_tracing_solution_type = np.zeros((4, 2), dtype=np.int) * np.nan
travel_times = np.zeros((4, 2)) * np.nan
travel_distances = np.zeros((4, 2)) * np.nan

ice = medium.southpole_simple()

lss = ['-', '--', ':']
colors = ['b','g','r']
fig, ax = plt.subplots(1, 1)
ax.plot(x1[0], x1[2], 'ko')
for i, x in enumerate([x2, x3, x4, x5]):
    print('finding solutions for ', x)
    r = ray.ray_tracing(x1,x,ice)
    r.find_solutions()
    if(r.has_solution()):
        for iS in range(r.get_number_of_solutions()):
            ray_tracing_C0[i, iS] = r.get_results()[iS]['C0']
            ray_tracing_solution_type[i, iS] = r.get_solution_type(iS)
            print "     Solution %d, Type %d: "%(iS,ray_tracing_solution_type[i, iS])
            R = r.get_path_length(iS)  # calculate path length
            T = r.get_travel_time(iS)  # calculate travel time
            print "     Ray Distance %.3f and Travel Time %.3f"%(R/units.m,T/units.ns)
            receive_vector = r.get_receive_vector(iS)
            receive_vectors[i, iS] = receive_vector
            zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)
            print "     Receiving Zenith %.3f and Azimuth %.3f "%(zenith/units.deg, azimuth/units.deg)

            #to readout the actual trace, we have to flatten to 2D
            dX = x - x1
            dPhi = -np.arctan2(dX[1],dX[0])
            c,s = np.cos(dPhi), np.sin(dPhi)
            R = np.array(((c,-s,0),(s,c,0),(0,0,1)))
            X1r = x1
            X2r = np.dot(R, x-x1) + x1
            x1_2d = np.array([X1r[0],X1r[2]])
            x2_2d = np.array([X2r[0],X2r[2]])
            r_2d = ray.ray_tracing_2D(ice)
            yy,zz = r_2d.get_path(x1_2d,x2_2d,ray_tracing_C0[i, iS])
            ax.plot(yy, zz, '{}'.format(php.get_color_linestyle(i)), label='{} C0 = {:.4f}'.format(ray_tracing_solution_type[i,iS], ray_tracing_C0[i,iS]))
            ax.plot(x2_2d[0], x2_2d[1], '{}{}-'.format('d',php.get_color(i)))

ax.legend()
ax.set_xlabel("y [m]")
ax.set_ylabel("z [m]")
fig.tight_layout()
fig.savefig('example.png')
