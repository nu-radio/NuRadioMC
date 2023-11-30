import matplotlib.pyplot as plt
import numpy as np
import time
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
import NuRadioReco.framework.electric_field
import logging
from radiotools import helper as hp
from radiotools import plthelpers as php
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('raytracing')
# ray.cpp_available=False

x1 = np.array([0, 0., -149.]) * units.m

x2 = np.array([0, 0., 100]) * units.m
x3 = np.array([200, 0., 100]) * units.m
x4 = np.array([1000, 0., 100]) * units.m
x5 = np.array([10000, 0., 100]) * units.m

x_starts = [x1, x1, x1, x1, x5]
x_stops = [x2, x3, x4, x5, x1]
N = 5

receive_vectors = np.zeros((N, 2, 3)) * np.nan
ray_tracing_C0 = np.zeros((N, 2)) * np.nan
ray_tracing_C1 = np.zeros((N, 2)) * np.nan
ray_tracing_solution_type = np.zeros((N, 2), dtype=int) * np.nan
travel_times = np.zeros((N, 2)) * np.nan
travel_distances = np.zeros((N, 2)) * np.nan

ice = medium.southpole_simple()

if 0:  # for debug purpuses, plot the objective function
    fig2, ax2 = plt.subplots(1, 1)
    for i, (x_start, x_stop) in enumerate(zip(x_starts, x_stops)):
        r2d = ray.ray_tracing_2D(ice, log_level=logging.WARNING)
        logC0s = np.linspace(-0.9, 10, 10)
        oo = [r2d.obj_delta_y(t, x_start[np.array([0,2])], x_stop[np.array([0,2])]) for t in logC0s]
        ax2.plot(logC0s, oo, "-o")
        plt.show()

fig, ax = plt.subplots(1, 1)
for i, (x_start, x_stop) in enumerate(zip(x_starts, x_stops)):
    ax.plot(x_start[0], x_start[2], 'ko')
    print(f'finding solutions for {x_start} to {x_stop}')
    r = ray.ray_tracing(ice, log_level=logging.WARNING, use_cpp=False)
    r.set_start_and_end_point(x_start, x_stop)
    r.find_solutions()
    if(r.has_solution()):
        for iS in range(r.get_number_of_solutions()):
            ray_tracing_C0[i, iS] = r.get_results()[iS]['C0']
            ray_tracing_solution_type[i, iS] = r.get_solution_type(iS)
            print("     Solution %d, Type %d: " % (iS, ray_tracing_solution_type[i, iS]))
            R = r.get_path_length(iS)  # calculate path length
            R2 = r.get_path_length(iS, analytic=False)  # calculate path length
            T = r.get_travel_time(iS)  # calculate travel time
            T2 = r.get_travel_time(iS, analytic=False)  # calculate travel time
            print(f"     Ray Distance {R/units.m:.3f}m {R2/units.m:.3f}m and Travel Time {T/units.ns:.3f}ns {T2/units.ns:.3f}ns")
            receive_vector = r.get_receive_vector(iS)
            receive_vectors[i, iS] = receive_vector
            zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)
            print("     Receiving Zenith %.3f and Azimuth %.3f " % (zenith / units.deg, azimuth / units.deg))
            
            # get focussing factor
            focusing = r.get_focusing(0)
            print(f"     focusing factor = {focusing:.8f}")

            att = r.get_attenuation(iS, np.array([100, 200]) * units.MHz)
            print(f"     attenuation: {att}")
            
            
            efield=NuRadioReco.framework.electric_field.ElectricField([0])
            efield.set_trace(np.ones((3,200)), 1)
            efield2 = r.apply_propagation_effects(efield, 0)
            
            
            xx, zz = r.get_ray_path(iS)

            # # to readout the actual trace, we have to flatten to 2D
            # dX = x - x_start
            # dPhi = -np.arctan2(dX[1], dX[0])
            # c, s = np.cos(dPhi), np.sin(dPhi)
            # R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            # X1r = x_start
            # X2r = np.dot(R, x - x_start) + x_start
            # x1_2d = np.array([X1r[0], X1r[2]])
            # x2_2d = np.array([X2r[0], X2r[2]])
            # r_2d = ray.ray_tracing_2D(ice)
            # yy, zz = r_2d.get_path(x1_2d, x2_2d, ray_tracing_C0[i, iS])
            ax.plot(xx, zz, '{}'.format(php.get_color_linestyle(i)), label='{} C0 = {:.4f}, f = {:.2f}'.format(ray_tracing_solution_type[i, iS], ray_tracing_C0[i, iS], focusing))
            ax.plot(x_stop[0], x_stop[2], '{}{}-'.format('d', php.get_color(i)))

ax.legend()
ax.set_xlabel("y [m]")
ax.set_ylabel("z [m]")
fig.tight_layout()
fig.savefig('example_to_air.png')
plt.show()
