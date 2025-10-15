"""
This is a simple temporary script to compare the C8 ray-tracer to the analytic NuRadioMC ray-tracer using a single-exponential ice model and 100 randomized points for a fixed detector position.
"""
import os
import numpy as np
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt

import time


n_datapoints = 100
n_output_parameters = int(2*4)

n_frequencies = 10
max_frequency = 1 * units.GHz
d_t = 0.5*1/max_frequency
frequencies = np.fft.rfftfreq(int(2*n_frequencies), d=d_t)

ice = medium.get_ice_model('southpole_2015')
#ice = medium.get_ice_model('uniform_ice')

from NuRadioMC.SignalProp.analyticraytracing import ray_tracing
propagator_1 = ray_tracing(ice,
                        n_reflections=1,
                        use_cpp=False)

from NuRadioMC.SignalProp.c8_ray_tracer import C8RayTracerIndividual
propagator_2 = C8RayTracerIndividual(ice,
                        n_reflections = 1,
                        min_step = 0.0001,
                        max_step = 1.0,
                        tolerance = 1e-8)

x_det = 0 * units.m
y_det = 0 * units.m
z_det = -100 * units.m

t0 = time.time()

# Generate raytracing solution:
X_array = np.zeros([n_datapoints, 3])
results_analytic = np.zeros([n_datapoints, n_output_parameters])
results_c8 = np.zeros([n_datapoints, n_output_parameters])

for i in range(n_datapoints):

    print("Progress:", i, "/", n_datapoints, " points ray-traced", end="\r")

    x0_vertex = np.random.random() * 2000 * units.m + 100 * units.m
    y0_vertex = 0
    z0_vertex = - 0.4 * np.sqrt(x0_vertex**2 + y0_vertex**2) - np.random.random() * 1000 * units.m # - np.random.random() * 1000 * units.m - 100 * units.m

    xyz_start = np.array([x0_vertex, y0_vertex, z0_vertex])
    X_array[i, :] = xyz_start
    xyz_stop = np.array([x_det, y_det, z_det])


    # propagator_1:
    propagator_1.set_start_and_end_point(xyz_start, xyz_stop)
    propagator_1.find_solutions()
    n_solutions = propagator_1.get_number_of_solutions()

    for i_solution in range(n_solutions):
        path_length = propagator_1.get_path_length(i_solution)
        travel_time = propagator_1.get_travel_time(i_solution)
        launch_vector = propagator_1.get_launch_vector(i_solution)
        launch_angle = np.arctan2(launch_vector[0], launch_vector[2])
        receive_vector = propagator_1.get_receive_vector(i_solution)
        receive_angle = np.arccos(receive_vector[2]/np.linalg.norm(receive_vector))
        # focusing_factor = propagator.get_focusing(i_solution) if not ray_tracing == "direct" else
        # attenuation = propagator.get_attenuation(i_solution, frequencies) if not ray_tracing == "direct" else 0

        if i_solution == 0:
            results_analytic[i, 0] = path_length
            results_analytic[i, 1] = travel_time
            results_analytic[i, 2] = launch_angle / units.deg
            results_analytic[i, 3] = receive_angle / units.deg
        elif i_solution == 1:
            results_analytic[i, 4] = path_length
            results_analytic[i, 5] = travel_time
            results_analytic[i, 6] = launch_angle / units.deg
            results_analytic[i, 7] = receive_angle / units.deg


    # propagator_2:
    propagator_2.set_start_and_end_point(xyz_start, xyz_stop)
    propagator_2.find_solutions()
    n_solutions = propagator_2.get_number_of_solutions()

    for i_solution in range(n_solutions):
        path_length = propagator_2.get_path_length(i_solution)
        travel_time = propagator_2.get_travel_time(i_solution)
        launch_vector = propagator_2.get_launch_vector(i_solution)
        launch_angle = np.arctan2(launch_vector[0], launch_vector[2])
        receive_vector = propagator_2.get_receive_vector(i_solution)
        receive_angle = np.arccos(receive_vector[2]/np.linalg.norm(receive_vector))
        
        if i_solution == 0:
            results_c8[i, 0] = path_length
            results_c8[i, 1] = travel_time
            results_c8[i, 2] = launch_angle / units.deg
            results_c8[i, 3] = receive_angle / units.deg
        elif i_solution == 1:
            results_c8[i, 4] = path_length
            results_c8[i, 5] = travel_time
            results_c8[i, 6] = launch_angle / units.deg
            results_c8[i, 7] = receive_angle / units.deg


t1 = time.time()


# plot results:
parameter_labels = ['path_length_1', 'travel_time_1', 'launch_angle_1', 'receive_angle_1', 'path_length_2', 'travel_time_2', 'launch_angle_2', 'receive_angle_2']
parameter_unit_labels = ['m', 'ns', 'deg', 'deg']

fig, ax = plt.subplots(2, 4, figsize=(15, 8), sharex=False, sharey=False)
ax = ax.flatten()
for i_param in range(n_output_parameters):
    ax[i_param].scatter(X_array[:, 0], results_analytic[:, i_param], color='C0', marker="o", alpha=0.8, s=10, label="Analytic")
    ax[i_param].scatter(X_array[:, 0], results_c8[:, i_param], color='C1', marker="x", alpha=0.8, s=10, label="C8")
    ax[i_param].set_title(parameter_labels[i_param])
    ax[i_param].legend()
    ax[i_param].set_xlim(0, np.max(X_array[:, 0])*1.1)
    ax[i_param].set_xlabel('Vertex position $x$ [m]')
    ax[i_param].set_ylabel(parameter_labels[i_param]+" ["+parameter_unit_labels[i_param%4]+"]")
fig.tight_layout()
plt.savefig('ray_tracing_results_x.png')

fig, ax = plt.subplots(2, 4, figsize=(15, 8), sharex=False, sharey=False)
ax = ax.flatten()
for i_param in range(n_output_parameters):
    ax[i_param].scatter(X_array[:, 2], results_analytic[:, i_param], color='C0', marker="o", alpha=0.8, s=10, label="Analytic")
    ax[i_param].scatter(X_array[:, 2], results_c8[:, i_param], color='C1', marker="x", alpha=0.8, s=10, label="C8")
    ax[i_param].set_title(parameter_labels[i_param])
    ax[i_param].legend()
    ax[i_param].set_xlim(np.min(X_array[:, 2])*1.1, 0)
    ax[i_param].set_xlabel('Vertex position $z$ [m]')
    ax[i_param].set_ylabel(parameter_labels[i_param]+" ["+parameter_unit_labels[i_param%4]+"]")
fig.tight_layout()
plt.savefig('ray_tracing_results_z.png')


fig, ax = plt.subplots(2, 4, figsize=(15, 8), sharex=False, sharey=False)
ax = ax.flatten()
for i_param in range(n_output_parameters):
    ax[i_param].scatter(X_array[:, 0], results_c8[:, i_param]-results_analytic[:, i_param], color='C3', marker="o", alpha=1, s=10, label=r"C8 $-$ Analytic")
    ax[i_param].axhline(0, color='k', ls='--', lw=1)
    ax[i_param].set_title(parameter_labels[i_param])
    ax[i_param].legend()
    ax[i_param].set_xlim(0, np.max(X_array[:, 0])*1.1)
    y_max = np.max(np.abs(results_c8[:, i_param]-results_analytic[:, i_param])) * 1.1
    ax[i_param].set_ylim(-y_max, y_max)
    ax[i_param].set_xlabel('Vertex position $x$ [m]')
    ax[i_param].set_ylabel("Delta "+parameter_labels[i_param]+" ["+parameter_unit_labels[i_param%4]+"]")
fig.tight_layout()
plt.savefig('ray_tracing_results_x_diff.png')

fig, ax = plt.subplots(2, 4, figsize=(15, 8), sharex=False, sharey=False)
ax = ax.flatten()
for i_param in range(n_output_parameters):
    ax[i_param].scatter(X_array[:, 2], results_c8[:, i_param]-results_analytic[:, i_param], color='C0', marker="o", alpha=1, s=10, label=r"C8 $-$ Analytic")
    ax[i_param].axhline(0, color='k', ls='--', lw=1)
    ax[i_param].set_title(parameter_labels[i_param])
    ax[i_param].legend()
    ax[i_param].set_xlim(np.min(X_array[:, 2])*1.1, 0)
    y_max = np.max(np.abs(results_c8[:, i_param]-results_analytic[:, i_param])) * 1.1
    ax[i_param].set_ylim(-y_max, y_max)
    ax[i_param].set_xlabel('Vertex position $z$ [m]')
    ax[i_param].set_ylabel("Delta "+parameter_labels[i_param]+" ["+parameter_unit_labels[i_param%4]+"]")
fig.tight_layout()
plt.savefig('ray_tracing_results_z_diff.png')


fig, ax = plt.subplots(2, 4, figsize=(15, 8), sharex=False, sharey=False)
ax = ax.flatten()
for i_param in range(n_output_parameters):
    diff = results_c8[:, i_param] - results_analytic[:, i_param]
    diff_max = np.max(np.abs(diff))
    sc = ax[i_param].scatter(X_array[:, 0], X_array[:, 2], c=diff, cmap="coolwarm", vmin=-diff_max, vmax=diff_max, marker="o", alpha=1, s=20, label=r"C8 $-$ Analytic")
    plt.sca(ax[i_param])
    plt.colorbar(sc, label="Delta "+parameter_labels[i_param]+" ["+parameter_unit_labels[i_param%4]+"]")
    ax[i_param].plot(x_det, z_det, ls="None", marker="*", color="k", markersize=10, label="Detector")
    ax[i_param].axhline(0, color='k', ls='--', lw=1)
    ax[i_param].axvline(0, color='k', ls='--', lw=1)
    ax[i_param].set_title(parameter_labels[i_param])
    ax[i_param].legend(loc='upper right')
    ax[i_param].set_xlim(-100, np.max(X_array[:, 0])*1.1)
    ax[i_param].set_ylim(np.min(X_array[:, 2])*1.1, 100)
    ax[i_param].set_xlabel('Vertex position $x$ [m]')
    ax[i_param].set_ylabel('Vertex position $z$ [m]')
fig.tight_layout()
plt.savefig('ray_tracing_results_2D.png')

print("Raytracing took", t1-t0, "s")