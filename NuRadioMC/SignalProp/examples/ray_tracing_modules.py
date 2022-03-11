from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig()

logger = logging.getLogger('ray_tracing_modules')
solution_types = propagation.solution_types
ray_tracing_modules = propagation.available_modules
plot_line_styles = ["-", "--", ":", "-."]

### This example shows the ray tracing results for the different 
### ray tracing modules available in NuRadioMC
ref_index_model = 'greenland_simple'
ice = medium.get_ice_model(ref_index_model)

# Let us work on the y = 0 plane
initial_point = np.array( [70, 0, -300] ) * units.m
final_point = np.array( [100, 0, -30] ) * units.m
attenuation_model = 'GL1'

fig, axs = plt.subplots(1, 2, figsize=(12,6))
for i_module, ray_tracing_module in enumerate(ray_tracing_modules):
    print('Testing ray tracing module: \'{}\''.format(ray_tracing_module))
    try:
        prop = propagation.get_propagation_module(ray_tracing_module)
    except ImportError as e:
        logger.warning("Failed to load {}".format(ray_tracing_module))
        logger.exception(e)

    # This function creates a ray tracing instance refracted index, attenuation model, 
    # number of frequencies # used for integrating the attenuation and interpolate afterwards, 
    # and the number of allowed reflections.
    rays = prop(ice, attenuation_model,
                n_frequencies_integration=25,
                n_reflections=0)

    rays.set_start_and_end_point(initial_point,final_point)
    rays.find_solutions()

    for i_solution in range(rays.get_number_of_solutions()):

        solution_int = rays.get_solution_type(i_solution)
        solution_type = solution_types[solution_int]

        path = rays.get_path(i_solution)
        # We can calculate the azimuthal angle phi to rotate the
        # 3D path into the 2D plane of the points. This is only 
        # necessary if we are not working in the y=0 plane
        launch_vector = rays.get_launch_vector(i_solution)
        phi = np.arctan(launch_vector[1]/launch_vector[0])
        axs[0].plot(
            path[:,0]/np.cos(phi), path[:,2], 
            label=ray_tracing_module+"; " + solution_type,
            ls = plot_line_styles[i_module]
        )

        # We can also get the 3D receiving vector at the observer position, for instance
        receive_vector = rays.get_receive_vector(i_solution)
        # Or the path length
        path_length = rays.get_path_length(i_solution)
        # And the travel time
        travel_time = rays.get_travel_time(i_solution)

    # We can also calculate the attenuation for a set of frequencies
    if ray_tracing_module == 'direct_ray': # no attenuation for direct_ray ray tracer
        continue
    sampling_rate_detector = 1 * units.GHz
    nyquist_frequency = 0.5 * sampling_rate_detector
    frequencies = np.linspace(50 * units.MHz, nyquist_frequency, 100)

    for i_solution in range(rays.get_number_of_solutions()):

        solution_int = rays.get_solution_type(i_solution)
        solution_type = solution_types[solution_int]

        attenuation = rays.get_attenuation(i_solution, frequencies, nyquist_frequency)

        axs[1].plot(
            frequencies/units.MHz, attenuation, 
            label=ray_tracing_module + "; " + solution_type,
            ls=plot_line_styles[i_module]
        )

axs[0].set_xlabel('horizontal coordinate [m]')
axs[0].set_ylabel('vertical coordinate [m]')
axs[0].legend()

axs[1].set_xlabel('Frequency [MHz]')
axs[1].set_ylabel('Attenuation factor')
axs[1].set_ylim((0,1))
axs[1].legend()
plt.show()