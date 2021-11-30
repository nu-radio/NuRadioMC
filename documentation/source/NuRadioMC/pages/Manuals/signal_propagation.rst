Signal Propagation
===================
Propagation module
------------------
The modules for the raytracing are stored in the folder **SignalProp**. All the propagation effects (attenuation, focussing) are also taken account for in the raytracer module itself. 
The configuration of the raytracing and propagation effects are specified in the ``config.yaml`` file under ``propagation`` with the following attributes:
* module: [string] the ray tracing method to use
* ice_model: [string] the description of the refractive index of the ice and all its special effects
* attenuation_model: [string] the description of the attenuation of the ice
* attenuation_ice: [boolean] whether the attenuation due to the propagation through the ice should be applied. Note: The 1/R amplitude scaling will be applied in either case.
* n_freq: [int] the number of frequencies where the attenuation length is calculated for. The remaining frequencies will be determined from a linear interpolation between the reference frequencies. The reference frequencies are equally spaced over the complete frequency range.
* focusing: [boolean] whether the focusing effect should be applied.
* focusing_limit: [float] the maximum amplification factor of the focusing correction
* n_reflections: [int] the maximum number of reflections off a reflective layer at the bottom of the ice layer

Below you find the **default settings** of the **config file**.
  
  .. code-block:: yaml

    propagation:
      module: analytic
      ice_model: southpole_2015
      attenuation_model: SP1
      attenuate_ice: True 
      n_freq: 25
      focusing: False
      focusing_limit: 2
      n_reflections: 0

How to implement new ice-models and information on all the available ice-models and attenuation models can be found in the documentation. 

Ray tracing
-----------
Ray tracing is the module to calculate the trajectory of the emitted radiaton. Depending on the ice model one wants to use, the user can specify which ray tracer method should be used by NuRadioMC. This can be done in the ``config.yaml`` file by setting the propagation module to the desired module name.
  
  .. code-block:: yaml

    propagation:
      module: [module name]

One has to keep in mind that it is possible some methods only work using specific classes of ice models.

Analytical ray tracer
_____________________
The analytical ray tracer of NuRadioMC is a method that can only handle the so called *simple* ice models. Simple ice models are media with a planar geometry and a refractive index with an exponential profile depending on the depth in the ice. The module name to be used in the config file for this method is ``analytic``. The analytic method is implemented in both python but also in c++ for more rapid solving which will automatically be used when all dependencies are available on the users machine. Below you'll find an overview of the mathematics and an example of the possible rays. The details of this method can be found in the `NuRadioMC paper <http://dx.doi.org/10.1140/epjc/s10052-020-7612-8>`__.

Take the following ice model:

  .. math::

    n(z) = n_{ice} - \Delta_n \exp(\frac{z}{z_0}) [Eq. (1)]

where z is the depth and :math:`n_{ice}`, :math:`\Delta_n`, :math:`z_0` are the parameters of the model. The ray trajectories in a planar medium
with an exponential refractive index can be calculated analytically using the optical variational principle - the time it takes for the ray to go from emitter to observer must be a stationary point. Not necessarily a minimum, as it is commonly said. Take, for instance, a ray reflected on a spherical mirror.

According to the variational principle, the ray path in a medium given by Eq. (1) can be written as:

  .. math::

    y(z) = \pm z_0 (n_{ice}^2 C_0^2 - 1)^{1/2} \cdot \ln\left(\frac{\gamma}{2 (d(\gamma^2 - b\gamma + d)^{1/2} - b\gamma + 2d}\right) + C_1,

where y is the horizontal coordinate, :math:`\gamma = \Delta_n \exp(z/z_0)`, :math:`b = 2n_{ice}`, and :math:`d = n_{ice}^2 - C_0^{-2}`. :math:`C_0` is an integration constant related to the angle at launch position and :math:`C_1`  is another integration constant that gives the starting point.

The ray path can be expressed in closed form, as well as the travel time and the path length. However, the calculation of the frequency-dependent attenuation length must be done numerically. To that effect, a C++ version of the code has been implemented, which is called from Python. If the user doesn't have this C++ extension compiled, the code tries to compile it itself, for which the user must have specified the GSLDIR variable, and this must be pointing to the GNU Scientific Library directory.

  .. code-block:: bash

    export GSLDIR=/path/to/my/GNU_Scientific_Library

Once GSLDIR is configured, the user can also compile it by hand executing the following instruction in the  SignalProp/CPPAnalyticRayTracing folder:

  .. code-block:: bash

    python setup.py build_ext --inplace


RadioPropa numerical ray tracer (in development)
_________________________________________________
For ice models other then the simple ones, one need a numerical ray tracer which is provided by the RadioPropa method. This method uses the RadioPropa package which is written in c++. Information on the installation of RadioPropa can found on https://github.com/nu-radio/RadioPropa. The module name for this method is ``radiopropa``.

  .. code-block:: yaml

    propagation:
      module: radiopropa

RadioPropa is a modular ray tracing code that solves the eikonal equation for a ray fired at a certain place in a certain direction using a Runge-Kutta method in arbitrary refractivity fields. The implemented NuRadio ray tracer uses this to scan a certain section of the ice in a iterative manner to see whether a channel will be hit or not as shown below

For now, this method can be used for a refractive index with any profile depending on the depth (only z, no x or y dependence) in the ice and some additional features like discontinuities or reflective/transmissive layers. In the future, more effect and the handling of more complex profiles will become available.

Example scripts
---------------

How to calculate an analytic ray path
______________________________________
The following code shows how to perform a analytic ray tracing and extract information on the solutions, such as trajectory, travel time, or attenuation.

  .. code-block:: Python

    from NuRadioMC.SignalProp import propagation
    from NuRadioMC.SignalProp.analyticraytracing import solution_types, ray_tracing_2D
    from NuRadioMC.utilities import medium
    from NuRadioReco.utilities import units
    import matplotlib.pyplot as plt
    import numpy as np

    prop = propagation.get_propagation_module('analytic')

    ref_index_model = 'greenland_simple'
    ice = medium.get_ice_model(ref_index_model)

    # Let us work on the y = 0 plane
    initial_point = np.array( [70, 0, -300] ) * units.m
    final_point = np.array( [100, 0, -30] ) * units.m
    attenuation_model = 'GL1'

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

        # To plot the ray path, we can use the 2D ray tracing class, which works on
        # a plane. Since we have been working on the y = 0 plane, we can construct
        # the 2D vectors without translations or rotations. Just ignore the y component.
        rays_2D = ray_tracing_2D(ice, attenuation_model)
        initial_point_2D = np.array( [initial_point[0], initial_point[2]] )
        final_point_2D = np.array( [final_point[0], final_point[2]] )
        C_0 = rays.get_results()[i_solution]['C0']

        xx, zz = rays_2D.get_path(initial_point_2D, final_point_2D, C_0)
        plt.plot(xx, zz, label=solution_type)

        # We can also get the 3D receiving vector at the observer position, for instance
        receive_vector = rays.get_receive_vector(i_solution)
        # Or the path length
        path_length = rays.get_path_length(i_solution)
        # And the travel time
        travel_time = rays.get_travel_time(i_solution)

    plt.xlabel('horizontal coordinate [m]')
    plt.ylabel('vertical coordinate [m]')
    plt.legend()
    plt.show()

    # We can also calculate the attenuation for a set of frequencies

    sampling_rate_detector = 1 * units.GHz
    nyquist_frequency = 0.5 * sampling_rate_detector
    frequencies = np.linspace(50 * units.MHz, nyquist_frequency, 100)

    for i_solution in range(rays.get_number_of_solutions()):

        solution_int = rays.get_solution_type(i_solution)
        solution_type = solution_types[solution_int]
   
        attenuation = rays.get_attenuation(i_solution, frequencies, nyquist_frequency)

        plt.plot(frequencies/units.MHz, attenuation, label=solution_type)

    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Attenuation factor')
    plt.ylim((0,1))
    plt.legend()
    plt.show()

How to calculate an radiopropa ray path
_________________________________________
The following code shows how to perform a ray tracing and extract  information on the solutions, such as trajectory, travel time, or attenuation.

  .. code-block:: Python

    from NuRadioMC.SignalProp import propagation
    from NuRadioMC.SignalProp.simple_radiopropa_tracer import solution_types, ray_tracing
    from NuRadioMC.utilities import medium
    from NuRadioReco.utilities import units
    import matplotlib.pyplot as plt
    import numpy as np

    prop = propagation.get_propagation_module('radiopropa')

    ref_index_model = 'greenland_simple'
    ice = medium.get_ice_model(ref_index_model)

    # Let us work on the y = 0 plane
    initial_point = np.array( [70, 0, -300] ) * units.m
    final_point = np.array( [100, 0, -30] ) * units.m
    attenuation_model = 'GL1'

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
        launch_vector = rays.get_launch_vector(i_solution))
        phi = np.arctan(launch_vector[1]/launch_vector[0])
        plt.plot(path[:,0]/np.cos(phi), path[:,2], label=solution_type)

        # We can also get the 3D receiving vector at the observer position, for instance
        receive_vector = rays.get_receive_vector(i_solution)
        # Or the path length
        path_length = rays.get_path_length(i_solution)
        # And the travel time
        travel_time = rays.get_travel_time(i_solution)

    plt.xlabel('horizontal coordinate [m]')
    plt.ylabel('vertical coordinate [m]')
    plt.legend()
    plt.show()

    # We can also calculate the attenuation for a set of frequencies

    sampling_rate_detector = 1 * units.GHz
    nyquist_frequency = 0.5 * sampling_rate_detector
    frequencies = np.linspace(50 * units.MHz, nyquist_frequency, 100)

    for i_solution in range(rays.get_number_of_solutions()):

        solution_int = rays.get_solution_type(i_solution)
        solution_type = solution_types[solution_int]
    
        attenuation = rays.get_attenuation(i_solution, frequencies, nyquist_frequency)

        plt.plot(frequencies/units.MHz, attenuation, label=solution_type)

    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Attenuation factor')
    plt.ylim((0,1))
    plt.legend()
    plt.show()
