Ice and attenuation models
================================

Ice model implementation
------------------------------
Ice models are the object in NuRadioMC that holds all the information of the ice needed to calculate the trajectory, namely: the refractive index at all the relevant points in space, boundary conditions and other special features that determine the ice medium. It can be found in the **utilities** module under ``medium.py`` and ``medium_base.py``. 

The IceModel and IceModel_Simple class
_______________________________________
``medium_base.py`` holds the framework of which all the specific ice models depends. The most basic class of ice models is the ``IceModel`` from which all final models should inherit directly or via some daughter classes. It represents a planar medium with a top and bottom boundary and in between a refractive index. A reflective bottom layer may be added as well.

    .. code-block:: Python

        class IceModel():
            def __init__(self,z_airBoundary=0*units.meter,z_bottom=None):
                self.z_airBoundary = z_airBoundary
                self.z_bottom = z_bottom

            def add_reflective_bottom(self,refl_z,refl_coef,refl_phase_shift):
                self.reflection = refl_z
                self.reflection_coefficient = refl_coef
                self.reflection_phase_shift = refl_phase_shift

            def get_index_of_refraction(self,position):
                return index_of_refraction_at_position

            def get_average_index_of_refraction(self,position1,position2):
                return average_index_of_refraction_between_position1_and_position2

            def get_gradient_of_index_of_refraction(self, position):
                return gradient_index_of_refraction_at_position

Besides this basic class, ``medium_base`` also holds already a daughter class ``IceModel_Simple``. This class can be use to build a so called simple ice model which is a model with a planar geometry and a refractive index with an exponential profile depending on the depth in the ice. Mathematically this translates in:

    .. math::

        n(z) = n_{ice} - \Delta_n \exp(z/z_0)

where z is the depth and :math:`n_{ice}`, :math:`\Delta_n`, `math:`z_0` are the parameters of the model. As an example of how to implement a specific simple ice model into ``medium.py`` is illustrated below with the help of the ``greenland_simple`` ice model.

    .. code-block:: Python

        class greenland_simple(IceModel_Simple):
            def __init__(self):
                # from C. Deaconu, fit to data from Hawley '08, Alley '88
                # rho(z) = 917 - 602 * exp (-z/37.25), using n = 1 + 0.78 rho(z)/rho_0
                super().__init__(
                    z_bottom = -3000*units.meter, 
                    n_ice = 1.78, 
                    z_0 = 37.25*units.meter, 
                    delta_n = 0.51)

RadioPropaIceWrapper
_____________________
The analytic ray tracer can only handle simple ice models. For other models you need the RadioPropa ray tracer. However, for this to work, the model needs a translation into proper RadioPropa object. To facilitate this, the ``RadioPropaIceWrapper`` class is defined in ``medium_base.py``. This class hold the index of refraction as a RadioPropa scalar field and the boundary conditions and special features in the relevant RadioPropa modules.
    
    .. code-block:: Python

        class RadioPropaIceWrapper():
            def __init__(self, ice_model_nuradio, scalar_field):
                self.__ice_model_nuradio = ice_model_nuradio
                self.__scalar_field = scalar_field
                # these are predined modules that are inherent to the ice model like
                # discontinuities in the refractive index, reflective or transmissive
                # layers, observers to confine the model in a certain space ...
                self.__modules = {}

                """
                here stands some code handling the first modules like air boundary and bottom boundary etc.
                """    
                 
            def get_modules(self):
                return self.__modules

            def get_module(self,name):
                return self.__modules[name]

            def add_module(self,name,module):
                self.__modules[name]=module

            def remove_module(self,name):
                self.__modules.pop(name)

            def replace_module(self,name,new_module):
                self.__modules[name] = new_module

            def get_scalar_field(self):
                return self.__scalar_field

The most important point is that the index of refraction has to be translated in a RadioPropa scalar field. For simple ice models all this is is handled automatically but for other models one needs to implement specific scalar field of the ice models in RadioPropa (``IceModel.h`` and ``IceModel.cpp``). To access the RadioPropaIceWrapper object from the ice model, an extra function is implemented in the ``IceModel`` that is inherited by all the daughter classes but should be adapted to the specific implemented ice models. For the ``IceModel_Simpe`` class this is already implemented and this is handled automatically when defining a new simple ice model.
    
    .. code-block:: Python

        import radiopropa as RP

        class IceModel_Simple():
            ...

            def get_ice_model_radiopropa(self):
                scalar_field = RP.IceModel_Simple(z_surface=self.z_airBoundary*RP.meter/units.meter, 
                                                 n_ice=self.n_ice, delta_n=self.delta_n, 
                                                 z_0=self.z_0*RP.meter/units.meter,
                                                 z_shift=self.z_shift*RP.meter/units.meter)
                return RadioPropaIceWrapper(self,scalar_field)

An example of the implementation of a non-simple model if given by ``greenland_firn`` in ``medium.py``. This model completely depends on an implementation through RadioPropa because it can only be used with RadioPropa.


Available models in NuRadioMC
---------------------------------

Simple ice models
____________________
In the table below we can find the different parameters for the simple ice refractive index models available in NuRadioMC. 

    .. csv-table:: Simple Ice Models
        :header: "Name", ":math:`n_{ice}`", ":math:`\Delta_n`", ":math:`z_0$ [m]`"

        `southpole_simple <https://iopscience.iop.org/article/10.1088/1475-7516/2018/07/055>`__ (RICE2014/SP), 1.78, 0.425, 71
        `southpole_2015 <https://iopscience.iop.org/article/10.1088/1475-7516/2018/07/055>`__ (SPICE2015/SP), 1.78, 0.423, 77
        `ARAsim_southpole <https://iopscience.iop.org/article/10.1088/1475-7516/2018/07/055>`__ (as implemented in AraSim), 1.78, 0.43, 75.75
        `mooresbay_simple <https://iopscience.iop.org/article/10.1088/1475-7516/2018/07/055>`__ (MB1), 1.78, 0.46, 34.5
        `mooresbay_simple_2 <https://iopscience.iop.org/article/10.1088/1475-7516/2018/07/055>`__ (MB2), 1.78, 0.481, 37
        `greenland_simple <https://www.cambridge.org/core/journals/journal-of-glaciology/article/rapid-techniques-for-determining-annual-accumulation-applied-at-summit-greenland/96F86ED8AC87EB6B578E5021229CB37B>`__, 1.78, 0.51, 37.25

The models ``mooresbay_simple`` and ``mooresbay_simple_2`` also contain a reflective layer at -576 m with a reflection coefficient of 0.82, mimicking the bottom layer of Ross Ice Shelf, in Antarctica. 


RadioPropa ice models
______________________
Besides the simple ice models above, there is also one other ice model implemented: `greenland_firn <https://arxiv.org/abs/1805.12576>`__

Attenuation model
___________________
NuRadioMC has also three attenuation models available. These models provide attenuation lengths that are depth- and frequency-dependent.

  * `GL1 <https://www.cambridge.org/core/journals/journal-of-glaciology/article/an-in-situ-measurement-of-the-radiofrequency-attenuation-in-ice-at-summit-station-greenland/69FBB917D29DD43EE4DCDCC3EC21EA9F>`__, for Greenland.
  * `MB1 <https://www.cambridge.org/core/journals/journal-of-glaciology/article/radar-absorption-basal-reflection-thickness-and-polarization-measurements-from-the-ross-ice-shelf-antarctica/28AFEB95A33A6FF5CAF613D533355129>`__, for Moore's Bay.
  * `SP1 <https://icecube.wisc.edu/~araproject/radio/\#icetabsorption>`__, for South Pole.


Using specific models
_______________________
Both the ice model and the attenuation model can be specified in the config file. As an example, if we want to use the ``greenland_simple`` ice model together with the GL1 attenuation, we have to write on the yaml configuration file:

    .. code-block:: yaml

        propagation:
            ice_model: greenland_simple
            attenuation_model: GL1

Example script
-------------------
The following snippet shows how the ice properties can be retrieved from NuRadioMC for an independent analysis.

    .. code-block:: Python

        from NuRadioMC.utilities import medium, attenuation
        from NuRadioReco.utilities import units

        # Retrieving refractive index at a point
        ref_index_model = 'greenland_simple'

        ref_index_medium = medium.get_ice_model(ref_index_model)
        z_coordinate = -100 * units.m
        antenna_position = [0, 0, z_coordinate]
        index_at_antenna = ref_index_medium.get_index_of_refraction(antenna_position)

        # Getting the attenuation length
        attenuation_model = 'GL1'
        frequency = 200 * units.MHz
        depth = -100 * units.m

        attenuation_length = attenuation.get_attenuation_length(depth, frequency, attenuation_model)

Birefringence Ice Models
----------------------------

Birefringence is an optional propagation setting in NuRadioMC which allows to simulate radio pulses propagating in anisotropic ice. The details about how the calculations in the propagation work can be found here `(Heyer & Glaser, 2023) <https://link.springer.com/article/10.1140/epjc/s10052-023-11238-y>`__. When using birefringence several options exist about what birefringence-ice-model to propagate in and what propagation code should be used for the propagation. 

There are several example scripts available demonstrating all available (``NuRadioMC/SignalProp/examples/birefringence_examples``) functions when dealing with birefringent ice. Check read_me.txt for a more detailed description of the examples and data used.

.. warning:: Using this code assumes that the ice flow points in the positive x-direction. Therefore, a rotation of the detector geometry into this coordinate system might be necessary.

Available Birefringence Ice Models
_____________________________________

The anisotropy of the ice at the South Pole was published here: `(Jordan et al., 2020) <https://www.cambridge.org/core/journals/annals-of-glaciology/article/modeling-ice-birefringence-and-oblique-radio-wave-propagation-for-neutrino-detection-at-the-south-pole/52A9412B1D502F453C3E1C497BA9FE39>`__ 

The anisotropy of the ice in Greenland was published here: `(RNO-G, 2022) <https://arxiv.org/abs/2212.10285>`__ 

To use these ice models in NuRadioMC the measurement data was interpolated using splines. As the measurements don't extend from the ice surface to bedrock or to account for measurement uncertainties, there is some freedom in how to interpolate the data. Different interpolations are indexed via capital letters, ``A`` always denoting the most reasonable interpolation. The files ``NuRadioMC/utilities/birefringence_models/IceModel_interpolation_southpole.py`` and ``NuRadioMC/utilities/birefringence_models/IceModel_interpolation_greenland.py`` can be used to adjust the interpolation method and come up with new ice models.

    .. csv-table:: South Pole Birefringence Ice Models
        :header: "Name", "description"

        southpole_A, assumes a constant index of refraction at shallow and deep depths
        southpole_B, assumes a converging index of refraction at shallow depths
        southpole_C, no birefringence as nx = ny = nz
        southpole_D, assumes a constant average over all depths
        southpole_E, assumes ny and nz to be the same value at the average of the two


    .. csv-table:: Greenland Birefringence Ice Models
        :header: "Name", "description"

        greenland_A, the most reasonable interpolation
        greenland_B, assumes ny and nx to be the same value at the average of the two
        greenland_C, assumes ny and nx to diverge more than the data indicates


Available Birefringence Propagation Options 
______________________________________________

There is an option to use RadioPropa `(birefringence branch) <https://github.com/nu-radio/RadioPropa/tree/birefrigence>`__  to speed up the pulse propagation. If both the ray tracing and the birefringence pulse propagation should be handled by the analytical ray tracer, use ``analytical`` in the config file. If the ray tracing should be handled by the analytical ray tracer but the birefringence pulse propagation by RadioPropa, use ``numerical`` in the config file. There is also the option to handle everything in RadioPropa.

Currently, the RadioPropa implementation of birefringence only supports the birefringence-ice-model ``southpole_A``.

Using specific models 
_______________________
Birefringence is an optional setting in a NuRadioMC simulation. To use it in a simulation the following lines should be added to the config file:

    .. code-block:: yaml

        propagation:
            birefringence: True  
            birefringence_propagation: 'analytical'  
            birefringence_model: 'southpole_A'
