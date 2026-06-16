"""
Module providing the ice models in NuRadioMC.

For more details on the implementation, and the available modules,
see the documentation :doc:`here </NuRadioMC/pages/Manuals/icemodels>`.

"""

from NuRadioMC.utilities import medium_base
import numpy as np
import os
from NuRadioReco.utilities import units

import functools

logger = medium_base.logger

try:
    import radiopropa as RP
    radiopropa_is_imported = True
except ImportError:
    radiopropa_is_imported = False

"""
1) When implementing a new model it should at least inherit from
'IceModel' from the module 'medium_base'. Overwrite all the function.
Inheritance from daughter classes like 'IceModelSimple' is also
possible and overwriting functions may not be needed in this case.

2) When implementing a new model and using the radiopropa numerical
tracer, do not forget to implement scalar field of the refractive index
also in the c++ code of radiopropa for a fast simulation. Implement the
model in IceModel.cpp and IceModel.h. Then edit the function to get the
radiopropa ice model, so it can be used in NuRadioMC. For example

        def get_ice_model_radiopropa(self):
            #args is a placeholder for the arguments needed by the specific module
            scalar field = radiopropa.New_IceModel(*args)
            return RadioPropaIceWrapper(self,scalar_field)

3) You can also choose to only implement the new ice model in radiopropa if
radiopropa is always necessary and make the new model in this script access
the c++ implemented model (e.g. green_firn model)


4) If you want to adjust (add, replace, remove) predefined modules
in the a RadioPropaIceWrapper object, you can do this by redefining the
'get_ice_model_radiopropa()' in your IceModel object. For exemple

        def get_ice_model_radiopropa(self):
            scalar field = radiopropa.IceModelSimple(*args)
            ice = RadioPropaIceWrapper(self,scalar_field)
            extra_dicontinuity = radiopropa.Discontinuity(*args)
            ice.add_module(extra_discontinuity)
            return ice
"""


class southpole_simple(medium_base.IceModelSimple):
    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 RICE2014/SP model
        # define model parameters (RICE 2014/southpole)
        super().__init__(
            z_bottom = -2820*units.meter,
            n_ice = 1.78,
            z_0 = 71.*units.meter,
            delta_n = 0.426,
            )


class southpole_2015(medium_base.IceModelSimple):
    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 SPICE2015/SP model
        super().__init__(
            z_bottom = -2820*units.meter,
            n_ice = 1.78,
            z_0 = 77.*units.meter,
            delta_n = 0.423,
            )


class ARAsim_southpole(medium_base.IceModelSimple):
    def __init__(self):
        # define model parameters (SPICE 2015/southpole)
        super().__init__(
            z_bottom = -2820*units.meter,
            n_ice = 1.78,
            z_0 = 75.75757575757576*units.meter,
            delta_n = 0.43,
            )



class ARA_2022(medium_base.IceModelSimple):
    def __init__(self):
        # define model parameters (ARA/southpole) -> https://journals.aps.org/prd/pdf/10.1103/PhysRevD.105.122006
        super().__init__(
            z_bottom = -2820*units.meter,
            n_ice = 1.78,
            z_0 = 49.5049505*units.meter,
            delta_n = 0.454,
            )


class birefringence_medium(medium_base.IceModelBirefringence):

    def __init__(self, bir_model='southpole_A'):
        f = self._load_binary_data(bir_model)
        self.load_birefringence_model(bir_model=f)

    @functools.lru_cache(maxsize=8)
    def _load_binary_data(self, bir_model):
        # from https://link.springer.com/article/10.1140/epjc/s10052-023-11238-y
        filepath = os.path.dirname(os.path.realpath(__file__)) + '/birefringence_models/birefringence_' + bir_model + '.npy'
        return np.load(filepath, allow_pickle=True)

class mooresbay_simple(medium_base.IceModelSimple):
    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 MB1 model
        super().__init__(
            n_ice = 1.78,
            z_0 = 34.5*units.meter,
            delta_n = 0.46,
            )

        # from https://doi.org/10.3189/2015JoG14J214
        self.add_reflective_bottom(
            refl_z = -576*units.m,
            refl_coef = 0.82,
            refl_phase_shift = 180*units.deg,
            )


class mooresbay_simple_2(medium_base.IceModelSimple):
    def __init__(self):\
        # from https://doi.org/10.1088/1475-7516/2018/07/055 MB2 model
        super().__init__(
            n_ice = 1.78,
            z_0 = 37*units.meter,
            delta_n = 0.481,
            )

        # from https://doi.org/10.3189/2015JoG14J214
        self.add_reflective_bottom(
            refl_z = -576*units.m,
            refl_coef = 0.82,
            refl_phase_shift = 180*units.deg,
            )


class greenland_simple(medium_base.IceModelSimple):
    def __init__(self):
        # from C. Deaconu, fit to data from Hawley '08, Alley '88
        # rho(z) = 917 - 602 * exp (-z/37.25), using n = 1 + 0.78 rho(z)/rho_0
        super().__init__(
            z_bottom = -3000*units.meter,
            n_ice = 1.78,
            z_0 = 37.25*units.meter,
            delta_n = 0.51,
            )

class greenland_simple_nils(medium_base.IceModelSimple):
    def __init__(self):
        # from C. Deaconu, fit to data from Hawley '08, Alley '88
        # rho(z) = 917 - 602 * exp (-z/37.25), using n = 1 + 0.78 rho(z)/rho_0
        super().__init__(
            z_bottom = -3000*units.meter,
            n_ice = 1.781,
            z_0 = 45.20*units.meter,
            delta_n = 0.485,
            )
                

class greenland_firn(medium_base.IceModel):
    """
    This model can only be used with the radiopropa raytracer.
    Therefor, the model is implemented through radiopropa.
    """
    def __init__(self):
        """
        initiation of a double exponential ice model at summit, Greenland

        The bottom defined here is a boundary condition used in simulations and
        should always be defined. Note: it is not the same as reflective bottom.
        The latter can be added using the `add_reflective_layer` function.

        The z_shift is a variable introduced to be able to shift the exponential
        up or down along the z direction. For simple models this is almost never
        but it is used to construct more complex ice models which rely on exp.
        profiles also

        Parameters
        ----------
        z_air_boundary: float, NuRadio length units
            z coordinate of the surface of the glacier
        z_bottom: float, NuRadio length units
            z coordinate of the bedrock/bottom of the glacier.
        z_firn: float, NuRadio length units
            z coordinate of the transition from the upper
            exponential profile to the lower one

        The following parameters can be found without (lower)
        and with (upper) the suffix of `_firn`

        n_ice: float, dimensionless
            refractive index of the deep bulk ice
        delta_n: float, NuRadio length units
            difference between n_ice and the refractive index
            of the snow at the surface
        z_0: float, NuRadio length units
            scale depth of the exponential
        z_shift: float, NuRadio length units
            up or down shift od the exponential profile
        """

        if not medium_base.radiopropa_is_imported:
            logger.error('This ice model depends fully on RadioPropa, which was not import, and can therefore not be used.'+
                         '\nMore info on https://github.com/nu-radio/RadioPropa')
            raise ImportError('This ice model depends fully on RadioPropa, which could not be imported')

        super().__init__(z_bottom = -3000*units.meter)
        self.z_firn = -14.9*units.meter

        self._scalarfield = RP.IceModel_Firn(
            z_surface = self.z_air_boundary*RP.meter/units.meter,
            z_firn = self.z_firn*RP.meter/units.meter,
            n_ice = 1.78,
            delta_n = 0.310,
            z_0 = 40.9*RP.meter,
            z_shift = -14.9*RP.meter,
            n_ice_firn = 1.78,
            delta_n_firn = 0.502,
            z_0_firn = 30.8*RP.meter,
            z_shift_firn = 0.*RP.meter,
            )

    def get_index_of_refraction(self, position):
        """
        returns the index of refraction at position.
        Overwrites function of the mother class

        Parameters
        ----------
        position: 3dim np.array
            point

        Returns
        -------
        n: float
            index of refraction
        """
        position = RP.Vector3d(*(position * RP.meter/units.meter))
        return self._scalarfield.getValue(position)

    def get_average_index_of_refraction(self, position1, position2):
        """
        returns the average index of refraction between two points
        Overwrites function of the mother class

        Parameters
        ----------
        position1: 3dim np.array
            point
        position2: 3dim np.array
            point

        Returns
        -------
        n_average: float
            averaged index of refraction between the two points
        """
        position1 = RP.Vector3d(*(position1 * RP.meter/units.meter))
        position2 = RP.Vector3d(*(position2 * RP.meter/units.meter))
        return self._scalarfield.getAverageValue(position1, position2)


    def get_gradient_of_index_of_refraction(self, position):
        """
        returns the gradient of index of refraction at position
        Overwrites function of the mother class

        Parameters
        ----------
        position: 3dim np.array
            point

        Returns
        -------
        n_nabla: np.array(3,)
            gradient of index of refraction at the point
        """
        pos = RP.Vector3d(*(position * RP.meter/units.meter))
        return self._scalarfield.getGradient(pos) * (1 / (units.meter/RP.meter))


    def _compute_default_ice_model_radiopropa(self):
        """
        Computes a default object holding the radiopropa scalarfield and necessary radiopropa
        moduldes that define the medium in radiopropa. It uses the parameters of the medium
        object to contruct the scalar field (using the firn ice model implementation
        in radiopropa) and some modules (like a discontinuity object for the air boundary).

        Overwrites function of the mother class

        Returns
        -------
        ice_model_radiopropa: RadioPropaIceWrapper
            object holding the radiopropa scalarfield and modules
        """
        return medium_base.RadioPropaIceWrapper(self, self._scalarfield)

class greenland_perturbation(greenland_firn):
    def __init__(self):
        greenland_firn.__init__(self)

    def _compute_default_ice_model_radiopropa(self, discontinuity=False):
        """
        Computes a default object holding the radiopropa scalarfield and necessary radiopropa
        moduldes that define the medium in radiopropa. It uses the parameters of the medium
        object to contruct some modules using the default computation of the firn model.
        An additional module for the perturbation layer is then added to the object.

        Overwrites function of the mother class

        Returns
        -------
        ice_model_radiopropa: RadioPropaIceWrapper
            object holding the radiopropa scalarfield and modules
        """
        ice = greenland_firn._compute_default_ice_model_radiopropa(self)
        #fraction from ArXiv 1805.12576 table IV last row
        perturbation_horz = RP.PerturbationHorizontal(-100*RP.meter,2*RP.meter, fraction=1)
        ice.add_module('horizontal perturbation',perturbation_horz)
        return ice

class greenland_poly5(medium_base.IceModelExponentialPolynomial):
    """
    Fifth-degree exponential polynomial model for Summit Station, Greenland by Oeyen B.
    https://doi.org/10.5281/zenodo.15067984
    """
    def __init__(self, density_factor=0.851 * (units.cm**3 / units.gram)):
        """
        initiation of the model based on the fitted coefficient

        The bottom defined here is a boundary condition used in simulations and
        should always be defined. Note: it is not the same as reflective bottom.
        The latter can be added using the `add_reflective_layer` function.
        """

        super().__init__(
            a=np.array([917, -62.2, 1177, -9051, 14360, -7024]) * (units.kg / units.m**3),
            z_0=74.6 * units.meter,
            density_factor=density_factor,
            z_bottom=-3000 * units.meter
        )


class uniform_ice(medium_base.IceModelSimple):
    """
    uniform ice with refractive index of typical deep ice (1.78)
    """
    def __init__(self, z_bottom=None):
        super().__init__(
            z_bottom = z_bottom,
            n_ice = 1.78,
            z_0 = 1*units.meter,
            delta_n = 0,
            )


def get_ice_model(name):
    """
    function to access the right ice model class by name of the class

    Parameters
    ----------
    name: string
        name of the class of the requested ice model

    Returns
    -------
    ice_model: IceModel object
        object of the class with the name of the requested model
    """
    if globals()[name]() == None:
        logger.error('The ice model you are trying to use is not implemented. Please choose another ice model or implement a new one.')
        raise NotImplementedError('The ice model you are trying to use is not implemented. Please choose another ice model or implement a new one.')
    else:
        return globals()[name]()


class greenland_simple_layered(medium_base.IceModelExpLayers):
    """
    Single layer refractive index model.
    
    greenland_simple model adapted to match the expected medium definition needed for the multi layer analytic raytracer. Used as a comparison to the single layer analytic raytracer.
    """
    def __init__(self):

        z_bottom = -3000*units.meter
        n_ice = 1.78
        z_0 = 37.25*units.meter
        delta_n = 0.51

        layers = [
            {
            "z_min": 0.0,
            "z_max": np.inf,
            "n_ice": 1.00001,
            "delta_n": 1e-6,
            "z_0": -8000,
            "region": "air",
            "region_name": "Air"
        },
            {
            "z_min": -3000.0,
            "z_max": 0.0,
            "n_ice": 1.78,
            "delta_n": 0.51,
            "z_0": 37.25,
            "region": "single",
            "region_name": "SingleModel"
        }]

        super().__init__(
            layers=layers
            #z_bottom=-3000.0,
        )


class greenland_simple_nils_layered(medium_base.IceModelExpLayers):
    """
    Single layer refractive index model.
    
    greenland_simple model adapted to match the expected medium definition needed for the multi layer analytic raytracer. Used as a comparison to the single layer analytic raytracer.
    """
    def __init__(self):

        z_bottom = -3000*units.meter
        n_ice = 1.78
        z_0 = 37.25*units.meter
        delta_n = 0.51

        layers = [
            {
            "z_min": 0.0,
            "z_max": np.inf,
            "n_ice": 1.00001,
            "delta_n": 1e-6,
            "z_0": -8000,
            "region": "air",
            "region_name": "Air"
        },
            {
            "z_min": -3000.0,
            "z_max": 0.0,
            "n_ice": 1.781,
            "delta_n": 0.485,
            "z_0": 45.20,
            "region": "single",
            "region_name": "SingleModel"
        }]

        super().__init__(
            layers=layers
            #z_bottom=-3000.0,
        )

class greenland_firn_layered(medium_base.IceModelExpLayers):
    """
    Two layer refractive index model.
     
    values taken from greenland_firn and adapted to match the expected medium definition needed for the multi layer analytic raytracer. Combination of firn layer (settling and freezing of snow in shallow ice) and bubbly ice.
    """
    def __init__(self):
        
        layers = [
            {
                "z_min": -14.9,
                "z_max": 0.0,
                "n_ice": 1.78,
                "delta_n": 0.502,
                "z_0": 30.8,
                "region": "firn",
                "region_name": "Firn"
            },
            {
                "z_min": -3000.0,
                "z_max": -14.9,
                "n_ice": 1.78,
                "delta_n": 0.446,
                "z_0": 40.9,
                "region": "ice",
                "region_name": "Ice"
            }
        ]

        super().__init__(
            layers=layers
            #z_bottom=-3000.0,
        )

class greenland_3exp_layered(medium_base.IceModelExpLayers):
    """
    Four layer refractive index model.
     
    Values for below the ice taken from https://github.com/philippwindischhofer/Reconal/blob/7204049c755a0678178821073fa73a476c49c491/defs.py#L72-L82. Combination of air layer above z=0.0, snow layer, firn layer (settling and freezing of snow in shallow ice) and bubbly ice.
    """
    def __init__(self):

        layers = [
            {
                "z_min": 0.0,
                "z_max": np.inf,
                "n_ice": 1.00027,
                "delta_n": 2.7e-4,
                "z_0": -8000.0,
                "region": "air",
                "region_name": "Air"
            },
            {
                "z_min": -14.9,
                "z_max": 0.0,
                "n_ice": 1.51188,
                "delta_n": 0.271579,
                "z_0": 1/0.114553,
                "region": "snow",
                "region_name": "Snow"
            },
            {
                "z_min": -80.5,
                "z_max": -14.9,
                "n_ice": 1.89957,
                "delta_n": 0.529715,
                "z_0": 1/0.0129175,
                "region": "firn",
                "region_name": "Firn"
            },
            {
                "z_min": -3000.0,
                "z_max": -80.5,
                "n_ice": 1.77468,
                "delta_n": 1.41573,
                "z_0": 1/0.0387882,
                "region": "bubbly_ice",
                "region_name": "Ice"
            }
        ]

        super().__init__(
            layers=layers
            #z_bottom=-3000.0,
        )


class greenland_3exp_nils_layered(medium_base.IceModelExpLayers):
    """
    Four layer refractive index model.
     
    Values for below the ice taken from https://github.com/philippwindischhofer/Reconal/blob/7204049c755a0678178821073fa73a476c49c491/defs.py#L72-L82. Combination of air layer above z=0.0, snow layer, firn layer (settling and freezing of snow in shallow ice) and bubbly ice.
    """
    def __init__(self):

        layers = [
            {
                "z_min": 0.0,
                "z_max": np.inf,
                "n_ice": 1.00027,
                "delta_n": 2.7e-4,
                "z_0": -8000.0,
                "region": "air",
                "region_name": "Air"
            },
            {
                "z_min": -14.9,
                "z_max": 0.0,
                "n_ice": 1.544,
                "delta_n": 0.272,
                "z_0": 15.88,
                "region": "snow",
                "region_name": "Snow"
            },
            {
                "z_min": -80.5,
                "z_max": -14.9,
                "n_ice": 1.855,
                "delta_n": 0.530255538,
                "z_0": 62.281809455,
                "region": "firn",
                "region_name": "Firn"
            },
            {
                "z_min": -3000.0,
                "z_max": -80.5,
                "n_ice": 1.778,
                "delta_n": 1.06592622966,
                "z_0": 29.343776516,
                "region": "bubbly_ice",
                "region_name": "Ice"
            }
        ]

        super().__init__(
            layers=layers
            #z_bottom=-3000.0,
        )

        
class southpole_simple_layered(medium_base.IceModelExpLayers):

    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055
        # RICE2014/SP model

        z_bottom = -2820 * units.meter
        n_ice = 1.78
        z_0 = 71. * units.meter
        delta_n = 0.426

        layers = [
            {
                "z_min": 0.0,
                "z_max": np.inf,
                "n_ice": 1.000001,
                "delta_n": 2.7e-6,
                "z_0": -8000.0,
                "region": "air",
                "region_name": "Air"
            },
            {
                "z_min": z_bottom,
                "z_max": 0.0,
                "n_ice": n_ice,
                "delta_n": delta_n,
                "z_0": z_0,
                "region": "ice",
                "region_name": "Ice"
            }
        ]

        super().__init__(layers=layers)


class southpole_2015_layered(medium_base.IceModelExpLayers):

    def __init__(self):

        # SPICE2015/SP model
        z_bottom = -2820 * units.meter
        n_ice = 1.78
        z_0 = 77. * units.meter
        delta_n = 0.423

        layers = [
            {
                "z_min": 0.0,
                "z_max": np.inf,
                "n_ice": 1.000001,
                "delta_n": 2.7e-6,
                "z_0": -8000.0,
                "region": "air",
                "region_name": "Air"
            },
            {
                "z_min": z_bottom,
                "z_max": 0.0,
                "n_ice": n_ice,
                "delta_n": delta_n,
                "z_0": z_0,
                "region": "ice",
                "region_name": "Ice"
            }
        ]

        super().__init__(layers=layers)

class ARAsim_southpole_layered(medium_base.IceModelExpLayers):

    def __init__(self):

        # SPICE 2015 / South Pole
        z_bottom = -2820 * units.meter
        n_ice = 1.78
        z_0 = 75.75757575757576 * units.meter
        delta_n = 0.43

        layers = [
            {
                "z_min": 0.0,
                "z_max": np.inf,
                "n_ice": 1.00027,
                "delta_n": 2.7e-4,
                "z_0": -8000.0,
                "region": "air",
                "region_name": "Air"
            },
            {
                "z_min": z_bottom,
                "z_max": 0.0,
                "n_ice": n_ice,
                "delta_n": delta_n,
                "z_0": z_0,
                "region": "ice",
                "region_name": "Ice"
            }
        ]

        super().__init__(
            layers=layers
        )

class ARA_2022_layered(medium_base.IceModelExpLayers):

    def __init__(self):

        # ARA South Pole model
        # https://journals.aps.org/prd/pdf/10.1103/PhysRevD.105.122006

        z_bottom = -2820 * units.meter
        n_ice = 1.78
        z_0 = 49.5049505 * units.meter
        delta_n = 0.454

        layers = [
            {
                "z_min": 0.0,
                "z_max": np.inf,
                "n_ice": 1.00027,
                "delta_n": 2.7e-4,
                "z_0": -8000.0,
                "region": "air",
                "region_name": "Air"
            },
            {
                "z_min": z_bottom,
                "z_max": 0.0,
                "n_ice": n_ice,
                "delta_n": delta_n,
                "z_0": z_0,
                "region": "ice",
                "region_name": "Ice"
            }
        ]

        super().__init__(
            layers=layers
        )

class mooresbay_simple_layered(medium_base.IceModelExpLayers):

    def __init__(self):

        # MB1 model
        # https://doi.org/10.1088/1475-7516/2018/07/055

        z_bottom = -576 * units.meter
        n_ice = 1.78
        z_0 = 34.5 * units.meter
        delta_n = 0.46


        layers = [
            {
                "z_min": 0.0,
                "z_max": np.inf,
                "n_ice": 1.00027,
                "delta_n": 2.7e-4,
                "z_0": -8000.0,
                "region": "air",
                "region_name": "Air"
            },
            {
                "z_min": z_bottom,
                "z_max": 0.0,
                "n_ice": n_ice,
                "delta_n": delta_n,
                "z_0": z_0,
                "region": "ice",
                "region_name": "Ice"
            }
        ]

        super().__init__(
            layers=layers
        )

class mooresbay_simple_2_layered(medium_base.IceModelExpLayers):

    def __init__(self):

        # MB2 model
        # https://doi.org/10.1088/1475-7516/2018/07/055

        z_bottom = -576 * units.meter
        n_ice = 1.78
        z_0 = 37.0 * units.meter
        delta_n = 0.481

        layers = [
            {
                "z_min": 0.0,
                "z_max": np.inf,
                "n_ice": 1.00027,
                "delta_n": 2.7e-4,
                "z_0": -8000.0,
                "region": "air",
                "region_name": "Air"
            },
            {
                "z_min": z_bottom,
                "z_max": 0.0,
                "n_ice": n_ice,
                "delta_n": delta_n,
                "z_0": z_0,
                "region": "ice",
                "region_name": "Ice"
            }
        ]

        super().__init__(
            layers=layers
        )


