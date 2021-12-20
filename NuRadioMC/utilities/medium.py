from NuRadioMC.utilities import medium_base
import numpy as np
from NuRadioReco.utilities import units
import logging
logging.basicConfig()

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
        z_air_boundary:  float, NuRadio length units
                         z coordinate of the surface of the glacier
        z_bottom:  float, NuRadio length units
                   z coordinate of the bedrock/bottom of the glacier.
        z_firn:  float, NuRadio length units
                 z coordinate of the transition from the upper 
                 exponential profile to the lower one

        The following parameters can be found without (lower)
        and with (upper) the suffix of `_firn`

        n_ice:  float, dimensionless
                refractive index of the deep bulk ice
        delta_n:  float, NuRadio length units
                  difference between n_ice and the refractive index
                  of the snow at the surface
        z_0:  float, NuRadio length units
              scale depth of the exponential
        z_shift:  float, NuRadio length units
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
        position:  3dim np.array
                    point

        Returns
        -------
        n:  float
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
        n_average:  float
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
        n_nabla:    (3,) np.array
                    gradient of index of refraction at the point
        """
        pos = RP.Vector3d(*(position * RP.meter/units.meter))
        return self._scalarfield.getGradient(pos) * (1 / (units.meter/RP.meter))

    
    def get_ice_model_radiopropa(self):
        """
        Returns an object holding the radiopropa scalarfield and necessary radiopropa moduldes 
        that define the medium in radiopropa. It uses the parameters of the medium object to 
        contruct some modules, like a discontinuity object for the air boundary. Additional modules
        can be added in this function
        
        Overwrites function of the mother class

        Returns
        -------
        ice:    RadioPropaIceWrapper
                object holding the radiopropa scalarfield and modules
        """
        ice = medium_base.RadioPropaIceWrapper(self, self._scalarfield)
        return ice





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