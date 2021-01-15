from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units
import logging
logging.basicConfig()

try:
    import radiopropa
    radiopropa_is_imported = True
except ImportError:
    radiopropa_is_imported = False

logger = logging.getLogger('ice_model')

class IceModel():
    def __init__(self,z_airBoundary=0*units.meter,z_bottom=None):
        self.z_airBoundary = z_airBoundary
        self.z_bottom = z_bottom

    def get_index_of_refraction(self,x):
        """
        returns the index of refraction at position x

        Parameters
        ---------
        x:  3dim np.array
            point

        Returns:
        --------
        n:  float
            index of refraction
        """
        return 0.

    def get_average_index_of_refraction(self,x1,x2):
        """
        returns the average index of refraction between two points

        Parameters
        ----------
        x1: 3dim np.array
            point
        x2: 3dim np.array
            point

        Returns
        -------
        n_average:  float
                    averaged index of refraction between the two points
        """
        return 0.

    def get_gradient_of_index_of_refraction(self, x):
        """
        returns the gradient of index of refraction at position x

        Parameters
        ----------
        x: 3dim np.array
            point

        Returns
        -------
        n_nabla:    (3,) np.array
                    gradient of index of refraction at the point
        """
        return np.array([0.,0.,0.])

    if not radiopropa_is_imported:
        logger.warning('The radiopropa dependancy was not import and can therefore not be used. \nMore info on https://github.com/nu-radio/RadioPropa')
    else:
        def get_ice_model_radiopropa(self):
            """
            returns the ice model to insert in radiopropa
            """
            # check if the scalar field of the model is also implemented in radiopropa
            # if not, one is build through the python wrapper is stead but this is very
            # slow. Implement the model also in radiopropa if you can. Be sure to use
            # the same name for the class as here.
            if hasattr(radiopropa,type(self).__name__ ):
                scalar_field = getattr(radiopropa,type(self).__name__)()
            else:
                logger.warning('The requested ice model has no counterpart implemented in RadioPropa.')
                user_input = input('WARNING: the requested ice model has no counterpart implemented in RadioPropa. Would you like to use the much slower python version? [y/n]:')
                if user_input in ['y','Y','yes','Yes','YES']:
                    scalar_field = self.ScalarField(self)
                else:
                    print('PROGRAM ABORTED')
                    exit()

            return self.IceModel_RadioPropa(self,scalar_field)


        class IceModel_RadioPropa():
            def __init__(self, ice_model_nuradio, scalar_field):
                self.__ice_model_nuradio = ice_model_nuradio
                self.__scalar_field = scalar_field
                # these are predined modules that are inherent to the ice model like
                # discontinuities in the refractive index, reflective or transmissive
                # layers, observers to confine the model in a certain space ...
                self.__modules = {}
                
                step = np.array([0,0,1])*units.centimeter
                air_boundary_pos = np.array([0,0,self.__ice_model_nuradio.z_airBoundary])
                air_boundary = radiopropa.Discontinuity(radiopropa.Plane(radiopropa.Vector3d(*(air_boundary_pos*(radiopropa.meter/units.meter))),
                                                        radiopropa.Vector3d(0,0,1)), self.__ice_model_nuradio.get_index_of_refraction(air_boundary_pos-step), 
                                                        self.__ice_model_nuradio.get_index_of_refraction(air_boundary_pos+step))
                self.__modules["air boundary"]=air_boundary
               
                boundary_above_surface = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(*((air_boundary_pos+100*step)*
                                                                    (radiopropa.meter/units.meter))), radiopropa.Vector3d(0,0,1)))
                air_observer = radiopropa.Observer()
                air_observer.setDeactivateOnDetection(True)
                air_observer.add(boundary_above_surface)
                self.__modules["air observer"] = air_observer
                
                bottom_boundary_pos = np.array([0,0,self.__ice_model_nuradio.z_bottom])
                boundary_bottom = radiopropa.ObserverSurface(radiopropa.Plane(radiopropa.Vector3d(*((bottom_boundary_pos)*
                                                            (radiopropa.meter/units.meter))), radiopropa.Vector3d(0,0,1)))
                bottom_observer = radiopropa.Observer()
                bottom_observer.setDeactivateOnDetection(True)
                bottom_observer.add(boundary_bottom)
                self.__modules["bottom observer"] = bottom_observer
                
                if hasattr(self.__ice_model_nuradio, 'reflection'):
                    reflection_pos = np.array([0,0,self.__ice_model_nuradio.reflection])
                    bottom_reflection = radiopropa.ReflectiveLayer(radiopropa.Plane(radiopropa.Vector3d(*(reflection_pos*(radiopropa.meter/units.meter))),
                                                                    radiopropa.Vector3d(0,0,1)),self.__ice_model_nuradio.reflection_coefficient)
                    self.__modules["bottom reflection"]=bottom_reflection

            def get_modules(self):
                """
                returns the predefined modules (like reflective or transmissive layers, 
                a discontinuity in refractive index, observers ...) of the ice for 
                to use in the radiopropa tracer

                Returns
                -------
                modules:    dictionary {name:module object}
                """
                return self.__modules

            def add_module(self,name,module):
                """
                add predefined modules (like reflective or transmissive layers, 
                a discontinuity in refractive index, observers ...) of the ice 
                to the dictionary

                Parameter
                -------
                name:   string
                        name to identify the module  
                module: radiopropa.Module (and all the daugther classes)
                        module to run in radiopropa
                """
                self.__modules[name]=module

            def remove_module(self,name):
                if name in self.__modules.keys():
                    self.__modules.pop(name)

            def replace_module(self,name,new_module):
                self.__modules[name] = new_module
                if name not in self.__modules.keys():
                    logger.info('Module with name {} does not exist yet and thus cannot be replaced, module just added'.format(name))

            def get_scalar_field(self):
                return self.__scalar_field

        class ScalarField(radiopropa.ScalarField):
            def __init__(self, ice_model_nuradio):
                radiopropa.ScalarField.__init__(self)
                self.__ice_model_nuradio = ice_model_nuradio
                
            def getValue(self,position): #name may not be changed because linked to c++ radiopropa module
                """
                returns the index of refraction at position for radiopropa tracer

                Parameters
                ---------
                position:   radiopropa.Vector3d
                            point

                Returns:
                --------
                n:  float
                    index of refraction
                """
                x = np.array([position.x,position.y,position.z]) *(radiopropa.meter/units.meter)
                return self.__ice_model_nuradio.get_index_of_refraction(x)

            def getGradient(self,position): #name may not be changed because linked to c++ radiopropa module
                """
                returns the gradient of index of refraction at position for radiopropa tracer

                Parameters
                ----------
                position:   radiopropa.Vector3d
                            point

                Returns
                -------
                n_nabla:    radiopropa.Vector3d
                            gradient of index of refraction at the point
                """
                x = np.array([position.x,position.y,position.z]) *(radiopropa.meter/units.meter)
                gradient = self.__ice_model_nuradio.get_gradient_of_index_of_refraction(x)
                return radiopropa.Vector3d(*gradient)


class IceModel_Exponential(IceModel):
    def __init__(self,z_airBoundary=0*units.meter,z_bottom=None,n_ice=None,z_0=None,delta_n=None):
        IceModel.__init__(self,z_airBoundary,z_bottom)
        self.n_ice = n_ice
        self.z_0 = z_0
        self.delta_n = delta_n

    def get_index_of_refraction(self, x):
        """
        overwrite inherited function
        """
        if (x[2] - self.z_airBoundary) <=0:
            return self.n_ice  - self.delta_n  * np.exp((x[2]-self.z_airBoundary) / self.z_0)
        else:
            return 1

    def get_average_index_of_refraction(self, x1, x2):
        """
        overwrite inherited function
        """
        if ((x1[2] - self.z_airBoundary) <=0) and ((x2[2] - self.z_airBoundary) <=0):
            return self.n_ice - self.delta_n * self.z_0 / (x2[2] - x1[2]) * (np.exp((x2[2]-self.z_airBoundary) / self.z_0) - np.exp((x1[2]-self.z_airBoundary) / self.z_0))
        else:
            return None

    def get_gradient_of_index_of_refraction(self, x):
        """
        overwrite inherited function
        """
        gradient = np.array([0.,0.,0.])
        if (x[2] - self.z_airBoundary) <=0:
            gradient[2] = - self.delta_n / self.z_0 * np.exp((x[2]-self.z_airBoundary) / self.z_0)
        return gradient

class IceModel_ReflectiveBottom(IceModel):
    def __init__(self,refl_z,refl_coef,refl_phase_shift):
        self.reflection = z_reflection  # from https://doi.org/10.3189/2015JoG14J214
        self.reflection_coefficient = reflection_coef  # from https://doi.org/10.3189/2015JoG14J214
        self.reflection_phase_shift = reflection_phase_shift