from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units
import logging
logging.basicConfig()

try:
    import radiopropa as RP
    radiopropa_is_imported = True
except ImportError:
    radiopropa_is_imported = False

logger = logging.getLogger('ice_model')

class IceModel():
    """
    Base class from which all ice models should inheret
    """
    def __init__(self,z_airBoundary=0*units.meter,z_bottom=None):
        self.z_airBoundary = z_airBoundary
        self.z_bottom = z_bottom

    def add_reflective_bottom(self,refl_z,refl_coef,refl_phase_shift):
        """
        function which adds a reflective bottom to your ice model
        """
        self.reflection = refl_z
        self.reflection_coefficient = refl_coef
        self.reflection_phase_shift = refl_phase_shift
        
        if not((self.z_bottom != None) and (self.z_bottom < self.reflection)):
            self.z_bottom = self.reflection-1*units.m

    def get_index_of_refraction(self,x):
        """
        returns the index of refraction at position x.
        Reimplement everytime in the specific model

        Parameters
        ---------
        x:  3dim np.array
            point

        Returns:
        --------
        n:  float
            index of refraction
        """
        pass

    def get_average_index_of_refraction(self,x1,x2):
        """
        returns the average index of refraction between two points
        Reimplement everytime in the specific model

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
        pass

    def get_gradient_of_index_of_refraction(self, x):
        """
        returns the gradient of index of refraction at position x
        Reimplement everytime in the specific model

        Parameters
        ----------
        x: 3dim np.array
            point

        Returns
        -------
        n_nabla:    (3,) np.array
                    gradient of index of refraction at the point
        """
        pass

    
    def get_ice_model_radiopropa(self):
        """
        if radiopropa is installed this will return the ice model to insert in radiopropa

        """
        if radiopropa_is_imported:
            # when implementing a new ice_model this part of the function should be ice model specific
            pass
        else:
            logger.error('The radiopropa dependancy was not import and can therefore not be used. \nMore info on https://github.com/nu-radio/RadioPropa')
            raise ImportError('RadioPropa could not be imported')


class IceModel_Simple(IceModel):
    """
    predefined ice model (to inherit from) with exponential shape
    """
    def __init__(self,n_ice,delta_n,z_0,z_shift=0*units.meter,z_airBoundary=0*units.meter,z_bottom=None):
        super().__init__(z_airBoundary,z_bottom)
        self.n_ice = n_ice
        self.delta_n = delta_n
        self.z_0 = z_0
        self.z_shift = z_shift

    def get_index_of_refraction(self, x):
        """
        overwrite inherited function
        """
        if (x[2] - self.z_airBoundary) <=0:
            return self.n_ice  - self.delta_n  * np.exp((x[2]-self.z_shift) / self.z_0)
        else:
            return 1

    def get_average_index_of_refraction(self, x1, x2):
        """
        overwrite inherited function
        """
        if ((x1[2] - self.z_airBoundary) <=0) and ((x2[2] - self.z_airBoundary) <=0):
            return self.n_ice - self.delta_n * self.z_0 / (x2[2] - x1[2]) * (np.exp((x2[2]-self.z_shift) / self.z_0) - np.exp((x1[2]-self.z_shift) / self.z_0))
        else:
            return None

    def get_gradient_of_index_of_refraction(self, x):
        """
        overwrite inherited function
        """
        gradient = np.array([0.,0.,0.])
        if (x[2] - self.z_airBoundary) <=0:
            gradient[2] = - self.delta_n / self.z_0 * np.exp((x[2]-self.z_shift) / self.z_0)
        return gradient

    def get_ice_model_radiopropa(self):
        """
        overwrite inherited function
        """
        if radiopropa_is_imported:
            scalar_field = RP.IceModel_Simple(z_surface=self.z_airBoundary*RP.meter/units.meter, 
                                            n_ice=self.n_ice, delta_n=self.delta_n, 
                                            z_0=self.z_0*RP.meter/units.meter,
                                            z_shift=self.z_shift*RP.meter/units.meter)
            return IceModel_RadioPropa(self,scalar_field)
        else:
            logger.error('The radiopropa dependancy was not import and can therefore not be used. \nMore info on https://github.com/nu-radio/RadioPropa')
            raise ImportError('RadioPropa could not be imported')



if radiopropa_is_imported:
    class IceModel_RadioPropa():
        """
        This class holds all the necessary variables for the radiopropa raytracer to work.
        When radiopropa is installed, this object will automatically be generated for a
        smooth handeling of the radiopropa ice model.
        """
        def __init__(self, ice_model_nuradio, scalar_field):
            # the ice model of NuRadioMC on which this object is based
            self.__ice_model_nuradio = ice_model_nuradio
            # this hold a radiopropa.scalarfield of the refractive index
            self.__scalar_field = scalar_field
            # these are predined modules that are inherent to the ice model like
            # discontinuities in the refractive index, reflective or transmissive
            # layers, observers to confine the model in a certain space ...
            self.__modules = {}
            
            step = np.array([0,0,1])*units.centimeter
            air_boundary_pos = np.array([0,0,self.__ice_model_nuradio.z_airBoundary])
            air_boundary = RP.Discontinuity(RP.Plane(RP.Vector3d(*(air_boundary_pos*(RP.meter/units.meter))),
                                RP.Vector3d(0,0,1)), self.__ice_model_nuradio.get_index_of_refraction(air_boundary_pos-step), 
                                self.__ice_model_nuradio.get_index_of_refraction(air_boundary_pos+step))
            self.__modules["air boundary"]=air_boundary
           
            boundary_above_surface = RP.ObserverSurface(RP.Plane(RP.Vector3d(*((air_boundary_pos+100*step)*
                                                                (RP.meter/units.meter))), RP.Vector3d(0,0,1)))
            air_observer = RP.Observer()
            air_observer.setDeactivateOnDetection(True)
            air_observer.add(boundary_above_surface)
            self.__modules["air observer"] = air_observer
            
            bottom_boundary_pos = np.array([0,0,self.__ice_model_nuradio.z_bottom])
            boundary_bottom = RP.ObserverSurface(RP.Plane(RP.Vector3d(*((bottom_boundary_pos)*
                                                (RP.meter/units.meter))), RP.Vector3d(0,0,1)))
            bottom_observer = RP.Observer()
            bottom_observer.setDeactivateOnDetection(True)
            bottom_observer.add(boundary_bottom)
            self.__modules["bottom observer"] = bottom_observer
            
            if hasattr(self.__ice_model_nuradio, 'reflection'):
                reflection_pos = np.array([0,0,self.__ice_model_nuradio.reflection])
                bottom_reflection = RP.ReflectiveLayer(RP.Plane(RP.Vector3d(*(reflection_pos*(RP.meter/units.meter))),
                                                    RP.Vector3d(0,0,1)),self.__ice_model_nuradio.reflection_coefficient)
                self.__modules["bottom reflection"]=bottom_reflection

        def get_modules(self):
            """
            returns the predefined modules (like reflective or transmissive layers, 
            a discontinuity in refractive index, observers ...) of the ice for 
            to use in the radiopropa tracer

            Returns
            -------
            modules: dictionary {name:module object}
                     dictionary of modules to run in radiopropa
            """
            return self.__modules

        def get_module(self,name):
            """
            returns the predefined module with that name (like reflective or 
            transmissive layers, a discontinuity in refractive index, observers 
            ...) of the ice for to use in the radiopropa tracer

            Returns
            -------
            module: module with this name
                    radiopropa.module object
            """
            if name not in self.__modules.keys():
                logger.error('Module with name {} does not exist.'.format(name))
                raise AttributeError('Module with name {} does not already exist.'.format(name))
            else:
                return self.__modules[name]

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
            if name in self.__modules.keys():
                logger.error('Module with name {} does already exist, use the replace_module function if you want to replace this module'.format(name))
                raise AttributeError('Module with name {} does already exist, use the replace_module function if you want to replace this module'.format(name))
            else:
                self.__modules[name]=module

        def remove_module(self,name):
            """
            removes predefined modules (like reflective or transmissive layers, 
            a discontinuity in refractive index, observers ...) of the ice 
            to the dictionary

            Parameter
            -------
            name:   string
                    name to identify the module to be removed
            """
            if name in self.__modules.keys():
                self.__modules.pop(name)

        def replace_module(self,name,new_module):
            """
            replaces predefined modules (like reflective or transmissive layers, 
            a discontinuity in refractive index, observers ...) of the ice 
            to the dictionary

            Parameter
            -------
            name:   string
                    name to identify the module to be replaced
            module: radiopropa.Module (and all the daugther classes)
                    new module to run in radiopropa
            """
            if name not in self.__modules.keys():
                logger.info('Module with name {} does not exist yet and thus cannot be replaced, module just added'.format(name))
            self.__modules[name] = new_module

        def get_scalar_field(self):
            """
            add predefined modules (like reflective or transmissive layers, 
            a discontinuity in refractive index, observers ...) of the ice 
            to the dictionary

            Parameter
            -------
            scalar_field: radiopropa.ScalarField
                          scalar field that holds the refractive index to use in radiopropa
            """
            return self.__scalar_field

    class ScalarFieldBuilder(RP.ScalarField):
        """
        If the requested ice model does not exist in radiopropa, this class can build
        one in stead. It is a radiopropa object but constructed through the python
        wrapper which is much slower. 
        """
        def __init__(self, ice_model_nuradio):
            RP.ScalarField.__init__(self)
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
            x = np.array([position.x,position.y,position.z]) *(RP.meter/units.meter)
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
            x = np.array([position.x,position.y,position.z]) *(RP.meter/units.meter)
            gradient = self.__ice_model_nuradio.get_gradient_of_index_of_refraction(x)
            return RP.Vector3d(*gradient)