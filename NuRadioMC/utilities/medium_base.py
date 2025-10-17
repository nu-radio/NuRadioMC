from NuRadioReco.utilities import units

from scipy import interpolate, integrate, linalg
import numpy as np
import logging

try:
    import radiopropa as RP
    # nu2rp_meter = RP.meter / units.meter
    # The line above gives 1. The reason why we are not using it is that it gives an error when building the docs:
    # TypeError: unsupported operand type(s) for /: 'meter' and 'int'
    # I do not understand why... Therefore we define it manually here
    nu2rp_meter = 1
    radiopropa_is_imported = True
except ImportError:
    radiopropa_is_imported = False

logger = logging.getLogger('NuRadioMC.ice_model')


class IceModel():
    """
    Base class from which all ice models should inheret
    """
    def __init__(self, z_air_boundary=0*units.meter, z_bottom=None):
        """
        initiaion of a basic ice model

        The bottom defined here is a boundary condition used in simulations and
        should always be defined. Note: it is not the same as reflective bottom.
        The latter can be added using the `add_reflective_layer` function.

        Parameters
        ----------
        z_air_boundary: float, NuRadio length units
            z coordinate of the surface of the glacier
        z_bottom: float, NuRadio length units
            z coordinate of the bedrock/bottom of the glacier.
        """
        self.z_air_boundary = z_air_boundary
        self.z_bottom = z_bottom
        self.reflection = None
        self.reflection_coefficient = None
        self.reflection_phase_shift = None
        self._ice_model_radiopropa = None

    def add_reflective_bottom(self, refl_z, refl_coef, refl_phase_shift):
        """
        function which adds a reflective bottom to your ice model

        Parameters
        ----------
        refl_z: float, NuRadio length units
            z coordinate of the bottom reflective layer
        refl_coef: float between 0 and 1
            fraction of the electric field that gets reflected
        refl_phase_shift: float, NuRadio angukar units
            phase shoft that the reflected electric field receives
        """
        self.reflection = refl_z
        self.reflection_coefficient = refl_coef
        self.reflection_phase_shift = refl_phase_shift

        if not ((self.z_bottom is not None) and (self.z_bottom < self.reflection)):
            # bottom should always be below the reflective layer
            self.z_bottom = self.reflection - 1*units.m

    def get_index_of_refraction(self, position):
        """ Returns the index of refraction at position.

        Has to be reimplement for every specific model.

        Parameters
        ----------
        position: 3dim np.array
            point

        Returns
        -------
        n: float
            index of refraction
        """
        logger.error('Function not implemented - this function has to be reimplemented for every specific model')
        raise NotImplementedError('Function not implemented - this function has to be reimplemented for every specific model')

    def get_average_index_of_refraction(self, position1, position2):
        """ Returns the average index of refraction between two points

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
        logger.warning(
            'Using general implementation of function which might be slow. '
            'For faster calculation, overwrite with an ice model specific function')

        def get_index_of_refraction(x, y, z):
            pos = np.array([x, y, z])
            return self.get_index_of_refraction(pos)

        ranges = [[position1[0],position2[0]],
                  [position1[1],position2[1]],
                  [position1[2],position2[2]]]

        int_result = integrate.nquad(get_index_of_refraction, ranges)
        n_average = int_result[0] / linalg.norm(position2 - position1)
        return n_average

    def get_gradient_of_index_of_refraction(self, position):
        """ Returns the gradient of index of refraction at position

        Has to be reimplement for every specific model.

        Parameters
        ----------
        position: 3dim np.array
            point

        Returns
        -------
        n_nabla: (3,) np.array
            Gradient of index of refraction at the point
        """
        logger.error('Function not implemented - this function has to be reimplemented for every specific model')
        raise NotImplementedError('Function not implemented - this function has to be reimplemented for every specific model')


    def _compute_default_ice_model_radiopropa(self):
        """
        Computes a default object holding the radiopropa scalarfield and necessary radiopropa
        moduldes that define the medium in radiopropa. It uses the parameters of the medium
        object to contruct some modules, like a discontinuity object for the air boundary.
        Additional modules can be added in this function

        This is the default and should always be overriden/implemented in new ice model!

        Returns
        -------
        ice_model_radiopropa: RadioPropaIceWrapper
            Object holding the radiopropa scalarfield and modules
        """
        if not radiopropa_is_imported:
            logger.error('The radiopropa dependancy was not import and can therefore not be used. \nMore info on https://github.com/nu-radio/RadioPropa')
            raise ImportError('RadioPropa could not be imported')

        # when implementing a new ice_model this part of the function should be ice model specific
        # if the new ice_model cannot be used in RadioPropa, this function should throw an error
        logger.error('Function not implemented - this function has to be reimplemented for every specific model')
        raise NotImplementedError('Function not implemented - this function has to be reimplemented for every specific model')

    def get_ice_model_radiopropa(self):
        """
        Returns an object holding the radiopropa scalarfield and necessary radiopropa moduldes
        that define the medium in radiopropa. If no specific model is set by the user it returns
        the default implemented model using the '_compute_default_ice_model_radiopropa' function.

        This seperation allows having the posibility to set a more specific/adjusted radiopropa
        ice model in case they need it, without losing the access to the default model.

        DO NOT OVERRIDE THIS FUNCTION

        Returns
        -------
        ice: RadioPropaIceWrapper
            Object holding the radiopropa scalarfield and modules
        """
        if not radiopropa_is_imported:
            logger.error('The radiopropa dependancy was not import and can therefore not be used. \nMore info on https://github.com/nu-radio/RadioPropa')
            raise ImportError('RadioPropa could not be imported')

        if self._ice_model_radiopropa is None:
            self._ice_model_radiopropa = self._compute_default_ice_model_radiopropa()

        return self._ice_model_radiopropa

    def set_ice_model_radiopropa(self, ice_model_radiopropa):
        """
        If radiopropa is installed, this function can be used
        to set a specific RadioPropaIceWrapper object as the
        ice model used for RadioPropa.

        DO NOT OVERRIDE THIS FUNCTION

        Parameters
        ----------
        ice_model_radioprop: RadioPropaIceWrapper
            Object holding the radiopropa scalarfield and modules

        """
        if not radiopropa_is_imported:
            logger.error('The radiopropa dependancy was not import and can therefore not be used. '
                         '\nMore info on https://github.com/nu-radio/RadioPropa')
            raise ImportError('RadioPropa could not be imported')

        self._ice_model_radiopropa = ice_model_radiopropa


class IceModelSimple(IceModel):
    """
    Predefined ice model (to inherit from) with exponential shape
    """
    def __init__(self,
                 n_ice,
                 delta_n,
                 z_0,
                 z_shift=0*units.meter,
                 z_air_boundary=0*units.meter,
                 z_bottom=None
                 ):

        """
        Initiaion of a simple exponential ice model

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

        super().__init__(z_air_boundary, z_bottom)
        self.n_ice = n_ice
        self.delta_n = delta_n
        self.z_0 = z_0
        self.z_shift = z_shift

    def get_index_of_refraction(self, position):
        """
        Returns the index of refraction at position.

        Parameters
        ----------
        position: 1D (3,) or 2D (n,3) numpy array
            Either one position or an array of positions for which the indices
            of refraction are returned,

        Returns
        -------
        n: float or 1D numpy array (n,)
            index of refraction
        """
        if isinstance(position, list) or position.ndim == 1:
            if (position[2] - self.z_air_boundary) <= 0:
                return self.n_ice - self.delta_n * np.exp((position[2] - self.z_shift) / self.z_0)
            else:
                return 1
        else:
            ior = self.n_ice - self.delta_n * np.exp((position[:, 2] - self.z_shift) / self.z_0)
            ior[position[:, 2] - self.z_air_boundary > 0] = 1.
            return ior

    def get_average_index_of_refraction(self, position1, position2):
        """
        Returns the average index of refraction between two points

        Parameters
        ----------
        position1: 1D (3,) or 2D (n,3) numpy array
            Either one position or an array of positions for which the indices
            of average refraction are returned
        position2: 1D (3,) or 2D (n,3) numpy array
            Either one position or an array of positions for which the indices
            of average refraction are returned

        Returns
        -------
        n_average: float of 1D numpy array (n,)
            averaged index of refraction between the two points
        """

        def exp_average(z_max, z_min):
            return (self.n_ice - self.delta_n * self.z_0 / (z_max - z_min)
                    * (np.exp((z_max-self.z_shift) / self.z_0) - np.exp((z_min-self.z_shift) / self.z_0)))

        if (isinstance(position1, list) or position1.ndim == 1) and (isinstance(position2, list) or position2.ndim == 1):
            zmax = max(position1[2], position2[2])
            zmin = min(position1[2], position2[2])
            if ((zmax - self.z_air_boundary) <=0):
                return exp_average(zmax, zmin)
            elif ((zmin - self.z_air_boundary) <=0):
                n1 = exp_average(self.z_air_boundary, zmin)
                n2 = 1
                return (n1 * (self.z_air_boundary - zmin) + n2 * (zmax - self.z_air_boundary)) / (zmax - zmin)
            else:
                return 1
        else:
            if all((position1[:,2] - self.z_air_boundary) <= 0) and all((position2[:,2] - self.z_air_boundary) <= 0):
                return exp_average(position1[:,2], position2[:,2])
            elif all((position1[:,2] - self.z_air_boundary) > 0) and all((position2[:,2] - self.z_air_boundary) > 0):
                return np.ones_like(position1[:,2])
            else:
                raise NotImplementedError('function cannot handle averages accros boundary when using arrays of positions.')

    def get_gradient_of_index_of_refraction(self, position):
        """
        Returns the gradient of index of refraction at position

        Parameters
        ----------
        position: 1D or 2D numpy array
            Either one position or an array of positions for which the gradient
            of index of refraction is returned

        Returns
        -------
        n_nabla: 1D (3,) or 2D (n,3) numpy array
            gradient of index of refraction at the point
        """
        def gradient_z(z):
            return -self.delta_n / self.z_0 * np.exp((z - self.z_shift) / self.z_0)

        if (isinstance(position, list) or position.ndim == 1):
            gradient = np.array([0,0,0])
            if (position[2] - self.z_air_boundary) <= 0:
                gradient[2] = gradient_z(position[2])
        else:
            gradient = gradient_z(position[:,2])
            gradient[position[:, 2] - self.z_air_boundary > 0] = 0
            gradient = np.stack((np.zeros_like(gradient),np.zeros_like(gradient),gradient),axis=1)

        return gradient

    def _compute_default_ice_model_radiopropa(self):
        """
        If radiopropa is installed this will compute and return a default object holding the
        radiopropa scalarfield and necessary radiopropa moduldes that define the medium in
        radiopropa. It uses the parameters of the medium object to contruct the scalar field
        using the simple ice model implementation in radiopropa and some modules, like a
        discontinuity object for the air boundary.

        Returns
        -------
        ice: RadioPropaIceWrapper
            Object holding the radiopropa scalarfield and modules
        """
        if not radiopropa_is_imported:
            logger.error('The radiopropa dependency was not import and can therefore not be used.'
                        +'\nMore info on https://github.com/nu-radio/RadioPropa')
            raise ImportError('RadioPropa could not be imported')

        scalar_field = RP.IceModel_Simple(
            z_surface=self.z_air_boundary * nu2rp_meter,
            n_ice=self.n_ice, delta_n=self.delta_n,
            z_0=self.z_0 * nu2rp_meter,
            z_shift=self.z_shift * nu2rp_meter
        )

        return RadioPropaIceWrapper(self, scalar_field)


class IceModelBirefringence(IceModelSimple):
    """
    predefined birefringence ice model (to inherit from) including different indieces of refraction for differnt directions
    """
    def __init__(self, bir_model):

        """
        initiaion of a birefringent ice model with an interpolation of the data as described in:
        https://link.springer.com/article/10.1140/epjc/s10052-023-11238-y

        Parameters
        ----------
        bire_model: string
            choose the interpolation to fit the measured refractive index data
            options include (A, B, C, D, E) description can be found under: NuRadioMC/NuRadioMC/utilities/birefringence_models/model_description

        """

        self.f1 = interpolate.UnivariateSpline._from_tck(bir_model[0])
        self.f2 = interpolate.UnivariateSpline._from_tck(bir_model[1])
        self.f3 = interpolate.UnivariateSpline._from_tck(bir_model[2])

    def get_birefringence_index_of_refraction(self, position):

        """
        Returns the birefringent index of refraction at any position, no density effects are included at this point.

        Parameters
        ----------
        position: 3dim np.array [x, y, z]
            position at which the ice model should be evaluated

        Returns
        -------
        n: list [nx, ny, nz]
            index of refraction for every direction
        """

        nx = self.f1( - position[2])
        ny = self.f2( - position[2])
        nz = self.f3( - position[2])

        return nx, ny, nz

class IceModelExponentialPolynomial(IceModel):
    """
    Predefined ice model (to inherit from) with polynomial exponential shape of degree n
    """
    def __init__(self, a, z_0, z_shift=0 * units.meter, z_air_boundary=0 * units.meter, z_bottom=None,
                 density_factor=0.8506 * (units.cm**3 / units.gram)):
        """
        Initiation of an exponential polynomial ice model.

        .. math::

            n(z) = 1 + density_factor * \sum_{i=0}^{n}( a_{i} * \exp^{i * (z - z_shift) / z_0} )

        for z_bottom < z < z_air_boundary, see eq. 5.2 in https://doi.org/10.5281/zenodo.15067984

        The bottom defined here is a boundary condition used in simulations and
        should always be defined. Note: it is not the same as reflective bottom.
        The latter can be added using the `add_reflective_layer` function.

        The z_shift is a variable introduced to be able to shift the model
        up or down along the z direction when this is needed to account for
        differences along station for example.

        Parameters
        ----------
        a: (n,) np.array of floats, NuRadio density units
            coefficients for the nth-degree exponential polynomial
            describing the vertical ice density profile
        z_0: float, NuRadio length units
            scale depth of the exponential

        [optional]
        z_shift: float, NuRadio length units
            up or down shift od the exponential profile
        z_air_boundary: float, NuRadio length units
            z coordinate of the surface of the glacier
        z_bottom: float, NuRadio length units
            z coordinate of the bedrock/bottom of the glacier.
        density_factor: float, NuRadio density units
            factor used to translate density to refractive index
            using Robin's equation.
        """

        super().__init__(z_air_boundary=z_air_boundary, z_bottom = z_bottom)
        self._z_0 = z_0
        self._a = a
        self._density_factor = density_factor
        self._z_air_boundary = z_air_boundary
        self._z_shift = z_shift

    def get_index_of_refraction(self, position):
        """
        Returns the index of refraction at position.

        Parameters
        ----------
        position: np.array of shape (3,) or (n,3)
            point(s) in space

        Returns
        -------
        n: float
            index of refraction
        """
        def ior(z):
            x = np.exp((z-self._z_shift)/self._z_0)
            rho = 0.
            for i, _ in enumerate(self._a):
                rho += self._a[i]*x**i
            return 1 + rho*self._density_factor

        if isinstance(position, list) or position.ndim == 1:
            if (position[2] - self.z_air_boundary) <= 0:
                return ior(position[2])
            else:
                return 1.
        else:
            ior = ior(position[:,2])
            ior[position[:, 2] - self.z_air_boundary > 0] = 1.
            return ior

    def get_average_index_of_refraction(self, position1, position2):
        """
        Returns the average index of refraction between two points

        Parameters
        ----------
        position1: np.array of shape (3,) or (n,3)
            point(s) in space
        position2: np.array of shape (3,) or (n,3)
            point(s) in space

        Returns
        -------
        n_average: float
            averaged index of refraction between the two points
        """
        if np.abs(position1 - position2)[2] < .1:
            return self.get_index_of_refraction(position1)
        def int_rho(z):
            x = np.exp((z-self._z_shift)/self._z_0)
            int_rho_z = self._a[0]*(z-self._z_shift)
            for i, _ in enumerate(self._a):
                if i == 0:
                    continue

                int_rho_z += (self._a[i]*self._z_0/i)*x**i
            return int_rho_z

        if (isinstance(position1, list) or position1.ndim == 1) and (isinstance(position2, list) or position2.ndim == 1):
            if (position1[2] - self.z_air_boundary) <= 0 and (position2[2] - self.z_air_boundary) <= 0:
                return ((int_rho(position2[2]) - int_rho(position1[2])) / (position2[2] - position1[2]))*self._density_factor + 1
            elif (position1[2] - self.z_air_boundary) > 0 and (position2[2] - self.z_air_boundary) > 0:
                return 1
            else:
                pos_min = position1
                pos_max = position2
                if position1[2]>position2[2]:
                    pos_min = position2
                    pos_max = position1
                integral_ice = abs(int_rho(self._z_air_boundary) - int_rho(pos_min[2]))*self._density_factor + 1
                integral_air = 1 * (pos_max[2] - self._z_air_boundary)
                return (integral_ice + integral_air) / abs(pos_max[2] - pos_min[2])
        else:
            if all((position1[:,2] - self.z_air_boundary) <= 0) and all((position2[:,2] - self.z_air_boundary) <= 0):
                return ((int_rho(position2[:,2]) - int_rho(position1[:,2])) / (position2[:,2] - position1[:,2]))*self._density_factor + 1
            elif all((position1[:,2] - self.z_air_boundary) > 0) and all((position2[:,2] - self.z_air_boundary) > 0):
                return np.ones_like(position1[:,2])
            else:
                raise NotImplementedError('function cannot handle averages accros boundary when using arrays of positions.')

    def get_gradient_of_index_of_refraction(self, position):
        """
        Returns the gradient of index of refraction at position

        Parameters
        ----------
        position: np.array of shape (3,) or (n, 3)
            point(s) in space

        Returns
        -------
        n_nabla: (3,) np.array
            gradient of index of refraction at the point
        """
        def dior_dz(z):
            x = np.exp((z-self._z_shift)/self._z_0)
            drho_dz = 0
            for i, _ in enumerate(self._a):
                drho_dz += (self._a[i]*i/self._z_0)*x**i
            return (drho_dz*self._density_factor)

        if (isinstance(position, list) or position.ndim == 1):
            if (position[2] - self.z_air_boundary) <= 0:
                return np.array([0, 0, dior_dz(position[2])])
            else:
                return np.array([0, 0, 0])
        else:
            dior = dior_dz(position[:,2])
            dior[position[:, 2] - self.z_air_boundary > 0] = 0
            return np.stack((np.zeros_like(dior),np.zeros_like(dior),dior),axis=1)


    def _compute_default_ice_model_radiopropa(self):
        """
        If radiopropa is installed this will compute and return a default object holding the
        radiopropa scalarfield and necessary radiopropa moduldes that define the medium in
        radiopropa. It uses the parameters of the medium object to contruct the scalar field
        using the polynomial ice model implementation in radiopropa and some modules, like a
        discontinuity object for the air boundary.

        If you want to add, remove or replace modules, you can use the returned object to adjust
        the model to your liking and set it using the set_ice_model_radiopropa function.

        Returns
        -------
        ice: RadioPropaIceWrapper
            object holding the radiopropa scalarfield and modules
        """
        if not radiopropa_is_imported:
            logger.error('The radiopropa dependency was not import and can therefore not be used.'
                         '\nMore info on https://github.com/nu-radio/RadioPropa')
            raise ImportError('RadioPropa could not be imported')

        coeff = RP.DoubleVector_1D()
        for ai in self._a:
            coeff.push_back(ai)

        scalar_field = RP.IceModel_Polynomial(
            coeff,
            self._z_0 * nu2rp_meter,
            self.z_air_boundary * nu2rp_meter,
            self._z_shift * nu2rp_meter,
            self._density_factor)

        return RadioPropaIceWrapper(self, scalar_field)


    def set_density_factor(self, density_factor):
        """
        Set the density factor to a new value

        Parameters
        ----------
        density_factor: float, NuRadio density units
            factor to translate density to refractive index
        """
        self._density_factor = density_factor


if radiopropa_is_imported:
    """
    RadioPropa is a C++ module dedicated for ray tracing. It is a seperate module and
    it has its own unit system. However, all object within NuRadio ecosystem are in the
    NuRadio uit system. Therefore, when passing argument from NuRadio to RadioPropa, or
    when receiving object from RadioPropa into NuRadio the units of object needed to be
    converted to the right unit system. Below is an example given for an object 'distance'

    - from NuRadio to RadioPropa:
        distance_in_meter = distance_in_nuradio / units.meter
        --> this converts the distance from NuRadio units into SI unit meter
        distance_in_radiopropa = distance_in_meter * radiopropa.meter
        --> this converts the distance from SI unit meter into RadioPropa units

    - from RadioPropa to NuRadio:
        distance_in_meter = distance_in_radiopropa / radiopropa.meter
        --> this converts the distance from RadioPropa units into SI unit meter
        distance_in_nuradio = distance_in_meter * units.meter
        --> this converts the distance from SI unit meter into NuRadio units
    """

    z_unit = RP.Vector3d(0, 0, 1)

    class RadioPropaIceWrapper():
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

            step = np.array([0, 0, 1]) * units.centimeter
            air_boundary_pos = np.array([0, 0, self.__ice_model_nuradio.z_air_boundary])

            air_boundary = RP.Discontinuity(
                RP.Plane(RP.Vector3d(0, 0, self.__ice_model_nuradio.z_air_boundary * nu2rp_meter), z_unit),
                self.__ice_model_nuradio.get_index_of_refraction(air_boundary_pos - step),
                self.__ice_model_nuradio.get_index_of_refraction(air_boundary_pos + step),
            )
            self.__modules["air boundary"] = air_boundary

            boundary_bottom = RP.ObserverSurface(
                RP.Plane(RP.Vector3d(0, 0, self.__ice_model_nuradio.z_bottom * nu2rp_meter), z_unit))

            bottom_observer = RP.Observer()
            bottom_observer.setDeactivateOnDetection(True)
            bottom_observer.add(boundary_bottom)
            self.__modules["bottom observer"] = bottom_observer

            if hasattr(self.__ice_model_nuradio, 'reflection') and self.__ice_model_nuradio.reflection is not None:
                bottom_reflection = RP.ReflectiveLayer(
                    RP.Plane(RP.Vector3d(0, 0, self.__ice_model_nuradio.reflection * nu2rp_meter), z_unit),
                    self.__ice_model_nuradio.reflection_coefficient,
                )
                self.__modules["bottom reflection"] = bottom_reflection

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

        def get_module(self, name):
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

        def add_module(self, name, module):
            """
            add predefined modules (like reflective or transmissive layers,
            a discontinuity in refractive index, observers ...) of the ice
            to the dictionary

            Parameters
            ----------
            name: string
                name to identify the module
            module: radiopropa.Module (and all the daugther classes)
                module to run in radiopropa
            """
            if name in self.__modules.keys():
                logger.error('Module with name {} does already exist, use the replace_module function if you want to replace this module'.format(name))
                raise AttributeError('Module with name {} does already exist, use the replace_module function if you want to replace this module'.format(name))
            else:
                self.__modules[name]=module

        def remove_module(self, name):
            """
            removes predefined modules (like reflective or transmissive layers,
            a discontinuity in refractive index, observers ...) of the ice
            to the dictionary

            Parameters
            ----------
            name: string
                name to identify the module to be removed
            """
            if name in self.__modules.keys():
                self.__modules.pop(name)

        def replace_module(self, name, new_module):
            """
            replaces predefined modules (like reflective or transmissive layers,
            a discontinuity in refractive index, observers ...) of the ice
            to the dictionary

            Parameters
            ----------
            name: string
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

            Parameters
            ----------
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

        def getValue(self, position): #name may not be changed because linked to c++ radiopropa module
            """
            returns the index of refraction at position for radiopropa tracer

            Parameters
            ----------
            position: radiopropa.Vector3d
                point

            Returns
            -------
            n: float
                index of refraction
            """
            pos = np.array([position.x, position.y, position.z]) / nu2rp_meter
            return self.__ice_model_nuradio.get_index_of_refraction(pos)

        def getGradient(self, position): #name may not be changed because linked to c++ radiopropa module
            """
            returns the gradient of index of refraction at position for radiopropa tracer

            Parameters
            ----------
            position: radiopropa.Vector3d
                point

            Returns
            -------
            n_nabla: radiopropa.Vector3d
                gradient of index of refraction at the point
            """
            pos = np.array([position.x, position.y, position.z]) / nu2rp_meter
            gradient = self.__ice_model_nuradio.get_gradient_of_index_of_refraction(pos) / nu2rp_meter
            return RP.Vector3d(*gradient)
