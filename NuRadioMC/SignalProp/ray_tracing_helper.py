"""
This class is used to optimize with Numba the computation time 
of long running ray-tracing computations.
All the methods are compiled upon first call of any item of the class,
and will remain available without any other required compilation trhoughout
the whole python run.
Since the compilation time can vary between 20 to 40 seconds, using this class
might be far less efficient for a low number of calculations.

"""
import numpy as np
try:
    from numba import jit, njit
    from numba.experimental import jitclass
    from numba import int32, float64, float32
    from numba import types


    numba_available = True
except ImportError:
    numba_available = False


if numba_available:
    spec = [
        ('n_ice', float64),               
        ('reflection', int32),
        ('z_0', float64),
        ('delta_n', float64),
        ('__b', float64)
    ]

@jitclass(spec)
class ray_tracing_helper_class():
    def __init__(self, n_ice, reflection, z_0, delta_n):
    
        self.n_ice = n_ice
        self.reflection = reflection
        self.z_0 = z_0
        self.delta_n = delta_n
        self.__b = 2 * self.n_ice
    """
        initialize 2D analytic ray tracing helper class

        Parameters
        ----------
        n_ice: float
            refractive index of the deep bulk ice
        reflection: int (default 0)
            the number of bottom reflections to consider
        z_0: float, NuRadio length units
            scale depth of the exponential
        delta_n:  float, NuRadio length units
            difference between n_ice and the refractive index
            of the snow at the surface

        """
            
    def get_C0_from_log(self, logC0):
        """
        transforms the fit parameter C_0 so that the likelihood looks better
        """
        return np.exp(logC0) + 1. / self.n_ice
    
    
    def get_delta_y(self, C_0, x1, x2, C0range=None, reflection=0, reflection_case=2):
        """
        calculates the difference in the y position between the analytic ray tracing path
        specified by C_0 at the position x2
        """
        C_0_first = C_0

        if C0range is None:
            C0range = [1. / self.n_ice, np.inf]
        else:
            C0range = [float(C0range[0]), float(C0range[1])]
        Corange_array = np.array(C0range ,  dtype=np.float64)
        if((C_0_first < Corange_array[0]) or(C_0_first > Corange_array[1])):
            return np.array([-np.inf])
        c = self.n_ice ** 2 - C_0 ** -2
        # we consider two cases here,
        # 1) the rays start rising -> the default case
        # 2) the rays start decreasing -> we need to find the position left of the start point that
        #    that has rising rays that go through the point x1
        if(reflection > 0 and reflection_case == 2):
            y_turn = self.get_y_turn(C_0_first, x1)
            dy = y_turn - x1[0]
            x1[0] = x1[0] - 2 * dy

        for i in range(reflection):
            # we take account reflections at the bottom layer into account via
            # 1) calculating the point where the reflection happens
            # 2) starting a ray tracing from this new point

            # determine y translation first
            C_1 = x1[0] - self.get_y_with_z_mirror(x1[1], C_0_first)
            if(hasattr(C_1, '__len__')):
                C_1 = C_1[0]

            x1 = self.get_reflection_point(C_0, C_1)

        # determine y translation first
        C_1 = x1[0] - self.get_y_with_z_mirror(x1[1], C_0_first)
        if(hasattr(C_1, '__len__')):
            C_1 = C_1[0]

        # for a given c_0, 3 cases are possible to reach the y position of x2
        # 1) direct ray, i.e., before the turning point
        # 2) refracted ray, i.e. after the turning point but not touching the surface
        # 3) reflected ray, i.e. after the ray reaches the surface
        gamma_turn, z_turn = self.get_turning_point(c)
        y_turn = self.get_y(gamma_turn, C_0_first, C_1)
        if(z_turn < x2[1]):  # turning points is deeper that x2 positions, can't reach target
            # the minimizer has problems finding the minimum if inf is returned here. Therefore, we return the distance
            # between the turning point and the target point + 10 x the distance between the z position of the turning points
            # and the target position. This results in a objective function that has the solutions as the only minima and
            # is smooth in C_0
            diff = ((z_turn - x2[1]) ** 2 + (y_turn - x2[0]) ** 2) ** 0.5 + 10 * np.abs(z_turn - x2[1])
            return -diff
#             return -np.inf
        if(y_turn > x2[0]):  # we always propagate from left to right
            # direct ray
            y2_fit = self.get_y(self.get_gamma(x2[1]), C_0_first, C_1)  # calculate y position at get_path position
            diff = (x2[0] - y2_fit)
            if(hasattr(diff, '__len__')):
                diff = diff[0]
            if(hasattr(x2[0], '__len__')):
                x2[0] = x2[0][0]

            return diff
        else:
            # now it's a bit more complicated. we need to transform the coordinates to
            # be on the mirrored part of the function
            z_mirrored = x2[1]
            gamma = self.get_gamma(z_mirrored)
            y2_raw = self.get_y(gamma, C_0_first, C_1)
            y2_fit = 2 * y_turn - y2_raw
            diff = (x2[0] - y2_fit)

            return -1 * diff

    
    def obj_delta_y_square(self, logC_0, x1, x2, reflection=0, reflection_case=2):
        """
        objective function to find solution for C0
        """
        C_0 = self.get_C0_from_log(logC_0[0])
        return self.get_delta_y(C_0, x1, x2, None, reflection=reflection, reflection_case=reflection_case) ** 2
    
    def get_y_turn(self, C_0, x1):
        """
        calculates the y-coordinate of the turning point. This is either the point of reflection off the ice surface
        or the point where the saddle point of the ray (transition from upward to downward going)

        Parameters
        ----------
        C_0: float
            C_0 parameter of function
        x1: typle
            (y, z) start position of ray
        """
        c = self.n_ice ** 2 - C_0 ** -2
        gamma_turn, z_turn = self.get_turning_point(c)
        C_1_temp ,  = self.get_y_with_z_mirror(x1[1], C_0)
        C_1 = x1[0] - C_1_temp
        y_turn = self.get_y(gamma_turn[0], C_0, C_1) 
        return y_turn
    
    def get_y_with_z_mirror(self, z, C_0, C_1=0.0):
        """
        analytic form of the ray tracing part given an exponential index of refraction profile

        this function automatically mirrors z values that are above the turning point,
        so that this function is defined for all z

        Parameters
        ----------
        z: (float or array)
            depth z
        C_0: (float)
            first parameter
        C_1: (float)
            second parameter
        """
        c = self.n_ice ** 2 - C_0 ** -2
        gamma_turn, z_turn = self.get_turning_point(c)
        y_turn = self.get_y(gamma_turn, C_0, C_1)
        if(not hasattr(z, '__len__')):
            if(z < z_turn):
                gamma = self.get_gamma(np.array([1])*z)
                return self.get_y(gamma, C_0, C_1)
            else:
                gamma = self.get_gamma(2 * z_turn - z)
                return 2 * y_turn - self.get_y(gamma, C_0, C_1)
        else:
            mask = z < z_turn
            res = np.zeros_like(z)
            zs = np.zeros_like(z)
            gamma = self.get_gamma(z[mask])
            zs[mask] = z[mask]
            res[mask] = self.get_y(gamma, C_0, C_1)
            gamma = self.get_gamma(2 * z_turn - z[~mask])
            res[~mask] = 2 * y_turn - self.get_y(gamma, C_0, C_1)
            zs[~mask] = 2 * z_turn - z[~mask]

            return res, zs
    def get_turning_point(self, c):
        """
        calculate the turning point, i.e. the maximum of the ray tracing path;
        parameter is c = self.medium.n_ice ** 2 - C_0 ** -2

        This is either the point of reflection off the ice surface
        or the point where the saddle point of the ray (transition from upward to downward going)

        Technically, the turning point is set to z=0 if the saddle point is above the surface.

        Parameters
        ----------
        c: float
            related to C_0 parameter via c = self.medium.n_ice ** 2 - C_0 ** -2

        Returns
        -------
        typle (gamma, z coordinate of turning point)
        """
        gamma2 = np.array([self.__b * 0.5 - (0.25 * self.__b ** 2 - c) ** 0.5])  # first solution discarded
        z2 = np.log(gamma2 / self.delta_n) * self.z_0
        if(z2 > 0):
            z2 = np.array([0], dtype=np.float64)  # a reflection is just a turning point at z = 0, i.e. cases 2) and 3) are the same
            gamma2 = self.get_gamma(z2)

        return gamma2  , z2

    def get_y(self, gamma, C_0, C_1):
        """
        analytic form of the ray tracing part given an exponential index of refraction profile

        Parameters
        ----------
        gamma: (float or array)
            gamma is a function of the depth z
        C_0: (float)
            first parameter
        C_1: (float)
            second parameter
        """
        c = self.n_ice ** 2 - C_0 ** -2
        # we take the absolute number here but we only evaluate the equation for
        # positive outcome. This is to prevent rounding errors making the root
        # negative
        root = np.abs(gamma ** 2 - gamma * self.__b + c)
        logargument = gamma / (2 * c ** 0.5 * (root) ** 0.5 - self.__b * gamma + 2 * c)
        result = self.z_0 * (self.n_ice ** 2 * C_0 ** 2 - 1) ** -0.5 * np.log(logargument) + C_1
        return result
    
    def get_gamma(self, z):
        """
        transforms z coordinate into gamma
        """
        return self.delta_n * (np.exp(z / self.z_0))
    
    def get_reflection_point(self, C_0, C_1):
        """
        calculates the point where the signal gets reflected off the bottom of the ice shelf

        Returns tuple (y,z)
        """
        c = self.n_ice ** 2 - C_0 ** -2
        _gamma_turn, z_turn = self.get_turning_point(c)
        x2 = np.array([0, self.reflection],dtype = np.float64)
        x2[0] ,  = self.get_y_with_z_mirror(-x2[1] + 2 * z_turn, C_0, C_1)
        return x2