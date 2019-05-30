from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.hatch import get_path
import time
import copy
from scipy.optimize import fsolve, minimize, basinhopping, root
from scipy import optimize, integrate, interpolate
import scipy.constants
from operator import itemgetter

from NuRadioMC.utilities import units
from NuRadioMC.utilities import attenuation as attenuation_util

import logging
logging.basicConfig()

# check if CPP implementation is available
cpp_available = False
try:
    from NuRadioMC.SignalProp.CPPAnalyticRayTracing import wrapper
    cpp_available = True
    print("using CPP version of ray tracer")
except:
    print("trying to compile the CPP extension on-the-fly")
    try:
        import subprocess
        import os
        subprocess.call(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "install.sh"))
        from NuRadioMC.SignalProp.CPPAnalyticRayTracing import wrapper
        cpp_available = True
        print("compilation was sucessful, using CPP version of ray tracer")
    except:
        print("compilation was not sucessful, using python version of ray tracer")
        cpp_available = False

"""
analytic ray tracing solution
"""
speed_of_light = scipy.constants.c * units.m / units.s

solution_types = {1: 'direct',
                  2: 'refracted',
                  3: 'reflected'}


class ray_tracing_2D():

    def __init__(self, medium, attenuation_model="SP1",
                 log_level=logging.WARNING,
                 n_frequencies_integration=25,
                 use_optimized_start_values=False):
        """
        initialize 2D analytic ray tracing class
        
        Parameters
        ----------
        medium: NuRadioMC.utilities.medium class
            details of the medium
        attenuation_model: string
            specifies which attenuation model to use (default 'SP1')
        log_level: logging.loglevel object
            controls verbosity (default WARNING)
        n_frequencies_integration: int
            specifies for how many frequencies the signal attenuation is being calculated
        use_optimized_start_value: bool
            if True, the initial C_0 paramter (launch angle) is set to the ray that skims the surface
            (default: False)
        
        """
        self.medium = medium
        self.attenuation_model = attenuation_model
        if(not self.attenuation_model in attenuation_util.model_to_int):
            raise NotImplementedError("attenuation model {} is not implemented".format(self.attenuation_model))
        self.attenuation_model_int = attenuation_util.model_to_int[self.attenuation_model]
        self.__b = 2 * self.medium.n_ice
        self.__logger = logging.getLogger('ray_tracing_2D')
        self.__logger.setLevel(log_level)
        self.__n_frequencies_integration = n_frequencies_integration
        self.__use_optimized_start_values = use_optimized_start_values

    def n(self, z):
        """
        refractive index as a function of depth
        """
        res = self.medium.n_ice - self.medium.delta_n * np.exp(z / self.medium.z_0)
    #     if(type(z) is float):
    #         if(z > 0):
    #             return 1.
    #         else:
    #             return res
    #     else:
    #         res[z > 0] = 1.
        return res

    def get_gamma(self, z):
        return self.medium.delta_n * np.exp(z / self.medium.z_0)

    def get_turning_point(self, c):
        """
        calculate the turning point, i.e. the maximum of the ray tracing path;
        parameter is c = self.medium.n_ice ** 2 - C_0 ** -2
        """
        gamma2 = self.__b * 0.5 - (0.25 * self.__b ** 2 - c) ** 0.5  # first solution discarded
        z2 = np.log(gamma2 / self.medium.delta_n) * self.medium.z_0

        return gamma2, z2

    def get_C_1(self, x1, C_0):
        """
        calculates constant C_1 for a given C_0 and start point x1
        """
        return x1[0] - self.get_y_with_z_mirror(x1[1], C_0)

    def get_c(self, C_0):
        return self.medium.n_ice ** 2 - C_0 ** -2

    def get_C0_from_log(self, logC0):
        """
        transforms the fit parameter C_0 so that the likelihood looks better
        """
        return np.exp(logC0) + 1. / self.medium.n_ice

    def get_y(self, gamma, C_0, C_1):
        """
        analytic form of the ray tracing part given an exponential index of refraction profile

        Parameters
        -------
        gamma: (float or array)
            gamma is a function of the depth z
        C_0: (float)
            first parameter
        C_1: (float)
            second parameter
        """
        c = self.medium.n_ice ** 2 - C_0 ** -2
        # we take the absolute number here but we only evaluate the equation for
        # positive outcome. This is to prevent rounding errors making the root
        # negative
        root = np.abs(gamma ** 2 - gamma * self.__b + c)
        logargument = gamma / (2 * c ** 0.5 * (root) ** 0.5 - self.__b * gamma + 2 * c)
        if(np.sum(logargument <= 0)):
            self.__logger.debug('log = {}'.format(logargument))
        result = self.medium.z_0 * (self.medium.n_ice ** 2 * C_0 ** 2 - 1) ** -0.5 * np.log(logargument) + C_1
        return result

    def get_y_diff(self, z_raw, C_0):
        """
        derivative dy(z)/dz
        """
        z = self.get_z_unmirrored(z_raw, C_0)
        c = self.medium.n_ice ** 2 - C_0 ** -2
        B = (0.2e1 * np.sqrt(c) * np.sqrt(-self.__b * self.medium.delta_n * np.exp(z / self.medium.z_0) + self.medium.delta_n ** 
                                          2 * np.exp(0.2e1 * z / self.medium.z_0) + c) - self.__b * self.medium.delta_n * np.exp(z / self.medium.z_0) + 0.2e1 * c)
        D = self.medium.n_ice ** 2 * C_0 ** 2 - 1
        E1 = -self.__b * self.medium.delta_n * np.exp(z / self.medium.z_0)
        E2 = self.medium.delta_n ** 2 * np.exp(0.2e1 * z / self.medium.z_0)
        E = (E1 + E2 + c)
        res = (-np.sqrt(c) * np.exp(z / self.medium.z_0) * self.__b * self.medium.delta_n + 0.2e1 * np.sqrt(-self.__b * self.medium.delta_n * np.exp(z / 
                                                                                                                                                     self.medium.z_0) + self.medium.delta_n ** 2 * np.exp(0.2e1 * z / self.medium.z_0) + c) * c + 0.2e1 * c ** 1.5) / B * E ** -0.5 * (D ** (-0.5))

        if(z != z_raw):
            res *= -1
        return res

    def get_y_with_z_mirror(self, z, C_0, C_1=0):
        """
        analytic form of the ray tracing part given an exponential index of refraction profile

        this function automatically mirrors z values that are above the turning point,
        so that this function is defined for all z

        Parameters
        -------
        z: (float or array)
            depth z
        C_0: (float)
            first parameter
        C_1: (float)
            second parameter
        """
        c = self.medium.n_ice ** 2 - C_0 ** -2
        gamma_turn, z_turn = self.get_turning_point(c)
        if(z_turn >= 0):
            # signal reflected at surface
            self.__logger.debug('signal reflects off surface')
            z_turn = 0
            gamma_turn = self.get_gamma(0)
        y_turn = self.get_y(gamma_turn, C_0, C_1)
        if(not hasattr(z, '__len__')):
            if(z < z_turn):
                gamma = self.get_gamma(z)
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

            self.__logger.debug('turning points for C_0 = {:.2f}, b= {:.2f}, gamma = {:.4f}, z = {:.1f}, y_turn = {:.0f}'.format(
                C_0, self.__b, gamma_turn, z_turn, y_turn))
            return res, zs

    def get_z_mirrored(self, x1, x2, C_0):
        """
        calculates the mirrored x2 position so that y(z) can be used as a continuous function
        """
        c = self.medium.n_ice ** 2 - C_0 ** -2
        C_1 = x1[0] - self.get_y_with_z_mirror(x1[1], C_0)
        gamma_turn, z_turn = self.get_turning_point(c)
        if(z_turn >= 0):
            # signal reflected at surface
            self.__logger.debug('signal reflects off surface')
            z_turn = 0
            gamma_turn = self.get_gamma(0)
        y_turn = self.get_y(gamma_turn, C_0, C_1)
        zstart = x1[1]
        zstop = x2[1]
        if(y_turn < x2[0]):
            zstop = zstart + np.abs(z_turn - x1[1]) + np.abs(z_turn - x2[1])
        x2_mirrored = [x2[0], zstop]
        return x2_mirrored

    def get_z_unmirrored(self, z, C_0):
        """
        calculates the unmirrored z position
        """
        c = self.medium.n_ice ** 2 - C_0 ** -2
        gamma_turn, z_turn = self.get_turning_point(c)
        if(z_turn >= 0):
            # signal reflected at surface
            self.__logger.debug('signal reflects off surface')
            z_turn = 0

        z_unmirrored = z
        if(z > z_turn):
            z_unmirrored = 2 * z_turn - z
        return z_unmirrored

    def ds(self, t, C_0):
        """
        helper to calculate line integral
        """
        return (self.get_y_diff(t, C_0) ** 2 + 1) ** 0.5

    def get_path_length(self, x1, x2, C_0):
        x2_mirrored = self.get_z_mirrored(x1, x2, C_0)
        gamma_turn, z_turn = self.get_turning_point(self.medium.n_ice ** 2 - C_0 ** -2)
        points = None
        if(x1[1] < z_turn and z_turn < x2_mirrored[1]):
            points = [z_turn]
        path_length = integrate.quad(self.ds, x1[1], x2_mirrored[1], args=(C_0), points=points)
        self.__logger.info("calculating path length from ({:.0f}, {:.0f}) to ({:.0f}, {:.0f}) = ({:.0f}, {:.0f}) = {:.2f} m".format(x1[0], x1[1], x2[0], x2[1],
                                                                                                                                    x2_mirrored[0],
                                                                                                                                    x2_mirrored[1],
                                                                                                                                    path_length[0] / units.m))
#         print('numeric {:.2f}'.format(path_length[0]))
        return path_length[0]

    def get_path_length_analytic(self, x1, x2, C_0):
        """
        analytic solution to calculate the distance along the path. This code is based on the analytic solution found
        by Ben Hokanson-Fasing and the pyrex implementation.
        """
        solution_type = self.determine_solution_type(x1, x2, C_0)

        z_deep = -500 * units.m
        launch_angle = self.get_launch_angle(x1, C_0)
        beta = self.n(x1[1]) * np.sin(launch_angle)
        alpha = self.medium.n_ice ** 2 - beta ** 2
#         print("launchangle {:.1f} beta {:.2g} alpha {:.2g}, n(z1) = {:.2g} n(z2) = {:.2g}".format(launch_angle/units.deg, beta, alpha, self.n(x1[1]), self.n(x2[1])))

        def l1(z):
            gamma = self.n(z) ** 2 - beta ** 2
            gamma = np.where(gamma < 0, 0, gamma)
            return self.medium.n_ice * self.n(z) - beta ** 2 - (alpha * gamma) ** 0.5

        def l2(z):
            gamma = self.n(z) ** 2 - beta ** 2
            gamma = np.where(gamma < 0, 0, gamma)
            return self.n(z) + gamma ** 0.5

        def get_s(z, deep=False):
            if(deep):
                return self.medium.n_ice * z / alpha ** 0.5
            else:
                #                 print(z, self.n(z), beta)
                #                 print(alpha**0.5, l1(z), l2(z))
                return self.medium.n_ice / alpha ** 0.5 * (-z + np.log(l1(z)) * self.medium.z_0) + np.log(l2(z)) * self.medium.z_0

        def get_path_direct(z1, z2):
            int1 = get_s(z1, z1 < z_deep)
            int2 = get_s(z2, z2 < z_deep)
#             print('analytic {:.4g} ({:.0f} - {:.0f}={:.4g}, {:.4g})'.format(
#                 int2 - int1, get_s(x2[1]), x1[1], x2[1], get_s(x1[1])))
            if (z1 < z_deep) == (z2 < z_deep):
                # z0 and z1 on same side of z_deep
                return int2 - int1
            else:
                int_diff = get_s(z_deep, deep=True) - get_s(z_deep, deep=False)
                if z1 < z2:
                    # z0 below z_deep, z1 above z_deep
                    return int2 - int1 + int_diff
                else:
                    # z0 above z_deep, z1 below z_deep
                    return int2 - int1 - int_diff

        if(solution_type == 1):
            return get_path_direct(x1[1], x2[1])
        else:
            if(solution_type == 3):
                z_turn = 0
            else:
                gamma_turn, z_turn = self.get_turning_point(self.medium.n_ice ** 2 - C_0 ** -2)
#             print('solution type {:d}, zturn = {:.1f}'.format(solution_type, z_turn))
            return get_path_direct(x1[1], z_turn) + get_path_direct(x2[1], z_turn)

    def get_travel_time(self, x1, x2, C_0):
        x2_mirrored = self.get_z_mirrored(x1, x2, C_0)

        def dt(t, C_0):
            z = self.get_z_unmirrored(t, C_0)
            return self.ds(t, C_0) / speed_of_light * self.n(z)

        gamma_turn, z_turn = self.get_turning_point(self.medium.n_ice ** 2 - C_0 ** -2)
        points = None
        if(x1[1] < z_turn and z_turn < x2_mirrored[1]):
            points = [z_turn]
        travel_time = integrate.quad(dt, x1[1], x2_mirrored[1], args=(C_0), points=points)
        self.__logger.info("calculating travel time from ({:.0f}, {:.0f}) to ({:.0f}, {:.0f}) = ({:.0f}, {:.0f}) = {:.2f} ns".format(
            x1[0], x1[1], x2[0], x2[1], x2_mirrored[0], x2_mirrored[1], travel_time[0] / units.ns))
        return travel_time[0]

    def get_travel_time_analytic(self, x1, x2, C_0):
        """
        analytic solution to calculate the time of flight. This code is based on the analytic solution found
        by Ben Hokanson-Fasing and the pyrex implementation.
        """
        solution_type = self.determine_solution_type(x1, x2, C_0)

        z_deep = -500 * units.m
        launch_angle = self.get_launch_angle(x1, C_0)
        beta = self.n(x1[1]) * np.sin(launch_angle)
        alpha = self.medium.n_ice ** 2 - beta ** 2
#         print("launchangle {:.1f} beta {:.2g} alpha {:.2g}, n(z1) = {:.2g} n(z2) = {:.2g}".format(launch_angle/units.deg, beta, alpha, self.n(x1[1]), self.n(x2[1])))

        def l1(z):
            gamma = self.n(z) ** 2 - beta ** 2
            gamma = np.where(gamma < 0, 0, gamma)
            return self.medium.n_ice * self.n(z) - beta ** 2 - (alpha * gamma) ** 0.5

        def l2(z):
            gamma = self.n(z) ** 2 - beta ** 2
            gamma = np.where(gamma < 0, 0, gamma)
            return self.n(z) + gamma ** 0.5

        def get_s(z, deep=False):
            if(deep):
                return self.medium.n_ice * (self.n(z) + self.medium.n_ice * (z / self.medium.z_0 - 1)) / (np.sqrt(alpha) / self.medium.z_0 * speed_of_light)
            else:
                gamma = self.n(z) ** 2 - beta ** 2
                gamma = np.where(gamma < 0, 0, gamma)
                log_1 = l1(z)
                log_2 = l2(z)
                return (((np.sqrt(gamma) + self.medium.n_ice * np.log(log_2) + 
                          self.medium.n_ice ** 2 * np.log(log_1) / np.sqrt(alpha)) * self.medium.z_0) - 
                        z * self.medium.n_ice ** 2 / np.sqrt(alpha)) / speed_of_light

        def get_ToF_direct(z1, z2):
            int1 = get_s(z1, z1 < z_deep)
            int2 = get_s(z2, z2 < z_deep)
#             print('analytic {:.4g} ({:.0f} - {:.0f}={:.4g}, {:.4g})'.format(
#                 int2 - int1, get_s(x2[1]), x1[1], x2[1], get_s(x1[1])))
            if (z1 < z_deep) == (z2 < z_deep):
                # z0 and z1 on same side of z_deep
                return int2 - int1
            else:
                int_diff = get_s(z_deep, deep=True) - get_s(z_deep, deep=False)
                if z1 < z2:
                    # z0 below z_deep, z1 above z_deep
                    return int2 - int1 + int_diff
                else:
                    # z0 above z_deep, z1 below z_deep
                    return int2 - int1 - int_diff

        if(solution_type == 1):
            return get_ToF_direct(x1[1], x2[1])
        else:
            if(solution_type == 3):
                z_turn = 0
            else:
                gamma_turn, z_turn = self.get_turning_point(self.medium.n_ice ** 2 - C_0 ** -2)
#             print('solution type {:d}, zturn = {:.1f}'.format(solution_type, z_turn))
            return get_ToF_direct(x1[1], z_turn) + get_ToF_direct(x2[1], z_turn)

    
    def __get_frequencies_for_attenuation(self, frequency, max_detector_freq):
            mask = frequency > 0
            nfreqs = min(self.__n_frequencies_integration, np.sum(mask))
            freqs = np.linspace(frequency[mask].min(), frequency[mask].max(), nfreqs)
            if(nfreqs < np.sum(mask) and max_detector_freq is not None):
                mask2 = frequency <= max_detector_freq
                nfreqs2 = min(self.__n_frequencies_integration, np.sum(mask2 & mask))
                freqs = np.linspace(frequency[mask2 & mask].min(), frequency[mask2 & mask].max(), nfreqs2)
                if(np.sum(~mask2) > 1):
                    freqs = np.append(freqs, np.linspace(frequency[~mask2].min(), frequency[~mask2].max(), nfreqs // 2))
            return freqs

    def get_attenuation_along_path(self, x1, x2, C_0, frequency, max_detector_freq):
        if(cpp_available):
            mask = frequency > 0
            freqs = self.__get_frequencies_for_attenuation(frequency, max_detector_freq)
            tmp = np.zeros_like(freqs)
            for i, f in enumerate(freqs):
                tmp[i] = wrapper.get_attenuation_along_path(
                    x1, x2, C_0, f, self.medium.n_ice, self.medium.delta_n, self.medium.z_0, self.attenuation_model_int)

            attenuation = np.ones_like(frequency)
            attenuation[mask] = np.interp(frequency[mask], freqs, tmp)
            return attenuation
        else:

            x2_mirrored = self.get_z_mirrored(x1, x2, C_0)

            def dt(t, C_0, frequency):
                z = self.get_z_unmirrored(t, C_0)
                return self.ds(t, C_0) / attenuation_util.get_attenuation_length(z, frequency, self.attenuation_model)

            # to speed up things we only calculate the attenuation for a few frequencies
            # and interpolate linearly between them
            mask = frequency > 0
            freqs = self.__get_frequencies_for_attenuation(frequency, max_detector_freq)
            gamma_turn, z_turn = self.get_turning_point(self.medium.n_ice ** 2 - C_0 ** -2)
            points = None
            if(x1[1] < z_turn and z_turn < x2_mirrored[1]):
                points = [z_turn]
            tmp = np.array([integrate.quad(dt, x1[1], x2_mirrored[1], args=(
                C_0, f), epsrel=5e-2, points=points)[0] for f in freqs])
            att_func = interpolate.interp1d(freqs, tmp)
            tmp2 = att_func(frequency[mask])
    #         tmp = np.array([integrate.quad(dt, x1[1], x2_mirrored[1], args=(C_0, f), epsrel=0.05)[0] for f in frequency[mask]])
            attenuation = np.ones_like(frequency)
            attenuation[mask] = np.exp(-1 * tmp2)
            self.__logger.info("calculating attenuation from ({:.0f}, {:.0f}) to ({:.0f}, {:.0f}) = ({:.0f}, {:.0f}) =  a factor {}".format(
                x1[0], x1[1], x2[0], x2[1], x2_mirrored[0], x2_mirrored[1], 1 / attenuation))
            return attenuation

    def get_angle(self, x, x_start, C_0):
        z = self.get_z_mirrored(x_start, x, C_0)[1]
        dy = self.get_y_diff(z, C_0)
        angle = np.arctan(dy)
        if(angle < 0):
            angle = np.pi + angle
        return angle

    def get_launch_angle(self, x1, C_0):
        return self.get_angle(x1, x1, C_0)

    def get_receive_angle(self, x1, x2, C_0):
        return np.pi - self.get_angle(x2, x1, C_0)

    def get_reflection_angle(self, x1, C_0):
        c = self.medium.n_ice ** 2 - C_0 ** -2
        C_1 = x1[0] - self.get_y_with_z_mirror(x1[1], C_0)
        gamma_turn, z_turn = self.get_turning_point(c)
        if(z_turn >= 0):
            gamma_turn = self.get_gamma(0)
            y_turn = self.get_y(gamma_turn, C_0, C_1)
            r = self.get_angle(np.array([y_turn, 0]), x1, C_0)
            self.__logger.debug(
                "reflecting off surface at y = {:.1f}m, reflection angle = {:.1f}deg".format(y_turn, r / units.deg))
            return r
        else:
            return None

    def get_path(self, x1, x2, C_0, n_points=1000):
        """
        for plotting purposes only, returns the ray tracing path between x1 and x2

        the result is only valid if C_0 is a solution to the ray tracing problem

        Parameters
        -------
        x1: array
            start position (y, z)
        x2: array
            stop position (y, z)
        C_0: (float)
            first parameter
        n_points: integer (optional)
            the number of coordinates to calculate

        Returns
        -------
        yy: array
            the y coordinates of the ray tracing path
        zz: array
            the z coordinates of the ray tracing path
        """
        c = self.medium.n_ice ** 2 - C_0 ** -2
        C_1 = x1[0] - self.get_y_with_z_mirror(x1[1], C_0)
        gamma_turn, z_turn = self.get_turning_point(c)
        if(z_turn >= 0):
            # signal reflected at surface
            self.__logger.debug('signal reflects off surface')
            z_turn = 0
            gamma_turn = self.get_gamma(0)
        y_turn = self.get_y(gamma_turn, C_0, C_1)
        zstart = x1[1]
        zstop = self.get_z_mirrored(x1, x2, C_0)[1]
        z = np.linspace(zstart, zstop, n_points)
        mask = z < z_turn
        res = np.zeros_like(z)
        zs = np.zeros_like(z)
        gamma = self.get_gamma(z[mask])
        zs[mask] = z[mask]
        res[mask] = self.get_y(gamma, C_0, C_1)
        gamma = self.get_gamma(2 * z_turn - z[~mask])
        res[~mask] = 2 * y_turn - self.get_y(gamma, C_0, C_1)
        zs[~mask] = 2 * z_turn - z[~mask]

        self.__logger.debug('turning points for C_0 = {:.2f}, b= {:.2f}, gamma = {:.4f}, z = {:.1f}, y_turn = {:.0f}'.format(
            C_0, self.__b, gamma_turn, z_turn, y_turn))
        return res, zs

    def obj_delta_y_square(self, logC_0, x1, x2):
        """
        objective function to find solution for C0
        """
        C_0 = self.get_C0_from_log(logC_0)
        return self.get_delta_y(C_0, x1, x2) ** 2

    def obj_delta_y(self, logC_0, x1, x2):
        """
        function to find solution for C0, returns distance in y between function and x2 position
        result is signed! (important to use a root finder)
        """
        C_0 = self.get_C0_from_log(logC_0)
        return self.get_delta_y(C_0, x1, x2)

    def get_delta_y(self, C_0, x1, x2, C0range=None):
        """
        calculates the difference in the y position between the analytic ray tracing path
        specified by C_0 at the position x2
        """
        if(C0range is None):
            C0range = [1. / self.medium.n_ice, np.inf]
        if(hasattr(C_0, '__len__')):
            C_0 = C_0[0]
        if((C_0 < C0range[0]) or(C_0 > C0range[1])):
            self.__logger.debug('C0 = {:.4f} out of range {:.0f} - {:.2f}'.format(C_0, C0range[0], C0range[1]))
            return -np.inf
        c = self.medium.n_ice ** 2 - C_0 ** -2
        # determine y translation first
        C_1 = x1[0] - self.get_y_with_z_mirror(x1[1], C_0)
        if(hasattr(C_1, '__len__')):
            C_1 = C_1[0]

        self.__logger.debug("C_0 = {:.4f}, C_1 = {:.1f}".format(C_0, C_1))

        # for a given c_0, 3 cases are possible to reach the y position of x2
        # 1) direct ray, i.e., before the turning point
        # 2) refracted ray, i.e. after the turning point but not touching the surface
        # 3) reflected ray, i.e. after the ray reaches the surface
        gamma_turn, z_turn = self.get_turning_point(c)
        if(z_turn > 0):
            z_turn = 0  # a reflection is just a turning point at z = 0, i.e. cases 2) and 3) are the same
            gamma_turn = self.get_gamma(z_turn)
        y_turn = self.get_y(gamma_turn, C_0, C_1)
        if(z_turn < x2[1]):  # turning points is deeper that x2 positions, can't reach target
            # the minimizer has problems finding the minimum if inf is returned here. Therefore, we return the distance
            # between the turning point and the target point + 10 x the distance between the z position of the turning points
            # and the target position. This results in a objective function that has the solutions as the only minima and 
            # is smooth in C_0 
            diff = ((z_turn - x2[1])**2 + (y_turn - x2[0])**2)**0.5 + 10 * np.abs(z_turn - x2[1])
            self.__logger.debug(
                "turning points (zturn = {:.0f} is deeper than x2 positon z2 = {:.0f}, setting distance to target position to {:.1f}".format(z_turn, x2[1], -diff))
            return -diff
#             return -np.inf
        self.__logger.debug('turning points is z = {:.1f}, y =  {:.1f}'.format(z_turn, y_turn))
        if(y_turn > x2[0]):  # we always propagate from left to right
            # direct ray
            y2_fit = self.get_y(self.get_gamma(x2[1]), C_0, C_1)  # calculate y position at get_path position
            diff = (x2[0] - y2_fit)
            if(hasattr(diff, '__len__')):
                diff = diff[0]
            if(hasattr(x2[0], '__len__')):
                x2[0] = x2[0][0]

            self.__logger.debug(
                'we have a direct ray, y({:.1f}) = {:.1f} -> {:.1f} away from {:.1f}, turning point = y={:.1f}, z={:.2f}, x0 = {:.1f} {:.1f}'.format(x2[1], y2_fit, diff, x2[0], y_turn, z_turn, x1[0], x1[1]))
            return diff
        else:
            # now it's a bit more complicated. we need to transform the coordinates to
            # be on the mirrored part of the function
            z_mirrored = x2[1]
            gamma = self.get_gamma(z_mirrored)
            self.__logger.debug("get_y( {}, {}, {})".format(gamma, C_0, C_1))
            y2_raw = self.get_y(gamma, C_0, C_1)
            y2_fit = 2 * y_turn - y2_raw
            diff = (x2[0] - y2_fit)
            
            self.__logger.debug('we have a reflected/refracted ray, y({:.1f}) = {:.1f} ({:.1f}) -> {:.1f} away from {:.1f} (gamma = {:.5g})'.format(
                z_mirrored, y2_fit, y2_raw, diff, x2[0], gamma))
            return -1 * diff

    def determine_solution_type(self, x1, x2, C_0):
        """ returns the type of the solution

        Parameters
        ----------
        x1: 2dim np.array
            start position
        x2: 2dim np.array
            stop position
        C_0: float
            C_0 value of ray tracing solution

        Returns
        -------
        solution_type: int
            * 1: 'direct'
            * 2: 'refracted'
            * 3: 'reflected
        """
        c = self.medium.n_ice ** 2 - C_0 ** -2
        C_1 = x1[0] - self.get_y_with_z_mirror(x1[1], C_0)
        gamma_turn, z_turn = self.get_turning_point(c)

        if(z_turn >= 0):
            z_turn = 0
            gamma_turn = self.get_gamma(0)
        y_turn = self.get_y(gamma_turn, C_0, C_1)
        if(x2[0] < y_turn):
            return 1
        else:
            if(z_turn == 0):
                return 3
            else:
                return 2

    def find_solutions(self, x1, x2, plot=False):
        """
        this function finds all ray tracing solutions

        prerequesite is that x2 is above and to the right of x1, this is not a violation of universality
        because this requirement can be achieved with a simple coordinate transformation

        returns an array of the C_0 paramters of the solutions (the array might be empty)
        """

        if(cpp_available):
            #             t = time.time()
            solutions = wrapper.find_solutions(x1, x2, self.medium.n_ice, self.medium.delta_n, self.medium.z_0)
#             print((time.time() -t)*1000.)
            return solutions
        else:

            tol = 1e-6
            results = []
            C0s = []  # intermediate storage of results

            # calculate optimal start value. The objective function becomes infinity if the turning point is below the z
            # position of the observer. We calculate the corresponding value so that the minimization starts at one edge
            # of the objective function
            # c = self.__b ** 2 / 4 - (0.5 * self.__b - np.exp(x2[1] / self.medium.z_0) * self.medium.n_ice) ** 2
            # C_0_start = (1 / (self.medium.n_ice ** 2 - c)) ** 0.5
            # R.L. March 15, 2019: This initial condition does not find a solution for e.g.:
            # emitter  at [-400.0*units.m,-732.0*units.m], receiver at [0., -2.0*units.m]

            if(self.__use_optimized_start_values):
                # take surface skimming ray as start value
                C_0_start, th_start = self.get_surf_skim_angle(x1)
                logC_0_start = np.log(C_0_start - 1. / self.medium.n_ice)
                self.__logger.debug(
                    'starting optimization with x0 = {:.2f} -> C0 = {:.3f}'.format(logC_0_start, C_0_start))
            else:
                logC_0_start = -1

            result = optimize.root(self.obj_delta_y_square, x0=logC_0_start, args=(x1, x2), tol=tol)

            if(plot):
                fig, ax = plt.subplots(1, 1)
            if(result.fun < 1e-7):
                if(plot):
                    self.plot_result(x1, x2, self.get_C0_from_log(result.x[0]), ax)
                if(np.round(result.x[0], 3) not in np.round(C0s, 3)):
                    C_0 = self.get_C0_from_log(result.x[0])
                    C0s.append(C_0)
                    solution_type = self.determine_solution_type(x1, x2, C_0)
                    self.__logger.info("found {} solution C0 = {:.2f}".format(solution_types[solution_type], C_0))
                    results.append({'type': solution_type,
                                    'C0': C_0,
                                    'C1': self.get_C_1(x1, C_0)})

            # check if another solution with higher logC0 exists
            logC0_start = result.x[0] + 0.0001
            logC0_stop = 100
            delta_start = self.obj_delta_y(logC0_start, x1, x2)
            delta_stop = self.obj_delta_y(logC0_stop, x1, x2)
        #     print(logC0_start, logC0_stop, delta_start, delta_stop, np.sign(delta_start), np.sign(delta_stop))
            if(np.sign(delta_start) != np.sign(delta_stop)):
                self.__logger.info("solution with logC0 > {:.3f} exists".format(result.x[0]))
                result2 = optimize.brentq(self.obj_delta_y, logC0_start, logC0_stop, args=(x1, x2))
                if(plot):
                    self.plot_result(x1, x2, self.get_C0_from_log(result2), ax)
                if(np.round(result2, 3) not in np.round(C0s, 3)):
                    C_0 = self.get_C0_from_log(result2)
                    C0s.append(C_0)
                    solution_type = self.determine_solution_type(x1, x2, C_0)
                    self.__logger.info("found {} solution C0 = {:.2f}".format(solution_types[solution_type], C_0))
                    results.append({'type': solution_type,
                                    'C0': C_0,
                                    'C1': self.get_C_1(x1, C_0)})
            else:
                self.__logger.info("no solution with logC0 > {:.3f} exists".format(result.x[0]))

            logC0_start = -100
            logC0_stop = result.x[0] - 0.0001
            delta_start = self.obj_delta_y(logC0_start, x1, x2)
            delta_stop = self.obj_delta_y(logC0_stop, x1, x2)
        #     print(logC0_start, logC0_stop, delta_start, delta_stop, np.sign(delta_start), np.sign(delta_stop))
            if(np.sign(delta_start) != np.sign(delta_stop)):
                self.__logger.info("solution with logC0 < {:.3f} exists".format(result.x[0]))
                result3 = optimize.brentq(self.obj_delta_y, logC0_start, logC0_stop, args=(x1, x2))

                if(plot):
                    self.plot_result(x1, x2, self.get_C0_from_log(result3), ax)
                if(np.round(result3, 3) not in np.round(C0s, 3)):
                    C_0 = self.get_C0_from_log(result3)
                    C0s.append(C_0)
                    solution_type = self.determine_solution_type(x1, x2, C_0)
                    self.__logger.info("found {} solution C0 = {:.2f}".format(solution_types[solution_type], C_0))
                    results.append({'type': solution_type,
                                    'C0': C_0,
                                    'C1': self.get_C_1(x1, C_0)})
            else:
                self.__logger.info("no solution with logC0 < {:.3f} exists".format(result.x[0]))

            if(plot):
                plt.show()

            return sorted(results, key=itemgetter('type'))

    def plot_result(self, x1, x2, C_0, ax):
        """
        helper function to visualize results
        """
        C_1 = self.get_C_1(x1, C_0)

        zs = np.linspace(x1[1], x1[1] + np.abs(x1[1]) + np.abs(x2[1]), 1000)
        yy, zz = self.get_y_with_z_mirror(zs, C_0, C_1)
        ax.plot(yy, zz, '-', label='C0 = {:.3f}'.format(C_0))
        ax.plot(x1[0], x1[1], 'ko')
        ax.plot(x2[0], x2[1], 'd')

    #     ax.plot(zz, yy, '-', label='C0 = {:.3f}'.format(C_0))
    #     ax.plot(x1[1], x1[0], 'ko')
    #     ax.plot(x2[1], x2[0], 'd')
        ax.legend()

#     def get_angle_from_C_0(self,C_0, z_pos,angoff=0):
#
#         '''
#         argument angoff is provided so that the function can be used for minimization in get_C_0_from_angle(),
#         in which case angoff is the angle for which the C_0 is sought and zero is returned when it is fouund.
#
#         output:
#             angle corresponding to C_0, minus offset angoff
#         '''
#
#
#         dydz = self.get_y_diff(z_pos,C_0)
# #        dydz = self.get_dydz_analytic(C_0, z_pos)
#
#         angle=np.arctan(dydz)
#
#         if(angle < 0):
#             angle = np.pi + angle
#         return angle - angoff

    def get_angle_from_logC_0(self, logC_0, z_pos, angoff=0):

        '''
        argument angoff is provided so that the function can be used for minimization in get_C_0_from_angle(),
        in which case angoff is the angle for which the C_0 is sought and zero is returned when it is fouund.

        C_0 has a smallest possible value at 1./self.medium.n_ice . When it approaches this value, very
        small changes in C_0 correspond to a given change in the angle. In order to prevent the root finding
        algorithm from crossing into the invalid range of C_0 at values smaller than 1./self.medium.n_ice,
        the root finding is done with the parameter logC_0 = np.log(C_0 - 1. / self.medium.n_ice), so it is
        not exactly the log of C_0 as the nome of this method implies.
        This is the same parameter transformation that is done for find_solutions()

        input:
            logC_0 = np.log(C_0 - 1. / self.medium.n_ice)
            angoff = angular offset
            z_pos  = z-position from where ray is emitted
        output:
            angle corresponding to C_0, minus offset angoff
        '''

        C_0 = self.get_C0_from_log(logC_0)

        dydz = self.get_y_diff(z_pos, C_0)
#        dydz = self.get_dydz_analytic(C_0, z_pos)
        angle = np.arctan(dydz)

        # print(dydz,angoffdydz)

        return angle - angoff

    def get_C_0_from_angle(self, anglaunch, z_pos):

        '''
        Find parameter C0 corresponding to a given launch angle and z-position of a ray.
        The parameter is found by means of a root finding procedure

        output:
            Complete output of optimisation procedure
            (result.x[0] is the C0 value found by optimisation procedure)

        '''

        # C_0 has a smallest possible value at 1./self.medium.n_ice . When it approaches this value, very
        # small changes in C_0 correspond to given change in the angle. In order to prevent the root finding
        # algorithm to cross into the invalid range of C_0 at  values smaller than 1./self.medium.n_ice,
        # the root finding is done with the parameter logC_0_start below. This is the same parameter transformation
        # that is done for find_solutions()

        C_0_start = 2.

        logC_0_start = np.log(C_0_start - 1. / self.medium.n_ice)

#        result = optimize.root(self.get_angle_from_C_0,np.pi/4.,args=(z_pos,anglaunch))
        result = optimize.root(self.get_angle_from_logC_0, logC_0_start, args=(z_pos, anglaunch))

        # want to return the complete instance of the result class; result value result.x[0] is logC_0,
        # but we want C_0, so replace it in the result class. This may not be good practice but it seems to be
        # more user-friendly than to return the value logC_0
        result.x[0] = copy.copy(self.get_C0_from_log(result.x[0]))

        return result

#     def get_dydz_analytic(self, C_0, z_pos):
#         '''
#         Implementation of derivative dy/dz obtained from the analytic expresion for y(z)
#         Returns dy/dz for a given z-position and C_0
#         '''
#
#         gamma = self.get_gamma(z_pos)
#
#         b = self.__b
#         c = self.medium.n_ice ** 2 - C_0 ** -2
#         root = np.abs(gamma ** 2 - gamma * b + c)
#         logargument = gamma / (2 * c ** 0.5 * (root) ** 0.5 - b * gamma + 2 * c)
#
#         dydz = 1/(C_0*np.sqrt(c))*(1 - np.sqrt(c)/np.sqrt(root)*(2*gamma-b)*logargument + b*logargument)
#
#         return dydz

    def get_z_from_n(self, n):
        '''
        get z from given n - equation from get_n solved for z
        '''

        return np.log((self.medium.n_ice - n) / self.medium.delta_n) * self.medium.z_0

    def get_surf_skim_angle(self, x1):

        '''
        For a given position x1 = [x,z] and depth profile self.n(), find the angle at which a beam must be
        emitted to "skim the surface", i.e. arrive horizontally (angle = 90 deg) at the surface;
        This is used to find the refraction zone.

        returns:
            C0crit: C0 of critical angle
            thcrit: critical angle
        '''

        nlaunch = self.n(x1[1])
        # by definition, z of critical angle is at surface, i.e. z=0
        zcrit = 0.
        nsurf = self.n(zcrit)

        sinthcrit = nsurf / nlaunch
        if sinthcrit <= 1:
            # ray goes from point with high optical thickness to point with lower optical thickness,
            # i.e. ray bending is towards horizontal
            thcrit = np.arcsin(sinthcrit)
            C0result = self.get_C_0_from_angle(thcrit, x1[1])
            C0crit = C0result.x[0]
        else:
            # ray goes from point with low optical thickness to point with higher optical thickness,
            # i.e. ray bending is towards vertical, no solution
            thcrit = None
            C0crit = None
            self.__logger.warning(' No solution for critical angle for z = {}!'.format(x1[1]))
        self.__logger.info(' critical angle for z = {} is {} !'.format(x1[1], thcrit / units.deg))
        self.__logger.info(' C0 for critical angle is {}'.format(C0crit))

        return C0crit, thcrit

    def is_in_refraction_zone(self, x1, x2, C0crit=None, plot=False):
        '''
        Find if receiver at x2 is in the refraction zone of emitter at x1. The refraction zone
        is the oposite of the shadow zone.

        If the C0 of the critical angle, C0crit, is provided, it will not be calculated. This is useful
        in case find_solutions() is called and C0crit is calculated in the process of determining the
        initial value for the minimization procedure.

        Returns True if x2 is in the refraction zone of x1 - note that the inverse statement is not
        necessarily true, i.e. when False is returned, it is possible that x2 is in the refraction
        zone nonetheless

        TODO:
        Why does the reference point not seem to lie exactly on the mirrored path?
        Instead of returning True/False, it might be useful to return  ycheck - x2[0] (in case x2[0]>ycrit),
        which gives some idea of how close the receiver is to the refraction zone. This could be used to
        define a "gray zone' and a 'far zone', in which the receiver is most definitely in the shadow zone
        '''

        refraction = False

        if C0crit == None:
            C0crit, thcrit = self.get_surf_skim_angle(x1)
        # z_crit = 0 and hence gamma_crit = delta_n by definition
        gcrit = self.medium.delta_n
        # the y-value where the ray hits z=0
        ycrit = self.get_y(gcrit, C0crit, self.get_C_1(x1, C0crit))

        if plot:
            plt.figure('in_refraction_zone')
            plt.grid(True)
            plt.plot(ycrit, 0, 'ro', label='turning point')
            yarray, zarray = self.get_path(x1, [ycrit, 0], C0crit)
            plt.plot(yarray, zarray, 'ko-', markersize=4, label='path')
            plt.plot(x1[0], x1[1], 'C1d', label='emitter')
            plt.plot(x2[0], x2[1], 'go', label='receiver')

        if x2[0] <= ycrit:
            # not in shadow zone
            refraction = True
            self.__logger.debug(' is_in_refraction_zone(): y-position of receiver smaller than ycrit!')
        else:
            # start horizontal ray at (y,z) = (ycrit,0)
            # experimentally this was found to give slightly different results than mirroring the array at the critical angle.
            # theoretically this is not quite unterstood
            C0check = self.get_C_0_from_angle(np.pi / 2., 0)
            C0check = C0check.x[0]
            gcheck = self.get_gamma(x2[1])
            # print('C0check, gcheck',C0check,gcheck)
            ycheck = -self.get_y(gcheck, C0check, self.get_C_1([ycrit, 0], C0check)) + 2 * ycrit
            # print('ycheck, x2[1]',ycheck,x2[1])
            if x2[0] < ycheck:
                refraction = True
            if plot:
                yarraymirr = -yarray + 2 * ycrit
                plt.plot(yarraymirr, zarray, 'mx-', markersize=4, label='mirrored path')
                # the reference point does not seem to lie exactly on the mirrored path but instead
                # ~1cm inside the path (i.e. towards the emmitter) which I do not understand.
                plt.plot(ycheck, x2[1], 'b+', label='reference point')

        if plot:
            plt.legend(fontsize='x-small')

        return refraction

    def get_tof_for_straight_line(self, x1, x2):
        '''
        Calculate the time of flight for a hypothatical ray travelling from x1 to x2 in a straight line.
        Such an array in general is not a solution consistant with Fermat's principle. It is however
        useful as a reference time or approximation for signals not explicable with geometric optics.

        output:
            time of flight for a ray travelling straight from x1 to x2
        '''

        dx = x2[0] - x1[0]
        dz = x2[1] - x1[1]
        n_ice = self.medium.n_ice
        delta_n = self.medium.delta_n
        z_0 = self.medium.z_0

        if dz > 0:
            return 1. / speed_of_light * np.sqrt((dx / dz) ** 2 + 1) * (
            n_ice * dz - delta_n * z_0 * (np.exp(x2[1] / z_0) - np.exp(x1[1] / z_0))
            )
        else:
            return self.n(x2[1]) / speed_of_light * dx

    def get_surface_pulse(self, x1, x2, infirn=False, angle='critical', chdraw=None, label=None):

        '''
        Calculate the time for a ray to travel from x1 to the surface and arriving at the surface
        with (a) critical angle or (b) Brewster angle, propagating in a straight line along the surface
        in air (n=1) in the firn at z=0 (n=self.n(0)) and then reaching the receiver by returning into the
        firn just at the point to reach the receiver at x2, entering the firn at the surface at the same
        angle it reached the surface from x1..

        Input:
            x1, x2: Arrays with x and z positions of emitter x1 and receiver x2
            infirn: Boolean. Set to True if surface ray travels in the firn, set to False (default) if it travels
                    in air.
            angle:  String specifying angle at which ray reaches/leaves the surface. Can be 'Brewster' or 'critical'
                    If neither of these is chosen, a warning is printed and angle is set to 'critical'
            chdraw: If None, do not draw the path of the ray. If the ray should be drawn, a string consistent with
                    the matplotlib.pyplot library has to be specified, e.g. 'r:' to draw a dotted red line.
                    It is assumed that an appropriate figure on which to draw the ray has been created and set as
                    current figure by the user before calling this method.
            label:  Label for plot
        '''

        draw = False
        if chdraw != None:
            draw = True

        if infirn == False:
            nlayer = 1.  # index of refraction at surface, default is n=1 for air
        else:
            nlayer = self.n(0)

        if angle == 'critical':
            # sin(th)=1,
            nxsin = 1.
        elif angle == 'Brewster':
            nxsin = np.sin(np.arctan(1. / self.n(0))) * self.n(0)
        else:
            self.__logger.warning(' unknown input angle=={}, using critical angle!!!'.format(angle))
            nxsin = 1.

        zsurf = 0
        gamma = self.get_gamma(zsurf)

        # print('nxsin = ',nxsin)
        # find emission angle for starting point x1 to hit the surface at the specified angle

        # look at time and distance it takes for the signal to travel from the emitter to the surface
        # and from the surface to the receiver
        tice = 0
        sice = 0
        for x in [x1, x2]:
            sinthemit = nxsin / self.n(x[1])
            th_emit = np.arcsin(sinthemit)
            C0result = self.get_C_0_from_angle(th_emit, x[1])
            C0_emit = C0result.x[0]

            # print(C0_emit)
            self.__logger.info(' emission angle for position {},{} is theta_emit= {}'.format(x[0], x[1], th_emit / units.deg))

            # x-coordinate where ray reaches surface; is always bigger than the x-position of the emitter
            # (i.e. ray travels "to the right")
            xsurf = self.get_y(gamma, C0_emit, self.get_C_1(x, C0_emit))
            sice += xsurf - x[0]
            self.__logger.info(' air pulse starting at x={}, z={} reaches surface at x={}'.format(x[0], x[1], xsurf))
            ttosurf = self.get_travel_time_analytic(x, [xsurf, zsurf], C0_emit)
            tice += ttosurf
            self.__logger.info(' travel time is {} ns.'.format(ttosurf / units.ns))

            if draw:
                z = np.linspace(x[1], zsurf, 1000, endpoint=True)
                y = self.get_y(self.get_gamma(z), C0_emit, C_1=self.get_C_1(x, C0_emit))
                if x == x1:
                    ysurf = [y[-1]]
                else:
                    y = -y + 2 * x2[0]
                    ysurf.append(y[-1])
                    plt.plot(ysurf, [0, 0], chdraw, label=label)

                plt.plot(y, z, chdraw)

        self.__logger.info(' time, distance travelled to and from surface: {}, {} '.format(tice, sice))

        sair = abs(x2[0] - x1[0]) - sice
        tair = sair * nlayer / speed_of_light
        self.__logger.info(' time, distance travelled at surface: {}, {}'.format(tair, sair))
        ttot = tice + tair
        if sair < 0:
            ttot = None
        return ttot

    def angular_diff(self, x_refl, z_refl, pulser_pos, receiver_pos, ipulssol, irxsol):

        '''
        This is a helper function to find a ray that is subject to specular reflection (no transmission) at the
        ice-water interface, e.g. for Moore's Bay. For a (virtual) emitter positioned at [x_refl,z_refl], it finds the
        emission angles such that the rays hit positions pulser_pos and receiver_pos. ipulssol = 0 or 1, respectively
        means that pulser_pos is hit directly or by means of reflection at the surface, respectively.
        irxsol is the equivalent parameter for receiver_pos.

        angular_diff can be used as the function to find a root of in the following fashion:

        result = optimize.root(raytr.angular_diff, x0=x_refl_start, args=(z_refl,pulser_pos,receiver_pos,ipulssol,irxsol))

        Then get the final value by x_refl = result.x[0] (z_refl is fixed)

        The idea is to treat the reflection point as a virtual emitter and then find the x-position at the predefined
        depth z_refl, for which the emisssion angles to pulser_pos and receiver_pos are the same (i.e. the output
        of angular_diff is zero). The x-position would be output "result" ofoptimize.root() above.

        output:
               float, is zero if the angles (w.r.t. the vertical) of rays emitted from [x_refl,z_refl] to
               positions pulser_pos and receiver_pos are the same or greater than zero, if this is not the case.
               For exact defintion, see "result" in code below
        '''

        # treat position of reflection as emitter and Rx/Tx as receivers
        pos_rx = [
            [receiver_pos[0], receiver_pos[1]],
            [x_refl - (pulser_pos[0] - x_refl), pulser_pos[1]]
        ]
        beta0 = None
        beta1 = None
        # solution for receiver
        solution0 = self.find_solutions([x_refl, z_refl], pos_rx[0], plot=False)
        if solution0 != []:
            C0rx = solution0[irxsol]['C0']
            beta0 = self.get_launch_angle([x_refl, z_refl], C0rx)
        # solution for pulser
        solution1 = self.find_solutions([x_refl, z_refl], pos_rx[1], plot=False)
        if solution1 != []:
            C0puls = solution1[ipulssol]['C0']
            beta1 = self.get_launch_angle([x_refl, z_refl], C0puls)

        if beta0 != None and beta1 != None:
            result = (np.tan(beta0) - np.tan(beta1)) ** 2
        else:
            result = np.inf  # set to infinity

        return result


class ray_tracing:
    """
    utility class (wrapper around the 2D analytic ray tracing code) to get
    ray tracing solutions in 3D for two arbitrary points x1 and x2
    """
    solution_types = {1: 'direct',
                      2: 'refracted',
                      3: 'reflected'}

    def __init__(self, x1, x2, medium, attenuation_model="SP1", log_level=logging.WARNING,
                 n_frequencies_integration=6):
        """
        class initilization

        Parameters
        ----------
        x1: 3dim np.array
            start point of the ray
        x2: 3dim np.array
            stop point of the ray
        medium: medium class
            class describing the index-of-refraction profile
        attenuation_model: string
            signal attenuation model (so far only "SP1" is implemented)
        log_level: logging object
            specify the log level of the ray tracing class
            * logging.ERROR
            * logging.WARNING
            * logging.INFO
            * logging.DEBUG
            default is WARNING
        n_frequencies_integration: int
            the number of frequencies for which the frequency dependent attenuation
            length is being calculated. The attenuation length for all other frequencies
            is obtained via linear interpolation.

        """
        self.__logger = logging.getLogger('ray_tracing')
        self.__logger.setLevel(log_level)
        self.__medium = medium
        self.__attenuation_model = attenuation_model
        self.__n_frequencies_integration = n_frequencies_integration

        self.__swap = False
        self.__X1 = x1
        self.__X2 = x2
        if(x2[2] < x1[2]):
            self.__swap = True
            self.__logger.debug('swap = True')
            self.__X2 = x1
            self.__X1 = x2

        dX = self.__X2 - self.__X1
        self.__dPhi = -np.arctan2(dX[1], dX[0])
        c, s = np.cos(self.__dPhi), np.sin(self.__dPhi)
        self.__R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        X1r = self.__X1
        X2r = np.dot(self.__R, self.__X2 - self.__X1) + self.__X1
        self.__logger.debug("X1 = {}, X2 = {}".format(self.__X1, self.__X2))
        self.__logger.debug('dphi = {:.1f}'.format(self.__dPhi / units.deg))
        self.__logger.debug("X2 - X1 = {}, X1r = {}, X2r = {}".format(self.__X2 - self.__X1, X1r, X2r))
        self.__x1 = np.array([X1r[0], X1r[2]])
        self.__x2 = np.array([X2r[0], X2r[2]])

        self.__logger.debug("2D points {} {}".format(self.__x1, self.__x2))
        self.__r2d = ray_tracing_2D(self.__medium, self.__attenuation_model, log_level=log_level,
                                    n_frequencies_integration=self.__n_frequencies_integration)

    def set_solution(self, C0s, C1s, solution_types):
        results = []
        for i in range(len(C0s)):
            if(not np.isnan(C0s[i])):
                results.append({'type': solution_types[i],
                                'C0': C0s[i],
                                'C1': C1s[i]})
        self.__results = results

    def find_solutions(self):
        """
        find all solutions between x1 and x2
        """
        self.__results = self.__r2d.find_solutions(self.__x1, self.__x2)

    def has_solution(self):
        """
        checks if ray tracing solution exists
        """
        return len(self.__results) > 0

    def get_number_of_solutions(self):
        """
        returns the number of solutions
        """
        return len(self.__results)

    def get_results(self):
        """
        returns dictionary of results (the parameters of the analytic ray path function)
        """
        return self.__results

    def get_solution_type(self, iS):
        """ returns the type of the solution

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        solution_type: int
            * 1: 'direct'
            * 2: 'refracted'
            * 3: 'reflected
        """
        return self.__r2d.determine_solution_type(self.__x1, self.__x2, self.__results[iS]['C0'])

    def get_path(self, iS, n_points=1000):
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError
        result = self.__results[iS]
        xx, zz = self.__r2d.get_path(self.__x1, self.__x2, result['C0'], n_points=n_points)
        path_2d = np.array([xx, np.zeros_like(xx), zz]).T
        dP = path_2d - np.array([self.__X1[0], 0, self.__X1[2]])
        MM = np.matmul(self.__R.T, dP.T)
        path = MM.T + self.__X1
        return path

    def get_launch_vector(self, iS):
        """
        calculates the launch vector (in 3D) of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        launch_vector: 3dim np.array
            the launch vector
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self.__results[iS]
        alpha = self.__r2d.get_launch_angle(self.__x1, result['C0'])
        launch_vector_2d = np.array([np.sin(alpha), 0, np.cos(alpha)])
        if self.__swap:
            alpha = self.__r2d.get_receive_angle(self.__x1, self.__x2, result['C0'])
            launch_vector_2d = np.array([-np.sin(alpha), 0, np.cos(alpha)])
        self.__logger.debug(self.__R.T)
        launch_vector = np.dot(self.__R.T, launch_vector_2d)
        return launch_vector

    def get_receive_vector(self, iS):
        """
        calculates the receive vector (in 3D) of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        receive_vector: 3dim np.array
            the receive vector
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self.__results[iS]
        alpha = self.__r2d.get_receive_angle(self.__x1, self.__x2, result['C0'])
        receive_vector_2d = np.array([-np.sin(alpha), 0, np.cos(alpha)])
        if self.__swap:
            alpha = self.__r2d.get_launch_angle(self.__x1, result['C0'])
            receive_vector_2d = np.array([np.sin(alpha), 0, np.cos(alpha)])
        receive_vector = np.dot(self.__R.T, receive_vector_2d)
        return receive_vector

    def get_reflection_angle(self, iS):
        """
        calculates the angle of reflection at the surface (in case of a reflected ray)

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        Returns
        -------
        reflection_angle: float or None
            the reflection angle (for reflected rays) or None for direct and refracted rays
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self.__results[iS]
        return self.__r2d.get_reflection_angle(self.__x1, result['C0'])

    def get_path_length(self, iS, analytic=True):
        """
        calculates the path length of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        analytic: bool
            If True the analytic solution is used. If False, a numerical integration is used. (default: True)

        Returns
        -------
        distance: float
            distance from x1 to x2 along the ray path
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self.__results[iS]
        if analytic:
            return self.__r2d.get_path_length_analytic(self.__x1, self.__x2, result['C0'])
        else:
            return self.__r2d.get_path_length(self.__x1, self.__x2, result['C0'])

    def get_travel_time(self, iS, analytic=True):
        """
        calculates the travel time of solution iS

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        analytic: bool
            If True the analytic solution is used. If False, a numerical integration is used. (default: True)

        Returns
        -------
        time: float
            travel time
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self.__results[iS]
        if(analytic):
            return self.__r2d.get_travel_time_analytic(self.__x1, self.__x2, result['C0'])
        else:
            return self.__r2d.get_travel_time(self.__x1, self.__x2, result['C0'])

    def get_attenuation(self, iS, frequency, max_detector_freq=None):
        """
        calculates the signal attenuation due to attenuation in the medium (ice)

        Parameters
        ----------
        iS: int
            choose for which solution to compute the launch vector, counting
            starts at zero

        frequency: array of floats
            the frequencies for which the attenuation is calculated

        max_detector_freq: float or None
            the maximum frequency of the final detector sampling
            (the simulation is internally run with a higher sampling rate, but the relevant part of the attenuation length
            calculation is the frequency interval visible by the detector, hence a finer calculation is more important)

        Returns
        -------
        attenuation: array of floats
            the fraction of the signal that reaches the observer
            (only ice attenuation, the 1/R signal falloff not considered here)
        """
        n = self.get_number_of_solutions()
        if(iS >= n):
            self.__logger.error("solution number {:d} requested but only {:d} solutions exist".format(iS + 1, n))
            raise IndexError

        result = self.__results[iS]
        return self.__r2d.get_attenuation_along_path(self.__x1, self.__x2, result['C0'], frequency, max_detector_freq)

    def get_ray_path(self, iS):
        return self.__r2d.get_path(self.__x1, self.__x2, self.__results[iS]['C0'], 10000)
