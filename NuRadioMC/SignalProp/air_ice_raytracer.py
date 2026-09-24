"""Raytracer for air-to-ice radio propagation via binary search on the surface entry point.

Adapted from the RNO-G antenna-positioning package
(https://github.com/RNO-G/antenna-positioning/blob/main/AntPosCal/ray_tracing/air_ice_raytracer.py).
Changes for use in NuRadioMC: the in-ice leg ends at one common depth just below the
surface (`SURFACE_DEPTH`), the air leg uses the same speed of light as the in-ice tracer,
the entry-point bisection continues when a midpoint lies beyond the reach of in-ice rays,
and accessor methods after `get_launch_angle` expose the path length, the in-ice
attenuation and the entry angles to the `air_ice` propagator and the table generator.
"""

import numpy as np
import scipy.constants
import NuRadioMC.SignalProp.analyticraytracing
from NuRadioReco.utilities import units


SURFACE_DEPTH = -0.01


class AirIceRaytracer:
    """Binary-search raytracer for air-to-ice propagation that finds the surface entry point satisfying Snell's law."""

    def __init__(
            self,
            ice_model,
            start_point=None,
            end_point=None,
            precision=.001 * units.deg,
            max_iterations=100,
            inice_kwargs=None
    ):
        """
        Class to perform raytracing between a transmitter above the surface and a receiver in the ice
        The raytracer works by finding the point at which the radio signal has to enter the ice, so that
        the direction of the ray entering the ice (after accounting for Snell's law) and the launch angle
        of the ray from the surface to the receiver match.

        :param ice_model: NuRadioMC.utilities.IceModelSimple object
            Object containing information about the index of refraction profile of the ice. Needs to
            be an exponention ice model
        :param start_point: numpy array of shape (3,)
            Position of the transmitting antenna
        :param end_point: numpy array of shape (3,)
            Position of the receiving antenna
        :param precision: float
            Miximum allowed angle between the ray entering the ice and the launch angle of the ray from the
            surface to the receiving antenna
        :param max_iterations: integer
            Number of iterations after which the optimization process is aborted if no solution was found
        :param inice_kwargs: dict or None
            Extra keyword arguments for the in-ice ray_tracing_2D (attenuation model, number of
            frequencies for the attenuation integration); needed when attenuation is requested
        """
        self.__ice_model = ice_model
        self.__start_point = None
        self.__end_point = None
        self.__start_2d = None
        self.__end_2d = None
        self.__search_range = None
        self.__surface_intersect_2d = None
        self.__positions_flipped = False
        self.__inice_raytracing_solution = None
        self.__precision = precision
        self.__max_iterations = max_iterations
        self.__speed_of_light = scipy.constants.c * units.m / units.second
        self.__inice_raytracer = NuRadioMC.SignalProp.analyticraytracing.ray_tracing_2D(
            self.__ice_model, **(inice_kwargs or {})
        )
        if start_point is not None and end_point is not None:
            self.set_start_end_points(
                start_point,
                end_point
            )

    def set_start_end_points(
            self,
            start_point,
            end_point
    ):
        """
        Set the position of the transmitting and receiving antennas.
        Can be skipped if antenna positions were already passed to the
        __init__ method.

        :param start_point: numpy array of shape (3,)
            Position of the transmitting antenna
        :param end_point: numpy array of shape (3,)
            Position of the receiving antenna
        :return: None
        """
        if (start_point[2] > 0 and end_point[2] > 0) or (start_point[2] < 0 and end_point[2] < 0):
            raise ValueError(
                'One of the antennas has to be above and one below the ice surface.'
            )
        self.__end_point = end_point
        self.__start_point = start_point
        if start_point[2] > 0:
            self.__positions_flipped = False
            self.__start_2d = np.array([
                np.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2),
                start_point[2]
            ])
            self.__end_2d = np.array([
                0,
                end_point[2]
            ])
        else:
            self.__positions_flipped = True
            self.__start_2d = np.array([
                np.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2),
                end_point[2]
            ])
            self.__end_2d = np.array([
                0,
                start_point[2]
            ])

    def run(self):
        """
        Run the raytracer.

        :return: None
        """
        if self.__start_point is None or self.__end_point is None:
            raise ValueError('Start and end points are not set. Did you run set_start_end_points()?')
        self.__find_optimization_start_points()
        self.__inice_raytracing_solution = None
        angle_error = np.pi
        for i in range(self.__max_iterations):
            angle_error = self.__search_step()
            if angle_error is None:
                break
            if angle_error <= self.__precision:
                break
        if angle_error is None or angle_error > self.__precision:
            self.__inice_raytracing_solution = None

    def get_ray_path(
            self,
            n_points=100
    ):
        """
        Get the path of the ray from the transmitting to the receiving antenna. Only useful after the run()
        method has been executed.

        :param n_points: integer
            Number of points on the ray path that should be returned
        :return: numpy array of shape (n_points, 3)
            An array of n_points points on the ray path.
        """
        l_1 = np.sqrt((self.__start_2d[0] - self.__surface_intersect_2d)**2 + self.__end_2d[1]**2)
        l_2 = np.sqrt(self.__surface_intersect_2d**2 + self.__start_2d[1]**2)
        n_inair = int(n_points * l_2 / (l_1 + l_2))
        n_inice = int(n_points * l_1 / (l_1 + l_2))
        n_inice += n_points - n_inair - n_inice
        path_2d = np.zeros((n_points, 2))
        inice_path = self.__inice_raytracer.get_path(
            self.__end_2d,
            np.array([self.__surface_intersect_2d, SURFACE_DEPTH]),
            self.__inice_raytracing_solution['C0'],
            n_inice
        )
        path_2d[:n_inice, 0] = inice_path[0]
        path_2d[:n_inice, 1] = inice_path[1]
        path_2d[n_inice:, 0] = np.linspace(self.__surface_intersect_2d, self.__start_2d[0], n_inair)
        path_2d[n_inice:, 1] = np.linspace(0, self.__start_2d[1], n_inair)
        path_3d = np.zeros((n_points, 3))
        direction = (self.__start_point - self.__end_point)[:2]
        direction /= np.sqrt(np.sum(direction**2))
        if self.__positions_flipped:
            direction *= -1.
        path_3d[:, :2] = np.outer(path_2d[:, 0], direction)
        path_3d[:, 2] = path_2d[:, 1]
        if self.__positions_flipped:
            path_3d[:, :2] += self.__start_point[:2]

        else:
            path_3d[:, :2] += self.__end_point[:2]
        return path_3d

    def get_signal_travel_time(
            self,
            path='total'
    ):
        """
        Returns the time the signal took to propagate from the transmitting to the receiving antenna.
        Only useful after the run() method has been executed.

        :param path: string, "total", "ice", or "air"
            Defines for which part of the ray path the propagation time should be returned
        :return: float
            Propagation time (in nanoseconds) between the transmitting and receiving antennas
        """
        inair_time = (np.sqrt((self.__surface_intersect_2d - self.__start_2d[0])**2 + self.__start_2d[1]**2)
                      / self.__speed_of_light)
        if path == 'air':
            return inair_time
        if self.__inice_raytracing_solution is None:
            return np.nan
        inice_time = self.__inice_raytracer.get_travel_time(
            self.__end_2d,
            np.array([self.__surface_intersect_2d, SURFACE_DEPTH]),
            self.__inice_raytracing_solution['C0']
        )
        if path == 'ice':
            return inice_time
        if path == 'total':
            return inice_time + inair_time
        raise ValueError('path parameter has to be either "total", "air" or "ice", not "{}".'.format(path))

    def get_surface_intersection_point(self):
        """
        Returns the coordinates at which the radio signal crosses from the air into the ice
        Only useful after the run() method has been executed.

        :return: numpy array of shape (3,)
            Coordinates of the point where the radio signal crosses from the air into the ice.
        """
        direction = (self.__start_point - self.__end_point)[:2]
        direction /= np.sqrt(np.sum(direction**2))
        intersection_point = np.zeros(3)
        intersection_point[:2] = direction * self.__surface_intersect_2d
        if self.__positions_flipped:
            intersection_point *= -1
            intersection_point[:2] += self.__start_point[:2]
        else:
            intersection_point[:2] += self.__end_point[:2]
        return intersection_point

    def get_receive_angle(self):
        """Return the angle at which the ray arrives at the receiving antenna (radians)."""
        return self.__inice_raytracer.get_receive_angle(
            self.__end_2d,
            np.array([self.__surface_intersect_2d, SURFACE_DEPTH]),
            self.__inice_raytracing_solution['C0']
        )

    def get_launch_angle(self):
        """Return the launch angle of the in-ice ray from the surface entry point (radians)."""
        if self.__inice_raytracing_solution is None:
            return np.nan
        return self.__inice_raytracer.get_launch_angle(
            self.__end_2d,
            self.__inice_raytracing_solution['C0']
        )

    def __find_optimization_start_points(self):
        """Initialize the binary search range for the surface intersection point."""
        x = - self.__start_2d[0] * self.__end_2d[1] / (self.__start_2d[1] - self.__end_2d[1])
        self.__search_range = np.array([0, x])

    def __search_step(self):
        """One binary search iteration: bisect on the Snell's law angle mismatch at the ice surface."""
        x = self.__search_range[0] + .5 * (self.__search_range[1] - self.__search_range[0])
        self.__surface_intersect_2d = x
        n = self.__ice_model.get_index_of_refraction([0, 0, SURFACE_DEPTH])
        alpha = np.arctan((self.__start_2d[0] - x) / self.__start_2d[1])
        inice_angle = np.pi - np.arcsin(np.sin(alpha) / n)
        solutions = self.__inice_raytracer.find_solutions(
            self.__end_2d,
            np.array([x, SURFACE_DEPTH])
        )
        for solution in solutions:
            angle = self.__inice_raytracer.get_receive_angle(
                self.__end_2d,
                np.array([x, SURFACE_DEPTH]),
                solution['C0'],
                solution['reflection'],
                solution['reflection_case']
            )
            if angle >= np.pi / 2.:
                self.__inice_raytracing_solution = solution
                if inice_angle > angle:
                    self.__search_range[1] = x
                else:
                    self.__search_range[0] = x
            return np.abs(inice_angle - angle)
        # No in-ice ray from the receiver reaches the surface this far out (the reachable
        # entry points form a contiguous interval starting at the receiver's vertical), so
        # the entry point must lie closer in: shrink the upper bound and keep bisecting.
        self.__search_range[1] = x
        return np.pi

    def get_path_length(self, path='total'):
        """Return the geometric path length in metres for 'total', 'ice' or 'air'."""
        inair = np.sqrt((self.__surface_intersect_2d - self.__start_2d[0])**2 + self.__start_2d[1]**2)
        if path == 'air':
            return inair
        if self.__inice_raytracing_solution is None:
            return np.nan
        inice = self.__inice_raytracer.get_path_length(
            self.__end_2d,
            np.array([self.__surface_intersect_2d, SURFACE_DEPTH]),
            self.__inice_raytracing_solution['C0']
        )
        if path == 'ice':
            return inice
        if path == 'total':
            return inice + inair
        raise ValueError('path parameter has to be either "total", "air" or "ice", not "{}".'.format(path))

    def get_inice_attenuation(self, frequency, max_detector_freq=None):
        """Return the frequency-dependent attenuation factor of the in-ice leg (air is lossless)."""
        return self.__inice_raytracer.get_attenuation_along_path(
            self.__end_2d,
            np.array([self.__surface_intersect_2d, SURFACE_DEPTH]),
            self.__inice_raytracing_solution['C0'],
            frequency,
            max_detector_freq
        )

    def get_entry_incidence_angles(self):
        """Return (angle in air, angle in ice) from the surface normal at the entry point, in radians."""
        theta_air = np.arctan((self.__start_2d[0] - self.__surface_intersect_2d) / self.__start_2d[1])
        theta_ice = np.pi - self.get_receive_angle()
        return theta_air, theta_ice

    def get_inice_solution(self):
        """Return the in-ice solution dictionary (C0, C1, reflection, reflection_case) or None."""
        return self.__inice_raytracing_solution

    def positions_flipped(self):
        """Return True when the start point given by the user is the in-ice one."""
        return self.__positions_flipped
