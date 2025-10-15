import logging

import numpy as np

from c8_tracer.c8_tracer_ext import Vec3
from c8_tracer.nrmc_interface import CreateNRMCWrappedRayTracer

from NuRadioReco.utilities import units
from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base

class C8RayTracerIndividual(ray_tracing_base):
    """
    C8 ray tracer in the style of the NRMC ray tracing base class
    This ray tracer finds the solutions each time they are queried
    and returns the path

    :param medium: class that inherits the `IceModel` class
    :param attenuation_model: unused by this class
    :param log_level: controls the granualrity of the print messages
    :param n_frequencies_integration: unused by this class
    :param config: unused by this class
    :param detector: unused by this class
    :param min_step: smallest step size that will be taken during propagation (meters)
    :param max_step: largest step size that will be taken during propagation (meters)
    :param tolerance: relative error tolerance that defines the adaptive step size

    """

    def __init__(
        self,
        medium,
        attenuation_model=None,
        log_level=logging.NOTSET,
        n_frequencies_integration=None,
        n_reflections=None,
        config=None,
        detector=None,
        min_step: float = 0.0001,
        max_step: float = 1.0,
        tolerance: float = 1e-8,
    ):
        if n_reflections != 1:
            raise RuntimeError(
                "This tracer only works for a single reflection at the surface"
            )

        super().__init__(
            medium=medium,
            attenuation_model=attenuation_model,
            log_level=log_level,
            n_frequencies_integration=n_frequencies_integration,
            n_reflections=n_reflections,
            config=config,
            detector=detector,
        )

        # function that returns a signal path for each of ONLY 2 solutions
        self._trace_to_point, self._tracer = CreateNRMCWrappedRayTracer(
            medium,
            min_step,
            max_step,
            tolerance,
        )

        self.set_config(config=config)

    def reset_solutions(self):
        self._X1 = None
        self._X2 = None
        self._results = None
        self._paths = None

    def find_solutions(self):
        self._results = self._trace_to_point(self._X1, self._X2)
        self._paths = [None] * len(self._results)

    def has_solution(self):
        return len(self._results) > 0

    def get_solution_type(self, iS):
        raise NotImplementedError("TODO, figure out what to return")

    def get_path(self, iS, n_points=1000):
        # ensure solutions exist
        if self._results is None:
            self.find_solutions()

        if self._paths[iS] is None:
            signal_path = self._tracer.GetSignalPath(
                Vec3(self._X1),
                Vec3(self._results[iS].emit),
                Vec3(self._X2),
                self._medium,
            )

            self._paths[iS] = np.array([np.array([x.x, x.y, x.z]) for x in signal_path])

        return self._paths[iS]

    def get_launch_vector(self, iS):
        return self._results[iS].emit

    def get_receive_vector(self, iS):
        return -self._results[iS].receive

    def get_reflection_angle(self, iS):
        if iS != 1:
            return None

        # find the point closest to z and ensure that it is plausibly
        # a reflection, note that the tracer will ensure that a path point
        # is with 5e-6 m of a reflection plane, so it won't be subtle
        path = self.get_path[iS]
        z_ref = self._medium.z_air_boundary

        dz = np.abs(path[:, 2] - z_ref)

        if dz > 1e-2:
            return None

        imin = np.argmin(dz)
        dr = path[imin] - path[imin - 1]
        dr /= np.linalg.norm(dr)

        # TODO: which is the angle they are consisering
        return abs(np.arccos(dr[2]))

    def get_path_length(self, iS):
        return self._results[iS].R_distance

    def get_travel_time(self, iS):
        return self._results[iS].propagation_time * units.s
