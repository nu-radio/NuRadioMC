"""Propagator that handles sources or receivers above the ice surface.

Pairs with both points in the ice are delegated to the analytic ray tracer, so in-ice
emitters simulate exactly as before. Pairs that cross the surface use the air-ice ray
tracer (straight air leg, Snell refraction at a flat surface, analytic in-ice leg) and
return a single solution of the direct type. Amplitude effects on the crossing path are
the in-ice attenuation and the Fresnel transmission coefficients at the entry point;
focusing is not applied to crossing paths.
"""

import logging

import numpy as np

from NuRadioMC.SignalProp import analyticraytracing
from NuRadioMC.SignalProp.air_ice_raytracer import AirIceRaytracer
from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base
from NuRadioReco.utilities import geometryUtilities, units


class air_ice_ray_tracing(ray_tracing_base):
    """Ray tracer for geometries with one end point in air, otherwise the analytic tracer."""

    def __init__(self, medium, attenuation_model=None, log_level=logging.NOTSET,
                 n_frequencies_integration=None, n_reflections=None, config=None,
                 detector=None, ray_tracing_2D_kwards={}, use_cpp=None,
                 precision=0.001 * units.deg, max_iterations=100):
        """Build the delegate analytic tracer and store the entry-point search settings.

        Args:
            precision: Largest allowed Snell mismatch at the entry point.
            max_iterations: Bisection steps before the entry-point search gives up.
        """
        super().__init__(medium, attenuation_model, log_level, n_frequencies_integration,
                         n_reflections, config, detector, ray_tracing_2D_kwards)
        self.__logger = logging.getLogger('NuRadioMC.SignalProp.air_ice_ray_tracing')
        self.__logger.setLevel(log_level)
        self._inice = analyticraytracing.ray_tracing(
            medium, attenuation_model=attenuation_model, log_level=log_level,
            n_frequencies_integration=n_frequencies_integration, n_reflections=n_reflections,
            config=config, detector=detector, ray_tracing_2D_kwards=ray_tracing_2D_kwards,
            use_cpp=use_cpp)
        self._precision = precision
        self._max_iterations = max_iterations
        self._cross = False
        self._air = None
        self._results = []

    def set_start_and_end_point(self, x1, x2):
        """Store the end points and decide whether the path crosses the surface."""
        super().set_start_and_end_point(x1, x2)
        self._both_in_air = self._X1[2] > 0 and self._X2[2] > 0
        self._cross = (self._X1[2] > 0) != (self._X2[2] > 0)
        self._air = None
        self._results = []
        if not self._cross and not self._both_in_air:
            self._inice.set_start_and_end_point(x1, x2)

    def find_solutions(self):
        """Find the crossing path with the air-ice tracer or delegate to the analytic tracer.

        Two points in the air are outside the model and get no solution.
        """
        if self._both_in_air:
            self._results = []
            return
        if not self._cross:
            self._inice.find_solutions()
            self._results = self._inice.get_results()
            return
        tracer = AirIceRaytracer(
            self._medium, self._X1, self._X2, precision=self._precision,
            max_iterations=self._max_iterations,
            inice_kwargs=dict(attenuation_model=self._attenuation_model,
                              n_frequencies_integration=self._n_frequencies_integration))
        try:
            tracer.run()
            travel_time = tracer.get_signal_travel_time('total')
        except Exception as err:
            self.__logger.debug(f'air-ice tracing failed: {err}')
            travel_time = np.nan
        if not np.isfinite(travel_time) or tracer.get_inice_solution() is None:
            self._results = []
            return
        self._air = tracer
        self._results = [dict(tracer.get_inice_solution(), type=1)]

    def _unit_horizontal(self, frm, to):
        """Unit horizontal vector from `frm` to `to`, or +x when they are vertically aligned."""
        d = np.array(to[:2], dtype=float) - np.array(frm[:2], dtype=float)
        n = np.linalg.norm(d)
        return d / n if n > 0 else np.array([1.0, 0.0])

    def _check(self, iS):
        if iS >= len(self._results):
            raise IndexError(f'solution {iS} requested but only {len(self._results)} exist')

    def get_solution_type(self, iS):
        """Crossing paths are reported as direct; in-ice pairs keep the analytic classification."""
        self._check(iS)
        return 1 if self._cross else self._inice.get_solution_type(iS)

    def get_path(self, iS, n_points=1000):
        """3D path from the start point to the end point."""
        self._check(iS)
        if not self._cross:
            return self._inice.get_path(iS, n_points)
        path = self._air.get_ray_path(n_points)
        return path if self._air.positions_flipped() else path[::-1]

    def get_launch_vector(self, iS):
        """Direction of the ray as it leaves the start point."""
        self._check(iS)
        if not self._cross:
            return self._inice.get_launch_vector(iS)
        entry = self._air.get_surface_intersection_point()
        if self._X1[2] > 0:
            v = entry - self._X1
            return v / np.linalg.norm(v)
        a = self._air.get_launch_angle()
        ux, uy = self._unit_horizontal(self._X1, entry)
        return np.array([np.sin(a) * ux, np.sin(a) * uy, np.cos(a)])

    def get_receive_vector(self, iS):
        """Direction the signal arrives from, seen at the end point."""
        self._check(iS)
        if not self._cross:
            return self._inice.get_receive_vector(iS)
        entry = self._air.get_surface_intersection_point()
        if self._X2[2] > 0:
            v = entry - self._X2
            return v / np.linalg.norm(v)
        a = self._air.get_launch_angle()
        ux, uy = self._unit_horizontal(self._X2, entry)
        return np.array([np.sin(a) * ux, np.sin(a) * uy, np.cos(a)])

    def get_reflection_angle(self, iS):
        """Crossing paths have no surface reflection."""
        self._check(iS)
        return None if self._cross else self._inice.get_reflection_angle(iS)

    def get_path_length(self, iS, analytic=True):
        """Geometric path length (air leg plus in-ice leg for crossing paths)."""
        self._check(iS)
        return self._air.get_path_length('total') if self._cross else self._inice.get_path_length(iS, analytic)

    def get_travel_time(self, iS, analytic=True):
        """Signal travel time (air leg plus in-ice leg for crossing paths)."""
        self._check(iS)
        return self._air.get_signal_travel_time('total') if self._cross else self._inice.get_travel_time(iS, analytic)

    def get_attenuation(self, iS, frequency, max_detector_freq=None):
        """Attenuation factor; only the in-ice leg attenuates on crossing paths."""
        self._check(iS)
        if not self._cross:
            return self._inice.get_attenuation(iS, frequency, max_detector_freq)
        return self._air.get_inice_attenuation(frequency, max_detector_freq)

    def apply_propagation_effects(self, efield, i_solution):
        """Apply in-ice attenuation and the Fresnel transmission at the entry point."""
        self._check(i_solution)
        if not self._cross:
            return self._inice.apply_propagation_effects(efield, i_solution)
        spec = efield.get_frequency_spectrum()
        if self._config is None or self._config['propagation']['attenuate_ice']:
            max_freq = self._max_detector_frequency or np.max(efield.get_frequencies())
            spec *= self.get_attenuation(i_solution, efield.get_frequencies(), max_freq)
        theta_air, theta_ice = self._air.get_entry_incidence_angles()
        n_surface = self._medium.get_index_of_refraction([0, 0, -0.1 * units.m])
        if self._X1[2] > 0:
            t_theta = geometryUtilities.get_fresnel_t_p(theta_air, n_2=n_surface, n_1=1.0)
            t_phi = geometryUtilities.get_fresnel_t_s(theta_air, n_2=n_surface, n_1=1.0)
        else:
            t_theta = geometryUtilities.get_fresnel_t_p(theta_ice, n_2=1.0, n_1=n_surface)
            t_phi = geometryUtilities.get_fresnel_t_s(theta_ice, n_2=1.0, n_1=n_surface)
        spec[1] *= t_theta
        spec[2] *= t_phi
        efield.set_frequency_spectrum(spec, efield.get_sampling_rate())
        return efield

    def get_output_parameters(self):
        """Same output layout as the analytic tracer so HDF5 files keep their schema."""
        return self._inice.get_output_parameters()

    def get_raytracing_output(self, i_solution):
        """Per-solution output; crossing paths report their in-ice segment parameters."""
        self._check(i_solution)
        if not self._cross:
            return self._inice.get_raytracing_output(i_solution)
        sol = self._results[i_solution]
        return {
            'ray_tracing_C0': sol['C0'],
            'ray_tracing_C1': sol.get('C1', np.nan),
            'focusing_factor': 1.0,
            'ray_tracing_reflection': 0,
            'ray_tracing_reflection_case': 1,
            'ray_tracing_solution_type': 1,
        }

    def get_number_of_raytracing_solutions(self):
        """Largest number of solutions any pair can have (set by the in-ice tracer)."""
        return self._inice.get_number_of_raytracing_solutions()

    def set_config(self, config):
        """Forward configuration changes to the delegate as well."""
        super().set_config(config)
        self._inice.set_config(config)
