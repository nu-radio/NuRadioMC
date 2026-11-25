"""
Air-to-ice propagation with the analytic ray tracers, compared with numerical integration
(scipy.integrate.quad) of the ray equations.

- the air layer of `IceModelContinuousExpLayers` and of `IceModelExpLayers.from_IceModel`
- plane-wave time differences, incl. a vertical plane wave (zenith 0)
- travel times from emitters in air to receivers in ice (multilayer ray tracer)
- geometries without support: beyond the direct reach, plane wave with a receiver above the surface
- the single-layer ray tracer (fixed n = 1 in air): numba and python backends, the C++ backend raises; focusing and
  fresnel coefficients against reference values

Runs as a script (all tests) or with pytest.
"""
import warnings
import numpy as np
from numpy import testing
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.SignalProp.AnalyticRayTracingImpl import single_layer_analytic_raytracer
from NuRadioMC.utilities import medium, medium_base
from NuRadioReco.utilities import units, constants

ice = medium.greenland_3exp_timing_2025()  # multilayer model with an exponential air layer
ice_simple = medium.greenland_simple()  # single-layer model, n = 1 in air
# where n(z) has kinks: layer boundaries of `ice`, incl. the surface (also the kink of `ice_simple`)
breaks = [layer["z_min"] for layer in ice.layers]


# refractive index profiles
def index_profile(model):
    return lambda z: model.get_index_of_refraction(np.array([0, 0, z]))


n = index_profile(ice)
n_simple = index_profile(ice_simple)


# integrands of a ray with ray parameter p (p = n sin(theta), constant along the ray) in a profile n(z), per dz
def horizontal_integrand(p, n_of_z):
    """dy/dz = tan(theta) = p / sqrt(n^2 - p^2): integrated, the horizontal distance."""
    return lambda z: p / np.sqrt(n_of_z(z)**2 - p**2)


def optical_path_integrand(p, n_of_z):
    """n ds/dz = n^2 / sqrt(n^2 - p^2): integrated and divided by c, the travel time."""
    return lambda z: n_of_z(z)**2 / np.sqrt(n_of_z(z)**2 - p**2)


def path_length_integrand(p, n_of_z):
    """ds/dz = n / sqrt(n^2 - p^2): integrated, the path length."""
    return lambda z: n_of_z(z) / np.sqrt(n_of_z(z)**2 - p**2)


def plane_wave_integrand(p, n_of_z):
    """n cos(theta) = sqrt(n^2 - p^2): integrated between two depths and divided by c, the plane-wave delay."""
    return lambda z: np.sqrt(n_of_z(z)**2 - p**2)


# numerical integration
def integrate(f, z1, z2):
    """int_z1^z2 f dz for z1 < z2, split at `breaks`."""
    points = [b for b in breaks if z1 < b < z2]
    with warnings.catch_warnings():
        # roundoff warnings: the requested 1e-12 is tighter than reachable, but far below the test tolerances
        warnings.simplefilter("ignore", IntegrationWarning)
        return quad(f, z1, z2, points=points, limit=500, epsabs=1e-12, epsrel=1e-12)[0]


# reference solution: p from the horizontal distance, then the travel time
def reference_direct_ray(n_of_z, distance, height, z_rx):
    """p and travel time of the direct ray from (distance, height) in air to (0, z_rx) in ice with profile n_of_z."""
    p = brentq(lambda p: integrate(horizontal_integrand(p, n_of_z), z_rx, height) - distance,
               1e-6, n_of_z(height) * (1 - 1e-14), xtol=1e-16, rtol=8.9e-16)
    return p, integrate(optical_path_integrand(p, n_of_z), z_rx, height) / constants.c


def test_air_layer():
    """n falls from N_AIR_SUMMIT at the surface to 1, with scale height AIR_SCALE_HEIGHT_SUMMIT."""
    n_air = [n(z) for z in [1e-6, 1 * units.km, medium.AIR_SCALE_HEIGHT_SUMMIT, 100 * units.km]]
    testing.assert_allclose(n_air[0], medium.N_AIR_SUMMIT, rtol=1e-9)
    testing.assert_allclose(n_air[2], 1 + (medium.N_AIR_SUMMIT - 1) / np.e, rtol=1e-12)
    assert np.all(np.diff(n_air) < 0), n_air


def test_air_layer_from_ice_model_and_set_params():
    """`from_IceModel` gets the default exponential air layer; `_set_params` without air arguments keeps the air."""
    converted = medium_base.IceModelExpLayers.from_IceModel(medium.greenland_simple())
    n_conv = [converted.get_index_of_refraction(np.array([0, 0, z])) for z in [1e-6, 8 * units.km, 100 * units.km]]
    testing.assert_allclose(n_conv[:2], [medium_base.N_AIR_SEA_LEVEL, 1 + (medium_base.N_AIR_SEA_LEVEL - 1) / np.e],
                            rtol=1e-12)
    assert np.all(np.diff(n_conv) < 0), n_conv

    perturbed = medium.greenland_3exp_timing_2025()
    perturbed._set_params(perturbed.nN, perturbed.delta_nN, perturbed.ls, perturbed.zs)
    testing.assert_allclose(perturbed.get_index_of_refraction(np.array([0, 0, 1e-6])), medium.N_AIR_SUMMIT, rtol=1e-12)


def test_plane_wave():
    """Two receivers on one vertical: dt = int sqrt(n^2 - p^2) dz / c with p = n_air(surface) * sin(zenith)."""
    z_low, z_high = -97 * units.m, -20 * units.m
    for compile_numba in [True, False]:
        r = ray.ray_tracing(ice, compile_numba=compile_numba)
        r.set_start_and_end_point(np.array([0, 0, z_low]), np.array([0, 0, z_high]))
        for zenith in np.array([0, 1e-6, 1, 30, 60, 89, 90]) * units.deg:
            p = medium.N_AIR_SUMMIT * np.sin(zenith)
            dt_ref = integrate(plane_wave_integrand(p, n), z_low, z_high) / constants.c
            testing.assert_allclose(r.get_time_difference_plane_wave(zenith, 0), dt_ref, atol=1e-4 * units.ns,
                                    err_msg=f"plane wave, zenith {zenith / units.deg} deg, numba {compile_numba}")


def test_plane_wave_receiver_above_surface():
    r = ray.ray_tracing(ice)
    r.set_start_and_end_point(np.array([0., 0, -97]), np.array([0., 0, 2]))
    testing.assert_raises(ValueError, r.get_time_difference_plane_wave, 0, 0)


# (distance, emitter height, receiver depth) in m. Near-horizontal rays from low emitters have p > 1 (allowed up to
# n at the emitter, as n > 1 near the surface). (7000, 1, -97) is at 76 % of the direct reach (9252 m), where p is
# within 1e-12 of n at the emitter; the 50 m emitters cover p = 1 -+ 2e-7 (alpha = 1 - p^2 -> 0 in the air layer).
AIR_TO_ICE_GEOMETRIES = [(100, 10, -97), (1000, 100, -97), (30, 1, -20), (3000, 1000, -20), (1271.4, 10, -97),
                         (2392, 10, -40), (5000, 1, -97), (5000, 100, -20), (7000, 1, -97),
                         (2500, 50, -97), (2501, 50, -97), (2501.25, 50, -97), (2502.5, 50, -97)]


def test_air_to_ice():
    """Multilayer ray tracer against `reference_direct_ray`; the path length against its integral."""
    r = ray.ray_tracing(ice, compile_numba=True)
    for distance, height, z_rx in AIR_TO_ICE_GEOMETRIES:
        distance, height, z_rx = distance * units.m, height * units.m, z_rx * units.m
        p, t_ref = reference_direct_ray(n, distance, height, z_rx)

        r.set_start_and_end_point(np.array([distance, 0, height]), np.array([0, 0, z_rx]))
        r.find_solutions()
        assert r.get_number_of_solutions() == 1, (distance, height, z_rx, r.get_number_of_solutions())
        testing.assert_allclose(r.get_travel_time(0), t_ref, atol=1e-3 * units.ns,
                                err_msg=f"air to ice, {distance}, {height}, {z_rx}")
        testing.assert_allclose(1 / r.get_results()[0]["C0"], p, rtol=1e-9)
        testing.assert_allclose(r.get_path_length(0), integrate(path_length_integrand(p, n), z_rx, height),
                                atol=1e-3 * units.m, err_msg=f"path length, {distance}, {height}, {z_rx}")


def test_beyond_direct_reach():
    """Beyond the direct reach (9252 m here) only rays that turn in the air reach the receiver: not supported."""
    r = ray.ray_tracing(ice, compile_numba=True)
    r.set_start_and_end_point(np.array([13880., 0, 1]), np.array([0., 0, -97]))
    r.find_solutions()
    assert r.get_number_of_solutions() == 0, r.get_number_of_solutions()


def test_vertical_air_path():
    """Emitter in air straight above the receiver: one solution."""
    r = ray.ray_tracing(ice, compile_numba=True)
    r.set_start_and_end_point(np.array([0., 0, 100]), np.array([0., 0, -50]))
    r.find_solutions()
    assert r.get_number_of_solutions() == 1, r.get_number_of_solutions()
    testing.assert_allclose(r.get_travel_time(0), integrate(n, -50, 100) / constants.c, atol=1e-3 * units.ns)


def test_single_layer_air_to_ice():
    """Single-layer ray tracer, numba and python backends, both directions, against `reference_direct_ray`."""
    geometries = [(100, 10, -97), (1000, 100, -97), (30, 1, -20), (3000, 1000, -20), (1271.4, 10, -97),
                  (2392, 10, -40), (5000, 1, -97), (5000, 100, -20), (13880, 1, -97)]
    for compile_numba in [True, False]:
        r = ray.ray_tracing(ice_simple, use_cpp=False, compile_numba=compile_numba)
        for distance, height, z_rx in geometries:
            distance, height, z_rx = distance * units.m, height * units.m, z_rx * units.m
            _, t_ref = reference_direct_ray(n_simple, distance, height, z_rx)
            for start, end in [([distance, 0, height], [0, 0, z_rx]), ([0, 0, z_rx], [distance, 0, height])]:
                r.set_start_and_end_point(np.array(start), np.array(end))
                r.find_solutions()
                assert r.get_number_of_solutions() == 1, (start, end, r.get_number_of_solutions())
                testing.assert_allclose(r.get_travel_time(0), t_ref, atol=1e-3 * units.ns,
                                        err_msg=f"single layer, numba {compile_numba}: {start} -> {end}")


def test_single_layer_cpp_raises():
    """The C++ backend has no air propagation."""
    if not single_layer_analytic_raytracer.cpp_available:
        return
    r = ray.ray_tracing(ice_simple, use_cpp=True)
    r.set_start_and_end_point(np.array([1000., 0, 100]), np.array([0., 0, -97]))
    testing.assert_raises(NotImplementedError, r.find_solutions)


def test_single_layer_focusing_fresnel_regression():
    """
    Single-layer ray tracer (python backend, southpole_simple): C0, focusing factor and fresnel transmission
    coefficients of the geometries of ``NuRadioMC/examples/SignalProp/E02ToAir.py`` against stored reference values
    (regression test; NaN = no solution).
    """
    x_ice = np.array([0, 0., -149.]) * units.m
    x_air = [np.array([d, 0., 100]) * units.m for d in [0, 200, 1000, 10000]]
    x_starts = [x_ice] * 4 + [x_air[3]]
    x_stops = x_air + [x_ice]
    c0, focusing, t_p, t_s = (np.full((5, 2), np.nan) for _ in range(4))
    r = ray.ray_tracing(medium.southpole_simple(), use_cpp=False)
    for i, (x_start, x_stop) in enumerate(zip(x_starts, x_stops)):
        r.set_start_and_end_point(x_start, x_stop)
        r.find_solutions()
        for i_s in range(r.get_number_of_solutions()):
            c0[i, i_s] = r.get_results()[i_s]['C0']
            focusing[i, i_s] = r.get_focusing(i_s)
            fresnel = r.get_fresnel_coefficients(i_s)[0]
            t_p[i, i_s], t_s[i, i_s] = fresnel['theta'], fresnel['phi']

    ray_tracing_C0_reference = np.array([
        [np.nan, np.nan],
        [1.3087854182874152, np.nan],
        [1.0064308743162975, np.nan],
        [1.000051230401809, np.nan],
        [1.000051230401809, np.nan],
    ])
    focusing_factors_reference = np.array([
        [np.nan, np.nan],
        [0.6745227377161311, np.nan],
        [0.11277309434909118, np.nan],
        [0.00904289875471403, np.nan],
        [0.8156210793664358, np.nan],
    ])
    transmission_t_p_reference = np.array([
        [np.nan, np.nan],
        [1.3157903900606893, np.nan],
        [2.210688211073233, np.nan],
        [2.654051751148146, np.nan],
        [0.02942659577064398, np.nan],
    ])
    transmission_t_s_reference = np.array([
        [np.nan, np.nan],
        [1.2681188663353486, np.nan],
        [1.7814111089911056, np.nan],
        [1.9780683188262487, np.nan],
        [0.02193168117375125, np.nan],
    ])
    testing.assert_allclose(c0, ray_tracing_C0_reference, rtol=1e-5, equal_nan=True)
    testing.assert_allclose(focusing, focusing_factors_reference, rtol=1e-5, equal_nan=True)
    testing.assert_allclose(t_p, transmission_t_p_reference, rtol=1e-5, equal_nan=True)
    testing.assert_allclose(t_s, transmission_t_s_reference, rtol=1e-5, equal_nan=True)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"{name} passed")
    print("T12test_air_to_ice passed without issues")
