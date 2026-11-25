"""
Air-to-ice propagation with the multilayer analytic ray tracer, compared with numerical quadrature.

- the air layer of `IceModelContinuousExpLayers`: n falls from n_air_surface at the surface to 1
- plane-wave time differences, incl. a vertical plane wave (zenith 0)
- travel times from emitters in air to receivers in ice
"""
import numpy as np
from numpy import testing
from scipy.integrate import quad
from scipy.optimize import brentq
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.SignalProp.AnalyticRayTracingImpl import single_layer_analytic_raytracer
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units, constants

ice = medium.greenland_3exp_timing_2025()


def n(z):
    return ice.get_index_of_refraction(np.array([0, 0, z]))


breaks = [layer["z_min"] for layer in ice.layers]


def integrate(f, z1, z2):
    points = [b for b in breaks if z1 < b < z2]
    return quad(f, z1, z2, points=points, limit=500, epsabs=1e-12, epsrel=1e-12)[0]


# air layer
H = medium.AIR_SCALE_HEIGHT_SUMMIT
n_air = [n(z) for z in [1e-6, 1 * units.km, H, 100 * units.km]]
testing.assert_allclose(n_air[0], medium.N_AIR_SUMMIT, rtol=1e-9)
testing.assert_allclose(n_air[2], 1 + (medium.N_AIR_SUMMIT - 1) / np.e, rtol=1e-12)
assert np.all(np.diff(n_air) < 0), n_air
print("air layer: n(0, 1 km, H, 100 km) =", n_air)

# the air layer of a model converted from a single-layer model, and _set_params keeping the air
from NuRadioMC.utilities import medium_base
converted = medium_base.IceModelExpLayers.from_IceModel(medium.greenland_simple())
n_conv = [converted.get_index_of_refraction(np.array([0, 0, z])) for z in [1e-6, 8 * units.km, 100 * units.km]]
testing.assert_allclose(n_conv[:2], [medium_base.N_AIR_SEA_LEVEL, 1 + (medium_base.N_AIR_SEA_LEVEL - 1) / np.e], rtol=1e-12)
assert np.all(np.diff(n_conv) < 0), n_conv
perturbed = medium.greenland_3exp_timing_2025()
perturbed._set_params(perturbed.nN, perturbed.delta_nN, perturbed.ls, perturbed.zs)
testing.assert_allclose(perturbed.get_index_of_refraction(np.array([0, 0, 1e-6])), medium.N_AIR_SUMMIT, rtol=1e-12)

# plane wave: for two receivers on one vertical, dt = int sqrt(n^2 - p^2) dz / c with p = n_air(surface) * sin(zenith)
z_low, z_high = -97 * units.m, -20 * units.m
for compile_numba in [True, False]:
    r = ray.ray_tracing(ice, compile_numba=compile_numba)
    r.set_start_and_end_point(np.array([0, 0, z_low]), np.array([0, 0, z_high]))
    for zenith in np.array([0, 1e-6, 1, 30, 60, 89, 90]) * units.deg:
        p = n_air[0] * np.sin(zenith)
        dt_ref = integrate(lambda z: np.sqrt(n(z)**2 - p**2), z_low, z_high) / constants.c
        dt = r.get_time_difference_plane_wave(zenith, 0)
        testing.assert_allclose(dt, dt_ref, atol=1e-4 * units.ns,
                                err_msg=f"plane wave, zenith {zenith / units.deg} deg, numba {compile_numba}")
        print(f"plane wave zenith {zenith / units.deg:8.6f} deg (numba {compile_numba}): dt = {dt:.5f} ns, "
              f"quadrature {dt_ref:.5f} ns")

# emitter in air -> receiver in ice: p from the horizontal distance, then the travel time by quadrature.
# Near-horizontal rays from low emitters have p > 1 (allowed up to n at the emitter, as n > 1 near the surface).
r = ray.ray_tracing(ice, compile_numba=True)
for distance, height, z_rx in [(100, 10, -97), (1000, 100, -97), (30, 1, -20), (3000, 1000, -20),
                               (1271.4, 10, -97), (2392, 10, -40), (5000, 1, -97), (5000, 100, -20)]:
    distance, height, z_rx = distance * units.m, height * units.m, z_rx * units.m
    horizontal = lambda p: integrate(lambda z: p / np.sqrt(n(z)**2 - p**2), z_rx, height)
    p = brentq(lambda p: horizontal(p) - distance, 1e-6, n(height) * (1 - 1e-12), xtol=1e-14)
    t_ref = integrate(lambda z: n(z)**2 / np.sqrt(n(z)**2 - p**2), z_rx, height) / constants.c

    r.set_start_and_end_point(np.array([distance, 0, height]), np.array([0, 0, z_rx]))
    r.find_solutions()
    assert r.get_number_of_solutions() == 1, r.get_number_of_solutions()
    t = r.get_travel_time(0)
    testing.assert_allclose(t, t_ref, atol=1e-3 * units.ns, err_msg=f"air to ice, {distance}, {height}, {z_rx}")
    testing.assert_allclose(1 / r.get_results()[0]["C0"], p, rtol=1e-9)
    print(f"air to ice: p = {p:.7f}, d = {distance:6.0f} m, h = {height:6.0f} m, z = {z_rx:4.0f} m: t = {t:.4f} ns, "
          f"quadrature {t_ref:.4f} ns")

# single-layer ray tracer (fixed n = 1 in air): numba and python backends; the C++ backend has no air propagation
ice_simple = medium.greenland_simple()


def n_simple(z):
    return ice_simple.n_ice - ice_simple.delta_n * np.exp(z / ice_simple.z_0)


def integrate_simple(f, z):
    return quad(f, z, 0, limit=500, epsabs=1e-12, epsrel=1e-12)[0]


def reach_simple(p, height, z_rx):
    return integrate_simple(lambda z: p / np.sqrt(n_simple(z)**2 - p**2), z_rx) + height * p / np.sqrt(1 - p**2)


geometries = [(100, 10, -97), (1000, 100, -97), (30, 1, -20), (3000, 1000, -20), (1271.4, 10, -97),
              (2392, 10, -40), (5000, 1, -97), (5000, 100, -20), (13880, 1, -97)]
for compile_numba in [True, False]:
    r = ray.ray_tracing(ice_simple, use_cpp=False, compile_numba=compile_numba)
    for distance, height, z_rx in geometries:
        distance, height, z_rx = distance * units.m, height * units.m, z_rx * units.m
        p = brentq(lambda p: reach_simple(p, height, z_rx) - distance, 1e-9, 1 - 1e-15, xtol=1e-16, rtol=1e-15)
        t_ref = (integrate_simple(lambda z: n_simple(z)**2 / np.sqrt(n_simple(z)**2 - p**2), z_rx)
                 + height / np.sqrt(1 - p**2)) / constants.c
        for start, end in [([distance, 0, height], [0, 0, z_rx]), ([0, 0, z_rx], [distance, 0, height])]:
            r.set_start_and_end_point(np.array(start), np.array(end))
            r.find_solutions()
            assert r.get_number_of_solutions() == 1, (start, end, r.get_number_of_solutions())
            testing.assert_allclose(r.get_travel_time(0), t_ref, atol=1e-3 * units.ns,
                                    err_msg=f"single layer, numba {compile_numba}: {start} -> {end}")
    print(f"single layer (numba {compile_numba}): {len(geometries)} air/ice geometries agree with quadrature")

if single_layer_analytic_raytracer.cpp_available:
    r = ray.ray_tracing(ice_simple, use_cpp=True)
    r.set_start_and_end_point(np.array([1000., 0, 100]), np.array([0., 0, -97]))
    testing.assert_raises(NotImplementedError, r.find_solutions)
    print("single layer (C++): raises NotImplementedError for a path through air")

print("T12test_air_to_ice passed without issues")
