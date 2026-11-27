#!/usr/bin/env python3
"""
Compare the ray tracing solutions of the C++ and the Python backend of the single-layer
analytic ray tracer, including reflections off a reflective bottom layer (Moore's Bay).

For random emitter positions, both backends must find the same solutions (number, solution type,
number of bottom reflections and reflection case) with the same C0, travel time, path length,
launch and receive vectors and attenuation.
"""
import numpy as np
from numpy import testing
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.single_layer_analytic_raytracer import cpp_available
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
from NuRadioReco.utilities.coding_helper import get_rng

if not cpp_available:
    raise RuntimeError("The C++ ray tracer is not available, but is needed for this test")

sites = {
    'southpole': dict(ice_model=medium.southpole_2015(), zmax=-2.5 * units.km, n_reflections=0),
    'mooresbay': dict(ice_model=medium.mooresbay_simple(), zmax=-0.5 * units.km, n_reflections=2),
}

n_events = 100
x_receiver = np.array([0., 0., -5.]) * units.m
frequencies = np.linspace(50, 500, 10) * units.MHz


def get_solutions(r, x):
    """ Returns the solutions of ray tracer `r` for emitter position `x`, sorted by (reflection, reflection_case, C0) """
    r.set_start_and_end_point(x, x_receiver)
    r.find_solutions()
    solutions = []
    for iS, result in enumerate(r.get_results()):
        solutions.append(dict(
            key=(result['reflection'], result['reflection_case'], r.get_solution_type(iS)),
            C0=result['C0'],
            travel_time=r.get_travel_time(iS),
            path_length=r.get_path_length(iS),
            launch_vector=r.get_launch_vector(iS),
            receive_vector=r.get_receive_vector(iS),
            attenuation=r.get_attenuation(iS, frequencies),
        ))
    return sorted(solutions, key=lambda s: (s['key'][:2], s['C0']))


for site, options in sites.items():
    rng = get_rng(1234)
    rr = rng.triangular(50, 1000, 1000, n_events) * units.m
    phi = rng.uniform(0, 2 * np.pi, n_events)
    zz = rng.uniform(options['zmax'], 0, n_events)
    points = np.array([rr * np.cos(phi), rr * np.sin(phi), zz]).T

    tracers = {
        backend: ray.ray_tracing(options['ice_model'], n_reflections=options['n_reflections'], use_cpp=use_cpp)
        for backend, use_cpp in [('C++', True), ('Python', False)]}

    n_solutions = 0
    for x in points:
        sols_cpp, sols_py = (get_solutions(r, x) for r in tracers.values())
        msg = f"{site}: emitter at {np.round(x / units.m, 2)} m"
        assert [s['key'] for s in sols_cpp] == [s['key'] for s in sols_py], \
            f"{msg}: solutions differ: C++ {[s['key'] for s in sols_cpp]}, Python {[s['key'] for s in sols_py]}"

        for s_cpp, s_py in zip(sols_cpp, sols_py):
            testing.assert_allclose(s_py['C0'], s_cpp['C0'], rtol=1e-5, err_msg=msg)
            testing.assert_allclose(s_py['travel_time'], s_cpp['travel_time'], atol=0.1 * units.ns, err_msg=msg)
            testing.assert_allclose(s_py['path_length'], s_cpp['path_length'], atol=1 * units.cm, err_msg=msg)
            testing.assert_allclose(s_py['launch_vector'], s_cpp['launch_vector'], atol=1e-5, err_msg=msg)
            testing.assert_allclose(s_py['receive_vector'], s_cpp['receive_vector'], atol=1e-5, err_msg=msg)
            testing.assert_allclose(s_py['attenuation'], s_cpp['attenuation'], rtol=1e-2, atol=1e-3, err_msg=msg)

        n_solutions += len(sols_cpp)

    print(f"{site}: C++ and Python agree for {n_solutions} solutions of {n_events} emitter positions")

print('T12test_compare_cpp_python passed without issues')
