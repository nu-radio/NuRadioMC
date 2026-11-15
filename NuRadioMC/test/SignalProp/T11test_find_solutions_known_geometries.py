import numpy as np
from numpy import testing
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.single_layer_analytic_raytracer import cpp_available
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units

# Geometry from PR #1192: the duplicate-solution check in find_solutions used to drop the second solution
# (it compared logC0 with C0, which coincided here after rounding)
start = np.array([0., 0., -525.]) * units.m
stop = np.array([116., 0., -287.]) * units.m
C0s_ref = [1.284546, 4.175292]

configs = {
    "single layer, python": (medium.southpole_2015(), dict(use_cpp=False, compile_numba=False)),
    "single layer, numba": (medium.southpole_2015(), dict(use_cpp=False, compile_numba=True)),
    "multilayer, python": (medium.southpole_2015_layered(), dict(use_cpp=False, compile_numba=False)),
    "multilayer, numba": (medium.southpole_2015_layered(), dict(use_cpp=False, compile_numba=True)),
}
if cpp_available:
    configs["single layer, C++"] = (medium.southpole_2015(), dict(use_cpp=True))

for name, (ice, kwargs) in configs.items():
    r = ray.ray_tracing(ice, n_reflections=0, **kwargs)
    r.set_start_and_end_point(start, stop)
    r.find_solutions()
    C0s = sorted(result['C0'] for result in r.get_results())
    testing.assert_allclose(C0s, C0s_ref, rtol=1e-5, err_msg=name)
    print(f"{name}: {np.round(C0s, 6)}")

print('T11test_find_solutions_known_geometries passed without issues')
