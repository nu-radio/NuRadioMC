import numpy as np
from numpy import testing
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.SignalProp.directRayTracing import direct_ray_tracing
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.single_layer_analytic_raytracer import cpp_available
from NuRadioMC.utilities import medium, medium_base
from NuRadioReco.utilities import units

# Geometry from PR #1192: the duplicate-solution check in find_solutions used to drop the second solution
# (it compared logC0 with C0, which coincided here after rounding)
start = np.array([0., 0., -525.]) * units.m
stop = np.array([116., 0., -287.]) * units.m
C0s_ref = [1.284546, 4.175292]

configs = {
    "single layer, python": (medium.southpole_2015(), dict(use_cpp=False, compile_numba=False)),
    "single layer, numba": (medium.southpole_2015(), dict(use_cpp=False, compile_numba=True)),
    "multilayer, python": (
        medium_base.IceModelExpLayers.from_IceModel(medium.southpole_2015()),
        dict(use_cpp=False, compile_numba=False)),
    "multilayer, numba": (
        medium_base.IceModelExpLayers.from_IceModel(medium.southpole_2015()),
        dict(use_cpp=False, compile_numba=True)),
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

# Bottom reflections requested for an ice model without a reflective bottom layer (incl. any multilayer model),
# or for a ray tracer without support for them, raise an error
ice_multilayer = medium_base.IceModelExpLayers.from_IceModel(medium.southpole_2015())
cases = [
    ("single layer", ray.ray_tracing, medium.southpole_2015(), dict(n_reflections=1), ValueError),
    ("multilayer", ray.ray_tracing, ice_multilayer, dict(n_reflections=1), ValueError),
    ("multilayer, config", ray.ray_tracing, ice_multilayer, dict(config={'propagation': {'n_reflections': 1}}), ValueError),
    ("direct", direct_ray_tracing, medium.mooresbay_simple(), dict(n_reflections=1), NotImplementedError),
]
for name, tracer, ice, kwargs, error in cases:
    try:
        tracer(ice, **kwargs)
    except error:
        print(f"{name} with {kwargs}: raises {error.__name__}")
    else:
        raise AssertionError(f"{name} ray tracer accepted bottom reflections ({kwargs})")

print('T11test_find_solutions_known_geometries passed without issues')
