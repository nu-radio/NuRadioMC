#!/usr/bin/env python3
"""
Check that the RadioPropa ray tracer raises NotImplementedError for the cases it does not support
(bottom reflections, start or end points above the ice, the 'minimizing' mode) and still works
for a plain in-ice geometry.

RadioPropa is an optional dependency: without it, the test is skipped.
"""
import copy
import importlib.util
import sys

import numpy as np

if importlib.util.find_spec("radiopropa") is None:
    print("radiopropa is not installed, skipping T12test_radiopropa_unsupported")
    sys.exit(0)

from NuRadioMC.SignalProp import radioproparaytracing
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units


def assert_raises_not_implemented(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except NotImplementedError as e:
        print(f"raised NotImplementedError as expected: {e}")
        return

    raise AssertionError(f"{func.__name__} did not raise NotImplementedError")


ice = medium.greenland_simple()
x_ice_deep = np.array([300., 0., -800.]) * units.m
x_ice_shallow = np.array([0., 0., -100.]) * units.m
x_air = np.array([0., 0., 10.]) * units.m

# bottom reflections: requested for an ice model with a reflective layer
assert_raises_not_implemented(radioproparaytracing.radiopropa_ray_tracing, medium.mooresbay_simple(), n_reflections=1)

# the same ice model without reflections is fine
radioproparaytracing.radiopropa_ray_tracing(medium.mooresbay_simple(), n_reflections=0)

# start or end point above the ice (issue #968), in both directions
rt = radioproparaytracing.radiopropa_ray_tracing(ice)
assert_raises_not_implemented(rt.set_start_and_end_point, x_air, x_ice_shallow)
assert_raises_not_implemented(rt.set_start_and_end_point, x_ice_shallow, x_air)

# 'minimizing' mode
config = copy.deepcopy(rt._config)
config["propagation"]["radiopropa"]["mode"] = "minimizing"
assert_raises_not_implemented(radioproparaytracing.radiopropa_ray_tracing, ice, config=config)

# a plain in-ice geometry still finds the direct and the reflected solution
rt = radioproparaytracing.radiopropa_ray_tracing(ice)
rt.set_start_and_end_point(x_ice_deep, x_ice_shallow)
rt.find_solutions()
assert rt.get_number_of_solutions() == 2, f"expected 2 solutions, found {rt.get_number_of_solutions()}"

print("T12test_radiopropa_unsupported passed without issues")
