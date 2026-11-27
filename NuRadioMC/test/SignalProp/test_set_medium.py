#!/usr/bin/env python3
"""
Tests for updating the medium of the analytic ray tracer on a per-ray basis.

The reference for every check is a freshly constructed ray tracer for the new medium: an
updated ray tracer has to reproduce it exactly, while not reproducing the results for the
old medium.
"""
import numpy as np

from NuRadioReco.utilities import units
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.MultilayerAnalyticRayTracing.multilayeranalyticraytracing import (
    multi_layer_ray_tracing_2D
)
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.single_layer_analytic_raytracer import ray_tracing_2D
from NuRadioMC.SignalProp.analyticraytracing import ray_tracing
from NuRadioMC.utilities import medium

X1 = np.array([0., 0., -500.])  # emitter position
X2 = np.array([300., 0., -5.])  # receiver position

ATTENUATION_FREQUENCIES = np.array([50., 100., 200.]) * units.MHz

# config with all propagation effects enabled, so that the (cached) attenuation and
# focusing factors are actually calculated
CONFIG = {'propagation': {'attenuate_ice': True, 'focusing': True, 'focusing_limit': 2,
                          'birefringence': False}}

def _trace(propagator, x1=X1, x2=X2):
    """Ray trace between `x1` and `x2` and return the number of solutions."""
    propagator.set_start_and_end_point(x1, x2)
    propagator.find_solutions()
    return propagator.get_number_of_solutions()

def _snapshot(propagator):
    """
    All ray tracing outputs that depend on the medium.

    Two ray tracers for the same medium and geometry have to produce identical snapshots,
    ray tracers for different media have to produce different ones.
    """
    snapshot = []
    for i_solution in range(propagator.get_number_of_solutions()):
        snapshot.append((
            propagator.get_C0(i_solution),
            propagator.get_solution_type(i_solution),
            propagator.get_travel_time(i_solution),
            propagator.get_path_length(i_solution),
            tuple(propagator.get_launch_vector(i_solution)),
            tuple(propagator.get_receive_vector(i_solution)),
            tuple(propagator.get_attenuation(i_solution, ATTENUATION_FREQUENCIES)),
        ))
    return snapshot

def _traced(model, config=None):
    """Construct a ray tracer for `model`, ray trace the reference geometry and return it."""
    propagator = ray_tracing(model, config=config)
    assert _trace(propagator) > 0, "reference geometry has no ray tracing solution"
    return propagator

def test_same_medium_type_update():
    """Updating to a medium of the same type reproduces a freshly constructed ray tracer."""
    model_a = medium.southpole_simple()
    model_b = medium.southpole_2015()

    propagator = _traced(model_a)
    snapshot_a = _snapshot(propagator)
    r2d_a = propagator._r2d

    propagator.set_medium(model_b)
    assert propagator._medium is model_b
    # solutions (and the geometry they belong to) are invalidated
    assert propagator._results is None
    assert propagator._x1 is None and propagator._x2 is None
    # the 2D raytracing module is kept and updated in place
    assert propagator._r2d is r2d_a
    assert isinstance(propagator._r2d, ray_tracing_2D)
    assert propagator._r2d.medium is model_b

    _trace(propagator)
    snapshot_b = _snapshot(propagator)
    assert snapshot_b == _snapshot(_traced(model_b))
    assert snapshot_b != snapshot_a

    # updating to the current medium is a no-op and keeps the solutions
    propagator.set_medium(model_b)
    assert propagator._results is not None
    assert _snapshot(propagator) == snapshot_b

def test_switch_between_medium_types():
    """Switching between single layer and multilayer media rebuilds the 2D raytracing module."""
    model_single = medium.southpole_simple()
    model_multi = medium.greenland_3exp_density()

    propagator = _traced(model_single)

    propagator.set_medium(model_multi)
    assert isinstance(propagator._r2d, multi_layer_ray_tracing_2D)
    assert propagator._r2d.medium is model_multi
    _trace(propagator)
    assert _snapshot(propagator) == _snapshot(_traced(model_multi))

    propagator.set_medium(model_single)
    assert isinstance(propagator._r2d, ray_tracing_2D)
    assert propagator._r2d.medium is model_single
    _trace(propagator)
    assert _snapshot(propagator) == _snapshot(_traced(model_single))

def test_caches_and_focusing_raytracer_invalidated():
    """
    Cached attenuation/focusing factors and the internal raytracer used for the numerical
    focusing correction must not survive a medium update.
    """
    model_a = medium.southpole_simple()
    model_b = medium.southpole_2015()

    propagator = _traced(model_a, config=CONFIG)
    attenuation_a = propagator.get_attenuation(0, ATTENUATION_FREQUENCIES)
    focusing_a = propagator.get_focusing(0)
    assert len(propagator._cache_attenuation) > 0
    assert len(propagator._cache_focusing) > 0
    # the numerical focusing correction uses a second raytracer that is bound to the medium
    assert hasattr(propagator, "_r1")

    propagator.set_medium(model_b)
    assert propagator._cache_attenuation == {}
    assert propagator._cache_focusing == {}
    assert not hasattr(propagator, "_r1")

    _trace(propagator)
    reference = _traced(model_b, config=CONFIG)
    assert np.allclose(propagator.get_attenuation(0, ATTENUATION_FREQUENCIES),
                       reference.get_attenuation(0, ATTENUATION_FREQUENCIES))
    assert np.isclose(propagator.get_focusing(0), reference.get_focusing(0))
    # the stored factors have to differ from the ones calculated before the update,
    # otherwise this test would not catch stale values
    assert not np.allclose(attenuation_a, propagator.get_attenuation(0, ATTENUATION_FREQUENCIES)) \
        or not np.isclose(focusing_a, propagator.get_focusing(0))

if __name__ == "__main__":
    for test in [test_same_medium_type_update, test_switch_between_medium_types,
                 test_caches_and_focusing_raytracer_invalidated]:
        print("{} ...".format(test.__name__))
        test()

    print("\nRay tracing medium update tests passed")
