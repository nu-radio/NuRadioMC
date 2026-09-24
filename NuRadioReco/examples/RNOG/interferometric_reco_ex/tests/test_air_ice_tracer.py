"""Properties of the air-to-ice ray tracer and of the `air_ice` propagator."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from NuRadioMC.SignalProp import analyticraytracing, propagation
from NuRadioMC.utilities.medium import greenland_simple
from NuRadioReco.utilities import units

air_ice_raytracer = pytest.importorskip('NuRadioMC.SignalProp.air_ice_raytracer')
airIceRayTracing = pytest.importorskip('NuRadioMC.SignalProp.airIceRayTracing')

C_NS = 0.299792458
ANT = np.array([0.0, 0.0, -97.55])
GEOMETRIES = [
    pytest.param((50.0, 5.0), id='rho50_h5'),
    pytest.param((100.0, 20.0), id='rho100_h20'),
    pytest.param((200.0, 50.0), id='rho200_h50'),
    pytest.param((100.0, 1.0), id='rho100_h1'),
    pytest.param((20.0, 100.0), id='rho20_h100'),
    pytest.param((300.0, 150.0), id='rho300_h150'),
]


@pytest.fixture(scope='module')
def ice():
    return greenland_simple()


@pytest.fixture(scope='module')
def tracer2d(ice):
    return analyticraytracing.ray_tracing_2D(ice)


def _trace(ice, rho, h, phi_deg=30.0):
    src = np.array([rho * np.cos(np.radians(phi_deg)), rho * np.sin(np.radians(phi_deg)), h])
    t = air_ice_raytracer.AirIceRaytracer(ice, src, ANT, precision=0.001 * units.deg, max_iterations=100)
    t.run()
    return src, t


def _fermat_time(tracer2d, rho, h, x_entry):
    """Travel time through a given surface entry point: straight air leg plus in-ice leg."""
    t_air = np.hypot(rho - x_entry, h) / C_NS
    sols = tracer2d.find_solutions(np.array([0.0, ANT[2]]), np.array([x_entry, -0.01]))
    times = []
    for s in sols:
        ang = tracer2d.get_receive_angle(np.array([0.0, ANT[2]]), np.array([x_entry, -0.01]),
                                         s['C0'], s['reflection'], s['reflection_case'])
        if ang >= np.pi / 2:
            times.append(tracer2d.get_travel_time(np.array([0.0, ANT[2]]), np.array([x_entry, -0.01]), s['C0']))
    return t_air + min(times) if times else np.inf


@pytest.mark.airice
@pytest.mark.parametrize('geom', GEOMETRIES)
def test_fermat_principle(ice, tracer2d, geom):
    """The Snell-matched entry point minimises the total travel time over all entry points."""
    rho, h = geom
    src, t = _trace(ice, rho, h)
    x_snell = np.hypot(*t.get_surface_intersection_point()[:2])
    res = minimize_scalar(lambda x: _fermat_time(tracer2d, rho, h, x), bounds=(0.5, rho - 1e-3),
                          method='bounded', options={'xatol': 1e-3})
    assert abs(res.x - x_snell) < 0.05, (res.x, x_snell)
    assert abs(res.fun - t.get_signal_travel_time('total')) < 0.01, (res.fun, t.get_signal_travel_time('total'))


@pytest.mark.airice
@pytest.mark.parametrize('geom', GEOMETRIES)
def test_snell_at_entry(ice, geom):
    """Angles on both sides of the entry point obey Snell's law to the tracer's precision."""
    rho, h = geom
    _, t = _trace(ice, rho, h)
    theta_air, theta_ice = t.get_entry_incidence_angles()
    n_surf = ice.get_index_of_refraction([0, 0, air_ice_raytracer.SURFACE_DEPTH])
    assert abs(np.sin(theta_air) - n_surf * np.sin(theta_ice)) < 2e-4


@pytest.mark.airice
@pytest.mark.parametrize('geom', GEOMETRIES)
def test_travel_time_bounds(ice, geom):
    """Total time exceeds the straight-line vacuum time and is below the straight-line deep-ice time."""
    rho, h = geom
    src, t = _trace(ice, rho, h)
    dist = np.linalg.norm(src - ANT)
    tt = t.get_signal_travel_time('total')
    assert dist / C_NS < tt < dist * 1.78 / C_NS
    assert t.get_signal_travel_time('air') + t.get_signal_travel_time('ice') == pytest.approx(tt)
    assert t.get_path_length('air') + t.get_path_length('ice') == pytest.approx(t.get_path_length('total'))


@pytest.mark.airice
def test_continuity_across_surface(ice):
    """Inside the exit cone a source just above the surface and one just below agree.

    From a 97.5 m deep antenna, rays leave the ice only from entry points within about
    80 m horizontally (total internal reflection beyond); farther out an air source is
    reached only by the grazing critical-angle path, which is faster than the in-ice
    direct ray to a point just below the surface, so continuity is expected only inside
    the cone.
    """
    rho = 60.0
    _, above = _trace(ice, rho, 0.02, phi_deg=0.0)
    rt = analyticraytracing.ray_tracing(ice)
    rt.set_start_and_end_point(np.array([rho, 0.0, -0.02]), ANT)
    rt.find_solutions()
    below = min(rt.get_travel_time(i) for i in range(rt.get_number_of_solutions()))
    assert abs(above.get_signal_travel_time('total') - below) < 0.2


@pytest.mark.airice
def test_swap_symmetry(ice):
    """Swapping start and end gives the same travel time and the same entry point."""
    src = np.array([80.0, 60.0, 30.0])
    a = air_ice_raytracer.AirIceRaytracer(ice, src, ANT)
    a.run()
    b = air_ice_raytracer.AirIceRaytracer(ice, ANT, src)
    b.run()
    assert a.get_signal_travel_time('total') == pytest.approx(b.get_signal_travel_time('total'), abs=1e-6)
    assert np.allclose(a.get_surface_intersection_point(), b.get_surface_intersection_point(), atol=1e-3)


@pytest.mark.airice
@pytest.mark.parametrize('geom', GEOMETRIES)
def test_propagator_matches_tracer_and_vectors(ice, geom):
    """The registered `air_ice` propagator reproduces the tracer and returns consistent vectors."""
    rho, h = geom
    src, t = _trace(ice, rho, h)
    prop_cls = propagation.get_propagation_module('air_ice')
    prop = prop_cls(ice)
    prop.set_start_and_end_point(src, ANT)
    prop.find_solutions()
    assert prop.get_number_of_solutions() == 1
    assert prop.get_solution_type(0) == 1
    assert prop.get_travel_time(0) == pytest.approx(t.get_signal_travel_time('total'), abs=1e-9)
    assert prop.get_path_length(0) == pytest.approx(t.get_path_length('total'), abs=1e-9)
    lv, rv = prop.get_launch_vector(0), prop.get_receive_vector(0)
    assert np.linalg.norm(lv) == pytest.approx(1.0) and np.linalg.norm(rv) == pytest.approx(1.0)
    assert lv[2] < 0, 'launch vector from an air source must point down'
    assert rv[2] > 0, 'receive vector at a deep antenna must point up toward the entry point'
    entry = t.get_surface_intersection_point()
    assert np.dot(lv[:2], (entry - src)[:2]) > 0
    assert np.dot(rv[:2], (entry - ANT)[:2]) > 0
    theta_air, _ = t.get_entry_incidence_angles()
    assert np.arccos(-lv[2]) == pytest.approx(theta_air, abs=1e-6)


@pytest.mark.airice
def test_propagator_delegates_in_ice_pairs(ice):
    """With both points in the ice the propagator returns the analytic tracer's solutions."""
    src = np.array([100.0, 0.0, -20.0])
    prop = propagation.get_propagation_module('air_ice')(ice)
    prop.set_start_and_end_point(src, ANT)
    prop.find_solutions()
    ref = analyticraytracing.ray_tracing(ice)
    ref.set_start_and_end_point(src, ANT)
    ref.find_solutions()
    assert prop.get_number_of_solutions() == ref.get_number_of_solutions() >= 1
    for i in range(ref.get_number_of_solutions()):
        assert prop.get_travel_time(i) == pytest.approx(ref.get_travel_time(i))
        assert prop.get_solution_type(i) == ref.get_solution_type(i)


@pytest.mark.airice
def test_propagator_has_no_air_to_air_solution(ice):
    """Two points in air are outside the model and must return no solution."""
    prop = propagation.get_propagation_module('air_ice')(ice)
    prop.set_start_and_end_point(np.array([0.0, 0.0, 10.0]), np.array([50.0, 0.0, 20.0]))
    prop.find_solutions()
    assert not prop.has_solution()
