#!/usr/bin/env python3
"""
Tests for NuRadioReco.detector.antennapattern

The tests are split into two groups: everything up to `test_provider_buffering` runs
without any antenna model file, the tests after it need RNOG_vpol_4inch_center_n1.73
(~10 MB, downloaded on first use).

Regression test for the bracketing of non-equidistant grids: the antenna models are
not required to be sampled on an equally spaced grid (e.g. RNOG_vpol_4inch_center_n1.73
has theta nodes [0, 30, 60, 70, 80, 90, 100, 110, 120, 150, 180] deg). Deriving the
node indices from the grid boundaries instead of looking them up produced
discontinuities of up to 70 % in the vector effective length.
"""
import numpy as np

from NuRadioReco.detector.antennapattern import (
    AntennaPattern, AntennaPatternAnalytic, AntennaPatternProvider, get_bracketing_indices,
    get_group_delay, interpolate_linear, interpolate_linear_vectorized, is_equidistant)
from NuRadioReco.utilities import units

TEST_MODEL = "RNOG_vpol_4inch_center_n1.73"


# --------------------------------------------------------------------------------------
# interpolation helpers
# --------------------------------------------------------------------------------------

def test_interpolate_linear():
    y0, y1 = 1 + 1j, 3 - 2j

    for method in ["complex", "magphase"]:
        # the endpoints have to be reproduced exactly by both methods
        assert np.isclose(interpolate_linear(0., 0., 1., y0, y1, method), y0)
        assert np.isclose(interpolate_linear(1., 0., 1., y0, y1, method), y1)

    # complex interpolation is linear in real and imaginary part
    assert np.isclose(interpolate_linear(0.5, 0., 1., y0, y1), 0.5 * (y0 + y1))

    # magphase interpolation is linear in magnitude
    mid = interpolate_linear(0.5, 0., 1., y0, y1, "magphase")
    assert np.isclose(np.abs(mid), 0.5 * (np.abs(y0) + np.abs(y1)))

    # a degenerate interval returns the first value
    assert interpolate_linear(5., 1., 1., y0, y1) == y0

    try:
        interpolate_linear(0.5, 0., 1., y0, y1, "does_not_exist")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("unknown interpolation method has to raise")


def test_interpolate_linear_vectorized():
    """The vectorized variant has to agree with the scalar one, including x0 == x1"""
    x = np.array([0.5, 1.0, 2.0])
    x0, x1 = np.array([0., 1., 2.]), np.array([1., 1., 3.])
    y0 = np.array([1 + 0j, 5 + 0j, 2 + 2j])
    y1 = np.array([3 + 0j, 9 + 0j, 4 + 4j])

    for method in ["complex", "magphase"]:
        vectorized = interpolate_linear_vectorized(x, x0, x1, y0, y1, method)
        scalar = [interpolate_linear(x[i], x0[i], x1[i], y0[i], y1[i], method) for i in range(len(x))]
        assert np.allclose(vectorized, scalar), "vectorized and scalar disagree for {}".format(method)


def test_get_bracketing_indices():
    nodes = np.array([0., 30., 60., 70., 80., 90., 100., 110., 120., 150., 180.])

    # values in between two nodes
    for x, expected in [(15., (0, 1)), (65., (2, 3)), (135., (8, 9)), (179.9, (9, 10))]:
        assert tuple(np.atleast_1d(i)[0] for i in get_bracketing_indices(x, nodes)) == expected, \
            "wrong bracket for x = {}".format(x)

    # values on a node have to be bracketed such that the interpolation returns the node
    for x in nodes:
        i_lower, i_upper = get_bracketing_indices(x, nodes)
        assert nodes[i_lower] <= x <= nodes[i_upper]
        assert i_upper - i_lower <= 1

    # out-of-range values are clamped to the outermost interval
    assert tuple(np.atleast_1d(i)[0] for i in get_bracketing_indices(-10., nodes)) == (0, 1)
    assert tuple(np.atleast_1d(i)[0] for i in get_bracketing_indices(200., nodes)) == (9, 10)

    # array input and a single-node grid
    i_lower, i_upper = get_bracketing_indices(np.array([15., 65.]), nodes)
    assert np.all(i_lower == [0, 2]) and np.all(i_upper == [1, 3])
    i_lower, i_upper = get_bracketing_indices(np.array([1., 2.]), np.array([5.]))
    assert np.all(i_lower == 0) and np.all(i_upper == 0)


def test_equidistant_shortcut():
    """On an equally spaced grid the fast index calculation must match the lookup"""
    assert not is_equidistant(np.array([0., 30., 60., 70., 80., 90., 100., 110., 120., 150., 180.]))
    assert is_equidistant(np.arange(0., 181., 5.))
    assert is_equidistant(np.arange(0., 181., 5.) + 1e-6 * np.arange(37))  # rounding of the raw file

    grid = np.arange(0., 181., 5.)
    x = np.concatenate([np.linspace(0, 180, 1001), grid])
    for i_slow, i_fast in zip(get_bracketing_indices(x, grid),
                              get_bracketing_indices(x, grid, equidistant=True)):
        assert np.all(grid[i_slow] == grid[i_fast])


def test_get_group_delay():
    """A pure delay has a constant group delay of exactly that delay"""
    df = 10 * units.MHz
    freqs = np.arange(0, 1000) * df
    delay = 12.3 * units.ns

    group_delay = get_group_delay(np.exp(-1j * 2 * np.pi * freqs * delay), df)
    assert np.allclose(group_delay, delay)


# --------------------------------------------------------------------------------------
# coordinate transformation (AntennaPatternBase)
# --------------------------------------------------------------------------------------

def test_antenna_rotation():
    antenna = AntennaPatternAnalytic("analytic_LPDA")
    own_orientation = [antenna._orientation_theta, antenna._orientation_phi,
                       antenna._rotation_theta, antenna._rotation_phi]

    # deploying the antenna exactly as it was simulated must not rotate anything
    rotation, inverse, polar_roll = antenna._get_antenna_rotation(*own_orientation)
    assert np.allclose(rotation, np.eye(3)) and np.allclose(inverse, np.eye(3))
    assert polar_roll, "the identity is a (zero) roll about the polar axis"

    # a roll about the polar axis is cancelled by the shift of the azimuth it causes
    for roll in [30 * units.deg, 90 * units.deg, 200 * units.deg]:
        rolled = list(own_orientation)
        rolled[3] += roll
        _, _, polar_roll = antenna._get_antenna_rotation(*rolled)
        assert polar_roll, "a roll of {:.0f} deg about the polar axis is not detected".format(
            roll / units.deg)

    # tilting the antenna is not
    tilted = [45 * units.deg, 0., 135 * units.deg, 0.]
    assert not antenna._get_antenna_rotation(*tilted)[2]

    for zenith, azimuth in [(30 * units.deg, 40 * units.deg), (120 * units.deg, 200 * units.deg)]:
        theta, phi = antenna._get_theta_and_phi(zenith, azimuth, *own_orientation)
        assert np.isclose(theta, zenith)
        assert np.isclose(np.exp(1j * phi), np.exp(1j * azimuth))  # phi is returned in (-180, 180] deg

    # orientation and rotation have to be perpendicular
    try:
        antenna._get_antenna_rotation(0., 0., 0., 0.)
    except AssertionError:
        pass
    else:
        raise AssertionError("a degenerate antenna orientation has to raise")


# --------------------------------------------------------------------------------------
# analytic antenna models
# --------------------------------------------------------------------------------------

def test_analytic_patterns():
    freqs = np.arange(50, 1001, 10) * units.MHz
    zeniths = np.arange(0, 181, 10) * units.deg
    azimuths = np.arange(0, 360, 30) * units.deg

    for model, max_VEL in [("analytic_LPDA", 0.55 * units.m),
                           ("analytic_VPol", 0.18 * units.m),
                           ("analytic_HPol", 0.055 * units.m)]:
        antenna = AntennaPatternAnalytic(model)
        scaled = AntennaPatternAnalytic(model, max_VEL=2 * max_VEL)
        orientation = [antenna._orientation_theta, antenna._orientation_phi,
                       antenna._rotation_theta, antenna._rotation_phi]

        peak, peak_scaled = 0, 0
        for zenith in zeniths:
            for azimuth in azimuths:
                vel = antenna.get_antenna_response_vectorized(freqs, zenith, azimuth, *orientation)
                assert np.all(np.isfinite(vel["theta"])) and np.all(np.isfinite(vel["phi"]))
                assert len(vel["theta"]) == len(freqs)
                peak = max(peak, np.abs(vel["theta"]).max(), np.abs(vel["phi"]).max())

                vel = scaled.get_antenna_response_vectorized(freqs, zenith, azimuth, *orientation)
                peak_scaled = max(peak_scaled, np.abs(vel["theta"]).max(), np.abs(vel["phi"]).max())

        assert np.isclose(peak, max_VEL), \
            "{}: peak |VEL| {:.4f} != max_VEL {:.4f}".format(model, peak, max_VEL)
        assert np.isclose(peak_scaled, 2 * max_VEL), "{}: max_VEL is not applied".format(model)

    # the cutoff frequency has to be picked up as well
    default = AntennaPatternAnalytic("analytic_VPol")
    shifted = AntennaPatternAnalytic("analytic_VPol", cutoff_freq=400 * units.MHz)
    assert shifted._cutoff_freq == 400 * units.MHz
    orientation = [default._orientation_theta, default._orientation_phi,
                   default._rotation_theta, default._rotation_phi]
    assert not np.allclose(
        default.get_antenna_response_vectorized(freqs, 90 * units.deg, 0., *orientation)["theta"],
        shifted.get_antenna_response_vectorized(freqs, 90 * units.deg, 0., *orientation)["theta"])


def test_parametric_phase():
    antenna = AntennaPatternAnalytic("analytic_LPDA")
    freqs = np.linspace(50, 1000, 100) * units.MHz

    for phase_type in ["frontlobe_lpda", "side_lpda", "back_lpda", "theoretical",
                       "VPol_third_order", "HPol_third_order"]:
        phase = antenna.parametric_phase(freqs, phase_type)
        assert np.shape(phase) == np.shape(freqs)
        assert np.all(np.isfinite(phase)), "{} returns non-finite values".format(phase_type)


# --------------------------------------------------------------------------------------
# AntennaPatternProvider
# --------------------------------------------------------------------------------------

def test_provider_buffering():
    provider = AntennaPatternProvider()

    # the provider is a singleton and buffers the (expensive to load) patterns
    assert AntennaPatternProvider() is provider
    antenna = provider.load_antenna_pattern("analytic_LPDA")
    assert provider.load_antenna_pattern("analytic_LPDA") is antenna

    # names starting with "analytic" are dispatched to the analytic implementation
    assert isinstance(antenna, AntennaPatternAnalytic)

    provider = AntennaPatternProvider()
    antenna = provider.load_antenna_pattern("analytic_LPDA")
    AntennaPatternProvider()  # __init__ runs again on the singleton
    assert provider.load_antenna_pattern("analytic_LPDA") is antenna

    provider = AntennaPatternProvider()
    provider.load_antenna_pattern("analytic_LPDA")
    antenna = provider.load_antenna_pattern("analytic_LPDA", max_VEL=1 * units.m)
    assert antenna._max_VEL == 1 * units.m



# --------------------------------------------------------------------------------------
# AntennaPattern, these need the antenna model file
# --------------------------------------------------------------------------------------

def test_interpolation_at_nodes():
    """At a grid node the interpolation has to return the tabulated value"""
    antenna = AntennaPattern(TEST_MODEL)

    for i_freq in [0, 100, antenna.n_freqs - 1]:
        for i_theta in range(antenna.n_theta):
            for i_phi in range(0, antenna.n_phi, 3):
                vel = antenna._get_antenna_response_vectorized_raw(
                    np.array([antenna.frequencies[i_freq]]),
                    antenna.theta_angles[i_theta], antenna.phi_angles[i_phi])

                assert np.allclose(vel[0], antenna.VEL_theta[i_freq, i_phi, i_theta], atol=1e-15)
                assert np.allclose(vel[1], antenna.VEL_phi[i_freq, i_phi, i_theta], atol=1e-15)


def test_magphase_agrees_at_nodes():
    """Both interpolation methods are only allowed to differ in between the nodes"""
    complex_ = AntennaPattern(TEST_MODEL, interpolation_method="complex")
    magphase = AntennaPattern(TEST_MODEL, interpolation_method="magphase")
    freqs = np.array([complex_.frequencies[100]])

    for i_theta in range(complex_.n_theta):
        for i_phi in range(0, complex_.n_phi, 3):
            theta, phi = complex_.theta_angles[i_theta], complex_.phi_angles[i_phi]
            assert np.allclose(
                np.array(complex_._get_antenna_response_vectorized_raw(freqs, theta, phi)),
                np.array(magphase._get_antenna_response_vectorized_raw(freqs, theta, phi)),
                atol=1e-15)


def test_phi_wrapping():
    """Azimuth angles outside the tabulated range are wrapped, not clipped"""
    antenna = AntennaPatternProvider().load_antenna_pattern(TEST_MODEL)
    freqs = np.array([200., 300.]) * units.MHz
    orientation = [0, 0, np.pi / 2, np.pi / 2]

    for azimuth in [30, 100, 200]:
        inside = antenna.get_antenna_response_vectorized(
            freqs, 1., azimuth * units.deg, *orientation)
        wrapped = antenna.get_antenna_response_vectorized(
            freqs, 1., (azimuth - 360) * units.deg, *orientation)

        for component in ["theta", "phi"]:
            assert np.allclose(inside[component], wrapped[component]), \
                "azimuth {} deg and {} deg give different responses".format(azimuth, azimuth - 360)


def test_out_of_range():
    """Directions outside the tabulated range return zeros"""
    antenna = AntennaPattern(TEST_MODEL)
    freqs = np.array([200., 300., 400.]) * units.MHz

    vel = antenna._get_antenna_response_vectorized_raw(freqs, 200 * units.deg, 0.)
    assert np.all(np.array(vel) == 0)

    # zeros still have to be returned per frequency, just like the in-range path
    assert np.shape(vel) == (2, len(freqs))

    # this model covers the full sphere, so the public method cannot reach the branch above
    # (a zenith of 200 deg maps to theta = 160 deg, which is tabulated)
    vel = antenna.get_antenna_response_vectorized(
        freqs, 200 * units.deg, 0., 0, 0, np.pi / 2, np.pi / 2)
    assert np.shape(vel["theta"]) == (len(freqs),)


def test_vel_is_continuous():
    """The interpolated VEL must not jump between two adjacent zenith angles"""
    antenna = AntennaPatternProvider().load_antenna_pattern(TEST_MODEL)

    # the theta/phi unit vectors are degenerate at the poles, stay away from them
    zeniths = np.arange(1, 179, 0.1) * units.deg
    orientation = [0, 0, np.pi / 2, np.pi / 2]

    for freq in np.array([100., 200., 600.]) * units.MHz:
        for azimuth in np.arange(0, 360, 45) * units.deg:
            response = [antenna.get_antenna_response_vectorized(
                np.array([freq]), zenith, azimuth, *orientation) for zenith in zeniths]

            for component in ["theta", "phi"]:
                vel = np.abs(np.array([r[component] for r in response])).flatten()
                steps = np.abs(np.diff(vel))

                # a smooth curve on this fine a grid stays close to its typical step size
                assert steps.max() < 10 * np.median(steps), \
                    "VEL_{} jumps by {:.2e} at zenith = {:.1f} deg (azimuth = {:.0f} deg, " \
                    "{:.0f} MHz), typical step is {:.2e}".format(
                        component, steps.max(), zeniths[np.argmax(steps)] / units.deg,
                        azimuth / units.deg, freq / units.MHz, np.median(steps))


if __name__ == "__main__":
    without_antenna_file = [
        test_interpolate_linear, test_interpolate_linear_vectorized, test_get_bracketing_indices,
        test_equidistant_shortcut, test_get_group_delay, test_antenna_rotation,
        test_analytic_patterns, test_parametric_phase, test_provider_buffering]

    with_antenna_file = [
        test_interpolation_at_nodes, test_magphase_agrees_at_nodes, test_phi_wrapping,
        test_out_of_range, test_vel_is_continuous]

    for test in without_antenna_file + with_antenna_file:
        print("{} ...".format(test.__name__))
        test()

    print("\nAntenna pattern tests passed")
