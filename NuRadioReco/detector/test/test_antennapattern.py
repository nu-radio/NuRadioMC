#!/usr/bin/env python3
"""
Tests for NuRadioReco.detector.antennapattern

The tests are split into two groups: everything up to `test_provider_buffering` runs
without any antenna model file, the tests after it need antenna models which are
downloaded on first use: RNOG_vpol_4inch_center_n1.73 (~10 MB) and, for the reference
values, RNOG_vpol_v3_5inch_center_n1.74 and createLPDA_100MHz_v2_InfFirn_n1.4 (~700 MB
each).

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
    assert np.allclose(antenna._get_antenna_rotation(*own_orientation), np.eye(3))

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

    # instantiating the provider again must not throw away the buffered patterns
    AntennaPatternProvider()
    assert provider.load_antenna_pattern("analytic_LPDA") is antenna

    # kwargs are part of the identity of a pattern and must not be served from the buffer
    configured = provider.load_antenna_pattern("analytic_LPDA", max_VEL=1 * units.m)
    assert configured._max_VEL == 1 * units.m
    assert configured is not antenna
    assert provider.load_antenna_pattern("analytic_LPDA", max_VEL=1 * units.m) is configured
    assert provider.load_antenna_pattern("analytic_LPDA") is antenna


# --------------------------------------------------------------------------------------
# AntennaPattern, these need the antenna model file
# --------------------------------------------------------------------------------------

def test_interpolation_at_nodes():
    """At a grid node the interpolation has to return the tabulated value"""
    antenna = AntennaPattern(TEST_MODEL)

    for i_freq in [0, 100, antenna.n_freqs - 1]:
        for i_theta in range(antenna.n_theta):
            for i_phi in range(0, antenna.n_phi, 3):
                index = antenna._get_index(i_freq, i_theta, i_phi)
                vel = antenna._get_antenna_response_vectorized_raw(
                    np.array([antenna.frequencies[i_freq]]),
                    antenna.theta_angles[i_theta], antenna.phi_angles[i_phi])

                assert np.allclose(vel[0], antenna.VEL_theta[index], atol=1e-15)
                assert np.allclose(vel[1], antenna.VEL_phi[index], atol=1e-15)


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


# --------------------------------------------------------------------------------------
# reference values, these need the RNOG_vpol_v3_5inch_center_n1.74 and
# createLPDA_100MHz_v2_InfFirn_n1.4 antenna models
# --------------------------------------------------------------------------------------

VPOL_MODEL = "RNOG_vpol_v3_5inch_center_n1.74"
LPDA_MODEL = "createLPDA_100MHz_v2_InfFirn_n1.4"

# the frequencies and directions below mix grid nodes with points in between them (the
# VPol model is sampled every 5 deg and ~2.13 MHz, the LPDA every 1 deg and 5 MHz), so both
# the tabulated values and the interpolation are covered
REFERENCE_FREQUENCIES = np.array([100., 250., 437.]) * units.MHz

# (orientation_theta, orientation_phi, rotation_theta, rotation_phi) in deg
REFERENCE_ORIENTATIONS = {
    "vpol_upright": (0, 0, 90, 0),
    "lpda_upward": (0, 0, 90, 0),  # boresight straight up
    "lpda_downward": (120, 0, 30, 0),  # boresight 30 deg below the horizon, pointing East
}

# VEL_theta and VEL_phi (in NuRadio units, i.e. m) at REFERENCE_FREQUENCIES, keyed by
# (model, orientation) and by the incoming direction (zenith, azimuth) in deg. These values
# are hard-coded on purpose: they pin down the interpolation and the coordinate
# transformations and only ever have to be regenerated if either is changed intentionally.
REFERENCE_VEL = {
    (VPOL_MODEL, "vpol_upright"): {
        (37.5, 23): ([-2.590587e-02-8.665878e-02j, +2.708036e-03+5.134874e-02j, -1.941625e-02-2.573250e-02j],
                     [+1.574257e-04-1.189295e-04j, -2.907987e-04+5.417262e-05j, +2.818347e-04-2.685509e-04j]),
        (60, 90): ([-8.842906e-02-8.594352e-02j, +1.003573e-01+1.761610e-02j, +3.602543e-02+2.405003e-02j],
                   [+1.221283e-05+2.995946e-06j, -2.656625e-06-1.709783e-06j, -8.886135e-06-5.124142e-06j]),
        (90, 210.5): ([-1.269029e-01+1.461460e-02j, -5.967338e-02-1.055009e-01j, -5.873987e-02+5.973344e-02j],
                      [+3.446606e-05+6.002938e-05j, -1.183580e-04+1.175912e-04j, +2.013560e-04+6.132837e-05j]),
        (123.5, 305): ([-7.918571e-02-8.934022e-02j, +8.702471e-02+3.345755e-02j, +1.331035e-02+3.210945e-02j],
                       [+3.802856e-05-1.812795e-05j, +4.049001e-06+8.622571e-05j, +2.272781e-05+5.639228e-05j]),
        (155, 0): ([-5.953972e-03-6.378338e-02j, -1.432429e-02+2.617367e-02j, +6.947442e-03-2.467225e-02j],
                   [0j, 0j, 0j]),  # phi = 0 deg is a symmetry plane of the VPol
    },
    (LPDA_MODEL, "lpda_upward"): {
        (37.5, 23): ([-1.068917e-01-8.854878e-02j, +6.456348e-02+1.988244e-02j, -2.568530e-02-3.191559e-02j],
                     [-3.351973e-01-2.703861e-01j, +1.753179e-01+6.223970e-02j, -7.659485e-02-7.037105e-02j]),
        (60, 90): ([-1.642099e-01-3.257877e-02j, -3.529879e-02-4.871512e-02j, +3.237249e-02+1.902306e-03j],
                   [+1.447680e-05+2.135807e-03j, +4.390991e-04-5.821575e-03j, -4.613082e-03+4.844393e-03j]),
        (90, 210.5): ([+3.979433e-03-2.175291e-03j, +4.135918e-03-4.818399e-03j, +2.784129e-03+2.111465e-03j],
                      [+1.905512e-01-1.162116e-01j, -1.094214e-01-3.755445e-02j, +5.116181e-02+5.014446e-02j]),
        (123.5, 305): ([+1.268950e-03+7.560181e-02j, +1.419240e-02+2.498945e-02j, +1.103123e-03+4.780411e-03j],
                       [+5.844797e-03+9.336023e-02j, +1.178195e-02+2.649801e-02j, +4.258309e-03+1.181137e-02j]),
        (155, 0): ([+3.437651e-03+2.315606e-03j, +4.778114e-03+3.814342e-03j, +3.236993e-03-6.114700e-03j],
                   [+7.007768e-02+9.615006e-02j, +8.457904e-03-2.404893e-02j, -1.340033e-02+1.534239e-02j]),
    },
    (LPDA_MODEL, "lpda_downward"): {
        (37.5, 23): ([+9.051290e-02-3.526327e-02j, -2.480212e-02-2.824809e-02j, -9.357417e-03+2.250649e-02j],
                     [+2.488400e-01-1.026628e-01j, -6.743154e-02-8.270836e-02j, -3.591645e-02+4.548868e-02j]),
        (60, 90): ([+4.836059e-02-9.430870e-02j, -2.321073e-02+2.159825e-02j, -1.994126e-02+4.408564e-03j],
                   [-2.455919e-03-1.801982e-04j, -2.495429e-03+3.725803e-03j, -3.037365e-03-3.058060e-04j]),
        (90, 210.5): ([-1.141920e-03+2.875596e-03j, +1.441551e-03+5.620259e-03j, -7.757950e-04+9.037358e-05j],
                      [+4.375625e-02+1.106481e-01j, +7.138241e-02+1.653943e-02j, -1.986827e-03-2.482480e-02j]),
        (123.5, 305): ([+1.561782e-01+8.630120e-02j, -5.759123e-02+3.860100e-02j, +4.264417e-02-2.978903e-02j],
                       [+2.027716e-01+1.103380e-01j, -7.621077e-02+4.369062e-02j, +4.538133e-02-3.746128e-02j]),
        (155, 0): ([-3.502938e-03-3.873231e-03j, -3.386163e-03+1.696049e-03j, -5.015626e-03+3.593128e-03j],
                   [+3.596386e-01+3.162327e-01j, -1.835327e-01-1.070991e-01j, +5.119614e-02+9.862683e-02j]),
    },
}


def test_reference_vel():
    """The interpolated VEL has to reproduce the hard-coded reference values"""
    provider = AntennaPatternProvider()

    for (model, orientation_name), reference in REFERENCE_VEL.items():
        antenna = provider.load_antenna_pattern(model)
        orientation = np.array(REFERENCE_ORIENTATIONS[orientation_name]) * units.deg

        for (zenith, azimuth), expected in reference.items():
            vel = antenna.get_antenna_response_vectorized(
                REFERENCE_FREQUENCIES, zenith * units.deg, azimuth * units.deg, *orientation)

            for component, reference_component in zip(["theta", "phi"], expected):
                # the atol is only relevant for the components which are ~ 0 by symmetry
                assert np.allclose(vel[component], reference_component,
                                   rtol=1e-5, atol=1e-6 * units.m), \
                    "{} ({}): VEL_{} at zenith = {} deg, azimuth = {} deg is\n{}\ninstead of\n{}".format(
                        model, orientation_name, component, zenith, azimuth,
                        np.array(vel[component]), np.array(reference_component))


def test_lpda_beam_follows_orientation():
    """Rotating the LPDA has to rotate its beam, not deform it"""
    antenna = AntennaPatternProvider().load_antenna_pattern(LPDA_MODEL)
    freqs = np.array([250.]) * units.MHz

    # scan the plane phi = 0 deg / 180 deg, which contains both boresights. Negative signed
    # zenith angles stand for azimuth = 180 deg.
    signed_zeniths = np.arange(-179., 180., 1.)

    peaks = []
    for orientation_name in ["lpda_upward", "lpda_downward"]:
        orientation = np.array(REFERENCE_ORIENTATIONS[orientation_name]) * units.deg
        vel = [antenna.get_antenna_response_vectorized(
            freqs, np.abs(zenith) * units.deg, np.pi * (zenith < 0), *orientation)
            for zenith in signed_zeniths]
        magnitude = np.array(
            [np.hypot(np.abs(v["theta"]), np.abs(v["phi"])) for v in vel]).flatten()

        # the beam of this model peaks 11 deg off the boresight
        i_peak = np.argmax(magnitude)
        boresight = REFERENCE_ORIENTATIONS[orientation_name][0]
        assert np.abs(signed_zeniths[i_peak] - boresight) < 15, \
            "{}: the beam peaks at a zenith angle of {:.0f} deg, the boresight is at {} deg".format(
                orientation_name, signed_zeniths[i_peak], boresight)
        peaks.append(magnitude[i_peak])

    assert np.isclose(*peaks, rtol=1e-6), \
        "the peak |VEL| changes with the orientation: {:.6f} vs {:.6f} m".format(*peaks)


if __name__ == "__main__":
    without_antenna_file = [
        test_interpolate_linear, test_interpolate_linear_vectorized, test_get_bracketing_indices,
        test_equidistant_shortcut, test_get_group_delay, test_antenna_rotation,
        test_analytic_patterns, test_parametric_phase, test_provider_buffering]

    with_antenna_file = [
        test_interpolation_at_nodes, test_magphase_agrees_at_nodes, test_phi_wrapping,
        test_out_of_range, test_vel_is_continuous, test_reference_vel,
        test_lpda_beam_follows_orientation]

    for test in without_antenna_file + with_antenna_file:
        print("{} ...".format(test.__name__))
        test()

    print("\nAntenna pattern tests passed")
