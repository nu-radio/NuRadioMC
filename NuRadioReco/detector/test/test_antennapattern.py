#!/usr/bin/env python3
"""
Tests for the antenna pattern interpolation.

Regression test for the bracketing of non-equidistant grids: the antenna models are
not required to be sampled on an equally spaced grid (e.g. RNOG_vpol_4inch_center_n1.73
has theta nodes [0, 30, 60, 70, 80, 90, 100, 110, 120, 150, 180] deg). Deriving the
node indices from the grid boundaries instead of looking them up produced
discontinuities of up to 70 % in the vector effective length.
"""
import numpy as np

from NuRadioReco.detector.antennapattern import AntennaPatternProvider, get_bracketing_indices
from NuRadioReco.utilities import units


def test_get_bracketing_indices():
    nodes = np.array([0., 30., 60., 70., 80., 90., 100., 110., 120., 150., 180.])

    # values in between two nodes
    for x, expected in [(15., (0, 1)), (65., (2, 3)), (135., (8, 9)), (179.9, (9, 10))]:
        assert tuple(np.atleast_1d(i)[0] for i in get_bracketing_indices(x, nodes)) == expected, \
            "wrong bracket for x = {}".format(x)

    # values on a node have to be bracketed such that the interpolation returns the node
    for i, x in enumerate(nodes):
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


def test_vel_is_continuous():
    """The interpolated VEL must not jump between two adjacent zenith angles"""
    provider = AntennaPatternProvider()
    antenna = provider.load_antenna_pattern("RNOG_vpol_4inch_center_n1.73")

    zeniths = np.arange(0, 180, 0.1) * units.deg
    freq = np.array([200]) * units.MHz
    orientation = [0, 0, np.pi / 2, np.pi / 2]

    for azimuth in np.arange(0, 360, 20) * units.deg:
        vel = np.array([np.abs(antenna.get_antenna_response_vectorized(
            freq, zenith, azimuth, *orientation)["theta"]) for zenith in zeniths]).flatten()

        steps = np.abs(np.diff(vel))
        # a smooth curve on this fine a grid stays close to its typical step size
        assert steps.max() < 10 * np.median(steps), \
            "VEL jumps by {:.2e} at zenith = {:.1f} deg (azimuth = {:.0f} deg), " \
            "typical step is {:.2e}".format(
                steps.max(), zeniths[np.argmax(steps)] / units.deg,
                azimuth / units.deg, np.median(steps))


if __name__ == "__main__":
    test_get_bracketing_indices()
    test_vel_is_continuous()
    print("Antenna pattern tests passed")
