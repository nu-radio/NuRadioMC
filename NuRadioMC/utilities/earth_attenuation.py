from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units
from NuRadioMC.utilities import cross_sections
from radiotools import helper as hp
import logging
logger = logging.getLogger("utilities.earth_attenuation")

AMU = 1.66e-27 * units.kg


def get_weight(theta_nu, pnu, flavors, mode='simple', cross_section_type='ctw',
               vertex_position=None, phi_nu=None):
    """
    calculates neutrino weight due to Earth absorption for different models

    Parameters
    ----------
    theta_nu: float or array of floats
        the zenith angle of the neutrino direction (where it came from, i.e., opposite to the direction of propagation)
    pnu: float or array of floats
        the momentum of the neutrino
    mode: string
        * 'simple': assuming interaction happens at the surface and approximating the Earth with constant density
        * 'core_mantle_crust_simple': assuming interaction happens at the surface and approximating the Earth with 3 layers of constant density
        * 'core_mantle_crust': approximating the Earth with 3 layers of constant density, path through Earth to interaction vertex is considered
        * 'PREM': density of Earth is parameterized as a fuction of radius, path through Earth to interaction vertex is considered
    cross_section_type: string
        'ghandi', 'ctw' or 'csms' (see description in `cross_sections.py`)
    vertex_position: 3-dim array or None (default)
        the position of the neutrino interaction
    phi_nu: float
        the azimuth angle of the neutrino direction
    """
    if(mode == 'simple'):
        return get_simple_weight(theta_nu, pnu, cross_section_type=cross_section_type)
    elif (mode == "core_mantle_crust_simple"):
        return get_core_mantle_crust_weight(theta_nu, pnu, flavors, cross_section_type=cross_section_type)
    elif (mode == "core_mantle_crust"):
        earth = CoreMantleCrustModel()
        direction = hp.spherical_to_cartesian(theta_nu, phi_nu)
        slant_depth = earth.slant_depth(vertex_position, direction)
        # by requesting the interaction length for a density of 1, we get it in units of length**2/weight
        L_int = cross_sections.get_interaction_length(pnu, density=1., flavor=flavors, inttype='total',
                                                      cross_section_type=cross_section_type)
        return np.exp(-slant_depth / L_int)
    elif (mode == "PREM"):
        earth = PREM()
        direction = hp.spherical_to_cartesian(theta_nu, phi_nu)
        slant_depth = earth.slant_depth(vertex_position, direction)
        # by requesting the interaction length for a density of 1, we get it in units of length**2/weight
        L_int = cross_sections.get_interaction_length(pnu, density=1., flavor=flavors, inttype='total',
                                                      cross_section_type=cross_section_type)
        return np.exp(-slant_depth / L_int)
    elif (mode == "None"):
        return 1.
    else:
        logger.error('mode {} not supported'.format(mode))
        raise NotImplementedError


def get_simple_weight(theta_nu, pnu, cross_section_type='ctw'):
    """
    calculates neutrino weight due to Earth absorption, i.e. probability of the
    neutrino to reach the detector

    simple parametrization using only momentum and zenith angle information
    of the neutrino, adapted from ShelfMC

    Parameters
    ----------
    theta_nu: float or array of floats
        the zenith angle of the neutrino direction (where it came from, i.e., opposite to the direction of propagation)
    pnu: float or array of floats
        the momentum of the neutrino
    """
    R_earth = 6357390 * units.m
    DensityCRUST = 2900 * units.kg / units.m ** 3
    if(theta_nu <= 0.5 * np.pi):  # coming from above
        return np.ones_like(theta_nu)
    else:  # coming from below
        sigma = cross_sections.get_nu_cross_section(pnu, flavors=0, cross_section_type=cross_section_type)
        d = -2 * R_earth * np.cos(theta_nu)
        return np.exp(-d * sigma * DensityCRUST / AMU)


def get_core_mantle_crust_weight(theta_nu, pnu, flavors, cross_section_type='ctw'):
    """
    calculates neutrino weight due to Earth absorption with a three layers earth model, i.e. probability of the
    neutrino to reach the detector

    simple parametrization using momentum, zenith angle, flavor and current type information
    of the neutrino

    parameters from ARAsim (2018)

    Parameters
    ----------
    theta_nu: float or array of floats
        the zenith angle of the neutrino direction (where it came from, i.e., opposite to the direction of propagation)
    pnu: float or array of floats
        the momentum of the neutrino
    flavors: float or array of floats
        the flavor of the neutrino
    """
    R_EARTH = 6.378140e6 * units.m
    densities = np.array([14000.0, 3400.0, 2900.0]) * units.kg / units.m ** 3  # inner layer, middle layer, outer layer
    radii = np.array([3.46e6 * units.m, R_EARTH - 4.0e4 * units.m, R_EARTH])  # average radii of boundaries between earth layers
    sigma = cross_sections.get_nu_cross_section(pnu, flavors, cross_section_type=cross_section_type)
    if(theta_nu <= 0.5 * np.pi):  # coming from above
        return np.ones_like(theta_nu)
    elif (theta_nu <= np.pi - np.arcsin(radii[1] / radii[2])):  # only go through the outer layer
        d_outer = -2 * R_EARTH * np.cos(theta_nu)
        weight = np.exp(-d_outer * sigma * densities[2] / AMU)
    elif (theta_nu <= np.pi - np.arcsin(radii[0] / radii[2])):  # only go through the outer and middle layer
        d_middle = 2 * np.sqrt(radii[1] * radii[1] - radii[2] * radii[2] * np.sin(np.pi - theta_nu) * np.sin(np.pi - theta_nu))
        d_outer = -2 * R_EARTH * np.cos(theta_nu) - d_middle
        weight = np.exp(-d_outer * sigma * densities[2] / AMU - d_middle * sigma * densities[1] / AMU)
    else:  # go through all three layers
        d_inner = 2 * np.sqrt(radii[0] * radii[0] - radii[2] * radii[2] * np.sin(np.pi - theta_nu) * np.sin(np.pi - theta_nu))
        d_middle = 2 * np.sqrt(radii[1] * radii[1] - radii[2] * radii[2] * np.sin(np.pi - theta_nu) * np.sin(np.pi - theta_nu)) - d_inner
        d_outer = -2 * R_EARTH * np.cos(theta_nu) - d_middle - d_inner
        weight = np.exp(-d_outer * sigma * densities[2] / AMU - d_middle * sigma * densities[1] / AMU - d_inner * sigma * densities[0] / AMU)
    return weight


# PREM class from pyrex: https://github.com/bhokansonfasig/pyrex/blob/d84a3270efa19fb4a21590510f7c3458845c9600/pyrex/earth_model.py
class PREM:
    """
    Class describing the Earth's density.

    Uses densities from the Preliminary reference Earth Model (PREM).

    Attributes
    ----------
    earth_radius : float
        Mean radius of the Earth (m).
    radii : tuple
        Boundary radii at which the functional form of the density of the
        Earth changes. The density function in `densities` at index `i`
        corresponds to the radius range from radius at index `i-1` to radius
        at index `i`.
    densities : tuple
        Functions which calculate the density of the Earth in a
        specific radius range as described by `radii`. The parameter of each
        function is the fractional radius, e.g. radius divided by
        `earth_radius`. Scalar values denote constant density over the range of
        radii.

    Notes
    -----
    The density calculation is based on the Preliminary reference Earth Model
    [1]_.

    References
    ----------
    .. [1] A. Dziewonski & D. Anderson, "Preliminary reference Earth model."
        Physics of the Earth and Planetary Interiors **25**, 297–356 (1981). :doi:`10.1016/0031-9201(81)90046-7`

    """
    earth_radius = 6.3710e6 * units.m

    radii = (1.2215e6 * units.m, 3.4800e6 * units.m, 5.7010e6 * units.m, 5.7710e6 * units.m, 5.9710e6 * units.m,
             6.1510e6 * units.m, 6.3466e6 * units.m, 6.3560e6 * units.m, 6.3680e6 * units.m, earth_radius)

    # `x` is fraction of earth radius
    densities = (
        lambda x: (13.0885 * units.g / units.cm ** 3 - 8.8381 * units.g / units.cm ** 3 * x ** 2),
        lambda x: 12.5815 * units.g / units.cm ** 3 - 1.2638 * units.g / units.cm ** 3 * x - 3.6426 * units.g / units.cm ** 3 * x ** 2 - 5.5281 * units.g / units.cm ** 3 * x ** 3,
        lambda x: 7.9565 * units.g / units.cm ** 3 - 6.4761 * units.g / units.cm ** 3 * x + 5.5283 * units.g / units.cm ** 3 * x ** 2 - 3.0807 * units.g / units.cm ** 3 * x ** 3,
        lambda x: 5.3197 * units.g / units.cm ** 3 - 1.4836 * units.g / units.cm ** 3 * x,
        lambda x: 11.2494 * units.g / units.cm ** 3 - 8.0298 * units.g / units.cm ** 3 * x,
        lambda x: 7.1089 * units.g / units.cm ** 3 - 3.8045 * units.g / units.cm ** 3 * x,
        lambda x: 2.691 * units.g / units.cm ** 3 + 0.6924 * units.g / units.cm ** 3 * x,
        2.9 * units.g / units.cm ** 3,
        2.6 * units.g / units.cm ** 3,
        1.02 * units.g / units.cm ** 3
    )

    def density(self, r):
        """
        Calculates the Earth's density at a given radius.

        Supports passing an array of radii or a single radius.

        Parameters
        ----------
        r : array_like
            Radius (m) at which to calculate density.

        Returns
        -------
        array_like
            Density (g/cm^3) of the Earth at the given radii.

        """
        r = np.array(r)
        radius_bounds = np.concatenate(([0], self.radii))
        conditions = list((lower <= r) & (r < upper) for lower, upper in
                          zip(radius_bounds[:-1], radius_bounds[1:]))
        return np.piecewise(r / self.earth_radius, conditions, self.densities)

    def slant_depth(self, endpoint, direction, step=500 * units.m):
        """
        Calculates the column density of a chord cutting through Earth.

        Integrates the Earth's density along the chord, resulting in a column
        density (or material thickness) with units of mass per area.

        Parameters
        ----------
        endpoint : array_like
            Vector position of the chord endpoint, in a coordinate system
            centered on the surface of the Earth (e.g. a negative third
            coordinate represents the depth below the surface).
        direction : array_like
            Vector direction of the chord, in a coordinate system
            centered on the surface of the Earth (e.g. a negative third
            coordinate represents the chord pointing into the Earth).
        step : float, optional
            Step size for the integration.

        Returns
        -------
        float
            Column density along the chord starting from `depth` and
            passing through the Earth at `angle`.

        See Also
        --------
        PREM.density : Calculates the Earth's density at a given radius.

        """
        # Convert to Earth-centric coordiante system (e.g. center of the Earth
        # is at (0, 0, 0))
        endpoint = np.array([endpoint[0], endpoint[1],
                             endpoint[2] + self.earth_radius])
        direction /= np.linalg.norm(direction)
        dot_prod = np.dot(endpoint, direction)
        # Check for intersection of line and sphere
        discriminant = dot_prod ** 2 - np.sum(endpoint ** 2) + self.earth_radius ** 2
        if discriminant <= 0:
            return 0
        # Calculate the distance at which the line intersects the sphere
        distance = -dot_prod + np.sqrt(discriminant)
        if distance <= 0:
            return 0
        # Parameterize line integral with ts from 0 to 1, with steps just under
        # the given step size (in meters)
        n_steps = int(distance / step)
        if distance % step:
            n_steps += 1
        ts = np.linspace(0, 1, n_steps)
        xs = endpoint[0] + ts * distance * direction[0]
        ys = endpoint[1] + ts * distance * direction[1]
        zs = endpoint[2] + ts * distance * direction[2]
        rs = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
        rhos = self.density(rs)
        return np.trapz(rhos * distance, ts)


class CoreMantleCrustModel(PREM):
    """
    Class describing the Earth's density.

    Uses densities from the Core-Mantle-Crust model. Parameters from ARASim

    Attributes
    ----------
    earth_radius : float
        Mean radius of the Earth (m).
    radii : tuple
        Boundary radii (m) at which the functional form of the density of the
        Earth changes. The density function in `densities` at index `i`
        corresponds to the radius range from radius at index `i-1` to radius
        at index `i`.
    densities : tuple
        Functions which calculate the density of the Earth (g/cm^3) in a
        specific radius range as described by `radii`. The parameter of each
        function is the fractional radius, e.g. radius divided by
        `earth_radius`. Scalar values denote constant density over the range of
        radii.

    """
    earth_radius = 6.378140e6 * units.m

    radii = (np.sqrt(1.2e13) * units.m, earth_radius - 4e4 * units.m, earth_radius)

    densities = (14 * units.g / units.cm ** 3, 3.4 * units.g / units.cm ** 3, 2.9 * units.g / units.cm ** 3)

