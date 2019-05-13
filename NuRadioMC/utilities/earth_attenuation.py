from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioMC.utilities import units
from NuRadioMC.utilities import cross_sections
import logging
logger = logging.getLogger("utilities.earth_attenuation")

R_earth = 6357390 * units.m
DensityCRUST = 2900 * units.kg / units.m ** 3
AMU = 1.66e-27 * units.kg
R_EARTH = 6.378140e6 * units.m
densities = np.array([14000.0, 3400.0, 2900.0]) * units.kg / units.m ** 3 # inner layer, middle layer, outer layer
radii = np.array([3.46e6 * units.m, R_EARTH - 4.0e4 * units.m, R_EARTH]) # average radii of boundaries between earth layers


def get_weight(theta_nu, pnu, flavors, mode='simple',cross_section_type = 'ghandi'):
    """
    calculates neutrino weight due to Earth absorption for different models

    Parameters
    ----------
    theta_nu: float or array of floats
        the zenith angle of the neutrino direction (where it came from, i.e., opposite to the direction of propagation)
    pnu: float or array of floats
        the momentum of the neutrino
    """
    if(mode == 'simple'):
        return get_simple_weight(theta_nu, pnu,cross_section_type = cross_section_type)
    elif (mode == "core_mantle_crust"):
        return get_core_mantle_crust_weight(theta_nu, pnu, flavors,cross_section_type = cross_section_type)
    elif (mode == "None"):
        return 1.
    else:
        logger.error('mode {} not supported'.format(mode))
        raise NotImplementedError


def get_simple_weight(theta_nu, pnu, cross_section_type = 'ghandi'):
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
    if(theta_nu <= 0.5 * np.pi):  # coming from above
        return np.ones_like(theta_nu)
    else:  # coming from below
        sigma = cross_sections.get_nu_cross_section(pnu, flavors = 0, cross_section_type = cross_section_type)
        d = - 2 * R_earth * np.cos(theta_nu)
        return np.exp(-d * sigma * DensityCRUST / AMU)

def get_core_mantle_crust_weight(theta_nu, pnu, flavors, cross_section_type='ctw'):
    """
    calculates neutrino weight due to Earth absorption with a three layers earth model, i.e. probability of the
    neutrino to reach the detector

    simple parametrization using momentum, zenith angle, flavor and current type information
    of the neutrino

    as implemented in ARAsim (2018)

    Parameters
    ----------
    theta_nu: float or array of floats
        the zenith angle of the neutrino direction (where it came from, i.e., opposite to the direction of propagation)
    pnu: float or array of floats
        the momentum of the neutrino
    flavors: float or array of floats
        the flavor of the neutrino
    """
    sigma = cross_sections.get_nu_cross_section(pnu, flavors, cross_section_type = cross_section_type)
    if(theta_nu <= 0.5 * np.pi):  # coming from above
        return np.ones_like(theta_nu)
    elif (theta_nu <= np.pi - np.arcsin(radii[1] / radii[2])): # only go through the outer layer
        d_outer = - 2 * R_EARTH * np.cos(theta_nu)
        weight = np.exp(-d_outer * sigma * densities[2] / AMU)
    elif (theta_nu <= np.pi - np.arcsin(radii[0] / radii[2])): # only go through the outer and middle layer
        d_middle = 2 * np.sqrt(radii[1] * radii[1] - radii[2] * radii[2] * np.sin(np.pi - theta_nu) * np.sin(np.pi - theta_nu))
        d_outer = - 2 * R_EARTH * np.cos(theta_nu) - d_middle
        weight = np.exp(-d_outer * sigma * densities[2] / AMU - d_middle * sigma * densities[1] / AMU)
    else: # go through all three layers
        d_inner = 2 * np.sqrt(radii[0] * radii[0] - radii[2] * radii[2] * np.sin(np.pi - theta_nu) * np.sin(np.pi - theta_nu))
        d_middle = 2 * np.sqrt(radii[1] * radii[1] - radii[2] * radii[2] * np.sin(np.pi - theta_nu) * np.sin(np.pi - theta_nu)) - d_inner
        d_outer = - 2 * R_EARTH * np.cos(theta_nu) - d_middle - d_inner
        weight = np.exp(-d_outer * sigma * densities[2] / AMU - d_middle * sigma * densities[1] / AMU - d_inner * sigma * densities[0] / AMU)
    return weight

