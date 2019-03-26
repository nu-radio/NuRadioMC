from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioMC.utilities import units
import logging
logger = logging.getLogger("utilities.earth_attenuation")

R_earth = 6357390 * units.m
DensityCRUST = 2900 * units.kg / units.m ** 3
AMU = 1.66e-27 * units.kg
R_EARTH = 6.378140e6 * units.m
densities = np.array([14000.0, 3400.0, 2900.0]) * units.kg / units.m ** 3 # inner layer, middle layer, outer layer
radii = np.array([3.46e6 * units.m, R_EARTH - 4.0e4 * units.m, R_EARTH]) # average radii of boundaries between earth layers

c0 = np.zeros((2, 2))
c1 = np.zeros((2, 2))
c2 = np.zeros((2, 2))
c3 = np.zeros((2, 2))
c4 = np.zeros((2, 2))
# [nu][neutral current]
c0[0][0] = -1.826;
c1[0][0] = -17.31;
c2[0][0] = -6.448;
c3[0][0] = 1.431;
c4[0][0] = -18.61;
# [nu][charged current]
c0[0][1] = -1.826;
c1[0][1] = -17.31;
c2[0][1] = -6.406;
c3[0][1] = 1.431;
c4[0][1] = -17.91;
# [nubar][neutral current]
c0[1][0] = -1.033;
c1[1][0] = -15.95;
c2[1][0] = -7.296;
c3[1][0] = 1.569;
c4[1][0] = -18.30;
# [nubar][charged current]
c0[1][1] = -1.033;
c1[1][1] = -15.95;
c2[1][1] = -7.247;
c3[1][1] = 1.569;
c4[1][1] = -17.72;

def m_fsigma(flavors, ccncs, x):
    if (flavors > 0):
        i = 0
    else:
        i = 1
    if (ccncs == 'nc'):
        j = 0
    elif (ccncs == 'cc'):
        j = 1
    else:
        logger.error('ccncs = {} not defined'.format(ccncs))
        raise NotImplementedError
    return np.power(10, c1[i][j] + c2[i][j] * np.log(x - c0[i][j]) + c3[i][j] * np.power(np.log(x - c0[i][j]), 2) + c4[i][j] / np.log(x - c0[i][j]))

def get_weight(theta_nu, pnu, flavors, mode='simple'):
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
        return get_simple_weight(theta_nu, pnu)
    elif (mode == "core_mantle_crust"):
        return get_arasim_simple_weight(theta_nu, pnu, flavors)
    elif (mode == "None"):
        return 1.
    else:
        logger.error('mode {} not supported'.format(mode))
        raise NotImplementedError


def get_simple_weight(theta_nu, pnu):
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
        sigma = (7.84e-40) * units.m2 * (pnu / units.GeV) ** 0.363
        d = - 2 * R_earth * np.cos(theta_nu)
        return np.exp(-d * sigma * DensityCRUST / AMU)

def get_arasim_simple_weight(theta_nu, pnu, flavors):
    """
    calculates neutrino weight due to Earth absorption with a three layers earth model, i.e. probability of the
    neutrino to reach the detector

    simple parametrization using momentum, zenith angle, flavor and current type information
    of the neutrino

    Parameters
    ----------
    theta_nu: float or array of floats
        the zenith angle of the neutrino direction (where it came from, i.e., opposite to the direction of propagation)
    pnu: float or array of floats
        the momentum of the neutrino
    flavors: float or array of floats
        the flavor of the neutrino
    """
    sigma = get_sigma(pnu, flavors)
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

def get_sigma(pnu, flavors):
    """
    calculates the cross section based on Connolly et al. 2011

    parametrization using momentum, flavor and current type information
    of the neutrino

    Parameters
    ----------
    pnu: float or array of floats
        the momentum of the neutrino
    flavors: float or array of floats
        the flavor of the neutrino
    """
    sigma_total = m_fsigma(flavors, 'nc', np.log10(pnu / units.GeV)) / 1e4 * units.m2 + m_fsigma(flavors, 'cc', np.log10(pnu / units.GeV)) / 1e4 * units.m2
    return sigma_total
