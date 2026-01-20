"""
This module (re)defines physics constants in NuRadio units.
"""

from scipy import constants as scipy_constants

from NuRadioReco.utilities import units

c = scipy_constants.c * units.m / units.s  #: Vacuum speed of light

epsilon_0 = scipy_constants.epsilon_0 * (units.coulomb / units.V / units.m)  #: Vacuum permittivity in NuRadioReco units

k_B = scipy_constants.k * units.joule / units.kelvin  #: Boltzman constant

e_mass = scipy_constants.physical_constants['electron mass energy equivalent in MeV'][0] * units.MeV
mu_mass = scipy_constants.physical_constants['muon mass energy equivalent in MeV'][0] * units.MeV
tau_mass = scipy_constants.physical_constants['tau mass energy equivalent in MeV'][0] * units.MeV
G_F = scipy_constants.physical_constants['Fermi coupling constant'][0] * units.GeV ** (-2)

pi_mass = 139.57061 * units.MeV
rho770_mass = 775.49 * units.MeV
rho1450_mass = 1465 * units.MeV
a1_mass = 1230 * units.MeV

tau_rest_lifetime = 290.3 * units.fs

density_ice = 0.9167 * units.g / units.cm ** 3
density_water = 997 * units.kg / units.m ** 3

