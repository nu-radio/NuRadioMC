import numpy as np
from NuRadioReco.utilities import units
import scipy.interpolate as interpolate
from scipy.integrate import quad

def get_auger_cr_flux(energy):
    data = np.loadtxt('Auger_combined_spectrum_ICRC_2019.txt', skiprows=3)
    E = 10**(data[:, 0]) * units.eV
    E_J = data[:, 1] * units.m**-2 * units.second**-1 * units.steradian**-1
    J = E_J / E
    Err_up = data[:, 2] * units.m**-2 * units.second**-1 * units.steradian**-1 / E
    Err_low = data[:, 3] * units.m**-2 * units.second**-1 * units.steradian**-1 / E

    print(J)
    get_flux = interpolate.interp1d(E, J, fill_value=0, bounds_error=False)

    return get_flux(energy)


def get_auger_flux_per_energy_bin(bin_edge_low, bin_edge_high):
    data = np.loadtxt('Auger_combined_spectrum_ICRC_2019.txt', skiprows=3)
    E = 10**(data[:, 0]) * units.eV
    E_J = data[:, 1] * units.m**-2 * units.second**-1 * units.steradian**-1
    J = E_J / E

    flux = interpolate.interp1d(E, J, fill_value=0, bounds_error=False)
    int_flux = quad(flux, bin_edge_low, bin_edge_high, limit=2 * E.shape[0], points=E)

    return int_flux[0]

'''https://astro.pages.rwth-aachen.de/astrotools/_modules/auger.html#spectrum_analyticq'''