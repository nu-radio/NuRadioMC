import numpy as np
from NuRadioReco.utilities import units
import scipy.interpolate as interpolate
from scipy.integrate import quad

def get_cr_flux(log10_energy, type='auger_19'):
    """
    Returns an scipy interpolation of the measured data. Output
    units are 1/(eV m^2 sr ns) for cosmic-ray energy in (in log10(E / eV)))
    :param log10_energy: Input energies (in log10(E / eV)))
    :param mode:
    :return: scipy interpolation of data in 1/(eV m^2 sr ns) (NuRadio base units)
    """
    if type == 'auger_19'
        data = np.loadt_xt('data/Auger_combined_spectrum_ICRC_2019.txt', skiprows=3)
        # from PRL paper 2020 based on ICRC 2019
        E = 10**(data[:, 0]) * units.eV
        E_J = data[:, 1] * units.m**-2 * units.second**-1 * units.steradian**-1
        J = E_J / E
        Err_up = data[:, 2] * units.m**-2 * units.second**-1 * units.steradian**-1 / E
        Err_low = data[:, 3] * units.m**-2 * units.second**-1 * units.steradian**-1 / E

    get_flux = interpolate.interp1d(E, J, fill_value=0, bounds_error=False)

    return get_flux(10**log10_energy)


def get_flux_per_energy_bin(log10_bin_edge_low, log10_bin_edge_high, type='auger_19'):
    """
    Returns an scipy integration of the measured data over given interval. Output
    units are 1/(eV m^2 sr ns) for cosmic-ray energy in (in log10(E / eV)))
    :param log10_bin_edge_low: Input energy lower bound (in log10(E / eV)))
    :param log10_bin_edge_high: Input energy upper bound (in log10(E / eV)))
    :param mode:
    :return: scipy intergration of data in 1/(eV m^2 sr ns) (NuRadio base units)
    """
    if type == 'auger_19'
        data = np.loadtxt('data/Auger_combined_spectrum_ICRC_2019.txt', skiprows=3)
        # from PRL paper 2020 based on ICRC 2019
        E = 10**(data[:, 0]) * units.eV
        E_J = data[:, 1] * units.m**-2 * units.second**-1 * units.steradian**-1
        J = E_J / E

    flux = interpolate.interp1d(E, J, fill_value=0, bounds_error=False)
    int_flux = quad(flux, 10**log10_bin_edge_low, 10**log10_bin_edge_high, limit=2 * E.shape[0], points=E)

    return int_flux[0]

def get_analytic_cr_spectrum(log10_energy, type="auger_19"):
    """
    Returns a analytic parametrization of the Auger energy spectrum
    units are 1/(eV m^2 sr ns)  for cosmic-ray energy in log10(E / eV)
    :param log10_energy: Input energies (in log10(E / eV))
    :type: take auger_17, auger_19 or TA_19
    :return: analytic parametrization of spectrum for given input energies in 1/(eV m^2 sr ns) (NuRadio base units)
    """
    # from https://astro.pages.rwth-aachen.de/astrotools/_modules/auger.html#spectrum_analyticq

    DSPECTRUM_ANALYTIC_15 = np.array([3.3e-19, 4.82e18, 42.09e18, 3.29, 2.6, 3.14])
    DSPECTRUM_ANALYTIC_17 = np.array([2.8e-19, 5.08e18, 39e18, 3.293, 2.53, 2.5])
    DSPECTRUM_ANALYTIC_19 = np.array([3.46e12, 1.5e17, 6.2e18, 12e18, 50e18, 2.92, 3.27, 2.2, 3.2, 5.4])
    SPECTRA_DICT_ANA = {15: DSPECTRUM_ANALYTIC_15, 17: DSPECTRUM_ANALYTIC_17, 19: DSPECTRUM_ANALYTIC_19}

    energy = 10 ** log10_energy  # type: np.ndarray

    if type == "auger_17":
        p = auger.SPECTRA_DICT_ANA[17]  # type: np.ndarray
        return (np.where(energy < p[1],
                        p[0] * (energy / p[1]) ** (-p[3]),
                        p[0] * (energy / p[1]) ** (-p[4]) * (1 + (p[1] / p[2]) ** p[5])
                        * (1 + (energy / p[2]) ** p[5]) ** -1)) * units.year * units.km**2

    elif type == "auger_19":
        p = auger.SPECTRA_DICT_ANA[19]  # type: np.ndarray
        return ((energy / p[0]) ** (-p[5]) * \
               (1 + (energy / p[1]) ** p[5]) / (1 + (energy / p[1]) ** p[6]) * \
               (1 + (energy / p[2]) ** p[6]) / (1 + (energy / p[2]) ** p[7]) * \
               (1 + (energy / p[3]) ** p[7]) / (1 + (energy / p[3]) ** p[8]) * \
               (1 + (energy / p[4]) ** p[8]) / (1 + (energy / p[4]) ** p[9]) ) * units.year * units.km**2

    elif type == "TA_19":
        p1 = -3.28
        p2 = -2.68
        p3 = -4.84
        E1 = 10 ** 18.69
        E2 = 10 ** 19.81
        c = 2.24e-30
        c1 = c * (E1 / 1e18) ** p1
        c2 = c1 * (E2 / E1) ** p2
        yy = np.where(energy < E1,
                 c * (energy / 1e18) ** p1,
                 np.where(energy < E2,
                          c1 * (energy / E1) ** p2,
                          c2 * (energy / E2) ** p3))
        # convert 1/m**2 to 1/km**2 abd second to year
        return (yy * 1e6 * 3.154 * 10 ** 7) * units.year * units.km**2

