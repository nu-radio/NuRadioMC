import numpy as np
from NuRadioReco.utilities import units
import scipy.interpolate as interpolate
from scipy.integrate import quad
import os.path

# References
# Auger ICRC 2019, https://www.auger.org/document-centre-public?task=download.send&id=5045&catid=78&m=0
# TA ICRC 2021, data provided by van Vliet

def get_measured_data(type):
    """
    Helper function get data points in NuRadio base units
    :param type: choose between auger_ICRC2019 and TA_ICRC2021
    :return: E, J, Err_up, Err_low in NuRadio base units
    """
    if type == 'auger_ICRC2019':
        abspath = os.path.dirname(os.path.abspath(__file__))
        data = np.loadtxt(os.path.join(abspath, 'data/Auger_combined_spectrum_ICRC_2019.txt'), skiprows=3)
        # from PRL paper 2020 based on ICRC 2019
        E = 10 ** (data[:, 0]) * units.eV
        E_J = data[:, 1] * units.m ** -2 * units.second ** -1 * units.steradian ** -1
        J = E_J / E
        J_err_low = data[:, 3] * units.m ** -2 * units.second ** -1 * units.steradian ** -1 / E
        J_err_up = data[:, 2] * units.m ** -2 * units.second ** -1 * units.steradian ** -1 / E

    if type == 'TA_ICRC2021':
        abspath = os.path.dirname(os.path.abspath(__file__))
        data = np.loadtxt(os.path.join(abspath, 'data/TA_combined_spectrum_ICRC_2021.txt'), skiprows=10)
        E = 10 ** (data[:, 0]) * units.eV
        J = data[:, 2] * units.eV ** -1 * units.m ** -2 * units.second ** -1 * units.steradian ** -1
        J_band_low = data[:, 3] * units.eV ** -1 * units.m ** -2 * units.second ** -1 * units.steradian ** -1
        J_band_up = data[:, 4] * units.eV ** -1 * units.m ** -2 * units.second ** -1 * units.steradian ** -1
        J_err_low = J - J_band_low
        J_err_up = J_band_up - J

    return E, J, J_err_low, J_err_up

def get_flux_interpolation(type):
    """
    Helper function to interpolate the measured data
    :param type: choose between auger_ICRC2019 and TA_ICRC2021
    :return: scipy interpolation of data in NuRadio base units (which are 1/(eV m^2 sr ns) )
    """
    E, J, J_err_low, J_err_up = get_measured_data(type)
    return interpolate.interp1d(E, J, fill_value=0, bounds_error=False)


def get_cr_flux(log10_energy, type='auger_ICRC2019'):
    """
    Returns an scipy interpolation of the measured data.
    :param log10_energy: Input energies (in log10(E / eV)))
    :param type: choose between auger_ICRC2019 and TA_ICRC2021
    :return: scipy interpolation of data in NuRadio base units (which are 1/(eV m^2 sr ns) )
    """
    flux_interpolation = get_flux_interpolation(type=type)

    return flux_interpolation(10**log10_energy)


def get_flux_per_energy_bin(log10_bin_edge_low, log10_bin_edge_high, type='auger_ICRC2019'):
    """
    Returns an scipy integration of the measured data over given interval.
    :param log10_bin_edge_low: Input energy lower bound (in log10(E / eV)))
    :param log10_bin_edge_high: Input energy upper bound (in log10(E / eV)))
    :param type: choose between auger_ICRC2019 and TA_ICRC2021
    :return: scipy integration of data in NuRadio base units (which are 1/(eV m^2 sr ns))
    """
    E, J, J_err_low, J_err_up = get_measured_data(type)

    flux_interpolation = get_flux_interpolation(type)

    integrated_flux = quad(flux_interpolation, 10**log10_bin_edge_low, 10**log10_bin_edge_high, limit=2 * E.shape[0], points=E)

    return integrated_flux[0]

def plot_measured_spectrum(ax=None, scale=2.7, type='auger_ICRC2019'):
    """
    Plot measured spectrum.
    :param ax: axis on which the data is plotted
    :param scale: scale factor for energy, default = 2.7
    :param type: choose between auger_ICRC2019 and TA_ICRC2021
    :return: plot of data without plt.show()
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    E, J, J_err_low, J_err_up = get_measured_data(type)

    E_J_scale_units = units.m**-2 * units.year**-1 * units.steradian**-1 * units.eV**(scale - 1)

    ax.errorbar(E, E**scale * J /E_J_scale_units,
                yerr=[(E**scale * J_err_low) / E_J_scale_units, (E**scale * J_err_up)/E_J_scale_units],
                marker='x', linewidth=1, markersize=8, ls='None', label='Measured by {}'.format(type))


    yl = r'$J(E)$ [m$^{-2}$ yr$^{-1}$ sr$^{-1}$ eV$^{%g}$]' % (scale - 1)
    if scale != 0:
        yl = r'$E^{%g}\,$' % scale + yl
    ax.set_ylabel(yl)
    ax.set_xlabel(r'$E$ [eV]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend()