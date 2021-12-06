import numpy as np
from NuRadioReco.utilities import units
import scipy.interpolate as interpolate
from scipy.integrate import quad
import os.path


# References
# data_auger_19: https://www.auger.org/document-centre-public?task=download.send&id=5045&catid=78&m=0
# data_TA_21: data provided by van Vliet
# analytic spectrum TA_19: https://pos.sissa.it/358/298/pdf
# analytic spectrum auger: https://git.rwth-aachen.de/astro/astrotools/-/blob/master/astrotools

def get_measured_data(type='data_auger_19'):
    """
    Helper function get data points in NuRadio base units

    Parameters
    ----------
    type: string
        choose between data_auger_19 and data_TA_21
    Returns
    -------
    E: array of floats
        energie in eV
    J: array of floats
        flux of cosmic ray
    Err_up: array of floats
        upper error bar on flux
    Err_low: array of floats
        lower error bar on flux
    """

    if type == 'data_auger_19':
        abspath = os.path.dirname(os.path.abspath(__file__))
        data = np.loadtxt(os.path.join(abspath, 'data/Auger_combined_spectrum_ICRC_2019.txt'), skiprows=3)
        # from PRL paper 2020 based on ICRC 2019
        E = 10 ** (data[:, 0]) * units.eV
        E_J = data[:, 1] * units.m ** -2 * units.second ** -1 * units.steradian ** -1
        J = E_J / E
        J_err_low = data[:, 3] * units.m ** -2 * units.second ** -1 * units.steradian ** -1 / E
        J_err_up = data[:, 2] * units.m ** -2 * units.second ** -1 * units.steradian ** -1 / E

    elif type == 'data_TA_21':
        abspath = os.path.dirname(os.path.abspath(__file__))
        data = np.loadtxt(os.path.join(abspath, 'data/TA_combined_spectrum_ICRC_2021.txt'), skiprows=10)
        E = 10 ** (data[:, 0]) * units.eV
        J = data[:, 2] * units.eV ** -1 * units.m ** -2 * units.second ** -1 * units.steradian ** -1
        J_band_low = data[:, 3] * units.eV ** -1 * units.m ** -2 * units.second ** -1 * units.steradian ** -1
        J_band_up = data[:, 4] * units.eV ** -1 * units.m ** -2 * units.second ** -1 * units.steradian ** -1
        J_err_low = J - J_band_low
        J_err_up = J_band_up - J

    else:
        raise NotImplementedError(f"Type {type} is not supported")

    return E, J, J_err_low, J_err_up


def get_interpolated_cr_flux(log10_energy, type='data_auger_19'):
    """
    Evaluates the scipy interpolation of the measured data at a given energy

    Parameters
    ----------
    log10_energy: float
        energies (in log10(E / eV)))
    type: string
        choose between data_auger_19 and data_TA_21
    Return
    -------
        scipy interpolation of data in NuRadio base units (which are 1/(eV m^2 sr ns) )
    """
    E, J, J_err_low, J_err_up = get_measured_data(type)
    log10_e = np.log10(E)
    flux_interpolation = interpolate.interp1d(log10_e, J, fill_value=0, bounds_error=True)
    return flux_interpolation(log10_energy)


def get_analytic_cr_flux(log10_energy, type="auger_19"):
    """
    Parameters
    ----------
    log10_energy: float
        energies (in log10(E / eV)))

    type: string
        choose between auger_17, auger_19 and TA_19
    Returns
    -------
    analytic parametrization of spectrum in NuRadio base units (which are 1/(eV m^2 sr ns)
    """
    energy = 10 ** log10_energy

    if type == "auger_17":
        p = np.array([2.8e-19, 5.08e18, 39e18, 3.293, 2.53, 2.5])
        spectrum = np.where(energy < p[1],
                            p[0] * (energy / p[1]) ** (-p[3]),
                            p[0] * (energy / p[1]) ** (-p[4]) * (1 + (p[1] / p[2]) ** p[5])
                            * (1 + (energy / p[2]) ** p[5]) ** -1)
        spectrum = spectrum * (units.eV * units.km ** 2 * units.sr * units.year) ** -1

    elif type == "auger_19":
        p = np.array([3.46e12, 1.5e17, 6.2e18, 12e18, 50e18, 2.92, 3.27, 2.2, 3.2, 5.4])
        spectrum = (energy / p[0]) ** (-p[5]) * \
                   (1 + (energy / p[1]) ** p[5]) / (1 + (energy / p[1]) ** p[6]) * \
                   (1 + (energy / p[2]) ** p[6]) / (1 + (energy / p[2]) ** p[7]) * \
                   (1 + (energy / p[3]) ** p[7]) / (1 + (energy / p[3]) ** p[8]) * \
                   (1 + (energy / p[4]) ** p[8]) / (1 + (energy / p[4]) ** p[9])
        spectrum = spectrum * (units.eV * units.km ** 2 * units.sr * units.year) ** -1

    elif type == "TA_19":
        p1 = -3.28
        p2 = -2.68
        p3 = -4.84
        E1 = 10 ** 18.69
        E2 = 10 ** 19.81
        c = 2.24e-30
        c1 = c * (E1 / 1e18) ** p1
        c2 = c1 * (E2 / E1) ** p2
        spectrum = np.where(energy < E1,
                            c * (energy / 1e18) ** p1,
                            np.where(energy < E2,
                                     c1 * (energy / E1) ** p2,
                                     c2 * (energy / E2) ** p3))
        spectrum = spectrum * (units.eV * units.m ** 2 * units.sr * units.s) ** -1

    else:
        raise NotImplementedError(f"Type {type} is not supported")

    return spectrum


def get_flux_per_energy_bin(log10e_min, log10e_max, type='auger_19'):
    """
    Returns an scipy integration of the measured data or the analytic spectrum over given interval.

    Parameters
    ----------
    log10e_min: float
        Input energy lower bound (in log10(E / eV)))
    log10e_max: float
        Input energy upper bound (in log10(E / eV)))
    type: string
        choose between data_auger_19, data_TA_21, auger_17, auger_19, TA_19

    Returns
    -------
    scipy integration of data in NuRadio base units (which are 1/(eV m^2 sr ns))
    """
    if type in ['auger_17', 'auger_19', 'TA_19']:
        def flux(x):
            """ Bring parametrized energy spectrum in right shape for quad() function """
            return get_analytic_cr_flux(np.log10(np.array([x])), type)[0]

        integrated_flux = quad(flux, 10 ** log10e_min, 10 ** log10e_max)
    else:
        raise NotImplementedError(f"Type {type} is not supported")

    return integrated_flux[0]


def get_cr_event_rate(log10energy=18, zenith=50*units.deg, a_eff=1, type="auger_19"):
    """
    Cosmic ray event rate at a specific energy and zenith angle assuming a detector with effective area
    'A_eff'. The detector projection and solid range are taken into account.
    The flux is calculated with the analytic spectrum.

    Parameters
    ----------
    log10energy: float
        energy in units log10(energy/eV)
    zenith: float
        zenith angle
    a_eff:
        effective area of detector
    type: string
        choose between auger_17, auger_19 and TA_19
    Returns
    -------
    eventrate for an isotropic flux with given energy and zenith angle.
    The differential flux is returned in d/d zenith_angle, not d/d solid_angle.
    """
    # The projected area, is the area visible to CR.
    # None of the horizontal CR reach the flat detector, all CR from above reach the detector
    projected_area = np.cos(zenith)

    solid_angle = 2 * np.pi * np.sin(zenith)

    flux = get_analytic_cr_flux(log10energy, type=type)

    return flux * projected_area * solid_angle * a_eff


def plot_measured_spectrum(ax=None, scale=2.7, type='data_auger_19',
                           base_units=False):
    """
    Plot measured spectrum. Attention: time unit is year instead of ns.

    Parameters
    ----------
    ax:
        axis on which the data is plotted
    scale:
        scale factor for energy, default = 2.7
    type: string
        choose between data_auger_19 and data_TA_21
    base_units: bool
       if False spectrum will be plotted in
        [m^{-2} yr^{-1} sr^{-1} eV^{%scale-1}] instead of NuRadio base units

    Returns
    -------
    plot of data without plt.show()
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    E, J, J_err_low, J_err_up = get_measured_data(type)

    if not base_units:
        E_J_scale_units = units.m ** -2 * units.year ** -1 * units.steradian ** -1 * units.eV ** (scale - 1)
        yl = r'$J(E)$ [m$^{-2}$ yr$^{-1}$ sr$^{-1}$ eV$^{%g}$]' % (scale - 1)

    else:
        E_J_scale_units = units.m ** -2 * units.ns ** -1 * units.steradian ** -1 * units.eV ** (scale - 1)
        yl = r'$J(E)$ [m$^{-2}$ ns$^{-1}$ sr$^{-1}$ eV$^{%g}$]' % (scale - 1)

    ax.errorbar(E, E ** scale * J / E_J_scale_units,
                yerr=[(E ** scale * J_err_low) / E_J_scale_units, (E ** scale * J_err_up) / E_J_scale_units],
                marker='x', linewidth=1, markersize=8, ls='None', label=type)

    if scale != 0:
        yl = r'$E^{%g}\,$' % scale + yl
    ax.set_ylabel(yl)
    ax.set_xlabel(r'$E$ [eV]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend()
