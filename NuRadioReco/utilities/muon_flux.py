import scipy.constants
import numpy as np
import json
import scipy.signal
from NuRadioReco.utilities import units, io_utilities
import astropy
import scipy.interpolate as interpolate
from scipy.integrate import quad
from scipy.integrate import dblquad

plot_flux = False

def get_flux(energy, file_surface_mu_flux='data/muon_flux_E_SIBYLL23c_GSF.pickle'):
    '''
    provides muon flux over the full zenith band as calculated with MCEq at a given energy. The CR model used
    is the GlobalSplineFit and the hadronic interaction model is Sibyll 2.3c.

    Parameters
    ----------
    energy: float
        energy at which flux is calculated (in eV)
    file_surface_mu_flux: string
        file with model for muon flux

    Returns
    -------
    flux of muon in NuRadio units. (GeV**-1 * m**-2 * ns**-1 sr**-1)
    '''
    data_surface_mu_flux = io_utilities.read_pickle(file_surface_mu_flux, encoding='latin1')
    J_e3 = np.array(data_surface_mu_flux['mu_total']) * units.GeV**2 * units.cm**-2 * units.s**-1 *units.sr**-1
    E_data = np.array(data_surface_mu_flux['e_grid']) * units.GeV

    def get_flux_norm(E):
        flux_E3 = interpolate.interp1d(E_data, J_e3, fill_value='extrapolate')
        flux = flux_E3(E)/E**3
        return flux

    return get_flux_norm(energy)


def get_flux_per_energy_bin(energy_bin_edge_low, energy_bin_edge_high):
    '''
    integrates the flux over the energy bin from energy_bin_edge_low to energy_bin_edge_high.
    The flux is defined in get_flux

    Parameters
    ----------
    energy_bin_edge_low: float
        lower edge of energy bin over which the flux is integrated (in eV)

    energy_bin_edge_high: float
        higher edge of energy bin over which the flux is integrated (in eV)

    Returns
    -------
    integrated flux of muon in NuRadio units.
    '''
    def flux(E):
        return get_flux(E)

    int_flux = quad(flux, energy_bin_edge_low, energy_bin_edge_high)

    return int_flux[0]

def get_flux_per_energy_and_zenith(energy, zenith, file_surface_mu_flux='data/muon_flux_E_theta_SIBYLL23c_GSF.pickle'):
    '''
    provides muon flux at a certain energy and zenith angle as calculated with MCEq at a given energy. The CR model used
    is the GlobalSplineFit and the hadronic interaction model is Sibyll 2.3c.

    Parameters
    ----------
    energy: float
        energy at which flux is calculated (in eV)
    zenith: float
        zenith angle at which flux is calculated (in rad)
    file_surface_mu_flux: string
        file with model for muon flux

    Returns
    -------
    flux of muon in NuRadio units
    '''
    data_surface_mu_flux = io_utilities.read_pickle(file_surface_mu_flux, encoding='latin1')
    J_e3 = np.array(data_surface_mu_flux['mu_total']) * units.GeV**2 * units.cm**-2 * units.s**-1
    E_data = np.array(data_surface_mu_flux['e_grid']) * units.GeV
    zen = np.array(data_surface_mu_flux['zen_grid']) * units.deg

    def get_flux_norm(E, theta):
        flux_E3 = interpolate.interp2d(E_data, zen, J_e3)
        flux = flux_E3(E, theta)/E**3
        return flux

    return get_flux_norm(energy, zenith)

def get_flux_per_energy_and_zenith_bin(energy_bin_edge_low, energy_bin_edge_high, zenith_bin_edge_low, zenith_bin_edge_high):
    '''
    integrates the flux over a certain energy and zenith bin.
    The flux is defined in get_flux_per_zenith_and_energy.

    Parameters
    ----------
    energy_bin_edge_low: float
        lower edge of energy bin over which the flux is integrated (in eV)
    energy_bin_edge_high: float
        higher edge of energy bin over which the flux is integrated (in eV)
    zenith_bin_edge_low: float
         lower edge of zenith bin over which flux is integrated (in rad)
     zenith_bin_edge_high: float
         higher edge of zenith bin over which flux is integrated (in rad)

    Returns
    -------
    integrated flux of muon in NuRadio units.
    '''
    def flux(E, theta):
        return get_flux_per_energy_and_zenith(E, theta)

    int_flux = dblquad(flux, np.cos(zenith_bin_edge_high), np.cos(zenith_bin_edge_low), lambda x: energy_bin_edge_low, lambda x: energy_bin_edge_high)
    return int_flux[0]


if plot_flux == True:
    units_flux = units.GeV**2.7 * units.cm**-2 * units.s**-1 * units.sr**-1

    import matplotlib.pyplot as plt
    file_surface_mu_flux='data/muon_flux_E_SIBYLL23c_GSF.pickle'
    data_surface_mu_flux = io_utilities.read_pickle(file_surface_mu_flux, encoding='latin1')
    J_e3 = np.array(data_surface_mu_flux['mu_total']) * units.GeV**2 * units.cm**-2 * units.s**-1 *units.sr**-1
    E_data = np.array(data_surface_mu_flux['e_grid']) * units.GeV

    plt.plot(E_data, J_e3/E_data**3)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'Flux data',fontsize=14)
    plt.xlabel('Energy [eV]',fontsize=14)
    #plt.show()
    #plt.close()

    plt.style.use({'figure.facecolor':'white'})
    plt.plot(1e18, get_flux_per_energy_and_zenith_bin(1e18, 1e19, 0*units.deg, 90*units.deg), marker='x', color='k', label='test 2d')
    plt.plot(1e18, get_flux_per_energy_bin(1e18, 1e19), marker='+', color='k', label='test 1d')

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'Flux',fontsize=14)
    plt.xlabel('Energy [eV]',fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.close()