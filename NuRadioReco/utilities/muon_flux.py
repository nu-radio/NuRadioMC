import numpy as np
from NuRadioReco.utilities import units, io_utilities
import scipy.interpolate as interpolate
from scipy.integrate import quad
from scipy.integrate import dblquad
import os

abspath = os.path.dirname(os.path.abspath(__file__))

def get_flux(energy, file_surface_mu_flux=os.path.join(abspath, 'data/muon_flux_E_SIBYLL23c_GSF.pickle')):
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

def get_flux_per_energy_and_zenith(energy, zenith, file_surface_mu_flux=os.path.join(abspath, 'data/muon_flux_E_theta_SIBYLL23c_GSF.pickle')):
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
    def get_flux_norm(E, cos_theta):
        flux_E3 = interpolate.interp2d(E_data, np.cos(zen), J_e3)
        flux = flux_E3(E, cos_theta)/E**3
        return flux

    return get_flux_norm(energy, np.cos(zenith))

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

    def flux(E, cos_theta):
        return get_flux_per_energy_and_zenith(E, cos_theta)

    int_flux = dblquad(flux, np.cos(zenith_bin_edge_high), np.cos(zenith_bin_edge_low), lambda x: energy_bin_edge_low, lambda x: energy_bin_edge_high)
    return int_flux[0]

if __name__ == "__main__":
    units_flux = units.GeV**2.7 * units.cm**-2 * units.s**-1 * units.sr**-1

    import matplotlib.pyplot as plt
    plt.style.use({'figure.facecolor':'white'})

    data_surface_mu_flux_1D = io_utilities.read_pickle(os.path.join(abspath,'data/muon_flux_E_SIBYLL23c_GSF.pickle') , encoding='latin1')
    J_e3_1D = np.array(data_surface_mu_flux_1D['mu_total']) * units.GeV**2 * units.cm**-2 * units.s**-1 *units.sr**-1
    E_data_1D = np.array(data_surface_mu_flux_1D['e_grid']) * units.GeV

    data_surface_mu_flux_2D = io_utilities.read_pickle(os.path.join(abspath,'data/muon_flux_E_theta_SIBYLL23c_GSF.pickle') , encoding='latin1')
    J_e3_2D = np.array(data_surface_mu_flux_2D['mu_total']) * units.GeV**2 * units.cm**-2 * units.s**-1 *units.sr**-1
    E_data_2D = np.array(data_surface_mu_flux_2D['e_grid']) * units.GeV
    zen_grid_2D = np.array(data_surface_mu_flux_2D['zen_grid']) * units.deg

    energy_bins = np.logspace(14, 19.5, 24)
    energy_bins_low = energy_bins[0:-1]
    energy_bins_high = energy_bins[1:]
    energy_bins_center = 0.5*(energy_bins_low+energy_bins_high)

    flux_1d = []
    flux_2d = []
    for i in np.arange(len(energy_bins_low)):
       flux_1d.append(get_flux_per_energy_bin(energy_bins_low[i], energy_bins_high[i]))
       flux_2d.append(get_flux_per_energy_and_zenith_bin(energy_bins_low[i], energy_bins_high[i], 50*units.deg, 70*units.deg))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.set_title('Scaled muon flux from MCEq')
    ax1.plot(E_data_1D, (J_e3_1D/E_data_1D**3) * E_data_1D**3.7 / units_flux, color='k', label='total')
    ax2.set_title('Unscaled muon flux from MCEq')
    ax2.plot(E_data_1D, J_e3_1D/E_data_1D**3, color='k', label='total')

    for i_zen, zen in enumerate(zen_grid_2D):
        ax1.plot(E_data_2D, (J_e3_2D[i_zen]/E_data_2D**3) * E_data_1D**3.7 / units_flux, label=f'{zen/units.deg:.2f}$\degree$')
        ax2.plot(E_data_2D, J_e3_2D[i_zen]/E_data_2D**3, label=f'{zen/units.deg:.2f}$\degree$')

    ax2.plot(1e16, get_flux(1e16), marker='x', label='test 1D')
    ax2.plot(1e17, get_flux_per_energy_and_zenith(1e17, 60*units.deg), marker='x', label='test 2D')
    ax3.step(energy_bins_center, flux_1d, label=r'Integrated flux 0$\degree$ - 90$\degree$')
    ax4.step(energy_bins_center, flux_2d, label=r'Integrated flux 50$\degree$ - 70$\degree$')
    ax1.set_ylabel(r'$E^{3.7}$ J [$GeV^{2.7}$ $cm^{-2}$ $s^{-1}$ $sr^{-1}$]')
    ax2.set_ylabel(r'J [$GeV^{-1}$ $cm^{-2}$ $s^{-1}$ $sr^{-1}$]')
    ax3.set_ylabel(r'J [$GeV^{-1}$ $cm^{-2}$ $s^{-1}$ $sr^{-1}$]')
    ax4.set_ylabel(r'J [$GeV^{-1}$ $cm^{-2}$ $s^{-1}$ $sr^{-1}$]')

    for ax in fig.get_axes():
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Muon energy [eV]')
        ax.legend()
    plt.show()
    plt.close()