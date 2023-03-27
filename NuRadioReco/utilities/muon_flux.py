from NuRadioReco.utilities import units
import numpy as np
import os
import pickle
import json
from scipy.interpolate import interp1d
from MCEq.core import MCEqRun
import crflux.models as crf
from functools import lru_cache
from scipy.integrate import quad
from scipy.interpolate import interp1d

mc_cm = 1.
mc_m = 1e2
mc_km = 1e3 * mc_m

mc_deg = 1.
mc_rad = 180. / np.pi

mc_eV = 1.e-9
mc_GeV = 1
mc_TeV = 1.e3
mc_PeV = 1.e6
mc_EeV = 1.e9

mc_s = 1
mc_ns = 1e-9

unc_file = 'Barr_uncertainties.json'
# calculated by anatoli for MCEq
with open(unc_file, 'r') as f:
    unc_dict = json.load(f)

CR_log_energies_GeV = np.array(unc_dict['CR']['log_energies_GeV'])
CR_log_unc = np.array(unc_dict['CR']['log_unc'])
hadronic_log_energies_GeV = np.array(unc_dict['hadronic']['log_energies_GeV'])
hadronic_log_unc = np.array(unc_dict['hadronic']['log_unc'])

def get_hadronic_unc(energy):
    finterp = interp1d(hadronic_log_energies_GeV, hadronic_log_unc, fill_value='extrapolate')
    log_uncertainties = finterp(np.log10(energy/units.GeV))
    uncertainties = 10 ** log_uncertainties
    if(uncertainties > 1):  # why are uncertainties limited in this way?
        uncertainties = 1
#     uncertainties[uncertainties > 1] = 1
    return uncertainties

def get_CR_unc(energy):
    finterp = interp1d(CR_log_energies_GeV, CR_log_unc, fill_value='extrapolate')
    log_uncertainties = finterp(np.log10(energy/units.GeV))
    uncertainties = 10 ** log_uncertainties
    if(uncertainties > 1):
        uncertainties = 1
#     uncertainties[uncertainties > 1] = 1
    return uncertainties

@lru_cache(maxsize=5000)
def get_mu_flux(theta, altitude=3200,
                       interaction_model='SIBYLL23C',  # High-energy hadronic interaction model
                       primary_model=(crf.GlobalSplineFitBeta, None),
                       particle_names=("total_mu+", "total_mu-")):  # cosmic ray flux at the top of the atmosphere
    """
    The function get_mu_flux returns the mceq object containing the information for the total muon flux at a
    given angle, altitude, interaction model, and cosmic ray model (primary model). 
    
    Parameters
    
    ----------

    theta: float
        zenith angle in rad       
    altitude: float
        altitude in meters
    interaction_model: str
        hadronic interaction model
        e.g. ['SIBYLL23C', 'EPOS-LHC', 'QGSJet-II-04']
    primary_model: tuple
        cosmic ray flux model
    particle_names: tuple  
        particle names to be considered 
        e.g. ("pr_mu+", "pr_mu-"), ("conv_mu+", "conv_mu-"), ("total_mu+", "total_mu-")
    
    Returns
    -------
    e_grid: array
        energy grid in eV
    flux: array
        flux in NuRadioReco units 1/(area * time * solid angle * energy)        
    """

    altitude *= mc_m
    mceq = MCEqRun(interaction_model=interaction_model,
                   primary_model=primary_model,
                   theta_deg=theta * mc_rad)  # theta is given in rad but mc unit is deg

    h_grid = np.linspace(50 * 1e3 * 1e2, 0, 500)  # altitudes from 0 to 50 km (in cm)
    X_grid = mceq.density_model.h2X(h_grid)

    alt_idx = np.abs(h_grid - altitude).argmin()

    mceq.solve(int_grid=X_grid) # solve equation system

    flux = None
    for particle_name in particle_names:
        if flux is None:
            flux = mceq.get_solution(particle_name, grid_idx=alt_idx, integrate=False)
        else:
            flux += mceq.get_solution(particle_name, grid_idx=alt_idx, integrate=False)

    flux *= mc_m ** 2 * mc_eV * mc_ns  # convert to NuRadioReco units
    e_grid = mceq.e_grid / mc_eV

    return e_grid, flux


def get_interp_angle_mu_flux(theta_min, theta_max, altitude=3200, n_steps=3, primary_model=(crf.GlobalSplineFitBeta, None),
                          interaction_model='SIBYLL23C', particle_names=("total_mu+", "total_mu-")):
    """
    The function get_int_angle_mu_flux returns the integrated muon flux from theta_min to theta_max, 
    for a given altitude, CR model, and hadronic interaction model. The integration is just a simple Riemannian sum 
    that should be enough, provided the band is small. 
    
    Returns zenith angle integrated flux in NuRadioReco units 1/(area * time  * energy)
    
    Parameters
    
    ----------
    energy: float
        energy in eV
    theta_min: float
        minimum zenith angle in rad
    theta_max: float
        maximum zenith angle in rad
    n_steps: int
        number of steps to use for the numerical integration

    Returns
    -------
    interpolator: function
        function that returns the flux for a given energy
    """
    angle_edges = np.arccos(np.linspace(np.cos(theta_max), np.cos(theta_min), n_steps + 1))
    angle_centers = 0.5 *(angle_edges[1:] + angle_edges[:-1])
    d_cos_theta = np.abs(np.cos(theta_min) - np.cos(theta_max))

    #print(f"integrating the flux from {theta_min/units.deg:.1f} deg to {theta_max/units.deg:.1f} deg by adding the flux from {angle_centers/units.deg}")

    flux = None
    for angle in angle_centers:
        e_grid, flux_tmp = get_mu_flux(angle, altitude, primary_model=primary_model,
                                        interaction_model=interaction_model, particle_names=particle_names)
        
        # solid angle element is sin(theta) dtheta dphi in spherical coordinates.
        flux_tmp *= np.sin(angle) * (d_cos_theta * 2 * np.pi ) / n_steps 
        if(flux is None):
            flux = flux_tmp
        else:
            flux += flux_tmp

    return interp1d(np.log10(e_grid), flux, kind='cubic')

def get_int_angle_mu_flux(energy, theta_min, theta_max, altitude=3200, n_steps=3, primary_model=(crf.GlobalSplineFitBeta, None),interaction_model='SIBYLL23C', particle_names=("total_mu+", "total_mu-")):
    """
    The function get_int_angle_mu_flux returns the integrated muon flux from theta_min to theta_max, 
    for a given altitude, CR model, and hadronic interaction model. The integration is just a simple Riemannian sum 
    that should be enough, provided the band is small. 
        
    Parameters
    
    ----------
    energy: float
        energy in eV
    theta_min: float
        minimum zenith angle in rad
    theta_max: float
        maximum zenith angle in rad
    altitude: float
        altitude in meters
    n_steps: int
        number of steps to use for the numerical integration

    Returns
    -------
    flux: float
        zenith angle integrated flux in NuRadioReco units 1/(area * time  * energy)    
    """

    finterp = get_interp_angle_mu_flux(theta_min, theta_max, altitude, n_steps=n_steps, primary_model=primary_model,
                                     interaction_model=interaction_model, particle_names=particle_names)
    
    return finterp(np.log10(energy))

def get_int_energy_angle_mu_flux(Emin, Emax, theta_min, theta_max, altitude, n_steps=3, primary_model=(crf.GlobalSplineFitBeta, None), interaction_model='SIBYLL23C', particle_names=("total_mu+", "total_mu-")):

    """
    Calculates the muon flux integrated over the energy and zenith angle range.  Returns flux in NuRadioReco units 1/(area * time)


    Parameters
    ----------
    Emin: float
        minimum energy in eV
    Emax: float
        maximum energy in eV
    theta_min: float
        minimum zenith angle in rad
    theta_max: float
        maximum zenith angle in rad
    altitude: float
        altitude in meters
    n_steps: int
        number of steps to use for the numerical integration

    Returns
    -------
    
    """

    def integrand(energy):
        return get_int_angle_mu_flux(energy, theta_min, theta_max, altitude, n_steps, primary_model, interaction_model, particle_names)  / (energy * np.log(10))

    flux, error = quad(integrand, Emin, Emax, epsabs=0, epsrel=1e-3)

    return flux