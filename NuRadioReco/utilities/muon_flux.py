from NuRadioReco.utilities import units
import numpy as np
import os
import glob
import json
import pickle
from scipy.interpolate import interp1d
from MCEq.core import MCEqRun
import crflux.models as crf
import crflux
from functools import lru_cache
from scipy.integrate import quad
import logging
logger = logging.getLogger('muon_flux')

# define MCEq units
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

h_grid = np.linspace(50 * 1e3 * 1e2, 0, 500)  # altitudes from 0 to 50 km (in cm)


@lru_cache(maxsize=5000)
def get_mu_flux(theta_deg,
                altitude,
                interaction_model='SIBYLL23C',  # High-energy hadronic interaction model
                primary_model=(crf.GlobalSplineFitBeta, None),
                particle_names=("total_mu+", "total_mu-")):  # cosmic ray flux at the top of the atmosphere
    """
    The function get_mu_flux_interp returns the mceq object containing the information for the total muon flux at a
    given angle, altitude, interaction model, and cosmic ray model (primary model).
    The following primary models are relevant:

        (crf.GlobalSplineFitBeta, None), for the GSF model, the most complete and up-to-date.

    Returns flux in NuRadioReco units 1/(area * time * solid angle * energy)
    """

    altitude *= mc_m
    mceq = MCEqRun(interaction_model=interaction_model,
                   primary_model=primary_model,
                   theta_deg=theta_deg * mc_rad)

    X_grid = mceq.density_model.h2X(h_grid)

    alt_idx = np.abs(h_grid - altitude).argmin()

    mceq.solve(int_grid=X_grid)

    flux = None
    for particle_name in particle_names:
        if flux is None:
            flux = mceq.get_solution(particle_name, grid_idx=alt_idx, mag=0, integrate=False)
        else:
            flux += mceq.get_solution(particle_name, grid_idx=alt_idx, mag=0, integrate=False)

    flux *= mc_m ** 2 * mc_eV * mc_ns  # convert to NuRadioReco units
    e_grid = mceq.e_grid / mc_eV

    return e_grid, flux


def get_int_angle_mu_flux(theta_min, theta_max, altitude, n_steps=3, primary_model=(crf.GlobalSplineFitBeta, None),
                          interaction_model='SIBYLL23C', particle_names=("total_mu+", "total_mu-")):
    """
    The function get_int_angle_mu_flux returns the integrated muon flux from theta_min to theta_max,
    for a given altitude, CR model, and hadronic interaction model. The integration is just a simple Riemannian sum
    that should be enough, provided the band is small.

    Returns zenith angle integrated flux in NuRadioReco units 1/(area * time  * energy)
    """

    angle_edges = np.linspace(theta_min, theta_max, n_steps + 1)
    step_theta = np.abs(angle_edges[1] - angle_edges[0])
    angles = 0.5 * (angle_edges[1:] + angle_edges[:-1])
    logger.info(f"integrating the flux from {theta_min / units.deg:.1f} deg to {theta_max / units.deg:.1f} deg by adding the flux from {angles / units.deg}")

    flux = None
    for angle in angles:
        e_grid, flux_tmp = get_mu_flux(angle, altitude, primary_model=primary_model,
                                       interaction_model=interaction_model, particle_names=particle_names)
        flux_tmp *= np.sin(angle)
        if flux is None:
            flux = flux_tmp
        else:
            flux += flux_tmp

    flux_norm = step_theta * 2 * np.pi  # We also integrate in azimuth
    flux *= flux_norm

    return interp1d(e_grid, flux, kind='cubic')


buffer = {}
file_buffer = "data/surface_muon_buffer.pkl"
if os.path.exists(file_buffer):
    fin = open(file_buffer, "rb")
    buffer = pickle.load(fin)
    fin.close()


def get_int_energy_mu_flux(Emin, Emax, theta_min, theta_max, altitude, n_steps=3, primary_model=(crf.GlobalSplineFitBeta, None),
                                    interaction_model='SIBYLL23C', particle_names=("total_mu+", "total_mu-")):
    """
    Returns zenith angle and energy integrated flux in NuRadioReco units 1/(area * time)
    """

    params = (np.round(Emin), np.round(Emax), np.round(theta_min, 6), np.round(theta_max, 6), np.round(altitude),
              n_steps, primary_model, interaction_model, particle_names)

    if params not in buffer:
        logger.info(f"calculating muon flux for {params}")
        finterp = get_int_angle_mu_flux(theta_min, theta_max, altitude, n_steps=n_steps, primary_model=primary_model,
                                        interaction_model=interaction_model, particle_names=particle_names)
        flux = quad(finterp, Emin, Emax)[0]
        buffer[params] = flux

        os.makedirs('data', exist_ok=True)
        with open(file_buffer, "wb") as fout:
            pickle.dump(buffer, fout, protocol=4)

    logger.info(f"muon flux from {Emin:.2g}eV to {Emax:.2g}eV and {theta_min / units.deg:.0f} to {theta_max / units.deg:.0f} deg = {buffer[params]:.2g} / m^2/ns")
    return buffer[params]
