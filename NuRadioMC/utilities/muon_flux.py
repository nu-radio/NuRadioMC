from NuRadioReco.utilities import units
import numpy as np
import os
import pickle
from scipy.interpolate import interp1d
from MCEq.core import MCEqRun
import crflux.models as crf
from functools import lru_cache

class MuonFlux:
    def __init__(self):

        self.mc_m = 1e2
        self.mc_rad = 180. / np.pi
        self.mc_eV = 1.e-9
        self.mc_ns = 1e-9

        self.__buffer = {}
        self.file_buffer = "data/surface_muon_buffer.pkl"
        if(os.path.exists(self.file_buffer)):
            fin = open(self.file_buffer, "rb")
            self.__buffer = pickle.load(fin)
            fin.close()


    @lru_cache(maxsize=5000)
    def get_mu_flux(self, theta, altitude=3200, interaction_model='SIBYLL23C', primary_model=(crf.GlobalSplineFitBeta, None), particle_names=("total_mu+", "total_mu-")):
        
        """
        The function get_mu_flux returns the muon flux at theta, for a given altitude, CR model, and hadronic interaction model. 
                
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
        e_grid: array of floats
            energy grid in eV
        flux: array of floats
            flux in NuRadioReco units 1/(area * time  * energy * steradian)
        """
         
        altitude *= self.mc_m
        mceq = MCEqRun(interaction_model=interaction_model,
                       primary_model=primary_model,
                       theta_deg=theta/units.deg)

        h_grid = np.linspace(50 * 1e3 * 1e2, 0, 500)
        X_grid = mceq.density_model.h2X(h_grid)

        alt_idx = np.abs(h_grid - altitude).argmin()

        mceq.solve(int_grid=X_grid)
        flux = None
        for particle_name in particle_names:
            if flux is None:
                flux = mceq.get_solution(particle_name, grid_idx=alt_idx, integrate=False)
            else:
                flux += mceq.get_solution(particle_name, grid_idx=alt_idx, integrate=False)

        flux *= self.mc_m ** 2 * self.mc_eV * self.mc_ns  # convert to NuRadioReco units
        e_grid = mceq.e_grid / self.mc_eV

        return e_grid, flux
    

    def get_interp_angle_mu_flux(self, theta_min, theta_max, altitude=3200, n_steps=3, primary_model=(crf.GlobalSplineFitBeta, None),
                          interaction_model='SIBYLL23C', particle_names=("total_mu+", "total_mu-")):
        """
        The function get_int_angle_mu_flux returns the interpolation function of the integrated muon flux from theta_min to theta_max, 
        The integration is just a simple Riemannian sum that should be enough, provided the band is small. 
        
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
            e_grid, flux_tmp = self.get_mu_flux(angle, altitude, primary_model=primary_model,
                                            interaction_model=interaction_model, particle_names=particle_names)
            
            # solid angle element is sin(theta) dtheta dphi in spherical coordinates.
            flux_tmp *= np.sin(angle) * (d_cos_theta * 2 * np.pi ) / n_steps 
            if(flux is None):
                flux = flux_tmp
            else:
                flux += flux_tmp

        return interp1d(np.log10(e_grid), flux, kind='cubic')


    def get_int_angle_mu_flux_buffered(self, energy, theta_min, theta_max, altitude=3200, n_steps=3, primary_model=(crf.GlobalSplineFitBeta, None),interaction_model='SIBYLL23C', particle_names=("total_mu+", "total_mu-")):

        """
        The function get_int_angle_mu_flux evalueates the integrated muon flux from theta_min to theta_max and caches the result.
            
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

        params = (np.round(energy), np.round(theta_min, 6), np.round(theta_max, 6), np.round(altitude),
                n_steps, primary_model, interaction_model, particle_names)

        if(params not in self.__buffer):
            print(f"calculating muon flux for {params}")
            finterp = self.get_interp_angle_mu_flux(theta_min, theta_max, altitude, n_steps=n_steps, primary_model=primary_model,
                                        interaction_model=interaction_model, particle_names=particle_names)

            flux = finterp(np.log10(energy))
            self.__buffer[params] = flux
            with open(self.file_buffer, "wb") as fout:
                pickle.dump(self.__buffer, fout, protocol=4)

        return self.__buffer[params]
    

    def get_e_grid(self, theta=50*units.deg, interaction_model='SIBYLL23C', primary_model=(crf.GlobalSplineFitBeta, None)):
        """
        Returns the energy grid for a given interaction model and primary model. Usually this is the same for all zenith angles.
            
        Parameters
        ----------
        theta_deg: float
            minimum zenith angle in rad
        interaction_model: str
            hadronic interaction model
        primary_model: tuple
            cosmic ray model

        Returns
        -------
        energies: array of floats
            energy grid in eV
        """
        mceq = MCEqRun(interaction_model=interaction_model, primary_model=primary_model, theta_deg=theta/units.deg)
        e_grid = mceq.e_grid / self.mc_eV
        return e_grid