from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units


def get_ice_model(name):
    if(name == "ARAsim_southpole"):
        return ARAsim_southpole()
    elif(name == "southpole_simple"):
        return southpole_simple()
    elif(name == "southpole_2015"):
        return southpole_2015()
    elif(name == "mooresbay_simple"):
        return mooresbay_simple()
    elif(name == "greenland_simple"):
        return greenland_simple()


class medium():

    def get_index_of_refraction(self, x):
        """
        returns the index of refraction at position x

        Parameters
        ---------
        x: 3dim np.array
            point

        Returns:
        --------
        n: float
            index of refraction
        """
        return self.n_ice - self.delta_n * np.exp(x[2] / self.z_0)

    def get_average_index_of_refraction(self, z1, z2):
        """
        returns the average index of refraction between two depths

        Parameters
        ----------
        z1: float
            depth 1
        z2: float
            depth 2

        Returns: depth averaged index of refraction
        """
        return self.n_ice - self.delta_n * self.z_0 / (z2 - z1) * (np.exp(z2 / self.z_0) - np.exp(z1 / self.z_0))


class southpole_simple(medium):

    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 RICE2014/SP model
        # define model parameters (RICE 2014/southpole)
        self.n_ice = 1.78
        self.z_0 = 71. * units.m
        self.delta_n = 0.426


class southpole_2015(medium):

    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 SPICE2015/SP model
        self.n_ice = 1.78
        self.z_0 = 77. * units.m
        self.delta_n = 0.423


class ARAsim_southpole(medium):

    def __init__(self):
        # define model parameters (SPICE 2015/southpole)
        self.n_ice = 1.78
        self.z_0 = 75.75757575757576 * units.m
        self.delta_n = 0.43


class mooresbay_simple(medium):

    # from https://doi.org/10.1088/1475-7516/2018/07/055 MB1 model
    def __init__(self):
        self.n_ice = 1.78
        self.z_0 = 34.5 * units.m
        self.delta_n = 0.46
        self.reflection = -576 * units.m  # from https://doi.org/10.3189/2015JoG14J214
        self.reflection_coefficient = 0.82  # from https://doi.org/10.3189/2015JoG14J214
        self.reflection_phase_shift = 180 * units.deg


class mooresbay_simple_2(medium):

    # from https://doi.org/10.1088/1475-7516/2018/07/055 MB2 model
    def __init__(self):
        self.n_ice = 1.78
        self.z_0 = 37 * units.m
        self.delta_n = 0.481
        self.reflection = -576 * units.m  # from https://doi.org/10.3189/2015JoG14J214
        self.reflection_coefficient = 0.82  # from https://doi.org/10.3189/2015JoG14J214
        self.reflection_phase_shift = 180 * units.deg


class greenland_simple(medium):

    # from C. Deaconu, fit to data from Hawley '08, Alley '88
    # rho(z) = 917 - 602 * exp (-z/37.25), using n = 1 + 0.78 rho(z)/rho_0
    def __init__(self):
        self.n_ice = 1.78
        self.z_0 = 37.25 * units.m
        self.delta_n = 0.51
