#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from NuRadioReco.utilities import units, io_utilities
from scipy import interpolate as intp
from scipy import integrate as int
from scipy import constants
from matplotlib import pyplot as plt
from radiotools import coordinatesystems as cstrafo
from NuRadioReco.utilities.metaclasses import Singleton
import os
import copy
import logging
import six
logger = logging.getLogger("SignalGen.ARZ")
logging.basicConfig()

######################
######################
# This code is based on "J. Alvarez-Muniz, P. Hansen, A. Romero-Wolf, E. Zas in preparation" which is an extension of
# Alvarez-Muñiz, J., Romero-Wolf, A., & Zas, E. (2011). Practical and accurate calculations of Askaryan radiation. Physical Review D - Particles, Fields, Gravitation and Cosmology, 84(10). https://doi.org/10.1103/PhysRevD.84.103003
# to hadronic showers. Thanks a lot to Jaime Alvarez-Muñiz for providing us with this unpublished work!
#####################
#####################

# define constants
# x0 = 36.08 * units.g / units.cm**2  # radiation length g cm^-2
rho = 0.924 * units.g / units.cm ** 3  # density g cm^-3
xmu = 12.566370e-7 * units.newton / units.ampere ** 2
c = 2.99792458e8 * units.m / units.s
# e = 1.602177e-19 * units.coulomb


def thetaprime_to_theta(thetaprime, xmax, R):
    """
    convertes a viewing angle relative to the shower maximum to a viewing angle relative to the start of the shower.
    """
    L = xmax / rho
    return thetaprime - np.arcsin((L * np.sin(np.pi - thetaprime)) / R)


def theta_to_thetaprime(theta, xmax, R):
    """
    converts a viewing angle relative to the start of the shower to a viewing angle relative to the shower maximum
    """
    L = xmax / rho
    b = R * np.sin(theta)
    a = R * np.cos(theta) - L
    return np.arctan2(b, a)


@six.add_metaclass(Singleton)
class ARZ(object):

    def __init__(self, seed=1234, interp_factor=1, interp_factor2=100, library=None,
                 arz_version='ARZ2020'):
        logger.warning("setting seed to {}".format(seed, interp_factor))
        self._random_generator = np.random.RandomState(seed)
        self._interp_factor = interp_factor
        self._interp_factor2 = interp_factor2
        self._random_numbers = {}
        self._version = (1, 2)
        # # load shower library into memory
        if(library is None):
            library = os.path.join(os.path.dirname(__file__), "shower_library/library_v{:d}.{:d}.pkl".format(*self._version))
        else:
            if(not os.path.exists(library)):
                logger.error("user specified shower library {} not found.".format(library))
                raise FileNotFoundError("user specified shower library {} not found.".format(library))
        self.__check_and_get_library()
        self.__set_model_parameters(arz_version)

        logger.warning("loading shower library ({}) into memory".format(library))
        self._library = io_utilities.read_pickle(library)

    def __check_and_get_library(self):
        """
        checks if shower library exists and is up to date by comparing the sha1sum. If the library does not exist
        or changes on the server, a new library will be downloaded.
        """
        path = os.path.join(os.path.dirname(__file__), "shower_library/library_v{:d}.{:d}.pkl".format(*self._version))

        download_file = False
        if(not os.path.exists(path)):
            logger.warning("shower library version {} does not exist on the local file system yet. It will be downloaded to {}".format(self._version, path))
            download_file = True

        if(os.path.exists(path)):
            BUF_SIZE = 65536 * 2 ** 4  # lets read stuff in 64kb chunks!
            import hashlib
            import json
            sha1 = hashlib.sha1()
            with open(path, 'rb') as f:
                while True:
                    data = f.read(BUF_SIZE)
                    if not data:
                        break
                    sha1.update(data)

            shower_directory = os.path.join(os.path.dirname(__file__), "shower_library/")
            with open(os.path.join(shower_directory, 'shower_lib_hash.json'), 'r') as fin:
                lib_hashs = json.load(fin)
                if("{:d}.{:d}".format(*self._version) in lib_hashs.keys()):
                    if(sha1.hexdigest() != lib_hashs["{:d}.{:d}".format(*self._version)]):
                        logger.warning("shower library {} has changed on the server. downloading newest version...".format(self._version))
                        download_file = True
                else:
                    logger.warning("no hash sum of {} available, skipping up-to-date check".format(os.path.basename(path)))
        if not download_file:
            return True
        else:
            import requests
            URL = 'http://arianna.ps.uci.edu/~arianna/data/ce_shower_library/library_v{:d}.{:d}.pkl'.format(*self._version)

            logger.info("downloading shower library {} from {}. This can take a while...".format(self._version, URL))
            r = requests.get(URL)
            if (r.status_code != requests.codes.ok):
                logger.error("error in download of antenna model")
                raise IOError("error in download of antenna model")
            with open(path, "wb") as code:
                code.write(r.content)
            logger.info("...download finished.")

    def __set_model_parameters(self, arz_version='ARZ2020'):
        """
        Sets the parameters for the form factor
        """
        if (arz_version == 'ARZ2019'):
            # Refit of ZHAireS results => factor 0.88 in Af_e
            self._Af_e = -4.5e-14 * 0.88 * units.V * units.s
            self._t0_e_pos = 0.057 * units.ns
            self._freq_e_pos = 2.87 / units.ns
            self._exp_e_pos = -3.00
            self._t0_e_neg = 0.030 * units.ns
            self._freq_e_neg = 3.05 / units.ns
            self._exp_e_neg = -3.50

            self._Af_p = -3.2e-14 * units.V * units.s
            self._t0_p_pos = 0.065 * units.ns
            self._freq_p_pos = 3.00 / units.ns
            self._exp_p_pos = -2.65
            self._t0_p_neg = 0.043 * units.ns
            self._freq_p_neg = 2.92 / units.ns
            self._exp_p_neg = -3.21
            self._include_em_factor = False
        elif (arz_version == 'ARZ2020'):
            self._Af_e = -4.445e-14 * units.V * units.s
            self._t0_e_pos = 0.0348 * units.ns
            self._freq_e_pos = 2.298 / units.ns
            self._exp_e_pos = -3.588
            self._t0_e_neg = 0.0203 * units.ns
            self._freq_e_neg = 2.616 / units.ns
            self._exp_e_neg = -4.043

            self._Af_p = -4.071e-14 * units.V * units.s
            self._t0_p_pos = 0.0391 * units.ns
            self._freq_p_pos = 2.338 / units.ns
            self._exp_p_pos = -3.320
            self._t0_p_neg = 0.0234 * units.ns
            self._freq_p_neg = 2.686 / units.ns
            self._exp_p_neg = -3.687
            self._include_em_factor = True
        else:
            raise ValueError('ARZ version does not exist. Please choose ARZ2019 or ARZ2020.')

    def em_fraction(self, energy):
        """
        Returns the energy fraction carried by the electromagnetic component of
        a hadronic shower
        """

        if not self._include_em_factor:
            return 1

        epsilon = np.log10(energy / units.eV)
        f_epsilon = -21.98905 - 2.32492 * epsilon
        f_epsilon += 0.019650 * epsilon ** 2 + 13.76152 * np.sqrt(epsilon)

        return f_epsilon

    def set_seed(self, seed):
        """
        allow to set a new random seed
        """
        self._random_generator.seed(seed)

    def set_interpolation_factor(self, interp_factor):
        """
        set interpolation factor of charge-excess profiles
        """
        self._interp_factor = interp_factor

    def set_interpolation_factor2(self, interp_factor):
        """
        set interpolation factor around peak of form factor
        """
        self._interp_factor2 = interp_factor

    def get_time_trace(self, shower_energy, theta, N, dt, shower_type, n_index, R, shift_for_xmax=False,
                       same_shower=False, iN=None, output_mode='trace', theta_reference='X0'):
        """
        calculates the electric-field Askaryan pulse from a charge-excess profile

        Parameters
        ----------
        shower_energy: float
            the energy of the shower
        theta: float
            viewing angle, i.e., the angle between shower axis and launch angle of the signal (the ray path)
        N: int
            number of samples in the time domain
        dt: float
            size of one time bin in units of time
        profile_depth: array of floats
            shower depth values of the charge excess profile
        profile_ce: array of floats
            charge-excess values of the charge excess profile
        shower_type: string (default "HAD")
            type of shower, either "HAD" (hadronic), "EM" (electromagnetic) or "TAU" (tau lepton induced)
        n_index: float (default 1.78)
            index of refraction where the shower development takes place
        R: float (default 1km)
            observation distance, the signal amplitude will be scaled according to 1/R
        interp_factor: int (default 10)
            interpolation factor of charge-excess profile. Results in a more precise numerical integration which might be beneficial
            for small vertex distances but also slows down the calculation proportional to the interpolation factor.
        shift_for_xmax: bool (default True)
            if True the observer position is placed relative to the position of the shower maximum, if False it is placed
            with respect to (0,0,0) which is the start of the charge-excess profile
        same_shower: bool (default False)
            if False, for each request a new random shower realization is choosen.
            if True, the shower from the last request of the same shower type is used. This is needed to get the Askaryan
            signal for both ray tracing solutions from the same shower.
        iN: int or None (default None)
            specify shower number
        output_mode: string
            * 'trace' (default): return only the electric field trace
            * 'Xmax': return trace and position of xmax in units of length
            * 'full' return trace, depth and charge_excess profile
        theta_reference: string (default: X0)
            * 'X0': viewing angle relativ to start of the shower
            * 'Xmax': viewing angle is relativ to Xmax, internally it will be converted to be relative to X0

        Returns: array of floats
            array of electric-field time trace in 'on-sky' coordinate system eR, eTheta, ePhi
        """
        if not shower_type in self._library.keys():
            raise KeyError("shower type {} not present in library. Available shower types are {}".format(shower_type, *self._library.keys()))

        # determine closes available energy in shower library
        energies = np.array([*self._library[shower_type]])
        iE = np.argmin(np.abs(energies - shower_energy))
        rescaling_factor = shower_energy / energies[iE]
        logger.info("shower energy of {:.3g}eV requested, closest available energy is {:.3g}eV. The amplitude of the charge-excess profile will be rescaled accordingly by a factor of {:.2f}".format(shower_energy / units.eV, energies[iE] / units.eV, rescaling_factor))
        profiles = self._library[shower_type][energies[iE]]
        N_profiles = len(profiles['charge_excess'])

        if(iN is None):
            if(same_shower):
                if(shower_type in self._random_numbers):
                    iN = self._random_numbers[shower_type]
                    logger.info("using previously used shower {}/{}".format(iN, N_profiles))
                else:
                    logger.warning("no previous random number for shower type {} exists. Generating a new random number.".format(shower_type))
                    iN = self._random_generator.randint(N_profiles)
                    self._random_numbers[shower_type] = iN
                    logger.info("picking profile {}/{} randomly".format(iN, N_profiles))
            else:
                iN = self._random_generator.randint(N_profiles)
                self._random_numbers[shower_type] = iN
                logger.info("picking profile {}/{} randomly".format(iN, N_profiles))
        else:
            logger.info("using shower {}/{} as specified by user".format(iN, N_profiles))

        profile_depth = profiles['depth']
        profile_ce = profiles['charge_excess'][iN] * rescaling_factor

        xmax = profile_depth[np.argmax(profile_ce)]
        if(theta_reference == 'Xmax'):
            thetat = copy.copy(theta)
            theta = thetaprime_to_theta(theta, xmax, R)
            logger.info("transforming viewing angle from {:.2f} to {:.2f}".format(thetat / units.deg, theta / units.deg))
        elif(theta_reference != 'X0'):
            raise NotImplementedError("theta_reference = '{}' is not implemented".format(theta_reference))

        vp = self.get_vector_potential_fast(shower_energy, theta, N, dt, profile_depth, profile_ce, shower_type, n_index, R,
                                            self._interp_factor, self._interp_factor2, shift_for_xmax)
        trace = -np.diff(vp, axis=0) / dt
#         trace = -np.gradient(vp, axis=0) / dt

        # use viewing angle relative to shower maximum for rotation into spherical coordinate system (that reduced eR component)
        thetaprime = theta_to_thetaprime(theta, xmax, R)
        cs = cstrafo.cstrafo(zenith=thetaprime, azimuth=0)
        trace_onsky = cs.transform_from_ground_to_onsky(trace.T)
        if(output_mode == 'full'):
            return trace_onsky, profile_depth, profile_ce
        elif(output_mode == 'Xmax'):
            xmax = profile_depth[np.argmax(profile_ce)]
            Lmax = xmax / rho
            return trace_onsky, Lmax
        return trace_onsky

    def get_last_shower_profile_id(self):
        """
        returns dict
            the index of the randomly selected shower profile per shower type
            key is the shower type (string)
            value is the index (int)
        """
        return self._random_numbers

    def get_vector_potential_fast(self, shower_energy, theta, N, dt, profile_depth, profile_ce,
                                  shower_type="HAD", n_index=1.78, distance=1 * units.m,
                                  interp_factor=1., interp_factor2=100., shift_for_xmax=False):
        """
        fast interpolation of time-domain calculation of vector potential of the
        Askaryan pulse from a charge-excess profile

        Note that the returned array has N+1 samples so that the derivative (the efield) will have N samples.

        The numerical integration was replaces by a sum using the trapeoiz rule using vectorized numpy operations

        Parameters
        ----------
        shower_energy: float
            the energy of the shower
        theta: float
            viewing angle, i.e., the angle between shower axis and launch angle of the signal (the ray path)
        N: int
            number of samples in the time domain
        dt: float
            size of one time bin in units of time
        profile_depth: array of floats
            shower depth values of the charge excess profile
        profile_ce: array of floats
            charge-excess values of the charge excess profile
        shower_type: string (default "HAD")
            type of shower, either "HAD" (hadronic), "EM" (electromagnetic) or "TAU" (tau lepton induced)
        n_index: float (default 1.78)
            index of refraction where the shower development takes place
        distance: float (default 1km)
            observation distance, the signal amplitude will be scaled according to 1/R
        interp_factor: int (default 1)
            interpolation factor of charge-excess profile. Results in a more precise numerical integration which might be beneficial
            for small vertex distances but also slows down the calculation proportional to the interpolation factor.
            if None, the interpolation factor will be calculated from the distance
        interp_factor2: int (default 100)
            interpolation just around the peak of the form factor
        shift_for_xmax: bool (default True)
            if True the observer position is placed relative to the position of the shower maximum, if False it is placed
            with respect to (0,0,0) which is the start of the charge-excess profile
        """

        ttt = np.arange(0, (N + 1) * dt, dt)
        ttt = ttt + 0.5 * dt - ttt.mean()
        if(len(ttt) != N + 1):
            ttt = ttt[:-1]
        N = len(ttt)

        xn = n_index
        cher = np.arccos(1. / n_index)
        beta = 1.

        profile_dense = profile_depth
        profile_ce_interp = profile_ce
        if(interp_factor != 1):
            profile_dense = np.linspace(min(profile_depth), max(profile_depth), interp_factor * len(profile_depth))
            profile_ce_interp = np.interp(profile_dense, profile_depth, profile_ce)
        length = profile_dense / rho
        dxmax = length[np.argmax(profile_ce_interp)]
    #     theta2 = np.arctan(R * np.sin(theta)/(R * np.cos(theta) - dxmax))
    #     logger.warning("theta changes from {:.2f} to {:.2f}".format(theta/units.deg, theta2/units.deg))

        # calculate antenna position in ARZ reference frame
        # coordinate system is with respect to an origin which is located
        # at the position where the primary particle is injected in the medium. The reference frame
        # is z = along shower axis, and x,y are two arbitray directions perpendicular to z
        # and perpendicular among themselves of course.
        # For instance to place an observer at a distance R and angle theta w.r.t. shower axis in the x,z plane
        # it can be simply done by putting in the input file the numerical values:
        X = np.array([distance * np.sin(theta), 0., distance * np.cos(theta)])
        if(shift_for_xmax):
            logger.info("shower maximum at z = {:.1f}m, shifting observer position accordingly.".format(dxmax / units.m))
            X = np.array([distance * np.sin(theta), 0., distance * np.cos(theta) + dxmax])
        logger.info("setting observer position to {}".format(X))

        def get_dist_shower(X, z):
            """
            Distance from position in shower depth z' to each antenna.
            Denominator in Eq. (22) PRD paper

            Parameters
            ----------
            X: 3dim np. array
                position of antenna in ARZ reference frame
            z: shower depth
            """
            return (X[0] ** 2 + X[1] ** 2 + (X[2] - z) ** 2) ** 0.5

        # calculate total charged track length
        xntot = np.sum(profile_ce_interp) * (length[1] - length[0])
        factor = -xmu / (4. * np.pi)
        fc = 4. * np.pi / (xmu * np.sin(cher))

        vp = np.zeros((N, 3))
        for it, t in enumerate(ttt):
            tobs = t + (get_dist_shower(X, 0) / c * xn)
            z = length

            R = get_dist_shower(X, z)
            arg = z - (beta * c * tobs - xn * R)

            # Note that Acher peaks at tt=0 which corresponds to the observer time.
            # The shift from tobs to tt=0 is done when defining argument
            tt = (-arg / (c * beta))  # Parameterisation of A_Cherenkov with t in ns

            mask = abs(tt) < 20. * units.ns
            if(np.sum(mask) == 0):  #
                vp[it] = 0
                continue

            profile_dense2 = profile_dense
            profile_ce_interp2 = profile_ce_interp
            abc = False
            if(interp_factor2 != 1):
                # we only need to interpolate between +- 1ns to achieve a better precision in the numerical integration
                # the following code finds the indices sourrounding the bins fulfilling these condition
                # please not that we often have two distinct intervals having -1 < tt < 1
                tmask = (tt < 1 * units.ns) & (tt > -1 * units.ns)
                gaps = (tmask[1:] ^ tmask[:-1])  # xor
                indices = np.arange(len(gaps))[gaps]  # the indices in between tt is within -+ 1ns
                if(len(indices) != 0):  # only interpolate if we have time within +- 1 ns of the observer time
                    # now we add the corner cases of having the tt array start or end with an entry fulfilling the condition
                    if(len(indices) % 2 != 0):
                        if((tt[0] < 1 * units.ns) and (tt[0] > -1 * units.ns) and indices[0] != 0):
                            indices = np.append(0, indices)
                        else:
                            if(indices[-1] != (len(tt) - 1)):
                                indices = np.append(indices, len(tt) - 1)
                    if(len(indices) % 2 == 0):  # this rejects the cases where only the first or the last entry fulfills the -1 < tt < 1 condition
                        dt = tt[1] - tt[0]

                        dp = profile_dense2[1] - profile_dense2[0]
                        if(len(indices) == 2):  # we have only one interval
                            i_start = indices[0]
                            i_stop = indices[1]
                            profile_dense2 = np.arange(profile_dense[i_start], profile_dense[i_stop], dp / interp_factor2)
                            profile_ce_interp2 = np.interp(profile_dense2, profile_dense[i_start:i_stop], profile_ce_interp[i_start:i_stop])
                            profile_dense2 = np.append(np.append(profile_dense[:i_start], profile_dense2), profile_dense[i_stop:])
                            profile_ce_interp2 = np.append(np.append(profile_ce_interp[:i_start], profile_ce_interp2), profile_ce_interp[i_stop:])
                        elif(len(indices) == 4):  # we have two intervals, hence, we need to upsample two distinct intervals and put the full array back together.
                            i_start = indices[0]
                            i_stop = indices[1]
                            profile_dense2 = np.arange(profile_dense[i_start], profile_dense[i_stop], dp / interp_factor2)
                            profile_ce_interp2 = np.interp(profile_dense2, profile_dense[i_start:i_stop], profile_ce_interp[i_start:i_stop])

                            i_start3 = indices[2]
                            i_stop3 = indices[3]
                            profile_dense3 = np.arange(profile_dense[i_start3], profile_dense[i_stop3], dp / interp_factor2)
                            profile_ce_interp3 = np.interp(profile_dense3, profile_dense[i_start3:i_stop3], profile_ce_interp[i_start3:i_stop3])

                            profile_dense2 = np.append(np.append(np.append(np.append(
                                                            profile_dense[:i_start], profile_dense2),
                                                               profile_dense[i_stop:i_start3]),
                                                                  profile_dense3),
                                                                       profile_dense[i_stop3:])
                            profile_ce_interp2 = np.append(np.append(np.append(np.append(
                                                    profile_ce_interp[:i_start],
                                                    profile_ce_interp2),
                                                    profile_ce_interp[i_stop:i_start3]),
                                                    profile_ce_interp3),
                                                    profile_ce_interp[i_stop3:])

                        else:
                            raise NotImplementedError("length of indices is not 2 nor 4")  # this should never happen
                        if 0:
                            abc = True
                            i_stop = len(profile_dense) - 1
                            from matplotlib import pyplot as plt
                            fig, ax = plt.subplots(1, 1)
                            ax.plot(tt, color='0.5')
                            ax.plot(np.arange(len(tmask))[tmask], tt[tmask], 'o')
                            ax.plot(indices, np.ones_like(indices), 'd')
            #                 ax.plot(np.arange(len(tmask))[gaps], tt[gaps], 'd')
                            plt.show()

                        # recalculate parameters for interpolated values
                        z = profile_dense2 / rho
                        R = get_dist_shower(X, z)
                        arg = z - (beta * c * tobs - xn * R)
                        tt = (-arg / (c * beta))
                        mask = abs(tt) < 20. * units.ns
                        tmask = (tt < 1 * units.ns) & (tt > -1 * units.ns)

            F_p = np.zeros_like(tt)
            # Cut fit above +/-5 ns

            u_x = X[0] / R
            u_y = X[1] / R
            u_z = (X[2] - z) / R
            beta_z = 1.
            vperp_x = u_x * u_z * beta_z
            vperp_y = u_y * u_z * beta_z
            vperp_z = -(u_x * u_x + u_y * u_y) * beta_z
            v = np.array([vperp_x, vperp_y, vperp_z])
            """
            Function F_p Eq.(15) PRD paper.
            """
            # Factor accompanying the F_p in Eq.(15) in PRD paper
            beta = 1.
            if(np.sum(mask)):
                # Choose Acher between purely electromagnetic, purely hadronic or mixed shower
                # Eq.(16) PRD paper.
                E_TeV = shower_energy / units.TeV
                Acher = np.zeros_like(tt)
                if(shower_type == "HAD"):
                    mask2 = tt > 0 & mask
                    if(np.sum(mask2)):
                        Acher[mask2] = self._Af_p * E_TeV * (np.exp(-np.abs(tt[mask2]) / self._t0_p_pos) +
                                              (1. + self._freq_p_pos * np.abs(tt[mask2])) ** self._exp_p_pos)  # hadronic
                    mask2 = tt <= 0 & mask
                    if(np.sum(mask2)):
                        Acher[mask2] = self._Af_p * E_TeV * (np.exp(-np.abs(tt[mask2]) / self._t0_p_neg) +
                                              (1. + self._freq_p_neg * np.abs(tt[mask2])) ** self._exp_p_neg)  # hadronic
                    Acher *= self.em_fraction(shower_energy)
                elif(shower_type == "EM"):
                    mask2 = tt > 0 & mask
                    if(np.sum(mask2)):
                        Acher[mask2] = self._Af_e * E_TeV * (np.exp(-np.abs(tt[mask2]) / self._t0_e_pos) +
                                              (1. + self._freq_e_pos * np.abs(tt[mask2])) ** self._exp_e_pos)  # electromagnetic
                    mask2 = tt <= 0 & mask
                    if(np.sum(mask2)):
                        Acher[mask2] = self._Af_e * E_TeV * (np.exp(-np.abs(tt[mask2]) / self._t0_e_neg) +
                                              (1. + self._freq_e_neg * np.abs(tt[mask2])) ** self._exp_e_neg)  # electromagnetic
                elif(shower_type == "TAU"):
                    logger.error("Tau showers are not yet implemented")
                    raise NotImplementedError("Tau showers are not yet implemented")
                else:
                    msg = "showers of type {} are not implemented. Use 'HAD', 'EM' or 'TAU'".format(shower_type)
                    logger.error(msg)
                    raise NotImplementedError(msg)
                # Obtain "shape" of Lambda-function from vp at Cherenkov angle
                # xntot = LQ_tot in PRD paper
                F_p[mask] = Acher[mask] * fc / xntot
    #         F_p[~mask] = 1.e-30 * fc / xntot
            F_p[~mask] = 0

            vp[it] = np.trapz(-v * profile_ce_interp2 * F_p / R, z)
            if  0:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 1)
                inte = -v * profile_ce_interp2 * F_p / R
                ax.plot(tt, inte[0], '-')
                ax.plot(tt, inte[1], '-')
                ax.plot(tt, inte[2], '-')
                ax.plot(tt[tmask], inte[0][tmask], 'o')
                ax.plot(tt[tmask], inte[1][tmask], 'o')
                ax.plot(tt[tmask], inte[2][tmask], 'o')
                ax.set_title("{}".format(vp[it]))
                plt.show()

        vp *= factor
        if 0:
            import matplotlib.pyplot as plt
            fig, (ax, ax2) = plt.subplots(1, 2)
            ax.plot(vp)
            print(vp.shape)
            t0 = -np.gradient(vp.T[0]) / dt
            t1 = -np.gradient(vp.T[1]) / dt
            t2 = -np.gradient(vp.T[2]) / dt
            trace2 = -np.diff(vp, axis=0) / dt
    #         print(trace.shape)
            ax2.plot(t0)
            ax2.plot(t1)
            ax2.plot(t2)

            ax2.plot(trace2.T[0], '--')
            ax2.plot(trace2.T[1], '--')
            ax2.plot(trace2.T[2], '--')
            plt.show()
        return vp

    def get_vector_potential(self, energy, theta, N, dt, y=1, ccnc='cc', flavor=12, n_index=1.78, R=1 * units.m,
                             profile_depth=None, profile_ce=None):
        """
        python transcription of original FORTRAN code
        """

        tt = np.arange(0, (N + 1) * dt, dt)
        tt = tt + 0.5 * dt - tt.mean()
    #     tmin = tt.min()
    #     tmax = tt.max()

    #     tmin = -100 * units.ns
    #     tmax = 100 * units.ns

    #     tt = np.arange(tmin, tmax, dt)
    #     tt += 0.5 * dt
        N = len(tt)

        xn = n_index
        cher = np.arccos(1. / n_index)
        beta = 1.

        # calculate antenna position in ARZ reference frame
        # coordinate system is with respect to an origin which is located
        # at the position where the primary particle is injected in the medium. The reference frame
        # is z = along shower axis, and x,y are two arbitray directions perpendicular to z
        # and perpendicular among themselves of course.
        # For instance to place an observer at a distance R and angle theta w.r.t. shower axis in the x,z plane
        # it can be simply done by putting in the input file the numerical values:
        X = np.array([R * np.sin(theta), 0., R * np.cos(theta)])

        def get_dist_shower(X, z):
            """
            Distance from position in shower depth z' to each antenna.
            Denominator in Eq. (22) PRD paper

            Parameters
            ----------
            X: 3dim np. array
                position of antenna in ARZ reference frame
            z: shower depth
            """
            return (X[0] ** 2 + X[1] ** 2 + (X[2] - z) ** 2) ** 0.5

        length = profile_depth / rho
        xnep = intp.interp1d(length, profile_ce, bounds_error=False, fill_value=0)

        # calculate total charged track length
        xntot = np.sum(profile_ce) * (length[1] - length[0])
        # print("{:.5g}".format(xntot))
        # res = int.quad(xnep, length.min(), length.max())
        # print("{:.5g} {:.5g}".format(*res))

        if 0:  # debug plot
            ll = np.linspace(length.min(), length.max(), 10000)
            plt.plot(ll, xnep(ll))
            plt.plot(length, N_e - N_p, 'o')
            plt.show()

        factor = -xmu / (4. * np.pi)

        def xintegrand(z, index, tobs):
            R = get_dist_shower(X, z)
            arg = z - (beta * c * tobs - xn * R)
            u_x = X[0] / R
            u_y = X[1] / R
            u_z = (X[2] - z) / R
            beta_z = 1.
            vperp_x = u_x * u_z * beta_z
            vperp_y = u_y * u_z * beta_z
            vperp_z = -(u_x * u_x + u_y * u_y) * beta_z
            v = np.array([vperp_x, vperp_y, vperp_z])[index]

            return -v * xnep(z) * F_p(arg) / R

        def F_p(arg):
            """
            Function F_p Eq.(15) PRD paper.
            """
            # Factor accompanying the F_p in Eq.(15) in PRD paper
            fc = 4. * np.pi / (xmu * np.sin(cher))
            beta = 1.

            # Note that Acher peaks at tt=0 which corresponds to the observer time.
            # The shift from tobs to tt=0 is done when defining argument
            tt = (-arg / (c * beta))  # Parameterisation of A_Cherenkov with t in ns
            # Cut fit above +/-5 ns
            if (abs(tt) > 5. * units.ns):
                return 1.e-30 * fc / xntot

            # Choose Acher between purely electromagnetic, purely hadronic or mixed shower
            # Eq.(16) PRD paper.
            E_TeV = energy / units.TeV
            em_fraction = self.em_fraction(energy)
            if (tt > 0):
                A_e = self._Af_e * E_TeV * (np.exp(-np.abs(tt) / self._t0_e_pos) +
                                      (1. + self._freq_e_pos * np.abs(tt)) ** self._exp_e_pos)  # electromagnetic
                A_p = self._Af_p * E_TeV * em_fraction * (np.exp(-np.abs(tt) / self._t0_p_pos) +
                                      (1. + self._freq_p_pos * np.abs(tt)) ** self._exp_p_pos)  # hadronic
            else:
                A_e = self._Af_e * E_TeV * (np.exp(-np.abs(tt) / self._t0_e_neg) +
                                      (1. + self._freq_e_neg * np.abs(tt)) ** self._exp_e_neg)  # electromagnetic
                A_p = self._Af_p * E_TeV * em_fraction * (np.exp(-np.abs(tt) / self._t0_p_pos) +
                                      (1. + self._freq_p_neg * np.abs(tt)) ** self._exp_p_neg)  # hadronic

            if(ccnc == 'nc'):
                Acher = y * A_p
            else:
                if(np.abs(flavor) == 12):
                    Acher = (1. - y) * A_e + y * A_p
                else:
                    Acher = 0

            # Obtain "shape" of Lambda-function from vp at Cherenkov angle
            # xntot = LQ_tot in PRD paper
            return Acher * fc / xntot

        vp = np.zeros((N, 3))
        for it, t in enumerate(tt):
            tobs = t + (get_dist_shower(X, 0) / c * xn)
            xmin = length.min()
            xmax = length.max()
            if(X[0] != 0):
                vp[it][0] = int.quad(xintegrand, xmin, xmax, args=(0, tobs))[0]
            if(X[1] != 0):
                vp[it][1] = int.quad(xintegrand, xmin, xmax, args=(1, tobs))[0]
            if(X[2] != 0):
                vp[it][2] = int.quad(xintegrand, xmin, xmax, args=(2, tobs))[0]
        vp *= factor
        return vp


class ARZ_tabulated(object):
    __instance = None

    def __new__(cls, seed=1234, library=None):
        if ARZ_tabulated.__instance is None:
            ARZ_tabulated.__instance = object.__new__(cls, seed, library)
        return ARZ_tabulated.__instance

    def __init__(self, seed=1234, library=None):
        logger.warning("setting seed to {}".format(seed))
        self._random_generator = np.random.RandomState(seed)
        self._random_numbers = {}
        self._version = (1, 1)
        # # load shower library into memory
        if(library is None):
            library = os.path.join(os.path.dirname(__file__), "shower_library/ARZ_library_v{:d}.{:d}.pkl".format(*self._version))
        else:
            if(not os.path.exists(library)):
                logger.error("user specified pulse library {} not found.".format(library))
                raise FileNotFoundError("user specified pulse library {} not found.".format(library))
        self.__check_and_get_library()

        logger.warning("loading pulse library into memory")
        self._library = io_utilities.read_pickle(library)

    def __check_and_get_library(self):
        """
        checks if pulse library exists and is up to date by comparing the sha1sum. If the library does not exist
        or changes on the server, a new library will be downloaded.
        """
        path = os.path.join(os.path.dirname(__file__), "shower_library/ARZ_library_v{:d}.{:d}.pkl".format(*self._version))

        download_file = False
        if(not os.path.exists(path)):
            logger.warning("ARZ library version {} does not exist on the local file system yet. It will be downloaded to {}".format(self._version, path))
            download_file = True

        if(os.path.exists(path)):
            BUF_SIZE = 65536 * 2 ** 4  # lets read stuff in 64kb chunks!
            import hashlib
            import json
            sha1 = hashlib.sha1()
            with open(path, 'rb') as f:
                while True:
                    data = f.read(BUF_SIZE)
                    if not data:
                        break
                    sha1.update(data)

            shower_directory = os.path.join(os.path.dirname(__file__), "shower_library/")
            with open(os.path.join(shower_directory, 'shower_lib_hash.json'), 'r') as fin:
                lib_hashs = json.load(fin)
                if("ARZ_{:d}.{:d}".format(*self._version) in lib_hashs.keys()):
                    if(sha1.hexdigest() != lib_hashs["{:d}.{:d}".format(*self._version)]):
                        logger.warning("pulse library {} has changed on the server. downloading newest version...".format(self._version))
                        download_file = True
                else:
                    logger.warning("no hash sum of {} available, skipping up-to-date check".format(os.path.basename(path)))
        if not download_file:
            return True
        else:
            import requests
            URL = 'http://arianna.ps.uci.edu/~arianna/data/ce_shower_library/ARZ_library_v{:d}.{:d}.pkl'.format(*self._version)

            logger.info("downloading pulse library {} from {}. This can take a while...".format(self._version, URL))
            r = requests.get(URL)
            if (r.status_code != requests.codes.ok):
                logger.error("error in download of antenna model")
                raise IOError("error in download of antenna model")
            with open(path, "wb") as code:
                code.write(r.content)
            logger.info("...download finished.")

    def set_seed(self, seed):
        """
        allow to set a new random seed
        """
        self._random_generator.seed(seed)

    def get_time_trace(self, shower_energy, theta, N, dt, shower_type, n_index, R,
                       same_shower=False, iN=None, output_mode='trace', theta_reference='X0'):
        """
        calculates the electric-field Askaryan pulse from a charge-excess profile

        Parameters
        ----------
        shower_energy: float
            the energy of the shower
        theta: float
            viewing angle, i.e., the angle between shower axis and launch angle of the signal (the ray path)
        N: int
            number of samples in the time domain
        dt: float
            size of one time bin in units of time
        shower_type: string (default "HAD")
            type of shower, either "HAD" (hadronic), "EM" (electromagnetic) or "TAU" (tau lepton induced)
        n_index: float (default 1.78)
            index of refraction where the shower development takes place
        R: float (default 1km)
            observation distance, the signal amplitude will be scaled according to 1/R
        same_shower: bool (default False)
            if False, for each request a new random shower realization is choosen.
            if True, the shower from the last request of the same shower type is used. This is needed to get the Askaryan
            signal for both ray tracing solutions from the same shower.
        iN: int or None (default None)
            specify shower number
        output_mode: string
            * 'trace' (default): return only the electric field trace
            * 'Xmax': return trace and position of xmax in units of length
        theta_reference: string (default: X0)
            * 'X0': viewing angle relativ to start of the shower
            * 'Xmax': viewing angle is relativ to Xmax, internally it will be converted to be relative to X0

        Returns: array of floats
            array of electric-field time trace in 'on-sky' coordinate system eR, eTheta, ePhi
        """
        if not shower_type in self._library.keys():
            raise KeyError("shower type {} not present in library. Available shower types are {}".format(shower_type, *self._library.keys()))

        # determine closes available energy in shower library
        energies = np.array(list(self._library[shower_type].keys()))
        iE = np.argmin(np.abs(energies - shower_energy))
        rescaling_factor = shower_energy / energies[iE]
        logger.info("shower energy of {:.3g}eV requested, closest available energy is {:.3g}eV. The pulse amplitude will be rescaled accordingly by a factor of {:.2f}".format(shower_energy / units.eV, energies[iE] / units.eV, rescaling_factor))
        profiles = self._library[shower_type][energies[iE]]
        N_profiles = len(profiles.keys())

        if(iN is None):
            if(same_shower):
                if(shower_type in self._random_numbers):
                    iN = self._random_numbers[shower_type]
                    logger.info("using previously used shower {}/{}".format(iN, N_profiles))
                else:
                    logger.warning("no previous random number for shower type {} exists. Generating a new random number.".format(shower_type))
                    iN = self._random_generator.randint(N_profiles)
                    self._random_numbers[shower_type] = iN
                    logger.info("picking profile {}/{} randomly".format(iN, N_profiles))
            else:
                iN = self._random_generator.randint(N_profiles)
                self._random_numbers[shower_type] = iN
                logger.info("picking profile {}/{} randomly".format(iN, N_profiles))
        else:
            logger.info("using shower {}/{} as specified by user".format(iN, N_profiles))

        thetas = profiles[iN].keys()
        iT = np.argmin(np.abs(thetas - theta))
        logger.info("selecting theta = {:.2f} ({:.2f} requested)".format(thetas[iT] / units.deg, theta))
        trace = profiles[iT]['trace']
        t0 = profiles[iT]['t0']
        Lmax = profiles[iT]['Lmax']
        trace2 = np.zeros(N)
        tcenter = N // 2 * dt
        tstart = t0 + tcenter
        i0 = np.int(np.round(tstart / dt))
        trace2[i0:(i0 + len(trace))] = trace

        trace2 *= self._library['meta']['R'] / R * rescaling_factor

        if(output_mode == 'Xmax'):
            return trace2, Lmax
        return trace2
