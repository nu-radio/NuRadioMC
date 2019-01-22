#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from NuRadioReco.utilities import units
from scipy import interpolate as intp
from scipy import integrate as int
from scipy import constants
from matplotlib import pyplot as plt
from radiotools import coordinatesystems as cstrafo
import os
import pickle
import logging
logger = logging.getLogger("SignalGen.ARZ")

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

# # load shower library into memory
with open(os.path.join(os.path.dirname(__file__), "shower_library/library_v1.pkl")) as fin:
    library = pickle.load(fin)


def get_time_trace(shower_energy, theta, N, dt, shower_type, n_index, R, interp_factor=1.):
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
        
    Returns: array of floats
        array of electric-field time trace in 'on-sky' coordinate system eR, eTheta, ePhi
    """
    if not shower_type in library.keys():
        raise KeyError("shower type {} not present in library. Available shower types are {}".format(shower_type, *library.keys()))

    # determine closes available energy in shower library
    energies = np.array(library[shower_type].keys())
    iE = np.argmin(np.abs(energies - shower_energy))
    rescaling_factor = shower_energy / energies[iE]
    logger.info("shower energy of {:.3g}eV requested, closest available energy is {:.3g}eV. The amplitude of the \
                    charge-excess profile will be rescaled accordingly by a factor of \
                    {:.2f}".format(shower_energy/units.eV, energies[iE]/units.eV, rescaling_factor))
    profiles = library[shower_type][energies[iE]]
    N_profiles = len(profiles['charge_excess'])
    iN = np.random.randint(N_profiles)
    logger.debug("picking profile {}/{} randomly".format(iN, N_profiles))
    profile_depth = profiles['depth']
    profile_ce = profiles['charge_excess'][iN] * rescaling_factor
    vp = get_vector_potential_fast(shower_energy, theta, N, dt, profile_depth, profile_ce, 
                                   shower_type, n_index, R, interp_factor)
    trace = -np.diff(vp, axis=0) / dt
    
    cs = cstrafo.cstrafo(zenith=theta, azimuth=0)
    trace_onsky = cs.transform_from_ground_to_onsky(trace.T)
    return trace_onsky


def get_vector_potential(energy, theta, N, dt, y=1, ccnc='cc', flavor=12, n_index=1.78, R=1 * units.m,
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
        # Refit of ZHAireS results => factor 0.88 in Af_e
        Af_e = -4.5e-14 * 0.88 * units.V * units.s
        Af_p = -3.2e-14 * units.V * units.s  # V s
        E_TeV = energy / units.TeV
        if (tt > 0):
            A_e = Af_e * E_TeV * (np.exp(-np.abs(tt) / (0.057 * units.ns)) + 
                                  (1. + 2.87 / units.ns * np.abs(tt)) ** (-3.00))  # electromagnetic
            A_p = Af_p * E_TeV * (np.exp(-np.abs(tt) / (0.065 * units.ns)) + 
                                  (1. + 3.00 / units.ns * np.abs(tt)) ** (-2.65))  # hadronic
        else:
            A_e = Af_e * E_TeV * (np.exp(-np.abs(tt) / (0.030 * units.ns)) + 
                                  (1. + 3.05 / units.ns * np.abs(tt)) ** (-3.50))  # electromagnetic
            A_p = Af_p * E_TeV * (np.exp(-np.abs(tt) / (0.043 * units.ns)) + 
                                  (1. + 2.92 / units.ns * np.abs(tt)) ** (-3.21))  # hadronic

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


def get_vector_potential_fast(shower_energy, theta, N, dt, profile_depth, profile_ce,
                              shower_type="HAD", n_index=1.78, R=1 * units.m,
                              interp_factor=10):
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
    R: float (default 1km)
        observation distance, the signal amplitude will be scaled according to 1/R
    interp_factor: int (default 10)
        interpolation factor of charge-excess profile. Results in a more precise numerical integration which might be beneficial 
        for small vertex distances but also slows down the calculation proportional to the interpolation factor. 
    """

    tt = np.arange(0, (N + 1) * dt, dt)
    tt = tt + 0.5 * dt - tt.mean()
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

    profile_dense = np.linspace(min(profile_depth), max(profile_depth), interp_factor * len(profile_depth))
    length = profile_dense / rho
    profile_ce_interp = np.interp(profile_dense, profile_depth, profile_ce)
    # calculate total charged track length
    xntot = np.sum(profile_ce_interp) * (length[1] - length[0])
    factor = -xmu / (4. * np.pi)
    fc = 4. * np.pi / (xmu * np.sin(cher))

    vp = np.zeros((N, 3))
    for it, t in enumerate(tt):
        tobs = t + (get_dist_shower(X, 0) / c * xn)
        z = length

        R = get_dist_shower(X, z)
        arg = z - (beta * c * tobs - xn * R)
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

        # Note that Acher peaks at tt=0 which corresponds to the observer time.
        # The shift from tobs to tt=0 is done when defining argument
        tt = (-arg / (c * beta))  # Parameterisation of A_Cherenkov with t in ns
        # Cut fit above +/-5 ns
        mask = abs(tt) < 5. * units.ns

        # Choose Acher between purely electromagnetic, purely hadronic or mixed shower
        # Eq.(16) PRD paper.
        # Refit of ZHAireS results => factor 0.88 in Af_e
        Af_e = -4.5e-14 * 0.88 * units.V * units.s
        Af_p = -3.2e-14 * units.V * units.s  # V s
        E_TeV = shower_energy / units.TeV
        Acher = np.zeros_like(tt)
        F_p = np.zeros_like(tt)
        if(np.sum(mask)):
            if(shower_type == "HAD"):
                mask2 = tt > 0 & mask
                if(np.sum(mask2)):
                    Acher[mask2] = Af_p * E_TeV * (np.exp(-np.abs(tt[mask2]) / (0.065 * units.ns)) + 
                                          (1. + 3.00 / units.ns * np.abs(tt[mask2])) ** (-2.65))  # hadronic
                mask2 = tt <= 0 & mask
                if(np.sum(mask2)):
                    Acher[mask2] = Af_p * E_TeV * (np.exp(-np.abs(tt[mask2]) / (0.043 * units.ns)) + 
                                          (1. + 2.92 / units.ns * np.abs(tt[mask2])) ** (-3.21))  # hadronic
            elif(shower_type == "EM"):
                mask2 = tt > 0 & mask
                if(np.sum(mask2)):
                    Acher[mask2] = Af_e * E_TeV * (np.exp(-np.abs(tt[mask2]) / (0.057 * units.ns)) + 
                                          (1. + 2.87 / units.ns * np.abs(tt[mask2])) ** (-3.00))  # electromagnetic
                mask2 = tt <= 0 & mask
                if(np.sum(mask2)):
                    Acher[mask2] = Af_e * E_TeV * (np.exp(-np.abs(tt[mask2]) / (0.030 * units.ns)) + 
                                          (1. + 3.05 / units.ns * np.abs(tt[mask2])) ** (-3.50))  # electromagnetic
            elif(shower_type == "TAU"):
                logger.error("Tau showers are not yet implemented")
                raise NotImplementedError("Tau showers are not yet implemented")
            else:
                msg ="showers of type {} are not implemented. Use 'HAD', 'EM' or 'TAU'".format(shower_type)
                logger.error(msg)
                raise NotImplementedError(msg)
            # Obtain "shape" of Lambda-function from vp at Cherenkov angle
            # xntot = LQ_tot in PRD paper
            F_p[mask] = Acher[mask] * fc / xntot
        F_p[~mask] = 1.e-30 * fc / xntot

        vp[it] = np.trapz(-v * profile_ce_interp * F_p / R, z)

    vp *= factor
    return vp


if __name__ == "__main__":
    energy = 1.e6 * units.TeV
    theta = 55 * units.deg
    R = 1 * units.km
    N = 512
    dt = 0.5 * units.ns
    n_index = 1.78
    y = 0.60146725
    ccnc = 'cc'
    flavor = 12  # e = 12, mu = 14, tau = 16

    cdir = os.path.dirname(__file__)
    bins, depth_e, N_e = np.loadtxt(os.path.join(cdir, "shower_library/nue_1EeV_CC_1_s0001.t1005"), unpack=True)
    bins, depth_p, N_p = np.loadtxt(os.path.join(cdir, "shower_library/nue_1EeV_CC_1_s0001.t1006"), unpack=True)
    depth_e *= units.g / units.cm ** 2
    depth_p *= units.g / units.cm ** 2
    depth_e -= 1000 * units.g / units.cm ** 2  # all simulations have an artificial offset of 1000 g/cm^2
    depth_p -= 1000 * units.g / units.cm ** 2
    # sanity check if files electron and positron profiles are compatible
    if (not np.all(depth_e == depth_p)):
        raise ImportError("electron and positron profile have different depths")

    vp = get_vector_potential(energy, theta, N, dt, y, ccnc, flavor, n_index, R, profile_depth=depth_e, profile_ce=(N_e - N_p))
    vp2 = get_vector_potential(energy, theta, N, dt, y, "EM", n_index, R, profile_depth=depth_e, profile_ce=(N_e - N_p))

    # generate time array
    tt = np.arange(0, (N + 1) * dt, dt)
    tt = tt + 0.5 * dt - tt.mean()

    t, Ax, Ay, Az = np.loadtxt("fortran_reference.dat", unpack=True)
    fig, ax = plt.subplots(1, 1)
    ax.plot(tt, vp[:, 0] / units.V / units.s)
    ax.plot(tt, vp[:, 1] / units.V / units.s)
    ax.plot(tt, vp[:, 2] / units.V / units.s)
    ax.plot(t, Ax, "C0--")
    ax.plot(t, Az, "C2--")
    ax.set_xlim(-2, 2)

    ax.set_xlabel("time [ns]")
    ax.set_ylabel("vector potential")

    mask = np.array([x in t for x in tt])
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, vp[:, 0][mask] / units.V / units.s / Ax)
    ax.plot(t, vp[:, 2][mask] / units.V / units.s / Az)
    ax.set_xlim(-2, 2)
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("python/fortran implementation")
    ax.set_ylim(0.8, 1.2)

    trace = get_time_trace(energy, theta, N, dt, y, ccnc, flavor, n_index, R)
    tt = np.arange(0, dt * N, dt)
    fig, ax = plt.subplots(1, 1)
    ax.plot(tt, trace[:, 0])
    ax.plot(tt, trace[:, 1])
    ax.plot(tt, trace[:, 2])
    fig.tight_layout()
    plt.show()
