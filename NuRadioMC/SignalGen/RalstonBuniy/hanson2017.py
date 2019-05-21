import numpy as np
from NuRadioMC.utilities import units, fft
from scipy import constants
from scipy.optimize import curve_fit
import logging
logger = logging.getLogger("hanson2017")
logger.setLevel(logging.DEBUG)


"""
Python transcription of the C++ implementation of https://github.com/918particle/AskaryanModule/tree/simplified/RalstonBuniy
as of 30th Oct. 2018. 
Implementation of J. Hanson and A. Conolly "Complex analysis of Askaryan radiation: A fully analytic treatment including the LPM effect and Cascade Form Factor."
Astropart. Phys. ????
"""
speed_of_light = constants.c * units.m / units.s
# c= 0.29972

_strictLowFreqLimit = True

NORM = 1.0

ICE_DENSITY = 0.9167  * units.g / units.cm**3
ICE_RAD_LENGTH = 36.08 * units.g / units.cm**2


def get_k(ff, n_index):
    return 2 * np.pi * ff / speed_of_light * n_index


def get_eta(k, _askaryanDepthA, _askaryanR, _askaryanTheta):
    return k * _askaryanDepthA**2 / _askaryanR * np.sin(_askaryanTheta)**2


def get_Iff(ff, n_index, _askaryanDepthA, _askaryanR, _askaryanTheta):
    COS_THETA_C = 1. / n_index
    k = get_k(ff, n_index)
    eta = get_eta(k, _askaryanDepthA, _askaryanR, _askaryanTheta)
    re_d = 1 - 3 * eta**2 * np.cos(_askaryanTheta) / np.sin(_askaryanTheta)**2 * \
        (np.cos(_askaryanTheta) - COS_THETA_C) / (1 + eta**2)
    im_d = -eta - 3 * eta**3 * np.cos(_askaryanTheta) / np.sin(_askaryanTheta)**2 * \
        (np.cos(_askaryanTheta) - COS_THETA_C) / (1 + eta**2)
    denom = re_d + 1j * im_d
    re_power = -0.5 * (k * _askaryanDepthA)**2 * (np.cos(_askaryanTheta) - COS_THETA_C)**2 / (1 + eta**2)
    im_power = -eta * 0.5 * (k * _askaryanDepthA)**2 * (np.cos(_askaryanTheta) - COS_THETA_C)**2 / (1 + eta**2)
    power = re_power + 1j * im_power
    return np.exp(power) / denom**0.5


def get_E_omega(ff, E, R, theta, n_index, em=True,
                lpm=True, use_form_factor=True,
                _rho0=1. / (np.sqrt(2.0 * np.pi) * 0.03 * units.m)):
    """
    calculates the frequncy spectrum of an Askaryan pulse 
    
    Parameters
    -----------
    ff: np.array of floats
        array of frequencies
    E: float
        shower energy
    R: float
        distance from vertex to observer
    theta: float
        viewing angle
    n_index: float
        index of refraction at the shower
    em: bool (default True)
        switch between EM and had. showers
    lpm: bool (default True)
        enable/disable LPD effect
    
    """

    _Nmax, _askaryanDepthA = get_N_AskDepthA(E, em, lpm)
    COS_THETA_C = 1. / n_index
    k = get_k(ff, n_index)
    eta = get_eta(k, _askaryanDepthA, R, theta)
    I_FF = get_Iff(ff, n_index, _askaryanDepthA, R, theta)
    nu = speed_of_light * k / (2.0 * np.pi)
    logger.debug("a {}, nmax {}, R {}".format(_askaryanDepthA, _Nmax, R))
    norm = 2.52e-7 * 1e3 * _askaryanDepthA * _Nmax * nu / R / NORM  # the additional *1e3 comes from putting all the units in the constant of Eq. (10). The left side of the equation is in MHz, whereas the right side is in GHz
    # Kinematic factor, psi...checked JCH March 8th, 2016...fixed missing sin(theta)
    psi = np.sin(theta) * np.sin(k * R) + 1j * (-np.sin(theta) * np.cos(k * R))
    # radial component (imaginary part is zero)...checked JCH March 8th, 2016
    rComp_num = -(np.cos(theta) - COS_THETA_C) / np.sin(theta)
    rComp = I_FF * norm * psi * rComp_num
    # theta component (has real and imaginary parts)...checked JCH March 8th, 2016
    thetaComp_num = 1 + eta**2 / (1 + eta)**2 * COS_THETA_C / np.sin(theta)**2 * (np.cos(theta) - COS_THETA_C) + \
        1j * (-eta / (1 + eta)**2 * COS_THETA_C / np.sin(theta)**2 * (np.cos(theta) - COS_THETA_C))
    thetaComp = I_FF * norm * psi * thetaComp_num
    logger.debug("IFF[0] {:.2g}, norm {:.2g}, psi[0] {:.2g}, thetaComp_num {:.2g}".format(I_FF[1], norm[1], psi[1], thetaComp_num[1]))

    if use_form_factor:
        a = k / _rho0
        b = np.sin(theta) / (2.0 * np.pi)**0.5
        atten = (1 + a**2 * b**2)**-1.5
        rComp *= atten
        thetaComp *= atten

#     rComp *= units.V / units.m / units.MHz
#     thetaComp *= units.V / units.m / units.MHz
    return rComp, thetaComp

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x-mu)**2/2/sigma**2)


def get_N_AskDepthA(E, em=True, lpm=True):
    if em:
        E_CRIT = 0.073 * units.GeV  # GeV
        max_x = 5000.0  # maximum number of radiation lengths
        dx = 0.01  # small enough bin in depth for our purposes.
        x_start = 0.01  # starting radiation length
        # Greissen EM shower profile from Energy E in GeV.

        x = np.arange(x_start, max_x, dx)
        a = 0.31 / (np.log(E / E_CRIT))**0.5
        b = x
        c = 1.5 * x
        d = np.log((3 * x) / (x + 2 * np.log(E / E_CRIT)))
        nx = a * np.exp(b - c * d)

    else:  # hadronic shower profile
        # Gaisser-Hillas hadronic shower parameterization
        max_x = 200000.0 * units.g /units.cm**2 # maximum depth in g/cm^2
        dx = 1.0  * units.g /units.cm**2 # small enough bin in depth for our purposes.
        x_start = dx  # depth in g/cm^2
        S0 = 0.11842
        X0 = 39.562 * units.g /units.cm**2 # g/cm^2
        l = 113.03  * units.g /units.cm**2# g/cm^2
        Ec = 0.17006 * units.GeV  # GeV
        Xmax = X0 * np.log(E / Ec)
        x = np.arange(x_start, max_x, dx)
        a = S0 * E / Ec * (Xmax - l) / Xmax * np.exp(Xmax / l - 1)
        b = pow(x / (Xmax - l), Xmax / l)
        c = np.exp(-x / l)
        nx = a * b * c
    # find location of maximum, and charge excess from Fig. 5.9, compare in cm not m.
    n_max_position = np.argmax(nx)
    n_max = np.max(nx)
    if em:
        excess = 0.09 + dx * n_max_position * ICE_RAD_LENGTH / ICE_DENSITY / 100.
    else:
        excess = 0.09 + dx * n_max_position / ICE_DENSITY * 1.0e-2
    Nmax = excess * n_max / 1000.0
    logger.debug("Nmax {}, excess {}, n_max {}".format(Nmax, excess, n_max))
    

    fit_region_cut = 0.95 # We want to perform a fit for the regions with an excess charge
                          # 10% close to the maximum
    cut_left = np.argwhere((nx[:n_max_position] / nx[n_max_position]) > fit_region_cut)[0][0]
    cut_right = np.argwhere((nx[n_max_position:] / nx[n_max_position]) < fit_region_cut)[0][0]+n_max_position
    fit_width = cut_right-cut_left
    max_vicinity = nx[n_max_position-fit_width:n_max_position+fit_width]/nx[n_max_position]
    x_fit = np.arange(0, len(max_vicinity), 1)
    sigma = curve_fit(gauss, x_fit, max_vicinity)[0]
    if em:
        _askaryanDepthA = dx * sigma[2] / ICE_DENSITY * ICE_RAD_LENGTH 
    else:
        _askaryanDepthA = dx * sigma[2] / ICE_DENSITY
    logger.debug("a (before LPM = {}".format(_askaryanDepthA))

    if(em and lpm):
        p1 = -2.8564e2
        p2 = 7.8140e1
        p3 = -8.3893
        p4 = 4.4175e-1
        p5 = -1.1382e-2
        p6 = 1.1493e-4
        e = np.log10(E/units.eV)  # log_10 of Energy in eV
        log10_shower_depth = p1 + p2 * e + p3 * e**2 + p4 * e**3 + p5 * e**4 + p6 * e**5
        a = 10.0**log10_shower_depth
        # Right here, record the reduction in n_max_position that I don't believe in.
        if _strictLowFreqLimit:
            logger.debug("strict_lowfeq  Nmax = {:.2g}, a= {} priora = {}".format(Nmax, a, _askaryanDepthA/units.m, Nmax))
            Nmax = Nmax / (a / _askaryanDepthA)
        _askaryanDepthA = a
    logger.debug("a = {:.2f}m, Nmax = {}".format(_askaryanDepthA/units.m, Nmax))
    return Nmax, _askaryanDepthA

def get_time_trace(energy, theta, N, dt, is_em_shower, n_index, R, LPM=True, a=None):
    freqs = np.fft.rfftfreq(N, dt)
    eR, eTheta = get_E_omega(freqs, energy, R, theta, n_index, is_em_shower, LPM)
    traceR = np.fft.irfft(eR) / dt
    traceTheta = np.fft.irfft(eTheta) / dt
    return np.array([traceR, traceTheta, np.zeros_like(traceTheta)])


def get_frequency_spectrum(energy, theta, N, dt, is_em_shower, n, R, LPM=True, a=None):
    eR, eTheta, ePhi = get_time_trace(energy, theta, N, dt, is_em_shower, n, R, LPM, a)
    return np.array([fft.time2freq(eR), fft.time2freq(eTheta), fft.time2freq(ePhi)])

