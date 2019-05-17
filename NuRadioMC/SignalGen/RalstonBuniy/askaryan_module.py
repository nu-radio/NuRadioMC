from NuRadioMC.SignalGen.RalstonBuniy import create_askaryan
from NuRadioMC.utilities import units, fft
from scipy.optimize import curve_fit
import numpy as np

ICE_DENSITY = 0.9167 # g/cm^2
ICE_RAD_LENGTH = 36.08 # g/cm^2
_strictLowFreqLimit = False

def gauss(x, A, mu, sigma):

    return A * np.exp(-(x-mu)**2/2/sigma**2)

def get_N_AskDepthA_2(E, em=True, lpm=True):
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
        max_x = 200000.0  # maximum depth in g/cm^2
        dx = 1.0  # small enough bin in depth for our purposes.
        x_start = dx  # depth in g/cm^2
        S0 = 0.11842
        X0 = 39.562  # g/cm^2
        l = 113.03  # g/cm^2
        Ec = 0.17006 * units.GeV  # GeV
        Xmax = X0 * np.log(E / Ec)
        x = np.arange(x_start, max_x, dx)
        a = S0 * E / Ec * (Xmax - l) / Xmax * np.exp(Xmax / l - 1)
        b = pow(x / (Xmax - l), Xmax / l)
        c = np.exp(-x / l)
        nx = a * b * c
    # find location of maximum, and charge excess from Fig. 5.9, compare in cm not m.
    n_max = np.argmax(nx)
    if em:
        excess = 0.09 + dx * n_max * ICE_RAD_LENGTH / ICE_DENSITY * 1.0e-4
    else:
        excess = 0.09 + dx * n_max / ICE_DENSITY * 1.0e-4
    Nmax = excess * n_max / 1000.0
    # find depth, which is really the FWHM of this Greisen formula.
    i = np.argwhere((nx / nx[n_max]) > 0.606531)[0][0]
    j = np.argwhere((nx[(i + 1):] / nx[n_max]) < 0.606531)[0]
    _askaryanDepthA = dx * (j - i) / ICE_DENSITY * ICE_RAD_LENGTH / 100.0  # meters

    fit_region_cut = 0.95 # We want to perform a fit for the regions with an excess charge
                         # 10% close to the maximum
    cut_left = np.argwhere((nx[:n_max] / nx[n_max]) > fit_region_cut)[0][0]
    cut_right = np.argwhere((nx[n_max:] / nx[n_max]) < fit_region_cut)[0][0]+n_max
    fit_width = cut_right-cut_left
    max_vicinity = nx[n_max-fit_width:n_max+fit_width]/nx[n_max]
    x_fit = np.arange(0, len(max_vicinity), 1)
    sigma = curve_fit(gauss, x_fit, max_vicinity)[0]
    if em:
        _askaryanDepthA = dx * sigma[2] / ICE_DENSITY * ICE_RAD_LENGTH / 100.0  # meters
    else:
        _askaryanDepthA = dx * sigma[2] / ICE_DENSITY / 100.0  # meters

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
        # Right here, record the reduction in n_max that I don't believe in.
        if _strictLowFreqLimit:
            Nmax = Nmax / (a / _askaryanDepthA)
        _askaryanDepthA = a

    return _askaryanDepthA

def get_time_trace(energy, theta, N, dt, is_em_shower, n, R, LPM=True, a=None):
    if(a is None):
        a = get_N_AskDepthA_2(energy, em=is_em_shower, lpm=True)
    freqs = np.fft.rfftfreq(N, dt)
    eR, eTheta, ePhi = create_askaryan.get_frequency_spectrum(energy, theta, freqs, is_em_shower, n, R, LPM, a)
    ZHS_norm = 2 # ZHS Fourier transform factor
    traceR = np.fft.irfft(eR) / dt / ZHS_norm
    traceTheta = np.fft.irfft(eTheta) / dt / ZHS_norm
    tracePhi = np.fft.irfft(ePhi) / dt / ZHS_norm
    return np.array([traceR, traceTheta, tracePhi])


def get_frequency_spectrum(energy, theta, N, dt, is_em_shower, n, R, LPM=True, a=None):
    eR, eTheta, ePhi = get_time_trace(energy, theta, N, dt, is_em_shower, n, R, LPM, a)
    return np.array([fft.time2freq(eR), fft.time2freq(eTheta), fft.time2freq(ePhi)])
