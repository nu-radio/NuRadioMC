from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioMC.utilities import units

"""

Analytic parametrizations of the radio pulse produced by an in-ice particle shower.

Generic functions to provide the frequency spectrum and the pulse in the time domain
are defined. All models/parametrizations should be added to each of these functions,
such that different parametrizations can be exchanged by just modifying the 'model'
argument of the respective function.

The following models are implemented
 * Alvarez2000, 10.1103/PhysRevD.62.063001

"""


def get_parametrizations():
    """ returns a list of all implemented parametrizations """
    return ['Alvarez2000']


def get_frequency_spectrum(Enu, f, y=0.2,
                           nu_flavor='e', currentint='charged',
                           view_angle=np.arccos(1 / 1.78),
                           cherenkov_angle=np.arccos(1 / 1.78),
                           model='Alvarez2000'):
    """
    returns the magnitude of the frequency spectrum of the neutrino radio signal

    Parameters
    ----------
    Enu : float
        energy of the neutrino
    f : float or array
        frequency
    y: float
        inelasticity y is defined at the fraction of the neutrino energy that goes
        into the hadronic part.
    nu_flafor: string one of ['e', 'mu', 'tau']
        the neutrino flavor. default 'e' (electron neutrino) This information
        is neccessary to determine how much energy is going into hadronic and
        electromagnetic showers
    currentint: string one of ['charged', 'neutral']
        charged or current interaction
    view_angle: float or array
        viewangle: angle between shower axis (neutrino direction) and the line
        of sight between interaction and detector, default is the cherenkov_angle
    cherenkov_angle: float or array
        the cherenkov angle, default is 55.8deg = arccos(1 / 1.78)

    Returns
    -------
    E: float or array
        the amplitude for the given frequency

    """
    if(model == 'Alvarez2000'):
        fem = 0  # electrogmatnetic fraction
        fhad = 0  # hadroninc fraction
        if(currentint == 'neutral'):
            fhad = y
        else:
            if(nu_flavor == 'e'):
                fem = (1 - y)
                fhad = y
            elif(nu_flavor == 'mu'):
                fhad = y
            elif(nu_flavor == 'tau'):
                fhad = y

        Elpm = 2e15 * units.eV
        dThetaEM = np.deg2rad(2.7) * 500 * units.MHz / f * (Elpm / (0.14 * Enu + Elpm)) ** 0.3

        epsilon = np.log10(Enu / units.TeV)
        dThetaHad = 0
        if (epsilon >= 0 and epsilon <= 2):
            dThetaHad = 500 * units.MHz / f * (2.07 - 0.33 * epsilon + 7.5e-2 * epsilon ** 2)
        elif (epsilon > 2 and epsilon <= 5):
            dThetaHad = 500 * units.MHz / f * (1.74 - 1.21e-2 * epsilon)
        elif(epsilon > 5 and epsilon <= 7):
            dThetaHad = 500 * units.MHz / f * (4.23 - 0.785 * epsilon + 5.5e-2 * epsilon ** 2)
        elif(epsilon > 7):
            dThetaHad = 500 * units.MHz / f * (4.23 - 0.785 * 7 + 5.5e-2 * 7 ** 2) * (1 + (epsilon - 7) * 0.075)

        f0 = 1.15 * units.GHz
        E = 2.53e-7 * Enu / units.TeV * f / f0 / (1 + (f / f0) ** 1.44)
        E *= units.V / units.m / units.MHz
        E *= np.sin(view_angle) / np.sin(cherenkov_angle)

        res = fem * E * np.exp(-np.log(2) * ((view_angle - cherenkov_angle) / dThetaEM) ** 2) + \
            fhad * E * np.exp(-np.log(2) * ((view_angle - cherenkov_angle) / dThetaHad) ** 2)

        return res


def get_frequency_spectrum_phases(Enu, f, y=0.2,
                                  nu_flavor='e', currentint='charged',
                                  view_angle=np.arccos(1 / 1.78),
                                  cherenkov_angle=np.arccos(1 / 1.78),
                                  model='Alvarez2000'):
    """
    returns the phases of the frequency spectrum of the neutrino radio signal

    The 'Alvarez2000' model does not provide a parametrization of the phases, therefore
    a constant phase is assumed.
    Parameters
    ----------
    Enu : float
        energy of the neutrino
    f : float or array
        frequency
    y: float
        inelasticity y is defined at the fraction of the neutrino energy that goes
        into the hadronic part.
    nu_flafor: string one of ['e', 'mu', 'tau']
        the neutrino flavor. default 'e' (electron neutrino) This information
        is neccessary to determine how much energy is going into hadronic and
        electromagnetic showers
    currentint: string one of ['charged', 'neutral']
        charged or current interaction
    view_angle: float or array
        viewangle: angle between shower axis (neutrino direction) and the line
        of sight between interaction and detector, default is the cherenkov_angle
    cherenkov_angle: float or array
        the cherenkov angle, default is 55.8deg = arccos(1 / 1.78)

    Returns
    -------
    phases: float or array
        the phases for the given frequency

    """
    if(model == 'Alvarez2000'):
        return np.ones_like(f)


def get_time_trace(Enu, y=0.2,
                   nu_flavor='e', currentint='charged',
                   view_angle=np.arccos(1 / 1.78),
                   cherenkov_angle=np.arccos(1 / 1.78),
                   model='Alvarez2000',
                   flow=100 * units.MHz, fhigh=500 * units.MHz,
                   df=10 * units.MHz):
    """
    returns the time trace of the neutrino radio signal

    The 'Alvarez2000' model does not provide a parametrization of the phases, therefore
    a constant phase is assumed.

    Parameters
    ----------
    Enu : float
        energy of the neutrino
    y: float
        inelasticity y is defined at the fraction of the neutrino energy that goes
        into the hadronic part.
    nu_flafor: string one of ['e', 'mu', 'tau']
        the neutrino flavor. default 'e' (electron neutrino) This information
        is neccessary to determine how much energy is going into hadronic and
        electromagnetic showers
    currentint: string one of ['charged', 'neutral']
        charged or current interaction
    view_angle: float or array
        viewangle: angle between shower axis (neutrino direction) and the line
        of sight between interaction and detector, default is the cherenkov_angle
    cherenkov_angle: float or array
        the cherenkov angle, default is 55.8deg = arccos(1 / 1.78)
    flow : float
        lower frequency cutoff
    fhigh : float
        higher frequency cutoff
    df: float
        frequency resolution


    Returns
    -------
    times: array
        time bins of trace
    trace: array
        the neutrino pulse in the time domain

    """

    ff = np.arange(10 * units.Hz, fhigh + df, df)
#     print(ff)
    mag = get_frequency_spectrum(Enu, ff, y, nu_flavor, currentint, view_angle, cherenkov_angle, model)
    mag[ff < flow] = 0
    phases = get_frequency_spectrum_phases(Enu, ff, y, nu_flavor, currentint, view_angle, cherenkov_angle, model)
    spec = mag * np.exp(1j * phases)  # complex spectrum
    # do inverse fft to obtain time trace
    trace = np.fft.irfft(spec, norm='ortho') / 2 ** 0.5  # watch the proper normalization to keep the power conserved

    df = ff[1] - ff[0]
    sampling_rate = len(spec) * df
    tt = np.linspace(0, (len(trace) + 1) / sampling_rate * 0.5, len(trace))
#     print(np.fft.rfftfreq(len(trace), tt[1] - tt[0]))
    return tt, np.roll(trace, len(trace) // 2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ff = np.linspace(100, 5000, 500) * units.MHz
    mag = get_frequency_spectrum(1e18, ff)
    tt, trace = get_time_trace(1e18, flow=0, fhigh=20 * units.GHz)

    fig, (ax, ax2) = plt.subplots(1, 2)
    ax.plot(ff, mag)
    ax.plot(np.fft.rfftfreq(len(trace), tt[1] - tt[0]),
            np.abs(np.fft.rfft(trace, norm='ortho') * 2 ** 0.5))
    ax2.plot(tt, np.roll(trace, len(trace) // 2))
    plt.tight_layout()
    plt.show()
