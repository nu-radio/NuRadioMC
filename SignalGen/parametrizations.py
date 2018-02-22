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


def get_frequency_spectrum(Eem, f, dTheta=0, model='Alvarez2000'):
    """
    returns the magnitude of the frequency spectrum of the neutrino radio signal

    Parameters
    ----------
    Eem : float or array
        energy of the electromagnetic shower
    f : float or array
        frequency

    Returns
    -------
    E: float or array
        the amplitude for the given frequency

    """
    if(model == 'Alvarez2000'):
        f0 = 1.15 * units.GHz
        E = 2.53e-7 * Eem / units.TeV * f / f0 / (1 + (f / f0) ** 1.44)
        E *= units.V / units.m / units.MHz

        return E


def get_frequency_spectrum_phases(Eem, f, dTheta=0, model='Alvarez2000'):
    """
    returns the phases of the frequency spectrum of the neutrino radio signal

    The 'Alvarez2000' model does not provide a parametrization of the phases, therefore
    a constant phase is assumed.

    Parameters
    ----------
    Eem : float or array
        energy of the electromagnetic shower
    f : float or array
        frequency

    Returns
    -------
    phases: float or array
        the phases for the given frequency

    """
    if(model == 'Alvarez2000'):
        return np.ones_like(f)


def get_time_trace(Eem, flow=100 * units.MHz, fhigh=500 * units.MHz,
                   df=10 * units.MHz, dTheta=0,
                   model='Alvarez2000'):
    """
    returns the time trace of the neutrino radio signal

    The 'Alvarez2000' model does not provide a parametrization of the phases, therefore
    a constant phase is assumed.

    Parameters
    ----------
    Eem : float or array
        energy of the electromagnetic shower
    flow : float
        lower frequency cutoff
    fhigh : float
        higher frequency cutoff
    df: float
        frequency resolution
    dTheta: float



    Returns
    -------
    times: array
        time bins of trace
    trace: array
        the neutrino pulse in the time domain

    """

    ff = np.arange(0, fhigh + df, df)
#     print(ff)
    mag = get_frequency_spectrum(Eem, ff, dTheta, model)
    mag[ff < flow] = 0
    phases = get_frequency_spectrum_phases(Eem, ff, dTheta, model)
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
