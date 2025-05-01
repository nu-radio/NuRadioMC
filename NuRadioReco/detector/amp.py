import numpy as np
from NuRadioReco.utilities import units
from scipy import interpolate as intp
from matplotlib import pyplot as plt
import os


def get_amp_response(frequencies, amp_name):
    """
    Get measured amplifier responses.
    Very simple script for currently NTU amps.

    """
    # parse input file and convert to default units
    directory = os.path.dirname(os.path.abspath(__file__))
    freqs2, phase, t = np.loadtxt(os.path.join(
        directory, 'amps/{}/LNA_PHASE.CSV'.format(amp_name)), unpack=True, skiprows=3, delimiter=',')
    freqs, logmag, t = np.loadtxt(os.path.join(
        directory, 'amps/{}/LNA_LOGMAG.CSV'.format(amp_name)), unpack=True, skiprows=3, delimiter=',')
    freqs *= units.Hz
    freqs2 *= units.Hz
    phase *= units.deg
    linmag = 10**(logmag / 20.)

    get_phase = intp.interp1d(freqs2, np.unwrap(phase))
    get_linmag = intp.interp1d(freqs, linmag)

    amp_response = np.zeros_like(frequencies, dtype=complex)
    mask = (frequencies > max(freqs.min(), freqs2.min())) & (frequencies < min(freqs.max(), freqs2.max()))
    amp_response[mask] = get_linmag(frequencies[mask]) * np.exp(1j * get_phase(frequencies[mask]))
    return amp_response


if __name__ == "__main__":
    ff = np.linspace(10 * units.MHz, 500 * units.MHz, 100)
    df = ff[1] - ff[0]
    fig, (ax, ax2) = plt.subplots(1, 2)
    response = get_amp_response(ff, "NTU01")
    ax.plot(ff / units.MHz, np.abs(response))
    group_delay = -1. / 2 / np.pi * np.diff(np.unwrap(np.angle(response))) / df
    #     ax2.plot(ff / units.MHz, np.unwrap(np.angle(response1)) / units.deg)
    ax2.plot((ff[1:] + ff[:-1]) * 0.5 / units.MHz, group_delay / units.ns)
    ax.legend()
    ax2.legend()
    fig.tight_layout()
    plt.show()
