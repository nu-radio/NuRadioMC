import numpy as np
from scipy.signal.windows import hann
from scipy import constants
from NuRadioReco.utilities import units


def half_hann_window(length, half_percent=None, hann_window_length=None):
    """
    Produce a half-Hann window. This is the Hann window from SciPY with ones inserted in the middle to make the window
    `length` long. Note that this is different from a Hamming window.

    Parameters
    ----------
    length : int
        The desired total length of the window
    half_percent : float, default=None
        The percentage of `length` at the beginning **and** end that should correspond to half of the Hann window
    hann_window_length : int, default=None
        The length of the half the Hann window. If `half_percent` is set, this value will be overwritten by it.
    """
    if half_percent is not None:
        hann_window_length = int(length * half_percent)
    elif hann_window_length is None:
        raise ValueError("Either half_percent or half_window_length should be set!")
    hann_window = hann(2 * hann_window_length)

    half_hann_widow = np.ones(length, dtype=np.double)
    half_hann_widow[:hann_window_length] = hann_window[:hann_window_length]
    half_hann_widow[-hann_window_length:] = hann_window[hann_window_length:]

    return half_hann_widow


def calculate_filtered_thermal_noise_amplitude(temperature, frequencies, filter, resistance=50):
    """
    Calculate the amplitude of the filtered (amplified!) thermal noise for a given temperature, bandwidth and resistance.

    Calculation of Vrms. For details see from elog:1566 and https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
    (last two Eqs. in "noise voltage and power" section) or our wiki
    https://nu-radio.github.io/NuRadioMC/NuRadioMC/pages/HDF5_structure.html.

    Parameters
    ----------
    temperature : float
        The temperature in Kelvin
    bandwidth : float
        The bandwidth in Hz
    resistance : float (Default: 50)
        The resistance in Ohm
    """

    # Bandwidth, i.e., \Delta f in equation
    integrated_channel_response = np.trapz(np.abs(filter) ** 2, frequencies)

    vrms = (temperature * resistance * constants.k * integrated_channel_response / units.Hz) ** 0.5
    return vrms
