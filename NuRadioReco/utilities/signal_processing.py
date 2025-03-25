from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioReco.framework.sim_station import SimStation

from scipy.signal.windows import hann
from scipy import constants
import numpy as np

import logging
logger = logging.getLogger("NuRadioReco.utilities.signal_processing")


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


def add_cable_delay(station, det, sim_to_data=None, trigger=False, logger=None):
    """
    Add or subtract cable delay by modifying the ``trace_start_time``.

    Parameters
    ----------
    station: Station
        The station to add the cable delay to.

    det: Detector
        The detector description

    trigger: bool
        If True, take the time delay from the trigger channel response.
        Only possible if ``det`` is of type `rnog_detector.Detector`. (Default: False)

    logger: logging.Logger, default=None
        If set, use ``logger.debug(..)`` to log the cable delay.

    See Also
    --------
    NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay : module that automatically applies / corrects for cable delays.
    """
    assert sim_to_data is not None, "``sim_to_data`` is None, please specify."

    add_or_subtract = 1 if sim_to_data else -1
    msg = "Add" if sim_to_data else "Subtract"

    if trigger and not isinstance(det, detector.rnog_detector.Detector):
        raise ValueError("Simulating extra trigger channels is only possible with the `rnog_detector.Detector` class.")

    for channel in station.iter_channels():

        if trigger:
            if not channel.has_extra_trigger_channel():
                continue

            channel = channel.get_trigger_channel()
            cable_delay = det.get_cable_delay(station.get_id(), channel.get_id(), trigger=True)

        else:
            # Only the RNOG detector has the argument `trigger`. Default is false
            cable_delay = det.get_cable_delay(station.get_id(), channel.get_id())

        if logger is not None:
            logger.debug(f"{msg} {cable_delay / units.ns:.2f}ns "
                        f"of cable delay to channel {channel.get_id()}")

        channel.add_trace_start_time(add_or_subtract * cable_delay)


def calculate_vrms_from_temperature(temperature, bandwidth=None, response=None, impedance=50 * units.ohm, freqs=None):
    """ Helper function to calculate the noise vrms from a given noise temperature and bandwidth.

    For details see https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
    (sec. "Maximum transfer of noise power") or our wiki
    https://nu-radio.github.io/NuRadioMC/NuRadioMC/pages/HDF5_structure.html

    Parameters
    ----------
    temperature: float
        The noise temperature of the channel in Kelvin
    bandwidth: float or tuple of 2 floats (list of 2 floats) (default: None)
        If single float, this argument is interpreted as the effective bandwidth. If tuple, the argument is
        interpreted as the lower and upper frequency of the bandwidth. Can be `None` if `response` is specified.
    response: `NuRadioReco.detector.response.Response` (default: None)
        If not None, the response of the channel is taken into account to calculate the noise vrms.
    impedance: float (default: 50)
        Electrical impedance of the channel in Ohm.
    freqs: array_like (default: None -> np.arange(0, 2500, 0.1) * units.MHz)
        Frequencies at which the response is evaluated. Only used if `response` is not None.

    Returns
    -------
    vrms_per_channel: float
        The vrms of the channel
    """
    if bandwidth is None and response is None:
        raise ValueError("Please specify bandwidth or response")

    if impedance > 1000 * units.ohm:
        logger.warning(f"Impedance is {impedance / units.ohm:.2f} Ohm, did you forget to specify the unit?")

    # (effective) bandwidth, i.e., \Delta f in equation
    if response is None:
        if not isinstance(bandwidth, (float, int)):
            bandwidth = bandwidth[1] - bandwidth[0]
    else:
        freqs = freqs or np.arange(0, 2500, 0.1) * units.MHz
        bandwidth = np.trapz(np.abs(response(freqs)) ** 2, freqs)

    return (temperature * impedance * bandwidth * constants.k * units.joule / units.kelvin) ** 0.5
