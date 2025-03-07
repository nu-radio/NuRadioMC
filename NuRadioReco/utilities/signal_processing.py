from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioReco.framework.sim_station import SimStation

from scipy.signal.windows import hann
import numpy as np


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
