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
    Add or subtract cable delay by modifying the `trace_start_time`.

    Parameters
    ----------
    station: Station
        The station to add the cable delay to.

    det: Detector
        The detector description

    trigger: bool
        If True, take the time delay from the trigger channel response.
        Only possible if `det` is of type `rnog_detector.Detector`. (Default: False)

    logger: logging.Logger, default=None
        If set, use `logger.debug(..)` to log the cable delay.
    """
    assert sim_to_data is not None, "`sim_to_data` is None, please specify."

    add_or_subtract = 1 if sim_to_data else -1
    msg = "Add" if sim_to_data else "Subtract"

    for channel in station.iter_channels():
        cable_delay = det.get_cable_delay(station.get_id(), channel.get_id())
        if logger is not None:
            logger.debug(f"{msg} {cable_delay / units.ns:.2f}ns "
                        f"of cable delay to channel {channel.get_id()}")

        channel.add_trace_start_time(add_or_subtract * cable_delay)


def add_cable_delay_by_rolling(station, det, trigger=False, logger=None):
    """
    Add cable delay to a channel.

    All channels will continue to have the same trace start time. Pulses are rolled
    to reflect the cable delay. All rolled, and thus unphysical, samples are cut.
    This is only possible if the traces initally were sufficiently long. I it verified
    whether the trace start time of the channel is larger than that of the electric fields.

    Parameters
    ----------
    station: Station
        The station to add the cable delay to.

    det: Detector
        The detector description

    trigger: bool
        If True, take the time delay from the trigger channel response.
        Only possible if `det` is of type `rnog_detector.Detector`. (Default: False)

    logger: logging.Logger, default=None
        If set, use `logger.debug(..)` to log the cable delay.
    """
    if isinstance(station, SimStation):
        add_cable_delay(station, det, sim_to_data=True, trigger=trigger, logger=logger)
        return

    if trigger and not isinstance(det, detector.rnog_detector.Detector):
        raise ValueError("Simulating extra trigger channels is only possible with the `rnog_detector.Detector` class.")

    # This function assumes that the trace_start_time of all channels is the station is the same

    cable_delays = []
    for channel in station.iter_channels():
        if trigger:
            if not channel.has_extra_trigger_channel():
                continue

            cable_delays.append(
                det.get_cable_delay(station.get_id(), channel.get_id(), trigger=True))

        else:
            # Only the RNOG detector has the argument `trigger`. Default is false
            cable_delays.append(det.get_cable_delay(station.get_id(), channel.get_id()))

    rolled_samples = []
    for channel, delta_time in zip(station.iter_channels(), cable_delays):
        if trigger:
            if channel.has_extra_trigger_channel():
                channel = channel.get_trigger_channel()
            else:
                continue

        if logger is not None:
            logger.debug(f"Shift channel {channel.get_id()} by {delta_time / units.ns:.2f}ns")

        if delta_time:
            roll_samples = int(delta_time * channel.get_sampling_rate())
            # Keep the trace length even
            if roll_samples % 2 != 0:
                roll_samples -= 1

            trace = np.roll(channel.get_trace(), roll_samples)
            channel.set_trace(trace, "same")
            rolled_samples.append(roll_samples)

            delta_time = delta_time - roll_samples * channel.get_sampling_rate()
            channel.apply_time_shift(delta_time)

    # Assumes all channels have the same sampling rate
    max_rolled_sample = np.max(rolled_samples)
    print(max_rolled_sample / channel.get_sampling_rate())
    new_trace_start_time = channel.get_trace_start_time() + max_rolled_sample / channel.get_sampling_rate()

    print("cable :", new_trace_start_time)
    for channel in station.iter_channels():

        for efield in station.get_electric_fields_for_channels([channel.get_id()]):
            if efield.get_trace_start_time() < new_trace_start_time:
                raise ValueError(
                    "The trace start time of the channel is larger than of that "
                    "of the electric field after adding the cable delay."
                )

        trace = channel.get_trace()
        trace = trace[max_rolled_sample:]
        channel.set_trace(trace, "same")
        channel.set_trace_start_time(new_trace_start_time)
