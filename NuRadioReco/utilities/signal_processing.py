from NuRadioReco.utilities import units
from NuRadioReco.detector import detector

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


def add_cable_delay(station, det, sim_to_data, trigger=False, logger=None):
        """
        Add or subtract cable delay to a channel.

        Parameters
        ----------
        station: Station
            The station to add the cable delay to.

        det: Detector
            The detector description
        sim_to_data: bool
            If True, the cable delay is added. If False, the cable delay is subtracted.

        trigger: bool
            If True, take the time delay from the trigger channel response.
            Only possible if `det` is of type `rnog_detector.Detector`. (Default: False)

        logger: logging.Logger, default=None
            If set, use `logger.debug(..)` to log the cable delay.
        """

        if sim_to_data:
            new_trace_start_times = []
            for channel in station.iter_channels():
                if trigger:
                    if not channel.has_extra_trigger_channel():
                        continue

                    if isinstance(det,detector.detector_base.DetectorBase):
                        cable_delay = det.get_cable_delay(station.get_id(), channel.get_id())
                    else:
                        cable_delay = det.get_cable_delay(station.get_id(), channel.get_id(), trigger=True)
                    new_trace_start_times.append(channel.get_trigger_channel().get_trace_start_time() + cable_delay)

                else:
                    cable_delay = det.get_cable_delay(station.get_id(), channel.get_id())
                    new_trace_start_times.append(channel.get_trace_start_time() + cable_delay)

            new_trace_start_time = np.min(new_trace_start_times)
            delta_times = np.array(new_trace_start_times) - new_trace_start_time
            if logger is not None:
                logger.info(f"New traces start time after adding the cable delay is {new_trace_start_time / units.ns:.2f}ns ")

            for channel, delta_time in zip(station.iter_channels(), delta_times):
                if trigger:
                    if channel.has_extra_trigger_channel():
                        channel = channel.get_trigger_channel()
                    else:
                        continue

                if logger is not None:
                    logger.debug(f"Shift channel {channel.get_id()} by {delta_time / units.ns:.2f}ns")

                if delta_time:
                    # tmp code block to try to get the same results as before
                    roll_samples = int(delta_time * channel.get_sampling_rate())
                    delta_time = delta_time - roll_samples * channel.get_sampling_rate()
                    trace = np.roll(channel.get_trace(), roll_samples)
                    channel.set_trace(trace, channel.get_sampling_rate())
                    channel.apply_time_shift(delta_time)

                channel.set_trace_start_time(new_trace_start_time)
        else:
            for channel in station.iter_channels():
                cable_delay = det.get_cable_delay(station.get_id(), channel.get_id())
                if logger is not None:
                    logger.debug(f"Subtract {cable_delay / units.ns:.2f}ns "
                                f"of cable delay to channel {channel.get_id()}")

                channel.add_trace_start_time(-cable_delay)
