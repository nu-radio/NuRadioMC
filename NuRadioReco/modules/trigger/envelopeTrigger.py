from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.trigger.highLowThreshold import get_majority_logic
from NuRadioReco.framework.trigger import EnvelopeTrigger
import NuRadioReco.utilities.fft
import numpy as np
import scipy.signal
import copy
import time
import logging

logger = logging.getLogger('envelopeTrigger')


def get_envelope_triggers(trace, threshold):  # define trigger constraint for each channel

    """
    calculates a Hilbert-envelope based trigger

    Parameters
    ----------
    trace: array of floats
        the signal trace
    threshold: float
        the threshold
    Returns
    -------
    triggered bins: array of bools
        the bins where the trigger condition is satisfied
    """

    return np.abs(scipy.signal.hilbert(trace)) > threshold


class triggerSimulator:
    """
    Calculate a simple amplitude trigger depending on the Hilbert-envelope.
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self):
        return

    @register_run()
    def run(self, evt, station, det, passband, order, threshold, coinc_window, number_coincidences=2, triggered_channels=None, trigger_name='envelope_trigger'):
        """
        Simulates simple threshold trigger based on an Hilbert-envelope of the trace. Passband of the trigger, coincidence
        window within different channels should have triggered, and the number of channels needed to trigger can be specified.

        Parameters
        ----------
        evt: Event
            Event to run the module on
        station: Station
            Station to run the module on
        det: Detector
            The detector description
        passband: list
            Passband of the filter to apply before the trigger
        order: int
            Order of the butterworth filter to apply before the trigger
        threshold: float or dict of floats
            threshold above (or below) a trigger is issued, absolute amplitude
            a dict can be used to specify a different threshold per channel where the key is the channel id
        number_coincidences: int
            number of channels that are required in coincidence to trigger a station
        triggered_channels: array of ints or None
            channels ids that are triggered on, if None trigger will run on all channels
        coinc_window: float
            time window in which number_coincidences channels need to trigger
        trigger_name: string
            a unique name of this particular trigger
        """

        t = time.time()  # absolute time of system

        sampling_rate = station.get_channel(det.get_channel_ids(station.get_id())[0]).get_sampling_rate()
        dt = 1. / sampling_rate

        triggered_bins_channels = []
        channels_that_passed_trigger = []

        if triggered_channels is None:  # caveat: all channels start at the same time
            for channel in station.iter_channels():
                channel_trace_start_time = channel.get_trace_start_time()
                break
        else:
            channel_trace_start_time = station.get_channel(triggered_channels[0]).get_trace_start_time()

        for channel in station.iter_channels():
            # get filter
            frequencies = channel.get_frequencies()

            f = np.zeros_like(frequencies, dtype=complex)
            mask = frequencies > 0
            b, a = scipy.signal.butter(order, passband, 'bandpass', analog=True)  # Numerator (b) and denominator (a) polynomials of the IIR filter
            w, h = scipy.signal.freqs(b, a, frequencies[mask])  # w :The angular frequencies at which h was computed. h :The frequency response.
            f[mask] = h

            # apply filter
            freq_spectrum_fft = channel.get_frequency_spectrum()
            freq_spectrum_fft_copy = copy.copy(freq_spectrum_fft)  # copy spectrum so it is only changed within the trigger module
            sampling_rate = channel.get_sampling_rate()

            freq_spectrum_fft_copy *= f
            trace_filtered = NuRadioReco.utilities.fft.freq2time(freq_spectrum_fft_copy, sampling_rate)

            # apply envelope trigger to each channel
            channel_id = channel.get_id()

            trace = trace_filtered
            if triggered_channels is not None and channel_id not in triggered_channels:
                logger.debug("skipping channel {}".format(channel_id))
                continue
            if channel.get_trace_start_time() != channel_trace_start_time:
                logger.warning('Channel has a trace_start_time that differs from '
                               '        the other channels. The trigger simulator may not work properly')

            if(isinstance(threshold, dict)):
                threshold_tmp = threshold[channel_id]
            else:
                threshold_tmp = threshold
            triggered_bins = get_envelope_triggers(trace, threshold_tmp)
            triggered_bins_channels.append(triggered_bins)

            if True in triggered_bins:
                channels_that_passed_trigger.append(channel.get_id())

        # check for coincidences with get_majority_logic(tts, number_of_coincidences=2,
        # time_coincidence=32 * units.ns, dt=1 * units.ns)
        # returns:
        # triggered: bool; returns True if majority logic is fulfilled --> has_triggered
        # triggered_bins: array of ints; the bins that fulfilled the trigger --> triggered_bins
        # triggered_times = triggered_bins * dt: array of floats;
        # the trigger times relative to the trace --> triggered_times

        has_triggered, triggered_bins, triggered_times = get_majority_logic(triggered_bins_channels,
                                                                            number_coincidences, coinc_window, dt)

        trigger = EnvelopeTrigger(trigger_name, passband, order, threshold, number_coincidences, coinc_window, triggered_channels)
        trigger.set_triggered_channels(channels_that_passed_trigger)

        if has_triggered:
            trigger.set_triggered(True)
            trigger.set_trigger_time(triggered_times.min() + channel_trace_start_time)  # trigger_time = time from the beginning of the trace
            logger.debug("station has triggered")

        else:
            trigger.set_triggered(False)
            trigger.set_trigger_time(None)
            logger.debug("station has NOT triggered")

        station.set_trigger(trigger)
        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
