import logging
import time
import numpy as np
from NuRadioReco.framework.trigger import RadiantAUXTrigger
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.modules.trigger.highLowThreshold import get_majority_logic
from NuRadioReco.utilities import fft
import matplotlib.pyplot as plt
import copy
from NuRadioReco.utilities import units
logger = logging.getLogger('radiant_aux_trigger')

def tanh_func(x, b, c):
    return 0.5*(np.tanh((x-b)/c) + 1)

class triggerSimulator:
    """
    Simulates the radiant auxiliar trigger on the surface based on a schottky diode. The parametrization is based on lab measurments of the full radiant board and fitted with a tanh function.
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self):
        return
    
    def get_filtered_trace_trigger(self, trace):
        """
        filters the trace in the frequency band according to the bandpass filter (80-200MHz) implemented in the trigger path of the radiant surface channels.

        Parameters
        ----------
        trace: array of floats
            the signal trace

        Returns
        -------
        filtered trace: array of floats
            the filtered trace
        """
        n_samples = len(trace)
        spectrum = fft.time2freq(trace, self.sampling_rate)
        freqs = np.fft.rfftfreq(n_samples, 1 / self.sampling_rate)
        filtering_out = np.where( (freqs < 0.08) | (freqs > 0.2)) # or!
        filtered_spectrum = spectrum.copy()
        filtered_spectrum [filtering_out] *= 0.0
        filtered_trace = fft.freq2time(filtered_spectrum, self.sampling_rate)

        return filtered_trace

    def get_threshold(self, trace, threshold_sigma, int_window=11*units.ns):
        """
        calculates the trigger threshold in (V^2/ns) given a multiple of the noise rms of a single trace.
        The noise rms is calulated from the first 130ns of each trace. The efieldToVoltageConverter() adds emtpy samples before each signal pulse, 
        this pre_pulse_time is currently set to 200ns, therefore 130ns is a safe choice for a noise time window.
        The threshold per trace should model the in-situ servoing of the trigger threshold to the present noise level.

        Parameters
        ----------
        trace: array of floats
            the signal trace
        threshold_sigma: float
            the threshold as a factor of noise rms
        int_window: float
            the time over which the trace is integrated in ns, default is 11ns determined by the radiant board

        Returns
        -------
        threshold: float
            the threshold in units V^2/ns

        """
        noise_samples = int(130*units.ns * self.sampling_rate)
        noise = self.get_power_int_trace(trace[10:noise_samples], int_window)
        threshold = threshold_sigma * np.std(noise)
        return threshold
        
    def get_power_int_trace(self, trace, int_window):
        """
        calculats a power integration over a given time window

        Parameters
        ----------
        trace: array of floats
            the trace
        window: float
            the integration windown in ns

        Returns
        -------
        integrated power: array of floats
            the integrated power trace
        """
        dt = 1. / self.sampling_rate
        i_window = int(int_window / dt)
        power = trace ** 2
        int_power = np.convolve(power, np.ones(i_window, dtype=int), 'valid') * dt
        return int_power

    def get_trigger_response(self, trace, threshold_sigma, int_window):
        """
        checks if which sample bins of a trace satisfies the trigger condition

        Parameters
        ----------
        trace: array of floats
            the signal trace
        threshold: float
            the threshold as a factor of rms noise
        window: float
            the integration window in ns

        Returns
        -------
        triggered bins: array of bools
            the bins where the trigger condition is satisfied
        threshold: float
            the threshold value in power integrated units (V^2/ns)
        """
        filtered_trace = self.get_filtered_trace_trigger(trace)

        power_int_trace = self.get_power_int_trace(filtered_trace, int_window)
        threshold = self.get_threshold(filtered_trace, threshold_sigma, int_window)
        triggered_bins = np.array(power_int_trace) > threshold
        return (triggered_bins, threshold)

    @register_run()
    def run(self, evt, station, det, threshold_sigma=30, int_window=11, coinc_window=60, number_coincidences=2, triggered_channels=None, trigger_name='radiant_aux_trigger'):
        """
        Simulates the radiant auxiliar trigger based on a schottky diode. The parametrization is based on lab measurments of the full radiant board and fitted with a tanh function.
        The trigger is applied to the channels specified in triggered_channels. The coincidence window indicates the time within different channels should have triggered, 
        in order to trigger the station. The number of coincidences specifies the number of channels that need to have triggered within the coincidence window.
        
        Parameters
        ----------
        evt: Event
            Event to run the module on
        station: Station
            Station to run the module on
        det: Detector
            The detector description
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

        considered_channels = copy.copy(triggered_channels)
        if triggered_channels is None:
            considered_channels = list(station.get_channel_ids())
        
        self.sampling_rate = station.get_channel(considered_channels[0]).get_sampling_rate()
        dt = 1. / self.sampling_rate

        triggered_bins_channels = []
        channels_that_passed_trigger = []
        channel_trace_start_time = station.get_channel(considered_channels[0]).get_trace_start_time()
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if considered_channels is not None and channel_id not in considered_channels:
                logger.debug("skipping channel {}".format(channel_id))
                continue

            if channel.get_trace_start_time() != channel_trace_start_time:
                logger.warning('Channel has a trace_start_time that differs from '
                               '        the other channels. The trigger simulator may not work properly')

            (triggered_bins, threshold) = self.get_trigger_response(channel.get_trace(), threshold_sigma, int_window)
            triggered_bins_channels.append(triggered_bins)

            if np.any(triggered_bins):
                channels_that_passed_trigger.append(channel.get_id())
                
        has_triggered, triggered_bins, triggered_times = get_majority_logic(triggered_bins_channels,
                                                                            number_coincidences, coinc_window, dt)
        trigger = RadiantAUXTrigger(trigger_name, threshold_sigma, int_window, number_coincidences, coinc_window, considered_channels)
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