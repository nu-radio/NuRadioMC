from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.trigger.highLowThreshold import get_majority_logic
from NuRadioReco.framework.trigger import RNOGSurfaceTrigger
from NuRadioReco.utilities import units
import NuRadioReco.framework.base_trace as base_trace
import NuRadioReco.utilities.fft
import numpy as np
import scipy.signal
import copy
import time
import logging

logger = logging.getLogger('rnog_surface_trigger')

def schottky_diode(trace, threshold, temperature=250*units.kelvin, Vbias=2*units.volt):
    '''
    Returns the absolute maximum of the diode response depending on the maximum input signal.

    Options for temperature are [300K, 273K, 250K], for the bias voltages [2, 1.5, 1, 0.5]V.
    The triggerpath reduces the voltage by -10dB, hence the threshold is small. The linear fit is obtained from
    measurements stored in rnog_surface_trigger_measurements.

    Parameters
    ----------
    trace: array of floats
        the signal trace
    threshold: float
        the threshold
    temperature: float
        temperature of the board with diode
    Vbias: float
        applied bias voltage to the diode board

    Returns
    -------
    trigger: list of booleans
        trigger evaluation for each entry of the trace
    '''

    if temperature == 300 *units.kelvin:  #measurements taken at 300K
        if Vbias == 2*units.volt:
            a = 38.10032
            b = -9.19654194e-08
        if Vbias == 1.5*units.volt: #measurements taken at 1.6V
            a = 34.6274877
            b = -8.45140238e-05
        if Vbias == 1*units.volt:
            a = 24.6683322
            b = -1.03679002e-03
        if Vbias == 0.5*units.volt:  #measurements taken at 0.6V
            a = 16.0005295
            b = -5.83972425e-04

    if temperature == 273*units.kelvin: #measurements taken at 273K
        if Vbias == 2*units.volt:
            a = 45.9684369
            b = -6.15792586e-06
        if Vbias == 1.5*units.volt:
            a = 38.80661
            b = -3.01491791e-04
        if Vbias == 1*units.volt:
            a = 29.45406
            b = -7.71227505e-04
        if Vbias == 0.5*units.volt:
            a = 15.4192195
            b = -5.10530795e-04

    if temperature == 250*units.kelvin: #measurements taken at 248K
        if Vbias == 2*units.volt:
            a = 55.0380132
            b = -2.13447979e-05
        if Vbias == 1.5*units.volt:
            a = 49.3192486
            b = -1.67121000e-03
        if Vbias == 1*units.volt:
            a = 37.4351427
            b = -9.76368002e-04
        if Vbias == 0.5*units.volt:
            a = 17.4574667
            b = -5.90216182e-04

    v_in = (trace)**2
    v_out = a * v_in + b
    return v_out > threshold

class triggerSimulator:
    """
    Calculates the RNO_G surface trigger with a bandpass filter, a -10db attenuator and a schottky_diode.
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self):
        return

    @register_run()
    def run(self, evt, station, det, threshold, coinc_window=60*units.ns, number_coincidences=1, triggered_channels=[13, 16, 19], temperature=250*units.kelvin, Vbias=2*units.volt, trigger_name='rnog_surface_trigger'):
        """
        Run the surface trigger module and write trigger status into station.

        Parameters
        ----------
        evt: Event
            Event to run the module on
        station: Station
            Station to run the module on
        det: Detector
            The detector description
        threshold: float or dict of floats
            threshold above (or below) a trigger is issued, absolute amplitude
            a dict can be used to specify a different threshold per channel where the key is the channel id
        number_coincidences: int
            number of channels that are required in coincidence to trigger a station
        triggered_channels: array of ints or None
            channels ids that are triggered on, if None trigger will run on all channels
        coinc_window: float
            time window in which number_coincidences channels need to trigger
        temperature: float
            temperature of the board with diode in Kelvin
        Vbias: float
            applied bias voltage to the diode board in Volt
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
            channel_id = channel.get_id()
            logger.debug(f'channel id {channel_id}')
            if triggered_channels is not None and channel_id not in triggered_channels:
                logger.debug("skipping channel {}".format(channel_id))
                continue
            if channel.get_trace_start_time() != channel_trace_start_time:
                logger.warning('Channel has a trace_start_time that differs from '
                               '        the other channels. The trigger simulator may not work properly')

            frequencies = channel.get_frequencies()
            logger.debug(f'trace before trigger {np.abs(np.max(channel.get_trace()))}')
            trace_filtered = channel.base_trace.get_filtered_trace([80 * units.MHz, 180 * units.MHz], 'cheby1', order=3, rp=5)
            logger.debug(f'trace after bandpass {np.abs(np.max(trace_filtered))}')
            # apply -10dB attenuator of signal chain
            trace_filtered *= 10**(-10/20)
            logger.debug(f'trace after attenuator {np.abs(np.max(trace_filtered))}')
            trace = trace_filtered

            if(isinstance(threshold, dict)):
                threshold_tmp = threshold[channel_id]
            else:
                threshold_tmp = threshold
            triggered_bins = schottky_diode(trace, threshold_tmp)
            triggered_bins_channels.append(triggered_bins)
            if True in triggered_bins:
                channels_that_passed_trigger.append(channel.get_id())

        # check for coincidences with get_majority_logic(tts, number_of_coincidences, time_coincidence, dt)
        # returns:
        # triggered: bool; returns True if majority logic is fulfilled --> has_triggered
        # triggered_bins: array of ints; the bins that fulfilled the trigger --> triggered_bins
        # triggered_times = triggered_bins * dt: array of floats;
        # the trigger times relative to the trace --> triggered_times

        has_triggered, triggered_bins, triggered_times = get_majority_logic(triggered_bins_channels,
                                                                            number_coincidences, coinc_window, dt)

        trigger = RNOGSurfaceTrigger(trigger_name, threshold, number_coincidences, coinc_window, triggered_channels)
        trigger.set_triggered_channels(channels_that_passed_trigger)

        if has_triggered:
            trigger.set_triggered(True)
            trigger.set_trigger_time(triggered_times.min()+channel_trace_start_time)  # trigger_time = time from moment of first interaction
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
