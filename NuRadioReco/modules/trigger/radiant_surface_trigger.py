import logging
import time
import numpy as np
from NuRadioReco.framework.trigger import RadiantSurfaceTrigger
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.trigger.highLowThreshold import get_majority_logic

logger = logging.getLogger('radiant_surface_trigger')

def tanh_func(x, b, c):
    return 0.5*(np.tanh((x-b)/c) + 1)

def get_fit_params(vrms):
    b = 0.01486614
    c = 0.02164392
    return b, c

def calc_sliding_vpp(data, window_size=30, start_index=0, end_index=None):
    if end_index is None:
        end_index = len(data)
    vpps = []
    indices = []
    h = window_size // 2
    for i in range(start_index, end_index):
        window = data[i-h:i+h]
        vpp = np.max(window) - np.min(window)
        indices.append(i)
        vpps.append(vpp)
    return vpps, indices

def get_vpps_vrms_snr(trace):
    """ Calculate vpp, vrms and snr from trace """

    all_vpps, indices = calc_sliding_vpp(trace)
    max_vpp = np.max(all_vpps)
    vrms = np.std(trace[:800])
    snr = (max_vpp / (2 * vrms))
    return all_vpps, vrms, snr

def get_trigger_response(trace):  # define trigger constraint for each channel

    """
    check if the trace satisfies the trigger condition

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
    vpps, vrms, snr = get_vpps_vrms_snr(trace)
    b, c = get_fit_params(vrms)

    return tanh_func(vpps, b, c) > 0.5


class triggerSimulator:
    """
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self):
        return

    @register_run()
    def run(self, evt, station, det, coinc_window=60, number_coincidences=2, triggered_channels=[13, 16, 19], trigger_name='radian_surface_trigger'):
        """
        Simulates the radiant surface trigger based on a schottky diode. The parametrization is based on lab measurments of the full radiant board and fitted with a tanh function.
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
            if triggered_channels is not None and channel_id not in triggered_channels:
                logger.debug("skipping channel {}".format(channel_id))
                continue

            if channel.get_trace_start_time() != channel_trace_start_time:
                logger.warning('Channel has a trace_start_time that differs from '
                               '        the other channels. The trigger simulator may not work properly')

            triggered_bins = get_trigger_response(trace=channel.get_trace())
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

        trigger = RadiantSurfaceTrigger(trigger_name, number_coincidences, coinc_window, triggered_channels)
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