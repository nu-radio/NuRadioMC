from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.trigger import SimplePhasedTrigger
import numpy as np
from scipy import signal
from scipy import constants
import logging

logger = logging.getLogger('phasedArrayTriggerSimulator')

speed_of_light = constants.c * units.m / units.s



def get_channel_trace_start_time(station, triggered_channels):

    channel_trace_start_time = None
    for channel_id in triggered_channels:
        channel = station.get_channel(channel_id)
        if channel_trace_start_time is None:
            channel_trace_start_time = channel.get_trace_start_time()
        elif channel_trace_start_time != channel.get_trace_start_time():
            error_msg = 'Phased array channels do not have matching trace start times. '
            error_msg += 'This module is not prepared for this case.'
            raise ValueError(error_msg)

    return channel_trace_start_time



def calculate_trigger(station, det, threshold, triggered_channels, window_time, ref_index, upsampling_factor):
    """
    add documentation of function
    """
    
    # to get the sampling rate per channel do
    sampling_rate = det.get_sampling_frequency(station.get_id(), channel_id)
    # to get the position of the antenna do
    antenna_position = det.get_relative_position(station.get_id(), channel_id)  # relative coordinates with respect to the surface
    
    traces = {}
    # get and upsample traces
    for channel_id in triggered_channels:
        trace = station.get_channel(channel_id).get_trace()  # retrieve the voltage trace
        
        # you probably want to check that the current sampling rate of the channel corresponds to the detector sampling rate
        if(station.get_channel(channel_id).get_sampling_frequency() != sampling_rate):
            raise ValueError(f"current sampling rate of channel {channel_id} of {station.get_channel(channel_id).get_sampling_frequency()/units.MHz:.0f} MHz does not correspond to detector sampling rate of {sampling_rate/units.MHz:.0f}MHz")
        
        if(upsampling_factor >= 2):  # optinally upsample the trace
            trace = signal.resample(trace, upsampling_factor * len(trace))
        traces[channel_id] = trace

    # calculate the beams and time delays
    
    # assuming the dictionary beams contains the relevant quantities for the beam
    for beam in beams.values():
        phased_trace = np.zeros_like(traces[0])  # initialize empty trace
        for channel_id in triggered_channels:
            phased_trace += np.roll(traces[channel_id], beam['delay'][channel_id])
        # calculate if summed trace fulfills power trigger
        # probably loop over samples from which the window will be calculated
        integrated_power = np.sum(phased_trace[window_start:(window_start + window_length)])
        if(integrated_power > threshold):
            # jippi, we have a trigger


class triggerSimulator:
    """
    Calculates the trigger for a phased array with a primary and a secondary beam.
    The channels that participate in both beams and the pointing angle for each
    subbeam can be specified.

    See https://arxiv.org/pdf/1809.04573.pdf
    """

    def __init__(self):
        self.__t = 0
        self.__pre_trigger_time = None
        self.__debug = None
        self.begin()

    def begin(self, debug=False, pre_trigger_time=100 * units.ns):
        self.__pre_trigger_time = pre_trigger_time
        self.__debug = debug

    @register_run()
    def run(self, evt, station, det,
            threshold,
            triggered_channels=None,
            trigger_name='simple_phased_threshold',
            set_not_triggered=False,
            window_time=10.67 * units.ns,
            ref_index=1.75,
            upsampling_factor=1):
        """
        simulates phased array trigger for each event

        Describe module here

        Parameters
        ----------
        evt: Event object
            Description of the current event
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        threshold: float
            threshold above (or below) a trigger is issued, absolute amplitude
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        trigger_name: string
            name for the trigger
        set_not_triggered: bool (default: False)
            if True not trigger simulation will be performed and this trigger will be set to not_triggered
        window_time: float
            Width of the time window used in the power integration
        ref_index: float
            refractive index for beam forming
        upsampling_factor: integer
            Upsampling factor. The trace will be a upsampled to a
            sampling frequency int_factor times higher than the original one
            before conversion to digital

        Returns
        -------
        is_triggered: bool
            True if the triggering condition is met
        """

        upsampling_factor = int(upsampling_factor)

        if (triggered_channels is None):
            triggered_channels = [channel.get_id() for channel in station.iter_channels()]

        if set_not_triggered:  # we can force this module to always save a negative trigger. This is used to speed up the calculation, e.g. with a separate "pretrigger" module. If this trigger did not trigger, we can skip the evaluation of this more complicated trigger.
            is_triggered = False
        else:
            # check if all channels have the same start time
            # Daniels module required them to be equal (it this the reality?) So far I simulated slightly different
            # cable lengths for each antenna resulting in different start times of the recorded traces.
            channel_trace_start_time = get_channel_trace_start_time(station, triggered_channels)

            calculate_trigger(station, det,
                              threshold=threshold,
                              triggered_channels=triggered_channels,
                              window_time=window_time,
                              ref_index=ref_index,
                              upsampling_factor=upsampling_factor)

        # create trigger object that stores properties of the trigger settings
        trigger = SimplePhasedTrigger(trigger_name, threshold, triggered_channels)
        trigger.set_triggered(is_triggered)  # set the trigger to True or False
        if is_triggered:  # set trigger time
            trigger.set_trigger_time(channel_trace_start_time)
        else:
            trigger.set_trigger_time(None)
        station.set_trigger(trigger)  # add trigger to station object

        return is_triggered

    def end(self):
        pass
