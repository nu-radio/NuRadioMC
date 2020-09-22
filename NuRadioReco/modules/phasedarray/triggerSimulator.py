from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.trigger import SimplePhasedTrigger
from NuRadioReco.modules.analogToDigitalConverter import analogToDigitalConverter
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import logging

#logger = logging.getLogger('phasedTriggerSimulator')
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('runstrawman') #phasedTriggerSimulator')

cspeed = constants.c * units.m / units.s

main_low_angle = -53. * units.deg
main_high_angle = 47. * units.deg
default_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 15))

class triggerSimulator:
    """
    Calculates the trigger for a phased array with a primary beam.
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

    def get_antenna_positions(self, station, det, triggered_channels=None, component=2):
        """
        Calculates the vertical coordinates of the antennas of the detector
        """

        ant_pos = [det.get_relative_position(station.get_id(), channel.get_id())[component]
                   for channel in station.iter_channels()
                   if channel.get_id() in triggered_channels]

        return np.array(ant_pos)

    def get_beam_rolls(self, station, det, triggered_channels,
                       phasing_angles=default_angles, ref_index=1.75,
                       trigger_adc=False):
        """
        Calculates the delays needed for phasing the array.
        """
        station_id = station.get_id()
        sampling_rate = None

        for channel in station.iter_channels(use_channels=triggered_channels):
            channel_id = channel.get_id()

            if(trigger_adc):
                sampling_rate_ = det.get_channel(station_id, channel_id)["trigger_adc_sampling_frequency"]
            else:
                sampling_rate_ = det.get_channel(station_id, channel_id)["adc_sampling_frequency"]

            if(sampling_rate_ != sampling_rate and sampling_rate != None):
                error_msg = 'Phased array channels do not have matching sampling rates. '
                error_msg += 'Please specify a common sampling rate.'
                raise ValueError(error_msg)
            else:
                sampling_rate = sampling_rate_

        time_step = 1. / sampling_rate

        ant_z = self.get_antenna_positions(station, det, triggered_channels, 2)

        self.check_vertical_string(station, det, triggered_channels)
        beam_rolls = []
        ref_z = (np.max(ant_z) + np.min(ant_z)) / 2

        for angle in phasing_angles:

            delay = (ant_z - ref_z) / cspeed * ref_index * np.sin(angle)
            roll = np.array(np.round(delay / time_step)).astype(int) # so number of entries to shift
            subbeam_rolls = dict(zip(triggered_channels, roll))

            logger.debug("angle:", angle / units.deg)
            logger.debug(subbeam_rolls)

            beam_rolls.append(subbeam_rolls)

        return beam_rolls

    def get_channel_trace_start_time(self, station, triggered_channels):

        channel_trace_start_time = None
        for channel in station.iter_channels(use_channels=triggered_channels):
            if channel_trace_start_time is None:
                channel_trace_start_time = channel.get_trace_start_time()
            elif channel_trace_start_time != channel.get_trace_start_time():
                error_msg = 'Phased array channels do not have matching trace start times. '
                error_msg += 'This module is not prepared for this case.'
                raise ValueError(error_msg)

        return channel_trace_start_time

    def check_vertical_string(self, station, det, triggered_channels):
        """
        Checks if the triggering antennas lie in a straight vertical line
        """

        cut = 1.e-3 * units.m
        ant_x = self.get_antenna_positions(station, det, triggered_channels, 0)
        diff_x = np.abs(ant_x - ant_x[0])
        ant_y = self.get_antenna_positions(station, det, triggered_channels, 1)
        diff_y = np.abs(ant_y - ant_y[0])
        if (sum(diff_x) > cut or sum(diff_y) > cut):
            raise NotImplementedError('The phased triggering array should lie on a vertical line')

    #def analog_to_digital(self, trace, ref_voltage, n_bits):
    #    lsb_voltage = ref_voltage / (2 ** (n_bits - 1) - 1)        
    #    digital_trace = np.floor(trace / lsb_voltage)
    #    digital_trace[digital_trace > (2 ** (n_bits - 1) - 1)] = (2 ** (n_bits - 1) - 1)
    #    digital_trace[digital_trace < -(2 ** (n_bits - 1) - 1)] = -(2 ** (n_bits - 1) - 1)
    #    digital_trace = digital_trace.astype(int)
    #    return digital_trace

    def powerSum(self, coh_sum, window=32, step=16, adc_output='voltage'):
        '''
        calculate power summed over a length defined by 'window', overlapping at intervals defined by 'step'
        '''
        num_frames = int(np.floor((len(coh_sum)-window) / step))
        
        if(adc_output == 'voltage'):
            coh_sum_squared = (coh_sum * coh_sum).astype(np.float)
        elif(adc_output == 'counts'):
            coh_sum_squared = (coh_sum * coh_sum).astype(np.int)

        coh_sum_windowed = np.lib.stride_tricks.as_strided(coh_sum_squared, (num_frames, window),
                                                           (coh_sum_squared.strides[0]*step, coh_sum_squared.strides[0]))
        power = np.sum(coh_sum_windowed, axis=1)
        
        return power.astype(np.float)/window, num_frames

    def phase_signals(self, traces, beam_rolls):
        phased_traces = [[] for i in range(len(beam_rolls))]

        running_i = 0
        for subbeam_rolls in beam_rolls:
            phased_trace = np.zeros(len(traces[0]))
            for channel_id in traces:

                trace = traces[channel_id]
                phased_trace += np.roll(trace, subbeam_rolls[channel_id])

            phased_traces[running_i] = phased_trace
            running_i += 1

        return phased_traces

    @register_run()
    def run(self, evt, station, det,
            Vrms = None,
            threshold = 60 * units.mV,
            triggered_channels=None,
            trigger_name='simple_phased_threshold',
            phasing_angles=default_angles,
            set_not_triggered=False,
            window_time=10.67 * units.ns,
            ref_index=1.75,
            cut_times=(None, None),
            trigger_adc=False, # by default, assumes the trigger ADC is the same as the channels ADC
            adc_output='voltage'):

        """
        simulates phased array trigger for each event

        Several channels are phased by delaying their signals by an amount given
        by a pointing angle. Several pointing angles are possible in order to cover
        the sky. The array triggered_channels controls the channels that are phased,
        according to the angles phasing_angles. 

        Parameters
        ----------
        evt: Event object
            Description of the current event
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        Vrms: 
            RMS of a single channel... so can probably do better later, seems like a weird way to get this value
        threshold: float
            threshold above (or below) a trigger is issued, absolute amplitude
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        trigger_name: string
            name for the trigger
        phasing_angles: array of float
            pointing angles for the primary beam
        set_not_triggered: bool (default: False)
            if True not trigger simulation will be performed and this trigger will be set to not_triggered
        window_time: float
            Width of the time window used in the power integration
        ref_index: float
            refractive index for beam forming
        cut_times: (float, float) tuple
            Times for cutting the trace. This helps reducing the number of noise-induced triggers.
        trigger_adc: bool, default True
            If True, analog to digital conversion is performed. It must be specified in the
            detector file. See analogToDigitalConverter module for information

        Returns
        -------
        is_triggered: bool
            True if the triggering condition is met
        """

        if(triggered_channels is None):
            triggered_channels = [channel.get_id() for channel in station.iter_channels()]

        if(adc_output != 'voltage' and adc_output != 'counts'):
            error_msg = 'ADC output type must be "counts" or "voltage". Currently set to:'+str(adc_output)
            raise ValueError(error_msg)

        ADC = analogToDigitalConverter()

        is_triggered = False
        trigger_delays = {}

        if not(set_not_triggered): # lol, double negative
            logger.debug("trigger channels:", triggered_channels)

            channel_trace_start_time = self.get_channel_trace_start_time(station, triggered_channels)
            
            beam_rolls = self.get_beam_rolls(station, det,                                              
                                             triggered_channels, 
                                             phasing_angles,
                                             ref_index=ref_index, 
                                             trigger_adc=trigger_adc)

            Nant = len(beam_rolls[0])
            squared_mean_threshold = Nant * threshold ** 2

            traces = {}

            for channel in station.iter_channels(use_channels=triggered_channels):
                channel_id = channel.get_id()
                station_id = station.get_id()

                random_clock_offset = np.random.uniform(0, 1)

                # So, this adc_reference_voltage .... 

                trace = ADC.get_digital_trace(station, det, channel,
                                              Vrms=Vrms,
                                              trigger_adc=trigger_adc,
                                              clock_offset=random_clock_offset,
                                              adc_type='perfect_floor_comparator',
                                              adc_output=adc_output)

                if(trigger_adc):
                    time_step = 1 / det.get_channel(station_id, channel_id)['trigger_adc_sampling_frequency']
                else:
                    time_step = 1 / det.get_channel(station_id, channel_id)['adc_sampling_frequency']

                times = np.arange(len(trace), dtype=np.float) * time_step
                times += channel.get_trace_start_time()
                                    
                traces[channel_id] = trace[:]

            phased_traces = self.phase_signals(traces, beam_rolls)
            
            for phased_trace in phased_traces:                

                # Create a sliding window
                squared_mean, num_frames = self.powerSum(coh_sum=phased_trace, window=32, step=16, adc_output=adc_output)

                if True in (squared_mean > squared_mean_threshold):
                    trigger_delays = {} # Need to figure out this element
                    logger.debug("Station has triggered")
                    is_triggered = True

        # Create a trigger object to be returned to the station
        trigger = SimplePhasedTrigger(trigger_name, threshold, triggered_channels, phasing_angles, trigger_delays)                                      
                                      
        trigger.set_triggered(is_triggered)

        if is_triggered:
            trigger.set_trigger_time(channel_trace_start_time)
        else:
            trigger.set_trigger_time(None)

        station.set_trigger(trigger)

        return is_triggered

    def end(self):
        pass
