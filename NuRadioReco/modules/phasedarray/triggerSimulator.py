from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units

from NuRadioReco.framework.trigger import SimplePhasedTrigger
from NuRadioReco.modules.analogToDigitalConverter import analogToDigitalConverter
import logging
import scipy
import numpy as np
from scipy import constants

logger = logging.getLogger('phasedTriggerSimulator')

cspeed = constants.c * units.m / units.s

main_low_angle = np.deg2rad(-55.0)
main_high_angle = -1.0 * main_low_angle
default_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))


class triggerSimulator:
    """
    Calculates the trigger for a phased array with a primary beam.
    The channels that participate in both beams and the pointing angle for each
    subbeam can be specified.

    See https://arxiv.org/pdf/1809.04573.pdf
    """

    def __init__(self, log_level=logging.WARNING):
        self.__t = 0
        self.__pre_trigger_time = None
        self.__debug = None
        logger.setLevel(log_level)
        self.begin()

    def begin(self, debug=False, pre_trigger_time=100 * units.ns):
        self.__pre_trigger_time = pre_trigger_time
        self.__debug = debug

    def get_antenna_positions(self, station, det, triggered_channels=None, component=2):
        """
        Calculates the vertical coordinates of the antennas of the detector

        Parameters
        ----------
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        component: int
            Which cartesian coordinate to return

        Returns
        -------
        ant_pos: array of floatss
            Desired antenna position in requested coordinate

        """

        ant_pos = {}
        for channel in station.iter_channels(use_channels=triggered_channels):
            ant_pos[channel.get_id()] = det.get_relative_position(station.get_id(), channel.get_id())[component]

        return ant_pos

    def calculate_time_delays(self, station, det,
                              triggered_channels,
                              phasing_angles=default_angles,
                              ref_index=1.75,
                              sampling_frequency=None):

        """
        Calculates the delays needed for phasing the array.

        Parameters
        ----------
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        phasing_angles: array of float
            pointing angles for the primary beam
        ref_index: float
            refractive index for beam forming
        sampling_frequency: float
            Rate of the ADC used

        Returns
        -------
        beam_rolls: array of dicts of keys=antenna and content=delay
        """

        if(triggered_channels is None):
            triggered_channels = [channel.get_id() for channel in station.iter_channels()]

        time_step = 1. / sampling_frequency

        ant_z = self.get_antenna_positions(station, det, triggered_channels, 2)

        self.check_vertical_string(station, det, triggered_channels)
        ref_z = np.max(np.fromiter(ant_z.values(), dtype=float))

        # Need to add in delay for trigger delay
        cable_delays = {}
        for channel in station.iter_channels(use_channels=triggered_channels):
            cable_delays[channel.get_id()] = det.get_cable_delay(station.get_id(), channel.get_id())

        beam_rolls = []
        for angle in phasing_angles:

            delays = []
            for key in ant_z:
                delays += [-(ant_z[key] - ref_z) / cspeed * ref_index * np.sin(angle) - cable_delays[key]]

            delays -= np.max(delays)

            roll = np.array(np.round(np.array(delays) / time_step)).astype(int)

            subbeam_rolls = dict(zip(triggered_channels, roll))

            # logger.debug("angle:", angle / units.deg)
            # logger.debug(subbeam_rolls)

            beam_rolls.append(subbeam_rolls)

        return beam_rolls

    def get_channel_trace_start_time(self, station, triggered_channels):
        """
        Finds the start time of the desired traces.
        Throws an error if all the channels dont have the same start time.

        Parameters
        ----------
        station: Station object
            Description of the current station
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        Returns
        -------
        channel_trace_start_time: float
            Channel start time
        """

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
        Throws error if not.

        Parameters
        ----------
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        """

        cut = 1.e-3 * units.m
        ant_x = np.fromiter(self.get_antenna_positions(station, det, triggered_channels, 0).values(), dtype=float)
        diff_x = np.abs(ant_x - ant_x[0])
        ant_y = np.fromiter(self.get_antenna_positions(station, det, triggered_channels, 1).values(), dtype=float)
        diff_y = np.abs(ant_y - ant_y[0])
        if (sum(diff_x) > cut or sum(diff_y) > cut):
            raise NotImplementedError('The phased triggering array should lie on a vertical line')

    def power_sum(self, coh_sum, window, step, adc_output='voltage'):
        """
        Calculate power summed over a length defined by 'window', overlapping at intervals defined by 'step'

        Parameters
        ----------
        coh_sum: array of floats
            Phased signal to be integrated over
        window: int
            Power integral window
            Units of ADC time ticks
        step: int
            Time step in power integral. If equal to window, there is no time overlap
            in between neighboring integration windows
            Units of ADC time ticks.
        adc_output: string

            - 'voltage' to store the ADC output as discretised voltage trace
            - 'counts' to store the ADC output in ADC counts

        Returns
        -------
        power:
            Integrated power in each integration window
        num_frames
            Number of integration windows calculated

        """

        if(adc_output != 'voltage' and adc_output != 'counts'):
            error_msg = 'ADC output type must be "counts" or "voltage". Currently set to:' + str(adc_output)
            raise ValueError(error_msg)

        num_frames = int(np.floor((len(coh_sum) - window) / step))

        if(adc_output == 'voltage'):
            coh_sum_squared = (coh_sum * coh_sum).astype(np.float)
        elif(adc_output == 'counts'):
            coh_sum_squared = (coh_sum * coh_sum).astype(int)

        coh_sum_windowed = np.lib.stride_tricks.as_strided(coh_sum_squared, (num_frames, window),
                                                           (coh_sum_squared.strides[0] * step, coh_sum_squared.strides[0]))
        power = np.sum(coh_sum_windowed, axis=1)

        return power.astype(np.float) / window, num_frames

    def phase_signals(self, traces, beam_rolls):
        """
        Phase signals together given the rolls

        Parameters
        ----------
        traces: 2D array of floats
            Signals from the antennas to be phased together.
        beam_rolls: 2D array of floats
            The amount to shift each signal before phasing the
            traces together

        Returns
        -------
        phased_traces: array of arrays
        """

        phased_traces = [[] for i in range(len(beam_rolls))]

        running_i = 0
        for subbeam_rolls in beam_rolls:

            phased_trace = np.zeros(len(list(traces.values())[0]))
            for channel_id in traces:

                trace = traces[channel_id]
                phased_trace += np.roll(trace, subbeam_rolls[channel_id])

            phased_traces[running_i] = phased_trace
            running_i += 1

        return phased_traces

    def phased_trigger(self, station, det,
                       Vrms=None,
                       threshold=60 * units.mV,
                       triggered_channels=None,
                       phasing_angles=default_angles,
                       ref_index=1.75,
                       trigger_adc=False,  # by default, assumes the trigger ADC is the same as the channels ADC
                       clock_offset=0,
                       adc_output='voltage',
                       trigger_filter=None,
                       upsampling_factor=1,
                       window=32,
                       step=16):
        """
        simulates phased array trigger for each event

        Several channels are phased by delaying their signals by an amount given
        by a pointing angle. Several pointing angles are possible in order to cover
        the sky. The array triggered_channels controls the channels that are phased,
        according to the angles phasing_angles.

        Parameters
        ----------
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        Vrms: float
            RMS of the noise on a channel, used to automatically create the digitizer
            reference voltage. If set to None, tries to use reference voltage as defined
            int the detector description file.
        threshold: float
            threshold above (or below) a trigger is issued, absolute amplitude
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        phasing_angles: array of float
            pointing angles for the primary beam
        ref_index: float
            refractive index for beam forming
        trigger_adc: bool, default True
            If True, analog to digital conversion is performed. It must be specified in the
            detector file. See analogToDigitalConverter module for information
        clock_offset: float
            Overall clock offset, for adc clock jitter reasons
        trigger_filter: array floats
            Freq. domain of the response to be applied to post-ADC traces
            Must be length for "MC freq"
        upsampling_factor: integer
            Upsampling factor. The trace will be a upsampled to a
            sampling frequency int_factor times higher than the original one
            after conversion to digital
        window: int
            Power integral window
            Units of ADC time ticks
        step: int
            Time step in power integral. If equal to window, there is no time overlap
            in between neighboring integration windows.
            Units of ADC time ticks

        Returns
        -------
        is_triggered: bool
            True if the triggering condition is met
        trigger_delays: dictionary
            the delays for the primary channels that have caused a trigger.
            If there is no trigger, it's an empty dictionary
        trigger_time: float
            the earliest trigger time
        trigger_times: dictionary
            all time bins that fulfil the trigger condition per beam. The key is the beam number.
        """

        if(triggered_channels is None):
            triggered_channels = [channel.get_id() for channel in station.iter_channels()]

        if(adc_output != 'voltage' and adc_output != 'counts'):
            error_msg = 'ADC output type must be "counts" or "voltage". Currently set to:' + str(adc_output)
            raise ValueError(error_msg)

        ADC = analogToDigitalConverter()

        is_triggered = False
        trigger_delays = {}

        logger.debug(f"trigger channels: {triggered_channels}")

        traces = {}
        for channel in station.iter_channels(use_channels=triggered_channels):
            channel_id = channel.get_id()

            trace = np.array(channel.get_trace())

            trace, adc_sampling_frequency = ADC.get_digital_trace(station, det, channel,
                                                                  Vrms=Vrms,
                                                                  trigger_adc=trigger_adc,
                                                                  clock_offset=clock_offset,
                                                                  return_sampling_frequency=True,
                                                                  adc_type='perfect_floor_comparator',
                                                                  adc_output=adc_output,
                                                                  trigger_filter=None)

            # Upsampling here, linear interpolate to mimic an FPGA internal upsampling
            if not isinstance(upsampling_factor, int):
                try:
                    upsampling_factor = int(upsampling_factor)
                except:
                    raise ValueError("Could not convert upsampling_factor to integer. Exiting.")

            if(upsampling_factor >= 2):
                '''
                # Zero inserting and filtering
                upsampled_trace = upsampling_fir(trace, adc_sampling_frequency,
                                                 int_factor=upsampling_factor, ntaps=1)

                channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
                ff = np.fft.rfftfreq(len(upsampled_trace), 1.0 / adc_sampling_frequency / upsampling_factor)
                filt = channelBandPassFilter.get_filter(ff, 0, 0, None, passband=[0, 240 * units.MHz], filter_type="cheby1", order=9, rp=.1)

                upsampled_trace = np.fft.irfft(np.fft.rfft(upsampled_trace) * filt)
                '''

                # FFT upsampling
                new_len = len(trace) * upsampling_factor
                upsampled_trace = scipy.signal.resample(trace, new_len)

                '''
                # Linear interpolation
                x = np.arange(len(trace))
                f_trace = interp1d(x, trace, kind='linear', fill_value=(trace[0], trace[-1]), bounds_error=False)
                x_new = np.arange(len(trace) * upsampling_factor) / upsampling_factor
                upsampled_trace = f_trace(x_new)
                '''

                #  If upsampled is performed, the final sampling frequency changes
                trace = upsampled_trace[:]

                if(len(trace) % 2 == 1):
                    trace = trace[:-1]

                adc_sampling_frequency *= upsampling_factor

            time_step = 1.0 / adc_sampling_frequency

            traces[channel_id] = trace[:]

        beam_rolls = self.calculate_time_delays(station, det,
                                                triggered_channels,
                                                phasing_angles,
                                                ref_index=ref_index,
                                                sampling_frequency=adc_sampling_frequency)

        phased_traces = self.phase_signals(traces, beam_rolls)

        trigger_time = None
        trigger_times = {}
        channel_trace_start_time = self.get_channel_trace_start_time(station, triggered_channels)

        trigger_delays = {}
        for iTrace, phased_trace in enumerate(phased_traces):

            # Create a sliding window
            squared_mean, num_frames = self.power_sum(coh_sum=phased_trace, window=window, step=step, adc_output=adc_output)

            if True in (squared_mean > threshold):
                trigger_delays[iTrace] = {}

                for channel_id in beam_rolls[iTrace]:
                    trigger_delays[iTrace][channel_id] = beam_rolls[iTrace][channel_id] * time_step

                triggered_bins = np.atleast_1d(np.squeeze(np.argwhere(squared_mean > threshold)))
                # logger.debug(f"Station has triggered, at bins {triggered_bins}")
                # logger.debug(trigger_delays)
                # logger.debug(f"trigger_delays {trigger_delays[iTrace][triggered_channels[0]]}")
                is_triggered = True
                trigger_times[iTrace] = trigger_delays[iTrace][triggered_channels[0]] + triggered_bins * step * time_step + channel_trace_start_time
                # logger.debug(f"trigger times  = {trigger_times[iTrace]}")
        if is_triggered:
            # logger.debug(trigger_times)
            trigger_time = min([x.min() for x in trigger_times.values()])
            # logger.debug(f"minimum trigger time is {trigger_time:.0f}ns")

        return is_triggered, trigger_delays, trigger_time, trigger_times

    @register_run()
    def run(self, evt, station, det,
            Vrms=None,
            threshold=60 * units.mV,
            triggered_channels=None,
            trigger_name='simple_phased_threshold',
            phasing_angles=default_angles,
            set_not_triggered=False,
            ref_index=1.75,
            trigger_adc=False,  # by default, assumes the trigger ADC is the same as the channels ADC
            clock_offset=0,
            adc_output='voltage',
            trigger_filter=None,
            upsampling_factor=1,
            window=32,
            step=16):

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
        Vrms: float
            RMS of the noise on a channel, used to automatically create the digitizer
            reference voltage. If set to None, tries to use reference voltage as defined
            int the detector description file.
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
        ref_index: float
            refractive index for beam forming
        trigger_adc: bool, default True
            If True, analog to digital conversion is performed. It must be specified in the
            detector file. See analogToDigitalConverter module for information
        clock_offset: float
            Overall clock offset, for adc clock jitter reasons
        adc_output: string

            - 'voltage' to store the ADC output as discretised voltage trace
            - 'counts' to store the ADC output in ADC counts

        trigger_filter: array floats
            Freq. domain of the response to be applied to post-ADC traces
            Must be length for "MC freq"
        upsampling_factor: integer
            Upsampling factor. The trace will be a upsampled to a
            sampling frequency int_factor times higher than the original one
            after conversion to digital
        window: int
            Power integral window
            Units of ADC time ticks
        step: int
            Time step in power integral. If equal to window, there is no time overlap
            in between neighboring integration windows.
            Units of ADC time ticks
        Returns
        -------
        is_triggered: bool
            True if the triggering condition is met
        """

        if(triggered_channels is None):
            triggered_channels = [channel.get_id() for channel in station.iter_channels()]

        if(adc_output != 'voltage' and adc_output != 'counts'):
            error_msg = 'ADC output type must be "counts" or "voltage". Currently set to:' + str(adc_output)
            raise ValueError(error_msg)

        is_triggered = False
        trigger_delays = {}

        if(set_not_triggered):
            is_triggered = False
            trigger_delays = {}
        else:
            is_triggered, trigger_delays, trigger_time, trigger_times = self.phased_trigger(station=station,
                                                                             det=det,
                                                                             Vrms=Vrms,
                                                                             threshold=threshold,
                                                                             triggered_channels=triggered_channels,
                                                                             phasing_angles=phasing_angles,
                                                                             ref_index=ref_index,
                                                                             trigger_adc=trigger_adc,
                                                                             clock_offset=clock_offset,
                                                                             adc_output=adc_output,
                                                                             trigger_filter=trigger_filter,
                                                                             upsampling_factor=upsampling_factor,
                                                                             window=window,
                                                                             step=step)

        # Create a trigger object to be returned to the station
        trigger = SimplePhasedTrigger(trigger_name, threshold, channels=triggered_channels,
                                      primary_angles=phasing_angles, trigger_delays=trigger_delays)

        trigger.set_triggered(is_triggered)

        if is_triggered:
            trigger.set_trigger_time(trigger_time)
            trigger.set_trigger_times(trigger_times)
        else:
            trigger.set_trigger_time(None)

        station.set_trigger(trigger)

        return is_triggered

    def end(self):
        pass
