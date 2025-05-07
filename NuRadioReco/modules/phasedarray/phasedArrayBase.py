import logging
import numpy as np
from scipy import constants

from NuRadioReco.utilities import units
from NuRadioReco.modules.analogToDigitalConverter import analogToDigitalConverter


logger = logging.getLogger('NuRadioReco.PhasedArrayTriggerBase')
cspeed = constants.c * units.m / units.s

class PhasedArrayBase():
    """
    Base class for all phased array trigger modules.
    """

    def __init__(self, log_level=logging.NOTSET):
        self.__pre_trigger_time = None
        self.__debug = None
        logger.setLevel(log_level)
        self._adc_to_digital_converter = analogToDigitalConverter()
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

    def calculate_time_delays(
            self, station, det,
            triggered_channels,
            phasing_angles=None,
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

        if triggered_channels is None:
            triggered_channels = [channel.get_id() for channel in station.iter_trigger_channels()]

        time_step = 1. / sampling_frequency

        ant_z = self.get_antenna_positions(station, det, triggered_channels, 2)

        self.check_vertical_string(station, det, triggered_channels)
        ref_z = np.max(np.fromiter(ant_z.values(), dtype=float))

        # Need to add in delay for trigger delay
        cable_delays = {}
        for channel in station.iter_trigger_channels(use_channels=triggered_channels):
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
        for channel in station.iter_trigger_channels(use_channels=triggered_channels):

            if channel_trace_start_time is None:
                channel_trace_start_time = channel.get_trace_start_time()

            elif channel_trace_start_time != channel.get_trace_start_time():
                raise ValueError(
                    'Phased array channels do not have matching trace start times. '
                    'This module is not prepared for this case.')

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

        if sum(diff_x) > cut or sum(diff_y) > cut:
            raise NotImplementedError('The phased triggering array should lie on a vertical line')

    def phase_signals(self, traces, beam_rolls, adc_output="voltage", saturation_bits=None):
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

            if adc_output == 'counts' and saturation_bits is not None:
                phased_trace[phased_trace>2**(saturation_bits-1)-1] = 2**(saturation_bits-1) - 1
                phased_trace[phased_trace<-2**(saturation_bits-1)] = -2**(saturation_bits-1)

            phased_traces[running_i] = phased_trace
            running_i += 1

        return phased_traces

    def power_sum(self, coh_sum, window, step, adc_output='voltage', averaging_divisor=None):
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
            Options:

                - 'voltage' to store the ADC output as discretised voltage trace
                - 'counts' to store the ADC output in ADC counts

        averaging_divisor: int (default None)
            Power integral divisor for averaging. If not specified,
            the divisor is the same as the summation window.

        Returns
        -------
        power:
            Integrated power in each integration window
        num_frames
            Number of integration windows calculated

        """

        # If not specified, the divisor is the same as the summation window.
        if averaging_divisor is None:
            averaging_divisor = window

        if adc_output not in ['voltage', 'counts']:
            raise ValueError(f'ADC output type must be "counts" or "voltage". Currently set to: {adc_output}')

        num_frames = int(np.floor((len(coh_sum) - window) / step))

        coh_sum_squared = (coh_sum * coh_sum)

        # if(adc_output == 'voltage'):
        #     coh_sum_squared = (coh_sum * coh_sum).astype(float)
        # elif(adc_output == 'counts'):
        #     coh_sum_squared = (coh_sum * coh_sum).astype(int)

        coh_sum_windowed = np.lib.stride_tricks.as_strided(
            coh_sum_squared, (num_frames, window),
            (coh_sum_squared.strides[0] * step, coh_sum_squared.strides[0]))

        power = np.sum(coh_sum_windowed, axis=1)
        return_power = power.astype(float) / averaging_divisor

        if adc_output=='counts':
            return_power = np.round(return_power)

        return return_power, num_frames

    def end(self):
        pass
