import logging
import functools
import numpy as np
from scipy import constants
from scipy.signal import hilbert, firwin

from NuRadioReco.utilities import units, signal_processing
from NuRadioReco.modules.analogToDigitalConverter import analogToDigitalConverter


logger = logging.getLogger('NuRadioReco.PhasedArrayTriggerBase')
cspeed = constants.c * units.m / units.s

main_low_angle = np.deg2rad(-55.0)
main_high_angle = -1.0 * main_low_angle
default_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))

class PhasedArrayBase():
    """
    Base class for all phased array trigger modules.
    """

    def __init__(self, log_level=logging.NOTSET):
        self.__pre_trigger_time = None
        self.__debug = None
        logger.setLevel(log_level)
        self._adc_to_digital_converter = analogToDigitalConverter()
        self.buffered_delays = {}
        self.begin()

    def begin(self, debug=False, pre_trigger_time=100 * units.ns):
        self.__pre_trigger_time = pre_trigger_time
        self.__debug = debug

    def _get_antenna_positions(self, station, det, triggered_channels, component=2):
        """
        Calculates the vertical coordinates of the antennas of the detector

        Parameters
        ----------
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        triggered_channels: array of ints
            Channels ids of the channels that form the primary phasing array.
        component: int (default: 2)
            Which cartesian coordinate to return

        Returns
        -------
        ant_pos: dict
            Dictionary of keys=antenna and content=position
        """
        return np.array([det.get_relative_position(station.get_id(), channel_id)[component]
            for channel_id in triggered_channels])

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

        if station.get_id() in self.buffered_delays:
            return self.buffered_delays[station.get_id()]

        if triggered_channels is None:
            triggered_channels = [channel.get_id() for channel in station.iter_trigger_channels()]

        ant_z = self._get_antenna_positions(station, det, triggered_channels, 2)
        self.check_vertical_string(station, det, triggered_channels)
        ref_z = np.max(ant_z)
        try:
            # RNO-G Detector
            cable_delays = np.array([det.get_cable_delay(station.get_id(), channel_id, trigger=True) for channel_id in triggered_channels])
        except:
            # Others without trigger kwarg
            cable_delays = np.array([det.get_cable_delay(station.get_id(), channel_id) for channel_id in triggered_channels])

        group_delays = np.zeros(len(triggered_channels))

        for i, channel in enumerate(triggered_channels):
            try:
                resp = det.get_signal_chain_response(station.get_id(), channel, trigger=True)
                group_delays[i] = resp.calculate_time_delay()
            except:
                pass

        beam_rolls = []

        for angle in phasing_angles:
            delays = (ant_z - ref_z) / cspeed * ref_index * np.sin(angle) - cable_delays - group_delays
            delays -= np.min(delays)
            roll = np.array(np.round(delays * sampling_frequency)).astype(int)
            subbeam_rolls = dict(zip(triggered_channels, roll))
            beam_rolls.append(subbeam_rolls)

        self.buffered_delays[station.get_id()] = beam_rolls

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
        ant_x = self._get_antenna_positions(station, det, triggered_channels, 0)
        diff_x = np.abs(ant_x - ant_x[0])
        ant_y = self._get_antenna_positions(station, det, triggered_channels, 1)
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

        for beam_i, subbeam_rolls in enumerate(beam_rolls):
            phased_trace = np.zeros(len(list(traces.values())[0]))

            for channel_id in traces:
                trace = traces[channel_id]
                phased_trace += np.roll(trace, int(subbeam_rolls[channel_id]))

            if adc_output == 'counts' and saturation_bits is not None:
                phased_trace[phased_trace>2**(saturation_bits-1)-1] = 2**(saturation_bits-1) - 1
                phased_trace[phased_trace<-2**(saturation_bits-1)] = -2**(saturation_bits-1)

            phased_traces[beam_i] = phased_trace

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

        coh_sum_windowed = np.lib.stride_tricks.as_strided(
            coh_sum_squared, (num_frames, window),
            (coh_sum_squared.strides[0] * step, coh_sum_squared.strides[0]))

        power = np.sum(coh_sum_windowed, axis=1)
        return_power = power.astype(float) / averaging_divisor

        if adc_output == 'counts':
            return_power = np.round(return_power)

        return return_power, num_frames

    def get_traces(self, station, det, triggered_channels=None,
            apply_digitization=False, adc_kwargs={},
            upsampling_kwargs={}):
        """
        Get the traces from the station object

        Parameters
        ----------
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        triggered_channels: array of ints (default: None)
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        apply_digitization: bool
            If True, the traces are digitized
        adc_kwargs: dict
            Keyword arguments for the ADC
        upsampling_kwargs: dict
            Keyword arguments for the upsampling function

        Returns
        -------
        traces: 2D array of floats
            Signals from the antennas to be phased together.
        sampling_frequency: float
            Sampling frequency of the traces
        """

        adc_output = adc_kwargs.get('adc_output')

        if adc_output not in ['voltage', 'counts']:
            raise ValueError(f'ADC output type must be "counts" or "voltage". Currently set to: {adc_output}')

        traces = {}
        final_sampling_frequency = None
        for channel in station.iter_trigger_channels(use_channels=triggered_channels):
            if apply_digitization:
                trace, adc_sampling_frequency = self._adc_to_digital_converter.get_digital_trace(
                    station, det, channel, return_sampling_frequency=True, **adc_kwargs)
            else:
                adc_sampling_frequency = channel.get_sampling_rate()
                trace = channel.get_trace()

            if upsampling_kwargs.get("upsampling_factor") >= 2:
                trace, adc_sampling_frequency = signal_processing.digital_upsampling(
                    trace, adc_sampling_frequency, **upsampling_kwargs
                )

            if final_sampling_frequency is None:
                final_sampling_frequency = adc_sampling_frequency

            elif final_sampling_frequency != adc_sampling_frequency:
                raise ValueError(
                    'Phased array channels do not have matching sampling frequencies. '
                    'This module is not prepared for this case.')

            traces[channel.get_id()] = trace

        logger.debug("trigger channels: {}".format(traces.keys()))
        return traces, final_sampling_frequency


    def hilbert_envelope(self, coh_sum, adc_output='voltage', ideal_transformer=False, hilbert_n_taps=31, hilbert_coeff_gain=1):

        if ideal_transformer:
            imag_an = np.imag(hilbert(coh_sum))

            if adc_output=='counts':
                imag_an = np.round(imag_an)

            envelope = np.sqrt(coh_sum**2 + imag_an**2)

        else:
            assert hilbert_n_taps % 2 != 0, "Num taps MUST be odd for a hilbert transformer"

            sin_factor = np.sin(np.linspace(-(hilbert_n_taps-1)/2, (hilbert_n_taps-1)/2, hilbert_n_taps))
            lp_filter = -1 * firwin(hilbert_n_taps, cutoff=0.25, pass_zero=False, fs=1)
            hil = 2 * sin_factor * lp_filter

            if hilbert_coeff_gain != 1:
                hil = np.round(hil * hilbert_coeff_gain) / hilbert_coeff_gain

            imag_an = np.convolve(coh_sum, hil, mode='full')[len(hil) // 2 : len(coh_sum) + len(hil) // 2]

            if adc_output == 'counts':
                imag_an = np.rint(imag_an)

            # simple square root calculation taken from section 13.2 in Understanding Digital Signal Processing by Richard Lyons
            envelope = np.max(np.array((coh_sum, imag_an)), axis=0) + (3 / 8) * np.min(np.array((coh_sum, imag_an)), axis=0)

        if adc_output == 'counts':
            envelope = np.rint(envelope)

        return envelope

    def phased_trigger(
            self, station, det,
            threshold=60 * units.mV,
            triggered_channels=None,
            phasing_angles=default_angles,
            ref_index=1.75,
            apply_digitization=False,
            adc_kwargs=dict(
                adc_output='voltage'),
            upsampling_kwargs=dict(
                upsampling_factor=1,
                upsampling_method='fft',
                coeff_gain=128,
                filter_taps=31),
            saturation_bits=8,
            window=32,
            step=16,
            averaging_divisor=None,
            hilbert_transformer_kwargs=dict(
                ideal_transformer=False,
                hilbert_n_taps=31,
                hilbert_coeff_gain=128),
            mode="power_sum",
        ):
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
        threshold: float
            threshold above (or below) a trigger is issued, absolute amplitude
        triggered_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        phasing_angles: array of float
            pointing angles for the primary beam
        ref_index: float (default 1.75)
            refractive index for beam forming
        apply_digitization: bool (default False)
            Perform the quantization with the analogToDigitalConverter module.
        adc_kwargs: dict
            Keyword arguments for the ADC module. Only used if `apply_digitization == True`.
            For arguments, see the documentation of the analogToDigitalConverter module.
        upsampling_kwargs: dict
            For arguments, see the documentation of the digital_upsampling function
        saturation_bits: int (default None)
            Determines what the coherenty summed waveforms will saturate to if using adc counts
        window: int (default 32)
            Power integration window for averaging
            Units of ADC time ticks
        step: int (default 16)
            Time step in power integral. If equal to window, there is no time overlap
            in between neighboring integration windows.
            Units of ADC time ticks
        averaging_divisor: int (default 32)
            Power integral divisor for averaging (division by 2^n much easier in firmware)
            Units of ADC time ticks
        ideal_transformer: bool (default False)
            If true, use scipy.hilbert to compute an exact Hilbert envelope with scipy.hilbert and np.sqrt.
            Otherwise, a firmware-like FIR-based Hilbert transformer will be used and with an approximation
            function to a square root to calculate the envelope.
        hilbert_n_taps: int (defult 31)
            Number of taps used to construct the FIR-based Hilbert transformer
        hilbert_coeff_gain: int (default 128)
            Rescaling factor to quantize the FIR-based Hilbert transformer
        mode: string (default: "power_sum")
            The mode of the trigger. Can be either "power_sum" or "hilbert_env".

                - "power_sum": uses the power_sum method to calculate the trigger
                - "hilbert_env": uses the hilbert envelope method to calculate the trigger

        Returns
        -------
        is_triggered: bool
            True if the triggering condition is met
        trigger_delays: dictionary
            the delays for the primary channels that have caused a trigger.
            If there is no trigger, it's an empty dictionary
        trigger_time: float
            the earliest trigger time with respect to first interaction time.
        trigger_times: dictionary
            all time bins that fulfil the trigger condition per beam. The key is the beam number. Time with respect to first interaction time.
        maximum_amps: list of floats (length equal to that of `phasing_angles`)
            the maximum value of all the integration windows for each of the phased waveforms
        n_trigs: int
            total number of triggers that happened for all beams across the full traces
        triggered_beams: list
            list of bools for which beams triggered
        """

        adc_output = adc_kwargs.get('adc_output')

        traces, adc_sampling_frequency = self.get_traces(
            station, det,
            triggered_channels=triggered_channels,
            apply_digitization=apply_digitization,
            adc_kwargs=adc_kwargs,
            upsampling_kwargs=upsampling_kwargs
        )
        triggered_channels = np.array(list(traces.keys()))

        time_step = 1.0 / adc_sampling_frequency
        beam_rolls = self.calculate_time_delays(
            station, det, triggered_channels,
            phasing_angles, ref_index=ref_index,
            sampling_frequency=adc_sampling_frequency)

        phased_traces = self.phase_signals(traces, beam_rolls, adc_output=adc_output, saturation_bits=saturation_bits)

        if adc_output == "counts":
            threshold = np.trunc(threshold)

        channel_trace_start_time = self.get_channel_trace_start_time(station, triggered_channels)
        maximum_amps = np.zeros(len(phased_traces))

        trigger_delays = {}
        n_trigs = 0
        triggered_beams = []
        trigger_time = None
        trigger_times = {}

        for iTrace, phased_trace in enumerate(phased_traces):
            is_triggered = False

            if mode == "power_sum":
                # Create a sliding window
                sig_trace, _ = self.power_sum(
                    coh_sum=phased_trace, window=window, step=step, averaging_divisor=averaging_divisor, adc_output=adc_output)
            elif mode == "hilbert_env":
                coeff_gain = upsampling_kwargs.get("coeff_gain")
                sig_trace = self.hilbert_envelope(
                    coh_sum=phased_trace, adc_output=adc_output, **hilbert_transformer_kwargs)
            else:
                raise ValueError("mode must be either 'power_sum' or 'hilbert_env'")

            maximum_amps[iTrace] = np.max(sig_trace)

            if np.any(sig_trace > threshold):
                is_triggered = True

                n_trigs += np.sum(sig_trace > threshold)
                trigger_delays[iTrace] = {channel_id: beam_rolls[iTrace][channel_id] * time_step
                    for channel_id in beam_rolls[iTrace]}

                triggered_bins = np.atleast_1d(np.squeeze(np.argwhere(sig_trace > threshold)))
                trigger_times[iTrace] = np.abs(np.min(list(trigger_delays[iTrace]))) + triggered_bins * step * time_step + channel_trace_start_time

                logger.debug(
                    "Station has triggered, at bins {}\n".format(triggered_bins) +
                    "Trigger delays: {}\n".format(trigger_delays[iTrace][triggered_channels[0]]) +
                    "Trigger time is {}ns\n".format(trigger_times[iTrace])
                )

            triggered_beams.append(is_triggered)

        is_triggered = np.any(triggered_beams)

        if is_triggered:
            trigger_time = np.amin([x.min() for x in trigger_times.values()])
            logger.debug("Trigger condition satisfied!\n"
                "All trigger times: {}\n".format(trigger_times) +
                "Minimum trigger time is {:.0f}ns".format(trigger_time))

        return is_triggered, trigger_delays, trigger_time, trigger_times, maximum_amps, n_trigs, triggered_beams
