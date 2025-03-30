from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.utilities import signal_processing
from NuRadioReco.framework.trigger import DigitalEnvelopePhasedTrigger
from NuRadioReco.modules.phasedarray.phasedArray import phasedArray
from NuRadioReco.modules.analogToDigitalConverter import analogToDigitalConverter
import logging
import scipy
import numpy as np
from scipy import constants
from scipy.signal import hilbert
import matplotlib.pyplot as plt
logger = logging.getLogger('phasedEnvelopeTriggerSimulator')

cspeed = constants.c * units.m / units.s

main_low_angle = np.deg2rad(-55.0)
main_high_angle = -1.0 * main_low_angle
default_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))


class triggerSimulator(phasedArray):
    """
    Calculates the trigger for a phased array with primary beams.

    The channels that participate in both beams and the pointing angle for each
    subbeam can be specified.

    Calculates Hilbert Envelope using digital FIR filters.
    """

    def hilbert_envelope(self, coh_sum, adc_output='voltage', coeff_gain=1, ideal_transformer=True):

        if ideal_transformer:

            imag_an=np.imag(hilbert(coh_sum))

            if adc_output=='counts':
                imag_an = np.round(imag_an)

            envelope=np.sqrt(coh_sum**2 + imag_an**2)

        else:
            #firmware like

            #31 sample fir transformer
            #hil=[ -0.0424413 , 0. , -0.0489708 , 0. , -0.0578745 , 0. , -0.0707355 , 0. , -0.0909457 , 0. , -0.127324 , 0.
            #       , -0.2122066 , 0. , -0.6366198 , 0., 0.6366198 , 0. , 0.2122066 , 0. , 0.127324 ,0. , 0.0909457 , 0. , .0707355
            #       , 0. , 0.0578745 , 0. , 0.0489708 , 0. , 0.0424413 ]

            #middle 15 coefficients ^
            hil = np.array([ -0.0909457 , 0. , -0.127324 , 0. , -0.2122066 , 0. , -0.6366198 , 0. , 0.6366198 , 0. , 0.2122066 ,
                            0. , 0.127324 , 0. , 0.0909457 ])

            if coeff_gain!=1:
                hil = np.round(hil * coeff_gain) / coeff_gain

            imag_an = np.convolve(coh_sum, hil, mode='full')[len(hil)//2 : len(coh_sum) + len(hil)//2]

            if adc_output=='counts':
                imag_an = np.rint(imag_an)

            envelope = np.max(np.array((coh_sum,imag_an)), axis=0) + (3 / 8) * np.min(np.array((coh_sum,imag_an)), axis=0)

        if adc_output=='counts':
            envelope = np.rint(envelope)

        return envelope

    def envelope_trigger(self, station, det,
                       Vrms=None,
                       threshold=60 * units.mV,
                       trigger_channels=None,
                       phasing_angles=default_angles,
                       ref_index=1.75,
                       trigger_adc=False,  # by default, assumes the trigger ADC is the same as the channels ADC
                       clock_offset=0,
                       adc_output='voltage',
                       trigger_filter=None,
                       upsampling_factor=1,
                       apply_digitization=False,
                       upsampling_method='fft',
                       coeff_gain=128,
                       filter_taps=31,
                       saturation_bits=8,
                       ideal_transformer=False
                       ):
        """
        simulates phased array trigger for each event

        Several channels are phased by delaying their signals by an amount given
        by a pointing angle. Several pointing angles are possible in order to cover
        the sky. The array trigger_channels controls the channels that are phased,
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
        trigger_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        phasing_angles: array of float
            pointing angles for the primary beam
        ref_index: float (default 1.75)
            refractive index for beam forming
        trigger_adc: bool (default True)
            If True, uses the ADC settings from the trigger. It must be specified in the
            detector file. See analogToDigitalConverter module for information
            (see option `apply_digitization`)
        clock_offset: float (default 0)
            Overall clock offset, for adc clock jitter reasons (see `apply_digitization`)
        adc_output: string (default 'voltage')
            - 'voltage' to store the ADC output as discretised voltage trace
            - 'counts' to store the ADC output in ADC counts and apply integer based math
        trigger_filter: array floats (default None)
            Freq. domain of the response to be applied to post-ADC traces
            Must be length for "MC freq"
        upsampling_factor: integer (default 1)
            Upsampling factor. The trace will be a upsampled to a
            sampling frequency int_factor times higher than the original one
            after conversion to digital
        apply_digitization: bool (default True)
            Perform the quantization of the ADC. If set to true, should also set options
            `trigger_adc`, `adc_output`, `clock_offset`
        upsampling_method: str (default 'fft')
            Choose between FFT, FIR, or Linear Interpolaion based upsampling methods
        coeff_gain: int (default 1)
            If using the FIR upsampling, this will convert the floating point output of the
            scipy filter to a fixed point value by multiplying by this factor and rounding to an
            int.
        filter_taps: int (default )
            If doing FIR upsampling, this determine the number of filter coefficients
        saturation_bits: int (default None)
            Determines what the coherenty summed waveforms will saturate to if using adc counts
        ideal_transformer: bool (default False)
            To use ideal Hilbert transformer and enveloping or to use approximate firmware-like
            calculation
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

        if(trigger_channels is None):
            trigger_channels = [channel.get_id() for channel in station.iter_trigger_channels()]

        if(adc_output != 'voltage' and adc_output != 'counts'):
            error_msg = 'ADC output type must be "counts" or "voltage". Currently set to:' + str(adc_output)
            raise ValueError(error_msg)

        ADC = analogToDigitalConverter()

        is_triggered = False
        trigger_delays = {}

        logger.debug(f"trigger channels: {trigger_channels}")

        traces = {}
        for channel in station.iter_trigger_channels(use_channels=trigger_channels):
            channel_id = channel.get_id()

            trace = np.array(channel.get_trace())

            if apply_digitization:
                trace, adc_sampling_frequency = ADC.get_digital_trace(station, det, channel,
                                                                  Vrms=Vrms,
                                                                  trigger_adc=trigger_adc,
                                                                  clock_offset=clock_offset,
                                                                  return_sampling_frequency=True,
                                                                  adc_type='perfect_floor_comparator',
                                                                  adc_output=adc_output,
                                                                  trigger_filter=None)
            else:
                adc_sampling_frequency = channel.get_sampling_rate()

            if not isinstance(upsampling_factor, int):
                try:
                    upsampling_factor = int(upsampling_factor)
                except:
                    raise ValueError("Could not convert upsampling_factor to integer. Exiting.")

            if(upsampling_factor >= 2):
                upsampled_trace, new_sampling_frequency = signal_processing.digital_upsampling(trace, adc_sampling_frequency, upsampling_method=upsampling_method,
                                                                            upsampling_factor=upsampling_factor, coeff_gain=coeff_gain,
                                                                            adc_output=adc_output, filter_taps=filter_taps)

                #  If upsampled is performed, the final sampling frequency changes
                trace = upsampled_trace[:]

                if(len(trace) % 2 == 1):
                    trace = trace[:-1]

            traces[channel_id] = trace[:]

        adc_sampling_frequency *= upsampling_factor

        time_step = 1.0 / adc_sampling_frequency
        beam_rolls = self.calculate_time_delays(station, det,
                                                trigger_channels,
                                                phasing_angles,
                                                ref_index=ref_index,
                                                sampling_frequency=adc_sampling_frequency)

        phased_traces = self.phase_signals(traces, beam_rolls, adc_output=adc_output, saturation_bits=saturation_bits)

        if adc_output == "counts":
            threshold=np.trunc(threshold)

        trigger_time = None
        trigger_times = {}
        channel_trace_start_time = self.get_channel_trace_start_time(station, trigger_channels)

        trigger_delays = {}
        maximum_amps = np.zeros(len(phased_traces))
        n_trigs = 0
        triggered_beams = []

        for iTrace, phased_trace in enumerate(phased_traces):
            is_triggered=False
            hilbert_env = self.hilbert_envelope(coh_sum=phased_trace, adc_output=adc_output, coeff_gain=coeff_gain, ideal_transformer=ideal_transformer)
            maximum_amps[iTrace] = np.max(hilbert_env)

            if True in (hilbert_env>threshold):
                n_trigs += len(np.where((hilbert_env > threshold)==True)[0])
                trigger_delays[iTrace] = {}
                for channel_id in beam_rolls[iTrace]:
                    trigger_delays[iTrace][channel_id] = beam_rolls[iTrace][channel_id] * time_step
                triggered_bins = np.atleast_1d(np.squeeze(np.argwhere(hilbert_env > threshold)))
                is_triggered = True
                trigger_times[iTrace] = np.abs(np.min(list(trigger_delays[iTrace]))) + triggered_bins * time_step + channel_trace_start_time

            triggered_beams.append(is_triggered)

        is_triggered=np.any(triggered_beams)

        if is_triggered:
            logger.debug("Trigger condition satisfied!")
            logger.debug("all trigger times", trigger_times)
            trigger_time = min([x.min() for x in trigger_times.values()])
            logger.debug(f"minimum trigger time is {trigger_time:.0f}ns")

        return is_triggered, trigger_delays, trigger_time, trigger_times, maximum_amps, n_trigs, triggered_beams

    @register_run()
    def run(self, evt, station, det,
            Vrms=None,
            threshold=60 * units.mV,
            trigger_channels=None,
            trigger_name='digital_envelope_phased_threshold',
            phasing_angles=default_angles,
            set_not_triggered=False,
            ref_index=1.75,
            trigger_adc=False,  # by default, assumes the trigger ADC is the same as the channels ADC
            clock_offset=0,
            adc_output='voltage',
            trigger_filter=None,
            upsampling_factor=1,
            apply_digitization=True,
            upsampling_method='fft',
            coeff_gain=128,
            filter_taps=45,
            saturation_bits=8,
            ideal_transformer=False,
            return_n_triggers=False,
            **kwargs
            ):

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
        trigger_channels: array of ints
            channels ids of the channels that form the primary phasing array
            if None, all channels are taken
        trigger_name: string
            name for the trigger
        phasing_angles: array of float
            pointing angles for the primary beam
        set_not_triggered: bool (default False)
            if True not trigger simulation will be performed and this trigger will be set to not_triggered
        ref_index: float (default 1.75)
            refractive index for beam forming
        trigger_adc: bool, (default True)
            If True, uses the ADC settings from the trigger. It must be specified in the
            detector file. See analogToDigitalConverter module for information
        clock_offset: float (default 0)
            Overall clock offset, for adc clock jitter reasons
        adc_output: string (default 'voltage')

            - 'voltage' to store the ADC output as discretised voltage trace
            - 'counts' to store the ADC output in ADC counts

        trigger_filter: array floats (default None)
            Freq. domain of the response to be applied to post-ADC traces
            Must be length for "MC freq"
        upsampling_factor: integer (default 1)
            Upsampling factor. The trace will be a upsampled to a
            sampling frequency int_factor times higher than the original one
            after conversion to digital
        window: int (default 32)
            Power integral window
            Units of ADC time ticks
        step: int (default 16)
            Time step in power integral. If equal to window, there is no time overlap
            in between neighboring integration windows.
            Units of ADC time ticks
        apply_digitization: bool (default True)
            Perform the quantization of the ADC. If set to true, should also set options
            `trigger_adc`, `adc_output`, `clock_offset`
        upsampling_method: str (default 'fft')
            Choose between FFT, FIR, or Linear Interpolaion based upsampling methods
        coeff_gain: int (default 1)
            If using the FIR upsampling, this will convert the floating point output of the
            scipy filter to a fixed point value by multiplying by this factor and rounding to an
            int.
        filter_taps: int (default 45)
            If using FIR upsampling these are the number of filter coefficients
        saturation_bits: int (default 8)
            If using counts, determines how large the coherently summed waveforms will saturate
        ideal_transformer: bool (default False)
            To use ideal Hilbert transformer and enveloping or to use approximate firmware-like
            calculation
        return_n_triggers: bool (default False)
            To better estimate simulated thresholds one should count the total triggers
            in the entire trace for each beam. If true, this return the total trigger number.

        Returns
        -------
        is_triggered: bool
            True if the triggering condition is met
        n_triggers: int (Optional)
            Count of the total number of triggers in all beamformed traces
        """

        if(trigger_channels is None):
            trigger_channels = [channel.get_id() for channel in station.iter_trigger_channels()]

        if(adc_output != 'voltage' and adc_output != 'counts'):
            error_msg = 'ADC output type must be "counts" or "voltage". Currently set to:' + str(adc_output)
            raise ValueError(error_msg)

        is_triggered = False
        trigger_delays = {}

        if(set_not_triggered):
            is_triggered = False
            trigger_delays = {}
            triggered_beams = []
        else:
            is_triggered, trigger_delays, trigger_time, trigger_times,\
                  maximum_amps, n_triggers, triggered_beams = self.envelope_trigger(station=station,
                                                                                    det=det,
                                                                                    Vrms=Vrms,
                                                                                    threshold=threshold,
                                                                                    trigger_channels=trigger_channels,
                                                                                    phasing_angles=phasing_angles,
                                                                                    ref_index=ref_index,
                                                                                    trigger_adc=trigger_adc,
                                                                                    clock_offset=clock_offset,
                                                                                    adc_output=adc_output,
                                                                                    trigger_filter=trigger_filter,
                                                                                    upsampling_factor=upsampling_factor,
                                                                                    apply_digitization=apply_digitization,
                                                                                    upsampling_method=upsampling_method,
                                                                                    coeff_gain=coeff_gain,
                                                                                    filter_taps=filter_taps,
                                                                                    saturation_bits=saturation_bits,
                                                                                    ideal_transformer=ideal_transformer
                                                                                    )

        # Create a trigger object to be returned to the station
        trigger = DigitalEnvelopePhasedTrigger(
            trigger_name,
            threshold,
            trigger_channels=trigger_channels,
            phasing_angles=phasing_angles,
            trigger_delays=trigger_delays,
            maximum_amps=maximum_amps
        )

        trigger.set_triggered(is_triggered)

        if is_triggered:
            #trigger_time(s)= time(s) from start of trace + start time of trace with respect to moment of first interaction = trigger time from moment of first interaction; time offset to interaction time (channel_trace_start_time) already recognized in self.phased_trigger
            trigger.set_trigger_time(trigger_time)
            trigger.set_trigger_times(trigger_times)
        else:
            trigger.set_trigger_time(None)

        station.set_trigger(trigger)

        if return_n_triggers:
            return is_triggered, n_triggers
        else:
            return is_triggered
