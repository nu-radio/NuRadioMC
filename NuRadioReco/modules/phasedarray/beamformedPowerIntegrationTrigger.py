from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.trigger import PowerIntegrationPhasedTrigger
from NuRadioReco.modules.phasedarray.phasedArrayBase import PhasedArrayBase, default_angles

import numpy as np
import logging
logger = logging.getLogger('NuRadioReco.beamformedPowerIntegrationTrigger')

class BeamformedPowerIntegrationTrigger(PhasedArrayBase):
    """
    Calculates the trigger for a phased array with a primary beam.

    The channels that participate in both beams and the pointing angle for each
    subbeam can be specified.

    See https://arxiv.org/pdf/1809.04573.pdf
    """

    @register_run()
    def run(self, evt, station, det,
            Vrms=None,
            threshold=60 * units.mV,
            trigger_channels=None,
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
            averaging_divisor=None,
            step=16,
            apply_digitization=True,
            upsampling_method='fft',
            coeff_gain=128,
            filter_taps=45,
            saturation_bits=8,
            return_n_triggers=False,
            **kwargs
            ):

        """
        simulates phased array trigger for each event

        Several channels are phased by delaying their signals by an amount given
        by a pointing angle. Several pointing angles are possible in order to cover
        the sky. The array trigger_channels controls the channels that are phased,
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
            Power integration window for averaging
            Units of ADC time ticks
        averaging_divisor: int (default 32)
            Power integral divisor for averaging (division by 2^n much easier in firmware)
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
        if set_not_triggered:
            is_triggered = False
            trigger_delays = {}
            maximum_amps = np.zeros_like(phasing_angles)
        else:
            is_triggered, trigger_delays, trigger_time, trigger_times, \
                maximum_amps, n_triggers, triggered_beams = self.phased_trigger(
                    station=station, det=det,
                    threshold=threshold,
                    trigger_channels=trigger_channels,
                    phasing_angles=phasing_angles,
                    ref_index=ref_index,
                    apply_digitization=apply_digitization,
                    adc_kwargs=dict(
                        Vrms=Vrms,
                        trigger_adc=trigger_adc,
                        clock_offset=clock_offset,
                        adc_output=adc_output,
                        trigger_filter=trigger_filter),
                    upsampling_kwargs=dict(
                        upsampling_factor=upsampling_factor,
                        upsampling_method=upsampling_method,
                        coeff_gain=coeff_gain,
                        filter_taps=filter_taps),
                    saturation_bits=saturation_bits,
                    step=step,
                    window=window,
                    averaging_divisor=averaging_divisor,
                    ideal_transformer=False,
                    mode="power_sum",
            )

        # Create a trigger object to be returned to the station
        trigger = PowerIntegrationPhasedTrigger(
            trigger_name,
            threshold,
            trigger_channels=trigger_channels,
            primary_angles=phasing_angles,
            trigger_delays=trigger_delays,
            window_size=window,
            step_size=step,
            maximum_amps=maximum_amps
        )

        trigger.set_triggered(is_triggered)

        if is_triggered:
            #trigger_time(s)= time(s) from start of trace + start time of trace with
            # respect to moment of first interaction = trigger time from moment of first
            # interaction; time offset to interaction time (channel_trace_start_time)
            # already recognized in self.phased_trigger
            trigger.set_trigger_time(trigger_time)
            trigger.set_trigger_times(trigger_times)
        else:
            trigger.set_trigger_time(None)

        station.set_trigger(trigger)

        if return_n_triggers:
            return is_triggered, n_triggers
        else:
            return is_triggered
