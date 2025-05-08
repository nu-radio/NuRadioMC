from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.trigger import AnalogEnvelopePhasedTrigger
from NuRadioReco.modules.phasedarray.phasedArrayBase import PhasedArrayBase, default_angles
from NuRadioReco.utilities.diodeSimulator import diodeSimulator

import numpy as np
import logging

logger = logging.getLogger('NuRadioReco.analogBeamformedEnvelopeTrigger')


class AnalogBeamformedEnvelopeTrigger(PhasedArrayBase):
    """
    Calculates the trigger for a envelope phased array.
    The channels that participate in the beam forming and the pointing angle for each
    beam can be specified. The envelope filter is implemented using the model
    for the ARA tunnel diode specified in utilities.diodeSimulator.

    See https://arxiv.org/pdf/1903.11043.pdf
    and https://elog.phys.hawaii.edu/elog/anita_notes/080827_041639/powertrigger.pdf
    """

    def envelope_trigger(
            self,
            station,
            det,
            phasing_angles,
            ref_index,
            triggered_channels,
            envelope_type="diode",
            threshold_factor=None,
            power_mean=None,
            power_std=None,
            output_passband=(None, 200 * units.MHz),
            threshold=None,
            trigger_adc=False,
            apply_digitization=False,
            adc_output="voltage"
        ):
        """
        Calculates the envelope trigger for a certain phasing configuration.
        Beams are formed. Then, each channel to be phased is filtered with a
        tunnel diode, the outputs are phased and, if the minimum is lower than
        the number of antennas times (power_mean - power_std * np.abs(threshold_factor)),
        a trigger is created.

        Parameters
        ----------
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        beam_rolls: array of ints
            Contains the integers for rolling the voltage traces (delays)
        triggered_channels: array of ints
            Ids of the triggering channels
        threshold_factor: float
            the threshold factor
        power_mean: float
            mean of the noise trace after being filtered with the diode
        power_std: float
            standard deviation of the noise trace after being filtered with the
            diode. power_mean and power_std can be calculated with the function
            calculate_noise_parameters from utilities.diodeSimulator
        output_passband: (float, float) tuple
            Frequencies for a 6th-order Butterworth filter to be applied after
            the diode filtering.
        cut_times: (float, float) tuple
            Times for cutting the trace after diode filtering. This helps reducing
            the number of noise-induced triggers. Doing it the other way, that is,
            cutting and then filtering, will create two artificial pulses on the
            edges of the trace.
        trigger_adc: bool
            If True, analog to digital conversion is performed. It must be specified in the
            detector file. See analogToDigitalConverter module for information

        Returns
        -------
        is_triggered: bool
            True if the triggering condition is met
        trigger_delays: dictionary
            the delays for the primary channels that have caused a trigger.
            If there is no trigger, it's an empty dictionary
        """
        station_id = station.get_id()

        diode = diodeSimulator(output_passband)

        traces = {}
        for channel in station.iter_channels(use_channels=triggered_channels):
            channel_id = channel.get_id()
            adc_sampling_frequency = channel.get_sampling_rate()
            time_step = 1 / channel.get_sampling_rate()

            if trigger_adc:
                trace = self._adc_to_digital_converter.get_digital_trace(
                    station, det, channel,
                    trigger_adc=trigger_adc,
                    random_clock_offset=True,
                    adc_type='perfect_floor_comparator',
                    diode=diode)

                time_step = 1 / det.get_channel(station_id, channel_id)['trigger_adc_sampling_frequency']
                times = np.arange(len(trace), dtype=float) * time_step
                times += channel.get_trace_start_time()

            else:
                trace = diode.tunnel_diode(channel)  # get the enveloped trace
                times = np.copy(channel.get_times())  # get the corresponding time bins

            traces[channel_id] = trace[:]

        beam_rolls = self.calculate_time_delays(station, det,
                                                triggered_channels,
                                                phasing_angles,
                                                ref_index=ref_index,
                                                sampling_frequency=adc_sampling_frequency)

        phased_traces = self.phase_signals(traces, beam_rolls)


        trigger_time = None
        trigger_times = {}
        trigger_delays = {}
        n_trigs = 0
        triggered_beams = []
        maximum_amps = np.zeros(len(phased_traces))

        for iTrace, phased_trace in enumerate(phased_traces):
            is_triggered = False

            # Number of antennas: primary beam antennas
            Nant = len(beam_rolls[iTrace])


            low_trigger = power_mean - power_std * np.abs(threshold_factor)
            low_trigger *= Nant

            threshold_passed = np.min(phased_trace) < low_trigger

            if threshold_passed:
                is_triggered = True
                for channel_id in beam_rolls:
                    trigger_delays[channel_id] = beam_rolls[channel_id] * time_step
                logger.debug("Station has triggered")

            triggered_beams.append(is_triggered)

        is_triggered = np.any(triggered_beams)

        if is_triggered:
            logger.debug("Trigger condition satisfied!")
            logger.debug("all trigger times", trigger_times)
            trigger_time = min([x.min() for x in trigger_times.values()])
            logger.debug(f"minimum trigger time is {trigger_time:.0f}ns")

        return is_triggered, trigger_delays, trigger_time, trigger_times, maximum_amps, n_trigs, triggered_beams

    @register_run()
    def run(self, evt, station, det,
            trigger_name='envelope_phased_threshold',
            triggered_channels=None,
            phasing_angles=default_angles,
            set_not_triggered=False,
            ref_index=1.75,
            envelope_type="diode",
            threshold_factor=None,
            power_mean=None,
            power_std=None,
            output_passband=(None, 200 * units.MHz),
            threshold=None,
            trigger_adc=False,
            apply_digitization=False,
            adc_output="voltage",
            return_n_triggers=False
            ):
        """
        simulates phased array trigger for each event

        Several channels are phased by delaying their signals by an amount given
        by a pointing angle. Several pointing angles are possible in order to cover
        the sky. The array triggered_channels controls the channels that are phased,
        according to the angles phasing_angles. A secondary phasing that is added
        to this primary phasing is possible, and it is controlled by the parametres
        secondary_channels and secondary_phasing_angles.

        Parameters
        ----------
        evt: Event object
            Description of the current event
        station: Station object
            Description of the current station
        det: Detector object
            Description of the current detector
        threshold_factor: float
            threshold factor to be used for the envelope trigger
        power_mean: float
            mean of the noise trace after being filtered with the diode
        power_std: float
            standard deviation of the noise trace after being filtered with the
            diode. power_mean and power_std can be calculated with the function
            calculate_noise_parameters from utilities.diodeSimulator
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
        output_passband: (float, float) tuple
            Frequencies for a 6th-order Butterworth filter to be applied after
            the diode filtering.
        cut_times: (float, float) tuple
            Times for cutting the trace after diode filtering. This helps reducing
            the number of noise-induced triggers. Doing it the other way, that is,
            cutting and then filtering, will create two artificial pulses on the
            edges of the trace.
        trigger_adc: bool
            If True, analog to digital conversion is performed. It must be specified in the
            detector file. See analogToDigitalConverter module for information

        Returns
        -------
        is_triggered: bool
            True if the triggering condition is met
        """

        if triggered_channels is None:
            triggered_channels = [channel.get_id() for channel in station.iter_channels()]

        if power_mean is None or power_std is None:
            raise ValueError('The power_mean or power_std parameters are not defined. '
                'Please define them. You can use the calculate_noise_parameters '
                'function in utilities.diodeSimulator to do so.')

        if set_not_triggered:
            is_triggered = False
            trigger_delays = {}

        else:
            logger.debug("primary channels: {}".format(triggered_channels))

            is_triggered, trigger_delays, trigger_time, trigger_times, n_triggers = self.envelope_trigger(
                station,
                det,
                phasing_angles,
                ref_index,
                triggered_channels=triggered_channels,
                threshold_factor=threshold_factor,
                power_mean=power_mean,
                power_std=power_std,
                output_passband=output_passband,
                threshold=threshold,
                trigger_adc=trigger_adc,
                apply_digitization=apply_digitization,
                adc_output=adc_output
            )

        trigger = AnalogEnvelopePhasedTrigger(
            trigger_name, threshold_factor, power_mean, power_std,
            triggered_channels, phasing_angles, trigger_delays,
            output_passband)

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
