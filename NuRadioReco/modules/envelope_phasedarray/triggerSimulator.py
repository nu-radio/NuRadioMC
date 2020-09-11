from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.trigger import EnvelopePhasedTrigger
from NuRadioReco.modules.phasedarray.triggerSimulator import triggerSimulator as phasedTrigger
from NuRadioReco.modules.phasedarray.triggerSimulator import get_beam_rolls, get_channel_trace_start_time
from NuRadioReco.utilities.diodeSimulator import diodeSimulator
from NuRadioReco.modules.analogToDigitalConverter import analogToDigitalConverter
import numpy as np
from scipy import constants
import logging
logger = logging.getLogger('phasedTriggerSimulator')

cspeed = constants.c * units.m / units.s

main_low_angle = -50. * units.deg
main_high_angle = 50. * units.deg
default_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 30))


class triggerSimulator(phasedTrigger):
    """
    Calculates the trigger for a envelope phased array.
    The channels that participate in the beam forming and the pointing angle for each
    beam can be specified. The envelope filter is implemented using the model
    for the ARA tunnel diode specified in utilities.diodeSimulator.

    See https://arxiv.org/pdf/1903.11043.pdf
    and https://elog.phys.hawaii.edu/elog/anita_notes/080827_041639/powertrigger.pdf
    """

    def envelope_trigger(self,
                         station,
                         det,
                         beam_rolls,
                         triggered_channels,
                         threshold_factor,
                         power_mean,
                         power_std,
                         output_passband=(None, 200 * units.MHz),
                         cut_times=(None, None),
                         trigger_adc=False):
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
            time_step = 1 / channel.get_sampling_rate()

            if trigger_adc:

                ADC = analogToDigitalConverter()
                trace = ADC.get_digital_trace(station, det, channel,
                                              trigger_adc=trigger_adc,
                                              random_clock_offset=True,
                                              adc_type='perfect_floor_comparator',
                                              diode=diode)
                time_step = 1 / det.get_channel(station_id, channel_id)['trigger_adc_sampling_frequency']
                times = np.arange(len(trace), dtype=np.float) * time_step
                times += channel.get_trace_start_time()

            else:

                trace = diode.tunnel_diode(channel)  # get the enveloped trace
                times = np.copy(channel.get_times())  # get the corresponding time bins

            if cut_times != (None, None):
                left_bin = np.argmin(np.abs(times - cut_times[0]))
                right_bin = np.argmin(np.abs(times - cut_times[1]))
                trace[0:left_bin] = 0
                trace[right_bin:None] = 0

            traces[channel_id] = trace[:]

        for subbeam_rolls in beam_rolls:

            phased_trace = None
            # Number of antennas: primary beam antennas
            Nant = len(beam_rolls[0])

            for channel_id in traces:

                trace = traces[channel_id]

                if(phased_trace is None):
                    phased_trace = np.roll(trace, subbeam_rolls[channel_id])
                else:
                    phased_trace += np.roll(trace, subbeam_rolls[channel_id])

            low_trigger = power_mean - power_std * np.abs(threshold_factor)
            low_trigger *= Nant

            threshold_passed = np.min(phased_trace) < low_trigger

            if threshold_passed:
                trigger_delays = {}
                for channel_id in subbeam_rolls:
                    trigger_delays[channel_id] = subbeam_rolls[channel_id] * time_step
                logger.debug("Station has triggered")
                return True, trigger_delays

        return False, {}

    @register_run()
    def run(self, evt, station, det,
            threshold_factor=6.5,
            power_mean=None,
            power_std=None,
            triggered_channels=None,
            trigger_name='envelope_phased_threshold',
            phasing_angles=default_angles,
            set_not_triggered=False,
            ref_index=1.75,
            output_passband=(None, 200 * units.MHz),
            cut_times=(None, None),
            trigger_adc=False):
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

        if (triggered_channels is None):
            triggered_channels = [channel.get_id() for channel in station.iter_channels()]

        if (power_mean is None) or (power_std is None):
            error_msg = 'The power_mean or power_std parameters are not defined. '
            error_msg += 'Please define them. You can use the calculate_noise_parameters '
            error_msg += 'function in utilities.diodeSimulator to do so.'
            raise ValueError(error_msg)

        if set_not_triggered:

            is_triggered = False
            trigger_delays = {}

        else:

            channel_trace_start_time = get_channel_trace_start_time(station, triggered_channels)

            logger.debug("primary channels:", triggered_channels)
            beam_rolls = get_beam_rolls(station, det, triggered_channels,
                                        phasing_angles, ref_index=ref_index)

            is_triggered, trigger_delays = self.envelope_trigger(station, det, beam_rolls,
                                                                 triggered_channels, threshold_factor,
                                                                 power_mean, power_std, output_passband, cut_times,
                                                                 trigger_adc)

        trigger = EnvelopePhasedTrigger(trigger_name, threshold_factor, power_mean, power_std,
                                        triggered_channels, phasing_angles, trigger_delays,
                                        output_passband)
        trigger.set_triggered(is_triggered)
        if is_triggered:
            trigger.set_trigger_time(channel_trace_start_time)
        else:
            trigger.set_trigger_time(None)
        station.set_trigger(trigger)

        return is_triggered

    def end(self):
        pass
