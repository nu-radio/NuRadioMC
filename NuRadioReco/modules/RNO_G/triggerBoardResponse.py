import logging
import numpy as np

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.analogToDigitalConverter import analogToDigitalConverter
from NuRadioReco.utilities import units

logger = logging.getLogger("NuRadioReco.triggerBoardResponse")


class triggerBoardResponse:
    """
    Simulates the response of the trigger board, nominally the "flower board"
    Includes:
    * analog frequency filter
    * ADC gain to fix the noise RMS to a specified number of bits
    * (optional) applies digitization to the waveforms
    """

    def __init__(self, log_level=logging.WARNING):
        logger.setLevel(log_level)
        self.logger = logger
        self.__t = 0
        self.begin()

    def begin(self, adc_input_range=2 * units.volt, clock_offset=0.0, adc_output="voltage"):
        """
        Parameters
        ----------
        adc_input_range : float (default: 2V)
            the voltage range of the ADC (should be given in units of volts)
        clock_offset: bool
            If True, a random clock offset between -1 and 1 clock cycles is added
        adc_output: string
            Options:

            * 'voltage' to store the ADC output as discretised voltage trace
            * 'counts' to store the ADC output in ADC counts

        """

        self._adc = analogToDigitalConverter()
        self._clock_offset = clock_offset
        self._adc_output = adc_output

        # the fields that need to exist in the detector description for this module to work
        self._mandatory_fields = ["trigger_adc_nbits", "trigger_adc_noise_nbits"]

        # Table 21 in https://www.analog.com/media/en/technical-documentation/data-sheets/hmcad1511.pdf
        self._triggerBoardAmplifications = np.array([1, 1.25, 2, 2.5, 4, 5, 8, 10, 12.5, 16, 20, 25, 32, 50])
        self._adc_input_range = adc_input_range

    def apply_trigger_filter(self, station, trigger_channels, trigger_filter):
        """
        Applies the requested trigger filter to the `trigger_channels`

        Parameters
        ----------
        station : Station
            Station to use
        trigger_channels : list
            Channels that this function should be applied to
        trigger_filter : function
            set of interpolations describing the `gain` and `phase` of the filter
            (see function `load_amp_response` in file `./detector/RNO_G/analog_components.py`)

        """

        for channel_id in trigger_channels:
            channel = station.get_trigger_channel(channel_id)

            # calculate and apply trigger filters
            freqs = channel.get_frequencies()
            filt = trigger_filter(freqs)
            channel.set_frequency_spectrum(channel.get_frequency_spectrum() * filt, channel.get_sampling_rate())

    def get_avg_vrms(self, station, trigger_channels, trace_split=20):
        """
        Estimates the RMS voltage of the triggering antennas by splitting the waveforms
        into chunks and taking the median of standard deviation of the chunks

        Parameters
        ----------
        station : Station
            Station to use
        trigger_channels : list
            Channels that this function should be applied to
        trace_split : int (default: 9)
            How many chunks each of the waveforms will be split into before calculating
            the standard deviation

        Returns
        -------
        approx_vrms : float
            the median RMS voltage of the waveforms

        """

        avg_vrms = 0
        for channel_id in trigger_channels:
            channel = station.get_trigger_channel(channel_id)
            trace = np.array(channel.get_trace())
            trace = trace[: int(trace_split * int(len(trace) / trace_split))].reshape((trace_split, -1))
            approx_vrms = np.median(np.std(trace, axis=1))
            logger.debug(f"    Ch: {channel_id}\tObser Vrms: {approx_vrms / units.mV:0.3f} mV")
            avg_vrms += approx_vrms

        avg_vrms /= len(trigger_channels)
        self.logger.debug(f"Average Vrms: {avg_vrms / units.mV:0.3f} mV")
        return approx_vrms

    def apply_adc_gain(self, station, det, trigger_channels, avg_vrms=None):
        """
        Calculates and applies the gain adjustment such that the correct number
        of "noise bits" are realized. The ADC has fixed possible gain values and
        this module sets the one that is closest-to-but-greater-than the ideal value

        Parameters
        ----------
        station : Station
            Station to use
        det : Detector
            The detector description
        trigger_channels : list
            Channels that this function should be applied to
        avg_rms : float (default: None)
            The Vrms of the trigger channels including the trigger board filters
            If set to `None`, this will be estimated using the waveforms

        Returns
        -------
        vrms_after_gain : float
            the RMS voltage of the waveforms after the gain has been applied

        ideal_vrms: float
            the ideal vrms, as measured on the ADC capacitors

        """

        if avg_vrms is None:
            avg_vrms = self.get_avg_vrms(station, trigger_channels)

        self.logger.debug("Applying gain at ADC level")

        if not hasattr(avg_vrms, "__len__"):
            avg_vrms = np.full_like(trigger_channels, avg_vrms, dtype=float)

        vrms_after_gain = []
        for channel_id, vrms in zip(trigger_channels, avg_vrms):
            det_channel = det.get_channel(station.get_id(), channel_id)

            noise_bits = det_channel["trigger_adc_noise_nbits"]
            total_bits = det_channel["trigger_adc_nbits"]
            volts_per_adc = self._adc_input_range / (2 ** total_bits - 1)
            ideal_vrms = volts_per_adc * (2 ** (noise_bits - 1))

            msg = f"\t Ch: {channel_id}\t Target Vrms: {ideal_vrms / units.mV:0.3f} mV"
            msg += f"\t V/ADC: {volts_per_adc / units.mV:0.3f} mV"
            self.logger.debug(msg)

            # find the ADC gain from the possible values that makes the realized
            # vrms closest-to-but-greater-than the ideal value
            amplified_vrms_values = vrms * self._triggerBoardAmplifications
            mask = amplified_vrms_values > ideal_vrms
            if np.any(mask):
                gain_to_use = self._triggerBoardAmplifications[mask][0]
                vrms_after_gain.append(amplified_vrms_values[mask][0])
            else:
                gain_to_use = self._triggerBoardAmplifications[-1]
                vrms_after_gain.append(amplified_vrms_values[-1])

            channel = station.get_trigger_channel(channel_id)
            self.logger.debug(f"\t Ch: {channel_id}\t Actuall Vrms: {np.std(channel.get_trace() * gain_to_use) / units.mV:0.3f} mV")
            channel.set_trace(channel.get_trace() * gain_to_use, channel.get_sampling_rate())
            self.logger.debug(f"\t Used Vrms: {vrms_after_gain[-1] / units.mV:0.3f} mV" + f"\tADC Gain {gain_to_use}")
            eff_noise_bits = np.log2(vrms_after_gain[-1] / volts_per_adc) + 1
            self.logger.debug(f"\t Eff noise bits: {eff_noise_bits:0.2f}\tRequested: {noise_bits}")

        return np.array(vrms_after_gain), ideal_vrms


    def digitize_trace(self, station, det, trigger_channels, vrms):
        for channel_id in trigger_channels:
            channel = station.get_trigger_channel(channel_id)

            digitized_trace, adc_sampling_frequency = self._adc.get_digital_trace(
                station,
                det,
                channel,
                Vrms=vrms,
                trigger_adc=True,
                adc_type="perfect_floor_comparator",
                trigger_filter=None,  # Applied already
                clock_offset=self._clock_offset,
                adc_output=self._adc_output,
                return_sampling_frequency=True,
                channel_id=channel_id,
            )

            channel.set_trace(digitized_trace, adc_sampling_frequency)

    @register_run()
    def run(self, evt, station, det, trigger_channels, vrms=None, apply_adc_gain=True,
            digitize_trace=True):
        """
        Applies the additional filters on the trigger board and performs a gain amplification
        to get the correct number of trigger bits.

        Parameters
        ----------
        evt : Event
            Event to run the module on
        station : Station
            Station to run the module on
        det : Detector
            The detector description
        trigger_channels : list
            Channels that this module should consider applying the filter/board response
        vrms : float (default: None)
            The Vrms of the trigger channels including the trigger board filters
            If set to `None`, this will be estimated using the waveforms
        apply_adc_gain : bool (default: True)
            Apply the gain shift to achieve the specified level of noise bits
        digitize_trace : bool (default: True)
            Apply the quantization to the voltages (uses `NuRadioReco.modules.analogToDigitalConverter` to do so)

        Returns
        -------
        trigger_board_vrms : float
            the RMS voltage of the waveforms on the trigger board after applying the ADC gain
        """
        self.logger.debug("Applying the RNO-G trigger board response")

        if vrms is None:
            vrms = self.get_avg_vrms(station, trigger_channels)

        if apply_adc_gain:
            trigger_board_vrms, ideal_vrms = self.apply_adc_gain(station, det, trigger_channels, vrms)
        else:
            trigger_board_vrms = vrms
            ideal_vrms = vrms

        if digitize_trace:
            self.digitize_trace(station, det, trigger_channels, ideal_vrms)

        return trigger_board_vrms

    def end(self):
        from datetime import timedelta

        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        return dt
