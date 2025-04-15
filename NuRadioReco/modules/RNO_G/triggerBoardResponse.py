from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.analogToDigitalConverter import analogToDigitalConverter
from NuRadioReco.utilities import units

import numpy as np
import logging
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
        self._log_level = log_level
        self.begin()

    def begin(self, clock_offset=0.0, adc_output="voltage"):
        """
        Parameters
        ----------
        clock_offset: bool
            If True, a random clock offset between -1 and 1 clock cycles is added
        adc_output: string
            Options:

            * 'voltage' to store the ADC output as discretised voltage trace
            * 'counts' to store the ADC output in ADC counts

        """
        self._adc = analogToDigitalConverter(log_level=self._log_level)
        self._clock_offset = clock_offset
        self.adc_output = adc_output

        # Table 21 in https://www.analog.com/media/en/technical-documentation/data-sheets/hmcad1511.pdf
        self._triggerBoardAmplifications = np.array([1, 1.25, 2, 2.5, 4, 5, 8, 10, 12.5, 16, 20, 25, 32, 50])
        self._adc_input_range = None
        self._nbits = None

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

    def get_noise_vrms_per_trigger_channel(self, station, trigger_channels, trace_split=20):
        """
        Estimates the RMS voltage of the triggering antennas by splitting the waveforms
        into chunks and taking the median of standard deviation of the chunks.

        Parameters
        ----------
        station : Station
            Station to use
        trigger_channels : list
            Channels that this function should be applied to
        trace_split : int (default: 20)
            How many chunks each of the waveforms will be split into before calculating
            the standard deviation

        Returns
        -------
        vrms : list of floats
            RMS voltage of the waveforms
        """
        vrms = np.zeros(len(trigger_channels))
        for idx, channel_id in enumerate(trigger_channels):
            channel = station.get_trigger_channel(channel_id)
            trace = channel.get_trace()

            n_samples_to_split = trace_split * (len(trace) // trace_split)
            trace = trace[:n_samples_to_split].reshape((trace_split, -1))
            approx_vrms = np.median(np.std(trace, axis=1))

            vrms[idx] = approx_vrms

        logger.debug("obs. Vrms {} mV".format(np.around(vrms / units.mV, 3)))
        return vrms

    def apply_adc_gain(self, station, det, trigger_channels, vrms_noise=None):
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
        vrms_noise : float (default: None)
            The (noise) Vrms of the trigger channels including the trigger board filters
            If set to `None`, this will be estimated using the waveforms.

        Returns
        -------
        vrms_after_gain : float
            the RMS voltage of the waveforms after the gain has been applied

        ideal_vrms: float
            the ideal vrms, as measured on the ADC capacitors

        """

        if vrms_noise is None:
            vrms_noise = self.get_noise_vrms_per_trigger_channel(station, trigger_channels)
            logger.debug("obs. Vrms {} mV".format(np.around(vrms_noise / units.mV, 3)))

        logger.debug("Applying gain at ADC level")

        if not hasattr(vrms_noise, "__len__"):
            vrms_noise = np.full_like(trigger_channels, vrms_noise, dtype=float)

        vrms_after_gain = []
        for channel_id, vrms in zip(trigger_channels, vrms_noise):
            det_channel = det.get_channel(station.get_id(), channel_id)
            noise_count = det_channel["trigger_adc_noise_count"]
            total_bits = det_channel["trigger_adc_nbits"]
            adc_input_range = det_channel["trigger_adc_max_voltage"] - det_channel["trigger_adc_min_voltage"]

            volts_per_adc = adc_input_range / (2 ** total_bits - 1)
            ideal_vrms = volts_per_adc * noise_count

            if self._adc_input_range is None:
                self._adc_input_range = adc_input_range
            else:
                assert self._adc_input_range == adc_input_range, "ADC input range is not consistent across channels"

            if self._nbits is None:
                self._nbits = total_bits
            else:
                assert self._nbits == total_bits, "ADC bits are not consistent across channels"


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

            # Apply gain
            channel = station.get_trigger_channel(channel_id)
            channel.set_trace(channel.get_trace() * gain_to_use, channel.get_sampling_rate())

            # Calculate the effective number of noise bits
            eff_noise_count = vrms_after_gain[-1] / volts_per_adc

            logger.debug("\t Ch {}: ampl. Vrms {:0.3f} ({:.3f}) mV (gain: {}, eff. noise count {:0.2f})".format(
                channel_id, np.std(channel.get_trace()) / units.mV, vrms_after_gain[-1] / units.mV, gain_to_use, eff_noise_count))
        logger.debug("Target Vrms: {:0.3f} mV; Target noise count: {}".format(ideal_vrms / units.mV, noise_count))

        return np.array(vrms_after_gain), ideal_vrms


    def digitize_trace(self, station, det, trigger_channels, vrms):
        """
        Digitizes the traces of the trigger channels.

        This function uses the `NuRadioReco.modules.analogToDigitalConverter` module to digitize the traces.
        The resulting digitized traces are either in discrete voltage values or in ADC counts
        (depeding on the argument `adc_output`).

        Parameters
        ----------
        station : Station
            Station to use
        det : Detector
            The detector description
        trigger_channels : list
            Channels that this function should be applied to
        vrms : float
            The (noise) RMS voltage of the trigger channels including the trigger board filters.
            This can be used to simulate a dynamic range of the ADC which depends on the noise level.
        """
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
                adc_output=self.adc_output,
                return_sampling_frequency=True,
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
        logger.debug("Applying the RNO-G trigger board response")

        if vrms is None:
            vrms = self.get_noise_vrms_per_trigger_channel(station, trigger_channels)

        if apply_adc_gain:
            equalized_vrms, ideal_vrms = self.apply_adc_gain(station, det, trigger_channels, vrms)
        else:
            equalized_vrms = vrms
            ideal_vrms = vrms

        if digitize_trace:
            self.digitize_trace(station, det, trigger_channels, ideal_vrms)
            if self.adc_output == "counts":
                lsb_voltage = self._adc_input_range / (2 ** self._nbits - 1)
                # We do not floor/convert the vrms to integers here. But this has to happen before the trigger.
                equalized_vrms = equalized_vrms / lsb_voltage
                logger.debug("obs. Vrms {} ADC".format(equalized_vrms))

        return equalized_vrms

    def end(self):
        pass