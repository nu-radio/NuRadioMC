from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.analogToDigitalConverter import analogToDigitalConverter
from NuRadioReco.utilities import units, fft

import numpy as np
import logging
logger = logging.getLogger("NuRadioReco.triggerBoardResponse")

main_low_angle = np.deg2rad(-60)
main_high_angle = np.deg2rad(60)
phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 12))
sampling_rate = 472 * units.MHz

#define the keyword arguments for the trigger modules here. these settings emulate the trigger firmware
#the keys should only appear here and not in as named keyword arguments!
powerTriggerKwargs={
                "phasing_angles": phasing_angles,
                "ref_index": 1.75,
                "trigger_adc": True,
                "trigger_filter": None,
                "apply_digitization": False,
                "adc_output": "counts",
                "upsampling_factor": 4,
                "upsampling_method": "fir",
                "window": 24,
                "step": 8,
                "saturation_bits": 8,
                "filter_taps":45,
                "coeff_gain": 256 }

envelopeTriggerKwargs={
                "phasing_angles": phasing_angles,
                "ref_index": 1.75,
                "trigger_adc": True,
                "trigger_filter": None,
                "apply_digitization": False,
                "adc_output": "counts",
                "upsampling_factor": 1,
                "upsampling_method": "fir",
                "saturation_bits": 8,
                "filter_taps":45,
                "coeff_gain": 256,
                "ideal_transformer": False}

highLowTriggerKwargs={
                        "high_low_window": 6 / sampling_rate,
                        "coinc_window": 20 / sampling_rate,
                        "number_concidences": 2,
                        "step": 4}


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

    def apply_adc_gain(self, station, det, trigger_channels, vrms_noise=None, gain_values=None):
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
        gain_values : list (default: None)
            If set these will be applied to the channel. Otherwise gains are recalculated.

        Returns
        -------
        vrms_after_gain : list
            the RMS voltage of the waveforms after the gain has been applied
        ideal_vrms: float
            the ideal vrms, as measured on the ADC capacitors
        ret_gain_values: list
            gain values applied to each channel
        """

        if vrms_noise is None:
            vrms_noise = self.get_noise_vrms_per_trigger_channel(station, trigger_channels)
            logger.debug("obs. Vrms {} mV".format(np.around(vrms_noise / units.mV, 3)))

        logger.debug("Applying gain at ADC level")

        if not hasattr(vrms_noise, "__len__"):
            vrms_noise = np.full_like(trigger_channels, vrms_noise, dtype=float)

        vrms_after_gain = []
        ret_gain_values = []
        for channel_id, vrms in zip(trigger_channels, vrms_noise):
            det_channel = det.get_channel(station.get_id(), channel_id)
            noise_count = det_channel["trigger_adc_noise_count"]
            total_bits = det_channel["trigger_adc_nbits"]
            adc_input_range = det_channel["trigger_adc_max_voltage"] - det_channel["trigger_adc_min_voltage"]

            volts_per_adc = adc_input_range / (2 ** total_bits - 1)
            ideal_vrms = volts_per_adc * noise_count

            if gain_values is not None:
                vrms_after_gain.append(vrms * gain_values[channel_id])
                channel = station.get_trigger_channel(channel_id)
                channel.set_trace(channel.get_trace() * gain_values[channel_id], channel.get_sampling_rate())
                ret_gain_values.append(gain_values[channel_id])

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

            ret_gain_values.append(gain_to_use)

        return np.array(vrms_after_gain), np.array(vrms_after_gain)/volts_per_adc, ideal_vrms, np.array(ret_gain_values)

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
                trigger_filter=None,
                clock_offset=self._clock_offset,
                adc_output=self.adc_output,
                return_sampling_frequency=True,
            )

            channel.set_trace(digitized_trace, adc_sampling_frequency)

    @register_run()
    def run(self, evt, station, det, trigger_channels, vrms=None, apply_adc_gain=True,
            digitize_trace=True, gain_values=None, recalculate_rms=False):
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
        ret_gain_values : list
            the gain values applied to each channel
        """
        logger.debug("Applying the RNO-G trigger board response")

        if vrms is None:

            vrms = self.get_noise_vrms_per_trigger_channel(station, trigger_channels)

        if apply_adc_gain:
            equalized_vrms, equalized_adc_vrms, ideal_vrms, ret_gain_values = self.apply_adc_gain(station, det, trigger_channels, vrms, gain_values)
        else:
            equalized_vrms = vrms
            ideal_vrms = vrms
            ret_gain_values = None

        if digitize_trace:
            self.digitize_trace(station, det, trigger_channels, ideal_vrms)
            if self.adc_output == "counts":
                lsb_voltage = self._adc_input_range / (2 ** self._nbits - 1)
                # We do not floor/convert the vrms to integers here. But this has to happen before the trigger.
                equalized_vrms = equalized_vrms / lsb_voltage
                logger.debug("obs. Vrms {} ADC".format(equalized_vrms))

        return equalized_vrms, ret_gain_values

    def end(self):
        pass

if __name__=='__main__':

    from NuRadioReco.detector import detector
    import NuRadioReco.modules.channelGenericNoiseAdder
    import NuRadioReco.modules.channelBandPassFilter
    from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator
    import matplotlib.pyplot as plt
    from scipy import constants

    rnogHarwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()
    rnogHarwareResponse.begin(trigger_channels=[0,1,2,3])
    rnogADCResponse = triggerBoardResponse(log_level=logging.ERROR)
    rnogADCResponse.begin(adc_input_range=2 * units.volt, clock_offset=0.0, adc_output="counts")
    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

    det_file = 'RNO_G/RNO_single_station_only_PA.json'
    det = detector.Detector(source='json',json_filename=det_file)

    station_id = 11
    channel_ids = np.arange(4)

    n_samples  = 1024
    sampling_rate = 472 * units.MHz
    dt = 1 / sampling_rate
    ff = np.fft.rfftfreq(n_samples, dt)
    max_freq = ff[-1]
    min_freq = 0
    fff = np.linspace(min_freq, max_freq, 10000)

    four_filters_highres = {}
    rf_filter_highres = rnogHarwareResponse.get_filter(fff, station_id, channel_ids[0], det,sim_to_data=True, is_trigger=True)
    chain_filter_highres = rf_filter_highres

    for i in channel_ids:
        four_filters_highres[i] = chain_filter_highres
    Vrms = 1
    noise_temp = 300
    bandwidth = {}
    Vrms_ratio = {}
    amplitude = {}
    per_channel_vrms = []

    for i in channel_ids:
        integrated_channel_response = np.trapz(np.abs(four_filters_highres[i]) ** 2, fff)
        rel_channel_response = np.trapz(np.abs(four_filters_highres[i]) ** 2, fff)
        bandwidth[i] = integrated_channel_response
        Vrms_ratio[i] = np.sqrt(rel_channel_response / (max_freq - min_freq))
        chan_vrms = (noise_temp * 50 * constants.k * integrated_channel_response / units.Hz) ** 0.5
        per_channel_vrms.append(chan_vrms)
        amplitude[i] = chan_vrms / Vrms_ratio[i]

    station = NuRadioReco.framework.station.Station(station_id)
    evt = NuRadioReco.framework.event.Event(0, 0)

    channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
    channelGenericNoiseAdder.begin()

    for channel_id in channel_ids:

        spectrum = channelGenericNoiseAdder.bandlimited_noise(min_freq, max_freq, n_samples, sampling_rate, amplitude[channel_id],
                                                              type="rayleigh", time_domain=False)

        trace = fft.freq2time(spectrum * four_filters_highres[channel_id], sampling_rate)

        channel = NuRadioReco.framework.channel.Channel(channel_id)
        channel.set_trace(trace, sampling_rate)
        station.add_trigger_channel(channel)

    fig , ax  = plt.subplots(1, 2, sharex=True, figsize=(11,7))

    for channel_id in channel_ids:
        ch = station.get_channel(channel_id)
        ax[0].plot(ch.get_times(), ch.get_trace(), label='ch %i' % channel_id)

    chan_rms, gain_values = rnogADCResponse.run(evt, station, det, requested_channels=channel_ids,
                                 digitize_trace=True, apply_adc_gain=True)

    for channel_id in channel_ids:
        ch = station.get_channel(channel_id)
        ax[1].plot(ch.get_times(), ch.get_trace(), label='ch %i' % channel_id)

    ax[0].set_title('Raw Voltage Trace')
    ax[0].set_ylabel('Voltage [V]')
    ax[0].set_xlabel('Samples')
    ax[0].legend(loc='upper right')
    ax[1].set_xlabel('Samples')
    ax[1].set_ylabel('Voltage [ADC]')
    ax[1].legend(loc='upper right')
    ax[1].set_title('Gain Eq. and Digitized Trace')
    fig.tight_layout()
    plt.show()
