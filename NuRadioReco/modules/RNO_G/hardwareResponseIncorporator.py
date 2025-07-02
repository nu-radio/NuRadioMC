import NuRadioReco.modules.channelAddCableDelay
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units, fft
import NuRadioReco.framework.station

from NuRadioReco.detector.RNO_G import analog_components
from NuRadioReco.detector import detector

import numpy as np
import copy
import time
import logging


class hardwareResponseIncorporator:
    """
    Incorporates the compex response of the RNO-G hardware. The response is obtained from the detector description.
    The response is applied in the frequency domain.
    """

    def __init__(self):
        self.logger = logging.getLogger(
            "NuRadioReco.RNOG.hardwareResponseIncorporator")
        self.__time_delays = {}
        self.__t = 0
        self.__mingainlin = None
        self.trigger_channels = None
        self.channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()

    def begin(self, trigger_channels=None):
        """
        Parameters
        ----------
        trigger_channels: list of int
            List of channels for which an extra trigger channel with a different response is used. (Default: None)
        """
        self.trigger_channels = trigger_channels


    def get_filter(self, frequencies, station_id, channel_id, det,
                   temp=293.15, sim_to_data=False, phase_only=False,
                   mode=None, mingainlin=None, is_trigger=False):
        """
        Helper function to return the filter that the module applies.

        Parameters
        ----------

        frequencies: array of floats
            the frequency array for which the filter should be returned

        station_id: int
            the station id

        channel_id: int
            the channel id

        det: detector instance
            the detector

        temp: float
            temperature in Kelvin, better in the range [223.15 K , 323.15 K]

        sim_to_data: bool (default False)
            If False, deconvolve the hardware response.
            If True, convolve with the hardware response

        phase_only: bool (default False)
            if True, only the phases response is applied but not the amplitude response

        mode: {None, 'phase_only', 'relative'}, default None
            Options:

            * 'phase_only': only the phases response is applied but not the amplitude response
              (identical to phase_only=True )
            * 'relative': gain of amp is divided by maximum of the gain, i.e. at the maximum of the
              filter response is 1 (before applying cable response). This makes it easier to compare
              the filtered to unfiltered signal
            * None : default, gain and phase effects are applied 'normally'

        mingainlin: float
            In frequency ranges where the gain gets very small, the reconstruction of the original signal (obtained by
            dividing the measured signal by the gain) leads to excessively high values, due to the effect of
            post-amplifier noise. In order to mitigate this effect, a minimum gain (linear scale!) as fraction of the
            maximum gain can be defined. If specified, any gain value smaller than mingainlin will be replaced by mingainlin.

            Note: The adjustment to the minimal gain is NOT visible when getting the amp response from
            ``analog_components.get_amplifier_response()``

        is_trigger: bool
            Use trigger channel response instead. Only relevant for RNO-G. (Default: False)

        Returns
        -------
        array of complex floats
            the complex filter amplitudes
        """

        if isinstance(det, detector.rnog_detector.Detector):
            resp = det.get_signal_chain_response(station_id, channel_id, is_trigger)
            amp_response = resp(frequencies)
        elif isinstance(det, detector.detector_base.DetectorBase):
            amp_type = det.get_amplifier_type(station_id, channel_id)
            # it reads the log file. change this to load_amp_measurement if you want the RI file
            amp_response = analog_components.load_amp_response(amp_type)
            amp_response = amp_response['gain'](
                frequencies, temp) * amp_response['phase'](frequencies)
        else:
            raise NotImplementedError("Detector type not implemented")

        if mingainlin is not None:
            mingainlin = float(mingainlin)
            ampmax = np.max(np.abs(amp_response))
            iamp_gain_low = np.where(
                np.abs(amp_response) < (mingainlin * ampmax))
            amp_response[iamp_gain_low] = (
                mingainlin * ampmax) * np.exp(1j * np.angle(amp_response[iamp_gain_low]))

        cable_response = 1
        if mode is None:
            pass
        elif mode == 'phase_only':
            cable_response = np.ones_like(cable_response) * np.exp(1j * np.angle(cable_response))
            amp_response = np.ones_like(amp_response) * np.exp(1j * np.angle(amp_response))
        elif mode == 'relative':
            ampmax = np.max(np.abs(amp_response))
            amp_response /= ampmax
        else:
            raise NotImplementedError(f"Operating mode \"{mode}\" has not been implemented.")

        if sim_to_data:
            return amp_response * cable_response
        else:
            return 1. / (amp_response * cable_response)

    @register_run()
    def run(self, evt, station, det, temp=293.15, sim_to_data=False, phase_only=False, mode=None, mingainlin=None):
        """
        Switch sim_to_data to go from simulation to data or otherwise.
        The option zero_noise can be used to zero the noise around the pulse. It is unclear, how useful this is.

        Parameters
        ----------
        evt: Event
            Event to run the module on

        station: Station
            Station to run the module on

        det: Detector
            The detector description

        temp: temperature in Kelvin, better in the range [223.15 K , 323.15 K]

        sim_to_data: bool (default False)

            * if False, deconvolve the hardware response
            * if True, convolve with the hardware response

        phase_only: bool (default False)
            if True, only the phases response is applied but not the amplitude response

        mode: string or None, default None
            Options:

            * 'phase_only': only the phases response is applied but not the amplitude response
              (identical to phase_only=True)
            * 'relative': gain of amp is divided by maximum of the gain, i.e. at the maximum of the
              filter response is 1 (before applying cable response). This makes it easier to compare
              the filtered to unfiltered signal
            * None: default, gain and phase effects are applied 'normally'

        mingainlin: float
            In frequency ranges where the gain gets very small, the reconstruction of the original signal (obtained by
            dividing the measured signal by the gain) leads to excessively high values, due to the effect of
            post-amplifier noise. In order to mitigate this effect, a minimum gain (linear scale!) as fraction of the
            maximum gain can be defined. If specified, any gain value smaller than mingainlin will be replaced by mingainlin.

            Note: The adjustment to the minimal gain is NOT visible when getting the amp response from
            ``analog_components.get_amplifier_response()``
        """

        self.__mingainlin = mingainlin
        if phase_only:
            mode = 'phase_only'
            self.logger.warning(
                'Please use option mode=''phase_only'' in the future, use of option phase_only will be phased out')

        t = time.time()

        if self.trigger_channels is not None and not isinstance(det, detector.rnog_detector.Detector):
            raise ValueError("Simulating extra trigger channels is only possible with the `rnog_detector.Detector` class.")

        has_trigger_channels = False
        for channel in station.iter_channels():
            frequencies = channel.get_frequencies()
            trace_fft = channel.get_frequency_spectrum()

            filter = self.get_filter(
                frequencies, station.get_id(), channel.get_id(), det, temp, sim_to_data, phase_only, mode, mingainlin)

            if (self.trigger_channels is not None and
                channel.get_id() in self.trigger_channels and
                isinstance(station, NuRadioReco.framework.station.Station)):
                """
                Create a copy of the channel and apply the readout and trigger channel response respectively.
                We do this here under the assumption that up to this point no difference between the two channels
                had to be made. This is acutally not strictly true. The cable delay is already added in the
                efieldToVoltageConverter module. I.e., the assumption is made that the cable delay is no different
                between the two. While this might be true/a good approximation for the moment it is not given that
                this holds for the future. You have been warned!

                See Also: https://nu-radio.github.io/NuRadioMC/NuRadioReco/pages/event_structure.html#channel for a bit more context.
                """
                trig_filter = self.get_filter(
                    frequencies, station.get_id(), channel.get_id(), det, temp, sim_to_data,
                    phase_only, mode, mingainlin, is_trigger=True)

                trig_trace_fft = trace_fft * trig_filter
                # zero first bins to avoid DC offset
                trig_trace_fft[0] = 0

                # Add trigger channel
                trig_channel = copy.deepcopy(channel)
                trig_channel.set_frequency_spectrum(
                    trig_trace_fft, channel.get_sampling_rate())

                channel.set_trigger_channel(trig_channel)
                has_trigger_channels = True

            trace_fft *= filter
            # zero first bins to avoid DC offset
            trace_fft[0] = 0

            # hardwareResponse incorporator should always be used in conjunction with bandpassfilter
            # otherwise, noise will be blown up
            channel.set_frequency_spectrum(
                trace_fft, channel.get_sampling_rate())

        if not sim_to_data:
            if not evt.has_been_processed_by_module('channelAddCableDelay', station.get_id()):
                self.logger.warning(
                    "The hardwareResponseIncorporator module should _not_ be used to remove the cable delay "
                    "from data anymore. Please use channelAddCableDelay module for this (before running "
                    "the hardwareResponseIncorporator module). The channelAddCableDelay was not applied "
                    "to this event, hence, you are receiving this warning. The cable delay is now "
                    "removed. Please add the channelAddCableDelay module to your processing chain "
                    "to avoid this warning in the future (in that case the cable delay will not be "
                    "removed by this module).")

                # Subtraces the cable delay. For `sim_to_data=True`, the cable delay is added
                # in the efieldToVoltageConverter or with the channelCableDelayAdder
                # (if efieldToVoltageConverterPerEfield was used).
                self.channelAddCableDelay.run(evt, station, det, mode='subtract')

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        return dt

    def __calculate_time_delays_amp(self, amp_type):
        """
        helper function to calculate the time delay of the amp for a delta pulse
        """
        amp_response_f = analog_components.load_amp_response(amp_type)
        # assume a huge sampling rate to have a good time resolution
        sampling_rate = 10 * units.GHz
        n = 2 ** 12
        trace = np.zeros(n)
        trace[n // 2] = 1
        max_time = trace.argmax() / sampling_rate
        spec = fft.time2freq(trace, sampling_rate)
        ff = np.fft.rfftfreq(n, 1. / sampling_rate)
        amp_response_gain = amp_response_f['gain'](ff)
        amp_response_phase = amp_response_f['phase'](ff)
        mask = (ff < 70 * units.MHz) & (ff > 40 * units.MHz)
        spec[~mask] = 0
        trace2 = fft.freq2time(
            spec * amp_response_gain * amp_response_phase, sampling_rate)
        max_time2 = np.abs(trace2).argmax() / sampling_rate
        return max_time2 - max_time

    def get_time_delay(self, amp_type):
        if amp_type not in self.__time_delays.keys():
            # not yet calculated -> calculate the time delay
            self.__time_delays[amp_type] = self.__calculate_time_delays_amp(
                amp_type)
            self.logger.info(
                "time delays of amp {} have not yet been calculated -> calculating -> time delay is {:.2f} ns".format(
                    amp_type, self.__time_delays[amp_type] / units.ns))

        return self.__time_delays[amp_type]

    def get_mingainlin(self):
        return self.__mingainlin


if __name__ == "__main__":
    import os
    import datetime
    import matplotlib.pyplot as plt

    file_dir = os.path.dirname(__file__)

    detectorfile = os.path.join(
        file_dir, "../../detector/RNO_G/RNO_single_station.json")
    det_old = detector.generic_detector.GenericDetector(
        json_filename=detectorfile,
        default_station=11, antenna_by_depth=False)

    det = detector.rnog_detector.Detector(log_level=logging.DEBUG, over_write_handset_values={
        "sampling_frequency": 2.4 * units.GHz}, always_query_entire_description=True)
    det.update(datetime.datetime(2022, 8, 2, 0, 0))

    hri = hardwareResponseIncorporator()

    frequencies = np.linspace(0, 1) * units.GHz
    filter_old = hri.get_filter(
        frequencies, station_id=11, channel_id=0, det=det_old, sim_to_data=True)
    filter = hri.get_filter(frequencies, station_id=11,
                            channel_id=0, det=det, sim_to_data=True)

    fig, ax = plt.subplots()

    ax.plot(frequencies, np.abs(filter_old), label="old detector class")
    ax.plot(frequencies, np.abs(filter), label="new detector class")

    ax.set_yscale("log")
    ax.set_xlabel("frequency / GHz")
    ax.grid(which="both")
    ax.set_ylim(1e-2, 5e3)
    ax.legend()
    fig.tight_layout()
    plt.show()
