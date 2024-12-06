from NuRadioReco.detector.ARA import analog_components
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
import numpy as np
import time
import logging

logger = logging.getLogger("NuRadioReco.ARA.hardwareResponseIncorporator")


class hardwareResponseIncorporator:
    """
    Incorporates the gain and phase induced by the ARA hardware.


    """

    def __init__(self):
        self.__debug = False
        self.__time_delays = {}
        self.__t = 0
        self.begin()

    def begin(self, debug=False):
        self.__debug = debug

    def add_cable_delay(self, station, det, channel, sim_to_data):
        """
        Add or subtract cable delay to a channel.

        Parameters
        ----------
        station: Station
            The station to add the cable delay to.

        det: Detector
            The detector description

        channel: Channel
            The channel to add the cable delay to.

        sim_to_data: bool
            If True, the cable delay is added. If False, the cable delay is subtracted.
        """
        cable_delay = det.get_cable_delay(station.get_id(), channel.get_id())

        if sim_to_data:
            channel.add_trace_start_time(cable_delay)
            self.logger.debug(f"Add {cable_delay / units.ns:.2f}ns "
                            f"of cable delay to channel {channel.get_id()}")

        else:
            channel.add_trace_start_time(-cable_delay)
            self.logger.debug(f"Subtract {cable_delay / units.ns:.2f}ns "
                            f"of cable delay to channel {channel.get_id()}")

    @register_run()
    def run(self, evt, station, det, sim_to_data=False):
        """
        Switch sim_to_data to go from simulation to data or otherwise.
        """
        t = time.time()
        channels = station.iter_channels()

        for channel in channels:

            frequencies = channel.get_frequencies()
            system_response = analog_components.get_system_response(frequencies)
            trace_fft = channel.get_frequency_spectrum()

            if sim_to_data:

                trace_after_system_fft = trace_fft * system_response['gain'] * system_response['phase']
                # zero first bins to avoid DC offset
                trace_after_system_fft[0] = 0
                channel.set_frequency_spectrum(trace_after_system_fft, channel.get_sampling_rate())

            else:
                trace_before_system_fft = np.zeros_like(trace_fft)
                trace_before_system_fft[np.abs(system_response['gain']) > 0] = trace_fft[np.abs(system_response['gain']) > 0] / (system_response['gain'] * system_response['phase'])[np.abs(system_response['gain']) > 0]
                channel.set_frequency_spectrum(trace_before_system_fft, channel.get_sampling_rate())

            self.add_cable_delay(station, det, channel, sim_to_data)

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
