from NuRadioReco.detector.ARA import analog_components
from NuRadioReco.modules.base.module import register_run
import numpy as np
from NuRadioReco.utilities import units
import time
import logging

logger = logging.getLogger("hardwareResponseIncorporator")


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

    @register_run()
    def run(self, evt, station, det, sim_to_data=False):
        """
        Switch sim_to_data to go from simulation to data or otherwise.
        """
        t = time.time()
        station_id = station.get_id()
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

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt

