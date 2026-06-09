from NuRadioReco.detector.ARA import analog_components
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.channelAddCableDelay import add_cable_delay

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

    def get_filter(self, frequencies, station_id=None, channel_id=None, det=None, sim_to_data=False):
    """
    Helper function to return the filter that the module applies. Loads entire system response from interpolating data of ARA_Electronics_TotalGain_TwoFilters.txt.

    Parameters
    ----------

    frequencies: array of floats
        the frequency array for which the filter should be returned

    station_id: int (default None)
        the station id

    channel_id: int (default None)
        the channel id

    det: detector instance (default None)
        the detector

    sim_to_data: bool (default False)
        If False, deconvolve the hardware response.
        If True, convolve with the hardware response

    Returns
    -------
    array of complex floats
        the complex filter amplitudes
    """

    if station_id is not None or channel_id is not None or det is not None:
        logger.warning("get_filter() warning: station_id/channel_id/det provided but not used."
                       "ARA system response is not channel-dependent")

    analog_components.load_system_response()
    system_response = analog_components.get_system_response(frequencies)
    system_complex_response = system_response['gain'] * system_response['phase']

    if sim_to_data:
        return system_complex_response
    else:
        return 1. / system_complex_response
        
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
                trace_before_system_fft[np.abs(system_response['gain']) > 0] = (
                    trace_fft[np.abs(system_response['gain']) > 0] /
                    (system_response['gain'] * system_response['phase'])[np.abs(system_response['gain']) > 0]
                )
                channel.set_frequency_spectrum(trace_before_system_fft, channel.get_sampling_rate())

        if not sim_to_data:
            # Subtraces the cable delay. For `sim_to_data=True`, the cable delay is added
            # in the efieldToVoltageConverter or with the channelCableDelayAdder
            # (if efieldToVoltageConverterPerEfield was used).
            add_cable_delay(station, det, sim_to_data=False, logger=self.logger)

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
