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
    
        system_response = analog_components.get_system_response(frequencies)
        system_complex_response = system_response['gain'] * system_response['phase']

        if sim_to_data:
            return system_complex_response
        else:
            filt = np.zeros_like(system_complex_response)
            mask = np.abs(system_complex_response) > 0
            filt[mask] = 1. / system_complex_response[mask]
            return filt
        
    @register_run()
    def run(self, evt, station, det, sim_to_data=False):
        """
        Switch sim_to_data to go from simulation to data or otherwise.
        """
        t = time.time()
        channels = station.iter_channels()

        for channel in channels:

            frequencies = channel.get_frequencies()
            trace_fft = channel.get_frequency_spectrum()

            filt = self.get_filter(frequencies, station.get_id(), channel.get_id(), det, sim_to_data=sim_to_data)
            trace_after_system_fft = trace_fft * filt
            
            if sim_to_data:
                # zero first bins to avoid DC offset
                trace_after_system_fft[0] = 0

            channel.set_frequency_spectrum(trace_after_system_fft, channel.get_sampling_rate())

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
