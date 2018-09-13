import numpy as np
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from NuRadioReco.utilities import ice
from NuRadioReco.framework.parameters import channelParameters as chp
# from detector import antennamodel
from NuRadioReco.detector import antennapattern
from radiotools import coordinatesystems
import copy
import time
import logging
import fractions
from scipy import signal
from decimal import Decimal
import NuRadioReco.framework.channel
from NuRadioMC.utilities import fft
logger = logging.getLogger('calculateAmplitudePerRaySolution')


class calculateAmplitudePerRaySolution:
    """
    Convolves the signal from each ray tracing solution with the antenna response
    and calculates the maximum signal amplitude
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self, debug=False):
        """
        begin method, sets general parameters of module

        """
        self.__debug = debug
        self.antenna_provider = antennapattern.AntennaPatternProvider()

    def run(self, evt, station, det):
        t = time.time()

        # access simulated efield and high level parameters
        sim_station = station.get_sim_station()
        sim_station_id = sim_station.get_id()
        event_time = sim_station.get_station_time()

        # loop over all channels
        for sim_channel in sim_station.iter_channels():

            # one channel might contain multiple channels to store the signals from multiple ray paths,
            # so we loop over all simulated channels with the same id,
            # convolve each trace with the antenna response for the given angles
            # and everything up in the time domain
            channel_id = sim_channel[0].get_id()
            logger.debug('channel id {}'.format(channel_id))
            channel = NuRadioReco.framework.channel.Channel(channel_id)
            channel_spectrum = None
            if(self.__debug):
                from matplotlib import pyplot as plt
                fig, axes = plt.subplots(2, 1)
            for sim_channel2 in sim_channel:
                channel_id = sim_channel2.get_id()

                zenith = sim_channel2[chp.zenith]
                azimuth = sim_channel2[chp.azimuth]
                
                ff = sim_channel2.get_frequencies()
                efield_fft = sim_channel2.get_frequency_spectrum()

                # get antenna pattern for current channel
                antenna_model = det.get_antenna_model(sim_station_id, channel_id, zenith)
                antenna_pattern = self.antenna_provider.load_antenna_pattern(antenna_model, interpolation_method='complex')
                ori = det.get_antanna_orientation(sim_station_id, channel_id)
                logger.debug("zen {:.0f}, az {:.0f}".format(zenith / units.deg, azimuth / units.deg))
                VEL = antenna_pattern.get_antenna_response_vectorized(ff, zenith, azimuth, *ori)

                # Apply antenna response to electric field
                voltage_fft = efield_fft[2] * VEL['phi'] + efield_fft[1] * VEL['theta']

                # Remove DC offset
                voltage_fft[np.where(ff < 5 * units.MHz)] = 0.
                
                voltage = fft.freq2time(voltage_fft)
                h = np.abs(signal.hilbert(voltage))
                maximum = np.abs(voltage).max()
                maximum_envelope = h.max()
                
                sim_channel2.set_parameter(chp.maximum_amplitude, maximum)
                sim_channel2.set_parameter(chp.maximum_amplitude_envelope, maximum_envelope)
                
        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
