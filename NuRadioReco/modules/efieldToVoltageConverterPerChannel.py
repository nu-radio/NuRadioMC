import numpy as np
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from NuRadioReco.utilities import ice
# from detector import antennamodel
from NuRadioReco.detector import antennapattern
from radiotools import coordinatesystems
import copy
import time
import logging
import NuRadioReco.framework.channel
logger = logging.getLogger('efieldToVoltageConverterPerChannel')


class efieldToVoltageConverterPerChannel:
    """
    Module that should be used to convert simulations to data.
    It assumes that at least one efield is given per channel as input. It will
    convolve the electric field with the corresponding antenna response for the
    incoming direction specified in the channel object.
    The station id, defines antenna location and antenna type.
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self, debug=False, uncertainty={}):
        self.__debug = debug
        self.__uncertainty = uncertainty
        # some uncertainties are systematic, fix them here
        if('sys_dx' in self.__uncertainty):
            self.__uncertainty['sys_dx'] = np.random.normal(0, self.__uncertainty['sys_dx'])
        if('sys_dy' in self.__uncertainty):
            self.__uncertainty['sys_dy'] = np.random.normal(0, self.__uncertainty['sys_dy'])
        if('sys_dz' in self.__uncertainty):
            self.__uncertainty['sys_dz'] = np.random.normal(0, self.__uncertainty['sys_dz'])
        if('sys_amp'in self.__uncertainty):
            for iCh in self.__uncertainty['sys_amp'].keys():
                self.__uncertainty['sys_amp'][iCh] = np.random.normal(1, self.__uncertainty['sys_amp'][iCh])
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
            channel = NuRadioReco.framework.channel.Channel(channel_id)
            channel_spectrum = None
            for sim_channel2 in sim_channel:
                channel_id = sim_channel2.get_id()
                ff = sim_channel2.get_frequencies()
                efield_fft = sim_channel2.get_frequency_spectrum()

                zenith = sim_channel2['zenith']
                azimuth = sim_channel2['azimuth']

                # get antenna pattern for current channel
                antenna_model = det.get_antenna_model(sim_station_id, channel_id, zenith)
                antenna_pattern = self.antenna_provider.load_antenna_pattern(antenna_model)
                ori = det.get_antanna_orientation(sim_station_id, channel_id)
                VEL = antenna_pattern.get_antenna_response_vectorized(ff, zenith, azimuth, *ori)

                # Apply antenna response to electric field
                voltage_fft = efield_fft[2] * VEL['phi'] + efield_fft[1] * VEL['theta']

                # Remove DC offset
                voltage_fft[np.where(ff < 5 * units.MHz)] = 0.
                if('amp' in self.__uncertainty):
                    voltage_fft *= np.random.normal(1, self.__uncertainty['amp'][channel_id])
                if('sys_amp' in self.__uncertainty):
                    voltage_fft *= self.__uncertainty['sys_amp'][channel_id]

                if(channel_spectrum is None):
                    channel_spectrum = voltage_fft
                else:
                    channel_spectrum += voltage_fft
            channel.set_frequency_spectrum(channel_spectrum, sim_channel2.get_sampling_rate())

            station.add_channel(channel)
        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
