import numpy as np
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.detector import antennapattern
import time
import logging
from scipy import signal
from NuRadioReco.utilities import fft
logger = logging.getLogger('calculateAmplitudePerRaySolution')


class calculateAmplitudePerRaySolution:
    """
    Convolves the signal from each ray tracing solution with the antenna response
    and calculates the maximum signal amplitude
    """

    def __init__(self):
        self.__t = 0
        self.__debug = None
        self.antenna_provider = None
        self.begin()

    def begin(self, debug=False):
        """
        begin method, sets general parameters of module

        """
        self.__debug = debug
        self.antenna_provider = antennapattern.AntennaPatternProvider()

    @register_run()
    def run(self, evt, station, det):
        t = time.time()

        # access simulated efield and high level parameters
        sim_station = station.get_sim_station()
        sim_station_id = sim_station.get_id()

        # loop over all channels
        for efield in sim_station.get_electric_fields():
            # one efield might be valid for multiple channels, hence we loop over all channels this efiels is valid for,
            # convolve each trace with the antenna response for the given angles
            # and transform it to the time domain to calculate the max. amplitude
            channel_ids = efield.get_channel_ids()
            for channel_id in channel_ids:
                logger.debug('channel id {}'.format(channel_id))

                zenith = efield[efp.zenith]
                azimuth = efield[efp.azimuth]

                ff = efield.get_frequencies()
                efield_fft = efield.get_frequency_spectrum()

                # get antenna pattern for current channel
                antenna_model = det.get_antenna_model(sim_station_id, channel_id, zenith)
                antenna_pattern = self.antenna_provider.load_antenna_pattern(antenna_model, interpolation_method='complex')
                ori = det.get_antenna_orientation(sim_station_id, channel_id)
                logger.debug("zen {:.0f}, az {:.0f}".format(zenith / units.deg, azimuth / units.deg))
                VEL = antenna_pattern.get_antenna_response_vectorized(ff, zenith, azimuth, *ori)

                # Apply antenna response to electric field
                voltage_fft = efield_fft[2] * VEL['phi'] + efield_fft[1] * VEL['theta']

                # Remove DC offset
                voltage_fft[np.where(ff < 5 * units.MHz)] = 0.

                voltage = fft.freq2time(voltage_fft, efield.get_sampling_rate())
                h = np.abs(signal.hilbert(voltage))
                maximum = np.abs(voltage).max()
                maximum_envelope = h.max()

                if not efield.has_parameter(efp.max_amp_antenna):
                    efield[efp.max_amp_antenna] = {}
                    efield[efp.max_amp_antenna_envelope] = {}
                efield[efp.max_amp_antenna][channel_id] = maximum
                efield[efp.max_amp_antenna_envelope][channel_id] = maximum_envelope

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
