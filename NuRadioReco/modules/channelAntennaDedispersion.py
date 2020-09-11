from NuRadioReco.modules.base.module import register_run
import numpy as np
import logging
from NuRadioReco.utilities import units
from NuRadioReco.detector.antennapattern import AntennaPatternProvider

logger = logging.getLogger('channelAntennaDedispersion')


class channelAntennaDedispersion:

    def __init__(self):

        self._provider = AntennaPatternProvider()

#     @lru_cache(maxsize=32)  # hashing doen't make sense because ff array is always different
    def _get_response(self, det, station_id, channel_id, ff):
        antenna_name = det.get_antenna_model(station_id, channel_id)
        antenna = self._provider.load_antenna_pattern(antenna_name)
        zen_ori, az_ori, zen_rot, az_rot = det.get_antenna_orientation(station_id, channel_id)
        if("LPDA" in antenna_name):
            zen = zen_ori  # the sensitive direction of an LPDA is the boresight direction
            az = az_ori
        elif("bicone" in antenna_name or "dipole" in antenna_name):
            zen = 90 * units.deg + zen_ori  # the sensitive direction of a dipole is perpendicular to its orientatoin
            az = 0
        else:
            raise AttributeError(f"antenna name {antenna_name} can't be interpreted")
        VEL = antenna.get_antenna_response_vectorized(ff, zen, az, zen_ori, az_ori, zen_rot, az_rot)
        polarization = "phi"
        if(np.sum(np.abs(VEL['theta'])) > np.sum(np.abs(VEL['phi']))):
            polarization = "theta"
        response = np.exp(1j * np.angle(VEL[polarization]))
        return response

    @register_run()
    def run(self, evt, station, det, debug=False):
        """
        parameters
        ----------
        """
        for channel in station.iter_channels():
            ff = channel.get_frequencies()
            response = self._get_response(det, station.get_id(), channel.get_id(), tuple(ff))
            if debug:
                trace = channel.get_trace()
                tt = channel.get_times()
                from matplotlib import pyplot as plt
                fig, ax = plt.subplots(1, 1)
                ax.plot(tt, trace)
            channel.set_frequency_spectrum(channel.get_frequency_spectrum() / response, sampling_rate=channel.get_sampling_rate())
            if debug:
                trace = channel.get_trace()
                ax.plot(tt, trace, '--')
                plt.show()

    def end(self):
        pass
