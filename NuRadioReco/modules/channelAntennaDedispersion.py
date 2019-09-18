from NuRadioReco.modules.base.module import register_run
import numpy as np
import copy
import logging
from NuRadioReco.utilities import units
from scipy import signal
from NuRadioReco.detector.antennapattern import AntennaPatternProvider
from functools import lru_cache

logger = logging.getLogger('channelAntennaDedispersion')


class channelAntennaDedispersion:
    """
    
    """

    def __init__(self):

        self._provider = AntennaPatternProvider()

#     @lru_cache(maxsize=32)  # hashing doen't make sense because ff array is always different
    def _get_response(self, det, station_id, channel_id, ff):
        antenna_name = det.get_antenna_model(station_id, channel_id)
        antenna = self._provider.load_antenna_pattern(antenna_name)
        zen_ori, az_ori, zen_rot, az_rot = det.get_antenna_orientation(station_id, channel_id)
        zen, az = None, None
        if("LPDA" in antenna_name):
            zen = zen_ori
            az = az_ori
        elif("bicone" in antenna_name or "dipole" in antenna_name):
            zen = 90 * units.deg + zen_ori
            az = 0
        else:
            raise AttributeError(f"antenna name {antenna_name} can't be interpreted")
        VEL = antenna.get_antenna_response_vectorized(ff, zen, az, zen_ori, az_ori, zen_rot, az_rot)
        polarization = "phi"
        if(np.sum(np.abs(VEL['theta'])) > np.sum(np.abs(VEL['phi']))):
            polarization = "theta"
        response = np.exp(1j * np.angle(VEL[polarization]))
#         fig, ax = plt.subplots(1, 1)
#         ax.plot(ff / units.MHz, np.abs(VEL['theta']), label='theta')
#         ax.plot(ff / units.MHz, np.abs(VEL['phi']), label='phi')
#         ax.set_title(f"{antenna_name} - {polarization}")
#         ax.legend()
#         plt.show()
        return response

    @register_run()
    def run(self, evt, station, det):
        """
        parameters
        ----------
        """
        for channel in station.iter_channels():
            ff = channel.get_frequencies()
            response = self._get_response(det, station.get_id(), channel.get_id(), tuple(ff))

#             trace = channel.get_trace()
#             tt = channel.get_times()
#             from matplotlib import pyplot as plt
#             fig, ax = plt.subplots(1, 1)
#             ax.plot(tt, trace)
            channel.set_frequency_spectrum(channel.get_frequency_spectrum() / response, sampling_rate=channel.get_sampling_rate())
#             trace = channel.get_trace()
#             ax.plot(tt, trace, '--')
#             plt.show()

#             sampling_rate = channel.get_sampling_rate()
#             window = signal.tukey(len(trace), filter_size)
#             trace *= window
#             trace = np.append(np.zeros(np.int(np.round(prepend * sampling_rate))), trace)
#             trace = np.append(trace, np.zeros(np.int(np.round(append * sampling_rate))))
#             channel.set_trace(trace, sampling_rate)

    def end(self):
        pass
