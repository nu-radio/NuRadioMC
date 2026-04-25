from NuRadioReco.modules.base.module import register_run
import numpy as np
import logging
from NuRadioReco.utilities import units
from NuRadioReco.detector.antennapattern import AntennaPatternProvider

logger = logging.getLogger('NuRadioReco.channelAntennaDedispersion')


class channelAntennaDedispersion:
    """
    The goal of this module is to unfold the group delay introduced by the antenna at a point where the
    direction of the incoming signal is not known. Hence, we choose the most sensitive direction (and polarisation) 
    for each antenna as a proxy. The goal is to improve the signal-to-noise ratio on the channel level.

    For a proper reconstruction of the electric field, please look at the voltage2EFieldConverter module.
    """
    def __init__(self):


        self._provider = AntennaPatternProvider()

        # relative to the antenna orientation
        self.antennas_most_sensitive_directions = {
            "LPDA": [0, 0], # the sensitive direction of an LPDA is the boresight direction
            "bicone": [90 * units.deg, 0], # the sensitive direction of a dipole is perpendicular to its orientation
            "vpol": [90 * units.deg, 0], # the sensitive direction of a dipole is perpendicular to its orientation
            "hpol": [0, 0], # the sensitive direction of a dipole is along it's direction
        }

    def _get_response(self, det, station_id, channel_id, ff):
        antenna_name = det.get_antenna_model(station_id, channel_id)
        antenna = self._provider.load_antenna_pattern(antenna_name)
        zen_ori, az_ori, zen_rot, az_rot = det.get_antenna_orientation(station_id, channel_id)

        match = next(
            (key_name for key_name in self.antennas_most_sensitive_directions if key_name.lower() in antenna_name.lower()), None
        )
        if match is None:
            raise AttributeError(f"Antenna name {antenna_name} can't be interpreted. "
                f"Available names are: {list(self.antennas_most_sensitive_directions.keys())}")

        zen = zen_ori + self.antennas_most_sensitive_directions[match][0]
        az = az_ori + self.antennas_most_sensitive_directions[match][1]

        VEL = antenna.get_antenna_response_vectorized(ff, zen, az, zen_ori, az_ori, zen_rot, az_rot)
        polarization = "phi"
        if np.sum(np.abs(VEL['theta'])) > np.sum(np.abs(VEL['phi'])):
            polarization = "theta"

        response = np.exp(1j * np.angle(VEL[polarization]))
        return response

    @register_run()
    def run(self, evt, station, det, add_ant=None, debug=False):
        if add_ant is not None:
            if not isinstance(add_ant, dict):
                raise ValueError("To add antennas provide a dictionary with antenna names as keys and direction arrays as values")
            for ant_name, direction in add_ant.items():
                if not isinstance(direction, (list, np.ndarray)) or len(direction) != 2:
                    raise ValueError(f"Antenna direction for {ant_name} requires zenith and azimuth")
                self.antennas_most_sensitive_directions[ant_name] = direction

        for channel in station.iter_channels():
            ff = channel.get_frequencies()
            response = self._get_response(det, station.get_id(), channel.get_id(), tuple(ff))

            if debug:
                from matplotlib import pyplot as plt

                trace = channel.get_trace()
                tt = channel.get_times()
                fig, ax = plt.subplots(1, 1)
                ax.plot(tt, trace)

            channel.set_frequency_spectrum(
                channel.get_frequency_spectrum() / response, sampling_rate=channel.get_sampling_rate())

            if debug:
                trace = channel.get_trace()
                ax.plot(tt, trace, '--')
                plt.show()

    def end(self):
        pass
