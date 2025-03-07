from NuRadioReco.utilities import units

from collections import defaultdict
import logging
import numpy as np
import json
import os


class Detector:

    def __init__(
            self, position_path=None, channel_file=None,
            detector_altitude=460 * units.m, maximum_radius=600 * units.m):
        """
        Simple class to describe an ideal SKA detector.

        All individual channels (= single antenna) within a receiver unit (= dual polarised antenna)
        are described with a JSON file. The channels within one receiver, i.e., those with the same 
        unique position have the same `channel_group_id`. In the context of this class the 
        `channel_group_id` is also referred to as `antenna_id`. Each channel has a unique ID.
        The channels' IDs are constructed as follows: `channel_group_id * 10 + pol`
        `pol` is (currently) either 0 or 1 for the first or second polarization respectively. 

        As of February 2025, the SKA positions are specified in a set of files following a specific
        directory structure. In the directory there should be a layout.txt file which contains the
        positions of the stations. For each station there should be a subdirectory which also contains
        a layout.txt file with the positions of the antennas. The `position_path` should point to the
        root directory of this structure.

        Parameters
        ----------
        position_path: str (Default: None)
            Path to the directory which contains the layout.txt files with the station positions and
            subdirectories for each station, as explained above. If None, the detector is left empty
            and the positions will have to be added manually.
        channel_file: str (Default: None)
            Path to the JSON file which contains the channel information. If None,
            the default file "ska_channels.json" in the same directory as this file
            is used.
        detector_altitude: float (Default: 460 * units.m)
            Altitude of the detector in meters.
        maximum_radius: float (Default: 600 * units.m)
            Maximum radius of stations to be included when reading from file.
        """

        self.logger = logging.getLogger("NuRadioReco.detector.SKA.detector")
        self.detector_altitude = detector_altitude
        self.maximum_radius = maximum_radius

        if channel_file is None:
            channel_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "ska_channels.json"
            )

        with open(channel_file, "r") as f:
            channel_data_json = json.load(f)["channels"]

        self.ref_channel_ids = []
        self.channel_data = {}
        for cinfo in channel_data_json.values():
            self.ref_channel_ids.append(cinfo["channel_id"])
            self.channel_data[cinfo["channel_id"]] = cinfo

        self._antenna_positions = None
        self._station_positions = None
        if position_path is not None:
            self.read_antenna_positions(position_path, maximum_radius=maximum_radius)

    def read_antenna_positions(self, base_path, maximum_radius=600 * units.m):
        """ Reads the antenna positions from the given path.
        This function expects a path to the root directory containing all information is a certain
        format.

        Parameters
        ----------
        base_path: str
            Path to the root directory containing the layout.txt.
        maximum_radius: float (Default: 600 * units.m)
            Maximum radius of stations to be included when reading from file.
        """
        assert self._antenna_positions is None, "Antenna positions already read. Cannot read again."
        self._antenna_positions = defaultdict(dict)
        self._station_positions = {}

        station_position_file = os.path.join(base_path, 'layout.txt')
        if not os.path.exists(station_position_file):
            self.logger.error(f"File {station_position_file} does not exist. Cannot read station positions. Exiting.")
            raise FileNotFoundError(f"File {station_position_file} does not exist. Cannot read station positions. Exiting.")

        station_positions = np.loadtxt(station_position_file)

        for station_id, station_position in enumerate(station_positions):

            if np.linalg.norm(station_position[:2]) > maximum_radius:
                continue

            antenna_position_file = os.path.join(base_path, f"station{station_id:03d}", "layout.txt")
            if not os.path.exists(antenna_position_file):
                self.logger.error(f"File {antenna_position_file} does not exist. Cannot read antenna positions. Exiting.")
                raise FileNotFoundError(f"File {antenna_position_file} does not exist. Cannot read antenna positions. Exiting.")

            self._station_positions[station_id] = station_position

            antenna_positions = np.loadtxt(antenna_position_file)
            antenna_positions_3d = np.zeros((antenna_positions.shape[0], 3))
            antenna_positions_3d[:, :2] = antenna_positions
            antenna_positions_3d[:, 2] = self.detector_altitude

            # create two entries: One for each antenna (arm) in a dual-polarized antenna (i.e., channels 0 and 1)
            for antenna_id, antenna_position in enumerate(antenna_positions_3d):
                self._antenna_positions[station_id][antenna_id] = antenna_position

    def add_antenna_position(self, station_id, antenna_id, position):
        """ Adds an antenna position to the detector. """
        if self._antenna_positions is None:
            self._antenna_positions = defaultdict(dict)
        self._antenna_positions[station_id][antenna_id] = position

    def add_station_position(self, station_id, position):
        """ Adds a station position to the detector. """
        if self._station_positions is None:
            self._station_positions = {}
        self._station_positions[station_id] = position

    def _get_reference_channel_id(self, station_id, channel_id):
        """ Returns the reference channel ID for the given station and channel ID.

        The reference channel ID is the last digit of the channel ID.

        Parameters
        ----------
        station_id: int
            Station ID.
        channel_id: int
            Channel ID.

        Returns
        -------
        ref_id: int
            Reference channel ID.
        """
        ref_channel_id = int(str(channel_id)[-1])  # take the last digit
        if ref_channel_id not in self.ref_channel_ids:
            self.logger.error(f"Reference channel ID {ref_channel_id} (inferred from {channel_id}) "
                            "not found in the reference channel list.")
            raise ValueError(f"Reference channel ID {ref_channel_id} (inferred from {channel_id}) "
                            "not found in the reference channel list.")
        return ref_channel_id

    def get_channel_ids(self, station_id):
        """ Returns all channel ids of one station (sorted) """
        assert self._antenna_positions is not None, "No antennas added yet. Cannot get channel IDs."
        antenna_ids = np.array(list(self._antenna_positions[station_id].keys()), dtype=int)
        channel_ids = np.hstack(
            [antenna_ids * 10, antenna_ids * 10 + 1], dtype=int)
        channel_ids.sort()
        return np.array(channel_ids)

    def get_station_ids(self):
        """ Returns all station ids """
        assert self._antenna_positions is not None, "No antennas added yet. Cannot get station IDs."
        return np.array(list(self._antenna_positions.keys()), dtype=int)

    def get_cable_delay(self, station_id=None, channel_id=None):
        channel_id = self._get_reference_channel_id(station_id, channel_id)
        return self.channel_data[channel_id]["cab_time_delay"]

    def get_site(self, station_id=None):
        return "ska"

    def get_absolute_position(self, station_id):
        """ Return the station position """
        return self._station_positions[station_id]

    def get_relative_position(self, station_id, channel_id):
        """ Return the relative position of the antenna in the station (relative to station position) """
        antenna_id = self.get_channel_group_id(station_id, channel_id)
        return self._antenna_positions[station_id][antenna_id]

    def get_antenna_model(self, station_id=None, channel_id=None, zenith_antenna=None):
        """ Returns the antenna model """
        channel_id = self._get_reference_channel_id(station_id, channel_id)
        return self.channel_data[channel_id]["ant_type"]

    def get_antenna_orientation(self, station_id=None, channel_id=None):
        """ Returns the channel's 4 orientation angles in rad """
        channel_id = self._get_reference_channel_id(station_id, channel_id)
        d = self.channel_data[channel_id]
        return np.deg2rad([d["ant_orientation_theta"], d["ant_orientation_phi"],
                           d["ant_rotation_theta"], d["ant_rotation_phi"]])

    def get_site_coordinates(self, station_id=None):
        """ Returns latitude and longitude of SKA in degrees """
        return -26.825, 116.764

    def get_channel_group_id(self, station_id, channel_id):
        """ Return the channel_group_id for a given channel_id.

        The channel_group_id associates channels which are at the same position (i.e., on the same antenna).
        Hence, the channel_group_id is the antenna_id.

        Parameters
        ----------
        station_id: int
            Station ID.
        channel_id: int
            Channel ID.

        Returns
        -------
        channel_group_id: int
            The channel_group_id or antenna_id.
        """
        if channel_id > 1:
            antenna_id = int(str(channel_id)[:-1]) # take all but the last digit
        else:
            antenna_id = 0

        return antenna_id

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    det = Detector(position_path=sys.argv[1])
    print(det.get_station_ids())
    fig, ax = plt.subplots()
    for stid in det.get_station_ids():
        for chid in det.get_channel_ids(stid):

            pos = det.get_relative_position(stid, chid)
            ax.plot(pos[0], pos[1], 'k.', alpha=0.1)

    ax.set_aspect(1)
    plt.show()
