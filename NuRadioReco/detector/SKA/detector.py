import numpy as np
import json
import os

class Detector:

    def __init__(self, json_file=None):

        if json_file is None:
            json_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "ska_channels.json"
            )

        with open(json_file, "r") as f:
            self.json = json.load(f)

        self.channel_ids = []
        self.channel_data = {}
        for cinfo in self.json["channels"].values():
            self.channel_ids.append(cinfo["channel_id"])
            self.channel_data[cinfo["channel_id"]] = cinfo

    def _get_channel_id_from_station_channel_pair(self, station_id, channel_id):
        if station_id is None:
            return channel_id
        # To be changed ...
        return channel_id

    def get_channel_ids(self, station_id=None):
        return self.channel_ids

    def get_cable_delay(self, station_id=None, channel_id=None):
        channel_id = self._get_channel_id_from_station_channel_pair(station_id, channel_id)
        return self.channel_data[channel_id]["cab_time_delay"]

    def get_site(self, station_id=None):
        return "ska"

    def get_relative_position(self, station_id=None, channel_id=None):
        channel_id = self._get_channel_id_from_station_channel_pair(station_id, channel_id)
        return np.array([self.channel_data[channel_id]["ant_position_x"],
                         self.channel_data[channel_id]["ant_position_y"],
                         self.channel_data[channel_id]["ant_position_z"]])

    def get_antenna_model(self, station_id=None, channel_id=None, zenith_antenna=None):
        """ Returns the antenna model """
        channel_id = self._get_channel_id_from_station_channel_pair(station_id, channel_id)
        return self.channel_data[channel_id]["ant_type"]

    def get_antenna_orientation(self, station_id=None, channel_id=None):
        """ Returns the channel's 4 orientation angles in rad """
        channel_id = self._get_channel_id_from_station_channel_pair(station_id, channel_id)
        d = self.channel_data[channel_id]
        return np.deg2rad([d["ant_orientation_theta"], d["ant_orientation_phi"],
                           d["ant_rotation_theta"], d["ant_rotation_phi"]])

    def get_site_coordinates(self, station_id=None):
        """ Returns latitude and longitude of SKA in degrees """
        return -26.825, 116.764