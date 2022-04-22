from pymongo import MongoClient
import six
import os
import pymongo
import datetime
from astropy.time import Time
from pprint import pprint
import logging
import urllib.parse
import json
from bson import json_util #bson dicts are used by pymongo
import numpy as np
import pandas as pd
from NuRadioReco.utilities import units
import NuRadioReco.utilities.metaclasses
from NuRadioReco.detector import detector_mongo
from NuRadioReco.detector import detector


mongo_detector = detector_mongo.Detector(database_connection="test")

input_detector = detector.Detector(json_filename = "RNO_G/RNO_season_2021.json")

input_detector.update(Time.now())

station = input_detector.get_station(input_detector.get_station_ids()[0])

mongo_detector.add_station(96,"dummy3",(station['pos_altitude'], station["pos_easting"], station["pos_northing"]),
            station["commission_time"], station["decommission_time"])



def add_dummy_channels(input_detector, station_id = 96):
    for channel_id in input_detector.get_channel_ids(input_detector.get_station_ids()[0]):
        channel = input_detector.get_channel(input_detector.get_station_ids()[0], channel_id)
        amp = channel["amp_type"]
        if amp == "rno_surface":
            signal_chain = [ "Golden_Coax", "Golden_Surface"]
        elif amp == "iglu":
            signal_chain = ["Golden_IGLU", "Golden_Fiber", "Golden_DRAB"]
        else:
            print(f"unknown amp type {amp}")
            continue

        mongo_detector.add_channel_to_station(station_id,
                                   channel_id,
                                   signal_chain = signal_chain,
                                   ant_name = channel["ant_comment"],
                                   ant_ori_theta = channel["ant_orientation_theta"],
                                   ant_ori_phi = channel["ant_orientation_phi"],
                                   ant_rot_theta = channel["ant_rotation_theta"],
                                   ant_rot_phi = channel["ant_rotation_phi"],
                                   ant_position = [channel["ant_position_x"],channel["ant_position_y"],channel["ant_position_z"]],
                                   channel_type = channel["ant_type"],
                                   commission_time = channel["commission_time"])

# add_dummy_channels(input_detector)

# def add_dummy_devices(input_detector, station_id = 97):
#     for device_id in input_detector.get_device_ids(input_detector.get_station_ids()[2]):
#         device = input_detector.get_device(input_detector.get_station_ids()[2], device_id)
#         amp = channel["amp_type"]
#         if amp == "surface_pulser":
#             signal_chain = ["Golden_Coax", "Golden_Surface"]
#         elif amp == "deep_pulser":
#             signal_chain = ["Golden_IGLU", "Golden_Fiber", "Golden_DRAB"]
#         else:
#             print(f"unknown amp type {amp}")
#             continue
#
#         mongo_detector.add_channel_to_station(station_id,
#                                    channel_id,
#                                    signal_chain = signal_chain,
#                                    ant_name = channel["ant_comment"],
#                                    ant_ori_theta = channel["ant_orientation_theta"],
#                                    ant_ori_phi = channel["ant_orientation_phi"],
#                                    ant_rot_theta = channel["ant_rotation_theta"],
#                                    ant_rot_phi = channel["ant_rotation_phi"],
#                                    ant_position = [channel["ant_position_x"],channel["ant_position_y"],channel["ant_position_z"]],
#                                    channel_type = channel["ant_type"],
#                                    commission_time = channel["commission_time"])
