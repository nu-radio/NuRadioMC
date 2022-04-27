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
from NuRadioReco.detector import generic_detector
user=os.getenv('RNOG_DB_USER')
pw=os.getenv('RNOG_DB_PW')
mongo_detector = detector_mongo.Detector(database_connection=f"mongodb+srv://{user}:{pw}@cluster0-fc0my.mongodb.net/test?retryWrites=true&w=majority", database_name="RNOG_test")


fiber_names = {
        4:'A1',
        8:'A2',
        3:'A3',
        2:'A4',
        1:'A5',
        0:'A6',
        5:'ABLUE',
        7:'ABROWN',
        6:'AGREEN',
        ##'pulserPower':'AORANGE',
        ##'pulserB':'B1',
        11:'B2',
        10:'B3',
        9:'B4',
        ##'pulserC':'C1',
        21:'C2',
        22:'C3',
        23:'C4'}

channel_to_reference_measurement = {
        9:  1,
        10: 2,
        11: 3,
        23: 1,
        22: 2,
        21: 3}

short_fiber_names = {
        4: '1A1_SHORT',
        8: '1A2_SHORT',
        3: '1A3_SHORT',
        2: '1A4_SHORT',
        1: '1A5_SHORT',
        0: '1A6_SHORT',
        11:'1B2_SHORT',
        10:'1B3_SHORT',
        9: '1B4_SHORT',
        21:'1C2_SHORT',
        22:'1C3_SHORT',
        23:'1C4_SHORT'}        

hardware_number = {11: 5, 21: 7, 22: 6}

def add_channels_to_database(input_detector, station_id, target_station_id=None, commission_time=None):
    if target_station_id is None:
        target_station_id = station_id
    for channel_id in input_detector.get_channel_ids(station_id):
        channel = input_detector.get_channel(station_id, channel_id)
        amp = channel["amp_type"]
        if target_station_id == 999:
            if amp == "rno_surface":
                signal_chain = [{'type': 'surface_chain','uname': f'ref_surface', 'weight': 1}]
            elif amp == "iglu":
                if channel_id in channel_to_reference_measurement:
                    mapped_channel_id = channel_to_reference_measurement[channel_id]
                else:
                    mapped_channel_id = channel_id
                signal_chain = [{'type': 'downhole','uname': f'ref_channel{mapped_channel_id}', 'weight': 1}]
                if channel_id in short_fiber_names:
                    signal_chain.append({'type': 'CABLE','uname': short_fiber_names[channel_id], 'weight': 1})
            else:
                print(f"unknown amp type {amp}")
                continue
        else:    
            if amp == "rno_surface":
                signal_chain = [{'type': "SURFACE", 'uname': "Golden_Surface", 'weight': 1},
                                {'type': "surfCABLE", 'uname': "Golden_Coax", 'weight': 1}]
            elif amp == "iglu":
                signal_chain = [{'type': "IGLU", 'uname': "Golden_IGLU", 'weight': 1},
                                {'type': "CABLE",'uname': f"{hardware_number[station_id]}{fiber_names[channel_id]}_FULL", 'weight': 1},
                                {'type': "DRAB", 'uname': "Golden_DRAB", 'weight': 1}]
            else:
                print(f"unknown amp type {amp}")
                continue

        if commission_time is None:
            commission_time = channel["commission_time"]
        mongo_detector.add_channel_to_station(target_station_id,
                                   channel_id,
                                   signal_chain = signal_chain,
                                   ant_name = channel["ant_comment"],
                                   ant_ori_theta = channel["ant_orientation_theta"],
                                   ant_ori_phi = channel["ant_orientation_phi"],
                                   ant_rot_theta = channel["ant_rotation_theta"],
                                   ant_rot_phi = channel["ant_rotation_phi"],
                                   ant_position = [channel["ant_position_x"],channel["ant_position_y"],channel["ant_position_z"]],
                                   channel_type = channel["ant_type"],
                                   commission_time = commission_time)


# THE DEPLOYED RNOG 2021 DETECTOR
input_detector = detector.Detector(json_filename = "RNO_G/RNO_season_2021.json")
input_detector.update(Time.now())

station_ids = input_detector.get_station_ids()
for station_id in station_ids:
    station = input_detector.get_station(station_id)
    station_name = station["station_type"]

    mongo_detector.add_station(station_id, station_name,(station['pos_altitude'], station["pos_easting"], station["pos_northing"]),
                                            station["commission_time"], station["decommission_time"])

    add_channels_to_database(input_detector, station_id)

# ADD REFERENCE STATION

comm_time = input_detector.get_station(input_detector.get_station_ids()[0])["commission_time"]
decomm_time = input_detector.get_station(input_detector.get_station_ids()[0])["decommission_time"]

input_detector = generic_detector.GenericDetector(json_filename = "RNO_G/RNO_single_station.json", create_new=True)
input_detector.update(Time.now())

station_ids = input_detector.get_station_ids()
for station_id in station_ids:
    station = input_detector.get_station(station_id)

    target_station_id = 999 # explicitly set for reference station
    station_name = "reference"

    mongo_detector.add_station(target_station_id, station_name,(station['pos_altitude'], station["pos_easting"], station["pos_northing"]),
                                              comm_time, decomm_time)
    add_channels_to_database(input_detector, station_id, target_station_id, comm_time)

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
