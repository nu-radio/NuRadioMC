from pymongo import MongoClient
import os
import pymongo
import datetime
# pprint library is used to make the output look more pretty
from pprint import pprint
import logging
import urllib.parse
import numpy as np
from NuRadioReco.utilities import units

logging.basicConfig()
logger = logging.getLogger("database")
logger.setLevel(logging.DEBUG)

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient("mongodb+srv://detector_write:detector_write@cluster0-fc0my.mongodb.net/test?retryWrites=true&w=majority")
# client = MongoClient("localhost")

# use db connection from environment, pw and user need to be percent escaped
# MONGODB_URL = os.environ.get('MONGODB_URL')
# if MONGODB_URL is None:
#     logging.warning('MONGODB_URL not set, defaulting to "localhost"')
#     MONGODB_URL = 'localhost'
# client = MongoClient(MONGODB_URL)
# mongo_password = urllib.parse.quote_plus(os.environ.get('mongo_password'))
# mongo_user = urllib.parse.quote_plus(os.environ.get('mongo_user'))
# mongo_server = os.environ.get('mongo_server')
# if mongo_server is None:
#     logging.warning('variable "mongo_server" not set')
# if None in [mongo_user, mongo_server]:
#     logging.warning('"mongo_user" or "mongo_password" not set')
# start client
# client = MongoClient("mongodb://{}:{}@{}".format(mongo_user, mongo_password, mongo_server), tls=True)
db = client.RNOG_test


def get_surface_board_names():
    """
    returns the unique names of all surface boards

    Returns list of strings
    """
    return db.surface_boards.distinct("name")

# def insert_amp_board_channel_S12(board_name, Sparameter, channel_id, ff, mag, phase):
#     """
#     inserts a new S12 measurement of one channel of an amp board
#     If the board dosn't exist yet, it will be created.
#     """
#     db.amp_boards.update_one({'name': board_name},
#                               {"$push" :{'channels': {
#                                   'id': channel_id,
#                                   'last_updated': datetime.datetime.utcnow(),
#                                   'S_parameter': Sparameter,
#                                   'frequencies': list(ff),
#                                   'mag': list(mag),
#                                   'phase': list(phase)
#                                   }}},
#                              upsert=True)


def surface_board_channel_set_not_working(board_name, channel_id):
    """
    inserts a new S parameter measurement of one channel of an amp board
    If the board dosn't exist yet, it will be created.

    Parameters
    ---------
    board_name: string
        the unique identifier of the board
    channel_id: int
        the channel id
    S_data: array of floats
        1st collumn: frequencies
        2nd/3rd collumn: S11 mag/phase
        4th/5th collumn: S12 mag/phase
        6th/7th collumn: S21 mag/phase
        8th/9th collumn: S22 mag/phase

    """
    db.surface_boards.update_one({'name': board_name},
                                  {"$push":{'channels': {
                                      'id': channel_id,
                                      'last_updated': datetime.datetime.utcnow(),
                                      'function_test': False,
                                      }}},
                                 upsert=True)


def surface_board_channel_add_Sparameters(board_name, channel_id, temp, S_data):
    """
    inserts a new S parameter measurement of one channel of an amp board
    If the board dosn't exist yet, it will be created.

    Parameters
    ---------
    board_name: string
        the unique identifier of the board
    channel_id: int
        the channel id
    S_data: array of floats
        1st collumn: frequencies
        2nd/3rd collumn: S11 mag/phase
        4th/5th collumn: S12 mag/phase
        6th/7th collumn: S21 mag/phase
        8th/9th collumn: S22 mag/phase

    """
    S_names = ["S11", "S12", "S21", "S22"]
    for i in range(4):
        db.surface_boards.update_one({'name': board_name},
                                {"$push": {'channels': {
                                      'surface_channel_id': channel_id,
                                      'last_updated': datetime.datetime.utcnow(),
                                      'function_test': True,
                                      'measurement_temp': temp,
                                      'S_parameter': S_names[i],
                                      'frequencies': list(S_data[0]),
                                      'mag': list(S_data[2 * i + 1]),
                                      'phase': list(S_data[2 * i + 2])
                                      }}},
                                 upsert=True)

# DRAB


def DRAB_set_not_working(board_name):
    """
    inserts a new S parameter measurement of one channel of an amp board
    If the board dosn't exist yet, it will be created.

    Parameters
    ---------
    board_name: string
        the unique identifier of the board
    """
    db.DRAB.insert_one({'name': board_name,
                          'last_updated': datetime.datetime.utcnow(),
                          'function_test': False,
                              })


def DRAB_add_Sparameters(board_name, channel_id, temp, S_data):
    """
    inserts a new S parameter measurement of one channel of an amp board
    If the board dosn't exist yet, it will be created.

    Parameters
    ---------
    board_name: string
        the unique identifier of the board
    S_data: array of floats
        1st collumn: frequencies
        2nd/3rd collumn: S11 mag/phase
        4th/5th collumn: S12 mag/phase
        6th/7th collumn: S21 mag/phase
        8th/9th collumn: S22 mag/phase

    """
    S_names = ["S11", "S12", "S21", "S22"]
    for i in range(4):
        db.DRAB.update_one({'name': board_name},
                                {"$push": {'channels': {
                                      'drab_channel_id': channel_id,
                                      'last_updated': datetime.datetime.utcnow(),
                                      'function_test': True,
                                      'measurement_temp': temp,
                                      'S_parameter': S_names[i],
                                      'frequencies': list(S_data[0]),
                                      'mag': list(S_data[2 * i + 1]),
                                      'phase': list(S_data[2 * i + 2])
                                      }}},
                                 upsert=True)

# VPol


def VPol_set_not_working(VPol_name):
    """
    inserts that the VPol is broken.
    If the antenna dosn't exist yet, it will be created.

    Parameters
    ---------
    VPol_name: string
        the unique identifier of the board
    """
    db.VPol.insert_one({'name': VPol_name,
                          'last_updated': datetime.datetime.utcnow(),
                          'function_test': False,
                              })


def VPol_add_Sparameters(VPol_name, S_data):
    """
    inserts a new S11 measurement of a VPol.
    If the Antenna dosn't exist yet, it will be created.

    Parameters
    ---------
    board_name: string
        the unique identifier of the antenna
    S_data: array of floats
        1st collumn: frequencies
        2ndcollumn: S11 mag (VSWR)


    """

    db.VPol.insert_one({'name': VPol_name,
                                'last_updated': datetime.datetime.utcnow(),
                                 'function_test': True,
                                 'S_parameter': 'S11',
                                 'frequencies': list(S_data[0]),
                                 'mag': list(S_data[1]),
                              })

# Cables


def Cable_set_not_working(cable_name):
    """
    inserts that the cable is broken.
    If the cable dosn't exist yet, it will be created.

    Parameters
    ---------
    cable_name: string
        the unique identifier of the board
    """
    db.CABLE.insert_one({'name': cable_name,
                          'last_updated': datetime.datetime.utcnow(),
                          'function_test': False,
                              })


def CABLE_add_Sparameters(cable_name, Sm_data, Sp_data):
    """
    inserts a new S21 measurement of a cable.
    If the cable dosn't exist yet, it will be created.

    Parameters
    ---------
    cable_name: string
        the unique identifier of the antenna
    S_data: array of floats
        1st collumn: frequencies
        2nd collumn: S21 mag (dB)
        3nd collumn: S21 phase (deg)


    """

    db.CABLE.insert_one({'name': cable_name,
                                'last_updated': datetime.datetime.utcnow(),
                                 'function_test': True,
                                 'S_parameter': 'S21',
                                 'frequencies': list(Sm_data[0]),
                                 'mag': list(Sm_data[1]),
                                 'phase': list(Sp_data[1]),
                              })


def surfCable_set_not_working(cable_name):
    """
    inserts that the cable is broken.
    If the cable dosn't exist yet, it will be created.

    Parameters
    ---------
    cable_name: string
        the unique identifier of the board
    """
    db.surfCABLE.insert_one({'name': cable_name,
                          'last_updated': datetime.datetime.utcnow(),
                          'function_test': False,
                              })


def surfCABLE_add_Sparameters(cable_name, Sm_data, Sp_data):
    """
    inserts a new S21 measurement of a SURFACE (11m) cable.
    If the cable dosn't exist yet, it will be created.

    Parameters
    ---------
    cable_name: string
        the unique identifier of the antenna
    S_data: array of floats
        1st collumn: frequencies
        2nd collumn: S21 mag (dB)
        3nd collumn: S21 phase (deg)


    """

    db.surfCABLE.insert_one({'name': cable_name,
                                'last_updated': datetime.datetime.utcnow(),
                                 'function_test': True,
                                 'S_parameter': 'S21',
                                 'frequencies': list(Sm_data[0]),
                                 'mag': list(Sm_data[1]),
                                 'phase': list(Sp_data[1]),
                              })


#### add IGLU board
def IGLU_board_channel_set_not_working(board_name, channel_id):
    """
    inserts a new S parameter measurement of one channel of an amp board
    If the board dosn't exist yet, it will be created.

    Parameters
    ---------
    board_name: string
        the unique identifier of the board
    channel_id: int
        the channel id
    S_data: array of floats
        1st collumn: frequencies
        2nd/3rd collumn: S11 mag/phase
        4th/5th collumn: S12 mag/phase
        6th/7th collumn: S21 mag/phase
        8th/9th collumn: S22 mag/phase

    """
    db.IGLU.update_one({'name': board_name},
                          {"$push":{'channels': {
                              'iglue_channel_id': channel_id,
                              'last_updated': datetime.datetime.utcnow(),
                              'function_test': False,
                              }}},
                         upsert=True)


def IGLU_board_channel_add_Sparameters_with_DRAB(board_name, channel_id, drab_id, temp, S_data):
    """
    inserts a new S parameter measurement of one channel of an IGLU board
    If the board dosn't exist yet, it will be created.

    Parameters
    ---------
    board_name: string
        the unique identifier of the board
    channel_id: int
        the channel id
    drab_id: string
        the unique name of the DRAB unit
    S_data: array of floats
        1st collumn: frequencies
        2nd/3rd collumn: S11 mag/phase
        4th/5th collumn: S12 mag/phase
        6th/7th collumn: S21 mag/phase
        8th/9th collumn: S22 mag/phase

    """
    S_names = ["S11", "S12", "S21", "S22"]
    for i in range(4):
        db.IGLU.update_one({'name': board_name},
                                  {"$push":{'channels': {
                                      'iglu_channel_id': channel_id,
                                      'last_updated': datetime.datetime.utcnow(),
                                      'function_test': True,
                                      'DRAB-id': drab_id,
                                      'measurement_temp': temp,
                                      'S_parameter_DRAB': S_names[i],
                                      'frequencies': list(S_data[0]),
                                      'mag': list(S_data[2 * i + 1]),
                                      'phase': list(S_data[2 * i + 2])
                                      }}},
                                 upsert=True)


def IGLU_board_channel_add_Sparameters_without_DRAB(board_name, channel_id, temp, S_data):
    """
    inserts a new S parameter measurement of one channel of an IGLU board
    If the board dosn't exist yet, it will be created.

    Parameters
    ---------
    board_name: string
        the unique identifier of the board
    channel_id: int
        the channel id
    S_data: array of floats
        1st collumn: frequencies
        2nd/3rd collumn: S11 mag/phase
        4th/5th collumn: S12 mag/phase
        6th/7th collumn: S21 mag/phase
        8th/9th collumn: S22 mag/phase

    """
    S_names = ["S11", "S12", "S21", "S22"]
    for i in range(4):
        db.IGLU.update_one({'name': board_name},
                                  {"$push":{'channels': {
                                      'iglu_channel_id': channel_id,
                                      'last_updated': datetime.datetime.utcnow(),
                                      'function_test': True,
                                      'measurement_temp': temp,
                                      'S_parameter_wo_DRAB': S_names[i],
                                      'frequencies': list(S_data[0]),
                                      'mag': list(S_data[2 * i + 1]),
                                      'phase': list(S_data[2 * i + 2])
                                      }}},
                                 upsert=True)


def add_channel_to_station(station_id,
                           channel_id,
                           signal_chain,
                           ant_name,
                           ant_ori_theta,
                           ant_ori_phi,
                           ant_rot_theta,
                           ant_rot_phi,
                           ant_position,
                           type):
    db.station.update_one({'station_id': station_id,
                           'channels': [{
                               'channel_id': channel_id,
                               'ant_name': ant_name, 
                               'ant_position': ant_position,
                               'ant_ori_theta': ant_ori_theta,
                               'ant_ori_phi': ant_ori_phi,
                               'ant_rot_theta': ant_rot_theta,
                               'ant_rot_phi': ant_rot_phi,
                               'signal_ch':"a"
                               }]
                           })
    pass


#TODO add functions from detector class
def get_channel(station_id, channel_id):
    """
    returns a dictionary of all channel parameters

    Parameters
    ---------
    station_id: int
        the station id
    channel_id: int
        the channel id

    Return
    -------------
    dict of channel parameters
    """
    return None


def get_absolute_position(station_id):
    """
    get the absolute position of a specific station

    Parameters
    ---------
    station_id: int
        the station id

    Returns
    ----------------
    3-dim array of absolute station position in easting, northing and depth wrt. to snow level at
    time of measurement
    """
    easting, northing, altitude = 0, 0, 0
    unit_xy = units.m
    return np.array([easting, northing, altitude])


def get_relative_position(station_id, channel_id):
    """
    get the relative position of a specific channels/antennas with respect to the station center

    Parameters
    ---------
    station_id: int
        the station id
    channel_id: int
        the channel id

    Returns
    ---------------
    3-dim array of relative station position
    """
    return np.array([None, None, None])


def get_number_of_channels(station_id):
    """
    Get the number of channels per station

    Parameters
    ---------
    station_id: int
        the station id

    Returns int
    """
    res = []
    return len(res)

def get_channel_ids(station_id):
    """
    get the channel ids of a station

    Parameters
    ---------
    station_id: int
        the station id

    Returns list of ints
    """
    channel_ids = []
    return sorted(channel_ids)


def get_cable_delay(station_id, channel_id):
    """
    returns the cable delay of a channel

    Parameters
    ---------
    station_id: int
        the station id
    channel_id: int
        the channel id

    Returns float (delay time)
    """
    return None

def get_cable_type_and_length(station_id, channel_id):
    """
    returns the cable type (e.g. LMR240) and its length

    Parameters
    ---------
    station_id: int
        the station id
    channel_id: int
        the channel id

    Returns tuple (string, float)
    """
    return None, None

def get_antenna_type(station_id, channel_id):
    """
    returns the antenna type

    Parameters
    ---------
    station_id: int
        the station id
    channel_id: int
        the channel id

    Returns string
    """
    return None

def get_antenna_deployment_time(station_id, channel_id):
    """
    returns the time of antenna deployment

    Parameters
    ---------
    station_id: int
        the station id
    channel_id: int
        the channel id

    Returns datetime
    """
    return None

def get_antenna_orientation(station_id, channel_id):
    """
    returns the orientation of a specific antenna

    Parameters
    ---------
    station_id: int
        the station id
    channel_id: int
        the channel id

    Returns
    ---------------
    tuple of floats
        * orientation theta: orientation of the antenna, as a zenith angle (0deg is the zenith, 180deg is straight down); for LPDA: outward along boresight; for dipoles: upward along axis of azimuth  al symmetry
        * orientation phi: orientation of the antenna, as an azimuth angle (counting from East counterclockwise); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symme  try
        * rotation theta: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
        * rotation phi: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
    """
    return None, None, None, None

def get_amplifier_type(station_id, channel_id):
    """
    returns the type of the amplifier

    Parameters
    ---------
    station_id: int
        the station id
    channel_id: int
        the channel id

    Returns string
    """
    return None

def get_amplifier_measurement(station_id, channel_id):
    """
    returns a unique reference to the amplifier measurement

    Parameters
    ---------
    station_id: int
        the station id
    channel_id: int
        the channel id

    Returns string
    """
    return None

def get_amplifier_response(station_id, channel_id, frequencies):
    """
    Returns the amplifier response for the amplifier of a given channel

    Parameters:
    ---------------
    station_id: int
        The ID of the station
    channel_id: int
        The ID of the channel
    frequencies: array of floats
        The frequency array for which the amplifier response shall be returned
    """
    amp_gain = np.zeros_like(frequencies)
    amp_phase = np.zeros_like(frequencies)
    return amp_gain * amp_phase

#TODO: needed?
def get_antenna_model(station_id, channel_id, zenith=None):
    """
    determines the correct antenna model from antenna type, position and orientation of antenna

    so far only infinite firn and infinite air cases are differentiated

    Parameters
    ---------
    station_id: int
        the station id
    channel_id: int
        the channel id
    zenith: float or None (default)
        the zenith angle of the incoming signal direction

    Returns string
    """

    #antenna_type = get_antenna_type(station_id, channel_id)
    #antenna_relative_position = get_relative_position(station_id, channel_id)
    return None

#TODO: needed?
def get_noise_temperature(station_id, channel_id):
    """
    returns the noise temperature of the channel

    Parameters
    ----------
    station_id: int
        station id
    channel_id: int
        the channel id

    """
    return None


def get_signal_chain(station_id, channel_id):
    """
    returns a dictionary of all signal chain items

    Parameters
    ---------
    station_id: int
        the station id
    channel_id: int
        the channel id

    Return
    -------------
    dict of signal chain items
    """
    return None
