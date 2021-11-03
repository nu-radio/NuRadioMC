from pymongo import MongoClient
import six
import os
import pymongo
import datetime
# pprint library is used to make the output look more pretty
from pprint import pprint
import logging
import urllib.parse
import numpy as np
import pandas as pd
from NuRadioReco.utilities import units
import NuRadioReco.utilities.metaclasses
logging.basicConfig()
logger = logging.getLogger("database")
logger.setLevel(logging.DEBUG)


@six.add_metaclass(NuRadioReco.utilities.metaclasses.Singleton)
class Detector(object):

    def __init__(self, database_connection="test"):

        if database_connection == "local":
            MONGODB_URL = "localhost"
            self.__mongo_client = MongoClient(MONGODB_URL)
            self.db = self.__mongo_client.RNOG_live
        elif database_connection == "env_url":
            # connect to MongoDB, change the << MONGODB_URL >> to reflect your own connection string
            MONGODB_URL = os.environ.get('MONGODB_URL')
            if MONGODB_URL is None:
                logger.warning('MONGODB_URL not set, defaulting to "localhost"')
                MONGODB_URL = 'localhost'
            self.__mongo_client = MongoClient(MONGODB_URL)
            self.db = self.__mongo_client.RNOG_live
        elif database_connection == "env_pw_user":
            # use db connection from environment, pw and user need to be percent escaped
            mongo_password = urllib.parse.quote_plus(os.environ.get('mongo_password'))
            mongo_user = urllib.parse.quote_plus(os.environ.get('mongo_user'))
            mongo_server = os.environ.get('mongo_server')
            if mongo_server is None:
                logger.warning('variable "mongo_server" not set')
            if None in [mongo_user, mongo_server]:
                logger.warning('"mongo_user" or "mongo_password" not set')
            # start client
            self.__mongo_client = MongoClient("mongodb://{}:{}@{}".format(mongo_user, mongo_password, mongo_server), tls=True)
            self.db = self.__mongo_client.RNOG_live
        elif database_connection == "test":
            self.__mongo_client = MongoClient("mongodb+srv://RNOG_test:TTERqY1YWBYB0KcL@cluster0-fc0my.mongodb.net/test?retryWrites=true&w=majority")
            self.db = self.__mongo_client.RNOG_test
        elif database_connection == "RNOG_public":
            self.__mongo_client = MongoClient("mongodb+srv://RNOG_read:7-fqTRedi$_f43Q@cluster0-fc0my.mongodb.net/test?retryWrites=true&w=majority")
            self.db = self.__mongo_client.RNOG_live
        else:
            logger.error('specify a defined database connection ["local", "env_url", "env_pw_user", "test"]')

        logger.info("database connection to {} established".format(self.db.name))

        self.__current_time = None

        self.__modification_timestamps = self._query_modification_timestamps()
        self.__buffered_period = None

        # just for testing
        logger.info("setting detector time to current time")
        self.update(datetime.datetime.now())
    # TODO do we need this?
    #def __error(self, frame):
    #    pass

    def update(self, timestamp):
        logger.info("updating detector time to {}".format(timestamp))
        self.__current_time = timestamp
        self._update_buffer()


    def get_surface_board_names(self):
        """
        returns the unique names of all surface boards

        Returns list of strings
        """
        return self.db.SURFACE.distinct("name")

    # def insert_amp_board_channel_S12(self, board_name, Sparameter, channel_id, ff, mag, phase):
    #     """
    #     inserts a new S12 measurement of one channel of an amp board
    #     If the board dosn't exist yet, it will be created.
    #     """
    #     self.db.amp_boards.update_one({'name': board_name},
    #                               {"$push" :{'channels': {
    #                                   'id': channel_id,
    #                                   'last_updated': datetime.datetime.utcnow(),
    #                                   'S_parameter': Sparameter,
    #                                   'frequencies': list(ff),
    #                                   'mag': list(mag),
    #                                   'phase': list(phase)
    #                                   }}},
    #                              upsert=True)

    def surface_board_channel_set_not_working(self, board_name, channel_id):
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
        self.db.SURFACE.update_one({'name': board_name},
                                      {"$push":{'measurements': {
                                          'id': surface_channel_id,
                                          'last_updated': datetime.datetime.utcnow(),
                                          'function_test': False,
                                          }}},
                                     upsert=True)

    def surface_board_channel_add_Sparameters(self, board_name, channel_id, temp, S_data,
                                              measurement_time, primary_measurement, time_delay, protocol):
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
        measurement_time: timestamp
            the time of the measurement
        primary_measurement: bool
            indicates the primary measurement to be used for analysis
        time_delay: array of floats
            the absolute time delay of each S parameter measurement (e.g. the group delay at
            a reference frequency)
        protocol: string
            details of the testing enviornment
        """
        S_names = ["S11", "S12", "S21", "S22"]
        for i in range(4):
            self.db.SURFACE.update_one({'name': board_name},
                                    {"$push": {'measurements': {
                                          'surface_channel_id': channel_id,
                                          'last_updated': datetime.datetime.utcnow(),
                                          'function_test': True,
                                          'measurement_temp': temp,
                                          'measurement_time': measurement_time,
                                          'primary_measurement': primary_measurement,
                                          'measurement_protocol': protocol,
                                          'time_delay': time_delay[i],
                                          'S_parameter': S_names[i],
                                          'frequencies': list(S_data[0]),
                                          'mag': list(S_data[2 * i + 1]),
                                          'phase': list(S_data[2 * i + 2])
                                          }}},
                                     upsert=True)

    # DRAB

    def DRAB_set_not_working(self, board_name):
        """
        inserts a new S parameter measurement of one channel of an amp board
        If the board dosn't exist yet, it will be created.

        Parameters
        ---------
        board_name: string
            the unique identifier of the board
        """
        self.db.DRAB.insert_one({'name': board_name,
                              'last_updated': datetime.datetime.utcnow(),
                              'function_test': False,
                                  })

    def DRAB_add_Sparameters(self, board_name, channel_id, iglu_id, temp, S_data,
                             measurement_time, primary_measurement, time_delay, protocol):
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
        measurement_time: timestamp
            the time of the measurement
        primary_measurement: bool
            indicates the primary measurement to be used for analysis
        time_delay: array of floats
            the absolute time delay of each S parameter measurement (e.g. the group delay at
            a reference frequency)
        protocol: string
            details of the testing enviornment

        """
        S_names = ["S11", "S12", "S21", "S22"]
        for i in range(4):
            self.db.DRAB.update_one({'name': board_name},
                                    {"$push": {'measurements': {
                                          'drab_channel_id': channel_id,
                                          'last_updated': datetime.datetime.utcnow(),
                                          'function_test': True,
                                          'IGLU_id': iglu_id,
                                          'measurement_temp': temp,
                                          'measurement_time': measurement_time,
                                          'primary_measurement': primary_measurement,
                                          'measurement_protocol': protocol,
                                          'time_delay': time_delay[i],
                                          'S_parameter': S_names[i],
                                          'frequencies': list(S_data[0]),
                                          'mag': list(S_data[2 * i + 1]),
                                          'phase': list(S_data[2 * i + 2])
                                          }}},
                                     upsert=True)

    # VPol

    def VPol_set_not_working(self, VPol_name):
        """
        inserts that the VPol is broken.
        If the antenna dosn't exist yet, it will be created.

        Parameters
        ---------
        VPol_name: string
            the unique identifier of the board
        """
        self.db.VPol.insert_one({'name': VPol_name,
                              'last_updated': datetime.datetime.utcnow(),
                              'function_test': False,
                                  })

    def VPol_add_Sparameters(self, VPol_name, S_data):
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

        self.db.VPol.insert_one({'name': VPol_name,
                                    'last_updated': datetime.datetime.utcnow(),
                                     'function_test': True,
                                     'S_parameter': 'S11',
                                     'frequencies': list(S_data[0]),
                                     'mag': list(S_data[1]),
                                  })

    # Cables

    def Cable_set_not_working(self, cable_name):
        """
        inserts that the cable is broken.
        If the cable dosn't exist yet, it will be created.

        Parameters
        ---------
        cable_name: string
            the unique identifier of the board
        """
        self.db.CABLE.insert_one({'name': cable_name,
                              'last_updated': datetime.datetime.utcnow(),
                              'function_test': False,
                                  })

    def CABLE_add_Sparameters(self, cable_name, Sm_data, Sp_data):
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

        self.db.CABLE.insert_one({'name': cable_name,
                                    'last_updated': datetime.datetime.utcnow(),
                                     'function_test': True,
                                     'S_parameter': 'S21',
                                     'frequencies': list(Sm_data[0]),
                                     'mag': list(Sm_data[1]),
                                     'phase': list(Sp_data[1]),
                                  })

    def surfCable_set_not_working(self, cable_name):
        """
        inserts that the cable is broken.
        If the cable dosn't exist yet, it will be created.

        Parameters
        ---------
        cable_name: string
            the unique identifier of the board
        """
        self.db.surfCABLE.insert_one({'name': cable_name,
                              'last_updated': datetime.datetime.utcnow(),
                              'function_test': False,
                                  })

    def surfCABLE_add_Sparameters(self, cable_name, Sm_data, Sp_data):
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

        self.db.surfCABLE.insert_one({'name': cable_name,
                                    'last_updated': datetime.datetime.utcnow(),
                                     'function_test': True,
                                     'S_parameter': 'S21',
                                     'frequencies': list(Sm_data[0]),
                                     'mag': list(Sm_data[1]),
                                     'phase': list(Sp_data[1]),
                                  })

    #### add IGLU board
    def IGLU_board_channel_set_not_working(self, board_name):
        """
        inserts a new S parameter measurement of one channel of an amp board
        If the board dosn't exist yet, it will be created.

        Parameters
        ---------
        board_name: string
            the unique identifier of the board

        """
        self.db.IGLU.update_one({'name': board_name},
                              {"$push":{'measurements': {
                                  'last_updated': datetime.datetime.utcnow(),
                                  'function_test': False,
                                  }}},
                             upsert=True)

    def IGLU_board_channel_add_Sparameters_with_DRAB(self, board_name, drab_id,
                                                     temp, S_data, measurement_time,
                                                     primary_measurement, time_delay, protocol):
        """
        inserts a new S parameter measurement of one channel of an IGLU board
        If the board dosn't exist yet, it will be created.

        Parameters
        ---------
        board_name: string
            the unique identifier of the board
        drab_id: string
            the unique name of the DRAB unit
        temp: int
            the temperature at which the measurement was taken
        S_data: array of floats
            1st collumn: frequencies
            2nd/3rd collumn: S11 mag/phase
            4th/5th collumn: S12 mag/phase
            6th/7th collumn: S21 mag/phase
            8th/9th collumn: S22 mag/phase
        measurement_time: timestamp
            the time of the measurement
        primary_measurement: bool
            indicates the primary measurement to be used for analysis
        time_delay: array of floats
            the absolute time delay of each S parameter measurement (e.g. the group delay at
            a reference frequency)
        protocol: string
            details of the testing enviornment

        """
        S_names = ["S11", "S12", "S21", "S22"]
        for i in range(4):
            self.db.IGLU.update_one({'name': board_name},
                                      {"$push":{'measurements': {
                                          'last_updated': datetime.datetime.utcnow(),
                                          'function_test': True,
                                          'DRAB-id': drab_id,
                                          'measurement_temp': temp,
                                          'measurement_time': measurement_time,
                                          'primary_measurement': primary_measurement,
                                          'measurement_protocol': protocol,
                                          'time_delay': time_delay[i],
                                          'S_parameter_DRAB': S_names[i],
                                          'frequencies': list(S_data[0]),
                                          'mag': list(S_data[2 * i + 1]),
                                          'phase': list(S_data[2 * i + 2])
                                          }}},
                                     upsert=True)

    def IGLU_board_channel_add_Sparameters_without_DRAB(self, board_name, temp,
                                                        S_data, measurement_time,
                                                        primary_measurement, time_delay, protocol):
        """
        inserts a new S parameter measurement of one channel of an IGLU board
        If the board dosn't exist yet, it will be created.

        Parameters
        ---------
        board_name: string
            the unique identifier of the board
        temp: int
            the temperature at which the measurement was taken
        S_data: array of floats
            1st collumn: frequencies
            2nd/3rd collumn: S11 mag/phase
            4th/5th collumn: S12 mag/phase
            6th/7th collumn: S21 mag/phase
            8th/9th collumn: S22 mag/phase
        measurement_time: timestamp
            the time of the measurement
        primary_measurement: bool
            indicates the primary measurement to be used for analysis
        time_delay: array of floats
            the absolute time delay of each S parameter measurement (e.g. the group delay at
            a reference frequency)
        protocol: string
            details of the testing enviornment

        """
        S_names = ["S11", "S12", "S21", "S22"]
        for i in range(4):
            self.db.IGLU.update_one({'name': board_name},
                                      {"$push":{'measurements': {
                                          'last_updated': datetime.datetime.utcnow(),
                                          'function_test': True,
                                          'measurement_temp': temp,
                                          'measurement_time': measurement_time,
                                          'primary_measurement': primary_measurement,
                                          'measurement_protocol': protocol,
                                          'time_delay': time_delay[i],
                                          'S_parameter_wo_DRAB': S_names[i],
                                          'frequencies': list(S_data[0]),
                                          'mag': list(S_data[2 * i + 1]),
                                          'phase': list(S_data[2 * i + 2])
                                          }}},
                                     upsert=True)

    def add_station(self,
                    station_id,
                    station_name,
                    position,  # in GPS UTM coordinates
                    commission_time,
                    decommission_time=datetime.datetime(2080, 1, 1)):
        if(self.db.station.count_documents({'id': station_id}) > 0):
            logger.error(f"station with id {station_id} already exists. Doing nothing.")
        else:
            self.db.station.insert_one({'id': station_id,
                                        'name': station_name,
                                        'position': list(position),
                                        'commission_time': commission_time,
                                        'decommission_time': decommission_time
                                        })

    def add_channel_to_station(self,
                               station_id,
                               channel_id,
                               signal_chain,
                               ant_name,
                               ant_ori_theta,
                               ant_ori_phi,
                               ant_rot_theta,
                               ant_rot_phi,
                               ant_position,
                               channel_type,
                               commission_time,
                               decommission_time=datetime.datetime(2080, 1, 1)):
        unique_station_id = self.db.station.find_one({'id': station_id})['_id']
        if(self.db.station.count_documents({'id': station_id, "channels.id": channel_id}) > 0):
            logger.warning(f"channel with id {channel_id} already exists")

        self.db.station.update_one({'_id': unique_station_id},
                               {"$push": {'channels': {
                                   'id': channel_id,
                                   'ant_name': ant_name,
                                   'ant_position': list(ant_position),
                                   'ant_ori_theta': ant_ori_theta,
                                   'ant_ori_phi': ant_ori_phi,
                                   'ant_rot_theta': ant_rot_theta,
                                   'ant_rot_phi': ant_rot_phi,
                                   'type': channel_type,
                                   'commission_time': commission_time,
                                   'decommission_time': decommission_time,
                                   'signal_ch': signal_chain
                                   }}
                               })

    # TODO add functions from detector class
    def get_station(self, station_id):
        """
        returns a dictionary of all station parameters

        Parameters
        ---------
        station_id: int
            the station id

        Return
        -------------
        dict of station parameters
        """
        return self.__db["stations"][station_id]

    def get_channel(self, station_id, channel_id):
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
        return self.get_station(station_id)["channels"][channel_id]

    def get_absolute_position(self, station_id):
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
        easting, northing, altitude = self.get_station(station_id)['position']
        return np.array([easting, northing, altitude])

    def get_relative_position(self, station_id, channel_id):
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
        chan = self.get_channel(station_id, channel_id)
        rel_pos = np.array(chan["ant_position"])
        return rel_pos

    def get_number_of_channels(self, station_id):
        """
        Get the number of channels per station

        Parameters
        ---------
        station_id: int
            the station id

        Returns int
        """
        return len(self.get_channel_ids(station_id))

    def get_channel_ids(self, station_id):
        """
        get the channel ids of a station

        Parameters
        ---------
        station_id: int
            the station id

        Returns list of ints
        """
        channel_ids = self.get_station(station_id)["channels"].keys()
        return sorted(channel_ids)

    def get_cable_delay(self, station_id, channel_id):
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

    #TODO this should go to the signal chain I guess... there could be more than one cable
    def get_cable_type_and_length(self, station_id, channel_id):
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

    def get_antenna_type(self, station_id, channel_id):
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
        chan = self.get_channel(station_id, channel_id)
        return  chan['type']

    def get_antenna_deployment_time(self, station_id, channel_id):
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
        chan = self.get_channel(station_id, channel_id)
        return chan['commission_time']

    def get_antenna_orientation(self, station_id, channel_id):
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
        chan = self.get_channel(station_id, channel_id)
        return chan["ant_ori_theta"], chan["ant_ori_phi"], chan["ant_rot_theta"], chan["ant_rot_phi"]

    def get_antenna(self, station_id, channel_id):
        """
        returns the antenna that belongs to a channel

        Parameters
        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns
        ---------------
        antenna from db
        """

        return None

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

    def get_amplifier_measurement(self, station_id, channel_id):
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

    def get_amplifier_response(self, station_id, channel_id, frequencies):
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

    # TODO: needed?
    def get_antenna_model(self, station_id, channel_id, zenith=None):
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

        # antenna_type = get_antenna_type(station_id, channel_id)
        # antenna_relative_position = get_relative_position(station_id, channel_id)
        return None

    # TODO: needed?
    def get_noise_temperature(self, station_id, channel_id):
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

    def get_signal_chain(self, station_id, channel_id):
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
        chan = self.get_channel(station_id, channel_id)
        return chan["signal_ch"]

    #TODO not needed any more?!
    #def find_db_entry(self, station_id, channel_id=None):
    #    """
    #    returns a dictionary with a database entry
    #
    #    Parameters
    #    ---------
    #    station_id: int
    #        the station id
    #    channel_id: int
    #        the channel id
    #
    #    Return
    #    -------------
    #    dict with database entry
    #    """
    #    if channel_id is None:
    #        aggregation = [agg_station(station_id), agg_first()]
    #    else:
    #        aggregation = [agg_station(station_id),
    #                  agg_unwind_channels(),
    #                  agg_channel(channel_id, self.__current_time),
    #                  agg_first()]
    #    res = self.db.station.aggregate(aggregation).next()
    #    return res

    def _update_buffer(self, force=False):
        """
        updates the buffer if need be

        Parameters
        ---------
        force: bool
            update even if the period has already been buffered
        """

        # digitize the current time of the detector to check if this period is already buffered or not
        current_period = np.digitize(self.__current_time.timestamp(), self.__modification_timestamps)
        if self.__buffered_period == current_period:
            logger.info("period already buffered")
        else:
            logger.info("buffering new period")
            self.__buffered_period = current_period
            #TODO do buffering, needs to be implemented, take whole db for now
            self.__db = {}
            self._buffer_stations()
            self._buffer_hardware_components()


    def _buffer_stations(self):
        """ write stations and channels for the current time to the buffer """

        # grouping dictionary is needed to undo the unwind
        grouping_dict = {"_id": "$_id", "channels": {"$push": "$channels"}}
        # add other keys that belong to a station
        for key in list(self.db.station.find_one().keys()):
            if key in grouping_dict:
                continue
            else:
                grouping_dict[key] = {"$first": "${}".format(key)}

        time = self.__current_time
        time_filter = [{"$match": {
                            'commission_time': {"$lte" : time},
                            'decommission_time': {"$gte" : time}}},
                       {"$unwind": '$channels'},
                       {"$match": {
                            'channels.commission_time': {"$lte" : time},
                            'channels.decommission_time': {"$gte" : time}}},
                       { "$group": grouping_dict}]
        stations_for_buffer = list(self.db.station.aggregate(time_filter))

        # convert nested lists of dicts to nested dicts of dicts
        self.__db["stations"] = dictionarize_nested_lists(stations_for_buffer, parent_key="id", nested_field="channels", nested_key="id")


    def _find_hardware_components(self):
        """
        returns hardware component names grouped by hardware type at the current detector time

        Return
        -------------
        dict with hardware component names grouped by hardware type
        """
        time = self.__current_time
        # filter for current time, and return only hardware component types and names
        component_filter = [{"$match": {
                              'commission_time': {"$lte" : time},
                              'decommission_time': {"$gte" : time}}},
                         {"$unwind": '$channels'},
                         {"$match": {
                              'channels.commission_time': {"$lte" : time},
                              'channels.decommission_time': {"$gte" : time}}},
                         {"$unwind": '$channels.signal_ch'},
                         {"$project": {"_id": False, "channels.signal_ch.uname": True,  "channels.signal_ch.type": True}}]

        # convert retult to the dict of hardware types / names
        components = {}
        for item in list(self.db.station.aggregate(component_filter)):
            ch_type = item['channels']['signal_ch']['type']
            ch_uname = item['channels']['signal_ch']['uname']
            # only add each component once
            if ch_type not in components:
                components[ch_type] = set([])
            components[ch_type].add(ch_uname)
        # convert sets to lists
        for key in components.keys():
            components[key] = list(components[key])
        return components

    def _buffer_hardware_components(self):
        """ buffer all the components which appear in the current detector """
        component_dict = self._find_hardware_components()

        for hardware_type in component_dict:
            #TODO select more accurately. is there a "primary" mesurement field? is S_parameter_XXX matching possible to reduce data?
            matching_components = list(self.db[hardware_type].aggregate([{"$match": {"name": {"$in": component_dict[hardware_type]}}}]))#,
                                       #{"$unwind": "$channels"}])
                                       #{"$match": {"channels.S_parameter_DRAB": "S12"}}])
                                       #{"$limit": 1}])
            #list to dict conversion using "id" as keys
            self.__db[hardware_type] = dictionarize_nested_lists(matching_components, parent_key="name", nested_field="channels", nested_key="id")

    def get_hardware_component(self, hardware_type, name):
        """
        get a specific hardware from the component buffer

        Return
        -------------
        dict of hardware component properties
        """

        return self.__db[hardware_type][name]

    def get_hardware_channel(self, hardware_type, name, channel):
        """
        get a channel for a hardware from the component buffer

        Return
        -------------
        dict of hardware channel info
        """
        component = self.__db[hardware_type][name]
        return component[channel]

    def get_signal_ch_hardware(self, station_id, channel_id):
        """
        get a list of component dicts for the signal chain

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Return
        -------------
        list of hardware components
        """

        channel = self.get_channel(station_id, channel_id)
        components = []
        for component in channel["signal_ch"]:
            components.append(self.get_hardware_channel(component['type'], component['uname'], component['channel_id']))
        return components

    # TODO this is probably not needed, nless we want to update on a per-station level (but buffering should be fast)
    def _query_modification_timestamps_per_station(self):
        """
        collects all the timestamps from the database for which some modifications happened
        ((de)commissioning of stations and channels).

        Return
        -------------
        dict with modification timestamps per station.id
        """
        # get distinct set of stations:
        station_ids = self.db.station.distinct("id")
        modification_timestamp_dict = {}

        for station_id in station_ids:
            # get set of (de)commission times for stations
            station_times_comm = self.db.station.distinct("commission_time", {"id": station_id})
            station_times_decomm = self.db.station.distinct("decommission_time", {"id": station_id})

            # get set of (de)commission times for channels
            channel_times_comm = self.db.station.distinct("channels.commission_time", {"id": station_id})
            channel_times_decomm = self.db.station.distinct("channels.decommission_time", {"id": station_id})

            mod_set = np.unique([*station_times_comm,
                                 *station_times_decomm,
                                 *channel_times_comm,
                                 *channel_times_decomm])
            mod_set.sort()
            # store timestamps, which can be used with np.digitize
            modification_timestamp_dict[station_id]= [mod_t.timestamp() for mod_t in mod_set]
        return modification_timestamp_dict


    def _query_modification_timestamps(self):
        """
        collects all the timestamps from the database for which some modifications happened
        ((de)commissioning of stations and channels).

        Return
        -------------
        list of modification timestamps
        """
        # get set of (de)commission times for stations
        station_times_comm = self.db.station.distinct("commission_time")
        station_times_decomm = self.db.station.distinct("decommission_time")

        # get set of (de)commission times for channels
        channel_times_comm = self.db.station.distinct("channels.commission_time")
        channel_times_decomm = self.db.station.distinct("channels.decommission_time")
        mod_set = np.unique([*station_times_comm,
                             *station_times_decomm,
                             *channel_times_comm,
                             *channel_times_decomm])
        mod_set.sort()
        # store timestamps, which can be used with np.digitize
        modification_timestamps = [mod_t.timestamp() for mod_t in mod_set]
        return modification_timestamps

def dictionarize_nested_lists(nested_lists, parent_key="id", nested_field="channels", nested_key="id"):
    """ mongodb aggregate returns lists of dicts, which can be converted to dicts of dicts """
    res = {}
    for parent in nested_lists:
        res[parent[parent_key]] = parent
        if nested_field in parent:
            daughter_list = parent[nested_field]
            daughter_dict = {}
            for daughter in daughter_list:
                if nested_key in daughter:
                    daughter_dict[daughter[nested_key]] = daughter
                #else:
                #    logger.warning(f"trying to access unavailable nested key {nested_key} in field {nested_field}. Nothing to be done.")
            # replace list with dict
            res[parent[parent_key]][nested_field] = daughter_dict
    return res
