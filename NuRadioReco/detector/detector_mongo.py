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
from NuRadioReco.utilities import units
import NuRadioReco.utilities.metaclasses
logging.basicConfig()
logger = logging.getLogger("database")
logger.setLevel(logging.DEBUG)


@six.add_metaclass(NuRadioReco.utilities.metaclasses.Singleton)
class Detector(object):

    def __init__(self):
        # connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
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

        self.__mongo_client = MongoClient("mongodb+srv://detector_write:detector_write@cluster0-fc0my.mongodb.net/test?retryWrites=true&w=majority")

        self.db = self.__mongo_client.RNOG_test
        logger.info("database connection to {} established".format("RNOG_test"))

        self.__current_time = None
        # just for testing
        logger.info("setting detector time to current time")
        self.__current_time = datetime.datetime.now()

    def __error(self, frame):
        pass

    def update(self, timestamp):
        logger.info("updating detector time to {}".format(timestamp))
        self.__current_time = timestamp

    def get_surface_board_names(self):
        """
        returns the unique names of all surface boards

        Returns list of strings
        """
        return self.db.surface_boards.distinct("name")

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
        self.db.surface_boards.update_one({'name': board_name},
                                      {"$push":{'channels': {
                                          'id': channel_id,
                                          'last_updated': datetime.datetime.utcnow(),
                                          'function_test': False,
                                          }}},
                                     upsert=True)

    def surface_board_channel_add_Sparameters(self, board_name, channel_id, temp, S_data):
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
            self.db.surface_boards.update_one({'name': board_name},
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

    def DRAB_add_Sparameters(self, board_name, channel_id, iglu_id, temp, S_data):
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
            self.db.DRAB.update_one({'name': board_name},
                                    {"$push": {'channels': {
                                          'drab_channel_id': channel_id,
                                          'last_updated': datetime.datetime.utcnow(),
                                          'function_test': True,
                                          'IGLU_id': iglu_id,
                                          'measurement_temp': temp,
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
                              {"$push":{'channels': {
                                  'last_updated': datetime.datetime.utcnow(),
                                  'function_test': False,
                                  }}},
                             upsert=True)

    def IGLU_board_channel_add_Sparameters_with_DRAB(self, board_name, drab_id, temp, S_data):
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

        """
        S_names = ["S11", "S12", "S21", "S22"]
        for i in range(4):
            self.db.IGLU.update_one({'name': board_name},
                                      {"$push":{'channels': {
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

    def IGLU_board_channel_add_Sparameters_without_DRAB(self, board_name, temp, S_data):
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

        """
        S_names = ["S11", "S12", "S21", "S22"]
        for i in range(4):
            self.db.IGLU.update_one({'name': board_name},
                                      {"$push":{'channels': {
                                          'last_updated': datetime.datetime.utcnow(),
                                          'function_test': True,
                                          'measurement_temp': temp,
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
        return None

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
        easting, northing, altitude = 0, 0, 0
        unit_xy = units.m
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
        return np.array([None, None, None])

    def get_number_of_channels(self, station_id):
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

    def get_channel_ids(self, station_id):
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
        return None

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
        return None

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
        return None, None, None, None

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
        return None
