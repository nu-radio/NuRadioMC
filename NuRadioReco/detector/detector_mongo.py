from pymongo import MongoClient
import six
import os
import sys
import urllib.parse
import datetime
import logging
from NuRadioReco.utilities import units
import NuRadioReco.utilities.metaclasses
import json
from bson import json_util #bson dicts are used by pymongo
import numpy as np
logging.basicConfig()
logger = logging.getLogger("database")
logger.setLevel(logging.DEBUG)



@six.add_metaclass(NuRadioReco.utilities.metaclasses.Singleton)
class Detector(object):

    def __init__(self, database_connection="test", database_name=None):

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
            self.__mongo_client = MongoClient("mongodb://read:EseNbGVaCV4pBBrt@radio.zeuthen.desy.de:27017/admin?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=true")
            #self.__mongo_client = MongoClient("mongodb+srv://RNOG_read:7-fqTRedi$_f43Q@cluster0-fc0my.mongodb.net/test?retryWrites=true&w=majority")
            self.db = self.__mongo_client.RNOG_live
        elif isinstance(database_connection, str):
            logger.info(f'trying to connect to {database_connection} ...')
            self.__mongo_client = MongoClient(database_connection)
            logger.info(f'looking for {database_name} ...')
            self.db = self.__mongo_client.get_database(database_name)
        else:
            logger.error('specify a defined database connection ["local", "env_url", "env_pw_user", "test"]')

        logger.info("database connection to {} established".format(self.db.name))

        self.__current_time = None

        self.__modification_timestamps = self._query_modification_timestamps()
        self.__buffered_period = None

        # just for testing
        # logger.info("setting detector time to current time")
        # self.update(datetime.datetime.now())


    def update(self, timestamp):
        logger.info("updating detector time to {}".format(timestamp))
        self.__current_time = timestamp
        self._update_buffer()

    def export_detector(self, filename="detector.json"):
        """ export the detector to file """

        if os.path.exists(filename):
            logger.error("Output file already exists.")
        else:
            self.__db["detector_time"] = self.__current_time
            with open(filename, 'w') as fp:
                fp.write(json_util.dumps(self.__db, indent=4, sort_keys=True))
                #Note: some output/timezone options can be set in bson.json_util.DEFAULT_JSON_OPTIONS
            logger.info("Output written to {}.".format(filename))

    def import_detector(self, filename):
        """ import the detector from file """
        if os.path.isfile(filename):
            logger.info("Importing detector from file {}".format(filename))

            self.__det = json.load(open(filename))
            self._current_time = self.__det["detector_time"]
        else:
            logger.error("Cannot import detector. File {} does not exist.".format(filename))

    # general

    def set_not_working(self, type, name):
        """
        inserts that the input unit is broken.
        If the input unit dosn't exist yet, it will be created.

        Parameters
        ---------
        type: string
            type of the input unit (HPol, VPol, surfCABLE, ...)
        name: string
            the unique identifier of the input unit
        """
        self.db[type].insert_one({'name': name,
                                        'last_updated': datetime.datetime.utcnow(),
                                        'function_test': False,
                                        })

    def update_primary(self, type, name):
        """
        updates the primary_measurement of previous entries to False

        Parameters
        ---------
        type: string
            type of the input unit (HPol, VPol, surfCABLE, ...)
        name: string
            the unique identifier of the input unit
        """

        self.db[type].update_one({'name': name},
                                {"$set": {"primary_measurement": False}})

    # antenna (VPol / HPol)

    def get_Antenna_names(self, antenna_type):
        names = self.db[antenna_type].distinct('name')
        names.insert(0, f'new {antenna_type}')
        return names

    def antenna_set_not_working(self, antenna_type, antenna_name):
        """
        inserts that the antenna is broken.
        If the antenna dosn't exist yet, it will be created.

        Parameters
        ---------
        antenna_type: string
            specify if it is a VPol or HPol antenna
        antenna_name: string
            the unique identifier of the antenna
        """
        self.db[antenna_type].insert_one({'name': antenna_name,
                                 'last_updated': datetime.datetime.utcnow(),
                                 'function_test': False,
                                 })

    def antenna_add_Sparameter(self, antenna_type, antenna_name, S_parameter, S_data, primary_measurement, protocol, units_arr):
        """
        inserts a new S measurement of a antenna.
        If the Antenna dosn't exist yet, it will be created.

        Parameters
        ---------
        antenna_type: string
            specify if it is a VPol or HPol antenna
        antenna_name: string
            the unique identifier of the antenna
        S_parameter: string
            specify which S_parameter is used (S11, ...)
        S_data: array of floats
            x and y data (the units are given as another input)
        primary_measurement: bool
            indicates the primary measurement to be used for analysis
        protocol: string
            details of the testing environment
        units_arr: string
            input y data
        """
        self.db[antenna_type].insert_one({'name': antenna_name,
                                 'last_updated': datetime.datetime.utcnow(),
                                 'function-test': True,
                                 'primary_measurement': primary_measurement,
                                 'measurement_protocol': protocol,
                                 'S_parameter': S_parameter,
                                 'Units mag': units_arr[1],
                                 'frequencies': list(S_data[0]),
                                 'mag': list(S_data[1])})

    def update_antenna_primary(self, antenna_type, antenna_name):
        """
        updates the primary_measurement of previous entries to False

        Parameters
        ---------
        antenna_type: string
            specify if it is a VPol or HPol antenna
        antenna_name: string
            the unique identifier of the antenna
        """

        self.db[antenna_type].update_one({'name': antenna_name},
                                {"$set": {"primary_measurement": False}})

    # cables

    def get_cable_names(self, cable_type):
        return self.db[cable_type].distinct('name')

    def cable_add_Sparameter(self, cable_type, cable_name, Sparameter, Sm_data, Sp_data, units_arr, primary_measurement, protocol):
        """
        inserts a new S21 measurement of a SURFACE (11m) cable.
        If the cable dosn't exist yet, it will be created.

        Parameters
        ---------
        cable_type: string
            type of the cable (surface or downhole)
        cable_name: string
            the unique identifier of the cable (station + channel + type)
        Sparameter: string
            specify which S parameter was measured
        Sm_data: array of floats
            magnitude data (frequencies will be saved in the 1st column, magnitude will be saved in the 2nd column
        Sp_data: array of floats
            phase data (phase data will be saved in the 3rd column)
        units_arr: list
            the units of the input y data
        primary_measurement: bool
            indicates the primary measurement to be used for analysis
        protocol: string
            details of the testing environment
        """

        self.db[cable_type].insert_one({'name': cable_name,
                                      'last_updated': datetime.datetime.utcnow(),
                                      'function_test': True,
                                      'primary_measurement': primary_measurement,
                                      'measurement_protocol': protocol,
                                      'S_parameter': Sparameter,
                                      'Units': [units_arr[1], units_arr[2]],
                                      'frequencies': list(Sm_data[0]),
                                      'mag': list(Sm_data[1]),
                                      'phase': list(Sp_data[1]), })


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


if __name__ == "__main__":
     test = sys.argv[1]
     det = Detector(test)