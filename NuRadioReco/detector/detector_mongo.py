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
from NuRadioReco.detector.webinterface import config
logging.basicConfig()
logger = logging.getLogger("database")
logger.setLevel(logging.DEBUG)



@six.add_metaclass(NuRadioReco.utilities.metaclasses.Singleton)
class Detector(object):

    def __init__(self, database_connection="env_pw_user", database_name=None):

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

    def update(self, timestamp, collection_name):
        logger.info("updating detector time to {}".format(timestamp))
        self.__current_time = timestamp
        self._update_buffer(collection_name)

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

    def rename_database_collection(self, old_name, new_name):
        """
        changes the name of a collection of the database
        If the new name already exists, the operation fails.

        Parameters
        ---------
        old_name: string
            old name of the collection
        new_name: string
            new name of the collection
        """
        self.db[old_name].rename(new_name)

    def set_not_working(self, type, name, primary_measurement, channel_id=None):
        """
        inserts that the input unit is broken.
        If the input unit dosn't exist yet, it will be created.

        Parameters
        ---------
        type: string
            type of the input unit (HPol, VPol, surfCABLE, ...)
        name: string
            the unique identifier of the input unit
        primary_measurement: boolean
            specifies if this measurement is used as the primary measurement
        channel_id: int
            channel-id of the measured object
        """

        if channel_id is None:
            self.db[type].update_one({'name':name},
                             {'$push':{'measurements': {
                                 'last_updated': datetime.datetime.utcnow(),
                                 'function_test': False,
                                 'primary_measurement': primary_measurement
                             }}}, upsert=True)
        else:
            self.db[type].update_one({'name': name},
                                     {'$push': {'measurements': {
                                         'last_updated': datetime.datetime.utcnow(),
                                         'function_test': False,
                                         'primary_measurement': primary_measurement,
                                         'channel_id': channel_id
                                     }}}, upsert=True)

    def is_primary_working(self, type, name):
        """
        checks if the primary measurement is set not working.

        Parameters
        ---------
        type: string
            type of the input unit (HPol, VPol, surfCABLE, ...)
        name: string
            the unique identifier of the input unit
        """

        component_filter = [{'$match': {'name': name}},
                            {'$unwind': '$measurements'},
                            {'$match': {'measurements.function_test': True,
                             'measurements.primary_measurement': True}}]

        entries = list(self.db[type].aggregate(component_filter))
        if len(entries) == 1:
            if entries[0]['name'] == name:
                return True
        elif len(entries) > 1:
            logger.error('More than one entry is found.')
            return False
        else:
            return False

    def update_primary(self, type, name, temperature=None, channel_id=None):
        """
        updates the primary_measurement of previous entries to False

        Parameters
        ---------
        type: string
            type of the input unit (HPol, VPol, surfCABLE, ...)
        name: string
            the unique identifier of the input unit
        temperature: int
            temperature at which the object was measured
        channel_id: int
            channel-id of the object
        """
        if self.is_primary_working(type, name):
            if temperature is None and channel_id is None:
                self.db[type].update_one({'name': name},
                                         {"$set": {"measurements.$[updateIndex].primary_measurement": False}},
                                         array_filters=[{"updateIndex.primary_measurement": True}])
            elif temperature is not None and channel_id is None:
                # a measured temperature is given, only update the entries with the same temperature
                self.db[type].update_one({'name': name},
                                         {"$set": {"measurements.$[updateIndex].primary_measurement": False}},
                                         array_filters=[{"updateIndex.measurement_temp": temperature}])
            elif temperature is None and channel_id is not None:
                # a channel-id is given, only update the entries with the same channel-id
                self.db[type].update_one({'name': name},
                                         {"$set": {"measurements.$[updateIndex].primary_measurement": False}},
                                         array_filters=[{"updateIndex.channel_id": channel_id}])
            else:
                # a measured temperature and channel-id is given, only update the entries with the same temperature and channel-id
                self.db[type].update_one({'name': name},
                                         {"$set": {"measurements.$[updateIndex].primary_measurement": False}},
                                         array_filters=[{"updateIndex.measurement_temp": temperature, "updateIndex.channel_id": channel_id}])
        else:
            if channel_id is not None:
                # a channel-id is given, only update the entries with the same channel-id
                self.db[type].update_one({'name': name},
                                         {"$set": {"measurements.$[updateIndex].primary_measurement": False}},
                                         array_filters=[{"updateIndex.channel_id": channel_id}])
            else:
                self.db[type].update_one({'name': name},
                                         {"$set": {"measurements.$[updateIndex].primary_measurement": False}},
                                         array_filters=[{"updateIndex.primary_measurement": True}])

    def get_object_names(self, object_type):
        return self.db[object_type].distinct('name')

    def get_collection_names(self):
        return self.db.list_collection_names()

    def create_empty_collection(self, collection_name):
        self.db.create_collection(collection_name)

    def clone_colletion_to_colletion(self, old_colletion, new_colletion):
        self.db[old_colletion].aggregate([{ '$match': {} }, { '$out': new_colletion}])

    # antenna (VPol / HPol)

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
        S_parameter: list of strings
            specify which S_parameter is used (S11, ...)
        S_data: array of floats
            x and y data (the units are given as another input)
        primary_measurement: bool
            indicates the primary measurement to be used for analysis
        protocol: string
            details of the testing environment
        units_arr: list of strings
            list of the input units (only y unit will be saved)
        """
        # update the entry with the measurement (if the entry doesn't exist it will be created)
        for spara in S_parameter:
            self.db[antenna_type].update_one({'name': antenna_name},
                                             {'$push': {'measurements': {
                                                    'function_test': True,
                                                    'last_updated': datetime.datetime.utcnow(),
                                                    'primary_measurement': primary_measurement,
                                                    'measurement_protocol': protocol,
                                                    'S_parameter': spara,
                                                    'y-axis units': [units_arr[1]],
                                                    'frequencies': list(S_data[0]),
                                                    'mag': list(S_data[1])
                                             }}}, upsert=True)

    # cables

    def cable_add_Sparameter(self, cable_type, cable_name, S_parameter, Sm_data, Sp_data, units_arr, primary_measurement, protocol):
        """
        inserts a new S21 measurement of a SURFACE (11m) cable.
        If the cable dosn't exist yet, it will be created.

        Parameters
        ---------
        cable_type: string
            type of the cable (surface or downhole)
        cable_name: string
            the unique identifier of the cable (station + channel + type)
        Sparameter: list of strings
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

        # update the entry with the measurement (if the entry doesn't exist it will be created)
        for spara in S_parameter:
            self.db[cable_type].update_one({'name': cable_name},
                                             {'$push': {'measurements': {
                                                 'function_test': True,
                                                 'last_updated': datetime.datetime.utcnow(),
                                                 'primary_measurement': primary_measurement,
                                                 'measurement_protocol': protocol,
                                                 'S_parameter': spara,
                                                 'y-axis units': [units_arr[1], units_arr[2]],
                                                 'frequencies': list(Sm_data[0]),
                                                 'mag': list(Sm_data[1]),
                                                 'phase': list(Sp_data[1])
                                             }}}, upsert=True)

    # IGLU/DRAB

    def load_board_information(self, type, board_name, info_names):
        infos = []
        # if self.db[type].find_one({'name': board_name})['function_test']:
        #     for name in info_names:
        #         infos.append(self.db[type].find_one({'name': board_name})['measurements'][0][name])

        #
        for i in range(len(self.db[type].find_one({'name': board_name})['measurements'])):
            if self.db[type].find_one({'name': board_name})['measurements'][i]['function_test']:
                for name in info_names:
                    infos.append(self.db[type].find_one({'name': board_name})['measurements'][i][name])
                break

        return infos

    # IGLU

    def iglu_add_Sparameters(self, page_name, S_names, board_name, drab_id, laser_id, temp, S_data, measurement_time, primary_measurement, time_delay, protocol, units_arr):
        """
        inserts a new S parameter measurement of IGLU board
        If the board dosn't exist yet, it will be created.

        Parameters
        ---------
        board_name: string
            the unique identifier of the board
        drab_id: string
            the unique name of the DRAB unit
        laser_id: string
            the serial number of the laser diode
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
        # update the entry with the measurement (if the entry doesn't exist it will be created)
        for i, spara in enumerate(S_names):
            self.db[page_name].update_one({'name': board_name},
                                            {'$push': {'measurements': {
                                               'function_test': True,
                                               'last_updated': datetime.datetime.utcnow(),
                                               'primary_measurement': primary_measurement,
                                               'measurement_protocol': protocol,
                                               'S_parameter': spara,
                                               'DRAB_id': drab_id,
                                               'laser_id': laser_id,
                                               'measurement_temp': temp,
                                               'time_delay': time_delay[i],
                                               'measurement_time': measurement_time,
                                               'y-axis units': [units_arr[1], units_arr[2]],
                                               'frequencies': list(S_data[0]),
                                               'mag': list(S_data[2*i+1]),
                                               'phase': list(S_data[2*i+2])
                                           }}}, upsert=True)

    # DRAB

    def drab_add_Sparameters(self, page_name, S_names, board_name, iglu_id, photodiode_id, channel_id, temp, S_data, measurement_time, primary_measurement, time_delay, protocol, units_arr):
        """
        inserts a new S parameter measurement of IGLU board
        If the board dosn't exist yet, it will be created.

        Parameters
        ---------
        board_name: string
            the unique identifier of the board
        drab_id: string
            the unique name of the DRAB unit
        laser_id: string
            the serial number of the laser diode
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
        # update the entry with the measurement (if the entry doesn't exist it will be created)
        for i, spara in enumerate(S_names):
            self.db[page_name].update_one({'name': board_name},
                                           {'$push': {'measurements': {
                                                'function_test': True,
                                                'last_updated': datetime.datetime.utcnow(),
                                                'primary_measurement': primary_measurement,
                                                'measurement_protocol': protocol,
                                                'S_parameter': spara,
                                                'IGLU_id': iglu_id,
                                                'photodiode_serial': photodiode_id,
                                                'channel_id': channel_id,
                                                'measurement_temp': temp,
                                                'time_delay': time_delay[i],
                                                'measurement_time': measurement_time,
                                                'y-axis units': [units_arr[1], units_arr[2]],
                                                'frequencies': list(S_data[0]),
                                                'mag': list(S_data[2 * i + 1]),
                                                'phase': list(S_data[2 * i + 2])
                                           }}}, upsert=True)

    # SURFACE

    def surface_add_Sparameters(self, page_name, S_names, board_name, channel_id, temp, S_data, measurement_time, primary_measurement, time_delay, protocol, units_arr):
        """
        inserts a new S parameter measurement of IGLU board
        If the board dosn't exist yet, it will be created.

        Parameters
        ---------
        board_name: string
            the unique identifier of the board
        drab_id: string
            the unique name of the DRAB unit
        laser_id: string
            the serial number of the laser diode
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
        # update the entry with the measurement (if the entry doesn't exist it will be created)
        for i, spara in enumerate(S_names):
            self.db[page_name].update_one({'name': board_name},
                                           {'$push': {'measurements': {
                                                'function_test': True,
                                                'last_updated': datetime.datetime.utcnow(),
                                                'primary_measurement': primary_measurement,
                                                'measurement_protocol': protocol,
                                                'S_parameter': spara,
                                                'channel_id': channel_id,
                                                'measurement_temp': temp,
                                                'time_delay': time_delay[i],
                                                'measurement_time': measurement_time,
                                                'y-axis units': [units_arr[1], units_arr[2]],
                                                'frequencies': list(S_data[0]),
                                                'mag': list(S_data[2 * i + 1]),
                                                'phase': list(S_data[2 * i + 2])
                                           }}}, upsert=True)

    # full downhole chain

    def downhole_add_Sparameters(self, page_name, S_names, board_name, breakout_id, breakout_cha_id, iglu_id, drab_id, temp, S_data, measurement_time, primary_measurement, time_delay, protocol, units_arr):
        """
        inserts a new S parameter measurement of IGLU board
        If the board dosn't exist yet, it will be created.

        Parameters
        ---------
        board_name: string
            the unique identifier of the board
        drab_id: string
            the unique name of the DRAB unit
        laser_id: string
            the serial number of the laser diode
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
        # update the entry with the measurement (if the entry doesn't exist it will be created)
        for i, spara in enumerate(S_names):
            self.db[page_name].update_one({'name': board_name},
                                           {'$push': {'measurements': {
                                                'function_test': True,
                                                'last_updated': datetime.datetime.utcnow(),
                                                'primary_measurement': primary_measurement,
                                                'measurement_protocol': protocol,
                                                'S_parameter': spara,
                                                'IGLU_id': iglu_id,
                                                'DRAB_id': drab_id,
                                                'breakout': breakout_id,
                                                'breakout_channel': breakout_cha_id,
                                                'measurement_temp': temp,
                                                'time_delay': time_delay[i],
                                                'measurement_time': measurement_time,
                                                'y-axis units': [units_arr[1], units_arr[2]],
                                                'frequencies': list(S_data[0]),
                                                'mag': list(S_data[2 * i + 1]),
                                                'phase': list(S_data[2 * i + 2])
                                           }}}, upsert=True)

    # stations

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
            logger.error(f"channel with id {channel_id} already exists. Doing nothing.")
            return 1

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

    def get_station_information(self, collection, station_id):
        """ get information from one station """

        # if the collection is empty, return an empty dict
        if self.db[collection].count_documents({'id': station_id}) == 0:
            return {}

        # grouping dictionary is needed to undo the unwind
        grouping_dict = {"_id": "$_id", "channels": {"$push": "$channels"}}
        # add other keys that belong to a station
        for key in list(self.db[collection].find_one().keys()):
            if key in grouping_dict:
                continue
            else:
                grouping_dict[key] = {"$first": "${}".format(key)}

        # filter to get all information from one station with station_id and with akitve commission time
        time = self.__current_time
        time_filter = [{"$match": {
            'commission_time': {"$lte": time},
            'decommission_time': {"$gte": time},
            'id': station_id}},
            {"$unwind": '$channels'},
            {"$group": grouping_dict}]
        # get all stations which fit the filter (should only be one)
        stations_for_buffer = list(self.db[collection].aggregate(time_filter))

        # transform the output of db.aggregate to a dict
        station_info = dictionarize_nested_lists(stations_for_buffer, parent_key="id", nested_field="channels", nested_key="id")

        return station_info

    # other

    def _update_buffer(self, collection_name, force=False):
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
            self._buffer_stations(collection_name)
            self._buffer_hardware_components()

    def _buffer_stations(self, collection_name):
        """ write stations and channels for the current time to the buffer """

        # grouping dictionary is needed to undo the unwind
        grouping_dict = {"_id": "$_id", "channels": {"$push": "$channels"}}
        # add other keys that belong to a station
        for key in list(self.db[collection_name].find_one().keys()):
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
        stations_for_buffer = list(self.db[collection_name].aggregate(time_filter))

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

        # convert result to the dict of hardware types / names
        components = {}
        for item in list(self.db.station.aggregate(component_filter)):
            if not 'signal_ch' in item['channels']:
                ## continue silently
                #logger.debug("'signal_ch' not in db entry, continuing")
                continue
            ch_type = item['channels']['signal_ch']['type']
            ch_uname = item['channels']['signal_ch']['uname']
            # only add each component once
            if ch_type not in components:
                components[ch_type] = set([])
            components[ch_type].add(ch_uname)
        # convert sets to lists
        for key in components.keys():
            components[key] = list(components[key])
            logger.info(f"found {len(components[key])} hardware components in {key}: {components[key]}")
        return components

    # TODO probably needs some modification
    def _buffer_hardware_components(self, S_parameters = ["S21"]):
        """
        buffer all the components which appear in the current detector

        Parameters
        ----------
        S_parameters: list
            list of S-parameters to buffer
        """
        component_dict = self._find_hardware_components()

        nested_key_dict = {"SURFACE": "surface_channel_id", "DRAB": "drab_channel_id"}
        for hardware_type in component_dict:
            # grouping dictionary is needed to undo the unwind
            grouping_dict = {"_id": "$_id", "measurements": {"$push": "$measurements"}}

            if self.db[hardware_type].find_one() is None:
                print(f"DB for {hardware_type} is empty. Skipping...")
                continue
            # add other keys that belong to a station
            for key in list(self.db[hardware_type].find_one().keys()):
                if key in grouping_dict:
                    continue
                else:
                    grouping_dict[key] = {"$first": "${}".format(key)}

            #TODO select more accurately. is there a "primary" mesurement field? is S_parameter_XXX matching possible to reduce data?
            matching_components = list(self.db[hardware_type].aggregate([{"$match": {"name": {"$in": component_dict[hardware_type]}}},
                                                                         {"$unwind": "$measurements"},
                                                                         {"$match": {"measurements.primary_measurement": True,
                                                                                     "measurements.S_parameter": {"$in": S_parameters}}},
                                                                         {"$group": grouping_dict}]))
            #TODO wind #only S21?
            #list to dict conversion using "name" as keys
            print(f"Component entries found in {hardware_type}: {len(matching_components)}")
            if len(matching_components)==0:
                continue
            try:
                self.__db[hardware_type] = dictionarize_nested_lists_as_tuples(matching_components,
                        parent_key="name",
                        nested_field="measurements",
                        nested_keys=("channel_id","S_parameter"))
            except:
                continue

            #print("there")
            #self.__db[hardware_type] = dictionarize_nested_lists(matching_components, parent_key="name", nested_field=None, nested_key=None)

        print("done...")

    # TODO this is probably not used, unless we want to update on a per-station level
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
        if nested_field in parent and (nested_field is not None):
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


def dictionarize_nested_lists_as_tuples(nested_lists, parent_key="name", nested_field="measurements", nested_keys=("channel_id","S_parameter")):
    """ mongodb aggregate returns lists of dicts, which can be converted to dicts of dicts """
    res = {}
    for parent in nested_lists:
        res[parent[parent_key]] = parent
        if nested_field in parent and (nested_field is not None):
            daughter_list = parent[nested_field]
            daughter_dict = {}
            for daughter in daughter_list:
                # measurements do not have a unique column which can be used as key for the dictionnary, so use a tuple instead for indexing
                dict_key = []
                for nested_key in nested_keys:
                    if nested_key in daughter:
                        dict_key.append(daughter[nested_key])
                    else:
                        dict_key.append(None)
                daughter_dict[tuple(dict_key)] = daughter
                #else:
                #    logger.warning(f"trying to access unavailable nested key {nested_key} in field {nested_field}. Nothing to be done.")
            # replace list with dict
            res[parent[parent_key]][nested_field] = daughter_dict
    return res


def get_measurement_from_buffer(hardware_db, S_parameter="S21", channel_id=None):
    if channel_id is None:
        measurements = list(filter(lambda document: document['S_parameter'] == S_parameter, hardware_db["measurements"]))
    else:
        measurements = list(filter(lambda document: (document['S_parameter'] == S_parameter) & (document["channel_id"] == channel_id), hardware_db["measurements"]))
    if len(measurements)>1:
        print("WARNING: more than one match for requested measurement found")
    return measurements


# det = Detector(config.DATABASE_TARGET)

if __name__ == "__main__":
     test = sys.argv[1]
     det = Detector(test)
