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
from bson import ObjectId
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

    def set_not_working(self, type, name, primary_measurement, channel_id=None, breakout_id=None, breakout_channel_id=None):
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

        # close the time period of the old primary measurement
        if primary_measurement and name in self.get_object_names(type):
            self.update_current_primary(type, name, channel_id=channel_id, breakout_id=breakout_id, breakout_channel_id=breakout_channel_id)

        # define the new primary measurement times
        primary_measurement_times = [{'start': datetime.datetime.utcnow(), 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)}]

        if channel_id is not None:
            self.db[type].update_one({'name': name},
                                     {'$push': {'measurements': {
                                         'id_measurement': ObjectId(),
                                         'last_updated': datetime.datetime.utcnow(),
                                         'function_test': False,
                                         'primary_measurement': primary_measurement_times,
                                         'channel_id': channel_id
                                     }}}, upsert=True)
        elif breakout_id is not None and breakout_channel_id is not None:
            self.db[type].update_one({'name': name},
                                     {'$push': {'measurements': {
                                         'id_measurement': ObjectId(),
                                         'last_updated': datetime.datetime.utcnow(),
                                         'function_test': False,
                                         'primary_measurement': primary_measurement_times,
                                         'breakout': breakout_id,
                                         'breakout_channel': breakout_channel_id
                                     }}}, upsert=True)
        else:
            self.db[type].update_one({'name': name},
                                     {'$push': {'measurements': {
                                         'id_measurement': ObjectId(),
                                         'last_updated': datetime.datetime.utcnow(),
                                         'function_test': False,
                                         'primary_measurement': primary_measurement_times
                                     }}}, upsert=True)

    # TODO: need to be adapted to the new structure
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

    # def update_primary(self, type, name, temperature=None, channel_id=None):
    #     """
    #     updates the primary_measurement of previous entries to False
    #
    #     Parameters
    #     ---------
    #     type: string
    #         type of the input unit (HPol, VPol, surfCABLE, ...)
    #     name: string
    #         the unique identifier of the input unit
    #     temperature: int
    #         temperature at which the object was measured
    #     channel_id: int
    #         channel-id of the object
    #     """
    #     if self.is_primary_working(type, name):
    #         if temperature is None and channel_id is None:
    #             self.db[type].update_one({'name': name},
    #                                      {"$set": {"measurements.$[updateIndex].primary_measurement": False}},
    #                                      array_filters=[{"updateIndex.primary_measurement": True}])
    #         elif temperature is not None and channel_id is None:
    #             # a measured temperature is given, only update the entries with the same temperature
    #             self.db[type].update_one({'name': name},
    #                                      {"$set": {"measurements.$[updateIndex].primary_measurement": False}},
    #                                      array_filters=[{"updateIndex.measurement_temp": temperature}])
    #         elif temperature is None and channel_id is not None:
    #             # a channel-id is given, only update the entries with the same channel-id
    #             self.db[type].update_one({'name': name},
    #                                      {"$set": {"measurements.$[updateIndex].primary_measurement": False}},
    #                                      array_filters=[{"updateIndex.channel_id": channel_id}])
    #         else:
    #             # a measured temperature and channel-id is given, only update the entries with the same temperature and channel-id
    #             self.db[type].update_one({'name': name},
    #                                      {"$set": {"measurements.$[updateIndex].primary_measurement": False}},
    #                                      array_filters=[{"updateIndex.measurement_temp": temperature, "updateIndex.channel_id": channel_id}])
    #     else:
    #         if channel_id is not None:
    #             # a channel-id is given, only update the entries with the same channel-id
    #             self.db[type].update_one({'name': name},
    #                                      {"$set": {"measurements.$[updateIndex].primary_measurement": False}},
    #                                      array_filters=[{"updateIndex.channel_id": channel_id}])
    #         else:
    #             self.db[type].update_one({'name': name},
    #                                      {"$set": {"measurements.$[updateIndex].primary_measurement": False}},
    #                                      array_filters=[{"updateIndex.primary_measurement": True}])

    def find_primary_measurement(self, type, name, primary_time, identification_label='name', channel_id=None, breakout_id=None, breakout_channel_id=None):
        """
                find the object_id of entry with name 'name' and gives the measurement_id of the primary measurement, return the id of the object and the measurement

                Parameters
                ---------
                type: string
                    type of the input unit (HPol, VPol, surfCABLE, ...)
                name: string
                    the unique identifier of the input unit
                primary_time: datetime.datetime
                    timestamp for the primary measurement
                channel_id: int
                    if there is a channel id for the object, the id is used in the search filter mask
        """

        # define search filter for the collection
        if breakout_channel_id is not None and breakout_id is not None:
            filter_primary = [{'$match': {identification_label: name}},
                              {'$unwind': '$measurements'},
                              {'$unwind': '$measurements.primary_measurement'},
                              {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                          'measurements.primary_measurement.end': {'$gte': primary_time},
                                          'measurements.breakout': breakout_id,
                                          'measurements.breakout_channel': breakout_channel_id}}]
        elif channel_id is not None:
            filter_primary = [{'$match': {identification_label: name}},
                              {'$unwind': '$measurements'},
                              {'$unwind': '$measurements.primary_measurement'},
                              {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                          'measurements.primary_measurement.end': {'$gte': primary_time},
                                          'measurements.channel_id': channel_id}}]
        else:
            filter_primary = [{'$match': {identification_label: name}},
                              {'$unwind': '$measurements'},
                              {'$unwind': '$measurements.primary_measurement'},
                              {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                          'measurements.primary_measurement.end': {'$gte': primary_time}}}]

        # get all entries matching the search filter
        matching_entries = list(self.db[type].aggregate(filter_primary))

        # extract the object and measurement id
        if len(matching_entries) > 1:
            # check if they are for different Sparameters
            s_parameter = []
            measurement_ids = []
            for entries in matching_entries:
                s_parameter.append(entries['measurements']['S_parameter'])
                measurement_ids.append(entries['measurements']['id_measurement'])
            if len(s_parameter) == len(set(s_parameter)):
                # all S_parameter are different
                object_id = matching_entries[0]['_id']
                measurement_id = measurement_ids
                return object_id, measurement_id
            else:
                logger.error('More than one primary measurement found.')
                # some S_parameter are the same
                return None, None
        elif len(matching_entries) > 4:
            logger.error('More primary measurements than Sparameters are found.')
            return None, None
        elif len(matching_entries) == 0:
            logger.error('No primary measurement found.')
            # the last zero is the information that no primary measurement was found
            return None, [0]
        else:
            object_id = matching_entries[0]['_id']
            measurement_id = matching_entries[0]['measurements']['id_measurement']
            return object_id, [measurement_id]

    def update_current_primary(self, type, name, identification_label='name', channel_id=None, breakout_id=None, breakout_channel_id=None):
        """
        updates the status of primary_measurement, set the timestamp of the current primary measurement to end at datetime.utcnow()

        Parameters
        ---------
        type: string
            type of the input unit (HPol, VPol, surfCABLE, ...)
        name: string
            the unique identifier of the input unit
        channel_id: int
                    if there is a channel id for the object, the id is used in the search filter mask
        """

        present_time = datetime.datetime.utcnow()

        # find the current primary measurement
        obj_id, measurement_id = self.find_primary_measurement(type, name, present_time, identification_label=identification_label, channel_id=channel_id, breakout_id=breakout_id, breakout_channel_id=breakout_channel_id)

        if obj_id is None and measurement_id[0] == 0:
            #  no primary measurement was found and thus there is no measurement to update
            pass
        else:
            for m_id in measurement_id:
                # get the old primary times
                filter_primary_times = [{'$match': {'_id': obj_id}},
                                        {'$unwind': '$measurements'},
                                        {'$match': {'measurements.id_measurement': m_id}}]

                info = list(self.db[type].aggregate(filter_primary_times))

                primary_times = info[0]['measurements']['primary_measurement']

                # update the 'end' time to the present time
                primary_times[-1]['end'] = present_time

                self.db[type].update_one({'_id': obj_id}, {"$set": {"measurements.$[updateIndex].primary_measurement": primary_times}}, array_filters=[{"updateIndex.id_measurement": m_id}])

    def get_object_names(self, object_type):
        return self.db[object_type].distinct('name')

    def get_collection_names(self):
        return self.db.list_collection_names()

    def create_empty_collection(self, collection_name):
        self.db.create_collection(collection_name)

    def clone_colletion_to_colletion(self, old_colletion, new_colletion):
        self.db[old_colletion].aggregate([{ '$match': {} }, { '$out': new_colletion}])

    def get_station_ids(self, collection):
        return self.db[collection].distinct('id')

    def __change_primary_object_measurement(self, object_type, object_name, search_filter, channel_id=None, breakout_id=None, breakout_channel_id=None):
        """

        helper function to change the current active primary measurement for a single antenna measurement

        Parameters
        ---------
        object_type: string
            type of the input unit (HPol, VPol, surfCABLE, ...)
        object_name: string
            the unique identifier of the object
        search_filter:
            specify the filter pipeline used for aggregate to find the measurement
        """

        present_time = datetime.datetime.utcnow()

        # get the information about the measurement specified in the search filter
        search_results = list(self.db[object_type].aggregate(search_filter))

        # extract the object and measurement id and current primary time array (only gives one measurement id)
        if len(search_results) > 1:
            logger.error('More than one measurement found.')
            object_id = None
            measurement_id = None
        elif len(search_results) == 0:
            logger.error('No measurement found.')
            object_id = None
            measurement_id = None
        else:
            object_id = search_results[0]['_id']
            measurement_id = search_results[0]['measurements']['id_measurement']
            primary_times = search_results[0]['measurements']['primary_measurement']

        # check if specified measurement is already the primary measurement (could be up to 4 measurement ids)
        current_obj_id, current_measurement_id = self.find_primary_measurement(object_type, object_name, present_time, channel_id=channel_id, breakout_id=breakout_id, breakout_channel_id=breakout_channel_id)
        for c_m_id in current_measurement_id:
            # find the current_measurement_id for the fitting S parameter
            filter_primary_times = [{'$match': {'_id': current_obj_id}},
                                    {'$unwind': '$measurements'},
                                    {'$match': {'measurements.id_measurement': c_m_id}}]

            info = list(self.db[object_type].aggregate(filter_primary_times))

            if info[0]['measurements']['S_parameter'] == search_results[0]['measurements']['S_parameter']:
                # the measurement id is fitting the S parameter

                if c_m_id == measurement_id and current_obj_id == object_id and measurement_id is not None:
                    logger.info('The specified measurement is already the primary measurement.')
                elif measurement_id is None or current_measurement_id is None:
                    pass
                else:
                    # update the old primary time (not using the 'update current primary measurement' function so that we can only update the entry of a single S parameter
                    primary_times_old = info[0]['measurements']['primary_measurement']
                    # # update the 'end' time to the present time
                    primary_times_old[-1]['end'] = present_time
                    self.db[object_type].update_one({'_id': object_id}, {"$set": {"measurements.$[updateIndex].primary_measurement": primary_times_old}}, array_filters=[{"updateIndex.id_measurement": c_m_id}])

                    # update the primary measurements of the specified measurements
                    if object_id is not None:
                        primary_times.append({'start': present_time, 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)})
                        self.db[object_type].update_one({'_id': object_id}, {"$set": {"measurements.$[updateIndex].primary_measurement": primary_times}}, array_filters=[{"updateIndex.id_measurement": measurement_id}])
            else:
                logger.error('S parameter not selected to be changed.')

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

        # close the time period of the old primary measurement
        if primary_measurement and antenna_name in self.get_object_names(antenna_type):
            self.update_current_primary(antenna_type, antenna_name)

        # define the new primary measurement times
        if primary_measurement:
            primary_measurement_times = [{'start': datetime.datetime.utcnow(), 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)}]
        else:
            primary_measurement_times = []

        # update the entry with the measurement (if the entry doesn't exist it will be created)
        for spara in S_parameter:
            self.db[antenna_type].update_one({'name': antenna_name},
                                             {'$push': {'measurements': {
                                                    'id_measurement': ObjectId(),
                                                    'function_test': True,
                                                    'last_updated': datetime.datetime.utcnow(),
                                                    'primary_measurement': primary_measurement_times,
                                                    'measurement_protocol': protocol,
                                                    'S_parameter': spara,
                                                    'y-axis_units': [units_arr[1]],
                                                    'frequencies': list(S_data[0]),
                                                    'mag': list(S_data[1])
                                             }}}, upsert=True)

    def change_primary_antenna_measurement(self, antenna_type, antenna_name, S_parameter, protocol, units_arr, function_test):
        """
        changes the current active primary measurement for a single antenna measurement

        Parameters
        ---------
        antenna_type: string
            specify if it is a VPol or HPol antenna
        antenna_name: string
            the unique identifier of the antenna
        S_parameter: list of strings
            specify which S_parameter is used (S11, ...)
        protocol: string
            details of the testing environment
        units_arr: list of strings
            list of the input units (only y unit will be saved)
        function_test: boolean
            describes if the channel is working or not
        """

        # find the entry specified by function arguments
        search_filter = [{'$match': {'name': antenna_name}},
                         {'$unwind': '$measurements'},
                         {'$match': {'measurements.function_test': function_test,
                                     'measurements.measurement_protocol': protocol,
                                     'measurements.S_parameter': S_parameter,
                                     'measurements.y-axis_units': units_arr}}]

        self.__change_primary_object_measurement(antenna_type, antenna_name, search_filter)

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

        # close the time period of the old primary measurement
        if primary_measurement and cable_name in self.get_object_names(cable_type):
            self.update_current_primary(cable_type, cable_name)

        # define the new primary measurement times
        if primary_measurement:
            primary_measurement_times = [{'start': datetime.datetime.utcnow(), 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)}]
        else:
            primary_measurement_times = []

        # update the entry with the measurement (if the entry doesn't exist it will be created)
        for spara in S_parameter:
            self.db[cable_type].update_one({'name': cable_name},
                                             {'$push': {'measurements': {
                                                 'id_measurement': ObjectId(),
                                                 'function_test': True,
                                                 'last_updated': datetime.datetime.utcnow(),
                                                 'primary_measurement': primary_measurement_times,
                                                 'measurement_protocol': protocol,
                                                 'S_parameter': spara,
                                                 'y-axis_units': [units_arr[1], units_arr[2]],
                                                 'frequencies': list(Sm_data[0]),
                                                 'mag': list(Sm_data[1]),
                                                 'phase': list(Sp_data[1])
                                             }}}, upsert=True)

    def change_primary_cable_measurement(self, cable_type, cable_name, S_parameter, protocol, units_arr, function_test):
        """
        changes the current active primary measurement for a single cable measurement

        Parameters
        ---------
        cable_type: string
            specify if it is a surface or downhole cable
        cable_name: string
            the unique identifier of the cable
        S_parameter: list of strings
            specify which S_parameter is used (S11, ...)
        protocol: string
            details of the testing environment
        units_arr: list of strings
            list of the input units (only y unit will be saved)
        function_test: boolean
            describes if the channel is working or not
        """

        # find the entry specified by function arguments
        search_filter = [{'$match': {'name': cable_name}},
                         {'$unwind': '$measurements'},
                         {'$match': {'measurements.function_test': function_test,
                                     'measurements.measurement_protocol': protocol,
                                     'measurements.S_parameter': S_parameter,
                                     'measurements.y-axis_units': units_arr}}]

        self.__change_primary_object_measurement(cable_type, cable_name, search_filter)

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

        # close the time period of the old primary measurement
        if primary_measurement and board_name in self.get_object_names(page_name):
            self.update_current_primary(page_name, board_name)

        # define the new primary measurement times
        if primary_measurement:
            primary_measurement_times = [{'start': datetime.datetime.utcnow(), 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)}]
        else:
            primary_measurement_times = []

        # update the entry with the measurement (if the entry doesn't exist it will be created)
        for i, spara in enumerate(S_names):
            self.db[page_name].update_one({'name': board_name},
                                            {'$push': {'measurements': {
                                               'id_measurement': ObjectId(),
                                               'function_test': True,
                                               'last_updated': datetime.datetime.utcnow(),
                                               'primary_measurement': primary_measurement_times,
                                               'measurement_protocol': protocol,
                                               'S_parameter': spara,
                                               'DRAB_id': drab_id,
                                               'laser_id': laser_id,
                                               'measurement_temp': temp,
                                               'time_delay': time_delay[i],
                                               'measurement_time': measurement_time,
                                               'y-axis_units': [units_arr[1], units_arr[2]],
                                               'frequencies': list(S_data[0]),
                                               'mag': list(S_data[2*i+1]),
                                               'phase': list(S_data[2*i+2])
                                           }}}, upsert=True)

    def change_primary_iglu_measurement(self, board_type, board_name, S_parameter, protocol, units_arr, function_test, drab_id, laser_id, temperature):
        """
        changes the current active primary measurement for a single board measurement

        Parameters
        ---------
        board_type: string
            specify the board type
        board_name: string
            the unique identifier of the board
        S_parameter: list of strings
            specify which S_parameter is used (S11, ...)
        protocol: string
            details of the testing environment
        units_arr: list of strings
            list of the input units (only y unit will be saved)
        function_test: boolean
            describes if the channel is working or not
        """

        # find the entry specified by function arguments
        search_filter = [{'$match': {'name': board_name}},
                         {'$unwind': '$measurements'},
                         {'$match': {'measurements.function_test': function_test,
                                     'measurements.measurement_protocol': protocol,
                                     'measurements.S_parameter': S_parameter,
                                     'measurements.DRAB_id': drab_id,
                                     'measurements.laser_id': laser_id,
                                     'measurements.measurement_temp': temperature,
                                     'measurements.y-axis_units': units_arr
                                     }}]

        self.__change_primary_object_measurement(board_type, board_name, search_filter)

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
        # close the time period of the old primary measurement
        if primary_measurement and board_name in self.get_object_names(page_name):
            self.update_current_primary(page_name, board_name, channel_id=channel_id)

        # define the new primary measurement times
        if primary_measurement:
            primary_measurement_times = [{'start': datetime.datetime.utcnow(), 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)}]
        else:
            primary_measurement_times = []

        # update the entry with the measurement (if the entry doesn't exist it will be created)
        for i, spara in enumerate(S_names):
            self.db[page_name].update_one({'name': board_name},
                                          {'$push': {'measurements': {
                                              'id_measurement': ObjectId(),
                                              'function_test': True,
                                              'last_updated': datetime.datetime.utcnow(),
                                              'primary_measurement': primary_measurement_times,
                                              'measurement_protocol': protocol,
                                              'S_parameter': spara,
                                              'IGLU_id': iglu_id,
                                              'photodiode_serial': photodiode_id,
                                              'channel_id': channel_id,
                                              'measurement_temp': temp,
                                              'time_delay': time_delay[i],
                                              'measurement_time': measurement_time,
                                              'y-axis_units': [units_arr[1], units_arr[2]],
                                              'frequencies': list(S_data[0]),
                                              'mag': list(S_data[2 * i + 1]),
                                              'phase': list(S_data[2 * i + 2])
                                          }}}, upsert=True)

    def change_primary_drab_measurement(self, board_type, board_name, S_parameter, iglu_id, photodiode_id, channel_id, temp, protocol, units_arr, function_test):
        """
        changes the current active primary measurement for a single board measurement

        Parameters
        ---------
        board_type: string
            specify the board type
        board_name: string
            the unique identifier of the board
        S_parameter: list of strings
            specify which S_parameter is used (S11, ...)
        protocol: string
            details of the testing environment
        units_arr: list of strings
            list of the input units (only y unit will be saved)
        function_test: boolean
            describes if the channel is working or not
        """

        # find the entry specified by function arguments
        search_filter = [{'$match': {'name': board_name}},
                         {'$unwind': '$measurements'},
                         {'$match': {'measurements.function_test': function_test,
                                     'measurements.measurement_protocol': protocol,
                                     'measurements.S_parameter': S_parameter,
                                     'measurements.IGLU_id': iglu_id,
                                     'measurements.photodiode_serial': photodiode_id,
                                     'measurements.channel_id': channel_id,
                                     'measurements.measurement_temp': temp,
                                     'measurements.y-axis_units': units_arr
                                     }}]

        self.__change_primary_object_measurement(board_type, board_name, search_filter, channel_id=channel_id)

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
        # close the time period of the old primary measurement
        if primary_measurement and board_name in self.get_object_names(page_name):
            self.update_current_primary(page_name, board_name, channel_id=channel_id)

        # define the new primary measurement times
        if primary_measurement:
            primary_measurement_times = [{'start': datetime.datetime.utcnow(), 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)}]
        else:
            primary_measurement_times = []

        # update the entry with the measurement (if the entry doesn't exist it will be created)
        for i, spara in enumerate(S_names):
            self.db[page_name].update_one({'name': board_name},
                                          {'$push': {'measurements': {
                                              'id_measurement': ObjectId(),
                                              'function_test': True,
                                              'last_updated': datetime.datetime.utcnow(),
                                              'primary_measurement': primary_measurement_times,
                                              'measurement_protocol': protocol,
                                              'S_parameter': spara,
                                              'channel_id': channel_id,
                                              'measurement_temp': temp,
                                              'time_delay': time_delay[i],
                                              'measurement_time': measurement_time,
                                              'y-axis_units': [units_arr[1], units_arr[2]],
                                              'frequencies': list(S_data[0]),
                                              'mag': list(S_data[2 * i + 1]),
                                              'phase': list(S_data[2 * i + 2])
                                          }}}, upsert=True)

    def change_primary_surface_measurement(self, board_type, board_name, S_parameter, channel_id, temp, protocol, units_arr, function_test):
        """
        changes the current active primary measurement for a single board measurement

        Parameters
        ---------
        board_type: string
            specify the board type
        board_name: string
            the unique identifier of the board
        S_parameter: list of strings
            specify which S_parameter is used (S11, ...)
        protocol: string
            details of the testing environment
        units_arr: list of strings
            list of the input units (only y unit will be saved)
        function_test: boolean
            describes if the channel is working or not
        """

        # find the entry specified by function arguments
        search_filter = [{'$match': {'name': board_name}},
                         {'$unwind': '$measurements'},
                         {'$match': {'measurements.function_test': function_test,
                                     'measurements.measurement_protocol': protocol,
                                     'measurements.S_parameter': S_parameter,
                                     'measurements.channel_id': channel_id,
                                     'measurements.measurement_temp': temp,
                                     'measurements.y-axis_units': units_arr
                                     }}]

        self.__change_primary_object_measurement(board_type, board_name, search_filter, channel_id=channel_id)

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
        # close the time period of the old primary measurement
        if primary_measurement and board_name in self.get_object_names(page_name):
            self.update_current_primary(page_name, board_name, breakout_id=breakout_id, breakout_channel_id=breakout_cha_id)

        # define the new primary measurement times
        if primary_measurement:
            primary_measurement_times = [{'start': datetime.datetime.utcnow(), 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)}]
        else:
            primary_measurement_times = []

        # update the entry with the measurement (if the entry doesn't exist it will be created)
        for i, spara in enumerate(S_names):
            self.db[page_name].update_one({'name': board_name},
                                          {'$push': {'measurements': {
                                              'id_measurement': ObjectId(),
                                              'function_test': True,
                                              'last_updated': datetime.datetime.utcnow(),
                                              'primary_measurement': primary_measurement_times,
                                              'measurement_protocol': protocol,
                                              'S_parameter': spara,
                                              'IGLU_id': iglu_id,
                                              'DRAB_id': drab_id,
                                              'breakout': breakout_id,
                                              'breakout_channel': breakout_cha_id,
                                              'measurement_temp': temp,
                                              'time_delay': time_delay[i],
                                              'measurement_time': measurement_time,
                                              'y-axis_units': [units_arr[1], units_arr[2]],
                                              'frequencies': list(S_data[0]),
                                              'mag': list(S_data[2 * i + 1]),
                                              'phase': list(S_data[2 * i + 2])
                                          }}}, upsert=True)

    def change_primary_downhole_measurement(self, board_type, board_name, S_parameter, breakout_id, breakout_cha_id, iglu_id, drab_id, temp, protocol, units_arr, function_test):
        """
        changes the current active primary measurement for a single board measurement

        Parameters
        ---------
        board_type: string
            specify the board type
        board_name: string
            the unique identifier of the board
        S_parameter: list of strings
            specify which S_parameter is used (S11, ...)
        protocol: string
            details of the testing environment
        units_arr: list of strings
            list of the input units (only y unit will be saved)
        function_test: boolean
            describes if the channel is working or not
        """

        # find the entry specified by function arguments
        search_filter = [{'$match': {'name': board_name}},
                         {'$unwind': '$measurements'},
                         {'$match': {'measurements.function_test': function_test,
                                     'measurements.measurement_protocol': protocol,
                                     'measurements.S_parameter': S_parameter,
                                     'measurements.IGLU_id': iglu_id,
                                     'measurements.DRAB_id': drab_id,
                                     'measurements.breakout': breakout_id,
                                     'measurements.breakout_channel': breakout_cha_id,
                                     'measurements.measurement_temp': temp,
                                     'measurements.y-axis_units': units_arr
                                     }}]

        self.__change_primary_object_measurement(board_type, board_name, search_filter, breakout_id=breakout_id, breakout_channel_id=breakout_cha_id)

    # stations (general)

    def decommission_a_station(self, collection, station_id, decomm_time):
        """
        function to decommission an active station in the db

        Parameters
        ---------
        collection: string
            name of the collection
        station_id: int
            the unique identifier of the station
        decomm_time: datetime
            time which should be used for updating the decommission time
        """
        # get the entry of the aktive station
        if self.db[collection].count_documents({'id': station_id}) == 0:
            logger.error(f'No active station {station_id} in the database')
        else:
            # filter to get all active stations with the correct id
            time = self.__current_time
            time_filter = [{"$match": {
                'commission_time': {"$lte": time},
                'decommission_time': {"$gte": time},
                'id': station_id}}]
            # get all stations which fit the filter (should only be one)
            stations = list(self.db[collection].aggregate(time_filter))
            if len(stations) > 1:
                logger.error('More than one active station was found.')
            else:
                object_id = stations[0]['_id']

                # change the commission/decomission time
                self.db[collection].update_one({'_id': object_id}, {'$set': {'decommission_time': decomm_time}})

    def add_general_station_info(self, collection, station_id, station_name, station_comment, commission_time, decommission_time=datetime.datetime(2080, 1, 1)):
        # check if an active station exist; if true, the active station will be decommissioned
        # filter to get all active stations with the correct id
        time = self.__current_time
        time_filter = [{"$match": {
            'commission_time': {"$lte": time},
            'decommission_time': {"$gte": time},
            'id': station_id}}]
        # get all stations which fit the filter (should only be one)
        stations = list(self.db[collection].aggregate(time_filter))

        if len(stations) > 0:
            self.decommission_a_station(collection, station_id, commission_time)

        # insert the new station
        self.db[collection].insert_one({'id': station_id,
                                    'name': station_name,
                                    'commission_time': commission_time,
                                    'decommission_time': decommission_time,
                                    'station_comment': station_comment
                                    })

    def decommission_a_channel(self, collection, station_id, channel_id, decomm_time):
        """
        function to decommission an active channel in the db

        Parameters
        ---------
        collection: string
            name of the collection
        station_id: int
            the unique identifier of the station
        channel_id: int
            the unique identifier of the channel
        decomm_time: datetime
            time which should be used for updating the decommission time
        """
        # get the entry of the aktive station
        if self.db[collection].count_documents({'id': station_id}) == 0:
            logger.error(f'No active station {station_id} in the database')
        else:
            # filter to get all active stations with the correct id
            time = self.__current_time
            time_filter = [{"$match": {
                'commission_time': {"$lte": time},
                'decommission_time': {"$gte": time},
                'id': station_id}}]
            # get all stations which fit the filter (should only be one)
            stations = list(self.db[collection].aggregate(time_filter))
            if len(stations) > 1:
                logger.error('More than one active station was found.')
            else:
                object_id = stations[0]['_id']

                # change the decommission time of a specific channel
                self.db[collection].update_one({'_id': object_id}, {'$set': {'channels.$[updateIndex].decommission_time': decomm_time}},
                                               array_filters=[{"updateIndex.id": channel_id}])

    def add_general_channel_info_to_station(self, collection, station_id, channel_id, signal_chain, ant_name, channel_type, channel_comment, commission_time, decommission_time=datetime.datetime(2080, 1, 1)):
        # get the current active station
        # filter to get all active stations with the correct id
        time = self.__current_time
        time_filter = [{"$match": {
            'commission_time': {"$lte": time},
            'decommission_time': {"$gte": time},
            'id': station_id}}]
        # get all stations which fit the filter (should only be one)
        stations = list(self.db[collection].aggregate(time_filter))

        if len(stations) != 1:
            logger.error('More than one or no active stations in the database')
            return 1

        unique_station_id = stations[0]['_id']

        # check if for this channel an entry already exists
        component_filter = [{'$match': {'_id': unique_station_id}},
                            {'$unwind': '$channels'},
                            {'$match': {'channels.id': channel_id}}]

        entries = list(self.db[collection].aggregate(component_filter))

        # check if the channel already exist, decommission the active channel first
        if entries != []:
            self.decommission_a_channel(collection, station_id, channel_id, commission_time)

        # insert the channel information
        self.db[collection].update_one({'_id': unique_station_id},
                               {"$push": {'channels': {
                                   'id': channel_id,
                                   'ant_name': ant_name,
                                   'type': channel_type,
                                   'commission_time': commission_time,
                                   'decommission_time': decommission_time,
                                   'signal_ch': signal_chain,
                                   'channel_comment': channel_comment
                                   }}
                               })

    def get_general_station_information(self, collection, station_id):
        """ get information from one station """

        # if the collection is empty, return an empty dict
        if self.db[collection].count_documents({'id': station_id}) == 0:
            return {}

        # filter to get all information from one station with station_id and with akitve commission time
        time = self.__current_time
        if time is None:
            logger.error('For the detector is no time set!')

        time_filter = [{"$match": {
            'commission_time': {"$lte": time},
            'decommission_time': {"$gte": time},
            'id': station_id}}]
        # get all stations which fit the filter (should only be one)
        stations_for_buffer = list(self.db[collection].aggregate(time_filter))

        # transform the output of db.aggregate to a dict
        station_info = dictionarize_nested_lists(stations_for_buffer, parent_key="id", nested_field="channels", nested_key="id")

        if 'channels' not in station_info[station_id].keys():
            station_info[station_id]['channels'] = {}

        return station_info

    # stations (position)

    def add_station_position(self, station_id, measurement_name, measurement_time, position, primary):
        """
        inserts a position measurement for a station into the database
        If the station dosn't exist yet, it will be created.

        Parameters
        ---------
        station_id: int
            the unique identifier of the station
        measurement_name: string
            the unique name of the position measurement
        measurement_time: string
            the time when the measurement was conducted
        position: list of floats
            the measured position of the three strings
        primary: bool
            indicates if the measurement will be used as the primary measurement from now on
        """
        collection_name = 'station_position'
        # close the time period of the old primary measurement
        if primary and station_id in self.db[collection_name].distinct('id'):
            self.update_current_primary(collection_name, station_id, identification_label='id')

        # define the new primary measurement times
        if primary:
            primary_measurement_times = [{'start': datetime.datetime.utcnow(), 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)}]
        else:
            primary_measurement_times = []

        # update the entry with the measurement (if the entry doesn't exist it will be created)
        self.db[collection_name].update_one({'id': station_id},
                                      {'$push': {'measurements': {
                                          'id_measurement': ObjectId(),
                                          'measurement_name': measurement_name,
                                          'last_updated': datetime.datetime.utcnow(),
                                          'primary_measurement': primary_measurement_times,
                                          'position': position,
                                          'measurement_time': measurement_time
                                      }}}, upsert=True)

    def change_primary_station_measurement(self):
        pass

    def get_collection_information(self, collection_name, station_id, primary_time=None, measurement_name=None, channel_id=None):
        """
        get the information for a specified collection (will only work for 'station_position', 'channel_position' and 'signal_chain')
        if the station does not exist, {} will be returned
        default: primary_time = current time and no measurement_name

        Parameters
        ---------
        collection_name: string
            specify the collection, from which the information should be extracted (will only work for 'station_position', 'channel_position' and 'signal_chain')
        station_id: int
            the unique identifier of the station
        primary_time: datetime.datetime
            elements which are/were primary at this time are selected
        measurement_name: string
            the unique name of the measurement
        channel_id: int
            unique identifier of the channel

        Returns
        ---------
        info
        """
        # if the collection is empty, return an empty dict
        if self.db[collection_name].count_documents({'id': station_id}) == 0:
            return {}

        # define the search filter
        if primary_time is not None:
            if measurement_name is None:
                if channel_id is None:
                    search_filter = [{'$match': {'id': station_id}},
                                     {'$unwind': '$measurements'},
                                     {'$unwind': '$measurements.primary_measurement'},
                                     {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                                 'measurements.primary_measurement.end': {'$gte': primary_time}}}]
                else:  # channel_id is not None
                    search_filter = [{'$match': {'id': station_id}},
                                     {'$unwind': '$measurements'},
                                     {'$match': {'measurements.channel_id': channel_id}},
                                     {'$unwind': '$measurements.primary_measurement'},
                                     {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                                 'measurements.primary_measurement.end': {'$gte': primary_time}}}]
            else:  # measurement_name is not None
                if channel_id is None:
                    search_filter = [{'$match': {'id': station_id}},
                                     {'$unwind': '$measurements'},
                                     {'$match': {'measurements.measurement_name': measurement_name}},
                                     {'$unwind': '$measurements.primary_measurement'},
                                     {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                                 'measurements.primary_measurement.end': {'$gte': primary_time}}}]
                else:  # channel_id is not None
                    search_filter = [{'$match': {'id': station_id}},
                                     {'$unwind': '$measurements'},
                                     {'$match': {'measurements.measurement_name': measurement_name,
                                                 'measurements.channel_id': channel_id}},
                                     {'$unwind': '$measurements.primary_measurement'},
                                     {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                                 'measurements.primary_measurement.end': {'$gte': primary_time}}}]
        else:  # primary time is None
            if measurement_name is not None:
                if channel_id is not None:
                    search_filter = [{'$match': {'id': station_id}},
                                     {'$unwind': '$measurements'},
                                     {'$match': {'measurements.measurement_name': measurement_name,
                                                 'measurements.channel_id': channel_id}}]
                else:  # channel_id is None
                    search_filter = [{'$match': {'id': station_id}},
                                     {'$unwind': '$measurements'},
                                     {'$match': {'measurements.measurement_name': measurement_name}}]
            else:  # measurement_name is None
                # take the current time as primary time and do not specify measurement name
                primary_time = datetime.datetime.utcnow()
                if channel_id is not None:
                    search_filter = [{'$match': {'id': station_id}},
                                     {'$unwind': '$measurements'},
                                     {'$match': {'measurements.channel_id': channel_id}},
                                     {'$unwind': '$measurements.primary_measurement'},
                                     {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                                 'measurements.primary_measurement.end': {'$gte': primary_time}}}]
                else:  # channel_id is None
                    search_filter = [{'$match': {'id': station_id}},
                                     {'$unwind': '$measurements'},
                                     {'$unwind': '$measurements.primary_measurement'},
                                     {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                                 'measurements.primary_measurement.end': {'$gte': primary_time}}}]

        search_result = list(self.db[collection_name].aggregate(search_filter))

        if search_result == []:
            return search_result

        # extract the measurement and object id
        object_id = []
        measurement_id = []
        for dic in search_result:
            object_id.append(dic['_id'])
            measurement_id.append(dic['measurements']['id_measurement'])

        # extract the information using the object and measurements id
        id_filter = [{'$match': {'_id': {'$in': object_id}}},
                     {'$unwind': '$measurements'},
                     {'$match': {'measurements.id_measurement': {'$in': measurement_id}}}]
        info = list(self.db[collection_name].aggregate(id_filter))

        return info

    def get_quantity_names(self, collection_name, wanted_quantity):
        """ returns a list with all measurement names, ids, ... or what is specified (example: wanted_quantity = measurements.measurement_name)"""
        return self.db[collection_name].distinct(wanted_quantity)

    # channels

    def add_channel_position(self, station_id, channel_number, measurement_name, measurement_time, position, orientation, rotation, primary):
        """
        inserts a position measurement for a channel into the database
        If the station dosn't exist yet, it will be created.

        Parameters
        ---------
        station_id: int
            the unique identifier of the station the channel belongs to
        channel_number: int
            unique identifier of the channel
        measurement_name: string
            the unique name of the position measurement
        measurement_time: string
            the time when the measurement was conducted
        position: list of floats
            the measured position of the channel
        orientation: dict
            orientation of the channel
        rotation: dict
            rotation of the channel
        primary: bool
            indicates if the measurement will be used as the primary measurement from now on
        """
        collection_name = 'channel_position'
        # close the time period of the old primary measurement
        if primary and station_id in self.db[collection_name].distinct('id'):
            self.update_current_primary(collection_name, station_id, identification_label='id', channel_id=channel_number)

        # define the new primary measurement times
        if primary:
            primary_measurement_times = [{'start': datetime.datetime.utcnow(), 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)}]
        else:
            primary_measurement_times = []

        # update the entry with the measurement (if the entry doesn't exist it will be created)
        self.db[collection_name].update_one({'id': station_id},
                                      {'$push': {'measurements': {
                                          'id_measurement': ObjectId(),
                                          'channel_id': channel_number,
                                          'measurement_name': measurement_name,
                                          'last_updated': datetime.datetime.utcnow(),
                                          'primary_measurement': primary_measurement_times,
                                          'position': position,
                                          'rotation': rotation,
                                          'orientation': orientation,
                                          'measurement_time': measurement_time
                                      }}}, upsert=True)

    def change_primary_channel_measurement(self):
        pass

    def add_channel_signal_chain(self, station_id, channel_number, config_name, sig_chain, primary, primary_components):
        """
        inserts a signal chain config for a channel into the database
        If the station dosn't exist yet, it will be created.

        Parameters
        ---------
        station_id: int
            the unique identifier of the station the channel belongs to
        channel_number: int
            unique identifier of the channel
        config_name: string
            the unique name of the signal chain configuration
        sig_chain: list of strings
            list of strings describing the signal chain
        primary: bool
            indicates if the measurement will be used as the primary measurement from now on
        primary_components: dict
            dates which say which measurement for each single component is used
        """
        collection_name = 'signal_chain'
        # close the time period of the old primary measurement
        if primary and station_id in self.db[collection_name].distinct('id'):
            self.update_current_primary(collection_name, station_id, identification_label='id', channel_id=channel_number)

        # define the new primary measurement times
        if primary:
            primary_measurement_times = [{'start': datetime.datetime.utcnow(), 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)}]
        else:
            primary_measurement_times = []

        # update the entry with the measurement (if the entry doesn't exist it will be created)
        self.db[collection_name].update_one({'id': station_id},
                                      {'$push': {'measurements': {
                                          'id_measurement': ObjectId(),
                                          'channel_id': channel_number,
                                          'measurement_name': config_name,
                                          'last_updated': datetime.datetime.utcnow(),
                                          'primary_measurement': primary_measurement_times,
                                          'sig_chain': sig_chain,
                                          'primary_components': primary_components
                                      }}}, upsert=True)

    def get_all_available_signal_chain_configs_old(self, collection, input_dic):
        """depending on the inputs, all possible configurations in the database are returned; Input example: {'name': 'Golden_IGLU', 'measurement_temp': 20}"""
        return_dic = {}
        check_value = True
        for key in input_dic.keys():
            if input_dic[key] != '':
                check_value = False
        if check_value:
            for key in input_dic.keys():
                if key == 'name':
                    return_dic[key] = self.get_quantity_names(collection, key)
                else:
                    return_dic[key] = self.get_quantity_names(collection, f'measurements.{key}')
        else:
            # define a search filter
            search_filter = []
            if 'name' in input_dic and input_dic['name'] != '':
                search_filter.append({'$match': {'name': input_dic['name']}})
            search_filter.append({'$unwind': '$measurements'})
            help_dic1 = {}
            help_dic2 = {}
            for key in input_dic.keys():
                if key != 'name' and input_dic[key] != '':
                    help_dic2[f'measurements.{key}'] = input_dic[key]
            if help_dic2 != {}:
                help_dic1['$match'] = help_dic2
                search_filter.append(help_dic1)

            search_result = list(self.db[collection].aggregate(search_filter))

            for key in input_dic.keys():
                help_list = []
                for entry in search_result:
                    if key in entry.keys():
                        help_list.append(entry[key])
                    else:
                        help_list.append(entry['measurements'][key])
                return_dic[key] = list(set(help_list))

        return return_dic

    def get_all_available_signal_chain_configs(self, collection, object_name, input_dic):
        """depending on the inputs, all possible configurations in the database are returned; Input example: 'iglu_boards', 'Golden_IGLU' {'measurement_temp': 20, 'DRAB_id': 'Golden_DRAB'}"""
        return_dic = {}
        if object_name is None:
            for key in input_dic.keys():
                return_dic[key] = self.get_quantity_names(collection, f'measurements.{key}')
        else:
            # define a search filter
            search_filter = []
            search_filter.append({'$match': {'name': object_name}})
            search_filter.append({'$unwind': '$measurements'})
            help_dic1 = {}
            help_dic2 = {}
            for key in input_dic.keys():
                if input_dic[key] is not None:
                    help_dic2[f'measurements.{key}'] = input_dic[key]
            if help_dic2 != {}:
                help_dic1['$match'] = help_dic2
                search_filter.append(help_dic1)
            print(search_filter)
            search_result = list(self.db[collection].aggregate(search_filter))

            for key in input_dic.keys():
                help_list = []
                for entry in search_result:
                    help_list.append(entry['measurements'][key])
                return_dic[key] = list(set(help_list))
        print(return_dic)
        return return_dic

    def change_primary_channel_signal_chain_configuration(self):
        pass

    def get_complete_station_information(self, station_id, primary_time=None, measurement_position=None, measurement_channel_position=None, measurement_signal_chain=None, measurement_device=None):
        """
        collects all available information about the station

        Parameters
        ---------
        station_id: int
            the unique identifier of the station the channel belongs to
        primary_time: datetime.datetime
            time used to check for the primary measurement
            if None and no measurement name given: the current time
        measurement_position: string
            if given (and primary=None) the measurement will be collected (even though it is not the primary measurement)
        measurement_channel_position: string
            if given (and primary=None) the measurement will be collected (even though it is not the primary measurement)
        measurement_signal_chain: string
            if given (and primary=None) the measurement will be collected (even though it is not the primary measurement)
        measurement_device: string
            if given (and primary=None) the measurement will be collected (even though it is not the primary measurement)

        Returns
        -------
        complete_info
        """
        complete_info = {}

        # load general station information
        general_info = self.get_general_station_information('station_rnog', station_id)

        # load the station position, channel position, signal_chain and device_position information
        if primary_time is None:
            if measurement_position is None:
                primary_time = datetime.datetime.utcnow()
                station_position_information = self.get_collection_information('station_position', station_id, primary_time=primary_time)
            else:  # measurement_position is not None
                station_position_information = self.get_collection_information('station_position', station_id, primary_time=None, measurement_name=measurement_position)
            if measurement_channel_position is None:
                primary_time = datetime.datetime.utcnow()
                channel_position_information = self.get_collection_information('channel_position', station_id, primary_time=primary_time)
            else:
                channel_position_information = self.get_collection_information('channel_position', station_id, primary_time=None, measurement_name=measurement_channel_position)
            if measurement_signal_chain is None:
                primary_time = datetime.datetime.utcnow()
                signal_chain_information = self.get_collection_information('signal_chain', station_id, primary_time=primary_time)
            else:
                signal_chain_information = self.get_collection_information('signal_chain', station_id, primary_time=None, measurement_name=measurement_signal_chain)
            if measurement_device is None:
                primary_time = datetime.datetime.utcnow()
                device_position_information = []
            else:
                device_position_information = []
        else:  # primary_time is not None
            station_position_information = self.get_collection_information('station_position', station_id, primary_time=primary_time)
            channel_position_information = self.get_collection_information('channel_position', station_id, primary_time=primary_time)
            signal_chain_information = self.get_collection_information('signal_chain', station_id, primary_time=primary_time)
            device_position_information = []

        # combine the station information
        general_station_dic = {k: general_info[station_id][k] for k in set(list(general_info[station_id].keys())) - set(['channels'])}  # create dic with all entries except 'channels'
        sta_pos_measurement_info = station_position_information[0]['measurements']
        complete_info.update(general_station_dic)
        complete_info['station_position'] = sta_pos_measurement_info

        # combine channel information
        general_channel_dic={}
        exclude_keys = ['signal_ch', 'id']
        for cha_key in general_info[station_id]['channels'].keys():
            help_dic = general_info[station_id]['channels'][cha_key]
            general_channel_dic[cha_key] = {k: help_dic[k] for k in set(list(help_dic.keys())) - set(exclude_keys)}

        # get the channel position information in the correct format
        channel_pos_dic = {}
        for cha_pos_dic in channel_position_information:
            channel_id = cha_pos_dic['measurements']['channel_id']
            channel_pos_dic[channel_id] = {k: cha_pos_dic['measurements'][k] for k in set(list(cha_pos_dic['measurements'].keys())) - set(['channel_id'])}

        # get the channel signal chain in the correct format
        channel_sig_chain_dic = {}
        for cha_sig_dic in signal_chain_information:
            channel_id = cha_sig_dic['measurements']['channel_id']
            channel_sig_chain_dic[channel_id] = {k: cha_sig_dic['measurements'][k] for k in set(list(cha_sig_dic['measurements'].keys())) - set(['channel_id'])}

        # got through the signal chain and collect the corresponding measurements
        for cha_id in channel_sig_chain_dic.keys():
            sig_chain = channel_sig_chain_dic[cha_id]['sig_chain']
            print(channel_sig_chain_dic[cha_id])
            # get all collection names to identify the different components
            collection_names = self.get_collection_names()
            measurement_components_dic = {}
            for sig_chain_key in sig_chain.keys():
                if sig_chain_key in collection_names:
                    # search if supplementary info (channel-id, etc.) are saved, if this is the case -> extract the information (important to find the final measurement)
                    supp_info = []
                    for sck in sig_chain.keys():
                        if sig_chain_key in sck and sig_chain_key != sck:
                            supp_info.append(sck)
                    # get the primary time of the component measurement -> important to find the measurement which should be used
                    primary_component = channel_sig_chain_dic[cha_id]['primary_components'][sig_chain_key]
                    # define a search filter
                    search_filter_sig_chain = []
                    search_filter_sig_chain.append({'$match': {'name': sig_chain[sig_chain_key]}})
                    search_filter_sig_chain.append({'$unwind': '$measurements'})
                    # if there is supp info -> add this to the search filter
                    help_dic1 = {}
                    help_dic2 = {}
                    if supp_info != []:
                        for si in supp_info:
                            help_dic2[f'measurements.{si[len(sig_chain_key) +1:]}'] = sig_chain[si]

                    if len(self.get_quantity_names(collection_name=sig_chain_key, wanted_quantity='measurements.S_parameter')) != 1:  # more than one S parameter is saved in the database
                        help_dic2['measurements.S_parameter'] = 'S21'
                    if help_dic2 != {}:
                        help_dic1['$match'] = help_dic2
                        search_filter_sig_chain.append(help_dic1)
                    # add the primary time
                    search_filter_sig_chain.append({'$unwind': '$measurements.primary_measurement'})
                    search_filter_sig_chain.append({'$match': {'measurements.primary_measurement.start': {'$lte': primary_component},
                                                               'measurements.primary_measurement.end': {'$gte': primary_component}}})

                    # find the correct measurement in the database and extract the measurement and object id
                    search_result_sig_chain = list(self.db[sig_chain_key].aggregate(search_filter_sig_chain))
                    freq = search_result_sig_chain[0]['measurements']['frequencies']  # 0: should only return a single valid entry
                    yunits = search_result_sig_chain[0]['measurements']['y-axis_units']
                    ydata = []
                    if 'mag' in search_result_sig_chain[0]['measurements'].keys():
                        ydata.append(search_result_sig_chain[0]['measurements']['mag'])
                    if 'phase' in search_result_sig_chain[0]['measurements'].keys():
                        ydata.append(search_result_sig_chain[0]['measurements']['phase'])

                    measurement_components_dic[sig_chain_key] = {'y_units': yunits, 'freq': freq, 'ydata': ydata}
            channel_sig_chain_dic[cha_id]['measurements_components'] = measurement_components_dic

        for cha_id in general_channel_dic.keys():
            if cha_id not in channel_pos_dic.keys():
                general_channel_dic[cha_id]['channel_position'] = {}
            else:
                general_channel_dic[cha_id]['channel_position'] = channel_pos_dic[cha_id]
            if cha_id not in channel_sig_chain_dic.keys():
                general_channel_dic[cha_id]['channel_signal_chain'] = {}
            else:
                general_channel_dic[cha_id]['channel_signal_chain'] = channel_sig_chain_dic[cha_id]

        complete_info['channels'] = general_channel_dic
        complete_info['devices'] = {}

        return complete_info

    def get_complete_channel_information(self):
        pass


    # devices (pulser, DAQ, windturbine, solar panels)
    # TODO: store the position information of these devices in a separate collection
    # TODO: make them readable by readout collection inofrmation

    def add_position_device_information(self):
        pass

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

    # runTable
    def get_runs(self, station_list, start_time, end_time, flag_list, trigger_list, min_duration, firmware_list):
        """ return all database entries fitting to the input parameter """

        # trigger
        trig_rf0 = 0
        trig_rf1 = 0
        trig_ext = 0
        trig_pps = 0
        trig_soft = 0
        if 'rf0' in trigger_list:
            trig_rf0 = 1
        if 'rf1' in trigger_list:
            trig_rf1 = 1
        if 'ext' in trigger_list:
            trig_ext = 1
        if 'pps' in trigger_list:
            trig_pps = 1
        if 'soft' in trigger_list:
            trig_soft = 1

        # define a search filter
        search_filter = [{"$match": {"station": {"$in": station_list},
                                     "time_start": {"$gte": start_time},
                                     "time_end": {"$lte": end_time},
                                     #"quality_flag": {"$in": flag_list},
                                     "duration": {"$gte": min_duration / units.second},
                                     "firmware_version": {"$in": firmware_list},
                                     "trigger_rf0_enabled": trig_rf0,
                                     "trigger_rf1_enabled": trig_rf1,
                                     "trigger_ext_enabled": trig_ext,
                                     "trigger_pps_enabled": trig_pps,
                                     "trigger_soft_enabled": trig_soft
                                     }}]

        search_results = list(self.db['runtable'].aggregate(search_filter))

        return search_results


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
