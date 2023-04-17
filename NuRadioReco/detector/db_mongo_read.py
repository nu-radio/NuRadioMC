import six
import os
import sys
import urllib.parse
import datetime
import json
import numpy as np
from functools import wraps
import collections

from pymongo import MongoClient
from bson import json_util  # bson dicts are used by pymongo

import NuRadioReco.utilities.metaclasses

import logging
logging.basicConfig()
logger = logging.getLogger("database")
logger.setLevel(logging.DEBUG)


def check_database_time(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        time = self.get_database_time()
        if time is None:
            logger.error('Database time is None.')
            raise ValueError('Database time is None.')
        return method(self, *method_args, **method_kwargs)
    return _impl


def filtered_keys(dict, exclude_keys):
    """ Creates a set (list) of dictionary keys with out 'exclude_keys' """
    if not isinstance(exclude_keys, list):
        exclude_keys = [exclude_keys]
    
    return set(list(dict.keys())) - set(exclude_keys)

@six.add_metaclass(NuRadioReco.utilities.metaclasses.Singleton)
class Database(object):

    def __init__(self, database_connection="env_pw_user", database_name=None):

        if database_connection == "env_pw_user":
            # use db connection from environment, pw and user need to be percent escaped
            mongo_server = os.environ.get('mongo_server')
            if mongo_server is None:
                logger.warning('variable "mongo_server" not set')

            mongo_password = urllib.parse.quote_plus(os.environ.get('mongo_password'))
            mongo_user = urllib.parse.quote_plus(os.environ.get('mongo_user'))
            if None in [mongo_user, mongo_password]:
                logger.warning('"mongo_user" or "mongo_password" not set')
            
            # start client
            self.__mongo_client = MongoClient("mongodb://{}:{}@{}".format(mongo_user, mongo_password, mongo_server), tls=True)
            self.db = self.__mongo_client.RNOG_live
        
        elif database_connection == "RNOG_public":
            # use read-only access to the RNO-G database
            self.__mongo_client = MongoClient("mongodb://read:EseNbGVaCV4pBBrt@radio.zeuthen.desy.de:27017/admin?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=true")
            self.db = self.__mongo_client.RNOG_live
        
        elif database_connection == "RNOG_test_public":
            # use readonly access to the RNO-G test database
            self.__mongo_client = MongoClient(
                "mongodb://RNOG_test_public:jrE5xO38D7wQweVR5doa@radio-test.zeuthen.desy.de:27017/admin?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=true")
            self.db = self.__mongo_client.RNOG_live
        
        elif database_connection == "connection_string":
            # use a connection string from the environment
            connection_string = os.environ.get('db_mongo_connection_string')
            self.__mongo_client = MongoClient(connection_string)
            self.db = self.__mongo_client.RNOG_test
        else:
            logger.error('specify a defined database connection ["env_pw_user", "connection_string", "RNOG_public", "RNOG_test_public"]')

        logger.info("database connection to {} established".format(self.db.name))

        self.__current_time = None

        self.__modification_timestamps = self._query_modification_timestamps()
        self.__buffered_period = None

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

    def find_primary_measurement(self, type, name, primary_time, identification_label='name', _id=None, id_label='channel', breakout_id=None, breakout_channel_id=None):
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
                _id: int
                    if there is a channel or device id for the object, the id is used in the search filter mask
                id_label: string
                    sets if a channel id ('channel') or device id ('device) is used

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
        elif _id is not None:
            filter_primary = [{'$match': {identification_label: name}},
                              {'$unwind': '$measurements'},
                              {'$unwind': '$measurements.primary_measurement'},
                              {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                          'measurements.primary_measurement.end': {'$gte': primary_time},
                                          f'measurements.{id_label}_id': _id}}]
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

    def get_object_names(self, object_type):
        return self.db[object_type].distinct('name')

    def get_collection_names(self):
        return self.db.list_collection_names()

    def get_station_ids(self, collection):
        return self.db[collection].distinct('id')

    # IGLU / DRAB
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
        # dictionarize the channel information
        station_info = dictionarize_nested_lists(stations_for_buffer, parent_key="id", nested_field="channels", nested_key="id")
        # dictionarize the device information
        station_info_help = dictionarize_nested_lists(stations_for_buffer, parent_key="id", nested_field="devices", nested_key="id")
        # print(station_info_help)
        station_info[station_id]['devices'] = station_info_help[station_id]['devices']

        if 'channels' not in station_info[station_id].keys():
            station_info[station_id]['channels'] = {}

        if 'devices' not in station_info[station_id].keys():
            station_info[station_id]['devices'] = {}

        return station_info

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
            # print(search_filter)
            search_result = list(self.db[collection].aggregate(search_filter))

            for key in input_dic.keys():
                help_list = []
                for entry in search_result:
                    help_list.append(entry['measurements'][key])
                return_dic[key] = list(set(help_list))
        # print(return_dic)
        return return_dic

    def get_complete_station_information(self, station_id, primary_time=None, measurement_position=None, measurement_channel_position=None, measurement_signal_chain=None, measurement_device_position=None):
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
        measurement_device_position: string
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
            if measurement_device_position is None:
                primary_time = datetime.datetime.utcnow()
                device_position_information = self.get_collection_information('device_position', station_id, primary_time=primary_time)
            else:
                device_position_information = self.get_collection_information('device_position', station_id, primary_time=None, measurement_name=measurement_device_position)
        else:  # primary_time is not None
            station_position_information = self.get_collection_information('station_position', station_id, primary_time=primary_time)
            channel_position_information = self.get_collection_information('channel_position', station_id, primary_time=primary_time)
            signal_chain_information = self.get_collection_information('signal_chain', station_id, primary_time=primary_time)
            device_position_information = self.get_collection_information('device_position', station_id, primary_time=primary_time)

        # combine the station information
        general_station_dic = {k: general_info[station_id][k] for k in set(list(general_info[station_id].keys())) - set(['channels', 'devices'])}  # create dic with all entries except 'channels'
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
            # print(channel_sig_chain_dic[cha_id])
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

        # combine device information
        general_device_dic = {}
        exclude_dev_keys = ['device_id']
        for dev_key in general_info[station_id]['devices'].keys():
            help_dev_dic = general_info[station_id]['devices'][dev_key]
            general_device_dic[dev_key] = {dk: help_dev_dic[dk] for dk in set(list(help_dev_dic.keys())) - set(exclude_dev_keys)}

        # get the device position information in the correct format
        device_pos_dic = {}
        for dev_pos_dic in device_position_information:
            device_id = dev_pos_dic['measurements']['device_id']
            device_pos_dic[device_id] = {dk: dev_pos_dic['measurements'][dk] for dk in set(list(dev_pos_dic['measurements'].keys())) - set(['device_id'])}

        for dev_id in general_device_dic.keys():
            if dev_id not in device_pos_dic.keys():
                general_device_dic[dev_id]['device_position'] = {}
            else:
                general_device_dic[dev_id]['device_position'] = device_pos_dic[dev_id]

        complete_info['devices'] = general_device_dic

        return complete_info

    def get_complete_channel_information(self, station_id, channel_id, primary_time=None, measurement_position=None, measurement_channel_position=None, measurement_signal_chain=None):
        """
                collects all available information about the input channel

                Parameters
                ---------
                station_id: int
                    the unique identifier of the station the channel belongs to
                channel_id: int
                    channel id for which all information will be returned
                primary_time: datetime.datetime
                    time used to check for the primary measurement
                    if None and no measurement name given: the current time
                measurement_position: string
                    if given (and primary=None) the measurement will be collected (even though it is not the primary measurement)
                measurement_channel_position: string
                    if given (and primary=None) the measurement will be collected (even though it is not the primary measurement)
                measurement_signal_chain: string
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
                channel_position_information = self.get_collection_information('channel_position', station_id, primary_time=primary_time, channel_id=channel_id)
            else:
                channel_position_information = self.get_collection_information('channel_position', station_id, primary_time=None, measurement_name=measurement_channel_position, channel_id=channel_id)
            if measurement_signal_chain is None:
                primary_time = datetime.datetime.utcnow()
                signal_chain_information = self.get_collection_information('signal_chain', station_id, primary_time=primary_time, channel_id=channel_id)
            else:
                signal_chain_information = self.get_collection_information('signal_chain', station_id, primary_time=None, measurement_name=measurement_signal_chain, channel_id=channel_id)
        else:  # primary_time is not None
            station_position_information = self.get_collection_information('station_position', station_id, primary_time=primary_time)
            channel_position_information = self.get_collection_information('channel_position', station_id, primary_time=primary_time, channel_id=channel_id)
            signal_chain_information = self.get_collection_information('signal_chain', station_id, primary_time=primary_time, channel_id=channel_id)

        # combine the station information
        general_station_dic = {k: general_info[station_id][k] for k in set(list(general_info[station_id].keys())) - set(['channels', 'devices'])}  # create dic with all entries except 'channels'
        sta_pos_measurement_info = station_position_information[0]['measurements']
        complete_info.update(general_station_dic)
        complete_info['station_position'] = sta_pos_measurement_info

        # combine channel information
        general_channel_dic = {}
        exclude_keys = ['signal_ch', 'id']
        help_dic = general_info[station_id]['channels'][channel_id]
        general_channel_dic[channel_id] = {k: help_dic[k] for k in set(list(help_dic.keys())) - set(exclude_keys)}

        # get the channel position information in the correct format
        channel_pos_dic = {}
        if len(channel_position_information) > 1:
            print('More than one channel found.')
            sys.exit(0)
        elif len(channel_position_information) == 0:
            pass
        else:
            cha_pos_dic = channel_position_information[0]
            channel_pos_dic[channel_id] = {k: cha_pos_dic['measurements'][k] for k in set(list(cha_pos_dic['measurements'].keys())) - set(['channel_id'])}

        # get the channel signal chain in the correct format
        channel_sig_chain_dic = {}
        if len(signal_chain_information) > 1:
            print('More than one channel found.')
            sys.exit(0)
        elif len(signal_chain_information) == 0:
            pass
        else:
            cha_sig_dic = signal_chain_information[0]
            channel_sig_chain_dic[channel_id] = {k: cha_sig_dic['measurements'][k] for k in set(list(cha_sig_dic['measurements'].keys())) - set(['channel_id'])}

            # got through the signal chain and collect the corresponding measurements
            sig_chain = channel_sig_chain_dic[channel_id]['sig_chain']
            # print(channel_sig_chain_dic[channel_id])
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
                    primary_component = channel_sig_chain_dic[channel_id]['primary_components'][sig_chain_key]
                    # define a search filter
                    search_filter_sig_chain = []
                    search_filter_sig_chain.append({'$match': {'name': sig_chain[sig_chain_key]}})
                    search_filter_sig_chain.append({'$unwind': '$measurements'})
                    # if there is supp info -> add this to the search filter
                    help_dic1 = {}
                    help_dic2 = {}
                    if supp_info != []:
                        for si in supp_info:
                            help_dic2[f'measurements.{si[len(sig_chain_key) + 1:]}'] = sig_chain[si]

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
            channel_sig_chain_dic[channel_id]['measurements_components'] = measurement_components_dic

        if channel_id not in channel_pos_dic.keys():
            general_channel_dic[channel_id]['channel_position'] = {}
        else:
            general_channel_dic[channel_id]['channel_position'] = channel_pos_dic[channel_id]
        if channel_id not in channel_sig_chain_dic.keys():
            general_channel_dic[channel_id]['channel_signal_chain'] = {}
        else:
            general_channel_dic[channel_id]['channel_signal_chain'] = channel_sig_chain_dic[channel_id]

        complete_info['channels'] = general_channel_dic

        return complete_info

    def get_complete_device_information(self, station_id, device_id, primary_time=None, measurement_position=None, measurement_device_position=None):
        """
                collects all available information about a device

                Parameters
                ---------
                station_id: int
                    the unique identifier of the station the channel belongs to
                device_id: int
                    the device id for which the information will be written out
                primary_time: datetime.datetime
                    time used to check for the primary measurement
                    if None and no measurement name given: the current time
                measurement_position: string
                    if given (and primary=None) the measurement will be collected (even though it is not the primary measurement)
                measurement_device_position: string
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
            if measurement_device_position is None:
                primary_time = datetime.datetime.utcnow()
                device_position_information = self.get_collection_information('device_position', station_id, primary_time=primary_time)
            else:
                device_position_information = self.get_collection_information('device_position', station_id, primary_time=None, measurement_name=measurement_device_position)
        else:  # primary_time is not None
            station_position_information = self.get_collection_information('station_position', station_id, primary_time=primary_time)
            device_position_information = self.get_collection_information('device_position', station_id, primary_time=primary_time)

        # combine the station information
        general_station_dic = {k: general_info[station_id][k] for k in set(list(general_info[station_id].keys())) - set(['channels', 'devices'])}  # create dic with all entries except 'channels'
        sta_pos_measurement_info = station_position_information[0]['measurements']
        complete_info.update(general_station_dic)
        complete_info['station_position'] = sta_pos_measurement_info

        # combine channel information
        general_channel_dic = {}
        exclude_keys = ['signal_ch', 'id']
        for cha_key in general_info[station_id]['channels'].keys():
            help_dic = general_info[station_id]['channels'][cha_key]
            general_channel_dic[cha_key] = {k: help_dic[k] for k in set(list(help_dic.keys())) - set(exclude_keys)}

        # combine device information
        general_device_dic = {}
        exclude_dev_keys = ['device_id']
        for dev_key in general_info[station_id]['devices'].keys():
            if dev_key == device_id:
                help_dev_dic = general_info[station_id]['devices'][dev_key]
                general_device_dic[dev_key] = {dk: help_dev_dic[dk] for dk in set(list(help_dev_dic.keys())) - set(exclude_dev_keys)}

        # get the device position information in the correct format
        device_pos_dic = {}
        for dev_pos_dic in device_position_information:
            if dev_pos_dic['measurements']['device_id'] == device_id:
                device_pos_dic[device_id] = {dk: dev_pos_dic['measurements'][dk] for dk in set(list(dev_pos_dic['measurements'].keys())) - set(['device_id'])}

        if device_id not in device_pos_dic.keys():
            general_device_dic[device_id]['device_position'] = {}
        else:
            general_device_dic[device_id]['device_position'] = device_pos_dic[device_id]

        complete_info['devices'] = general_device_dic

        return complete_info

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
