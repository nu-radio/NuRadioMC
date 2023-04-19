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

        # Set timestamp of database. This is used to determine which primary measurement is used
        self.__database_time = datetime.datetime.utcnow()
        
        # This is used to get commissioned stations/channels
        self.__detector_time = None
        
        self.__get_collection_names = None
        
        
    def set_database_time(self, time):
        ''' Set time(stamp) for database. This affects which primary measurement is used.
        
        Parameters
        ----------
        
        time: datetime.datetime 
            UTC time.
        '''
        self.__database_time = time
     
        
    def get_database_time(self):
        return self.__database_time


    def find_primary_measurement(
            self, type, name, primary_time, identification_label='name', _id=None, id_label='channel', 
            breakout_id=None, breakout_channel_id=None):
        """
        Find the object_id of entry with name 'name' and gives the measurement_id of the primary measurement, 
        return the id of the object and the measurement

        Parameters
        ----------
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
        if self.____get_collection_names is None:
            self.__get_collection_names =  self.db.list_collection_names()
        return self.__get_collection_names

    def get_station_ids(self, collection):
        return self.db[collection].distinct('id')

    def load_board_information(self, type, board_name, info_names):
        """ For IGLU / DRAB """
        infos = []
        for i in range(len(self.db[type].find_one({'name': board_name})['measurements'])):
            if self.db[type].find_one({'name': board_name})['measurements'][i]['function_test']:
                for name in info_names:
                    infos.append(self.db[type].find_one({'name': board_name})['measurements'][i][name])
                break

        return infos

    @check_database_time
    def get_general_station_information(self, collection, station_id, detector_time=None):
        """ Get information from one station
        
        Parameters
        ----------
        
        collection_name: string
            Specify the collection, from which the information should be extracted (e.g. "station_rnog")
        
        Returns
        -------
        
        info: dict
        """

        # if the collection is empty, return an empty dict
        if self.db[collection].count_documents({'id': station_id}) == 0:
            return {}
        
        if detector_time is None:
            detector_time = self.__database_time
            logger.info("Detector time is None, use database time.")

        # filter to get all information from one station with station_id and with active commission time
        time_filter = [{"$match": {
            'commission_time': {"$lte": detector_time},
            'decommission_time': {"$gte": detector_time},
            'id': station_id}}]

        # get all stations which fit the filter (should only be one)
        stations_for_buffer = list(self.db[collection].aggregate(time_filter))
        
        # transform the output of db.aggregate to a dict
        # dictionarize the channel information
        station_info = dictionarize_nested_lists(stations_for_buffer, parent_key="id", nested_field="channels", nested_key="id")
        
        # dictionarize the device information
        station_info_help = dictionarize_nested_lists(stations_for_buffer, parent_key="id", nested_field="devices", nested_key="id")

        station_info[station_id]['devices'] = station_info_help[station_id]['devices']

        # Add empty dicts if necessary
        for key in ['channels', 'devices']:
            if key not in station_info[station_id].keys():
                station_info[station_id][key] = {}

        return station_info


    @check_database_time
    def get_collection_information(self, collection_name, station_id, measurement_name=None, channel_id=None):
        """
        Get the information for a specified collection (will only work for 'station_position', 'channel_position' and 'signal_chain')
        if the station does not exist, {} will be returned. Return primary measurement unless measurement_name is specified.

        Parameters
        ----------
        
        collection_name: string
            Specify the collection, from which the information should be extracted (will only work for 'station_position', 
            'channel_position' and 'signal_chain')
        
        station_id: int
            The unique identifier of the station
        
        measurement_name: string
            The unique name of the measurement. (Default: None - return primary measurement)
        
        channel_id: int
            Unique identifier of the channel

        Returns
        -------
        
        info: list(dict)
        """
        
        # if the collection is empty, return an empty dict
        if self.db[collection_name].count_documents({'id': station_id}) == 0:
            return {}
        
        primary_time = self.__database_time
        
        # define the search filter
        search_filter = [{'$match': {'id': station_id}},
                         {'$unwind': '$measurements'}]
        
        if channel_id is not None and measurement_name is None:
            search_filter += [{'$match': {'measurements.channel_id': channel_id}}]  # append
        
        elif channel_id is None and measurement_name is not None:
            search_filter += [{'$match': {'measurements.measurement_name': measurement_name}}]  # append

        elif channel_id is not None and measurement_name is not None:
            search_filter += [{'$match': {'measurements.measurement_name': measurement_name,
                                          'measurements.channel_id': channel_id}}]  # append
        else:
            pass
                    
        if measurement_name is None:
            search_filter += [
                {'$unwind': '$measurements.primary_measurement'},
                {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                            'measurements.primary_measurement.end': {'$gte': primary_time}}}]
        else:
            # measurement identified by "measurement_name"
            pass    

        search_result = list(self.db[collection_name].aggregate(search_filter))

        # FS: The following code block seems unnecessary
        """
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
        """
        
        return search_result


    def get_quantity_names(self, collection_name, wanted_quantity):
        """ returns a list with all measurement names, ids, ... or what is specified (example: wanted_quantity = measurements.measurement_name)"""
        return self.db[collection_name].distinct(wanted_quantity)


    def get_all_available_signal_chain_configs(self, collection, object_name, input_dic):
        """ 
        Depending on the inputs, all possible configurations in the database are returned; 
        Input example: 'iglu_boards', 'Golden_IGLU' {'measurement_temp': 20, 'DRAB_id': 'Golden_DRAB'}
        """
        
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

        return return_dic


    def get_station_position(self, station_id, measurement_name=None):
        station_position_information = self.get_collection_information('station_position', station_id, measurement_name=measurement_name)

        if len(station_position_information) != 1:\
            raise ValueError

        return station_position_information[0]['measurements']
    

    def get_channels_position(self, station_id, measurement_name=None, channel_id=None):
        channel_position_information = self.get_collection_information(
            'channel_position', station_id, measurement_name=measurement_name, channel_id=channel_id)

        if channel_id is not None and len(channel_position_information) > 1:
            raise ValueError

        # get the channel position information in the correct format
        channel_pos_dic = {}
        for cha_pos_dic in channel_position_information:
            channel_id = cha_pos_dic['measurements']['channel_id']
            channel_pos_dic[channel_id] = {k: cha_pos_dic['measurements'][k] for k in 
                                           filtered_keys(cha_pos_dic['measurements'], ['channel_id'])}
        
        return channel_pos_dic
    
    
    def get_sig_chain_component_measurements(self, channel_signal_chain_dict):
        
        sig_chain = channel_signal_chain_dict['sig_chain']
        primary_components = channel_signal_chain_dict['primary_components']
        
        measurement_components_dic = {}
        
        for sig_chain_component in sig_chain:
            if sig_chain_component in self.get_collection_names():
                
                # Search for supplementary info (e.g., drab_board and drab_board_channel_id). If this is the case -> extract the information 
                # (important to find the final measurement)
                supp_info = []
                for sck in sig_chain:
                    if sig_chain_component in sck and sig_chain_component != sck:
                        supp_info.append(sck)
                                            
                # get the primary time of the component measurement -> important to find the measurement which should be used
                primary_component = primary_components[sig_chain_component]
                
                # define a search filter
                search_filter_sig_chain = [{'$match': {'name': sig_chain[sig_chain_component]}}, {'$unwind': '$measurements'}]
    
                # if there is supp info -> add this to the search filter
                help_dic = {}
                if len(supp_info):
                    for si in supp_info:
                        help_dic[f'measurements.{si[len(sig_chain_component)+1:]}'] = sig_chain[si]
                
                if len(self.get_quantity_names(collection_name=sig_chain_component, wanted_quantity='measurements.S_parameter')) != 1:  
                    # if more than one S parameter is saved in the database only chose S21
                    help_dic['measurements.S_parameter'] = 'S21'

                if len(help_dic):
                    search_filter_sig_chain.append({'$match': help_dic})
                
                # add the primary time
                search_filter_sig_chain.append({'$unwind': '$measurements.primary_measurement'})
                search_filter_sig_chain.append({'$match': {'measurements.primary_measurement.start': {'$lte': primary_component},
                                                            'measurements.primary_measurement.end': {'$gte': primary_component}}})
                # print(sig_chain_component, search_filter_sig_chain)
                # find the correct measurement in the database and extract the measurement and object id
                search_result_sig_chain = list(self.db[sig_chain_component].aggregate(search_filter_sig_chain))
                
                if len(search_result_sig_chain) != 1:
                    raise ValueError
                
                result_sig_chain = search_result_sig_chain[0]['measurements']
                
                freq = result_sig_chain['frequencies']  # 0: should only return a single valid entry
                yunits = result_sig_chain['y-axis_units']
                
                ydata = []
                if 'mag' in result_sig_chain:
                    ydata.append(result_sig_chain['mag'])
                if 'phase' in result_sig_chain:
                    ydata.append(result_sig_chain['phase'])

                measurement_components_dic[sig_chain_component] = {'y_units': yunits, 'freq': freq, 'ydata': ydata}
        
        return measurement_components_dic
    
    
    def get_channels_signal_chain(self, station_id, measurement_name=None, channel_id=None):
        signal_chain_information = self.get_collection_information('signal_chain', station_id, measurement_name=measurement_name, channel_id=channel_id)
        
        if channel_id is not None and len(signal_chain_information) > 1:
            raise ValueError

        # get the channel signal chain in the correct format
        channel_sig_chain_dic = {}
        for cha_sig_dic in signal_chain_information:
            channel_id = cha_sig_dic['measurements']['channel_id']
            channel_sig_chain_dic[channel_id] = {k: cha_sig_dic['measurements'][k] 
                                                 for k in filtered_keys(cha_sig_dic['measurements'], ['channel_id'])}

        # got through the signal chain and collect the corresponding measurements
        for cha_id, channel_signal_chain_dict in channel_sig_chain_dic.items():

            measurement_components_dic = self.get_sig_chain_component_measurements(channel_signal_chain_dict)

            channel_sig_chain_dic[cha_id]['measurements_components'] = measurement_components_dic

        return channel_sig_chain_dic

    
    def get_devices_position(self, station_id, measurement_name=None):
        device_position_information = self.get_collection_information('device_position', station_id, measurement_name=measurement_name)

        device_pos_dic = {}
        for dev_pos_dic in device_position_information:
            device_id = dev_pos_dic['measurements']['device_id']
            device_pos_dic[device_id] = {dk: dev_pos_dic['measurements'][dk] for dk in filtered_keys(dev_pos_dic['measurements'], ['device_id'])}

        return device_pos_dic
    
    
    def get_complete_station_information(
            self, station_id, measurement_position=None, measurement_channel_position=None, 
            measurement_signal_chain=None, measurement_device_position=None):
        """
        Collects all available information about the station

        Parameters
        ----------
        
        station_id: int
            The unique identifier of the station the channel belongs to

        measurement_position: string
            If not None, this measurement will be collected (even though it is not the primary measurement)
        
        measurement_channel_position: string
            If not None, this measurement will be collected (even though it is not the primary measurement)
        
        measurement_signal_chain: string
            If not None, this measurement will be collected (even though it is not the primary measurement)
        
        measurement_device_position: string
            If not None, this measurement will be collected (even though it is not the primary measurement)

        Returns
        -------
        
        complete_info: dict
        """
        complete_info = {}

        # load general station information
        general_info = self.get_general_station_information('station_rnog', station_id)

        # create dict without "channels" and devices
        general_station_dic = {k: general_info[station_id][k] for k in filtered_keys(general_info[station_id], ['channels', 'devices'])}
        complete_info.update(general_station_dic)

        # Add station position information
        sta_pos_measurement_info = self.get_station_position(station_id, measurement_name=measurement_position)
        complete_info['station_position'] = sta_pos_measurement_info

        # combine channel information
        general_channel_dic = {}
        exclude_keys = ['signal_ch', 'id']
        for cha_key in general_info[station_id]['channels']:
            help_dic = general_info[station_id]['channels'][cha_key]
            general_channel_dic[cha_key] = {k: help_dic[k] for k in filtered_keys(help_dic, exclude_keys)}

        # get the channel position / signal chain information in the correct format
        channel_pos_dic = self.get_channels_position(station_id, measurement_channel_position)
        channel_sig_chain_dic = self.get_channels_signal_chain(station_id, measurement_name=measurement_signal_chain)
        
        for cha_id in general_channel_dic:
            general_channel_dic[cha_id]['channel_position'] = channel_pos_dic[cha_id]
            general_channel_dic[cha_id]['channel_signal_chain'] = channel_sig_chain_dic[cha_id]

        complete_info['channels'] = general_channel_dic

        # combine device information
        general_device_dic = {}
        exclude_dev_keys = ['device_id']
        for dev_key in general_info[station_id]['devices']:
            help_dev_dic = general_info[station_id]['devices'][dev_key]
            general_device_dic[dev_key] = {dk: help_dev_dic[dk] for dk in filtered_keys(help_dev_dic, exclude_dev_keys)}

        # get the device position information in the correct format
        device_pos_dic = self.get_devices_position(station_id, measurement_device_position)
        if not len(device_pos_dic):
            logger.warn("Could not find device information.")
        else:
            for dev_id in general_device_dic.keys():
                general_device_dic[dev_id]['device_position'] = device_pos_dic[dev_id]

            complete_info['devices'] = general_device_dic

        return complete_info


    def get_complete_channel_information(
            self, station_id, channel_id,
            measurement_position=None, measurement_channel_position=None, measurement_signal_chain=None):
        """
        Collects all available information about the input channel

        Parameters
        ----------
        
        station_id: int
            The unique identifier of the station the channel belongs to
        
        channel_id: int
            Channel id for which all information will be returned
        
        measurement_position: string
            If not None, this measurement will be collected (even though it is not the primary measurement)
        
        measurement_channel_position: string
            If not None, this measurement will be collected (even though it is not the primary measurement)
        
        measurement_signal_chain: string
            If not None, this measurement will be collected (even though it is not the primary measurement)
        
        Returns
        -------
        
        complete_info
        """
        complete_info = {}

        # load general station information
        general_info = self.get_general_station_information('station_rnog', station_id)

        # create dict without "channels" and devices
        general_station_dic = {k: general_info[station_id][k] for k in filtered_keys(general_info[station_id], ['channels', 'devices'])}
        complete_info.update(general_station_dic)

        # Add station position information
        sta_pos_measurement_info = self.get_station_position(station_id, measurement_name=measurement_position)
        complete_info['station_position'] = sta_pos_measurement_info

        # combine channel information
        general_channel_dic = {}
        exclude_keys = ['signal_ch', 'id']
        help_dic = general_info[station_id]['channels'][channel_id]
        general_channel_dic[channel_id] = {k: help_dic[k] for k in set(list(help_dic.keys())) - set(exclude_keys)}

        # get the channel position information in the correct format
        channel_pos_dic = self.get_channels_position(station_id, measurement_name=measurement_channel_position, channel_id=channel_id)
        general_channel_dic[channel_id]['channel_position'] = channel_pos_dic[channel_id]

        channel_sig_chain_dic = self.get_channels_signal_chain(station_id, measurement_name=measurement_signal_chain, channel_id=channel_id)
        general_channel_dic[channel_id]['channel_signal_chain'] = channel_sig_chain_dic[channel_id]
        complete_info['channels'] = general_channel_dic

        return complete_info


    def get_complete_device_information(self, station_id, device_id, measurement_position=None, measurement_device_position=None):
        """
        Collects all available information about a device

        Parameters
        ----------
        
        station_id: int
            The unique identifier of the station the channel belongs to
        
        device_id: int
            The device id for which the information will be written out
        

        measurement_position: string
            If not None, this measurement will be collected (even though it is not the primary measurement)
        

        measurement_device_position: string
            If not None, this measurement will be collected (even though it is not the primary measurement)

        Returns
        -------
        complete_info
        """
        complete_info = {}

        # load general station information
        general_info = self.get_general_station_information('station_rnog', station_id)

        # create dict without "channels" and devices
        general_station_dic = {k: general_info[station_id][k] for k in filtered_keys(general_info[station_id], ['channels', 'devices'])}
        complete_info.update(general_station_dic)

        # Add station position information
        sta_pos_measurement_info = self.get_station_position(station_id, measurement_name=measurement_position)
        complete_info['station_position'] = sta_pos_measurement_info
        
        # combine device information
        general_device_dic = {}
        exclude_dev_keys = ['device_id']
        for dev_key in general_info[station_id]['devices']:
            help_dev_dic = general_info[station_id]['devices'][dev_key]
            general_device_dic[dev_key] = {dk: help_dev_dic[dk] for dk in filtered_keys(help_dev_dic, exclude_dev_keys)}

        # get the device position information in the correct format
        device_pos_dic = self.get_devices_position(station_id, measurement_device_position)
        for dev_id in general_device_dic.keys():
            general_device_dic[dev_id]['device_position'] = device_pos_dic[dev_id]

        complete_info['devices'] = general_device_dic

        return complete_info


def dictionarize_nested_lists(nested_lists, parent_key="id", nested_field="channels", nested_key="id"):
    """ mongodb aggregate returns lists of dicts, which can be converted to dicts of dicts """
    res = {}
    for parent in nested_lists:
        res[parent[parent_key]] = parent
        if nested_field in parent and nested_field is not None:
            daughter_list = parent[nested_field]
            daughter_dict = {}
            for daughter in daughter_list:
                if nested_key in daughter:
                    daughter_dict[daughter[nested_key]] = daughter

            res[parent[parent_key]][nested_field] = daughter_dict
    return res


def dictionarize_nested_lists_as_tuples(nested_lists, parent_key="name", nested_field="measurements", nested_keys=("channel_id", "S_parameter")):
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
