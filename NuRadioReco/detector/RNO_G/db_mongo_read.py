import six
import os
import urllib.parse
import datetime
import numpy as np
from functools import wraps
import collections
import copy

from pymongo import MongoClient
# from bson import json_util  # bson dicts are used by pymongo

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
        
        self.__station_collection = "station_rnog"
        
        
    def set_database_time(self, time):
        ''' Set time for database. This affects which primary measurement is used.
        
        Parameters
        ----------
        
        time: datetime.datetime 
            UTC time.
        '''
        if not isinstance(time, datetime.datetime):
            logger.error("Set invalid time for database. Time has to be of type datetime.datetime")
            raise TypeError("Set invalid time for database. Time has to be of type datetime.datetime")
        self.__database_time = time
        
        
    def set_detector_time(self, time):
        ''' Set time of detector. This controls which stations/channels are commissioned.
        
        Parameters
        ----------
        
        time: datetime.datetime 
            UTC time.
        '''
        if not isinstance(time, datetime.datetime):
            logger.error("Set invalid time for detector. Time has to be of type datetime.datetime")
            raise TypeError("Set invalid time for detector. Time has to be of type datetime.datetime")
        self.__detector_time = time
     
        
    def get_database_time(self):
        return self.__database_time
    
    def get_detector_time(self):
        return self.__detector_time

    def find_primary_measurement_old(
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
        filter_primary = [{'$match': {identification_label: name}},
                            {'$unwind': '$measurements'},
                            {'$unwind': '$measurements.primary_measurement'}]
        
        add_filter = {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                 'measurements.primary_measurement.end': {'$gte': primary_time}}}
        if breakout_channel_id is not None and breakout_id is not None:
            add_filter['$match'].update({'measurements.breakout': breakout_id,
                                         'measurements.breakout_channel': breakout_channel_id})

        elif _id is not None:
            add_filter['$match'].update({f'measurements.{id_label}_id': _id})
        
        filter_primary.append(add_filter)

        # get all entries matching the search filter
        matching_entries = list(self.db[type].aggregate(filter_primary))

        # extract the object and measurement id
        if len(matching_entries) > 1:
            # see if Sparameters are stored
            if 'S_parameter' in matching_entries[0]['measurements'].keys():
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
                    return None, [None]
            else:
                logger.error('More than one primary measurement found')
                return None, [None]
        elif len(matching_entries) > 4:
            logger.error('More primary measurements than Sparameters are found.')
            return None, [None]
        elif len(matching_entries) == 0:
            logger.error('No primary measurement found.')
            # the last zero is the information that no primary measurement was found
            return None, [0]
        else:
            object_id = matching_entries[0]['_id']
            measurement_id = matching_entries[0]['measurements']['id_measurement']
            return object_id, [measurement_id]
        
    def find_primary_measurement(self, type, name, primary_time, identification_label, data_dict):
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
        filter_primary = [{'$match': {identification_label: name}},
                            {'$unwind': '$measurements'},
                            {'$unwind': '$measurements.primary_measurement'}]
        
        add_filter = {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                 'measurements.primary_measurement.end': {'$gte': primary_time}}}
        
        data_dict_keys = data_dict.keys()

        if 'breakout_channel' in data_dict_keys and 'breakout' in data_dict_keys:
            add_filter['$match'].update({'measurements.breakout': data_dict['breakout'],
                                         'measurements.breakout_channel': data_dict['breakout_channel']})

        if 'channel_id' in data_dict_keys:
            add_filter['$match'].update({f'measurements.channel_id': data_dict['channel_id']})

        if 'S_parameter' in data_dict_keys:
            add_filter['$match'].update({'measurements.S_parameter': data_dict['S_parameter']})
        
        filter_primary.append(add_filter)

        # get all entries matching the search filter
        matching_entries = list(self.db[type].aggregate(filter_primary))

        # extract the object and measurement id
        if len(matching_entries) > 1:
            logger.error('More than one primary measurement found.')
            return None, [None]
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
        if self.__get_collection_names is None:
            self.__get_collection_names =  self.db.list_collection_names()
        return self.__get_collection_names

    def get_station_ids_of_collection(self, collection):
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
    def get_general_station_information(self, station_id):
        """ Get information from one station. Access information in the main collection.
        
        Parameters
        ----------
        
        station_id: int
            Station id
        
        Returns
        -------
        
        info: dict
        """

        # if the collection is empty, return an empty dict
        if self.db[self.__station_collection].count_documents({'id': station_id}) == 0:
            return {}
        
        if self.__detector_time is None:
            detector_time = self.__database_time
            logger.error("Detector time is None, use database time.")
        else:
            detector_time = self.__detector_time

        # filter to get all information from one station with station_id and with active commission time
        time_filter = [{"$match": {
            'commission_time': {"$lte": detector_time},
            'decommission_time': {"$gte": detector_time},
            'id': station_id}}]

        # get all stations which fit the filter (should only be one)
        stations_for_buffer = list(self.db[self.__station_collection].aggregate(time_filter))
        
        if len(stations_for_buffer) == 0:
            logger.warning('No corresponding station found!')
            return {}
        elif len(stations_for_buffer) > 1:
            err = f"Found to many stations (f{len(stations_for_buffer)}) for: station_id = {station_id}, and time = {detector_time}"
            logger.error(err)
            raise ValueError(err)
        
        # filter out all decommissioned channels and devices
        commissioned_info = copy.deepcopy(stations_for_buffer)
        for key in ['channels', 'devices']:
            for ientry, entry in enumerate(stations_for_buffer[0][key]):
                if entry['commission_time'] <= detector_time and entry['decommission_time'] >= detector_time:
                    pass
                else:
                    commissioned_info[0][key].pop(ientry)
        
        # transform the output of db.aggregate to a dict
        # dictionarize the channel information
        station_info = dictionarize_nested_lists(commissioned_info, parent_key="id", nested_field="channels", nested_key="id")
        
        # dictionarize the device information
        station_info_help = dictionarize_nested_lists(commissioned_info, parent_key="id", nested_field="devices", nested_key="id")

        station_info[station_id]['devices'] = station_info_help[station_id]['devices']

        # Add empty dicts if necessary
        for key in ['channels', 'devices']:
            if key not in station_info[station_id].keys():
                station_info[station_id][key] = {}

        return station_info

    @check_database_time
    def get_general_channel_information(self, station_id, channel_id):
        """ Get information from one channel. Access information in the main collection.
        
        Parameters
        ----------
        
        station_id: int
            specifiy the station id from which the channel information is taken
        
        channel_id: int
            specifiy the channel id
        
        Returns
        -------
        
        info: dict
        """

        # if the collection is empty, return an empty dict
        if self.db[self.__station_collection].count_documents({'id': station_id}) == 0:
            return {}
        
        if self.__detector_time is None:
            detector_time = self.__database_time
            logger.error("Detector time is None, use database time.")
        else:
            detector_time = self.__detector_time

        # filter to get all information from one channel with station_id and channel_id and active commission time
        time_filter = [{"$match": {
            'commission_time': {"$lte": detector_time},
            'decommission_time': {"$gte": detector_time},
            'id': station_id}},
            {'$unwind': '$channels'},
            {"$match": {'channels.commission_time': {"$lte": detector_time},
            'channels.decommission_time': {"$gte": detector_time},
            'channels.id': channel_id}}]

        # get all stations which fit the filter (should only be one)
        channel_info = list(self.db[self.__station_collection].aggregate(time_filter))

        if len(channel_info) == 0:
            logger.warning('No corresponding channel found!')
            return {}
        elif len(channel_info) > 1:
            err = (f"Found to many channels ({len(channel_info)}) for: station_id = {station_id}, "
                   f"channel_id = {channel_id}, and time = {detector_time}")
            logger.error(err)
            raise ValueError(err)
        
        # only return the channel information
        return channel_info[0]['channels']
    
    @check_database_time
    def get_general_device_information(self, station_id, device_id):
        """ Get information from one device. Access information in the main collection.
        
        Parameters
        ----------
        
        station_id: int
            specifiy the station id from which the device information is taken
        
        device_id: int
            specifiy the device id
        
        Returns
        -------
        
        info: dict
        """

        # if the collection is empty, return an empty dict
        if self.db[self.__station_collection].count_documents({'id': station_id}) == 0:
            return {}
        
        if self.__detector_time is None:
            detector_time = self.__database_time
            logger.error("Detector time is None, use database time.")
        else:
            detector_time = self.__detector_time

        # filter to get all information from one channel with station_id and channel_id and active commission time
        time_filter = [{"$match": {
            'commission_time': {"$lte": detector_time},
            'decommission_time': {"$gte": detector_time},
            'id': station_id}},
            {'$unwind': '$devices'},
            {"$match": {'devices.commission_time': {"$lte": detector_time},
            'devices.decommission_time': {"$gte": detector_time},
            'devices.id': device_id}}]

        # get all stations which fit the filter (should only be one)
        device_info = list(self.db[self.__station_collection].aggregate(time_filter))

        if len(device_info) == 0:
            logger.warning('No corresponding channel found!')
            return {}
        elif len(device_info) > 1:
            err = f"Found to many channels ({len(device_info)}) for: station_id = {station_id}, device_id = {device_id}, and time = {detector_time}"
            logger.error(err)
            raise ValueError(err)
        
        # only return the channel information
        return device_info[0]['devices']

        
    @check_database_time
    def get_collection_information_by_station_id(self, collection_name, station_id, measurement_name=None, channel_id=None, 
                                                 use_primary_time_with_measurement=False):
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
            Use the measurement name to select the requested data (not database time / primary time).
            If "use_primary_time_with_measurement" is True, use measurement_name and primary time to 
            find matching objects. (Default: None -> return measurement based on primary time)
        
        channel_id: int
            Unique identifier of the channel
            
        use_primary_time_with_measurement: bool
            If True (and measurement_name is not None), use measurement name and primary time to select objects.
            (Default: False)

        Returns
        -------
        
        info: list(dict)
        """
        
        # if the collection is empty, return an empty dict
        if self.db[collection_name].count_documents({'id': {'$regex': f'_stn{station_id}_'}}) == 0:
            return {}

        primary_time = self.__database_time
        
        # define the search filter
        search_filter = [{'$match': {'id': {'$regex': f'_stn{station_id}_'}}}, {'$unwind': '$measurements'}]

        if measurement_name is not None or channel_id is not None:
            search_filter.append({'$match': {}})

        if measurement_name is not None:
            # add {'measurements.measurement_name': measurement_name} to dict in '$match'
            search_filter[-1]['$match'].update(
                {'measurements.measurement_name': measurement_name})
            
        if channel_id is not None :
            # add {'measurements.channel_id': channel_id} to dict in '$match'
            search_filter[-1]['$match'].update({'measurements.channel_id': channel_id})
        

        if measurement_name is None or use_primary_time_with_measurement:
            search_filter += [
                {'$unwind': '$measurements.primary_measurement'},
                {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                            'measurements.primary_measurement.end': {'$gte': primary_time}}}]
        else:
            # measurement/object identified by soley by "measurement_name"
            pass
         
        search_result = list(self.db[collection_name].aggregate(search_filter))
        
        if search_result == []:
            return search_result
        
        # The following code block is necessary if the "primary_measurement" has several entries. Right now we always do that.
        
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


    @check_database_time
    def get_collection_information_by_id(self, collection_name, id, measurement_name=None, use_primary_time_with_measurement=False):
        """
        Get the information for a specified collection (will only work for 'station_position', 'channel_position' and 'signal_chain')
        if the id does not exist, {} will be returned. Return primary measurement unless measurement_name is specified.

        Parameters
        ----------
        
        collection_name: string
            Specify the collection, from which the information should be extracted (will only work for 'station_position', 
            'channel_position' and 'signal_chain')
        
        station_id: int
            The unique identifier of the station
        
        measurement_name: string
            Use the measurement name to select the requested data (not database time / primary time).
            If "use_primary_time_with_measurement" is True, use measurement_name and primary time to 
            find matching objects. (Default: None -> return measurement based on primary time)
        
        channel_id: int
            Unique identifier of the channel
            
        use_primary_time_with_measurement: bool
            If True (and measurement_name is not None), use measurement name and primary time to select objects.
            (Default: False)

        Returns
        -------
        
        info: list(dict)
        """
        
        # if the collection is empty, return an empty dict
        if self.db[collection_name].count_documents({'id': id}) == 0:
            return {}
        
        primary_time = self.__database_time
        
        # define the search filter
        search_filter = [{'$match': {'id': id}},
                         {'$unwind': '$measurements'}]
        
        if measurement_name is not None:
            search_filter.append({'$match': {'measurements.measurement_name': measurement_name}})
                                
        if measurement_name is None or use_primary_time_with_measurement:
            search_filter += [
                {'$unwind': '$measurements.primary_measurement'},
                {'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                            'measurements.primary_measurement.end': {'$gte': primary_time}}}]
        else:
            # measurement/object identified by soley by "measurement_name"
            pass    

        search_result = list(self.db[collection_name].aggregate(search_filter))
        
        if search_result == []:
            return search_result
        
        # The following code block is necessary if the "primary_measurement" has several entries. Right now we always do that.
        
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
        """ 
        Returns a list with all measurement names, ids, ... 
        or what is specified (example: wanted_quantity = measurements.measurement_name)
        """
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


    def get_station_position_identifier(self, station_id, detector_time):
        """ Get the station psoition identifier for the given station id and detector time. Access information in the main collection.
        
        Parameters
        ----------
        
        station id: int
            Specify the station for which the position identifier is return
        detector_time: datetime.datetime
            Time to select the commissioned station
        
        Returns
        -------
        
        station_position_id: str 
        """

        # if the collection is empty, return None
        if self.db[self.__station_collection].count_documents({'id': station_id}) == 0:
            return None

        # filter to get all information from one station with station_id and with active commission time
        time_filter = [{"$match": {
            'commission_time': {"$lte": detector_time},
            'decommission_time': {"$gte": detector_time},
            'id': station_id}}]

        # get all stations which fit the filter (should only be one)
        station_info = list(self.db[self.__station_collection].aggregate(time_filter))
        
        if len(station_info) == 0:
            logger.warning('No corresponding station found!')
            return None
        elif len(station_info) > 1:
            err = f"Found to many stations (f{len(station_info)}) for: station_id = {station_id}, and time = {detector_time}"
            logger.error(err)
            raise ValueError(err)

        return station_info[0]['id_position']
    

    def get_station_position(self, station_id=None, station_position_id=None, measurement_name=None,
                             verbose=False, use_primary_time_with_measurement=False):
        """ function to return the station position, returns primary unless measurement_name is not None """
        
        # if the station_position_id is given, the position is directly collected from the
        # station position collection (no need to look into the main collection again)
        if station_position_id is not None:
            pass
        # if station id is given, the station position identifier is collected from the main collection
        elif station_id is not None:
            # get the station_position_id
            # # set the detector time to the current time
            station_position_id = self.get_station_position_identifier(station_id, detector_time=self.get_detector_time())
        else:
            raise ValueError('Either the station_id or the station_position_id needes to be given!')

        # if measurement name is None, the primary measurement is returned
        collection_info = self.get_collection_information_by_id(
            'station_position', station_position_id, measurement_name=measurement_name, 
            use_primary_time_with_measurement=use_primary_time_with_measurement)

        # raise an error if more than one value is returned
        if len(collection_info) > 1:\
            raise ValueError
        
        # return empty dict if no measurement is found
        if len(collection_info) == 0:
            return {}

        # return the information
        if verbose:
            return collection_info[0]['measurements']
        else:
            return {'position': collection_info[0]['measurements']['position']}


    def get_channel_identifier(self, station_id, channel_id, detector_time):
        """ Get the channel position and signal identifier for the given station id, 
        channel id and detector time. Access information in the main collection.
        
        Parameters
        ----------
        
        station_id: int
            Specify the station for which the position identifier is return
        channel_id: int
            Specify the channel for which the position identifier is return
        detector_time: datetime.datetime
            Time to select the commissioned station
        
        Returns
        -------
        
        channel_position_id: str
        channel_signal_id: str 
        """

        # if the collection is empty, return None
        if self.db[self.__station_collection].count_documents({'id': station_id}) == 0:
            return None

        # filter to get all information from one channel with station_id and channel_id and active commission time
        time_filter = [{"$match": {
            'commission_time': {"$lte": detector_time},
            'decommission_time': {"$gte": detector_time},
            'id': station_id}},
            {'$unwind': '$channels'},
            {"$match": {'channels.commission_time': {"$lte": detector_time},
            'channels.decommission_time': {"$gte": detector_time},
            'channels.id': channel_id}}]

        # get all stations which fit the filter (should only be one)
        channel_info = list(self.db[self.__station_collection].aggregate(time_filter))

        if len(channel_info) == 0:
            logger.warning('No corresponding channel found!')
            return None, None
        elif len(channel_info) > 1:
            err = (f"Found to many channels ({len(channel_info)}) for: station_id = {station_id}, "
                   f"channel_id = {channel_id}, and time = {detector_time}")
            logger.error(err)
            raise ValueError(err)
        
        # only return the channel information
        return channel_info[0]['channels']['id_position'], channel_info[0]['channels']['id_signal']
    

    def get_channel_position(self, station_id=None, channel_id=None, channel_position_id=None, 
                             measurement_name=None, verbose=False, use_primary_time_with_measurement=False):
        """ function to return the channel position, returns primary unless measurement_name is not None """
        
        # if the channel_position_id is given, the position is directly collected from the channel position 
        # collection (no need to look into the main collection again)
        if channel_position_id is not None:
            pass
        # if channel id and station_id is given, the channel position identifier is collected from the main collection
        elif station_id is not None and channel_id is not None:
            # get the channel_position_id
            # # set the detector time to the current time
            self.set_detector_time(datetime.datetime.utcnow())
            channel_position_id, channel_signal_id = self.get_channel_identifier(
                station_id, channel_id, detector_time=self.get_detector_time())
        else:
            raise ValueError('Either the station_id + channel_id or the channel_position_id needes to be given!')

        # if measurement name is None, the primary measurement is returned
        collection_info = self.get_collection_information_by_id(
            'channel_position', channel_position_id, measurement_name=measurement_name, 
            use_primary_time_with_measurement=use_primary_time_with_measurement)

        # raise an error if more than one value is returned
        if len(collection_info) > 1:\
            raise ValueError
        # return empty dict if no measurement is found
        if len(collection_info) == 0:
            return {}

        # return the information
        if verbose:
            return collection_info[0]['measurements']
        else:
            return {k:collection_info[0]['measurements'][k] for k in ('position','rotation','orientation')}

    
    def get_device_position_identifier(self, station_id, device_id, detector_time):
        """ 
        Get the device position identifier for the given station id, device id and detector time. 
        Access information in the main collection.
        
        Parameters
        ----------
        
        station_id: int
            Specify the station for which the position identifier is return
        device_id: int
            Specify the channel for which the position identifier is return
        detector_time: datetime.datetime
            Time to select the commissioned station
        
        Returns
        -------
        
        device_position_id: str 
        """
        # if the collection is empty, return None
        if self.db[self.__station_collection].count_documents({'id': station_id}) == 0:
            return {}

        # filter to get all information from one device with station_id and device_id and active commission time
        time_filter = [{"$match": {
            'commission_time': {"$lte": detector_time},
            'decommission_time': {"$gte": detector_time},
            'id': station_id}},
            {'$unwind': '$devices'},
            {"$match": {'devices.commission_time': {"$lte": detector_time},
            'devices.decommission_time': {"$gte": detector_time},
            'devices.id': device_id}}]

        # get all stations which fit the filter (should only be one)
        device_info = list(self.db[self.__station_collection].aggregate(time_filter))

        if len(device_info) == 0:
            logger.warning('No corresponding device found!')
            return None
        elif len(device_info) > 1:
            err = (f"Found to many devices ({len(device_info)}) for: station_id = {station_id}, "  
                  f"channel_id = {device_id}, and time = {detector_time}")
            logger.error(err)
            raise ValueError(err)
        
        # only return the device position id
        return device_info[0]['devices']['id_position']


    def get_device_position(self, station_id=None, device_id=None, device_position_id=None, 
                            measurement_name=None, verbose=False, use_primary_time_with_measurement=False):
        """ function to return the device position, returns primary unless measurement_name is not None """
        
        # if the device_position_id is given, the position is directly collected from the device position collection 
        # (no need to look into the main collection again)
        if device_position_id is not None:
            pass
        # if device id and station_id is given, the device position identifier is collected from the main collection
        elif station_id is not None and device_id is not None:
            # get the device_position_id
            # # set the detector time to the current time
            self.set_detector_time(datetime.datetime.utcnow())
            device_position_id = self.get_device_position_identifier(station_id, device_id, detector_time=self.get_detector_time())
        else:
            raise ValueError('Either the station_id + device_id or the device_position_id needes to be given!')

        # if measurement name is None, the primary measurement is returned
        collection_info = self.get_collection_information_by_id(
            'device_position', device_position_id, measurement_name=measurement_name, 
            use_primary_time_with_measurement=use_primary_time_with_measurement)

        # raise an error if more than one value is returned
        if len(collection_info) > 1:\
            raise ValueError
        # return empty dict if no measurement is found
        if len(collection_info) == 0:
            return {}

        # return the information
        if verbose:
            return collection_info[0]['measurements']
        else:
            return {k:collection_info[0]['measurements'][k] for k in ('position','rotation','orientation')}
    

    def get_channel_signal_chain_measurement(self, station_id=None, channel_id=None, channel_signal_id=None, 
                                             measurement_name=None, verbose=False):
        """ function to return the channels signal chain information, returns primary unless measurement_name is not None """

        # if the channel_signal_id is given, the signal chain is directly collected from the signal chain collection 
        # (no need to look into the main collection again)
        if channel_signal_id is not None:
            pass
        # if channel_id and station_id is given, the signal chain identifier is collected from the main collection
        elif station_id is not None and channel_id is not None:
            # get the channel_signal_id
            # # set the detector time to the current time
            self.set_detector_time(datetime.datetime.utcnow())
            channel_position_id, channel_signal_id = self.get_channel_identifier(
                station_id, channel_id, detector_time=self.get_detector_time())
        else:
            raise ValueError('Either the station_id + channel_id or the channel_position_id needes to be given!')

        # if measurement name is None, the primary measurement is returned
        collection_info = self.get_collection_information_by_id(
            'signal_chain', channel_signal_id, measurement_name=measurement_name)

        # raise an error if more than one value is returned
        if len(collection_info) > 1:\
            raise ValueError
        # return empty dict if no measurement is found
        if len(collection_info) == 0:
            return {}

        # return the information
        if verbose:
            return collection_info[0]['measurements']
        else:
            return {k:collection_info[0]['measurements'][k] for k in ('VEL','sig_chain','primary_components')}


    def get_channel_signal_chain_component_data(self, component_type, component_id, supplementary_info, primary_time, verbose=True):
        """ returns the current primary measurement of the component, reads in the component collection"""
        
        # define a search filter
        search_filter = [{'$match': {'name': component_id}}, {'$unwind': '$measurements'}, {'$match': {}}]
    
        # if supplemenatry information exsits (like channel id, etc ...), update the search filter
        if supplementary_info != {}:
            for supp_info in supplementary_info.keys():
                search_filter[-1]['$match'].update({f'measurements.{supp_info}': supplementary_info[supp_info]})

        # add the S parameter to the search filter, only collect S21 parameter
        search_filter[-1]['$match'].update({f'measurements.S_parameter': 'S21'})

        search_filter.append({'$unwind': '$measurements.primary_measurement'})
        search_filter.append({'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                         'measurements.primary_measurement.end': {'$gte': primary_time}}})
        

        search_result = list(self.db[component_type].aggregate(search_filter))

        if len(search_result) != 1:
            raise ValueError('No or more than one measurement found!')

        measurement = search_result[0]['measurements']
        
        # remove 'id_measurement' object
        measurement.pop('id_measurement', None)

        if verbose:
            return measurement
        else:
            return {k:measurement[k] for k in ('name','channel_id','frequencies', 'mag', 'phase') if k in measurement.keys()}
    

    def get_complete_station_information(
            self, station_id, measurement_station_position=None, measurement_channel_position=None, 
            measurement_signal_chain=None, measurement_device_position=None, verbose=True):
        """
        Collects all available information about the station

        Parameters
        ----------
        
        station_id: int
            The unique identifier of the station the channel belongs to

        measurement_station_position: string
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

        # load general station information (dump of the main collection)
        general_info = self.get_general_station_information(station_id)
        
        #extract and delete the position identifier, drop the general channel and device information
        station_position_id = general_info[station_id]['id_position']
        general_channel_info = general_info[station_id]['channels']
        general_device_info = general_info[station_id]['devices']
        
        # remove '_id' object
        general_info[station_id].pop('_id')

        # get the station positions
        station_position = self.get_station_position(
            station_position_id=station_position_id, measurement_name=measurement_station_position, verbose=verbose)
        
        # remove 'id_measurement' object
        station_position.pop('id_measurement', None)
        
        # include the station position into the final dict
        general_info[station_id]['station_position'] = station_position

        # get the channel ids of all channels contained in the station
        channel_ids = list(general_channel_info.keys())

        # get the channel info
        channel_info = {}
        for cha_id in sorted(channel_ids):
            channel_info[cha_id] = self.get_complete_channel_information(
                station_id, cha_id, measurement_position=measurement_channel_position, 
                measurement_signal_chain=measurement_signal_chain, verbose=verbose)        

        # include the channel information into the final dict
        for channel_id in general_info[station_id]['channels']:
            # print(general_info[station_id]['channels'][channel_id].keys(), channel_info[channel_id].keys())
            general_info[station_id]['channels'][channel_id].update(channel_info[channel_id])

        # get the device ids of all devices contained in the station
        device_ids = list(general_device_info.keys())

        # get the device info
        device_info = {}
        for dev_id in sorted(device_ids):
            device_info[dev_id] = self.get_complete_device_information(
                station_id, dev_id, measurement_position=measurement_device_position,verbose=verbose)

        # include the device information into the final dict
        for device in general_info[station_id]['devices']:
            general_info[station_id]['devices'][device].update(device_info[device])

        return general_info
    
    
    def get_channel_signal_chain(self, channel_signal_id, measurement_name=None, verbose=True):
        """
        Returns the response data for a given signal chain.
        
        Parameters
        ----------
        
        channel_signal_id: str
            Indentifier of the signal chain
            
        Returns
        -------
        
        signal_chain: dict
            A dictinoary which among otherthings contains the "response_chain" which carries the measured response for the different 
            components in the signal chain.
        """

        # load the signal chain information:
        channel_sig_info = self.get_channel_signal_chain_measurement(
            channel_signal_id=channel_signal_id, measurement_name=measurement_name, verbose=verbose)
                
        # TODO: remove these hard-coded strings
        # get the component names 
        component_dict = channel_sig_info.pop('sig_chain')
        components = []
        components_id = []
        endings = ('_board', '_chain', '_cable')
        for key in component_dict.keys():
            if key.endswith(endings):
                components.append(key)
                components_id.append(component_dict[key])
        
        # get components data
        components_data = {}
        for component, component_id in zip(components, components_id):
            supp_info = {k[len(component)+1:]: component_dict[k] for k in component_dict.keys() if component in k and component != k}
            component_data = self.get_channel_signal_chain_component_data(
                component, component_id, supp_info, primary_time=self.__database_time, verbose=verbose)

            components_data[component] = {'name': component_id}
            components_data[component].update(component_data)

        # add/update the signal chain to the channel data
        channel_sig_info['response_chain'] = components_data
        
        return channel_sig_info


    def get_complete_channel_information(
            self, station_id, channel_id, measurement_position=None, measurement_signal_chain=None, verbose=True):
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
        
        measurement_signal_chain: string
            If not None, this measurement will be collected (even though it is not the primary measurement)
        
        Returns
        -------
        
        complete_info
        """

        # load general channel information
        general_info = self.get_general_channel_information(station_id, channel_id)

        #extract and delete the position/signal identifier
        position_id = general_info['id_position']
        signal_id = general_info['id_signal']

        # change name of signal chain entry to 'built_in_sig_chain'
        general_info['installed_components'] = general_info.pop('signal_ch')
        
        # load the channel position information:
        channel_pos_info = self.get_channel_position(
            channel_position_id=position_id, measurement_name=measurement_position, verbose=verbose)

        # include the channel position into the final dict
        general_info['channel_position'] = channel_pos_info

        channel_sig_info = self.get_channel_signal_chain(signal_id, measurement_signal_chain, verbose)
        
        # remove 'id_measurement' and 'channel_id' object
        channel_sig_info.pop('channel_id', None)
        
        general_info['signal_chain'] = channel_sig_info

        return general_info


    def get_complete_device_information(self, station_id, device_id, measurement_position=None, verbose=True):
        """
        Collects all available information about a device

        Parameters
        ----------
        
        station_id: int
            The unique identifier of the station the device belongs to
        
        device_id: int
            The device id for which the information will be written out
        
        measurement_position: string
            If not None, this measurement will be collected (even though it is not the primary measurement)
    

        Returns
        -------
        complete_info
        """
        complete_info = {}

        # load general device information
        general_info = self.get_general_device_information(station_id, device_id)

        #extract and delete the position identifier
        position_id = general_info.pop('id_position')

        # include the general info into the final dict
        complete_info.update(general_info)

        # load the device position information:
        device_pos_info = self.get_device_position(device_position_id=position_id, measurement_name=measurement_position, verbose=verbose)
        # remove 'id_measurement' and 'device_id' object
        device_pos_info.pop('id_measurement', None)
        device_pos_info.pop('device_id', None)

        # include the device position into the final dict
        complete_info['device_position'] = device_pos_info

        return complete_info
    
    
    def query_modification_timestamps_per_station(self):
        """
        Collects all the timestamps for station and channel (de)commissioning from the database.
        Combines those to get a list of timestamps when modifications happened which requiers to update the buffer.
        
        Returns
        -------
        
        station_data: dict(dict(list))
            Returns for each station (key = station.id) a dictionary with three entries:
            "modification_timestamps", "station_commission_timestamps", "station_decommission_timestamps" 
            each containing a list of timestamps. The former combines the latter two + channel (de)comission 
            timestamps.
        """
        # get distinct set of stations:
        station_ids = self.db[self.__station_collection].distinct("id")
        modification_timestamp_dict = {}
        for station_id in station_ids:
            # get set of (de)commission times for stations
            station_times_comm = self.db[self.__station_collection].distinct("commission_time", {"id": station_id})
            station_times_decomm = self.db[self.__station_collection].distinct("decommission_time", {"id": station_id})

            # get set of (de)commission times for channels
            channel_times_comm = self.db[self.__station_collection].distinct("channels.commission_time", {"id": station_id})
            channel_times_decomm = self.db[self.__station_collection].distinct("channels.decommission_time", {"id": station_id})

            mod_set = np.unique(station_times_comm + station_times_decomm + channel_times_comm + channel_times_decomm)
            mod_set.sort()
            station_times_comm.sort()
            station_times_decomm.sort()
            
            station_data = {
                "modification_timestamps": mod_set,
                "station_commission_timestamps": station_times_comm,
                "station_decommission_timestamps": station_times_decomm
            }
                        
            # store timestamps, which can be used with np.digitize
            modification_timestamp_dict[station_id] = station_data
        
        return modification_timestamp_dict


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
