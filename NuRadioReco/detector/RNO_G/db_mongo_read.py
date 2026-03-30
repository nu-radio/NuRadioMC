"""
Interface to the MongoDB that contains RNO-G hardware and calibration information

The :mod:`NuRadioReco.detector.RNO_G.db_mongo_read` module and the `Database` class herein mostly serve as the
backend of the `NuRadioReco.detector.RNO_G.rnog_detector.Detector` class. Most users
will want to use that class to obtain information about deployed RNO-G stations and hardware.
`NuRadioReco.detector.RNO_G.rnog_detector.Detector` class has an interface similar to that of
other detector descriptions in NuRadioMC, and is documented there.

However, for some specific use cases (e.g. finding measurements for individual hardware components
that have not been deployed to the field), one can use the `Database` class directly, using the
`Database.get_component_data` method.

"""

import six
import os
import urllib.parse
import datetime
import numpy as np
from functools import wraps
import copy
import re

from pymongo import MongoClient
import NuRadioReco.utilities.metaclasses
import astropy.time


import logging
logger = logging.getLogger("NuRadioReco.MongoDBRead")


def _convert_astro_time_to_datetime(time_astro):
    return time_astro.to_datetime()


def _check_database_time(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        time = self.get_database_time()
        if time is None:
            logger.error('Database time is None.')
            raise ValueError('Database time is None.')
        return method(self, *method_args, **method_kwargs)
    return _impl


@six.add_metaclass(NuRadioReco.utilities.metaclasses.Singleton)
class Database(object):

    def __init__(self, database_connection="RNOG_public", database_name=None, mongo_kwargs={}):
        """
        Interface to the RNO-G hardware database. This class uses the python API pymongo for the
        RNO-G MongoDB.

        This classes allows you to connect to preconfigured mongo clients or select your mongo client freely.
        The database is accesible with the `self.db` variable.

        Parameters
        ----------

        database_connection : str (Default: `"RNOG_public"`)
            Specify mongo client. You have 5 options:

                * `"env_pw_user"`: Connect to a server with the environmental variables
                  `mongo_server`, `mongo_user`, and `mongo_password`
                * `"RNOG_public"`: Preconfigured connection to read-only RNO-G Hardware DB
                * `"RNOG_test_public"`: Preconfigured connection to read-only RNO-G Test Hardware DB
                * `"connection_string"`: Use environmental variable `db_mongo_connection_string` to
                  connect to mongo server
                * `"mongodb*": Every string which starts with `"mongodb"` will be used to connect to
                  a mongo server

        database_name : str (Default: None -> `"RNOG_hardware_v0"`)
            Select the database by name. If None (default) is passed, set to `"RNOG_hardware_v0"`

        mongo_kwargs : dict (Default: `{}`)
            Additional arguments to pass to MongoClient.
        """

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
            connection_string = f"mongodb://{mongo_user}:{mongo_password}@{mongo_server}"
            mongo_kwargs["tls"] = True

        elif database_connection == "RNOG_public":
            # use read-only access to the RNO-G database
            connection_string = (
                "mongodb://read:EseNbGVaCV4pBBrt@radio.zeuthen.desy.de:27017/admin?authSource=admin&"
                "readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=true")

        elif database_connection == "RNOG_test_public":
            # use readonly access to the RNO-G test database
            connection_string = (
                "mongodb://RNOG_test_public:jrE5xO38D7wQweVR5doa@radio-test.zeuthen.desy.de:27017/admin?authSource=admin&"
                "readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=true")

        elif database_connection == "connection_string":
            # use a connection string from the environment
            connection_string = os.environ.get('db_mongo_connection_string')
        elif database_connection.startswith("mongodb"):
            connection_string = database_connection
        else:
            logger.error('specify a defined database connection '
                         '["env_pw_user", "connection_string", "RNOG_public", "RNOG_test_public", "mongodb..."]')


        self.__mongo_client = MongoClient(connection_string, **mongo_kwargs)

        if database_name is None:
            database_name = "RNOG_hardware_v0"

        if database_name not in self.__mongo_client.list_database_names():
            logger.error(f'Could not find database "{database_name}" in mongo client.')
            raise KeyError

        logger.info("Attempting to connect to the database ...")
        self.db = self.__mongo_client[database_name]
        logger.info(f"... connection to {self.db.name} established")

        # Set timestamp of database. This is used to determine which primary measurement is used
        self.__database_time = datetime.datetime.now(tz=datetime.timezone.utc)

        # This is used to get commissioned stations/channels
        self.__detector_time = None

        self.__get_collection_names = None

        self.__station_collection = "station_rnog"

        self.__digitizer_collection = "digitizer_configuration"



    def set_database_time(self, time):
        ''' Set time for database. This affects which primary measurement is used.

        Parameters
        ----------

        time: `datetime.datetime` or ``astropy.time.Time``
            UTC time.
        '''
        if isinstance(time, astropy.time.Time):
            time = _convert_astro_time_to_datetime(time)

        if not isinstance(time, datetime.datetime):
            logger.error("Set invalid time for database. Time has to be of type datetime.datetime")
            raise TypeError("Set invalid time for database. Time has to be of type datetime.datetime")

        if time.tzinfo is None:
            logger.warning(f"No timezone information provided for the database time {time}. Assuming UTC.")
            time = time.replace(tzinfo=datetime.timezone.utc)

        if time < datetime.datetime(2024, 11, 1, tzinfo=datetime.timezone.utc):
            logger.error("Set invalid time for database. Time has to be after 01.11.2024 (1st november 2024), before that the database was not used to describe the detector.")
            raise ValueError("Set invalid time for database. Time has to be after 01.11.2024 (1st november 2024), before that the database was not used to describe the detector.")

        self.__database_time = time


    def set_detector_time(self, time):
        ''' Set time of detector. This controls which stations/channels are commissioned.

        Parameters
        ----------

        time: `datetime.datetime` or ``astropy.time.Time``
            UTC time.
        '''
        if isinstance(time, astropy.time.Time):
            time = _convert_astro_time_to_datetime(time)

        if not isinstance(time, datetime.datetime):
            logger.error("Set invalid time for detector. Time has to be of type datetime.datetime")
            raise TypeError("Set invalid time for detector. Time has to be of type datetime.datetime")
        self.__detector_time = time


    def get_database_time(self):
        return self.__database_time

    def get_detector_time(self):
        return self.__detector_time

    def find_primary_measurement(self, collection_name, name, primary_time, identification_label, data_dict):
        """
        Find the object_id of entry with name 'name' and gives the measurement_id of the primary measurement,
        return the id of the object and the measurement

        Parameters
        ----------
        collection_name: string
            name of the collection that is searched (surface_board, iglu_board, ...)

        name: string
            the unique identifier of the input component

        primary_time: datetime.datetime
            timestamp for the primary measurement

        identification_label: string
            specify what kind of label is used for the identification ("name" or "id")

        data_dict: dict
            dictionary containing additional information that are used to search the database (e.g., channel_id, S_parameter)
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
        matching_entries = list(self.db[collection_name].aggregate(filter_primary))

        # extract the object and measurement id
        if len(matching_entries) > 1:
            logger.error('More than one primary measurement found.')
            return None, [None]
        elif len(matching_entries) == 0:
            logger.warning('No primary measurement found.')
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

    def get_station_ids(self):
        return self.db[self.__station_collection].distinct('id')

    def load_board_information(self, type, board_name, info_names):
        """ Load the information for a single component from the database (can be used for IGLU / DRAB) """
        infos = []
        board_data = self.db[type].find_one({'name': board_name})
        for i in range(len(board_data['measurements'])):
            if board_data['measurements'][i]['function_test']:
                for name in info_names:
                    infos.append(board_data['measurements'][i][name])
                break

        return infos


    def get_digitizer_configuration(self, config_id):
        """ Get digitizer configuration from the database. Access information in the digitizer collection.

        Parameters
        ----------

        config_id: int
            Identifier to get the correct configuration

        Returns
        -------

        config: dict
        """

        # filter to get the correct configuration
        filter = [{"$match": {'id': config_id}}]

        # query the configuration from the database
        config = list(self.db[self.__digitizer_collection].aggregate(filter))

        if len(config) > 1:
            err = f"Found to many digitizer configurations (f{len(config)}) for: config_id = {config_id}"
            logger.error(err)
            raise ValueError(err)
        elif len(config) == 0:
            err = f"Found no digitizer configuration for: config_id = {config_id}"
            logger.error(err)
            raise ValueError(err)

        return config[0]


    @_check_database_time
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
            for entry in stations_for_buffer[0][key]:
                if not entry['commission_time'] <= detector_time <= entry['decommission_time']:
                    commissioned_info[0][key].remove(entry)

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

        # load the signal / trigger digitizer configuration
        for digitizer_type in ["signal", "trigger"]:
            # get the correct key for the dict
            digitizer_key = digitizer_type + "_digitizer_config"

            # get the id and load the information from the digitizer collection
            digitizer_id = station_info[station_id][digitizer_key]
            station_info[station_id][digitizer_key] = self.get_digitizer_configuration(config_id=digitizer_id)

        return station_info

    @_check_database_time
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

    @_check_database_time
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

    @_check_database_time
    def get_collection_information(self, collection_name, search_by, obj_id, measurement_name=None, channel_id=None, use_primary_time=None):
        """
        Get the information for a specified collection (will only work for 'station_position', 'channel_position' and 'signal_chain')
        if the station does not exist, {} will be returned. Return primary measurement unless measurement_name is specified.

        Parameters
        ----------

        collection_name: string
            Specify the collection, from which the information should be extracted (will only work for 'station_position',
            'channel_position' and 'signal_chain')

        search_by: string
            Specify if the collection is searched by 'station_id' or 'id'. The latter is a position or signal chain identifier

        obj_id: string or int
            station id or position/signal_chain identifier

        measurement_name: string
            Use the measurement name to select the requested data. (Default: None -> return measurement based on primary time)

        channel_id: int
            Unique identifier of the channel. Only allowed if searched by 'station_id'

        use_primary_time: bool
            If True, use primary time to select objects.
            (Default: None -> Set to `use_primary_time = measurement_name is None`)

        Returns
        -------

        info: list(dict)
        """

        if use_primary_time is None:
            use_primary_time = measurement_name is None

        if search_by == 'station_id':
            id_dict = {'id': {'$regex': f'_stn{obj_id}_'}}
        elif search_by == 'id':
            id_dict = {'id': obj_id}
        else:
            raise ValueError('Only "station_id" and "id" are valid options for the "search_by" argument.')

        # if the collection is empty, return an empty dict
        if self.db[collection_name].count_documents(id_dict) == 0:
            return {}

        # define the search filter
        search_filter = [{'$match': id_dict},
                         {'$unwind': '$measurements'}]

        if measurement_name is not None or channel_id is not None:
            search_filter.append({'$match': {}})

        if measurement_name is not None:
            # add {'measurements.measurement_name': measurement_name} to dict in '$match'
            search_filter[-1]['$match'].update(
                {'measurements.measurement_name': measurement_name})

        if channel_id is not None:
            if search_by == 'id':
                raise ValueError('channel_id can only be used if the collection is searched by "station_id"')

            # add {'measurements.channel_id': channel_id} to dict in '$match'
            search_filter[-1]['$match'].update({'measurements.channel_id': channel_id})

        if use_primary_time:
            search_filter += [
                {'$unwind': '$measurements.primary_measurement'},
                {'$match': {'measurements.primary_measurement.start': {'$lte': self.__database_time},
                            'measurements.primary_measurement.end': {'$gte': self.__database_time}}}]
        else:
            # measurement/object identified by soley by "measurement_name"
            pass


        search_result = list(self.db[collection_name].aggregate(search_filter))

        if len(search_result) != 1:
            logger.warning(f'Found {len(search_result)} matching entries in collection {collection_name} for {search_filter}.')

        if search_result == []:
            return search_result

        # The following code block is necessary if the "primary_measurement" has several entries. Right now we always do that.
        # Extract the information using the object and measurements id
        id_filter = [{'$match': {'_id': {'$in': [dic['_id'] for dic in search_result]}}},
                     {'$unwind': '$measurements'},
                     {'$match': {'measurements.id_measurement':
                         {'$in': [dic['measurements']['id_measurement'] for dic in search_result]}}}]

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

            search_result = list(self.db[collection].aggregate(search_filter))

            for key in input_dic.keys():
                help_list = []
                for entry in search_result:
                    help_list.append(entry['measurements'][key])
                return_dic[key] = list(set(help_list))

        return return_dic


    def get_identifier(self, station_id, channel_device_id=None, component="station", what="signal"):
        """
        Get the identifier for a station/channel/device measurement,

        For station and device returns position identifer. For channel returns
        position and signal chain identifier.

        Access information in the main collection.

        Parameters
        ----------

        station id: int
            Specify the station for which the measurement identifier is return

        channel_device_id: int
            Specify the channel/device id. Only necessary if component="channel" or "device.
            (Default: None)

        component: str
            Specify for what you want to have the identifier(s):
            "station" (default), "channel", or "device"

        what: str
            For what to return the identifier: "position" (default) or "signal_chain" (only available for "channel")

        Returns
        -------

        position_id: str
            Unique identifier to find measurement in different collection
        """

        # if the collection is empty, return None
        if self.db[self.__station_collection].count_documents({'id': station_id}) == 0:
            return None

        detector_time = self.get_detector_time()

        # filter to get all information from one station with station_id and with active commission time
        time_filter = [{"$match": {
            'commission_time': {"$lte": detector_time},
            'decommission_time': {"$gte": detector_time},
            'id': station_id}}]

        if component == "channel" or component == "device":
            if channel_device_id is None:
                raise ValueError(f"Please provide a channel id.")

            comp_str = component + "s"
            time_filter += [{'$unwind': f'${comp_str}'},
                {"$match": {f'{comp_str}.commission_time': {"$lte": detector_time},
                f'{comp_str}.decommission_time': {"$gte": detector_time},
                f'{comp_str}.id': channel_device_id}}]
        elif component == "station":
            pass  # do nothing here
        else:
            err = (f"Requested identifer for unknown component: {component}. "
                "Only valid components are \"station\", \"channel\", \"device\".")
            logger.warning(err)
            raise ValueError(err)

        # get all stations which fit the filter (should only be one)
        info = list(self.db[self.__station_collection].aggregate(time_filter))

        if len(info) == 0:
            err = (f"Could not find corresponding station/channel/device "
                   f"({component}: station_id = {station_id}, channel/device id "
                   f"= {channel_device_id}, time = {detector_time}")
            logger.warning(err)
            raise ValueError(err)

        elif len(info) > 1:
            err = (f"Found to many stations/channels/devices (f{len(info)}) "
                   f"({component}: station_id = {station_id}, channel/device id "
                   f"= {channel_device_id}, time = {detector_time}")
            logger.error(err)
            raise ValueError(err)

        if component == "station":
            return info[0]['id_position']
        elif component == "channel":
            return info[0]['channels'][f'id_{what}']
        elif component == "device":
            # only return the device position id
            return info[0]['devices']['id_position']


    def get_position(self, station_id=None, channel_device_id=None, position_id=None,
                     measurement_name=None, use_primary_time=None,
                     component="station", verbose=False):
        """
        Function to return the channel position,
        returns primary unless measurement_name is not None
        """

        if use_primary_time is None:
            use_primary_time = measurement_name is None

        # If the channel_position_id is given, the position is directly collected from the channel position
        # collection (no need to look into the main collection again)
        if position_id is None:
            if station_id is None:
                raise ValueError('Either the position_id or station_id (+ channel_id/device_id) needes to be given!')

            position_id = self.get_identifier(
                station_id, channel_device_id, component=component, what='position')

        # if measurement name is None, the primary measurement is returned
        collection_info = self.get_collection_information(
            f'{component}_position', search_by='id', obj_id=position_id, measurement_name=measurement_name,
            use_primary_time=use_primary_time)

        # raise an error if more than one value is returned
        if len(collection_info) > 1:
            raise ValueError
        # return empty dict if no measurement is found
        if len(collection_info) == 0:
            return {}

        # return the information
        if verbose:
            return collection_info[0]['measurements']
        else:
            return {k: collection_info[0]['measurements'][k] for k in
                    ['position', 'rotation', 'orientation'] if k in collection_info[0]['measurements']}


    def get_channel_signal_chain_measurement(self, station_id=None, channel_id=None, channel_signal_id=None,
                                             measurement_name=None, verbose=False):
        """ function to return the channels signal chain information, returns primary unless measurement_name is not None """

        # if the channel_signal_id is given, the signal chain is directly collected from the signal chain collection
        # (no need to look into the main collection again)
        if channel_signal_id is None:

            if station_id is None :
                raise ValueError('Either the channel_signal_id or station_id + channel_id needes to be given!')

            channel_signal_id = self.get_identifier(
                station_id, channel_id, component="channel", what="signal")

        # if measurement name is None, the primary measurement is returned
        collection_info = self.get_collection_information(
            'signal_chain', search_by='id', obj_id=channel_signal_id, measurement_name=measurement_name)

        # raise an error if more than one value is returned
        if len(collection_info) > 1:
            raise ValueError("More than one signal chain measurement found!")

        # return empty dict if no measurement is found
        if len(collection_info) == 0:
            return {}

        # return the information
        if verbose:
            return collection_info[0]['measurements']
        else:
            return {k:collection_info[0]['measurements'][k] for k in ('VEL', 'response_chain', 'primary_components')}


    def get_component_data(self, collection_name, component_name, supplementary_info={}, primary_time=None, verbose=True, sparameter='S21', use_primary_time=True):
        """
        Returns a single measurement of a component. (e.g. gain of an IGLU)

        Parameters
        ----------
        collection_name : str
            Specify collection to look for component. Previously named `component_type`
            (collection names are typically the type of the component, e.g.. 'iglu_board').
        component_name : str
            Name of the component (e.g. 'C0069'). Previously named `component_id`.
        supplementary_info : dict
            Additional information to filter the measurement (e.g. {'channel_id': 0}).
        primary_time : datetime
            Time to use for the primary measurement (if None, the current time is used).
        verbose : bool
            If True, return the full measurement, otherwise return a simplified version.
        sparameter : str
            S-parameter to retrieve (e.g. 'S21').
        use_primary_time : bool
            If True, use the primary time to select the measurement.

        Examples
        --------

        .. code-block::

            import NuRadioReco.detector.RNO_G.db_mongo_read
            import datetime

            db = NuRadioReco.detector.RNO_G.db_mongo_read.Database()

            # gives you the entry in the database
            database_entry = db.get_component_data(
                collection_name='iglu_board',
                component_name='C0069',
                supplementary_info={}, # if you want a DRAB you have to specify the channel: {'channel_id':0}
                verbose=True,
                sparameter='S21', # you can also read the other S parameters
                primary_time=datetime.datetime.now())


            # extract the gain + phase data
            y_axis_units = database_entry['y-axis_units']
            frequencies = database_entry['frequencies']
            gain_data = database_entry['mag']
            phase_data = database_entry['phase']

        """

        if primary_time is None:
            primary_time = self.get_database_time()

        # define a search filter
        search_filter = [{'$match': {'name': component_name}}, {'$unwind': '$measurements'}, {'$match': {}}]

        # if supplementary information exists (like channel id, etc ...), update the search filter
        if supplementary_info != {}:
            for supp_info in supplementary_info.keys():
                search_filter[-1]['$match'].update({f'measurements.{supp_info}': supplementary_info[supp_info]})

        # add the S parameter to the search filter, only collect single S parameter
        search_filter[-1]['$match'].update({'measurements.S_parameter': sparameter})

        if use_primary_time:
            search_filter.append({'$unwind': '$measurements.primary_measurement'})
            search_filter.append({'$match': {'measurements.primary_measurement.start': {'$lte': primary_time},
                                            'measurements.primary_measurement.end': {'$gte': primary_time}}})

        search_result = list(self.db[collection_name].aggregate(search_filter))

        if len(search_result) != 1:
            raise ValueError(f'Found {len(search_result)} measurements with {search_filter}:\n {search_result}')

        measurement = search_result[0]['measurements']

        # remove 'id_measurement' object
        measurement.pop('id_measurement', None)

        if verbose:
            return measurement
        else:
            return {k:measurement[k] for k in ('name', 'channel_id', 'frequencies', 'mag', 'phase') if k in measurement.keys()}


    def get_time_dependent_factor(self, collection, search_id, measurement_name=None, use_primary_time=None):
        """
        Get the time dependent factor for a given station and channel id for the current detector time. 
        Time dependent factors are quantities that have commissioning and decommissioning times (e.g., gain calibration values or time delays).
        
        Parameters
        ----------
        collection: string
            Name of the collection in which the gain calibration is stored.

        search_id: int
            Combination of channel and station id. Used toegther with the detector time to select the correct absolute gain factor.

        measurement_name: string, optional
            Use the measurement name to select the requested data (not database time / primary time).
            If "use_primary_time" is explicitly set to True, use measurement_name and primary time to
            find matching objects. (Default: None -> return measurement based on primary time)

        use_primary_time: bool, optional
            If True, use primary time to find matching objects.
            (Default: None -> use primary time if measurement_name is None)

        Returns
        -------

        factor_dict: dict
        """

        if use_primary_time is None:
            use_primary_time = measurement_name is None

        # define the id that is used to search the collection
        id_dict = {'id': search_id,
                   'commission_time': {'$lte': self.__detector_time},
                   'decommission_time': {'$gte': self.__detector_time}}

        # if the collection is empty, return an empty dict
        if self.db[collection].count_documents(id_dict) == 0:
            logger.error(f'No entry found for id {search_id} at time {self.__detector_time}.')
            raise ValueError(f'No entry found for id {search_id} at time {self.__detector_time}.')

        # define the search filter
        search_filter = [{'$match': id_dict},
                         {'$unwind': '$measurements'}]

        search_filter.append({'$match': {}})

        if measurement_name is not None:
            # add {'measurements.measurement_name': measurement_name} to dict in '$match'
            search_filter[-1]['$match'].update(
                {'measurements.measurement_name': measurement_name})


        if use_primary_time:
            search_filter += [
                {'$unwind': '$measurements.primary_measurement'},
                {'$match': {'measurements.primary_measurement.start': {'$lte': self.__database_time},
                            'measurements.primary_measurement.end': {'$gte': self.__database_time}}}]
        else:
            # measurement/object identified by soley by id (and optional "measurement_name")
            pass

        search_result = list(self.db[collection].aggregate(search_filter))

        if search_result == []:
            logger.error(f'No entry found for id {search_filter} at time {self.__detector_time}.')
            raise ValueError(f'No entry found for id {search_filter} at time {self.__detector_time}.')
        elif len(search_result) > 1:
            logger.error(f'More than one entry found for id {search_filter} at time {self.__detector_time}.')
            raise ValueError(f'More than one entry found for id {search_filter} at time {self.__detector_time}.')

        # The following code block is necessary if the "primary_measurement" has several entries. Right now we always do that.
        # Extract the information using the object and measurements id
        id_filter = [
            {'$match': {'_id': {'$in': [dic['_id'] for dic in search_result]}}},
            {'$unwind': '$measurements'},
            {'$match': {'measurements.id_measurement':
                {'$in': [dic['measurements']['id_measurement'] for dic in search_result]}}}]

        info = list(self.db[collection].aggregate(id_filter))[0]

        # remove the measurements dict, and move the content to the main dict
        info.update(info.pop('measurements'))

        # remove 'id_measurement' and '_id' object
        info.pop('_id', None)
        info.pop('id_measurement', None)
        info["name"] = info.pop('id', None)

        return info


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
        station_position = self.get_position(
            position_id=station_position_id, measurement_name=measurement_station_position, verbose=verbose)

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

        measurement_name: str
            If not None, this measurement will be used to select the correct signal chain (not used in subsequent calls to
            `get_component_data` or `get_time_dependent_factor`)

        Returns
        -------

        signal_chain: dict
            A dictinoary which among otherthings contains the "response_chain" which carries the measured response for the different
            components in the signal chain.
        """

        # load the channel signal chain information (which components are used in the signal chain per channel):
        channel_sig_info = self.get_channel_signal_chain_measurement(
            channel_signal_id=channel_signal_id, measurement_name=measurement_name, verbose=verbose)

        for chain_key in ['response_chain', 'trigger_response_chain']:

            # Not every channel has a trigger response chain
            if chain_key not in channel_sig_info:
                continue

            # go through the component list query the corresponing measurements from the database (s parameters)
            for ice, component_entry in enumerate(channel_sig_info[chain_key]):
                if component_entry["collection"] in ["gain_calibration", "time_delays"]:
                    # only load a single calibration value
                    component_data = self.get_time_dependent_factor(
                        collection=component_entry["collection"], search_id=component_entry["id"])
                else:
                    # create a search dict with addtional informations
                    supp_info = {key: component_entry[key] for key in component_entry.keys() if re.search("(channel|breakout)", key)}

                    component_data = self.get_component_data(
                        collection_name=component_entry["collection"],
                        component_name=component_entry["name"],
                        supplementary_info=supp_info,
                        verbose=verbose,
                        sparameter='S21')

                # add the component data to the channel_sig_info dict
                channel_sig_info[chain_key][ice].update(component_data)

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

        # load the channel position information:
        channel_pos_info = self.get_position(
            position_id=position_id, measurement_name=measurement_position, verbose=verbose, component="channel")

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
        device_pos_info = self.get_position(position_id=position_id, measurement_name=measurement_position, verbose=verbose, component="device")
        # remove 'id_measurement' and 'device_id' object
        device_pos_info.pop('id_measurement', None)
        device_pos_info.pop('device_id', None)

        # include the device position into the final dict
        complete_info['device_position'] = device_pos_info

        return complete_info


    def query_modification_timestamps_per_station(self, station_ids=None):
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
        if isinstance(station_ids, int):
            station_ids = [station_ids]

        if station_ids is None:
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
