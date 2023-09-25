import NuRadioReco.detector.RNO_G.db_mongo_read
import NuRadioReco.utilities.metaclasses
import six
import datetime

from bson import ObjectId

import logging
logging.basicConfig()
logger = logging.getLogger("database")
logger.setLevel(logging.DEBUG)


@six.add_metaclass(NuRadioReco.utilities.metaclasses.Singleton)
class Database(NuRadioReco.detector.RNO_G.db_mongo_read.Database):

    # operation changing collections

    def rename_database_collection(self, old_name, new_name):
        """
        changes the name of a collection of the database
        If the new name already exists, the operation fails.

        Parameters
        ----------
        old_name: string
            old name of the collection
        new_name: string
            new name of the collection
        """
        self.db[old_name].rename(new_name)

    def create_empty_collection(self, collection_name):
        self.db.create_collection(collection_name)

    def clone_collection_to_collection(self, old_collection, new_collection):
        self.db[old_collection].aggregate(
            [{'$match': {}}, {'$out': new_collection}])

    # operation adding documents to a collection

    def set_not_working(self, type, name, primary_measurement, channel_id=None, breakout_id=None, breakout_channel_id=None):
        """
        inserts that the input unit is broken.
        If the input unit dosn't exist yet, it will be created.

        Parameters
        ----------
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
            self.update_current_primary(type, name, channel_id=channel_id,
                                        breakout_id=breakout_id, breakout_channel_id=breakout_channel_id)

        # define the new primary measurement times
        primary_measurement_times = [{'start': datetime.datetime.utcnow(
        ), 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)}]

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

    def add_entry_to_database(self, collection, identification_key, identification_value, primary_measurement, data_dict):
        """
        inserts a entry into the database.
        If the measurement dosn't exist yet, it will be created.
        Only works for the following collections: hpol, vpol, surface, ...

        Parameters
        ----------
        collection: string
            specify the collection in which the entry will be added
        identification_key: string
            specify the key used for identification (must be 'name' or 'id')
        identification_value: string
            specify the name of the entry (e.g. the name of the measurement or the channel position identifier)
        primary_measurement: bool
            indicates the primary measurement to be used for analysis
        data_dict: dict
            dictionary with all the information that should be saved for this entry
        """

        self.set_database_time(datetime.datetime.utcnow())

        # close the time period of the old primary measurement
        if primary_measurement and identification_value in self.db[collection].distinct(identification_key):
            self.update_current_primary(
                collection, identification_value, identification_label=identification_key, data_dict=data_dict)

        # define the new primary measurement times
        if primary_measurement:
            primary_measurement_times = [{'start': datetime.datetime.utcnow(
            ), 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)}]
        else:
            primary_measurement_times = []

        # update the entry with the measurement (if the entry doesn't exist it will be created)
        data_dict.update({'id_measurement': ObjectId(
        ), 'primary_measurement': primary_measurement_times, 'last_updated': datetime.datetime.utcnow()})
        self.db[collection].update_one({identification_key: identification_value},
                                       {'$push': {'measurements': data_dict}}, upsert=True)

    def add_general_station_info(self, collection, station_id, station_name, station_comment, commission_time, decommission_time=datetime.datetime(2080, 1, 1)):
        # check if an active station exist; if true, the active station will be decommissioned
        # filter to get all active stations with the correct id
        time = self.__detector_time
        time_filter = [{"$match": {
            'commission_time': {"$lte": time},
            'decommission_time': {"$gte": time},
            'id': station_id}}]
        # get all stations which fit the filter (should only be one)
        stations = list(self.db[collection].aggregate(time_filter))

        if len(stations) > 0:
            self.decommission_a_station(
                collection, station_id, commission_time)

        # create uniqe position identifier
        position_identifier = f'position_stn{station_id}_{commission_time.month}{commission_time.year}'

        # insert the new station
        self.db[collection].insert_one({'id': station_id,
                                        'name': station_name,
                                        'commission_time': commission_time,
                                        'decommission_time': decommission_time,
                                        'station_comment': station_comment,
                                        'id_position': position_identifier,
                                        'channels': [],
                                        'devices': []
                                        })

    def add_general_channel_info_to_station(self, collection, station_id, channel_id, signal_chain, ant_type, channel_comment, commission_time, decommission_time=datetime.datetime(2080, 1, 1)):
        # get the current active station
        # filter to get all active stations with the correct id
        time = self.__detector_time
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
            self.decommission_a_channel(
                collection, station_id, channel_id, commission_time)

        # create uniqe position and signal chain identifier
        position_identifier = f'position_stn{station_id}_cha{channel_id}_{commission_time.month}{commission_time.year}'
        signal_identifier = f'signal_stn{station_id}_cha{channel_id}_{commission_time.month}{commission_time.year}'

        # insert the channel information
        self.db[collection].update_one({'_id': unique_station_id},
                                       {"$push": {'channels': {
                                           'id': channel_id,
                                           'id_position': position_identifier,
                                           'id_signal': signal_identifier,
                                           'ant_type': ant_type,
                                           'commission_time': commission_time,
                                           'decommission_time': decommission_time,
                                           'installed_components': signal_chain,
                                           'channel_comment': channel_comment
                                       }}
        })

    def add_general_device_info_to_station(self, collection, station_id, device_id, device_name, device_comment, amp_name, commission_time, decommission_time=datetime.datetime(2080, 1, 1)):
        # get the current active station
        # filter to get all active stations with the correct id
        time = self.__detector_time
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

        # check if for this device an entry already exists
        component_filter = [{'$match': {'_id': unique_station_id}},
                            {'$unwind': '$devices'},
                            {'$match': {'device.id': device_id}}]

        entries = list(self.db[collection].aggregate(component_filter))

        # check if the device already exist, decommission the active device first
        if entries != []:
            self.decommission_a_device(
                collection, station_id, device_id, commission_time)

        # create uniqe position and identifier
        position_identifier = f'position_stn{station_id}_dev{device_id}_{commission_time.month}{commission_time.year}'

        # insert the device information
        self.db[collection].update_one({'_id': unique_station_id},
                                       {"$push": {'devices': {
                                           'id': device_id,
                                           'id_position': position_identifier,
                                           'device_name': device_name,
                                           'amp_name': amp_name,
                                           'commission_time': commission_time,
                                           'decommission_time': decommission_time,
                                           'device_comment': device_comment
                                       }}
        })

    def add_measurement_protocol(self, protocol_name):
        # insert the new measurement protocol
        self.db['measurement_protocol'].insert_one({'protocol': protocol_name,
                                                   'inserted': datetime.datetime.utcnow()})

    # operation that change the primary status of a measurement

    def update_current_primary(self, type, name, identification_label, data_dict):
        """
        updates the status of primary_measurement, set the timestamp of the current primary measurement to end at datetime.utcnow()

        Parameters
        ----------
        type: string
            type of the input unit (HPol, VPol, surfCABLE, ...)
        name: string
            the unique identifier of the input unit
        _id: int
            if there is a channel or device id for the object, the id is used in the search filter mask
        id_label: string
            sets if a channel id ('channel') or device id ('device) is used
        """

        present_time = self.__database_time
        print('present_time', present_time)

        # find the current primary measurement
        # obj_id, measurement_id = self.find_primary_measurement(type, name, present_time, identification_label=identification_label, _id=_id, id_label=id_label, breakout_id=breakout_id, breakout_channel_id=breakout_channel_id)
        obj_id, measurement_id = self.find_primary_measurement(
            type, name, present_time, identification_label=identification_label, data_dict=data_dict)
        print(obj_id, measurement_id)
        if obj_id is None and measurement_id[0] == 0:
            #  no primary measurement was found and thus there is no measurement to update
            pass
        elif obj_id is None and measurement_id == [None]:
            raise ValueError(
                'More than one primary measurements are found. Please contact the database support.')
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

                self.db[type].update_one({'_id': obj_id}, {"$set": {
                                         "measurements.$[updateIndex].primary_measurement": primary_times}}, array_filters=[{"updateIndex.id_measurement": m_id}])

    def __change_primary_object_measurement(self, object_type, object_name, search_filter, channel_id=None, breakout_id=None, breakout_channel_id=None):
        """

        helper function to change the current active primary measurement for a single antenna measurement

        Parameters
        ----------
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
        current_obj_id, current_measurement_id = self.find_primary_measurement(
            object_type, object_name, present_time, channel_id=channel_id, breakout_id=breakout_id, breakout_channel_id=breakout_channel_id)
        for c_m_id in current_measurement_id:
            # find the current_measurement_id for the fitting S parameter
            filter_primary_times = [{'$match': {'_id': current_obj_id}},
                                    {'$unwind': '$measurements'},
                                    {'$match': {'measurements.id_measurement': c_m_id}}]

            info = list(self.db[object_type].aggregate(filter_primary_times))

            if info[0]['measurements']['S_parameter'] == search_results[0]['measurements']['S_parameter']:
                # the measurement id is fitting the S parameter

                if c_m_id == measurement_id and current_obj_id == object_id and measurement_id is not None:
                    logger.info(
                        'The specified measurement is already the primary measurement.')
                elif measurement_id is None or current_measurement_id is None:
                    pass
                else:
                    # update the old primary time (not using the 'update current primary measurement' function so that we can only update the entry of a single S parameter
                    primary_times_old = info[0]['measurements']['primary_measurement']
                    # # update the 'end' time to the present time
                    primary_times_old[-1]['end'] = present_time
                    self.db[object_type].update_one({'_id': object_id}, {"$set": {
                                                    "measurements.$[updateIndex].primary_measurement": primary_times_old}}, array_filters=[{"updateIndex.id_measurement": c_m_id}])

                    # update the primary measurements of the specified measurements
                    if object_id is not None:
                        primary_times.append(
                            {'start': present_time, 'end': datetime.datetime(2100, 1, 1, 0, 0, 0)})
                        self.db[object_type].update_one({'_id': object_id}, {"$set": {
                                                        "measurements.$[updateIndex].primary_measurement": primary_times}}, array_filters=[{"updateIndex.id_measurement": measurement_id}])
            else:
                logger.error('S parameter not selected to be changed.')

    def change_primary_antenna_measurement(self, antenna_type, antenna_name, S_parameter, protocol, units_arr, function_test):
        """
        changes the current active primary measurement for a single antenna measurement

        Parameters
        ----------
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

        self.__change_primary_object_measurement(
            antenna_type, antenna_name, search_filter)

    def change_primary_cable_measurement(self, cable_type, cable_name, S_parameter, protocol, units_arr, function_test):
        """
        changes the current active primary measurement for a single cable measurement

        Parameters
        ----------
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

        self.__change_primary_object_measurement(
            cable_type, cable_name, search_filter)

    def change_primary_iglu_measurement(self, board_type, board_name, S_parameter, protocol, units_arr, function_test, drab_id, laser_id, temperature):
        """
        changes the current active primary measurement for a single board measurement

        Parameters
        ----------
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

        self.__change_primary_object_measurement(
            board_type, board_name, search_filter)

    def change_primary_drab_measurement(self, board_type, board_name, S_parameter, iglu_id, photodiode_id, channel_id, temp, protocol, units_arr, function_test):
        """
        changes the current active primary measurement for a single board measurement

        Parameters
        ----------
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

        self.__change_primary_object_measurement(
            board_type, board_name, search_filter, channel_id=channel_id)

    def change_primary_surface_measurement(self, board_type, board_name, S_parameter, channel_id, temp, protocol, units_arr, function_test):
        """
        changes the current active primary measurement for a single board measurement

        Parameters
        ----------
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

        self.__change_primary_object_measurement(
            board_type, board_name, search_filter, channel_id=channel_id)

    def change_primary_downhole_measurement(self, board_type, board_name, S_parameter, breakout_id, breakout_cha_id, iglu_id, drab_id, temp, protocol, units_arr, function_test):
        """
        changes the current active primary measurement for a single board measurement

        Parameters
        ----------
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

        self.__change_primary_object_measurement(
            board_type, board_name, search_filter, breakout_id=breakout_id, breakout_channel_id=breakout_cha_id)

    def change_primary_station_measurement(self):
        pass

    def change_primary_channel_measurement(self):
        pass

    def change_primary_channel_signal_chain_configuration(self):
        pass

    # operation that decommission a object

    def decommission_a_station(self, collection, station_id, decomm_time):
        """
        function to decommission an active station in the db

        Parameters
        ----------
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
                self.db[collection].update_one(
                    {'_id': object_id}, {'$set': {'decommission_time': decomm_time}})

    def decommission_a_channel(self, collection, station_id, channel_id, decomm_time):
        """
        function to decommission an active channel in the db

        Parameters
        ----------
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
            time = decomm_time
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

    def decommission_a_device(self, collection, station_id, device_id, decomm_time):
        """
        function to decommission an active device in the db

        Parameters
        ----------
        collection: string
            name of the collection
        station_id: int
            the unique identifier of the station
        device_id: int
            the unique identifier of the device
        decomm_time: datetime
            time which should be used for updating the decommission time
        """
        # get the entry of the active station
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

                # change the decommission time of a specific device
                self.db[collection].update_one({'_id': object_id}, {'$set': {'devices.$[updateIndex].decommission_time': decomm_time}},
                                               array_filters=[{"updateIndex.id": device_id}])
