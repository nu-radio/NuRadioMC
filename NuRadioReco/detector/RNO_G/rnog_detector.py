# Keep that the first import
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s:%(name)s:%(funcName)s : %(message)s", datefmt="%H:%M:%S")

from NuRadioReco.detector.RNO_G.db_mongo_read import Database, filtered_keys
import NuRadioReco.framework.base_trace
from NuRadioReco.utilities import units

from radiotools import helper
from scipy import interpolate
from functools import wraps

import datetime
import numpy as np
import collections
import pickle


def keys_not_in_dict(d, keys):
    """ Checks sequentially if a list of `keys` is in a dictionary.

    Example:
    keys = ["key1", "key2"]

    Returns False if d["key1"]["key2"] exsits, True otherwise.
    """
    d_tmp = d
    for key in keys:
        try:
            if key not in d_tmp:
                return True
        except KeyError:
            return True

        d_tmp = d_tmp[key]

    return False


def check_detector_time(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        self.get_detector_time()  # this will raise an error if time is not set
        return method(self, *method_args, **method_kwargs)
    return _impl


class Detector():
    def __init__(self, database_connection='RNOG_test_public', log_level=logging.INFO, over_write_handset_values={},
                 database_time=None, always_query_entire_description=True,
                 pickle_file=None):
        """

        Parameters
        ----------

        database_connection: str
            Allows to specify database connection. Passed to mongo-db interface. (Default: 'RNOG_test_public')

        log_level: enum
            Defines verbosity level of logger. (Default: logging.DEBUG)

        over_write_handset_values: dict
            Overwrite the default values for the manually set parameter which are not (yet) implemented in the database.
            (Default: {}, the acutally default values for the parameters in question are defined below)

        database_time: datetime.datetime
            Set database time which is used to select the primary measurement. By default (= None) the database time
            is set to now (time the code is running) to select the measurement which is now primary.

        always_query_entire_description: bool
            If True, query the entire detector describtion all at once when calling Detector.update(...) (if necessary).
            (Default: True)
        """

        self.logger = logging.getLogger("rno-g-detector")
        self.logger.setLevel(log_level)

        # Define default values for parameter not (yet) implemented in DB. Those values are taken for all channels.
        self.__default_values = {
            "noise_temperature": 300 * units.kelvin,
            "sampling_frequency": 3.2 * units.GHz,
            "number_of_samples": 2048,
            "is_noiseless": False,
            "cable_delay": 0.
        }

        if pickle_file is None:
            self._det_imported_from_file = False

            self.__db = Database(database_connection=database_connection)
            if database_time is not None:
                self.__db.set_database_time(database_time)

            self.logger.info(
                "Collect time periods of station commission/decommission ...")
            self._time_periods_per_station = self.__db.query_modification_timestamps_per_station()
            self.logger.info(
                f"Found the following stations in the database: {list(self._time_periods_per_station.keys())}")
            self.logger.debug("Register the following modification periods:")
            for key in self._time_periods_per_station:
                mods = self._time_periods_per_station[key]["modification_timestamps"]
                self.logger.debug(f"{key}: {mods}")

            # Used to keep track which time period is buffered. Index of 0, not buffered jet.
            self._time_period_index_per_station = collections.defaultdict(int)

            # This should be set with Detector.update(..) and corresponds to the time of a measurement. It will be use to
            # decide which components are commissioned at the time of the measurement
            self.__detector_time = None

            # Initialise the primary buffer
            self.__buffered_stations = collections.defaultdict(dict)

            self._query_all = always_query_entire_description
        else:
            self._query_all = None  # specific case for file imported detector descriptions
            self._det_imported_from_file = True

            import_dir = pickle.load(open(pickle_file, "rb"))
            if "version" in import_dir and import_dir["version"] == 1:
                self.__buffered_stations = import_dir["data"]
                self._time_periods_per_station = import_dir["periods"]
                self._time_period_index_per_station = {
                    st_id: 1 for st_id in self.__buffered_stations}
                self.__default_values = import_dir["default_values"]
            else:
                self.logger.error(f"{pickle_file} with unknown version.")
                raise ReferenceError(f"{pickle_file} with unknown version.")

        # Allow overwriting the hard-coded values
        self.__default_values.update(over_write_handset_values)

        info = f"Query entire detector description at once: {self._query_all}"

        info += "\nUsing the following hand-set values:"
        for key, value in self.__default_values.items():
            info += f"\n\t{key:<20}: {value}"

        self.logger.info(info)

    def export(self, filename):
        """
        Export the buffered detector description.

        Parameters
        ----------

        filename: str
            Filename of the exported detector description
        """

        periods = {}
        for station_id in self.__buffered_stations:

            # Remove decommissioned stations from the buffer completely
            if self.__buffered_stations[station_id] == {}:
                self.__buffered_stations.pop(station_id)
                continue

            idx = self._time_period_index_per_station[station_id]
            if idx == 0 or idx == len(self._time_periods_per_station[station_id]["modification_timestamps"]):
                self.logger.error("You try to export a decomissioned station")

            periods[station_id] = {"modification_timestamps":
                                   [self._time_periods_per_station[station_id]["modification_timestamps"][idx - 1],
                                    self._time_periods_per_station[station_id]["modification_timestamps"][idx]]
                                   }

        export_dict = {
            "version": 1,
            "data": self.__buffered_stations,
            "periods": periods,
            "default_values": self.__default_values
        }

        pickle.dump(export_dict, open(filename, "wb"))

    def _check_update_buffer(self):
        """
        Checks whether the correct detector description per station in in the current period.

        Returns
        -------

        need_update: dict(bool)
            Flag for each station if an update is needed
        """
        need_update = collections.defaultdict(bool)

        for station_id in self._time_periods_per_station:
            period = np.digitize(self.get_detector_time().timestamp(),
                                 [dt.timestamp() for dt in self._time_periods_per_station[station_id]["modification_timestamps"]])

            if period != self._time_period_index_per_station[station_id]:
                need_update[station_id] = True
            else:
                need_update[station_id] = False

            self._time_period_index_per_station[station_id] = period

        debug_str = "The following stations need to be updated:"
        for station_id in self._time_period_index_per_station:
            debug_str += f"\n\tStation {station_id} : {need_update[station_id]} (period: {self._time_period_index_per_station[station_id]})"

        self.logger.debug(debug_str)

        return need_update

    def __set_detector_time(self, time):
        ''' Set time of detector. This controls which stations/channels are commissioned.

        Only for internal use. Use detector.update() to set detector time from the outside.

        Parameters
        ----------

        time: datetime.datetime 
            UTC time.
        '''
        if not isinstance(time, datetime.datetime):
            self.logger.error(
                "Set invalid time for detector. Time has to be of type datetime.datetime")
            raise TypeError(
                "Set invalid time for detector. Time has to be of type datetime.datetime")
        self.__detector_time = time

    def get_detector_time(self):
        """
        Returns
        -------

        time: datetime.datetime
            Detector time
        """
        if self.__detector_time is None:
            raise ValueError(
                "Detector time is None. Please call detector.update(time) before.")
        return self.__detector_time

    def update(self, time):
        """ 
        Updates the detector. If configure in constructor this function with trigger the 
        database query.    

        Parameters
        ----------

        time: datetime.datetime 
            Unix time of measurement.        
        """
        self.logger.debug(f"Update detector to {time}")

        self.__set_detector_time(time)
        if not self._det_imported_from_file:
            self.__db.set_detector_time(time)

        update_buffer_for_station = self._check_update_buffer()
        any_update = np.any([v for v in update_buffer_for_station.values()])

        if self._det_imported_from_file and any_update:
            self.logger.error(
                "You have imported the detector description from a pickle file but it is not valid anymore. Full stop!")
            raise ValueError(
                "You have imported the detector description from a pickle file but it is not valid anymore. Full stop!")

        if any_update:
            for key in self.__buffered_stations:
                if update_buffer_for_station[station_id]:
                    # remove everything (could be handled smarter ...)
                    self.__buffered_stations[station_id] = {}

            for station_id, need_update in update_buffer_for_station.items():
                if need_update and self.has_station(station_id):
                    self._query_station_information(
                        station_id, all=self._query_all)

    @check_detector_time
    def get_station_ids(self):
        """
        Returns the list of all commissioned stations. 
        Returns
        -------

        station_ids: list(int)
            List of all commissioned station ids.
        """
        commissioned_stations = []

        for station_id, station_data in self._time_periods_per_station.items():
            for comm, decomm in zip(station_data["station_commission_timestamps"],
                                    station_data["station_decommission_timestamps"]):

                if comm < self.get_detector_time() and self.get_detector_time() < decomm:
                    commissioned_stations.append(station_id)

        return commissioned_stations
    
    @check_detector_time
    def has_station(self, station_id):
        """
        Returns true if the station is commission. First checks buffer. If not in buffer, queries (and buffers)
        the basic information of all stations and checks if station is among them. 

        Parameters
        ----------

        station_id: int
            Station id which uniquely idendifies a station

        Returns
        -------

        has_station: bool
            Returns True if the station could be found. A found station will be in the buffer.
        """
        # Check if station is in buffer ...
        self.logger.debug(
            f"Checking for station {station_id} at {self.get_detector_time()} ...")

        if station_id not in self._time_periods_per_station:
            self.logger.debug(f"Station {station_id} not found in database.")
            return False

        station_data = self._time_periods_per_station[station_id]
        for comm, decomm in zip(station_data["station_commission_timestamps"],
                                station_data["station_decommission_timestamps"]):

            if comm < self.get_detector_time() and self.get_detector_time() < decomm:
                self.logger.debug(f"Station {station_id} is commissioned!")
                return True

        self.logger.debug(f"Station {station_id} not commissioned!")
        return False

    def _query_station_information(self, station_id, all=True):
        """ 
        Query information about a specific station from the database via the db_mongo_read interface.
        You can query only information from the station_list collection (all=False) or the complete 
        information of the station (all=True).

        Parameters
        ----------

        station_id: int
            Station id

        all: bool
            If true, query all relevant information form a station including its channel and devices (position, signal chain, ...).
            If false, query only the information from the station list collection (describes a station with all channels and devices 
            with their (de)commissioning timestamps but not data like position, signal chain, ...)

        Returns
        -------
        None
        """

        if station_id in self.__buffered_stations and self.__buffered_stations[station_id] != {}:
            raise ValueError(
                f"Query information for station {station_id} which is still in buffer.")

        self.logger.info(
            f"Query information for station {station_id} at {self.get_detector_time()}")
        if all:
            station_information = self.__db.get_complete_station_information(
                station_id)
        else:
            station_information = self.__db.get_general_station_information(
                station_id)

        if len(station_information) != 1:
            raise ValueError(f"Could not query information of station {station_id} at {self.get_detector_time()}. "
                             f"Found {len(station_information)} entries in database.")

        self.__buffered_stations[station_id] = station_information[station_id]

        # time_filter = [{"$match": {
        #     'id': station_id,
        #     'commission_time': {"$lte": self.get_detector_time()},
        #     'decommission_time': {"$gte": self.get_detector_time()}}}]

        # # get all stations which fit the filter
        # station_information = list(self.__db.db["station_rnog"].aggregate(time_filter))

        # if len(station_information) != 1:
        #     raise ValueError(f"Could not query information of station {station_id} at {self.get_detector_time()}. "
        #                      f"Found {len(station_information)} entries in database.")

        # station_information[0].pop("id")
        # self.__buffered_stations[station_id] = station_information[0]

    # def get_full_station_information(self, station_id, update=False):
    #     """

    #     Parameters
    #     ----------

    #     station_id: int
    #         Station id

    #     update: bool
    #         Force to query full station information. (Default: False)

    #     Returns:

    #     full_info: dict
    #         Get dictionary of _all_ information of a commissioned station including all channel and device infomration
    #         such as the S-parameter of the amplifier
    #     """

    #     if station_id not in self.__buffered_stations or self.__buffered_stations[station_id] == {} or update:
    #         # query all information
    #         self.logger.debug(f"Query full information of station {station_id} at {self.get_detector_time()}")
    #         self.__buffered_stations[station_id] = self.__db.get_complete_station_information(station_id=station_id)
    #     else:
    #         info = self.__buffered_stations[station_id]
    #         # This should ne be necessary because we clean the buffer in Detector.update(...).
    #         # However, I keep it here because "twice holds longer" and in future we might implement a more smart way of updateing
    #         if info["commission_time"] > self.get_detector_time() or info["decommission_time"] < self.get_detector_time():
    #             self.__buffered_stations[station_id] = self.__db.get_complete_station_information(station_id=station_id)

    #     return self.__buffered_stations[station_id]

    def __get_channel(self, station_id, channel_id, with_position=False, with_signal_chain=False):
        """
        Get most basic channel information from buffer. If not in buffer query DB for all channels. 
        In particular this will query and buffer position information  

        Parameters
        ----------

        station_id: int
            The station id

        channel_id: int
            The channel id

        with_position: bool
            If True, check if channel position is available and if not query. (Default: False)

        with_signal_chain: bool
            If True, check if channel signal chain is available and if not query. (Default: False)

        Returns
        -------

        channel_info: dict
            Dict of channel information
        """
        if keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id]):
            raise KeyError(
                f"Could not find channel {channel_id} in detector description for station {station_id}.")

        if with_position and keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id, "channel_position"]):

            if keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id, "id_position"]):
                raise KeyError(
                    f"\"id_position\" not in buffer for station.channel {station_id}.{channel_id}. Did you call det.update(..)")

            channel_position_id = self.__buffered_stations[
                station_id]["channels"][channel_id]["id_position"]
            self.logger.debug(
                f"Query position of station.channel {station_id}.{channel_id} with id {channel_position_id}")

            channel_position_dict = self.__db.get_channel_position(
                channel_position_id=channel_position_id)
            self.__buffered_stations[station_id]["channels"][channel_id]['channel_position'] = channel_position_dict

        if with_signal_chain and keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id, "signal_chain"]):

            if keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id, "id_signal"]):
                raise KeyError(
                    f"\"id_signal\" not in buffer for station.channel {station_id}.{channel_id}. Did you call det.update(..)")

            signal_id = self.__buffered_stations[station_id]["channels"][channel_id]['id_signal']
            channel_sig_info = self.__db.get_channel_signal_chain(signal_id)
            channel_sig_info.pop('channel_id', None)

            self.__buffered_stations[station_id]["channels"][channel_id]['signal_chain'] = channel_sig_info

        return self.__buffered_stations[station_id]["channels"][channel_id]

    @check_detector_time
    def get_absolute_position(self, station_id):
        """
        Get the absolute position of a specific station (relative to site)

        Parameters
        ----------

        station_id: int
            the station id

        Returns
        -------

        pos: np.array(3,) 
            3-dim array of absolute station position in easting, northing and depth wrt. to snow level at
            time of measurement
        """
        self.logger.debug(
            f"Requesting station position for station {station_id}.")

        if keys_not_in_dict(self.__buffered_stations, [station_id, "station_position"]):

            if keys_not_in_dict(self.__buffered_stations, [station_id, "id_position"]):
                raise KeyError(
                    f"\"id_position\" not in buffer for station {station_id}. Did you call det.update(..)")

            station_position_id = self.__buffered_stations[station_id]["id_position"]
            self.logger.debug(
                f"Query position for id \"{station_position_id}\"")

            station_position = self.__db.get_station_position(
                station_position_id=station_position_id)
            self.__buffered_stations[station_id]["station_position"] = station_position

        return np.array(self.__buffered_stations[station_id]["station_position"]["position"])

    @check_detector_time
    def get_relative_position(self, station_id, channel_id):
        """
        Get the relative position of a specific channel/antenna with respect to the station center

        Parameters
        ----------

        station_id: int
            The station id

        channel_id: int
            The channel id

        Returns
        -------

        pos: np.array(3,)
            3-dim array of relative station position
        """
        channel_info = self.__get_channel(
            station_id, channel_id, with_position=True)
        return np.array(channel_info["channel_position"]['position'])

    @check_detector_time
    def get_channel_orientation(self, station_id, channel_id):
        """
        Returns the orientation of a specific channel/antenna

        Orientation:
            - theta: For LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
            - phi: Counting from East counterclockwise; for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry

        Rotation:
            - theta: Is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the tines
            - phi: Is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the tines

        Parameters
        ----------

        station_id: int
            The station id

        channel_id: int
            The channel id

        Returns
        -------
        orientation: array of floats
            (orientation_theta, orientation_phi, rotation_theta, rotation_phi): tuble of floats
        """

        channel_info = self.__get_channel(
            station_id, channel_id, with_position=True)
        orientation = channel_info['channel_position']["orientation"]
        rotation = channel_info['channel_position']["rotation"]

        return np.deg2rad([orientation["theta"], orientation["phi"], rotation["theta"], rotation["phi"]])

    def get_antenna_orientation(self, station_id, channel_id):
        """ Returns get_channel_orientation """
        return self.get_channel_orientation(station_id, channel_id)

    @check_detector_time
    def get_channel_signal_chain(self, station_id, channel_id):
        """

        Parameters
        ----------

        station_id: int
            The station id

        channel_id: int
            The channel id

        Returns
        -------

        channel_signal_chain: dict
            Returns dictionary which contains a list ("signal_chain") which contains 
            (the names of) components/response which are used to describe the signal
            chain of the channel
        """
        channel_info = self.__get_channel(
            station_id, channel_id, with_signal_chain=True)
        return self.__buffered_stations[station_id]["channels"][channel_id]["signal_chain"]

    @check_detector_time
    def get_signal_chain_response(self, station_id, channel_id):
        """

        Parameters
        ----------

        station_id: int
            The station id

        channel_id: int
            The channel id

        Returns
        -------

        response: rnog_detector.Response
            Returns combined response of the channel
        """
        signal_chain_dict = self.get_channel_signal_chain(
            station_id, channel_id)

        measurement_components_dic = signal_chain_dict["response_chain"]
        
        # Here comes a HACK
        components = list(measurement_components_dic.keys())
        is_equal = False
        if "drab_board" in components and "iglu_board" in components:
            
            is_equal = np.allclose(measurement_components_dic["drab_board"]["mag"],
                                   measurement_components_dic["iglu_board"]["mag"])
            
            if is_equal:
                self.logger.warn("Currently both, iglu and drab board are configured in the signal chain but their"
                                 " responses are the same (because we measure them together in the lab). Drop the drab board response.")

        responses = []
        for key, value in measurement_components_dic.items():
            if is_equal and key == "drab_board":
                continue
            
            ydata = [value["mag"], value["phase"]]
            responses.append(
                Response(value["frequencies"], ydata, value["y-axis_units"], name=key))

        return np.prod(responses)

    @check_detector_time
    def get_devices(self, station_id):
        """
        Get all devices for a particular station.

        Parameters
        ----------

        station_id: int
            Station id 

        Returns
        -------

        devices: dict(str)
            Dictonary of all devices with {id: name}.
        """

        if not self.has_station(station_id):
            self.logger.error(
                f"Station {station_id} not commissioned at {self.get_detector_time()}. Return empty device list")
            return []

        return {device["id"]: device["device_name"] for device in self.__buffered_stations[station_id]["devices"].values()}

    @check_detector_time
    def get_relative_position_device(self, station_id, device_id):
        """
        Get the relative position of a specific device with respect to the station center

        Parameters
        ----------

        station_id: int
            The station id

        device_id: int
            Device identifier

        Returns
        -------

        pos: np.array(3,)
            3-dim array of relative station position
        """
        if keys_not_in_dict(self.__buffered_stations, [station_id, "devices", device_id]):
            # All devices should have been queried with _query_station_information
            raise KeyError(f"Device {device_id} not in detector description.")

        if keys_not_in_dict(self.__buffered_stations, [station_id, "devices", device_id, "device_position"]):

            if keys_not_in_dict(self.__buffered_stations, [station_id, "devices", device_id, "id_position"]):
                raise KeyError(
                    f"\"id_position\" not in buffer for device {device_id}.")

            position_id = self.__buffered_stations[station_id]["devices"][device_id]["id_position"]

            device_pos_info = self.__db.get_device_position(
                device_position_id=position_id)
            self.__buffered_stations[station_id]["devices"][device_id]['device_position'] = device_pos_info

        return np.array(self.__buffered_stations[station_id]["devices"][device_id]["device_position"]["position"])

    # def _has_valid_parameter_in_buffer(self, key_list):
    #     """
    #     This function first checks if a parameter which is specified with a key_list (e.g. [station_id, "channels", channel_id, "channel_position"])
    #     exists in the buffer and second if it is still valid base on detector time (i.e., if corresponding station and channel are still commission).
    #     Remove station or channel from buffer if not commission.

    #     Parameters
    #     ----------

    #     key_list: list
    #         Specifies a parameter (see example above)

    #     Returns
    #     -------

    #     has_parameter:  bool
    #         True if key_list exists and is valid
    #     """

    #     if keys_not_in_dict(self.__buffered_stations, key_list):
    #         self.logger.debug("Parameter not in buffer: " + " / ".join([str(x) for x in key_list]))
    #         return False

    #     station_info = self.__buffered_stations[key_list[0]]

    #     if station_info["commission_time"] > self.get_detector_time() or station_info["decommission_time"] < self.get_detector_time():
    #         self.logger.debug(f"Station {key_list[0]} not commission anymore at {self.get_detector_time()}. Remove from buffer ...")
    #         self.__buffered_stations[key_list[0]] = {}  # clean buffer
    #         return False

    #     if key_list[1] == "channels" and len(key_list) >= 3:
    #         channel_info = station_info["channels"][key_list[2]]
    #         if station_info["commission_time"] > self.get_detector_time() or station_info["decommission_time"] < self.get_detector_time():
    #             self.logger.debug(f"Channel {key_list[2]} (of Station {key_list[0]}) not commission anymore at {self.get_detector_time()}. Remove from buffer ...")
    #             self.__buffered_stations[key_list[0]]["channels"][key_list[2]] = {}  # clean buffer
    #             return False

    #     self.logger.debug("Parameter in buffer and valid: " + " / ".join([str(x) for x in key_list]))
    #     return True

    @check_detector_time
    def get_number_of_channels(self, station_id):
        """
        Get number of channels for a particlular station. It will query the basic information of all stations in the 
        Database if necessary. Raises an error if the station is not commission. 

        Parameters
        ----------

        station_id: int
            Station id which uniquely idendifies a station

        Returns
        -------

        channel_ids: int
            Number of all channels for the requested station
        """

        if not self.has_station(station_id):
            err = f"Station id {station_id} not commission at {self.get_detector_time()}"
            self.logger.error(err)
            raise ValueError(err)

        # now station is in buffer (or an error was raised)
        channels = self.__buffered_stations[station_id]["channels"]

        return len(channels)

    @check_detector_time
    def get_channel_ids(self, station_id):
        """
        Get channel ids for a particlular station. It will query the basic information of all stations in the 
        Database if necessary. Raises an error if the station is not commission. 

        Parameters
        ----------

        station_id: int
            Station id which uniquely idendifies a station

        Returns
        -------

        channel_ids: list(int)
            A list of all channel ids for the requested station
        """

        if not self.has_station(station_id):
            err = f"Station id {station_id} not commission at {self.get_detector_time()}"
            self.logger.error(err)
            raise ValueError(err)

        # now station is in buffer (or an error was raised)
        channels = self.__buffered_stations[station_id]["channels"]

        return [ele["id"] for ele in channels.values()]
    
    def get_antenna_model(self, station_id, channel_id, zenith=None):
        """

        Parameters
        ----------
        station_id: int
            Station id
            
        channel_id: int
            Channel id

        Returns
        -------
        
        antenna_model: string
            Name of the antenna model (describing the Vector effective length VEL)
        """
        channel_info = self.__get_channel(
            station_id, channel_id, with_signal_chain=True)
        return channel_info["signal_chain"]["VEL"]
        

    def get_number_of_samples(self, station_id, channel_id):
        """ Get number of samples per station / channel """
        number_of_samples = self.__default_values["number_of_samples"]
        self.logger.warn(
            f"Return a hard-coded value of {number_of_samples} samples. This information is not (yet) implemented in the DB.")
        return number_of_samples

    def get_sampling_frequency(self, station_id, channel_id):
        """ Get sampling frequency per station / channel """
        sampling_frequency = self.__default_values["sampling_frequency"]
        self.logger.warn(
            f"Return a hard-coded value for the sampling frequency of {sampling_frequency / units.GHz} GHz. "
            "This information is not (yet) implemented in the DB.")
        return sampling_frequency

    def get_noise_temperature(self, station_id, channel_id):
        """ Get noise temperture per station / channel """
        noise_temperature = self.__default_values["noise_temperature"]
        self.logger.warn(
            f"Return a hard-coded value for the noise temperature of {noise_temperature / units.kelvin} K. "
            "This information is not (yet) implemented in the DB.")
        return noise_temperature

    def is_channel_noiseless(self, station_id, channel_id):
        is_noiseless = self.__default_values["is_noiseless"]
        self.logger.warn(
            f"Return a hard-coded value for \"is_noiseless\" of {is_noiseless}. "
            "This information is not (yet) implemented in the DB.")
        return is_noiseless

    def get_cable_delay(self, station_id, channel_id):
        cable_delay = self.__default_values["cable_delay"]
        if isinstance(cable_delay, float):    
            self.logger.warn(
                f"Return a hard-coded value for the cable delay of all channels of {cable_delay} ns. "
                "This information is not (yet) implemented in the DB.")
            return cable_delay
        elif isinstance(cable_delay, dict):
            self.logger.warn(
                f"Return a hard-coded value for the cable delay for channel {channel_id} of {cable_delay[channel_id]} ns. "
                "This information is not (yet) implemented in the DB.")
            return cable_delay[channel_id]
            
    def get_site(self, station_id):
        """
        This detector class is exclusive for the RNO-G detector at Summit Greenland.

        Returns
        -------

        site: str
            Returns "summit"
        """
        return "summit"

    def db(self):
        """ Temp. interface for development """
        return self.__db

    def set_detector_time(self, time):
        """ Temp. interface for development """
        self.__set_detector_time(time)


class Response:
    """
    This class provides an interface to read-in and apply the complex response functions of the
    various components of the signal chain of a RNO-G channel. The class allows to combine the 
    response of several components into one response by multiplying them.
    """

    def __init__(self, frequency, y, y_unit, name="default"):
        """
        Parameters
        ----------

        frequency: list(float)
            The frequency vector at which the response is measured. Unit has to be GHz.

        y: [list(float), list(float)]
            The measured response. First entry is the vector of the measured amplitude, the second entry is the measured phase.
            The unit of both entries is specified with the next argument.

        y_unit: [str, str]
            The first entry specifies the unit of the measured amplitude. Options are "dB", "MAG" and "mag".
            The second entry specifies the unit of the measured phase. Options are "rad" and "deg".

        name: str
            Give the response a name. This is only use for printing purposes. (Default: "default")        
        """

        self.__names = [name]

        self.__frequency = np.array(frequency) * units.GHz

        if y[0] is None or y[1] is None:
            raise ValueError

        y_ampl = np.array(y[0])
        y_phase = np.array(y[1])

        if y_unit[0] == "dB":
            gain = 10 ** (y_ampl / 20)
        elif y_unit[0] == "mag" or y_unit[0] == "MAG":
            gain = y_ampl
        else:
            raise KeyError

        if y_unit[1] == "deg":
            y_phase = np.deg2rad(y_phase)
        elif y_unit[1] == "rad":
            y_phase = y_phase
        else:
            raise KeyError

        self.__gains = [interpolate.interp1d(
            self.__frequency, gain, kind="linear", bounds_error=False, fill_value=0)]

        self.__phases = [interpolate.interp1d(
            self.__frequency, y_phase, kind="linear", bounds_error=False, fill_value=0)]

    def __call__(self, freq):
        """
        Returns the complex response for a given frequency.

        Parameters
        ----------

        freq: list(float)
            The frequencies for which to get the response.

        Returns

        response: np.array(np.complex128)
            The complex response at the desired frequencies
        """
        response = np.ones_like(freq, dtype=np.complex128)
        for gain, phase in zip(self.__gains, self.__phases):
            response *= gain(freq / units.GHz) * \
                np.exp(1j * phase(freq / units.GHz))

        return response

    def get_names(self):
        """ Get list of the names of all individual responses """
        return self.__names

    def __mul__(self, other):
        """ 
        Define multiplication operator for 
            - Other objects of the same class
            - Objects of type NuRadioReco.framework.base_trace

        TODO: Multiplication with base_trace will just work when the spectrum is one dimensional. Check if
        there is a case where the spectrum could be 3 (or more) dimensional and adjust the class

        """

        if isinstance(other, Response):
            # Store each response individually: append/concatenate lists of gains and phases.
            # The multiplication happens in __call__.
            self.__names += other.__names
            self.__gains += other.__gains
            self.__phases += other.__phases
            return self

        elif isinstance(other, NuRadioReco.framework.base_trace):
            spec = other.get_frequency_spectrum()
            freqs = other.get_frequencies()
            spec *= self(freqs)  # __call__
            return other

        else:
            raise TypeError

    def __rmul__(self, other):
        """ Same as mul """
        return self.__mul__(other)

    def __str__(self):
        return "Response of " + ", ".join(self.get_names()) + f": R(0.5 GHz) = {self(0.5 * units.GHz)}"


if __name__ == "__main__":
    det = Detector(log_level=logging.DEBUG, over_write_handset_values={
                   "sampling_frequency": 2.4 * units.GHz}, always_query_entire_description=False)

    det.update(datetime.datetime(2022, 8, 2, 0, 0))
    det.get_antenna_model(11, 0)
