from NuRadioReco.detector.RNO_G.db_mongo_read import Database, _convert_astro_time_to_datetime
from NuRadioReco.detector.response import Response
from NuRadioReco.utilities import units

from functools import wraps

import astropy.time
import datetime
import numpy as np
import collections
import json
import re
import copy
import bson
import lzma
import logging


def _json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, bson.objectid.ObjectId):
        return str(obj)
    elif isinstance(obj, Response):
        # Objects of type Response are not serializable
        pass
    else:
        raise TypeError ("Type %s not serializable" % type(obj))


def _keys_not_in_dict(d, keys):
    """ Checks sequentially if a list of `keys` is in a dictionary.

    Example:
    keys = ["key1", "key2"]

    Returns False if d["key1"]["key2"] exsits, True otherwise.
    """
    if isinstance(keys, str):
        keys = [keys]

    d_tmp = d
    for key in keys:
        try:
            if key not in d_tmp:
                return True
        except KeyError:
            return True

        d_tmp = d_tmp[key]

    return False


def _check_detector_time(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        self.get_detector_time()  # this will raise an error if time is not set
        return method(self, *method_args, **method_kwargs)
    return _impl


class Detector():
    def __init__(self, database_connection='RNOG_public', log_level=logging.NOTSET, over_write_handset_values=None,
                 database_time=None, always_query_entire_description=False, detector_file=None,
                 select_stations=None, create_new=False):
        """
        The RNO-G detector description.

        Parameters
        ----------
        database_connection : str (Default: 'RNOG_public')
            Allows to specify database connection. Passed to mongo-db interface.

        log_level : `logging.LOG_LEVEL` (Default: logging.NOTSET)
            Overrides verbosity level of logger. Propagates through to Response class logger.
            Other options are: `logging.WARNING`, `logging.DEBUG`, ...

        over_write_handset_values : dict (Default: {})
            Overwrite the default values for (channel) parameters which are not (yet) implemented in the database.
            You can not specify keys which already exist in the database. If the value is a dict it should be contain
            a value for each channel_id (key). (Default: None, the acutally default values for the parameters in question
            are defined below)

        database_time : `datetime.datetime` or `astropy.time.Time`
            Set database time which is used to select the primary measurement. By default (= None) the database time
            is set to now (time the code is running) to select the measurement which is now primary.

        always_query_entire_description : bool (Default: False)
            If True, query the entire detector describtion all at once when calling Detector.update(...) (if necessary).
            This value is currently set to ``False`` by default to avoid errors due to missing information in the database.

        detector_file : str
            File to import detector description instead of querying from DB. (Default: None -> query from DB)

        select_stations : int or list(int) (Default: None)
            Select a station or list of stations using their station ids for which the describtion is provided.
            This is useful for example in simulations when one wants to simulate only one station. The default None
            means to descibe all commissioned stations.

        create_new : bool (Default: False)
            If False, and a database already exists, the existing database will be used rather than initializing a
            new connection. Set to True to create a new database connection.

        Notes
        -----
        For more information about ``Detector`` objects in NuRadioMC, see
        https://nu-radio.github.io/NuRadioMC/NuRadioReco/pages/detector_tree.html

        """

        self.logger = logging.getLogger("NuRadioReco.RNOGDetector")
        self.__log_level = log_level
        self.logger.setLevel(self.__log_level)

        # Define default values for parameter not (yet) implemented in DB.
        self.__default_values = {
            "noise_temperature": 300 * units.kelvin,
            "is_noiseless": False,
        }

        self.additional_data = {}
        self.comment = ""

        if select_stations is not None and not isinstance(select_stations, list):
            select_stations = [select_stations]

        self.selected_stations = select_stations
        self.logger.info(f"Select the following stations (if possible): {select_stations}")
        self.__db = None
        if detector_file is None:
            self._det_imported_from_file = False

            self.__db = Database(database_connection=database_connection, create_new=create_new)
            if database_time is not None:
                self.__db.set_database_time(database_time)

            self.logger.info(
                "Collect time periods of station commission/decommission ...")
            self._time_periods_per_station = self.__db.query_modification_timestamps_per_station()
            self.logger.info(
                f"Found the following stations in the database: {list(self._time_periods_per_station.keys())}")

            if self.selected_stations is not None:
                # Filter stations: Only consider selected stations
                self._time_periods_per_station = {
                    station_id: value for station_id, value in self._time_periods_per_station.items()
                    if station_id in self.selected_stations
                }

            self.logger.debug("Register the following modification periods:")
            for key, value in self._time_periods_per_station.items():
                self.logger.debug(f'{key}: {value}["modification_timestamps"]')

            # Used to keep track which time period is buffered. Index of 0, not buffered jet.
            self._time_period_index_per_station = collections.defaultdict(int)

            # Initialise the primary buffer
            self.__buffered_stations = collections.defaultdict(dict)

            self._query_all = always_query_entire_description
        else:
            self._query_all = None  # specific case for file imported detector descriptions
            self._det_imported_from_file = True
            self._import_from_file(detector_file)

        # This should be set with Detector.update(..) and corresponds to the time of a measurement.
        # It will be use to decide which components are commissioned at the time of the measurement
        self.__detector_time = None

        # Allow overwriting the hard-coded values
        over_write_handset_values = over_write_handset_values or {}
        self.__default_values.update(over_write_handset_values)

        info = f"Query entire detector description at once: {self._query_all}"

        info += "\nUsing the following hand-set values:"
        n = np.amax([len(key) for key in self.__default_values.keys()]) + 3
        for key, value in self.__default_values.items():
            info += f"\n\t{key:<{n}}: {value}"

        self.logger.info(info)

        self.assume_inf = None  # Compatibility with other detectors classes
        self.antenna_by_depth = None  # Compatibility with other detectors classes

    def export(self, filename, json_kwargs=None, additional_data=None, drop_response_data=False, comment=None):
        """
        Export the buffered detector description.

        Parameters
        ----------
        filename: str
            Filename of the exported detector description

        json_kwargs: dict
            Arguments passed to json.dumps(..). (Default: None -> dict(indent=0, default=_json_serial))

        additional_data: dict (Default: None)
            If specified the content of this dict will be added to the exported detector description.

        drop_response_data: bool (Default: False)
            If True, the response data (frequency, mag, phase) will be dropped from the exported detector description.

        comment: str (Default: None)
            An optional comment describing this detector that will be added to the exported detector description.
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

            periods[station_id] = {
                "modification_timestamps":
                    [self._time_periods_per_station[station_id]["modification_timestamps"][idx - 1],
                    self._time_periods_per_station[station_id]["modification_timestamps"][idx]]
            }

        if drop_response_data:
            buffered_stations = copy.deepcopy(self.__buffered_stations)
            keys_to_drop = ['frequencies', 'mag', 'phase']
            for chain in ["response_chain", "trigger_response_chain"]:
                for station_id in buffered_stations:
                    for channel_id, channel_data in buffered_stations[station_id]["channels"].items():
                        if chain in channel_data['signal_chain']:
                            for names, component in channel_data['signal_chain'][chain].items():
                                for key in keys_to_drop:
                                    component.pop(key, None)
        else:
            buffered_stations = self.__buffered_stations

        export_dict = {
            "version": 1,
            "data": buffered_stations,
            "periods": periods,
            "default_values": self.__default_values
        }

        if additional_data is not None:
            export_dict["additional_data"] = additional_data

        if comment is not None:
            self.comment = "\n".join([self.comment, comment]).strip()
            
        export_dict["comment"] = self.comment

        if not filename.endswith(".xz"):
            if not filename.endswith(".json"):
                filename += ".json"
            filename += ".xz"
        elif not filename.endswith(".json.xz"):
            filename = filename.replace(".xz", ".json.xz")

        if json_kwargs is None:
            json_kwargs = dict(indent=0, default=_json_serial)

        self.logger.info(f"Export detector description to {filename}")
        with lzma.open(filename, "w") as f:
            f.write(json.dumps(export_dict, **json_kwargs).encode('utf-8'))


    def export_as_string(self, skip_signal_chain_response=True, dumps_kwargs=None):
        """
        Export the detector description as string using json.dumps

        Parameters
        ----------
        skip_signal_chain_response: bool
            If true drop the data of the response chain from the detector description (because this creates large files).
            (Default: True)

        dumps_kwargs: dict
            Arguments passed to json.dumps(..). (Default: None -> dict(indent=4, default=str))
        """
        export_dir = copy.deepcopy(self.__buffered_stations)

        if skip_signal_chain_response:
            for station_id in export_dir:
                for channel_id in export_dir[station_id]["channels"]:
                    for key in ["response_chain", "total_response", "trigger_response_chain", "total_trigger_response"]:
                        export_dir[station_id]["channels"][channel_id].pop(key, None)

        if dumps_kwargs is None:
            dumps_kwargs = dict(indent=4, default=str)

        return json.dumps(export_dir, **dumps_kwargs)

    def _import_from_file(self, detector_file):

        if detector_file.endswith(".json.xz"):
            with lzma.open(detector_file, "r") as f:
                import_dict = json.load(f)
        elif detector_file.endswith(".json"):
            with open(detector_file, "r") as f:
                import_dict = json.load(f)
        else:
            raise ValueError(f"Unknown filetype (extension), can not import detector from {detector_file}. "
                                "Allowed are only: \".json\" and \".json.xz\"")

        if "version" in import_dict and import_dict["version"] == 1:
            self.__buffered_stations = {}
            self.additional_data = import_dict.get("additional_data", None)
            self.comment = import_dict.get("comment", None)

            # need to convert station/channel/device id keys back to integers
            for station_id, station_data in import_dict["data"].items():
                if self.selected_stations is not None and int(station_id) not in self.selected_stations:
                    continue

                station_data["channels"] = {int(channel_id): channel_data for channel_id, channel_data in station_data["channels"].items()}
                station_data["devices"] = {int(device_id): device_data for device_id, device_data in station_data["devices"].items()}
                self.__buffered_stations[int(station_id)] = station_data

            # need to convert modification_timestamps back to datetime objects
            self._time_periods_per_station = {
                int(station_id): {"modification_timestamps":
                    [datetime.datetime.fromisoformat(v) for v in value["modification_timestamps"]]}
                for station_id, value in import_dict["periods"].items() if self.selected_stations is None
                or int(station_id) in self.selected_stations
            }

            # Set de/commission timestamps to the time period for which this config is valid
            for station_id in self._time_periods_per_station:
                modification_timestamps = self._time_periods_per_station[station_id]["modification_timestamps"]
                self._time_periods_per_station[station_id]["station_commission_timestamps"] = [modification_timestamps[0]]
                self._time_periods_per_station[station_id]["station_decommission_timestamps"] = [modification_timestamps[-1]]

            self._time_period_index_per_station = {
                st_id: 1 for st_id in self.__buffered_stations}
            self.__default_values = import_dict["default_values"]
        else:
            self.logger.error(f"{detector_file} with unknown version.")
            raise ReferenceError(f"{detector_file} with unknown version.")

        # print any potential comment present in this detector description
        if self.comment is not None:
            self.logger.info("\n".join(["Loaded detector description with comment:", self.comment]))

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
                                 [dt.timestamp() for dt in
                                  self._time_periods_per_station[station_id]["modification_timestamps"]])

            if period != self._time_period_index_per_station[station_id]:
                need_update[station_id] = True
            else:
                need_update[station_id] = False

            self._time_period_index_per_station[station_id] = period

        debug_str = "The following stations need to be updated:"
        for station_id in self._time_period_index_per_station:
            debug_str += (f"\n\tStation {station_id} : {need_update[station_id]} "
                          f"(period: {self._time_period_index_per_station[station_id]})")

        self.logger.debug(debug_str)

        return need_update

    def __set_detector_time(self, time):
        ''' Set time of detector. This controls which stations/channels are commissioned.

        Only for internal use. Use detector.update() to set detector time from the outside.

        Parameters
        ----------
        time: `datetime.datetime` or ``astropy.time.Time``
            UTC time.
        '''

        if isinstance(time, astropy.time.Time):
            time = _convert_astro_time_to_datetime(time)
        elif not isinstance(time, datetime.datetime):
            self.logger.error(
                "Set invalid time for detector. Time has to be of type `datetime.datetime`")
            raise TypeError(
                "Set invalid time for detector. Time has to be of type `datetime.datetime`")
        self.__detector_time = time

    def get_detector_time(self):
        """
        Returns
        -------

        time: `datetime.datetime`
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
        time: `datetime.datetime` or ``astropy.time.Time``
            Unix time of measurement.
        """
        if isinstance(time, astropy.time.Time):
            time = _convert_astro_time_to_datetime(time)

        if self.__detector_time is None:
            self.logger.info(f"Update detector to {time}")

        self.__set_detector_time(time)
        if not self._det_imported_from_file:
            self.__db.set_detector_time(time)

        update_buffer_for_station = self._check_update_buffer()
        any_update = np.any([v for v in update_buffer_for_station.values()])

        if self._det_imported_from_file and any_update:
            self.logger.warning(f"Update detector to {time}")
            self.logger.error(
                "You have imported the detector description from a pickle/json file but it is not valid anymore. Full stop!")
            raise ValueError(
                "You have imported the detector description from a pickle/json file but it is not valid anymore. Full stop!")

        if any_update:
            self.logger.info(f"Update detector to {time}")
            for key in self.__buffered_stations:
                if update_buffer_for_station[key]:
                    # remove everything (could be handled smarter ...)
                    self.__buffered_stations[key] = {}

            for station_id, need_update in update_buffer_for_station.items():
                if need_update and self.has_station(station_id):
                    self._query_station_information(station_id)

        # Return when buffer is not empty. This has to come first ...
        for station_id in self.__buffered_stations:
            if len(self.__buffered_stations[station_id]):
                return

        # ... and than second
        if len(self.__buffered_stations):
            return

        # When you reach this point something went wrong ...
        self.logger.warning(f"Empty detector for {time}!")


    @_check_detector_time
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

    @_check_detector_time
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

    def _query_station_information(self, station_id):
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
        """

        if station_id in self.__buffered_stations and self.__buffered_stations[station_id] != {}:
            raise ValueError(
                f"Query information for station {station_id} which is still in buffer.")

        self.logger.info(
            f"Query information for station {station_id} at {self.get_detector_time()}")
        if self._query_all:
            station_information = self.__db.get_complete_station_information(
                station_id)
        else:
            station_information = self.__db.get_general_station_information(
                station_id)

        if len(station_information) != 1:
            raise ValueError(f"Could not query information of station {station_id} at {self.get_detector_time()}. "
                             f"Found {len(station_information)} entries in database.")

        self.__buffered_stations[station_id] = station_information[station_id]

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
        if not self.has_station(station_id):
            err = f"Station id {station_id} not commission at {self.get_detector_time()}"
            self.logger.error(err)
            raise ValueError(err)

        if _keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id]):
            raise KeyError(
                f"Could not find channel {channel_id} in detector description for station {station_id}. Did you call det.update(...)?")

        if with_position and _keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id, "channel_position"]):

            if _keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id, "id_position"]):
                raise KeyError(
                    f"\"id_position\" not in buffer for station.channel {station_id}.{channel_id}. Did you call det.update(..)")

            position_id = self.__buffered_stations[
                station_id]["channels"][channel_id]["id_position"]
            self.logger.debug(
                f"Query position of station.channel {station_id}.{channel_id} with id {position_id}")

            channel_position_dict = self.__db.get_position(
                position_id=position_id, component="channel")
            self.__buffered_stations[station_id]["channels"][channel_id]['channel_position'] = channel_position_dict

        if with_signal_chain and _keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id, "signal_chain"]):

            if _keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id, "id_signal"]):
                raise KeyError(
                    f"\"id_signal\" not in buffer for station.channel {station_id}.{channel_id}. Did you call det.update(..)")

            signal_id = self.__buffered_stations[station_id]["channels"][channel_id]['id_signal']
            self.logger.debug(
                f"Query signal chain of station.channel {station_id}.{channel_id} with id {signal_id}")

            channel_sig_info = self.__db.get_channel_signal_chain(signal_id)
            channel_sig_info.pop('channel_id', None)

            self.__buffered_stations[station_id]["channels"][channel_id]['signal_chain'] = channel_sig_info

        return self.__buffered_stations[station_id]["channels"][channel_id]

    @_check_detector_time
    def get_channel(self, station_id, channel_id):
        """
        Returns a dictionary of all channel parameters

        Parameters
        ----------
        station_id: int
            The station id

        channel_id: int
            The channel id

        Returns
        -------
        channel_info: dict
            Dictionary of channel parameters
        """
        self.get_signal_chain_response(station_id, channel_id)  # this adds `total_response` to dict
        # Since we are not actually overwritting existing values we can use a shallow copy
        channel_data = copy.copy(self.__get_channel(station_id, channel_id, with_position=True, with_signal_chain=True))

        for key in self.__default_values:

            if key in channel_data:
                raise ValueError(f"{key} already in channel data. You can not update this in the this detector class. Use the ModDetector class.")

            if isinstance(self.__default_values[key], dict):
                channel_data[key] = self.__default_values[key][channel_id]
            else:
                channel_data[key] = self.__default_values[key]

        # Add ADC parameter to channel description. This is needed for ADC and trigger modules.
        for key, value in self.__buffered_stations[station_id]["signal_digitizer_config"].items():
            channel_data[f"adc_{key}"] = value

        for key, value in self.__buffered_stations[station_id]["trigger_digitizer_config"].items():
            channel_data[f"trigger_adc_{key}"] = value

        return channel_data

    @_check_detector_time
    def get_station(self, station_id):
        """
        Returns a dictionary of all station parameters/information including some channel information.
        The channel's (signal chain) response is not necessarily returned (they might be if they are
        already in the buffer but this is not ensured). To get the complete channel information call
        `self.get_channel(station_id, channel_id)`.

        Parameters
        ----------
        station_id: int
            The station id

        Returns
        -------
        station_info: dict
            Dictionary of station parameters/information
        """
        if not self.has_station(station_id):
            err = f"Station id {station_id} not commission at {self.get_detector_time()}"
            self.logger.error(err)
            raise ValueError(err)

        if not self._query_all:
            for ch in self.get_channel_ids(station_id):
                self.__get_channel(station_id, ch, with_position=True)  # stores all relevant information in buffer

        return self.__buffered_stations[station_id]

    @_check_detector_time
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

        if _keys_not_in_dict(self.__buffered_stations, [station_id, "station_position"]):

            if _keys_not_in_dict(self.__buffered_stations, [station_id, "id_position"]):
                raise KeyError(
                    f"\"id_position\" not in buffer for station {station_id}. Did you call det.update(..)")

            station_position_id = self.__buffered_stations[station_id]["id_position"]
            self.logger.debug(
                f"Query position for id \"{station_position_id}\"")

            station_position = self.__db.get_position(
                position_id=station_position_id)
            self.__buffered_stations[station_id]["station_position"] = station_position

        return np.array(self.__buffered_stations[station_id]["station_position"]["position"])

    @_check_detector_time
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

    @_check_detector_time
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

    @_check_detector_time
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
        return channel_info["signal_chain"]

    @_check_detector_time
    def get_amplifier_response(self, station_id, channel_id, frequencies):
        """
        Returns the complex response function for the entire signal chain of a channel.

        This includes not only the (main) amplifier but also cables and other components.
        Note that while group delays are appropriately accounted for, an overall time delay
        (mostly due to cable delay) has been removed and is instead accounted for by
        `get_time_delay`.

        Parameters
        ----------
        station_id: int
            The station id

        channel_id: int
            The channel id

        frequencies: array of floats
            Array of frequencies for which the response is returned

        Returns
        -------
        response: array of complex floats
            Complex response function

        See Also
        --------
        get_time_delay
        """
        response_func = self.get_signal_chain_response(station_id, channel_id)
        return response_func(frequencies)

    @_check_detector_time
    def get_signal_chain_response(self, station_id, channel_id, trigger=False):
        """
        Returns a `detector.response.Response` object which describes the complex response of the
        entire signal chain, i.e., the combined reponse of all components of one
        channel. For example: IGLU, fiber-cable, DRAB, coax-cable, RADIANT.

        Parameters
        ----------

        station_id: int
            The station id

        channel_id: int
            The channel id

        trigger: bool
            If True, the trigger channel resonse is returned. An error is raised if no trigger response exists. (Default: False)

        Returns
        -------

        response: `detector.response.Response`
            Returns combined response of the channel
        """
        signal_chain_dict = self.get_channel_signal_chain(
            station_id, channel_id)

        if trigger:
            response_chain_key = "trigger_response_chain"
            response_key = "total_trigger_response"

            if response_chain_key not in signal_chain_dict or "is_trigger_chain_absolute" not in signal_chain_dict:
                raise KeyError(f"No trigger response for station.channel {station_id}.{channel_id}")

            if not signal_chain_dict["is_trigger_chain_absolute"]:
                raise NotImplementedError("Relative trigger chains are not implemented yet.")
        else:
            response_chain_key = "response_chain"
            response_key = "total_response"

        # total_response can be None if imported from file
        if response_key not in signal_chain_dict or signal_chain_dict[response_key] is None:
            measurement_components_dic = signal_chain_dict[response_chain_key]

            # Here comes a HACK
            components = list(measurement_components_dic.keys())
            is_equal = False
            if "drab_board" in components and "iglu_board" in components:

                is_equal = np.allclose(
                    measurement_components_dic["drab_board"]["mag"],
                    measurement_components_dic["iglu_board"]["mag"])

                if is_equal:
                    self.logger.warn(
                        f"Station.channel {station_id}.{channel_id}: Currently both, "
                        "iglu and drab board are configured in the signal chain but their "
                        "responses are the same (because we measure them together in the lab). "
                        "Skip the drab board response.")

            responses = []
            for key, value in measurement_components_dic.items():

                # Skip drab_board if its equal with iglu (see warning above)
                if is_equal and key == "drab_board":
                    continue

                if "weight" not in value:
                    self.logger.warn(f"Component {key} does not have a weight. Assume a weight of 1 ...")
                weight = value.get("weight", 1)

                attenuator = value.get("attenuator", 0)

                if "time_delay" in value:
                    time_delay = value["time_delay"]
                else:
                    self.logger.warning(
                        f"The signal chain component \"{key}\" of station.channel "
                        f"{station_id}.{channel_id} has no time delay stored... "
                        "Set component time delay to 0")
                    time_delay = 0

                ydata = [value["mag"], value["phase"]]
                response = Response(value["frequencies"], ydata, value["y-axis_units"],
                                    time_delay=time_delay, weight=weight, name=key,
                                    station_id=station_id, channel_id=channel_id,
                                    log_level=self.__log_level,
                                    attenuator_in_dB=attenuator)

                responses.append(response)

            # Buffer object
            signal_chain_dict[response_key] = np.prod(responses)

        return signal_chain_dict[response_key]

    @_check_detector_time
    def get_signal_chain_components(self, station_id, channel_id):
        """
        Returns the names of all components in the signal chain.

        Parameters
        ----------

        station_id: int
            The station id

        channel_id: int
            The channel id

        Returns
        -------

        signal_chain_components: dict
            A dictionaries with the keys being the names of the components and the values their weights in
            the signal chain. (For an explanation of a weight see `detector.response.Response`)
        """
        signal_chain_dict = self.get_channel_signal_chain(
            station_id, channel_id)
        signal_chain_components = {
            key: value["weight"] for key, value in
                signal_chain_dict['response_chain'].items()}

        return signal_chain_components

    @_check_detector_time
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

    @_check_detector_time
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
        if _keys_not_in_dict(self.__buffered_stations, [station_id, "devices", device_id]):
            # All devices should have been queried with _query_station_information
            raise KeyError(f"Device {device_id} not in detector description.")

        if _keys_not_in_dict(self.__buffered_stations, [station_id, "devices", device_id, "device_position"]):

            if _keys_not_in_dict(self.__buffered_stations, [station_id, "devices", device_id, "id_position"]):
                raise KeyError(
                    f"\"id_position\" not in buffer for device {device_id}.")

            position_id = self.__buffered_stations[station_id]["devices"][device_id]["id_position"]

            device_pos_info = self.__db.get_device_position(
                device_position_id=position_id)
            self.__buffered_stations[station_id]["devices"][device_id]['device_position'] = device_pos_info

        return np.array(self.__buffered_stations[station_id]["devices"][device_id]["device_position"]["position"])

    @_check_detector_time
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

    @_check_detector_time
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

        zenith: float (Default: None)
            So far has no use in this class. Only defined to keep the interface
            in parity to other detector classes

        Returns
        -------

        antenna_model: string
            Name of the antenna model (describing the Vector effective length VEL)
        """
        channel_info = self.__get_channel(
            station_id, channel_id, with_signal_chain=True)
        return channel_info["signal_chain"]["VEL"]

    def get_antenna_type(self, station_id, channel_id):
        """
        Returns type of antenna, i.e., "VPol" or "HPol" or "LPDA", ...

        Parameters
        ----------
        station_id: int
            Station id

        channel_id: int
            Channel id

        Returns
        -------

        ant_type: string
            Accronym/abbrivatipn of the antenna type
        """
        channel_info = self.__get_channel(
            station_id, channel_id, with_signal_chain=True)
        return channel_info['ant_type']


    def get_number_of_samples(self, station_id, channel_id=None):
        """ Get number of samples for recorded waveforms

        All RNO-G channels have the same number of samples, the argument channel_id is not used but we keep
        it here for consistency with outer detector classes.

        Parameters
        ----------

        station_id: int
            Station id

        Returns
        -------

        number_of_samples: int
            Number of samples with which each waveform is recorded
        """

        if not self.has_station(station_id):
            err = f"Station id {station_id} not commission at {self.get_detector_time()}"
            self.logger.error(err)
            raise ValueError(err)

        if _keys_not_in_dict(self.__buffered_stations, [station_id, "signal_digitizer_config", "number_of_samples"]):
            raise KeyError(
                f"Could not find \"number_of_samples\" for station {station_id} in buffer. Did you call det.update(...)?")

        return int(self.__buffered_stations[station_id]["signal_digitizer_config"]['number_of_samples'])


    def get_sampling_frequency(self, station_id, channel_id=None, trigger=False):
        """ Get sampling frequency per station / channel

        All RNO-G channels have the same sampling frequency, the argument channel_id is not used but we keep
        it here for consistency with other detector classes.

        Parameters
        ----------
        station_id: int
            Station id

        channel_id: int (default: None)
            Not Used!

        trigger: bool
            If True, the sampling rate of the trigger board is returned (FLOWER). (Default: False)

        Returns
        -------
        sampling_rate: float
            Sampling frequency
        """
        if not self.has_station(station_id):
            err = f"Station id {station_id} not commission at {self.get_detector_time()}"
            self.logger.error(err)
            raise ValueError(err)

        if trigger:
            key = "trigger_digitizer_config"
        else:
            key = "signal_digitizer_config"

        if _keys_not_in_dict(self.__buffered_stations, [station_id, key, "sampling_frequency"]):
            raise KeyError(
                f"Could not find \"sampling_frequency\" for station {station_id} in buffer. Did you call det.update(...)?")

        return float(self.__buffered_stations[station_id][key]['sampling_frequency'])


    def get_noise_temperature(self, station_id, channel_id):
        """ Get noise temperture per station / channel """
        noise_temperature = self.get_channel(station_id, channel_id)["noise_temperature"]
        return noise_temperature


    def is_channel_noiseless(self, station_id, channel_id):
        is_noiseless = self.get_channel(station_id, channel_id)["is_noiseless"]
        return is_noiseless


    def get_cable_delay(self, station_id, channel_id, use_stored=True, trigger=False):
        """ Same as `get_time_delay`. Only here to keep the same interface as the other detector classes. """
        # FS: For the RNO-G detector description it is not easy to determine the cable delay alone
        # because it is not clear which reference components may need to be subtracted.
        # However, having the cable delay without amplifiers is anyway weird.
        return self.get_time_delay(station_id, channel_id, use_stored=use_stored, trigger=trigger)


    def _get_time_delay(self, station_id, channel_id, trigger=False):
        """ Returns the sum of the time delay of all components in the signal chain calculated from the phase.

        The function returns the values which were precalculated by the hardware database interface. They
        should correspond to the phase delay at ~200 MHz. This function returns the same values as
        `get_time_delay` or `get_cable_delay` if `use_stored==True` which is the default for those functions.
        However, Unlike `get_time_delay` or `get_cable_delay`, this function does not require the entire S21
        parameter measurement to be available (this is the case if the detector was exported to file with
        `drop_response_data==True` to reduce file size).

        Parameters
        ----------
        station_id: int
            The station id

        channel_id: int
            The channel id

        trigger: bool
            If True, the trigger channel resonse is returned. An error is raised
            if no trigger response exists. (Default: False)

        Returns
        -------
        time_delay: float
            Sum of the time delays of all components in the signal chain for one channel.

        Also see
        --------
        get_time_delay
        """
        signal_chain_dict = self.get_channel_signal_chain(
        station_id, channel_id)

        if trigger:
            response_chain_key = "trigger_response_chain"

            if response_chain_key not in signal_chain_dict or "is_trigger_chain_absolute" not in signal_chain_dict:
                raise KeyError(f"No trigger response for station.channel {station_id}.{channel_id}")

            if not signal_chain_dict["is_trigger_chain_absolute"]:
                raise NotImplementedError("Relative trigger chains are not implemented yet.")
        else:
            response_chain_key = "response_chain"

        measurement_components_dic = signal_chain_dict[response_chain_key]

        total_time_delay = 0
        for key, value in measurement_components_dic.items():

            if "weight" in value:
                weight = value["weight"]
            else:
                self.logger.warn(f"Component {key} does not have a weight. Assume a weight of 1 ...")
                weight = 1

            assert abs(weight) == 1, f"Weight is {weight}, only values of `-1` and `1` are currently supported."

            if "time_delay" in value:
                time_delay = value["time_delay"]
            else:
                self.logger.warning(
                    f"The signal chain component \"{key}\" of station.channel "
                    f"{station_id}.{channel_id} has no time delay stored... "
                    "Set component time delay to 0")
                time_delay = 0

            total_time_delay += weight * time_delay

        return total_time_delay


    def get_time_delay(self, station_id, channel_id, use_stored=True, trigger=False):
        """ Return the sum of the time delay of all components in the signal chain calculated from the phase.

        Parameters
        ----------
        station_id: int
            The station id

        channel_id: int
            The channel id

        use_stored: bool
            If True, take time delay as stored in DB rather than calculated from response. (Default: True)

        trigger: bool
            If True, the trigger channel resonse is returned. An error is raised if no trigger response exists.
            (Default: False)

        Returns
        -------
        time_delay: float
            Sum of the time delays of all components in the signal chain for one channel

        Notes
        -----
        IMPORTANT: The value returned by this function does *not* directly correspond to the overall time
        delay / cable delay of the requested channel! A residual group delay may be present and is accounted for
        by the response provided by `get_amplifier_response`.
        
        See Also
        --------
        get_cable_delay, get_amplifier_response
        """
        signal_chain_dict = self.get_channel_signal_chain(
            station_id, channel_id)

        if use_stored:
            try:
                resp = self.get_signal_chain_response(station_id, channel_id, trigger=trigger)
                return resp.get_time_delay()
            except KeyError: # in case the full S21 parameters are not stored
                return self._get_time_delay(station_id, channel_id, trigger=trigger)
        else:
            time_delay = 0
            if trigger and "trigger_response_chain" not in signal_chain_dict:
                raise KeyError(f"No trigger response for station.channel {station_id}.{channel_id}")

            prefix = "trigger_" if trigger else ""
            for key, value in signal_chain_dict[f"{prefix}response_chain"].items():
                ydata = [value["mag"], value["phase"]]
                # This is different from within `get_signal_chain_response` because we do set the time delay here
                # and thus we do not remove it from the response.
                response = Response(value["frequencies"], ydata, value["y-axis_units"],
                                    name=key, station_id=station_id, channel_id=channel_id,
                                    log_level=self.__log_level)

                weight = value.get("weight", 1)
                time_delay += weight * response._calculate_time_delay()

        return time_delay


    def get_site(self, station_id):
        """
        This detector class is exclusive for the RNO-G detector at Summit Greenland.

        Parameters
        ----------
        station_id: int
            the station ID (not used, only for compatibility with other detector classes)

        Returns
        -------
        site: str
            Returns "summit"
        """
        return "summit"


    def get_site_coordinates(self, station_id=None):
        """
        Get the (latitude, longitude) coordinates (in degrees) for the RNO-G detector site.

        Parameters
        ----------
        station_id: int
            the station ID (not used, only for compatibility with other detector classes)

        Returns
        -------
        coordinates: tuple(float, float)
            Tuple of latitude and longitude in degrees
        """
        return (72.57, -38.46)


    def get_database(self):
        """
        Returns the database connection

        Returns
        -------
        db: MongoClient
            Returns the database connection
        """
        if self.__db is None:
            self.logger.error("No database connection available. Return None")
            return None

        return self.__db


    def get_component(self, collection="coax_cable", component="daq_drab_flower_2024_avg"):
        """ Get the response of a component from the database.

        This function is a wrapper around the same named function in the database class.

        Parameters
        ----------
        collection: str (default: "coax_cable")
            The collection name in the database.
        component: str (default: "daq_drab_flower_2024_avg")
            The component name in the collection.

        Returns
        -------
        resp: `NuRadioReco.detector.response.Response`
            Returns the response of a component.

        See Also
        --------
        NuRadioReco.detector.RNO_G.db_mongo_read.Database.get_component_data
        """
        if component in self.additional_data and 'y-axis_units' in self.additional_data[component]:
            component_data = self.additional_data[component]
        else:
            db = self.get_database()
            if db is None:
                raise ValueError(
                    "No database connection. You probably imported the detector from a file. "
                    "Please use the DB connection to load the component data.")

            # load the s21 parameter measurement
            component_data = db.get_component_data(
                collection, component)
            self.additional_data[component] = component_data

        resp = Response(
            component_data["frequencies"], [component_data["mag"], component_data["phase"]], component_data['y-axis_units'],
            name=component, station_id=-1, channel_id=None)

        return resp


def produce_detector_files_for_all_time_periods(drop_response_data=False):
    """
    This function produces a detector file for each time period necessary
    """

    station_ids = [11, 12, 13, 21, 22, 23, 24]
    suffix = "_withoutS21" if drop_response_data else ""

    for station_id in station_ids:
        det = Detector(
            log_level=logging.DEBUG,
            always_query_entire_description=True,
            select_stations=station_id)

        db = det.get_database()
        ts_dict = db.query_modification_timestamps_per_station(station_id)

        ts = np.unique(np.hstack([station_ts['modification_timestamps'] for station_ts in ts_dict.values()]))
        for tdx in range(len(ts) - 1):
            t0 = ts[tdx]
            t1 = ts[tdx + 1]
            time = t0 + (t1 - t0) / 2

            det.update(time)
            if det.has_station(station_id):
                det.export(
                    f"rnog_detector_st{station_id}_{t0.strftime('%Y%m%d')}-{t1.strftime('%Y%m%d')}{suffix}",
                    drop_response_data=drop_response_data)


if __name__ == "__main__":
    # produce_detector_files_for_all_time_periods()
    from NuRadioReco.detector import detector

    det = detector.Detector(source="rnog_mongo", log_level=logging.DEBUG, always_query_entire_description=True,
                            database_connection='RNOG_public', select_stations=13)

    det.update(datetime.datetime(2023, 7, 2, 0, 0))
    response = det.get_signal_chain_response(station_id=13, channel_id=0)

    from NuRadioReco.framework import electric_field
    ef = electric_field.ElectricField(channel_ids=[0])
    ef.set_frequency_spectrum(np.ones(1025, dtype=complex), sampling_rate=2.4)

    # Multipy the response to a trace. The multiply operator takes care of everything
    trace_at_readout = ef * response

    # getting the complex response as array
    freq = np.arange(50, 1000) * units.MHz
    complex_resp = response(freq)
