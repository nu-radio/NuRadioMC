# Keep that the first import
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s:%(name)s:%(funcName)s : %(message)s", datefmt="%H:%M:%S")

from NuRadioReco.detector.RNO_G.db_mongo_read import Database, _convert_astro_time_to_datetime
import NuRadioReco.framework.base_trace
from NuRadioReco.utilities import units

from radiotools import helper
from scipy import interpolate
from functools import wraps
import astropy.time

import datetime
import numpy as np
import collections
import pickle
import json
import re
import copy
import bson
import lzma


def _json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, bson.objectid.ObjectId):
        return str(obj)
    elif isinstance(obj, Response):
        pass
    else:
        raise TypeError ("Type %s not serializable" % type(obj))


def _keys_not_in_dict(d, keys):
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


def _check_detector_time(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        self.get_detector_time()  # this will raise an error if time is not set
        return method(self, *method_args, **method_kwargs)
    return _impl


class Detector():
    def __init__(self, database_connection='RNOG_test_public', log_level=logging.INFO, over_write_handset_values={},
                 database_time=None, always_query_entire_description=True, detector_file=None,
                 select_stations=None):
        """

        Parameters
        ----------

        database_connection : str (Default: 'RNOG_test_public')
            Allows to specify database connection. Passed to mongo-db interface.

        log_level : `logging.LOG_LEVEL` (Default: logging.INFO)
            Defines verbosity level of logger. Other options are: `logging.WARNING`, `logging.DEBUG`, ...

        over_write_handset_values : dict (Default: {})
            Overwrite the default values for the manually set parameter which are not (yet) implemented in the database.
            (Default: {}, the acutally default values for the parameters in question are defined below)

        database_time : `datetime.datetime`
            Set database time which is used to select the primary measurement. By default (= None) the database time
            is set to now (time the code is running) to select the measurement which is now primary.

        always_query_entire_description : bool (Default: True)
            If True, query the entire detector describtion all at once when calling Detector.update(...) (if necessary).

        detector_file : str
            File to import detector description instead of querying from DB. (Default: None -> query from DB)

        select_stations : int or list(int) (Default: None)
            Select a station or list of stations using their station ids for which the describtion is provided.
            This is useful for example in simulations when one wants to simulate only one station. The default None
            means to descibe all commissioned stations.
        """

        self.logger = logging.getLogger("rno-g-detector")
        self.logger.setLevel(log_level)

        # Define default values for parameter not (yet) implemented in DB. Those values are taken for all channels.
        self.__default_values = {
            "noise_temperature": 300 * units.kelvin,
            "is_noiseless": False,
        }

        if select_stations is not None and not isinstance(select_stations, list):
            select_stations = [select_stations]

        self.selected_stations = select_stations
        self.logger.info(f"Select the following stations (if possible): {select_stations}")

        if detector_file is None:
            self._det_imported_from_file = False

            self.__db = Database(database_connection=database_connection)
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

            # This should be set with Detector.update(..) and corresponds to the time of a measurement. It will be use to
            # decide which components are commissioned at the time of the measurement
            self.__detector_time = None

            # Initialise the primary buffer
            self.__buffered_stations = collections.defaultdict(dict)

            self._query_all = always_query_entire_description
        else:
            self._query_all = None  # specific case for file imported detector descriptions
            self._det_imported_from_file = True
            self._import_from_file(detector_file)

        # Allow overwriting the hard-coded values
        self.__default_values.update(over_write_handset_values)

        info = f"Query entire detector description at once: {self._query_all}"

        info += "\nUsing the following hand-set values:"
        for key, value in self.__default_values.items():
            info += f"\n\t{key:<20}: {value}"

        self.logger.info(info)

    def export(self, filename, json_kwargs=None):
        """
        Export the buffered detector description.

        Parameters
        ----------

        filename: str
            Filename of the exported detector description

        json_kwargs: dict
            Arguments passed to json.dumps(..). (Default: None -> dict(indent=0, default=_json_serial))
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
                    export_dir[station_id]["channels"][channel_id]["signal_chain"].pop("response_chain", None)
                    export_dir[station_id]["channels"][channel_id]["signal_chain"].pop("total_response", None)

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

            # need to convert station/channel id keys back to integers
            for station_id, station_data in import_dict["data"].items():
                if self.selected_stations is not None and int(station_id) not in self.selected_stations:
                    continue

                station_data["channels"] = {int(channel_id): channel_data for channel_id, channel_data in station_data["channels"].items()}
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

        time: datetime.datetime or `astro.time.Time`
            UTC time.
        '''

        if isinstance(time, astropy.time.Time):
            time = _convert_astro_time_to_datetime(time)
        elif not isinstance(time, datetime.datetime):
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

        time: datetime.datetime or `astro.time.Time`
            Unix time of measurement.
        """
        if isinstance(time, astropy.time.Time):
            time = _convert_astro_time_to_datetime(time)

        self.logger.info(f"Update detector to {time}")

        self.__set_detector_time(time)
        if not self._det_imported_from_file:
            self.__db.set_detector_time(time)

        update_buffer_for_station = self._check_update_buffer()
        any_update = np.any([v for v in update_buffer_for_station.values()])

        if self._det_imported_from_file and any_update:
            self.logger.error(
                "You have imported the detector description from a pickle/json file but it is not valid anymore. Full stop!")
            raise ValueError(
                "You have imported the detector description from a pickle/json file but it is not valid anymore. Full stop!")

        if any_update:
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
        self.logger.error(f"Empty detector for {time}!")
        raise ValueError(f"Empty detector for {time}!")


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

        Returns
        -------
        None
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
    def get_signal_chain_response(self, station_id, channel_id):
        """
        Returns a `rnog_detector.Response` object which describes the complex response of the
        entire signal chain, i.e., the combined reponse of all components of one
        channel. For example: IGLU, fiber-cable, DRAB, coax-cable, RADIANT.

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

        # total_response can be None if imported from file
        if "total_response" not in signal_chain_dict or signal_chain_dict["total_response"] is None:
            measurement_components_dic = signal_chain_dict["response_chain"]

            # Here comes a HACK
            components = list(measurement_components_dic.keys())
            is_equal = False
            if "drab_board" in components and "iglu_board" in components:

                is_equal = np.allclose(measurement_components_dic["drab_board"]["mag"],
                                       measurement_components_dic["iglu_board"]["mag"])

                if is_equal:
                    self.logger.warn(f"Station.channel {station_id}.{channel_id}: Currently both, "
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
                                    station_id=station_id, channel_id=channel_id)


                responses.append(response)

            # Buffer object
            signal_chain_dict["total_response"] = np.prod(responses)

        return signal_chain_dict["total_response"]

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
            the signal chain. (For an explanation of a weight see `rnog_detector.Response`)
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

        if _keys_not_in_dict(self.__buffered_stations, [station_id, "number_of_samples"]):
            raise KeyError(
                f"Could not find \"number_of_samples\" for station {station_id} in buffer. Did you call det.update(...)?")

        return int(self.__buffered_stations[station_id]['number_of_samples'])


    def get_sampling_frequency(self, station_id, channel_id):
        """ Get sampling frequency per station / channel

        All RNO-G channels have the same sampling frequency, the argument channel_id is not used but we keep
        it here for consistency with outer detector classes.

        Parameters
        ----------

        station_id: int
            Station id

        Returns
        -------

        sampling_rate: int
            Sampling frequency
        """
        if not self.has_station(station_id):
            err = f"Station id {station_id} not commission at {self.get_detector_time()}"
            self.logger.error(err)
            raise ValueError(err)

        if _keys_not_in_dict(self.__buffered_stations, [station_id, "sampling_rate"]):
            raise KeyError(
                f"Could not find \"sampling_rate\" for station {station_id} in buffer. Did you call det.update(...)?")

        return float(self.__buffered_stations[station_id]['sampling_rate'])


    def get_noise_temperature(self, station_id, channel_id):
        """ Get noise temperture per station / channel """
        noise_temperature = self.__default_values["noise_temperature"]
        if isinstance(noise_temperature, float):
            self.logger.warn(
                f"Return a hard-coded value for the noise temperature of {noise_temperature / units.kelvin} K. "
                "This information is not (yet) implemented in the DB.")
            return noise_temperature
        elif isinstance(noise_temperature, dict):
            self.logger.warn(
                f"Return a hard-coded value for the noise temperature of {noise_temperature / units.kelvin} K for channel {channel_id}. "
                "This information is not (yet) implemented in the DB.")
            return noise_temperature[channel_id]
        else:
            raise ValueError("Unkown type for hard-coded value")

    def is_channel_noiseless(self, station_id, channel_id):
        is_noiseless = self.__default_values["is_noiseless"]
        self.logger.warn(
            f"Return a hard-coded value for \"is_noiseless\" of {is_noiseless} for all stations / channels. "
            "This information is not (yet) implemented in the DB.")
        return is_noiseless


    def get_cable_delay(self, station_id, channel_id, use_stored=True):
        """
        Return the cable delay of a signal chain as stored in the detector description.
        This interface is required by simulation.py. See get_time_delay for description of
        arguments.
        """
        return self.get_time_delay(station_id, channel_id, cable_only=True, use_stored=use_stored)

    def get_time_delay(self, station_id, channel_id, cable_only=False, use_stored=False):
        """ Return the sum of the time delay of all components in the signal chain calculated from the phase

        Parameters
        ----------

        station_id: int
            The station id

        channel_id: int
            The channel id

        cable_only: bool
            If True: Consider only cables to calculate delay. (Default: False)

        use_stored: bool
            If True, take time delay as stored in DB rather than calculated from response. (Default: False)

        Returns
        -------

        time_delay: float
            Sum of the time delays of all components in the signal chain for one channel
        """

        signal_chain_dict = self.get_channel_signal_chain(
            station_id, channel_id)

        time_delay = 0
        for key, value in signal_chain_dict["response_chain"].items():

            if re.search("cable", key) is None and cable_only:
                continue

            if use_stored:
                if "time_delay" not in value or "cable_delay" not in value:
                    self.logger.warning(
                        f"The signal chain component \"{key}\" of station.channel "
                        f"{station_id}.{channel_id} has no cable/time delay stored... Skip it")
                    continue

                try:
                    time_delay += value["time_delay"]
                except KeyError:
                    time_delay += value["cable_delay"]

            else:
                ydata = [value["mag"], value["phase"]]
                response = Response(value["frequencies"], ydata, value["y-axis_units"],
                                    name=key, station_id=station_id, channel_id=channel_id)

                time_delay += response._get_time_delay()

        return time_delay


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

    Examples
    --------

    .. code-block::
        response = det.get_signal_chain_response(station_id=24, channel_id=0)

        # Multipy the response to a trace. The multiply operator takes care of everything
        trace_at_readout = trace_at_antenna * response

        # getting the complex response as array
        freq = np.arange(50, 1000) * units.MHz
        complex_resp = response(freq)

    """

    def __init__(self, frequency, y, y_unit, time_delay=0, weight=1,
                 name="default", station_id=None, channel_id=None,
                 remove_time_delay=True, debug_plot=False):
        """
        Parameters
        ----------

        frequency : list(float)
            The frequency vector at which the response is measured. Unit has to be GHz.

        y : [list(float), list(float)]
            The measured response. First entry is the vector of the measured amplitude, the second entry is the measured phase.
            The unit of both entries is specified with the next argument.

        y_unit : [str, str]
            The first entry specifies the unit of the measured amplitude. Options are "dB", "MAG" and "mag".
            The second entry specifies the unit of the measured phase. Options are "rad" and "deg".

        time_delay : float (Default: 0)
            "Average" group delay. Read from database. Will be used to normalize the phase and stored alongside response

        weight : float (Default: 1)
            Specifies the weight with which this component reponse "adds" to the total signal-chain response or data.
            Its the exponent of the complex multiplicitive gain. That means that a value of 1 means to linear multiply this
            reponse while a value of -1 means to divide by this reponse.

        name : str (Default: "default")
            Give the response a name. This is only use for printing purposes.

        station_id : int (Default: None)
            Associated station id.

        channel_id : int (Default: None)
            Associated channel id.

        remove_time_delay : bool (Default: True)
            If True, remove `time_delay` from response.

        debug_plot : bool (Default: False)
            If True, produce a debug plot
        """

        self.__names = [name]
        self._station_id = station_id
        self._channel_id = channel_id

        self._sanity_check = True # Tmp

        if self._station_id is None or self._channel_id is None:
            logging.error(f"Station and channel id were not defined for response {name}. Please do that.")

        self.__frequency = np.array(frequency) * units.GHz

        if y[0] is None or y[1] is None:
            raise ValueError("Data for response incomplete, detected \"None\"")

        y_ampl, y_phase = np.array(y)
        if y_unit[0] == "dB":
            gain = 10 ** (y_ampl / 20)
        elif y_unit[0].lower() == "mag":
            gain = y_ampl
        else:
            raise KeyError

        if y_unit[1].lower() == "deg":
            if np.max(np.abs(y_phase)) < 2 * np.pi:
                logging.warning("Is the phase really in deg? Does not look like it... "
                                f"Do not convert to rad. Phase: {y_phase}")
            else:
                y_phase = np.deg2rad(y_phase)
        elif y_unit[1].lower() == "rad":
            y_phase = y_phase
        else:
            raise KeyError


        if remove_time_delay:
            y_phase_orig = np.copy(np.unwrap(y_phase))
            _response = gain * np.exp(1j * (y_phase + 2 * np.pi * time_delay * self.__frequency))
            y_phase = np.angle(_response)
        else:
            time_delay = 0

        y_phase = np.unwrap(y_phase)

        self.__gains = [interpolate.interp1d(
            self.__frequency, gain, kind="linear", bounds_error=False, fill_value=0)]

        self.__phases = [interpolate.interp1d(
            self.__frequency, y_phase, kind="linear", bounds_error=False, fill_value=0)]

        self.__weights = [weight]
        self.__time_delays = [weight * time_delay]

        if debug_plot:
            from matplotlib import pyplot as plt
            fig, axs = plt.subplots(3, 1, sharex=True)  #, gridspec_kw=dict(hspace=0.03))
            axs[0].set_title(name)
            frequency_interp = np.linspace(self.__frequency[0], self.__frequency[-1], 10000)
            axs[0].plot(self.__frequency, gain, "C0o", label="data", markersize=2)
            axs[0].plot(frequency_interp, self.__gains[0](frequency_interp), "C1--", lw=1, label="interpolation")

            if remove_time_delay:
                axs[1].plot(self.__frequency, y_phase_orig, "C0o", markersize=2, label="Original phase")
                ax2 = axs[1].twinx()
                ax2.plot(self.__frequency, y_phase, "C2o", markersize=2, label=f"Excl. time delay of {self.__time_delays[0]:.2f}ns")
                ax2.plot(frequency_interp, self.__phases[0](frequency_interp), "C3--", lw=1, label="interpolation")
                ax2.set_ylabel("norm. phase / rad")
                ax2.legend(fontsize=5)
            else:
                axs[1].plot(self.__frequency, y_phase, "C0o", markersize=2)
                axs[1].plot(frequency_interp, self.__phases[0](frequency_interp), "C1--", lw=1)

            axs[0].set_ylabel("gain")
            axs[1].set_ylabel("phase / rad")

            axs[0].legend(fontsize=5)
            # axs[1].legend(fontsize=5)

            group_delay = -np.diff(y_phase) / np.diff(self.__frequency)[0] / (2 * np.pi)
            axs[2].plot(self.__frequency[:-1], group_delay, "C0o", markersize=2)
            frequency_mask = np.all([50 * units.MHz < self.__frequency[:-1], self.__frequency[:-1] < 800 * units.MHz], axis=0)
            axs[2].set_ylim(np.amin(group_delay[frequency_mask]), np.amax(group_delay[frequency_mask]))
            axs[2].set_ylabel("group delay / ns")

            axs[-1].set_xlabel("frequency / GHz")
            fig.tight_layout()
            plt.savefig(f'{name}_{self._station_id}_{self._channel_id}_'
                        f'{datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")}_debug.png', transparent=False)
            plt.close()

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

        for gain, phase, weight in zip(self.__gains, self.__phases, self.__weights):

            _gain = gain(freq / units.GHz)

            # to avoid RunTime warning and NANs in total reponse
            if weight == -1:
                _gain = np.where(_gain > 0, _gain, 1e-6)

            response *= (_gain * np.exp(1j * phase(freq / units.GHz))) ** weight

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
            if self._station_id != other._station_id or \
                self._channel_id != other._channel_id:
                logging.error("It looks like you are combining responses from "
                                  f"two different channels: {self._station_id}.{self._channel_id} "
                                  f" vs {other._station_id}.{other._channel_id} (station_id.channel_id)")
            # Store each response individually: append/concatenate lists of gains and phases.
            # The multiplication happens in __call__.
            self.__names += other.__names
            self.__gains += other.__gains
            self.__phases += other.__phases
            self.__weights += other.__weights
            self.__time_delays += other.__time_delays
            return self

        elif isinstance(other, NuRadioReco.framework.base_trace.BaseTrace):

            if self._sanity_check:
                trace_length = other.get_number_of_samples() / other.get_sampling_rate()
                time_delay = self._get_time_delay()
                if time_delay > trace_length / 2:
                    logging.warning("The time shift appiled by the response is larger than half the trace length:\n\t"
                                    f"{time_delay:.2f} vs {trace_length:.2f}")

            spec = other.get_frequency_spectrum()
            freqs = other.get_frequencies()
            spec *= self(freqs)  # __call__
            other.add_trace_start_time(np.sum(self.__time_delays))
            other.set_frequency_spectrum(spec, sampling_rate="same")
            return other

        elif isinstance(other, np.ndarray):
            raise TypeError("You try to multiply a `Response` object with a numpy array, "
                            "only `Response` or `BaseTrace` is allowed. "
                            "Did you call `get_trace()` or `get_frequency_spectrum()` on `BaseTrace`?")

        else:
            raise TypeError(f"Response multiplied with unknown type: {type(other)}")

    def __rmul__(self, other):
        """ Same as mul """
        return self.__mul__(other)

    def __str__(self):
        ampl = 20 * np.log10(np.abs(self(0.5 * units.GHz)))
        return "Response of " + ", ".join([f"{name} ({weight})" for name, weight in zip(self.get_names(), self.__weights)]) \
            + f": |R(0.5 GHz)| = {ampl:.2f} dB (amplitude) ({np.sum(self.__time_delays):.2f} ns)"

    def plot(self, show=False, in_dB=True):
        import matplotlib.pyplot as plt

        freqs = np.linspace(0, 1.4) * units.GHz

        fig, ax = plt.subplots()
        for gain, weight, name in zip(self.__gains, self.__weights, self.__names):
            _gain = gain(freqs)

            if in_dB:
                mask = _gain > 0  # to avoid RunTime warning
                ax.plot(freqs[mask] / units.MHz, weight * 20 * np.log10(_gain[mask]), label=name)
            else:
                ax.plot(freqs / units.MHz, _gain, label=name)

        _gain = np.abs(self(freqs))
        if in_dB:
            mask = _gain > 0  # to avoid RunTime warning
            ax.plot(freqs[mask] / units.MHz, 20 * np.log10(_gain[mask]), color="k", label="total")
        else:
            ax.plot(freqs / units.MHz, _gain, color="k", label="total")

        ax.set_xlabel("frequency / MHz")
        if in_dB:
            ax.set_ylabel("gain / dB")
        else:
            ax.set_ylabel("gain")
            ax.set_yscale("log")

        ax.legend()

        if show:
            plt.show()
        else:
            return fig, ax

    def get_time_delay(self):
        """ Get time delay from DB """
        return np.sum(self.__time_delays)

    def _get_time_delay(self):
        """
        Calculate time delay from phase of the stored complex response function.
        This is not the time delay which is stored in the DB and which is used in
        the `__init__()` to normalize the reponse function. Rather, its the remaining
        group delay.

        The time delay is calculated as the mean between 195 and 205 MHz.

        Returns
        -------

        time_delay1 : float
            The time delay at ~ 200 MHz
        """

        freqs = np.arange(0.05, 1.2, 0.001) * units.GHz

        response = self(freqs)

        delta_freq = np.diff(freqs)

        phase = np.angle(response)
        time_delay = -np.diff(np.unwrap(phase)) / delta_freq / 2 / np.pi

        mask = np.all([195 * units.MHz < freqs, freqs < 250 * units.MHz], axis=0)[:-1]
        time_delay1 = np.mean(time_delay[mask])

        # fit the unwrapped phase with a linear function
        popt = np.polyfit(freqs, np.unwrap(phase), 1)
        time_delay2 = -popt[0] / (2 * np.pi)

        if np.abs(time_delay1 - time_delay2) > 0.1:
            logging.warning("Calculation of time delay. The two methods yield different results: "
                            f"{time_delay1:.1f} / {time_delay2:.1f} for {self.get_names()}. Return the former...")

        return time_delay1



if __name__ == "__main__":

    from NuRadioReco.detector import detector

    det = detector.Detector(source="rnog_mongo", log_level=logging.DEBUG, always_query_entire_description=False,
                            database_connection='RNOG_public', select_stations=24)

    det.update(datetime.datetime(2023, 8, 2, 0, 0))
