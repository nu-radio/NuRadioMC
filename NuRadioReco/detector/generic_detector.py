import logging
import json
from tinydb import Query
from tinydb_serialization import SerializationMiddleware
import NuRadioReco.detector.detector
from NuRadioReco.detector.detector import DateTimeSerializer
import copy

logger = logging.getLogger('NuRadioReco.genericDetector')

serialization = SerializationMiddleware()
serialization.register_serializer(DateTimeSerializer(), 'TinyDate')


class GenericDetector(NuRadioReco.detector.detector.Detector):
    """
    Used the same way as the main detector class, but works with incomplete
    detector descriptions.
    The user can define a default station. If any property is missing from one
    of the other station, the value from the default station will be used instead.
    If no channels are specified for a station, the channels from the default station
    will be used instead.
    For cases when the station design has a lot of channels, a default channel can
    also be defined. The default channel has to be part of the default station.
    It works the same way as the default station: Any property missing from one
    of the other channels will be taken from the default channel.
    The GenericDetector also ignores commission and decommision times and should
    therefore not be used for real data, but only for simulation studies.
    This detector only accepts json detector descriptions or dictionary.
    """

    def __init__(self, json_filename, default_station=None, default_channel=None, default_device=None, source='json',
                 dictionary=None,
                 assume_inf=True, antenna_by_depth=False):
        """
        Initialize the stations detector properties.

        Parameters
        ----------
        json_filename : str
            the path to the json detector description file (if first checks a path relative to this directory, then a
            path relative to the current working directory of the user)
            default value is 'ARIANNA/arianna_detector_db.json'
        default_station:
            This option is deprecated.
            ONLY in case no 'reference_station' is set in the station parameters of the detector description,
            the 'default_station' is used as reference for the time being.
            ID of the station that should be used as the default station.
            The default station needs to have a complete detector description. If a property is missing in any of the
            other stations, the value from the default station will be used instead.
        default_channel:
            This option is deprecated.
            ONLY in case no 'reference_channel' is set in the channel parameters of the detector description,
            the 'default_channel' is used as reference for the time being.
            ID of the channel that should be used as the default channel. This channel has to be part of the reference
            station and have a complete detector description. If a property is missing in any of the other channels,
            the value from the default channel will be used instead.
        default_device:
            This option is deprecated.
            ONLY in case no 'reference_device' is set in the device parameters of the detector description,
            the 'default_device' is used as reference for the time being.
            ID of the device that should be used as the default device. This channel has to be part of the reference
            station and have a complete detector description. If a property is missing in any of the other devices,
            the value from the default device will be used instead.
        source: str
            'json' or 'dictionary'
            default value is 'json'
            If 'json' is passed, the JSON dictionary at the location specified
            by json_filename will be used
            If 'dictionary' is passed, the dictionary specified by the parameter
            'dictionary' will be used
        dictionary: dict
            If 'dictionary' is passed to the parameter source, the dictionary
            passed to this parameter will be used for the detector description.
        assume_inf : Bool
            Default to True, if true forces antenna madels to have infinite boundary conditions, otherwise the antenna
            madel will be determined by the station geometry.
        antenna_by_depth: bool (default True)
            if True the antenna model is determined automatically depending on the depth of the antenna.
            This is done by appending e.g. '_InfFirn' to the antenna model name.
            if False, the antenna model as specified in the database is used.
        """

        if (default_station is None) and (default_channel is None) and (default_device is None):
            # load detector
            super(GenericDetector, self).__init__(source=source, json_filename=json_filename,
                                              dictionary=dictionary, assume_inf=assume_inf,
                                              antenna_by_depth=antenna_by_depth)
        else:
            if source == "json":
                # load json as dictionary and pass that to the detector, this is needed in order to not overwrite the json
                # when updating the table to include reference_station/channel/device
                with open(json_filename, "r") as json_input:
                    dictionary = json.load(json_input)
            super(GenericDetector, self).__init__(source="dictionary", json_filename=None,
                                              dictionary=dictionary, assume_inf=assume_inf,
                                              antenna_by_depth=antenna_by_depth)

            if default_station is not None:
                logger.warning("DeprecationWarning: replace default_station by setting a 'reference_station' for each station in the detector description. This allows to define multiple default station types")
                # fill default station info into 'reference_station' field for all stations in detector
                for sta in self._stations:
                    if 'reference_station' in sta.keys():
                        logger.warning(f"Station already has a reference station {sta['reference_station']}. Ignoring deprecated 'default_station'")
                    else:
                        logger.warning(f"Setting deprecated 'default_station' as reference station ({default_station}) as requested")
                        Station = Query()
                        self._stations.update({'reference_station': default_station}, (Station.station_id == sta["station_id"]))

            if default_channel is not None:
                logger.warning("DeprecationWarning: replace default_channel by setting a 'reference_channel' for each channel in the detector description. This allows to define multiple default channel types")
                # fill default channel info into 'reference_channel' field for all channels in detector
                for chan in self._channels:
                    if 'reference_channel' in chan.keys():
                        logger.warning(f"Channel already has a reference channel {chan['reference_channel']}. Ignoring deprecated 'default_channel'")
                    else:
                        logger.warning(f"Setting deprecated 'default_channel' as reference channel ({default_channel}) as requested")
                        Channel = Query()
                        self._channels.update({'reference_channel': default_channel}, (Channel.station_id == chan["station_id"]) & (Channel.channel_id == chan["channel_id"]))

            if default_device is not None:
                logger.warning("DeprecationWarning: replace default_device by setting a 'reference_device' for each device in the detector description. This allows to define multiple default device types")
                # fill default device info into 'reference_device' field for all devices in detector
                for dev in self._buffered_devices:
                    if 'reference_device' in dev.keys():
                        logger.warning(f"Device already has a reference device {dev['reference_device']}. Ignoring deprecated 'default_device'")
                    else:
                        logger.warning(f"Setting deprecated 'default_device' as reference device ({default_device}) as requested")
                        Device = Query()
                        self._devices.update({'reference_device': default_device}, (Device.station_id == dev["station_id"]) & (Device.device_id == dev["device_id"]))
            self.__default_station = default_station
        # TODO maybe these dicts/lists can be omitted
        # a lookup with one reference station for each station in the detector description
        self.__lookuptable_reference_station = {}
        self.__reference_station_ids = []
        self.__reference_device_ids = {}
        self.__reference_channel_ids = {}
        # TODO maybe these dicts.lists can be omitted
        # add all stations to the lookup
        for sta in self._stations.all():
            self._update_reference_station_lookup(sta)
        self.__reference_stations = {}
        Station = Query()
        for reference_station_id in self.__reference_station_ids:
            self.__reference_stations[reference_station_id] = self._stations.get((Station.station_id == reference_station_id))

        self.__station_changes_for_event = []
        self.__run_number = None
        self.__event_id = None

        # check if all  reference stations, reference_channels, and reference_devices are present
        Station = Query()
        for sta in self._stations.all():
            if "reference_station" in sta:
                ref = self._stations.get(
                  (Station.station_id == sta["reference_station"]))
                if ref is None:
                    raise ValueError(
                        'The reference station {} was not found in the detector description'.format(
                            ref["reference_station"]))

        Channel = Query()
        for chan in self._channels.all():
            if "reference_channel" in chan:
                ref = self._channels.get(
                  (Channel.station_id == chan["station_id"]) & (Channel.channel_id == chan["reference_channel"]))
                if ref is None:
                    raise ValueError(
                        'The reference channel {} of station {} was not found in the detector description'.format(
                            ref["reference_channel"], ref["station_id"]))

        Device = Query()
        for dev in self._devices.all():
              if "reference_device" in dev:
                  ref = self._devices.get(
                    (Device.station_id == dev["station_id"]) & (Device.channel_id == dev["reference_device"]))
                  if ref is None:
                      raise ValueError(
                          'The reference device {} of station {} was not found in the detector description'.format(
                              ref["reference_device"], ref["station_id"]))


    def _get_station(self, station_id):
        if station_id not in self._buffered_stations.keys():
            self._buffer(station_id)
        res = copy.copy(self._buffered_stations[station_id])
        if self.__run_number is not None and self.__event_id is not None:
            for change in self.__station_changes_for_event:
                if change['station_id'] == station_id and change['run_number'] == self.__run_number and change['event_id'] == self.__event_id:
                    for name, value in change['properties'].items():
                        res[name] = value
        return res

    def _query_station(self, station_id, raw=False):
        Station = Query()
        res = self._stations.get((Station.station_id == station_id))
        if res is None and not raw:
            logger.error("query for station {} returned no results".format(station_id))
            raise LookupError("query for station {} returned no results".format(station_id))
        if not raw:
            if "reference_station" in res.keys():
                ref = self._stations.get((Station.station_id == res["reference_station"]))
                for key in ref.keys():
                    if key not in res.keys():
                        # if a property is missing, we use the value from the reference station instead
                        res[key] = ref[key]
        return res

    def _query_channels(self, station_id, raw=False):
        Station = Query()
        sta = self._stations.get((Station.station_id == station_id))
        reference_station_id = station_id
        if "reference_station" in sta:
            reference_station_id = sta["reference_station"]

        Channel = Query()
        res = self._channels.search((Channel.station_id == station_id))

        if not raw:
            # if there are NO channels in this station defined, look up devices from the reference station
            if len(res) == 0:
                reference_channels = self._channels.search((Channel.station_id == reference_station_id))
                res = []
                for channel in reference_channels:
                    new_channel = copy.copy(channel)
                    new_channel['station_id'] = station_id
                    res.append(new_channel)

            # now we look if there are reference fields to fill. Will use reference_station_id, which is either the station or the reference
            for channel in res:
                if 'reference_channel' in channel:
                    # add to dictionary to keep track of reference channels TODO this is not really needed?
                    self.__reference_channel_ids[(station_id, channel['channel_id'])] = channel['reference_channel']
                    # there is a reference, so we have to get it
                    ref_chan = self._channels.get(
                            (Channel.station_id == reference_station_id) & (Channel.channel_id == channel['reference_channel']))
                    for key in ref_chan.keys():
                        if key not in channel.keys() and key != 'station_id' and key != 'channel_id':
                            channel[key] = ref_chan[key]
        return res

    def _query_devices(self, station_id, raw=False):
        # if the station has a reference, take this one to take the devices from
        Station = Query()
        sta = self._stations.get((Station.station_id == station_id))
        reference_station_id = station_id
        if "reference_station" in sta:
            reference_station_id = sta["reference_station"]

        Device = Query()
        res = self._devices.search((Device.station_id == station_id))

        if not raw:
            # if there are NO devices in this station defined, look up devices from the reference station
            if len(res) == 0:
                reference_devices = self._devices.search((Device.station_id == reference_station_id))
                res = []
                for device in reference_devices:
                    new_device = copy.copy(device)
                    new_device['station_id'] = station_id
                    res.append(new_device)

            # now we look if there are reference fields to fill. Will use reference_station_id, which is either the station or the reference
            for device in res:
                if 'reference_device' in device:
                    # add to dictionary to keep track of reference devices TODO this is not really needed?
                    self.__reference_device_ids[(station_id, device['device_id'])] = device['reference_device']
                    # there is a reference, so we have to get it
                    ref_dev = self._devices.get(
                            (Device.station_id == reference_station_id) & (Device.device_id == device['reference_device']))
                    for key in ref_dev.keys():
                        if key not in device.keys() and key != 'station_id' and key != 'device_id':
                            device[key] = ref_dev[key]
        return res

    def _buffer(self, station_id):
        self._buffered_stations[station_id] = self._query_station(station_id)
        channels = self._query_channels(station_id)
        self._buffered_channels[station_id] = {}
        for channel in channels:
            self._buffered_channels[station_id][channel['channel_id']] = channel
        devices = self._query_devices(station_id)
        self._buffered_devices[station_id] = {}
        for device in devices:
            self._buffered_devices[station_id][device['device_id']] = device

    def _update_reference_station_lookup(self, station_dict):
        """ add a station to the lookup of reference stations and update the list of default station ids """
        default_ids = set(self.__reference_station_ids)

        if 'reference_station' in station_dict:
            self.__lookuptable_reference_station[station_dict['station_id']] = station_dict['reference_station']
            default_ids.add(station_dict['reference_station'])
        else:
            self.__lookuptable_reference_station[station_dict['station_id']] = None

        self.__reference_station_ids = list(default_ids)

    def add_generic_station(self, station_dict):
        """
        Add a generic station to the detector. The station is treated like a
        generic station in the original detector description file, i.e. all missing
        properties and all channels will be taken from the reference station.
        If a station with the same ID already exists, this function does nothing.

        Parameters
        --------------
        station_dict: dictionary
            dictionary containing the station properties. Needs to at least include
            a station_id, any other missing parameters will be taken from the
            reference station
        """
        if "reference_station" not in station_dict and self.__default_station is not None:
            logger.warning('DeprecationWarning: Generating a station via `add_generic_station` that has no "reference_station" specified.')
            logger.warning(f'DeprecationWarning: Taking the deprecated "default_station" ({self.__default_station}) as "reference_station".')
            station_dict["reference_station"] = self.__default_station


        if station_dict['station_id'] in self._buffered_stations.keys():
            logger.warning('Station with ID {} already exists in buffer. Cannot add station with same ID'.format(
                station_dict['station_id']))
            return

        if station_dict['station_id'] not in self.__lookuptable_reference_station:
            self._update_reference_station_lookup(station_dict)

        reference_station_id = self.__lookuptable_reference_station[station_dict['station_id']]
        for key in self.__reference_stations[reference_station_id].keys():
            if key not in station_dict.keys():
                station_dict[key] = self.__reference_stations[reference_station_id][key]
        self._buffered_stations[station_dict['station_id']] = station_dict

        if reference_station_id not in self._buffered_channels.keys():
            self._buffer(reference_station_id)

        self._buffered_channels[station_dict['station_id']] = {}
        for i_channel, channel in self._buffered_channels[reference_station_id].items():
            new_channel = copy.copy(channel)
            new_channel['station_id'] = station_dict['station_id']
            self._buffered_channels[station_dict['station_id']][channel['channel_id']] = new_channel

        self._buffered_devices[station_dict['station_id']] = {}
        for i_device, device in self._buffered_devices[reference_station_id].items():
            new_device = copy.copy(device)
            new_device['station_id'] = station_dict['station_id']
            self._buffered_devices[station_dict['station_id']][device['device_id']] = new_device


    def add_station_properties_for_event(self, properties, station_id, run_number, event_id):
        """
        Adds an entry to the list of event-specific changes to the detector
        description.

        Parameters
        ------------------
        properties: dictionary
            Dictionary of the properties that should be changed, with keys being
            any of the property names in the detector description and values the
            values that these properties should be changed to
        station_id: integer
            ID of the station whose properties should be changed
        run_number: integer
            Run number of the event for which the changes are valid
        event_id: integer
            Event ID of the event for which the changes are valid
        """
        self.__station_changes_for_event.append({
            'run_number': run_number,
            'event_id': event_id,
            'station_id': station_id,
            'properties': properties
        })

    def get_station_properties_for_event(self, run_number, event_id, station_id=None):
        """
        Returns all event-specific changes that have been stored in the
        detector description for a given station and event

        Parameters
        ----------------------
        run_number: integer
            Run number of the event for which the changes should be returned
        event_id: integer
            Event ID of the event for which the changes should be returned
        station_id: integer or None
            ID of the station for which the changes should be returned
            If station_id is None, changes for all stations are returned
        """
        changes = []
        for change in self.__station_changes_for_event:
            if change['run_number'] == run_number and change['event_id'] == event_id:
                if station_id is None or change['station_id'] == station_id:
                    changes.append(change)
        return changes

    def set_event(self, run_number, event_id):
        """
        Sets the run number and event ID for which the detector description
        should be returned. This is needed if event-specific changes to the
        detector description have been stored. If run_number or event_id are
        not set (or are set to None), event-specific changes to the detector
        will be ignored

        Parameters
        ------------------
        run_number: integer
            Run number of the event the detector should be set to
        event_id: integer
            ID of the event the detector should be set to
        """
        self.__run_number = run_number
        self.__event_id = event_id

    def get_reference_station(self, station_id):
        """
        Get the properties of the reference station
        """
        return self.__reference_stations[station_id]

    def get_reference_station_id(self, station_id):
        """
        Get the properties of the reference station
        """
        return self.__reference_station_ids[station_id] 

    def get_reference_stations(self):
        """
        Get the properties of the reference stations
        """
        return self.__reference_stations

    def get_default_station(self):
        """
        Get the properties of the default station
        """
        return self.__reference_stations[self.get_default_station_id()]

    def get_reference_station_ids(self):
        """
        Get the whole diectionary of reference stations
        """
        return self.__reference_station_ids

    def get_default_station_id(self):
        """
        Get the ID of the default station
        """
        if len(self.__reference_station_ids) == 1:
            # only one default station, either passed as "reference_station" in detector description or in the ini
            return self.__reference_station_ids[0]
        else:
            # more than one "reference_station" passed in the detector description, return the first one
            logger.warning(
                f'more than one station id set as "reference station": {self.__reference_station_ids},\
                continue with first entry: {self.__reference_station_ids[0]}')
            return  self.__reference_station_ids[0]

    def get_default_channel(self):
        """
        Get the properties of the default channel
        """
        logger.warning("The use of 'default_channel' is deprecated. returning None")
        return None #self.__default_channel

    def get_default_channel_id(self):
        """
        Get the ID of the default channel
        """
        logger.warning("The use of 'default_channel' is deprecated. returning None")
        return None #self.__default_channel_id


    def get_raw_station(self, station_id):
        """
        Get the properties of a station as they are in the original detector
        description, i.e. without missing properties being replaced by those
        from the default station. Event-specific changes are also ignored.

        Parameters
        --------------------------
        station_id: integer
            ID of the requested station
        """
        if station_id in self._buffered_stations.keys():
            return self._buffered_stations[station_id]
        station = self._query_station(station_id, True)
        if station is None:
            return {
                "station_id": station_id
            }
        else:
            return station

    def get_raw_channel(self, station_id, channel_id):
        """
        Get the properties of a channel as they are in the original detector
        description, i.e. without missing properties being replaced by those
        from the default channel.

        Parameters
        --------------------------
        station_id: integer
            ID of the requested channel's station
        channel_id: integer
            ID of the requested channel
        """

        if station_id in self._buffered_channels.keys():
            if channel_id in self._buffered_channels[station_id].keys():
                return self._buffered_channels[station_id][channel_id]
        channels = self._query_channels(station_id, True)
        for channel in channels:
            if channel['channel_id'] == channel_id:
                return channel
        return None

    def has_station(self, station_id):
        if station_id in self._buffered_stations.keys():
            return True
        Station = Query()
        res = self._stations.get(Station.station_id == station_id)
        return res is not None

    # overwrite update function to do nothing
    def update(self, time):
        return
