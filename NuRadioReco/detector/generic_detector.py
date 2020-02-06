import os
import astropy.time
import logging
from tinydb import TinyDB, Query
from tinydb_serialization import SerializationMiddleware
from tinydb.storages import MemoryStorage
from tinydb_serialization import Serializer
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
    def __init__(self, json_filename, default_station, default_channel=None, source='json', dictionary=None, assume_inf=True):
        """
        Initialize the stations detector properties.

        Parameters
        ----------
        json_filename : str
            the path to the json detector description file (if first checks a path relative to this directory, then a
            path relative to the current working directory of the user)
            default value is 'ARIANNA/arianna_detector_db.json'
        default_station:
            ID of the station that should be used as the default station. The default station needs to have a complete detector
            description. If a property is missing in any of the other stations, the value from the default station will be used instead
        default_channel:
            ID of the channel that should be used as the default channel. This channel has to be part of the default station and have a
            complete detector description. If a property is missing in any of the other channels, the value from the default channel
            will be used instead.
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
            Default to True, if true forces antenna madels to have infinite boundary conditions, otherwise the antenna madel will be determined by the station geometry.
        """
        super(GenericDetector, self).__init__(source, json_filename, dictionary, assume_inf)

        self.__default_station_id = default_station
        self.__default_channel_id = default_channel
        self.__station_changes_for_event = []
        self.__run_number = None
        self.__event_id = None
        if not self.has_station(self.__default_station_id):
            raise ValueError('The default station {} was not found in the detector description'.format(self.__default_station_id))

        Station = Query()
        self.__default_station = self._stations.get((Station.station_id == self.__default_station_id))

        if default_channel is not None:
            Channel = Query()
            self.__default_channel = self._channels.get((Channel.station_id == default_station)&(Channel.channel_id == default_channel))
            if self.__default_channel is None:
                raise ValueError('The default channel {} of station {} was not found in the detector description'.format(default_channel, self.__default_station_id))
        else:
            self.__default_channel = None

    def _get_station(self, station_id):
        if(station_id not in self._buffered_stations.keys()):
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
        if(res is None and not raw):
            logger.error("query for station {} returned no results".format(station_id))
            raise LookupError("query for station {} returned no results".format(station_id))
        if not raw:
            for key in self.__default_station.keys():
                if key not in res.keys():
                    #   if a property is missing, we use the value from the default station instead
                    res[key] = self.__default_station[key]
        return res

    def _query_channels(self, station_id, raw=False):
        Channel = Query()
        res = self._channels.search((Channel.station_id == station_id))
        if not raw:
            if len(res) == 0:
                default_channels = self._channels.search((Channel.station_id == self.__default_station_id))
                res = []
                for channel in default_channels:
                    new_channel = copy.copy(channel)
                    new_channel['station_id'] = station_id
                    res.append(new_channel)
            if self.__default_channel is not None:
                for channel in res:
                    for key in self.__default_channel.keys():
                        if key not in channel.keys() and key != 'station_id':
                            channel[key] = self.__default_channel[key]
        return res

    def _buffer(self, station_id):
        self._buffered_stations[station_id] = self._query_station(station_id)
        channels = self._query_channels(station_id)
        self._buffered_channels[station_id] = {}
        for channel in channels:
            self._buffered_channels[station_id][channel['channel_id']] = channel

    def add_generic_station(self, station_dict):
        """
        Add a generic station to the detector. The station is treated like a
        generic station in the original detector description file, i.e. all missing
        properties and all channels will be taken from the default station.
        If a station with the same ID already exists, this function does nothing.

        Parameters
        --------------
        station_dict: dictionary
            dictionary containing the station properties. Needs to at least include
            a station_id, any other missing parameters will be taken from the
            default station
        """
        if station_dict['station_id'] in self._buffered_stations.keys():
            logger.warning('Station with ID {} already exists in buffer. Cannot add station with same ID'.format(station_dict['station_id']))
            return

        for key in self.__default_station.keys():
            if key not in station_dict.keys():
                station_dict[key] = self.__default_station[key]
        self._buffered_stations[station_dict['station_id']] = station_dict

        if self.__default_station_id not in self._buffered_channels.keys():
            self._buffer(self.__default_station_id)
        self._buffered_channels[station_dict['station_id']] = {}

        for i_channel, channel in self._buffered_channels[self.__default_station_id].items():
            new_channel = copy.copy(channel)
            new_channel['station_id'] = station_dict['station_id']
            self._buffered_channels[station_dict['station_id']][channel['channel_id']] = new_channel

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
            if (change['run_number'] == run_number and
            change['event_id'] == event_id):
                if station_id is None or change['station_id'] == station_id:
                    changes.append(change)
        return changes

    def set_event(self,run_number, event_id):
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

    def get_default_station(self):
        """
        Get the properties of the default station
        """
        return self.__default_station

    def get_default_station_id(self):
        """
        Get the ID of the default station
        """
        return self.__default_station_id

    def get_default_channel(self):
        """
        Get the properties of the default channel
        """
        return self.__default_channel

    def get_default_channel_id(self):
        """
        Get the ID of the default channel
        """
        return self.__default_channel_id

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
        return res != None

    # overwrite update function to do nothing
    def update(self, time):
        return
