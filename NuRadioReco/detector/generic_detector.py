import os
import astropy.time
import logging
from tinydb import TinyDB, Query
from tinydb_serialization import SerializationMiddleware
from tinydb.storages import MemoryStorage
from tinydb_serialization import Serializer
import NuRadioReco.detector.detector
from NuRadioReco.detector.detector import DateTimeSerializer

logger = logging.getLogger('genericDetector')
logging.basicConfig()

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
    This detector only accepts json detector descriptions.
    """
    def __init__(self, json_filename, default_station, default_channel=None, assume_inf=True):
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
        assume_inf : Bool
            Default to True, if true forces antenna madels to have infinite boundary conditions, otherwise the antenna madel will be determined by the station geometry.
        """
        super(GenericDetector, self).__init__('json', json_filename, assume_inf)
        self.__default_station_id = default_station
        if not self.has_station(self.__default_station_id):
            raise ValueError('The default station {} was not found in the detector description'.format(self.__default_station_id)) 
        Station = Query()
        self.__default_station = self._stations.get((Station.station_id == self.__default_station_id))

    def _query_station(self, station_id):
        Station = Query()
        res = self._stations.get((Station.station_id == station_id))
        if(res is None):
            logger.error("query for station {} returned no results".format(station_id))
            raise LookupError("query for station {} returned no results".format(station_id))
        for key in self.__default_station.keys():
            if key not in res.keys():
                #   if a property is missing, we use the value from the default station instead
                res[key] = self.__default_station[key]
        return res

    def _query_channels(self, station_id):
        Channel = Query()
        res = self._channels.search((Channel.station_id == station_id))
        if len(res) == 0:
            res = self._channels.search((Channel.station_id == self.__default_station_id))
        return res