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
    The user can define a default station and a default channel. If any
    information is missing, the value stored in the default station or default
    channel will be used instead.
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
        #if self.__current_time is None:
        #    raise ValueError("Detector time is not set. The detector time has to be set using the Detector.update() function before it can be used.")
        self.__default_station = self._stations.get((Station.station_id == self.__default_station_id))
    def get_station(self, station_id):
        return self.__query_station(station_id)
    def __query_station(self, station_id):
        Station = Query()
        res = self._stations.get((Station.station_id == station_id))
        if(res is None):
            logger.error("query for station {} returned no results".format(station_id))
            raise LookupError("query for station {} returned no results".format(station_id))
        for key in self.__default_station.keys():
            if key not in res.keys():
                res[key] = self.__default_station[key]
        return res
