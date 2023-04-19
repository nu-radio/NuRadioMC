import astropy
import datetime
import numpy as np

from NuRadioReco.detector.db_mongo_read import Database
from NuRadioReco.utilities import units
import collections
import logging
logging.basicConfig()


def keys_not_in_dict(d, keys):
    d_tmp = d
    for key in keys:
        try:
            if key not in d_tmp:
                return True
        except KeyError:
            return True
        
        d_tmp = d_tmp[key]
        
    return False



class RNOG_Detector():
    def __init__(self, database_connection='RNOG_test_public', log_level=logging.DEBUG):
        
        self.logger = logging.getLogger("RNO-G detector")
        self.logger.setLevel(log_level)
        
        self.__db = Database(database_connection=database_connection)
        
        self.__detector_time = None
        
        self.__buffered_stations = collections.defaultdict(dict)
        
        
    def update(self, time):
        """ Update detector
        
        Parameters
        ----------

        time: float 
            Unix timestamp in UTC.        
        """
        
        self.__detector_time = time
        

    def set_database_time(self, time):
        ''' Set time(stamp) for database. This affects which primary measurement is used.
        
        Parameters
        ----------
        
        time: float 
            Unix timestamp in UTC.
        '''
        self.__db.set_database_time(time)
        
        
    def get_full_station_from_buffer(self, station_id):
        if station_id in self.__buffered_stations:
            info = self.__buffered_stations[station_id]
            if info["commission_time"] > self.__detector_time or info["decommission_time"] < self.__detector_time:
                self.__buffered_stations[station_id] = self.__db.get_complete_station_information(station_id=station_id)
        else:
            self.__buffered_stations[station_id] = self.__db.get_complete_station_information(station_id=station_id)
        
        return self.__buffered_stations[station_id]
    
    
    def get_channel_from_buffer(self, station_id, channel_id):
        """
        Get channel information from buffer. If not in buffer query DB for all channels 
        
        Parameters
        ----------
        
        station_id: int
            The station id
        
        channel_id: int
            The channel id

        Returns
        -------
        
        channel_info: dict
            Position and orientation information.
        
        """
        if keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id]):

            if "channels" not in self.__buffered_stations:
                self.__buffered_stations[station_id]["channels"] = collections.defaultdict(dict)

            channel_position_dict = self.__db.get_channels_position(station_id)
            for cha_id in channel_position_dict:
                self.__buffered_stations[station_id]["channels"][cha_id]['channel_position'] = channel_position_dict[cha_id]
        
        return self.__buffered_stations[station_id]["channels"][channel_id]


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
        if keys_not_in_dict(self.__buffered_stations, [station_id, "station_position"]):
            station_position = self.__db.get_station_position(station_id)
            self.__buffered_stations[station_id]["station_position"] = station_position

        return self.__buffered_stations[station_id]["station_position"]["position"]
    
    
    def get_relative_position(self, station_id, channel_id):
        """
        Get the relative position of a specific channel/antenna or device with respect to the station center

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
        channel_info = self.get_channel_from_buffer(station_id, channel_id)
                            
        return channel_info["channel_position"]['position']


    def get_relative_position_device(self, station_id, device_id):
        """
        Get the relative position of a specific channel/antenna or device with respect to the station center

        Parameters
        ----------
        
        station_id: int
            The station id
            
        device_id: str
            Device name

        Returns
        -------
        
        pos: np.array(3,)
            3-dim array of relative station position
        """
        if keys_not_in_dict(self.__buffered_stations, [station_id, "devices", device_id]):
            
            if "devices" not in self.__buffered_stations[station_id]:
                self.__buffered_stations[station_id]["devices"] = collections.OrderedDict()

            device_position_information = self.__db.get_collection_information('device_position', station_id)
            
            if device_position_information is None:
                return None
            else:
                #TODO: Check if the code actually works
                for ele in device_position_information:
                    info = ele["measurements"]
                    self.__buffered_stations[station_id]["devices"][info["device_id"]] = info
                
        return self.__buffered_stations[station_id]["devices"][device_id]["position"]


    def get_channel_orientation(self, station_id, channel_id):
        """
        Returns the orientation of a specific channel/antenna

        Parameters
        ----------
        
        station_id: int
            The station id
        
        channel_id: int
            The channel id

        Returns
        -------
        
        tuple of floats
            * orientation theta: orientation of the antenna, as a zenith angle (0deg is the zenith, 180deg is straight down); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
            * orientation phi: orientation of the antenna, as an azimuth angle (counting from East counterclockwise); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
            * rotation theta: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
            * rotation phi: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
        """
        channel_info = self.get_channel_from_buffer(station_id, channel_id)
        orientation = channel_info['channel_position']["orientation"]
        rotation = channel_info['channel_position']["rotation"]

        return orientation["theta"], orientation["phi"], rotation["theta"], rotation["phi"]
    
    def get_channel_signal_chain(self, station_id, channel_id):
        pass
    
    
    def db(self):
        return self.__db


if __name__ == "__main__":
    det = RNOG_Detector()
    
    # det.db().get_collection_information("station_position", 11, "tape_measurement")
    # print(det.get_relative_position_device(11, None))
    # print(det.get_relative_position(11, 1))
    # det.get_full_station_from_buffer(11)
    det.get_absolute_position(11)
    print(det.get_relative_position(11, 11))
    print(det.get_channel_orientation(11, 11))
    # det.db().get_complete_channel_information(11, 0)
    
    # det.db().get_general_station_information("station_rnog", 11)
