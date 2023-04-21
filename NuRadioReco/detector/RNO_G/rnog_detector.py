import astropy
import datetime
import numpy as np

from radiotools import helper
from scipy import interpolate

from NuRadioReco.detector.db_mongo_read import Database
import NuRadioReco.framework.base_trace
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
        Get most basic channel information from buffer. If not in buffer query DB for all channels. 
        In particular this will query and buffer position information  
                
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
        print(channel_info)
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
        
        (orientation_theta, orientation_phi, rotation_theta, rotation_phi): tuble of floats
        """
        
        channel_info = self.get_channel_from_buffer(station_id, channel_id)
        orientation = channel_info['channel_position']["orientation"]
        rotation = channel_info['channel_position']["rotation"]

        return orientation["theta"], orientation["phi"], rotation["theta"], rotation["phi"]
    
    
    def get_channel_signal_chain(self, station_id, channel_id):
        
        if keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id, 'channel_signal_chain']):
            signal_chain = self.__db.get_channels_signal_chain(station_id, channel_id=channel_id)
            
            if "channels" not in self.__buffered_stations:
                self.__buffered_stations[station_id]["channels"] = collections.defaultdict(dict)
            
            self.__buffered_stations[station_id]["channels"][channel_id]["channel_signal_chain"] = signal_chain[channel_id]
            
        return self.__buffered_stations[station_id]["channels"][channel_id]["channel_signal_chain"]
    
    
    def get_signal_chain_response(self, station_id, channel_id):
        
        signal_chain_dict = self.get_channel_signal_chain(station_id, channel_id)
        
        if keys_not_in_dict(self.__buffered_stations, 
                [station_id, "channels", channel_id, 'channel_signal_chain', 'measurements_components']):
            measurement_components_dic = self.__db.get_sig_chain_component_measurements(signal_chain_dict)
            
            self.__buffered_stations[station_id]["channels"][channel_id] \
                ["channel_signal_chain"]['measurements_components'] = measurement_components_dic
            
        else:
            measurement_components_dic = self.__buffered_stations[station_id]["channels"][channel_id] \
                ["channel_signal_chain"]['measurements_components']
        
        response = None
        for key, value in measurement_components_dic.items():
            if response is None:
                response = Response(value["freq"], value["ydata"], value["y_units"], name=key)
            else:
                response *= Response(value["freq"], value["ydata"], value["y_units"], name=key)

        return response


    def get_station_id(self):
        pass      
            
    def get_number_of_channels(self, station_id):
        pass
    
    def get_channel_ids(self, station_id):
        pass
    
    def get_number_of_samples(self, station_id, channel_id):
        pass
    
    def get_sampling_frequency(self, station_id, channel_id):
        pass
    
    def get_noise_temperature(self, station_id, channel_id):
        pass
    
    def is_channel_noiseless(self, station_id, channel_id):
        pass
            
    def get_cable_delay(self, station_id, channel_id):
        pass
    
    def get_site(self, station_id):
        pass
            
    def db(self):
        return self.__db
    
    
class Response:
    
    def __init__(self, frequency, y, y_unit, name="default"):
        
        self.__names = [name]
        
        self.__frequency = np.array(frequency) * units.GHz

        if y[0] is None or y[1] is None:
            raise ValueError
        
        y_ampl = np.array(y[0])
        y_phase = np.array(y[1])
        
        if y_unit[0] == "dB":
            gain = helper.dB_to_linear(y_ampl)
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
        response = np.ones_like(freq, dtype=np.complex128)
        for gain, phase in zip(self.__gains, self.__phases):
            response *= gain(freq) * np.exp(1j * phase(freq))

        return response
    
    
    def get_names(self):
        return self.__names


    def __mul__(self, other):
        
        if isinstance(other, Response):
            self.__names += other.__names
            self.__gains += other.__gains
            self.__phases += other.__phases
            return self
       
        elif isinstance(other, NuRadioReco.framework.base_trace):
            spec = other.get_frequency_spectrum()
            freqs = other.get_frequencies()
            spec *= self(freqs)
            return other
        
        else:
            raise TypeError
        

    def __rmul__(self, other):
        return self.__mul__(other)


    def __str__(self):
        return "Response of " + ", ".join(self.get_names())


if __name__ == "__main__":
    det = RNOG_Detector()
    
    # det.db().get_collection_information("station_position", 11, "tape_measurement")
    # print(det.get_relative_position_device(11, None))
    # print(det.get_relative_position(11, 1))
    # det.get_full_station_from_buffer(11)
    print(det.get_channel_signal_chain(11, 11))
    print(det.get_signal_chain_response(11, 11))
    # print(det.get_relative_position(11, 11))
    # print(det.get_channel_orientation(11, 11))
    # det.db().get_complete_channel_information(11, 0)
    
    # det.db().get_general_station_information("station_rnog", 11)
