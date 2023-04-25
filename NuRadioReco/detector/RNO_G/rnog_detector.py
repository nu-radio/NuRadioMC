# Keep that the first import
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s:%(name)s:%(funcName)s : %(message)s", datefmt="%H:%M:%S")

import astropy
import datetime
import numpy as np

from radiotools import helper
from scipy import interpolate

from NuRadioReco.detector.RNO_G.db_mongo_read import Database, filtered_keys
import NuRadioReco.framework.base_trace
from NuRadioReco.utilities import units
import collections


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
    def __init__(self, database_connection='RNOG_test_public', log_level=logging.DEBUG, over_write_handset_values={}):
        
        self.logger = logging.getLogger("rno-g-detector")
        self.logger.setLevel(log_level)

        self.__db = Database(database_connection=database_connection)
        
        self.__detector_time = None
        
        self.__buffered_stations = collections.defaultdict(dict)
        
        # Define default values for parameter not (yet) implemented in DB. Those values are taken for all channels.
        self.__default_values = {
            "noise_temperature": 300 * units.kelvin,
            "sampling_frequency": 3.2 * units.GHz,
            "number_of_samples": 2048,
            "is_noiseless": False
        }
        
        self.__default_values.update(over_write_handset_values)
        
        info = "\nUsing the following hand-set values:"
        for key, value in self.__default_values.items():
            info += f"\n\t{key:<20}: {value}"
        
        self.logger.info(info)
        
        
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
    
    
    def get_channel_from_buffer(self, station_id, channel_id, with_position=False, with_signal_chain=False):
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
        if not self._has_valid_parameter_in_buffer([station_id, "channels", channel_id]):

            if "channels" not in self.__buffered_stations:
                self.__buffered_stations[station_id]["channels"] = collections.defaultdict(dict)
                
            time_filter = [
                {"$match": {
                    'id': station_id,
                    'channel_id': channel_id,
                    'commission_time': {"$lte": self.__detector_time},
                    'decommission_time': {"$gte": self.__detector_time}}}]

            # get all stations which fit the filter
            channel_information = list(self.__db.db["station_rnog"].aggregate(time_filter))
            print(channel_information, channel_information[0].keys())
            sys.exit()

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
        
        # if keys_not_in_dict(self.__buffered_stations, [station_id, "channels", channel_id, 'channel_signal_chain']):
        if not self._has_valid_parameter_in_buffer([station_id, "channels", channel_id, 'channel_signal_chain']):
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


    def get_station_ids(self):
        """ 
        Returns a list of all commissioned station ids. Queries basic station information of all stations
        and add stations to buffer.
        
        Simple command to get all configured station ids (without checking commission time):
            station_ids = self.__db.get_station_ids("station_rnog")
        
        What is missing sofar: When do we need to update the buffer.
        
        Returns
        -------
        
        ids: list
            List of all station ids.
        """
        
        # I.e., when the buffer is empty
        if not len(self.__buffered_stations.keys()):

            time_filter = [{"$match": {
                'commission_time': {"$lte": self.__detector_time},
                'decommission_time': {"$gte": self.__detector_time}}}]

            # get all stations which fit the filter
            stations_information = list(self.__db.db["station_rnog"].aggregate(time_filter))
            
            for station_info in stations_information:
                station_id = station_info["id"]
                for key in filtered_keys(station_info, "id"):
                    self.__buffered_stations[station_id][key] = station_info[key]
        
        return list(self.__buffered_stations.keys())
    
    
    def _has_valid_parameter_in_buffer(self, key_list):
        """
        This function first checks if a parameter which is specified with a key_list (e.g. [station_id, "channels", channel_id, "channel_position"])
        exists in the buffer and second if it is still valid base on detector time (i.e., if corresponding station and channel are still commission).
        Remove station or channel from buffer if not commission.
        
        Parameters
        ----------
        
        key_list: list
            Specifies a parameter (see example above)
        
        Returns
        -------
        
        has_parameter:  bool
            True if key_list exists and is valid
        """
        
        if keys_not_in_dict(self.__buffered_stations, key_list):
            self.logger.debug("Parameter not in buffer: " + " / ".join([str(x) for x in key_list]))
            return False
        
        station_info = self.__buffered_stations[key_list[0]]
        
        if station_info["commission_time"] > self.__detector_time or station_info["decommission_time"] < self.__detector_time:
            self.logger.debug(f"Station {key_list[0]} not commission anymore at {self.__detector_time}. Remove from buffer ...")
            self.__buffered_stations[key_list[0]] = {}  # clean buffer
            return False
        
        if key_list[1] == "channels" and len(key_list) >= 3:
            channel_info = station_info["channels"][key_list[2]]
            if station_info["commission_time"] > self.__detector_time or station_info["decommission_time"] < self.__detector_time:
                self.logger.debug(f"Channel {key_list[2]} (of Station {key_list[0]}) not commission anymore at {self.__detector_time}. Remove from buffer ...")
                self.__buffered_stations[key_list[0]]["channels"][key_list[2]] = {}  # clean buffer
                return False

        self.logger.debug("Parameter in buffer and valid: " + " / ".join([str(x) for x in key_list]))
        return True
    
    
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
        self.logger.debug(f"Checking for station {station_id} ...")
        if station_id not in self.__buffered_stations:
            self.logger.debug("Query database.")

            # ... if not check DB (this will also store them in self.__buffered_stations)
            station_ids = self.get_station_ids()

            return station_id in station_ids
        else:
            self.logger.debug(f"Found station {station_id} in buffer.")
            return True


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
            err = f"Station id {station_id} not commission at {self.__detector_time}"
            self.logger.error(err)
            raise ValueError(err)
            
        # now station is in buffer (or an error was raised)
        channels = self.__buffered_stations[station_id]["channels"]
                        
        return len(channels)
    
    
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
            err = f"Station id {station_id} not commission at {self.__detector_time}"
            self.logger.error(err)
            raise ValueError(err)
            
        # now station is in buffer (or an error was raised)
        channels = self.__buffered_stations[station_id]["channels"]

        return [ele["id"] for ele in channels]
    
    
    def get_number_of_samples(self, station_id, channel_id):
        """ Get number of samples per station / channel """
        number_of_samples = self.__default_values["number_of_samples"]
        self.logger.warn(f"Return a hard-coded value of {number_of_samples} samples. This information is not (yet) implemented in the DB.")
        return number_of_samples
    
    
    def get_sampling_frequency(self, station_id, channel_id):
        """ Get sampling frequency per station / channel """
        sampling_frequency = self.__default_values["sampling_frequency"]
        self.logger.warn(f"Return a hard-coded value of {sampling_frequency / units.GHz} GHz. This information is not (yet) implemented in the DB.")
        return  sampling_frequency
    
    
    def get_noise_temperature(self, station_id, channel_id):
        """ Get noise temperture per station / channel """
        noise_temperature = self.__default_values["noise_temperature"]
        self.logger.warn(f"Return a hard-coded value of {noise_temperature / units.kelvin} K. This information is not (yet) implemented in the DB.")
        return noise_temperature
    
    
    def is_channel_noiseless(self, station_id, channel_id):
        is_noiseless = self.__default_values["is_noiseless"]
        self.logger.warn(f"Return a hard-coded value of {is_noiseless}. This information is not (yet) implemented in the DB.")
        return is_noiseless
    

    def get_cable_delay(self, station_id, channel_id):
        self.logger.error("The cable delay is not yet implemented in the DB.")
        raise NotImplementedError("The cable delay is not yet implemented in the DB.")
    
    
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
            response *= gain(freq / units.GHz) * np.exp(1j * phase(freq / units.GHz))

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
        return "Response of " + ", ".join(self.get_names())


if __name__ == "__main__":
    det = RNOG_Detector(log_level=logging.DEBUG, over_write_handset_values={"sampling_frequency": 2.4 * units.GHz})
    # det.update(datetime.datetime(2022, 8, 2, 0, 0))
    
    # det.get_full_station_from_buffer(11)
    # det.update(datetime.datetime(2022, 8, 2, 0, 0))

    # s = det.get_channel_signal_chain(11, 11)
    # print(s.keys())
    
    time_filter = [
    {"$match": {
        'id': 11,
        'channels.id': 11,
        'commission_time': {"$lte": datetime.datetime(2022, 8, 2, 0, 0)},
        'decommission_time': {"$gte": datetime.datetime(2022, 8, 2, 0, 0)}}}]

    # get all stations which fit the filter
    channel_information = list(det.db().db["station_rnog"].aggregate(time_filter))
    print(len(channel_information[0]["channels"]))