import numpy as np
from NuRadioReco.utilities import units
from radiotools import helper as hp
import os
import inspect
import logging
from tinydb import TinyDB, Query
from tinydb_serialization import SerializationMiddleware
from tinydb.storages import MemoryStorage
import astropy.time
from datetime import datetime
from tinydb_serialization import Serializer
import six  # # used for compatibility between py2 and py3
logger = logging.getLogger('NuRadioReco.detector')


class DateTimeSerializer(Serializer):
    """
    helper class to serialize datetime objects with TinyDB
    """
    OBJ_CLASS = datetime  # The class this serializer handles

    def encode(self, obj):
        return obj.strftime('%Y-%m-%dT%H:%M:%S')

    def decode(self, s):
        return datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')


serialization = SerializationMiddleware()
serialization.register_serializer(DateTimeSerializer(), 'TinyDate')


def buffer_db(in_memory, filename=None):
    """
    buffers the complet SQL database into a TinyDB object (either in memory or into a local JSON file)
    """
    logger.info("buffering SQL database on-the-fly")
    db = None
    if(in_memory):
        db = TinyDB(storage=MemoryStorage)
    else:
        db = TinyDB(filename, storage=serialization, sort_keys=True, indent=4, separators=(',', ': '))
    db.purge()

    from NuRadioReco.detector import detector_sql
    sqldet = detector_sql.Detector()
    results = sqldet.get_everything_stations()
    table_stations = db.table('stations')
    table_stations.purge()
    for result in results:
        table_stations.insert({'station_id': result['st.station_id'],
                               'commission_time': result['st.commission_time'],
                               'decommission_time': result['st.decommission_time'],
                               'station_type': result['st.station_type'],
                               'position': result['st.position'],
                               'board_number': result['st.board_number'],
                               'MAC_address': result['st.MAC_address'],
                               'MBED_type': result['st.MBED_type'],
                               'pos_position': result['pos.position'],
                               'pos_measurement_time': result['pos.measurement_time'],
                               'pos_easting': result['pos.easting'],
                               'pos_northing': result['pos.northing'],
                               'pos_altitude': result['pos.altitude'],
                               'pos_zone': result['pos.zone'],
                               'pos_site': result['pos.site']})

    table_channels = db.table('channels')
    table_channels.purge()
    results = sqldet.get_everything_channels()
    for channel in results:
        table_channels.insert({ 'station_id': channel['st.station_id'],
                                'channel_id': channel['ch.channel_id'],
                                'commission_time': channel['ch.commission_time'],
                                'decommission_time': channel['ch.decommission_time'],
                                'ant_type': channel['ant.antenna_type'],
                                'ant_orientation_phi': channel['ant.orientation_phi'],
                                'ant_orientation_theta': channel['ant.orientation_theta'],
                                'ant_rotation_phi': channel['ant.rotation_phi'],
                                'ant_rotation_theta': channel['ant.rotation_theta'],
                                'ant_position_x': channel['ant.position_x'],
                                'ant_position_y': channel['ant.position_y'],
                                'ant_position_z': channel['ant.position_z'],
                                'ant_deployment_time': channel['ant.deployment_time'],
                                'ant_comment': channel['ant.comment'],
                                'cab_length': channel['cab.cable_length'],
                                'cab_reference_measurement': channel['cab.reference_measurement'],
                                'cab_time_delay': channel['cab.time_delay'],
                                'cab_id': channel['cab.cable_id'],
                                'cab_type': channel['cab.cable_type'],
                                'amp_type': channel['amps.amp_type'],
                                'amp_reference_measurement': channel['amps.reference_measurement'],
                                'adc_id': channel['adcs.adc_id'],
                                'adc_time_delay': channel['adcs.time_delay'],
                                'adc_nbits': channel['adcs.nbits'],
                                'adc_n_samples': channel['adcs.n_samples'],
                                'adc_sampling_frequency': channel['adcs.sampling_frequency']})

    results = sqldet.get_everything_positions()
    table_positions = db.table('positions')
    table_positions.purge()
    for result in results:
        table_positions.insert({
                               'pos_position': result['pos.position'],
                               'pos_measurement_time': result['pos.measurement_time'],
                               'pos_easting': result['pos.easting'],
                               'pos_northing': result['pos.northing'],
                               'pos_altitude': result['pos.altitude'],
                               'pos_zone': result['pos.zone'],
                               'pos_site': result['pos.site']})

    logger.info("sql database buffered")
    return db


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if Singleton._instances.get(cls, None) is None:
            Singleton._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return Singleton._instances[cls]


@six.add_metaclass(Singleton)
class Detector(object):
    """
    main detector class which provides access to the detector description

    This class provides functions for all relevant detector properties.
    """

    def __init__(self, source='json', json_filename='ARIANNA/arianna_detector_db.json',
                 dictionary=None, assume_inf=True):
        """
        Initialize the stations detector properties.

        Parameters
        ----------
        source : str
            'json', 'dictionary' or 'sql'
            default value is 'json'
            if dictionary is specified, the dictionary passed to __init__ is used
            if 'sql' is specified, the file 'detector_sql_auth.json' file needs to be present in this folder that
            specifies the sql server credentials (see 'detector_sql_auth.json.sample' for an example of the syntax)
        json_filename : str
            the path to the json detector description file (if first checks a path relative to this directory, then a
            path relative to the current working directory of the user)
            default value is 'ARIANNA/arianna_detector_db.json'
        assume_inf : Bool
            Default to True, if true forces antenna madels to have infinite boundary conditions, otherwise the antenna madel will be determined by the station geometry.
        """
        if(source == 'sql'):
            self._db = buffer_db(in_memory=True)
        elif source == 'dictionary':
            self._db = TinyDB(storage=MemoryStorage)
            self._db.purge()
            stations_table = self._db.table('stations', cache_size=1000)
            for station in dictionary['stations'].values():
                stations_table.insert(station)
            channels_table = self._db.table('channels', cache_size=1000)
            for channel in dictionary['channels'].values():
                channels_table.insert(channel)
        else:
            dir_path = os.path.dirname(os.path.realpath(__file__))  # get the directory of this file
            filename = os.path.join(dir_path, json_filename)
            if(not os.path.exists(filename)):
                # try local folder instead
                filename2 = json_filename
                if(not os.path.exists(filename2)):
                    logger.error("can't locate json database file {} or {}".format(filename, filename2))
                    raise NameError
                filename = filename2
            logger.warning("loading detector description from {}".format(os.path.abspath(filename)))
            self._db = TinyDB(filename, storage=serialization,
                               sort_keys=True, indent=4, separators=(',', ': '))

        self._stations = self._db.table('stations', cache_size=1000)
        self._channels = self._db.table('channels', cache_size=1000)
        self.__positions = self._db.table('positions', cache_size=1000)

        logger.info("database initialized")

        self._buffered_stations = {}
        self.__buffered_positions = {}
        self._buffered_channels = {}
        self.__valid_t0 = astropy.time.Time('2100-1-1')
        self.__valid_t1 = astropy.time.Time('1970-1-1')

        self.__noise_RMS = None

        self.__current_time = None

        self.__assume_inf = assume_inf

    def __query_channel(self, station_id, channel_id):
        Channel = Query()
        if self.__current_time is None:
            raise ValueError("Detector time is not set. The detector time has to be set using the Detector.update() function before it can be used.")
        res = self._channels.get((Channel.station_id == station_id) & (Channel.channel_id == channel_id)
                                           & (Channel.commission_time <= self.__current_time.datetime)
                                           & (Channel.decommission_time > self.__current_time.datetime))
        if(res is None):
            logger.error("query for station {} and channel {} at time {} returned no results".format(station_id, channel_id, self.__current_time))
            raise LookupError
        return res

    def _query_channels(self, station_id):
        Channel = Query()
        if self.__current_time is None:
            raise ValueError("Detector time is not set. The detector time has to be set using the Detector.update() function before it can be used.")
        return self._channels.search((Channel.station_id == station_id)
                                           & (Channel.commission_time <= self.__current_time.datetime)
                                           & (Channel.decommission_time > self.__current_time.datetime))

    def _query_station(self, station_id):
        Station = Query()
        if self.__current_time is None:
            raise ValueError("Detector time is not set. The detector time has to be set using the Detector.update() function before it can be used.")
        res = self._stations.get((Station.station_id == station_id)
                                       & (Station.commission_time <= self.__current_time.datetime)
                                       & (Station.decommission_time > self.__current_time.datetime))
        if(res is None):
            logger.error("query for station {} at time {} returned no results".format(station_id, self.__current_time.datetime))
            raise LookupError("query for station {} at time {} returned no results".format(station_id, self.__current_time.datetime))
        return res

    def __query_position(self, position_id):
        Position = Query()
        res = self.__positions.get((Position.pos_position == position_id))
        if self.__current_time is None:
            raise ValueError("Detector time is not set. The detector time has to be set using the Detector.update() function before it can be used.")
        if(res is None):
            logger.error("query for position {} at time {} returned no results".format(position_id, self.__current_time.datetime))
            raise LookupError("query for position {} at time {} returned no results".format(position_id, self.__current_time.datetime))
        return res

    def get_station_ids(self):
        """
        returns a sorted list of all station ids present in the database
        """
        station_ids = []
        res = self._stations.all()
        if(res is None):
            logger.error("query for stations returned no results")
            raise LookupError("query for stations returned no results")
        for a in res:
            if(a['station_id'] not in station_ids):
                station_ids.append(a['station_id'])
        return sorted(station_ids)

    def _get_station(self, station_id):
        if(station_id not in self._buffered_stations.keys()):
            self._buffer(station_id)
        return self._buffered_stations[station_id]

    def get_station(self, station_id):
        return self._get_station(station_id)

    def __get_position(self, position_id):
        if(position_id not in self.__buffered_positions.keys()):
            self.__buffer_position(position_id)
        return self.__buffered_positions[position_id]

    def __get_channels(self, station_id):
        if(station_id not in self._buffered_stations.keys()):
            self._buffer(station_id)
        return self._buffered_channels[station_id]

    def __get_channel(self, station_id, channel_id):
        if(station_id not in self._buffered_stations.keys()):
            self._buffer(station_id)
        return self._buffered_channels[station_id][channel_id]

    def _buffer(self, station_id):
        self._buffered_stations[station_id] = self._query_station(station_id)
        self.__valid_t0 = astropy.time.Time(self._buffered_stations[station_id]['commission_time'])
        self.__valid_t1 = astropy.time.Time(self._buffered_stations[station_id]['decommission_time'])
        channels = self._query_channels(station_id)
        self._buffered_channels[station_id] = {}
        for channel in channels:
            self._buffered_channels[station_id][channel['channel_id']] = channel
            self.__valid_t0 = max(self.__valid_t0, astropy.time.Time(channel['commission_time']))
            self.__valid_t1 = min(self.__valid_t1, astropy.time.Time(channel['decommission_time']))

    def __buffer_position(self, position_id):
        self.__buffered_positions[position_id] = self.__query_position(position_id)

    def __get_t0_t1(self, station_id):
        Station = Query()
        res = self._stations.get(Station.station_id == station_id)
        t0 = None
        t1 = None
        if(isinstance(res, list)):
            for station in res:
                if(t0 is None):
                    t0 = station['commission_time']
                else:
                    t0 = min(t0, station['commission_time'])
                if(t1 is None):
                    t1 = station['decommission_time']
                else:
                    t1 = max(t1, station['decommission_time'])
        else:
            t0 = res['commission_time']
            t1 = res['decommission_time']
        return astropy.time.Time(t0), astropy.time.Time(t1)

    def has_station(self, station_id):
        """
        checks if a station is present in the database

        Parameters
        ----------
        station_id: int
            the station id

        Returns bool
        """
        Station = Query()
        res = self._stations.get(Station.station_id == station_id)
        return res != None

    def get_unique_time_periods(self, station_id):
        """
        returns the time periods in which the station configuration (including all channels) was constant

        Parameters
        ----------
        station_id: int
            the station id

        Returns datetime tuple
        """
        up = []
        t0, t1 = self.__get_t0_t1(station_id)
        self.update(t0)
        while True:
            if(len(up) > 0 and up[-1] == t1):
                break
            self._buffer(station_id)
            if(len(up) == 0):
                up.append(self.__valid_t0)
            up.append(self.__valid_t1)
            self.update(self.__valid_t1)
        return up

    def update(self, time):
        """
        updates the detector description to a new time

        Parameters
        ----------
        timestamp: astropy.time.Time
            the time to update the detectordescription to
            for backward compatibility datetime is also accepted, but astropy.time is prefered
        """
        if isinstance(time, datetime):
            self.__current_time = astropy.time.Time(time)
        else:
            self.__current_time = time
        logger.info("updating detector time to {}".format(self.__current_time))
        if(not ((self.__current_time > self.__valid_t0) and (self.__current_time < self.__valid_t1))):
            self._buffered_stations = {}
            self._buffered_channels = {}
            self.__valid_t0 = astropy.time.Time('2100-1-1')
            self.__valid_t1 = astropy.time.Time('1970-1-1')

    def get_channel(self, station_id, channel_id):
        """
        returns a dictionary of all channel parameters

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns: dict of channel parameters
        """
        return self.__get_channel(station_id, channel_id)

    def get_absolute_position(self, station_id):
        """
        get the absolute position of a specific station

        Parameters
        ---------
        station_id: int
            the station id

        Returns: 3-dim array of absolute station position in easting, northing and depth wrt. to snow level at
        time of measurement
        """
        res = self._get_station(station_id)
        easting, northing, altitude = 0, 0, 0
        unit_xy = units.m
        if('pos_zone' in res and res['pos_zone'] == "SP-grid"):
            unit_xy = units.feet
        if(res['pos_easting'] is not None):
            easting = res['pos_easting'] * unit_xy
        if(res['pos_northing'] is not None):
            northing = res['pos_northing'] * unit_xy
        if(res['pos_altitude'] is not None):
            altitude = res['pos_altitude']
        return np.array([easting, northing, altitude])

    def get_absolute_position_site(self, site):
        """
        get the absolute position of a specific station

        Parameters
        ---------
        site: string
            the position identifier e.g. "G"

        Returns: 3-dim array of absolute station position in easting, northing and depth wrt. to snow level at
        time of measurement
        """
        res = self.__get_position(site)
        unit_xy = units.m
        if('pos_zone' in res and res['pos_zone'] == "SP-grid"):
            unit_xy = units.feet
        easting, northing, altitude = 0, 0, 0
        if(res['pos_easting'] is not None):
            easting = res['pos_easting'] * unit_xy
        if(res['pos_northing'] is not None):
            northing = res['pos_northing'] * unit_xy
        if(res['pos_altitude'] is not None):
            altitude = res['pos_altitude'] * units.m
        return np.array([easting, northing, altitude])

    def get_relative_position(self, station_id, channel_id):
        """
        get the relative position of a specific channels/antennas with respeect to the station center

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns: 3-dim array of relative station position
        """
        res = self.__get_channel(station_id, channel_id)
        return np.array([res['ant_position_x'], res['ant_position_y'], res['ant_position_z']])

    def get_relative_positions(self, station_id):
        """
        get the relative positions of all channels/antennas with respeect to the station center

        Parameters
        ---------
        station_id: int
            the station id

        Returns: List of 3-dim array of relative station positions
        """
        res = self.__get_channels(station_id)
        positions = np.zeros((len(res), 3))
        for i, r in enumerate(res.values()):
            logger.debug("position channel {}: {:.0f}m, {:.0f}m, {:.0f}m".format(r['channel_id'], r['ant_position_x'], r['ant_position_y'], r['ant_position_z']))
            positions[i] = [r['ant_position_x'], r['ant_position_y'], r['ant_position_z']]
        return positions

    def get_site(self, station_id):
        """
        get the site where the station is deployed (e.g. MooresBay or South Pole)

        Parameters
        ---------
        station_id: int
            the station id

        Returns string
        """

        res = self._get_station(station_id)
        return res['pos_site']

    def get_number_of_channels(self, station_id):
        """
        Get the number of channels per statoin

        Parameters
        ---------
        station_id: int
            the station id

        Returns int
        """
        res = self.__get_channels(station_id)
        return len(res)

    def get_channel_ids(self, station_id):
        """
        get the channel ids of a station

        Parameters
        ---------
        station_id: int
            the station id

        Returns list of ints
        """
        channel_ids = []
        for channel in self.__get_channels(station_id).values():
            channel_ids.append(channel['channel_id'])
        return channel_ids

    def get_parallel_channels(self, station_id):
        """
        get a list of parallel antennas

        Parameters
        ---------
        station_id: int
            the station id

        Returns list of list of ints
        """
        res = self.__get_channels(station_id)
        orientations = np.zeros((len(res), 4))
        antenna_types = []
        channel_ids = []
        for iCh, ch in enumerate(res.values()):
            channel_id = ch['channel_id']
            channel_ids.append(channel_id)
            antenna_types.append(self.get_antenna_type(station_id, channel_id))
            orientations[iCh] = self.get_antenna_orientation(station_id, channel_id)
            orientations[iCh][3] = hp.get_normalized_angle(orientations[iCh][3], interval=np.deg2rad([0, 180]))
        channel_ids = np.array(channel_ids)
        antenna_types = np.array(antenna_types)
        orientations = np.round(np.rad2deg(orientations))  # round to one degree to overcome rounding errors
        parallel_antennas = []
        for antenna_type in np.unique(antenna_types):
            for u_zen_ori in np.unique(orientations[:, 0]):
                for u_az_ori in np.unique(orientations[:, 1]):
                    for u_zen_rot in np.unique(orientations[:, 2]):
                        for u_az_rot in np.unique(orientations[:, 3]):
                            mask = (antenna_types == antenna_type) \
                                     & (orientations[:, 0] == u_zen_ori) & (orientations[:, 1] == u_az_ori) \
                                     & (orientations[:, 2] == u_zen_rot) & (orientations[:, 3] == u_az_rot)
                            if(np.sum(mask)):
                                parallel_antennas.append(channel_ids[mask])
        return np.array(parallel_antennas)

    def get_cable_delay(self, station_id, channel_id):
        """
        returns the cable delay of a channel

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns float (delay time)
        """
        res = self.__get_channel(station_id, channel_id)
        return res['cab_time_delay']

    def get_cable_type_and_length(self, station_id, channel_id):
        """
        returns the cable type (e.g. LMR240) and its length

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns typle (string, float)
        """
        res = self.__get_channel(station_id, channel_id)
        return res['cab_type'], res['cab_length'] * units.m

    def get_antenna_type(self, station_id, channel_id):
        """
        returns the antenna type

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns string
        """
        res = self.__get_channel(station_id, channel_id)
        return res['ant_type']

    def get_antenna_deployment_time(self, station_id, channel_id):
        """
        returns the time of antenna deployment

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns datetime
        """
        res = self.__get_channel(station_id, channel_id)
        return res['ant_deployment_time']

    def get_antenna_orientation(self, station_id, channel_id):
        """
        returns the orientation of a specific antenna

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns typle of floats
            * orientation theta: boresight direction (zenith angle, 0deg is the zenith, 180deg is straight down)
            * orientation phi: boresight direction (azimuth angle counting from East counterclockwise)
            * rotation theta: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector in plane of tines pointing away from connector
            * rotation phi: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector in plane of tines pointing away from connector
        """
        res = self.__get_channel(station_id, channel_id)
        return np.deg2rad([res['ant_orientation_theta'], res['ant_orientation_phi'], res['ant_rotation_theta'], res['ant_rotation_phi']])

    def get_amplifier_type(self, station_id, channel_id):
        """
        returns the type of the amplifier

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns string
        """
        res = self.__get_channel(station_id, channel_id)
        return res['amp_type']

    def get_amplifier_measurement(self, station_id, channel_id):
        """
        returns a unique reference to the amplifier measurement

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns string
        """
        res = self.__get_channel(station_id, channel_id)
        return res['amp_reference_measurement']

    def get_sampling_frequency(self, station_id, channel_id):
        """
        returns the sampling frequency

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns float
        """
        res = self.__get_channel(station_id, channel_id)
        return res['adc_sampling_frequency'] * units.GHz

    def get_number_of_samples(self, station_id, channel_id):
        """
        returns the number of samples of a channel

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns int
        """
        res = self.__get_channel(station_id, channel_id)
        return res['adc_n_samples']

    def get_antenna_model(self, station_id, channel_id, zenith=None):
        """
        determines the correct antenna model from antenna type, position and orientation of antenna

        so far only infinite firn and infinite air cases are differentiated

        Parameters
        ---------
        station_id: int
            the station id
        channel_id: int
            the channel id
        zenith: float or None (default)
            the zenith angle of the incoming signal direction

        Returns string
        """
        antenna_type = self.get_antenna_type(station_id, channel_id)
        antenna_relative_position = self.get_relative_position(station_id, channel_id)

        antenna_model = ""
        if(zenith is not None and (antenna_type == 'createLPDA_100MHz')):
            if(antenna_relative_position[2] > 0):
                antenna_model = "{}_InfAir".format(antenna_type)
                if((not self.__assume_inf) and zenith < 90 * units.deg):
                    antenna_model = "{}_z1cm_InAir_RG".format(antenna_type)
            else:  # antenna in firn
                antenna_model = "{}_InfFirn".format(antenna_type)
                if((not self.__assume_inf) and zenith > 90 * units.deg):  # signal comes from below
                    antenna_model = "{}_z1cm_InFirn_RG".format(antenna_type)
                    # we need to add further distinction here
        elif(not antenna_type.startswith('analytic')):
            if(antenna_relative_position[2] > 0):
                antenna_model = "{}_InfAir".format(antenna_type)
            else:
                antenna_model = "{}_InfFirn".format(antenna_type)
        else:
            antenna_model = antenna_type
        return antenna_model

    def get_noise_RMS(self, station_id, channel_id, stage='amp'):
        """
        returns the noise RMS that was precomputed from forced triggers

        Parameters
        ----------
        station_id: int
            station id
        channel_id: int
            the channel id, not used at the moment, only station averages are computed
        stage: string (default 'amp')
            specifies the stage of reconstruction you want the noise RMS for,
            `stage` can be one of
             * 'raw' (raw measured trace)
             * 'amp' (after the amp was deconvolved)
             * 'filt' (after the trace was highpass with 100MHz

        Returns
        -------
        RMS: float
            the noise RMS (actually it is the standard deviation but as the mean should be zero its the same)
        """
        if(self.__noise_RMS is None):
            import json
            detector_directory = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(detector_directory, 'noise_RMS.json'), 'r') as fin:
                self.__noise_RMS = json.load(fin)

        key = "{:d}".format(station_id)
        if(key not in self.__noise_RMS.keys()):
            rms = self.__noise_RMS['default'][stage]
            logger.warning("no RMS values for station {} available, returning default noise for stage {}: RMS={:.2g} mV".format(station_id, stage, rms / units.mV))
            return rms
        return self.__noise_RMS[key][stage]
