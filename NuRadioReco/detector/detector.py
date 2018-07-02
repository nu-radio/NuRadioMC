import numpy as np
from NuRadioReco.utilities import units
from radiotools import helper as hp
import os
import inspect
import logging
from tinydb import TinyDB, Query
from tinydb_serialization import SerializationMiddleware
from tinydb.storages import MemoryStorage
from datetime import datetime
from tinydb_serialization import Serializer
import six  # # used for compatibility between py2 and py3
logger = logging.getLogger('detector')


class DateTimeSerializer(Serializer):
    OBJ_CLASS = datetime  # The class this serializer handles

    def encode(self, obj):
        return obj.strftime('%Y-%m-%dT%H:%M:%S')

    def decode(self, s):
        return datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')


serialization = SerializationMiddleware()
serialization.register_serializer(DateTimeSerializer(), 'TinyDate')


def buffer_db(in_memory, filename=None):
    db = None
    if(in_memory):
        db = TinyDB(storage=MemoryStorage)
    else:
        db = TinyDB(filename, storage=serialization, sort_keys=True, indent=4, separators=(',', ': '))
    db.purge()
    table_stations = db.table('stations')
    table_stations.purge()

    import detector_sql
    sqldet = detector_sql.Detector()
    results = sqldet.get_everything_stations()
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
                               'pos_altitude': result['pos.altitude'],
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

    def __init__(self, source='json', json_filename='ARIANNA/arianna_detector_db.json'):

        if(source == 'sql'):
            self.__db = buffer_db(in_memory=True)
        else:
            dir_path = os.path.dirname(os.path.realpath(__file__))  # get the directory of this file
            filename = os.path.join(dir_path, json_filename)
            if(not os.path.exists(filename)):
                logger.error("can't locate json database file {}".format(filename))
                raise NameError
            self.__db = TinyDB(filename, storage=serialization,
                               sort_keys=True, indent=4, separators=(',', ': '))

        self.__stations = self.__db.table('stations', cache_size=1000)
        self.__channels = self.__db.table('channels', cache_size=1000)

        logger.info("database initialized")

        self.__buffered_stations = {}
        self.__buffered_channels = {}
        self.__valid_t0 = datetime(2100, 1, 1)
        self.__valid_t1 = datetime(1970, 1, 1)

        self.__noise_RMS = None

        self.__current_time = None
        # just for testing
        self.__current_time = datetime.now()

    def __query_channel(self, station_id, channel_id):
        Channel = Query()
        res = self.__channels.get((Channel.station_id == station_id) & (Channel.channel_id == channel_id)
                                           & (Channel.commission_time < self.__current_time)
                                           & (Channel.decommission_time > self.__current_time))
        if(res is None):
            logger.error("query for station {} and channel {} at time {} returned no results".format(station_id, channel_id, self.__current_time))
            raise LookupError
        return res

    def __query_channels(self, station_id):
        Channel = Query()
        return self.__channels.search((Channel.station_id == station_id)
                                           & (Channel.commission_time < self.__current_time)
                                           & (Channel.decommission_time > self.__current_time))

    def __query_station(self, station_id):
        Station = Query()
        res = self.__stations.get((Station.station_id == station_id)
                                       & (Station.commission_time < self.__current_time)
                                       & (Station.decommission_time > self.__current_time))
        if(res is None):
            logger.error("query for station {} at time {} returned no results".format(station_id, self.__current_time))
            raise LookupError
        return res

    def __get_station(self, station_id):
        if(station_id not in self.__buffered_stations.keys()):
            self.__buffer(station_id)
        return self.__buffered_stations[station_id]

    def __get_channels(self, station_id):
        if(station_id not in self.__buffered_stations.keys()):
            self.__buffer(station_id)
        return self.__buffered_channels[station_id]

    def __get_channel(self, station_id, channel_id):
        if(station_id not in self.__buffered_stations.keys()):
            self.__buffer(station_id)
        return self.__buffered_channels[station_id][channel_id]

    def __buffer(self, station_id):
        self.__buffered_stations[station_id] = self.__query_station(station_id)
        self.__valid_t0 = max(self.__valid_t0, self.__buffered_stations[station_id]['commission_time'])
        self.__valid_t1 = min(self.__valid_t0, self.__buffered_stations[station_id]['decommission_time'])
        channels = self.__query_channels(station_id)
        self.__buffered_channels[station_id] = {}
        for channel in channels:
            self.__buffered_channels[station_id][channel['channel_id']] = channel
            self.__valid_t0 = max(self.__valid_t0, channel['commission_time'])
            self.__valid_t1 = min(self.__valid_t0, channel['decommission_time'])

    def update(self, timestamp):
        logger.info("updating detector time to {}".format(timestamp))
        self.__current_time = timestamp
        if(not ((self.__current_time > self.__valid_t0) and (self.__current_time < self.__valid_t1))):
            self.__buffered_stations = {}
            self.__buffered_channels = {}
            self.__valid_t0 = datetime(2100, 1, 1)
            self.__valid_t1 = datetime(1970, 1, 1)

    def get_relative_position(self, station_id, channel_id):
        res = self.__get_channel(station_id, channel_id)
        return np.array([res['ant_position_x'], res['ant_position_y'], res['ant_position_z']])

    def get_relative_positions(self, station_id):
        res = self.__get_channels(station_id)
        positions = np.zeros((len(res), 3))
        for i, r in enumerate(res.values()):
            logger.debug("position channel {}: {:.0f}m, {:.0f}m, {:.0f}m".format(r['channel_id'], r['ant_position_x'], r['ant_position_y'], r['ant_position_z']))
            positions[i] = [r['ant_position_x'], r['ant_position_y'], r['ant_position_z']]
        return positions

    def get_site(self, station_id):
        res = self.__get_station(station_id)
        return res['pos_site']

    def get_number_of_channels(self, station_id):
        res = self.__get_channels(station_id)
        return len(res)

    def get_parallel_channels(self, station_id):
        res = self.__get_channels(station_id)
        orientations = np.zeros((len(res), 4))
        antenna_types = []
        channel_ids = []
        for iCh, ch in enumerate(res.values()):
            channel_id = ch['channel_id']
            channel_ids.append(channel_id)
            antenna_types.append(self.get_antenna_type(station_id, channel_id))
            orientations[iCh] = self.get_antanna_orientation(station_id, channel_id)
            orientations[iCh][3] = hp.get_normalized_angle(orientations[iCh][3], interval=np.deg2rad([0, 180]))
        channel_ids = np.array(channel_ids)
        antenna_types = np.array(antenna_types)
        orientations = np.round(np.rad2deg(orientations))  # round to one degree to overcome rounding errors
        parallel_antennas = []
        for antenna_type in np.unique(antenna_types):
#             print(antenna_type)
            for u_zen_ori in np.unique(orientations[:, 0]):
#                 print(u_zen_ori)
                for u_az_ori in np.unique(orientations[:, 1]):
#                     print("\t {}".format(u_az_ori))
                    for u_zen_rot in np.unique(orientations[:, 2]):
#                         print("\t\t{}".format(u_zen_rot))
                        for u_az_rot in np.unique(orientations[:, 3]):
#                             print("\t\t\t{}".format(u_az_rot))
#                             print("\t\t\t\t{}".format(antenna_types == antenna_type))
#                             print("\t\t\t\t{}".format(orientations[:, 0] == u_zen_ori))
#                             print("\t\t\t\t{}".format(orientations[:, 1] == u_az_ori))
#                             print("\t\t\t\t{}".format(orientations[:, 2] == u_zen_rot))
#                             print("\t\t\t\t{}".format(orientations[:, 3] == u_az_rot))
                            mask = (antenna_types == antenna_type) \
                                     & (orientations[:, 0] == u_zen_ori) & (orientations[:, 1] == u_az_ori) \
                                     & (orientations[:, 2] == u_zen_rot) & (orientations[:, 3] == u_az_rot)
#                             print("\t\t\t\t\t{}".format(mask))
                            if(np.sum(mask)):
                                parallel_antennas.append(channel_ids[mask])
        return np.array(parallel_antennas)

    def get_cable_delay(self, station_id, channel_id):
        res = self.__get_channel(station_id, channel_id)
        return res['cab_time_delay']

    def get_cable_type_and_length(self, station_id, channel_id):
        res = self.__get_channel(station_id, channel_id)
        return res['cab_type'], res['cab_length'] * units.m

    def get_antenna_type(self, station_id, channel_id):
        res = self.__get_channel(station_id, channel_id)
        return res['ant_type']

    def get_antanna_orientation(self, station_id, channel_id):
        """ returns the orientation of a specific antenna
        * orientation theta: boresight direction (zenith angle, 0deg is the zenith, 180deg is straight down)
        * orientation phi: boresight direction (azimuth angle counting from East counterclockwise)
        * rotation theta: rotation of the antenna, vector in plane of tines pointing away from connector
        * rotation phi: rotation of the antenna, vector in plane of tines pointing away from connector
        """
        res = self.__get_channel(station_id, channel_id)
        return np.deg2rad([res['ant_orientation_theta'], res['ant_orientation_phi'], res['ant_rotation_theta'], res['ant_rotation_phi']])

    def get_amplifier_type(self, station_id, channel_id):
        res = self.__get_channel(station_id, channel_id)
        return res['amp_type']

    def get_sampling_frequency(self, station_id, channel_id):
        res = self.__get_channel(station_id, channel_id)
        return res['adc_sampling_frequency'] * units.GHz

    def get_number_of_samples(self, station_id, channel_id):
        res = self.__get_channel(station_id, channel_id)
        return res['adc_n_samples']

    def get_antenna_model(self, station_id, channel_id, zenith=None):
        """
        determine correct antenna model from antenna type, position and orientation of antenna

        so far only infinite firn and infinite air cases are differentiated

        """

        antenna_type = self.get_antenna_type(station_id, channel_id)
        antenna_relative_position = self.get_relative_position(station_id, channel_id)

        antenna_model = ""
        if(zenith is not None and (antenna_type == 'createLPDA_100MHz')):
            if(antenna_relative_position[2] > 0):
                if(zenith < 90 * units.deg):
                    antenna_model = "{}_z1cm_InAir_RG".format(antenna_type)
                else:
                    antenna_model = "{}_InfAir".format(antenna_type)
            else:  # antenna in firn
                if(zenith > 90 * units.deg):  # signal comes from below
                    antenna_model = "{}_z1cm_InFirn_RG".format(antenna_type)
                    # we need to add further distinction here
                else:
                    antenna_model = "{}_InfFirn".format(antenna_type)
        else:
            if(antenna_relative_position[2] > 0):
                antenna_model = "{}_InfAir".format(antenna_type)
            else:
                antenna_model = "{}_InfFirn".format(antenna_type)
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
