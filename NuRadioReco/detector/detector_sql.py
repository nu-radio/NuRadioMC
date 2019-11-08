import numpy as np
from NuRadioReco.utilities import units
import os
import MySQLdb
import inspect
import datetime
import json
import logging
logger = logging.getLogger('sql detector')


class Detector(object):
    __instance = None

    def __new__(cls):
        if Detector.__instance is None:
            Detector.__instance = object.__new__(cls)
        return Detector.__instance

    def __init__(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))  # get the directory of this file
        filename = os.path.join(dir_path, "detector_sql_auth.json")
        fin = open(filename, 'r')
        mysql_opts = json.load(fin)
        fin.close()

        self.__mysql = MySQLdb.connect(host=mysql_opts['host'],
                                       user=mysql_opts['user'],
                                       passwd=mysql_opts['pass'],
                                       db=mysql_opts['DB'])

        self.__mysql.apilevel = "2.0"
        self.__mysql.threadsafety = 2
        self.__mysql.paramstyle = "format"
        logger.info("database connection to {} established".format(mysql_opts['DB']))

        self.__current_time = None
        # just for testing
        self.__current_time = datetime.datetime.now()

    def __error(self, frame):
        args, _, _, values = inspect.getargvalues(frame)
        out = 'request {} with arguments\n'.format(inspect.getframeinfo(frame)[2])
        for i in args[1:]:
            out += "    {} = {}\n".format(i, values[i])
        out += "   at time {}\n returned not result.".format(self.__current_time)
        logger.error(out)

    def update(self, timestamp):
        logger.info("updating detector time to {}".format(timestamp))
        self.__current_time = timestamp

    def get_everything(self, station_id):
        cursor = self.__mysql.cursor()

        fields = ['st.station_id', 'st.commission_time', 'st.decommission_time',
                  'st.station_type',
                  'st.position', 'st.board_number', 'st.MAC_address', 'st.MBED_type',
                  'ch.channel_id', 'ch.commission_time', 'ch.decommission_time',
                  'ant.antenna_type', 'ant.orientation_phi', 'ant.orientation_theta',
                  'ant.rotation_phi', 'ant.rotation_theta', 'ant.position_x',
                  'ant.position_y', 'ant.position_z', 'ant.deployment_time', 'ant.comment',
                  'cab.cable_type', 'cab.cable_length', 'cab.reference_measurement',
                  'cab.time_delay', 'cab.cable_id', 'cab.comment',
                  'pos.position', 'pos.measurement_time', 'pos.easting', 'pos.northing',
                  'pos.altitude', 'pos.site',
                  'amps.amp_type', 'amps.reference_measurement',
                  'adcs.adc_id', 'adcs.time_delay', 'adcs.nbits', 'adcs.sampling_frequency', 'adcs.n_samples'
                  ]
        field_str = ""
        for field in fields:
            field_str += field + ", "
        field_str = field_str[:-2]
        query = """
        SELECT {fields} FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN antennas AS ant USING(antenna_uid)
            JOIN cables AS cab USING(cable_uid)
            JOIN positions AS pos USING(position)
            JOIN amps USING(amp_uid)
            JOIN adcs USING(adc_uid)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id};
            """.format(fields=field_str, time=self.__current_time, station_id=station_id)
        cursor.execute(query)
        result = np.array(cursor.fetchall())
        if(len(result) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        result_dict = []
        for r in np.squeeze(result):
            t = {}
            for i, field_name in enumerate(fields):
                t[field_name] = r[i]
            result_dict.append(t)
        return result_dict

    def get_everything_channels(self):
        cursor = self.__mysql.cursor()

        fields = ['st.station_id', 'ch.channel_id', 'ch.commission_time', 'ch.decommission_time',
                  'ant.antenna_type', 'ant.orientation_phi', 'ant.orientation_theta',
                  'ant.rotation_phi', 'ant.rotation_theta', 'ant.position_x',
                  'ant.position_y', 'ant.position_z', 'ant.deployment_time', 'ant.comment',
                  'cab.cable_type', 'cab.cable_length', 'cab.reference_measurement',
                  'cab.time_delay', 'cab.cable_id', 'cab.comment',
                  'amps.amp_type', 'amps.reference_measurement',
                  'adcs.adc_id', 'adcs.time_delay', 'adcs.nbits', 'adcs.sampling_frequency', 'adcs.n_samples'
                  ]
        field_str = ""
        for field in fields:
            field_str += field + ", "
        field_str = field_str[:-2]
        query = """
        SELECT {fields} FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN antennas AS ant USING(antenna_uid)
            JOIN cables AS cab USING(cable_uid)
            JOIN positions AS pos USING(position)
            JOIN amps USING(amp_uid)
            JOIN adcs USING(adc_uid);
            """.format(fields=field_str)
        cursor.execute(query)
        result = np.array(cursor.fetchall())
        if(len(result) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        result_dict = []
        for r in np.squeeze(result):
            t = {}
            for i, field_name in enumerate(fields):
                t[field_name] = r[i]
            result_dict.append(t)
        return result_dict

    def get_everything_stations(self):
        cursor = self.__mysql.cursor()

        fields = ['st.station_id', 'st.commission_time', 'st.decommission_time',
                  'st.station_type',
                  'st.position', 'st.board_number', 'st.MAC_address', 'st.MBED_type',
                  'pos.position', 'pos.measurement_time', 'pos.easting', 'pos.northing', 'pos.zone',
                  'pos.altitude', 'pos.site']
        field_str = ""
        for field in fields:
            field_str += field + ", "
        field_str = field_str[:-2]
        query = """
        SELECT {fields} FROM stations AS st
            JOIN positions AS pos USING(position);
            """.format(fields=field_str)
        cursor.execute(query)
        result = np.array(cursor.fetchall())
        if(len(result) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        result_dict = []
        for r in np.squeeze(result):
            t = {}
            for i, field_name in enumerate(fields):
                t[field_name] = r[i]
            result_dict.append(t)
        return result_dict

    def get_everything_positions(self):
        cursor = self.__mysql.cursor()

        fields = ['pos.position', 'pos.measurement_time', 'pos.easting', 'pos.northing', 'pos.zone',
                  'pos.altitude', 'pos.site', 'pos.comment']
        field_str = ""
        for field in fields:
            field_str += field + ", "
        field_str = field_str[:-2]
        query = """
        SELECT {fields} FROM positions AS pos;
            """.format(fields=field_str)
        cursor.execute(query)
        result = np.array(cursor.fetchall())
        if(len(result) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        result_dict = []
        for r in np.squeeze(result):
            t = {}
            for i, field_name in enumerate(fields):
                t[field_name] = r[i]
            result_dict.append(t)
        return result_dict

    def get_absolute_position_site(self, pos):
        """
        returns the UTM coordinates
        
        Parameters
        ----------
        pos: string
            the position identifier (e.g. "A" or "X")
        Returns:
            * easting (float)
            * northing (float)
            * UTM zone (string)
            * altitude
            * measurement time
        """
        cursor = self.__mysql.cursor()
        query = """
        SELECT easting, northing, zone, altitude, measurement_time FROM positions
        WHERE position = '{position}' ORDER BY measurement_time DESC;
            """.format(position=pos)
        cursor.execute(query)
        position = np.array(cursor.fetchall())
        if(len(position) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        return np.squeeze(position)

    def get_relative_position(self, station_id, channel_id):
        cursor = self.__mysql.cursor()
        query = """
        SELECT position_x, position_y, position_z FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN antennas USING(antenna_uid)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id} AND ch.channel_id = {channel_id:d} ;
            """.format(time=self.__current_time, station_id=station_id, channel_id=channel_id)
        cursor.execute(query)
        position = np.array(cursor.fetchall())
        if(len(position) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        return np.squeeze(position)

    def get_relative_positions(self, station_id):
        cursor = self.__mysql.cursor()
        query = """
        SELECT position_x, position_y, position_z FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN antennas USING(antenna_uid)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id};
            """.format(time=self.__current_time, station_id=station_id)
        cursor.execute(query)
        position = np.array(cursor.fetchall())
        if(len(position) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
#         logger.debug('station {}: {}'.format(station_id, position))
        return position

    def get_site(self, station_id):
        cursor = self.__mysql.cursor()
        query = """
        SELECT site FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN positions USING(position)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id};
            """.format(time=self.__current_time, station_id=station_id)
        cursor.execute(query)
        site = np.array(cursor.fetchall())
        if(len(site) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
#         logger.debug('station {}: {}'.format(station_id, position))
        return site[0][0]

    def get_number_of_channels(self, station_id):
        cursor = self.__mysql.cursor()
        query = """
        SELECT channel_id FROM stations AS st
            JOIN channels AS ch USING(station_uid)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id};
            """.format(time=self.__current_time, station_id=station_id)
        cursor.execute(query)
        channel_ids = np.array(cursor.fetchall())
        if(len(channel_ids) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        return len(channel_ids)

    def get_cable_delay(self, station_id, channel_id):
        cursor = self.__mysql.cursor()
        query = """
        SELECT time_delay FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN cables USING(cable_uid)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id} AND ch.channel_id = {channel_id:d} ;
            """.format(time=self.__current_time, station_id=station_id, channel_id=channel_id)
        cursor.execute(query)
        delay = np.array(cursor.fetchall())
        if(len(delay) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
            return None
        return np.squeeze(delay)

    def get_cable_type_and_length(self, station_id, channel_id):
        cursor = self.__mysql.cursor()
        query = """
        SELECT cable_type, cable_length FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN cables USING(cable_uid)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id} AND ch.channel_id = {channel_id:d} ;
            """.format(time=self.__current_time, station_id=station_id, channel_id=channel_id)
        cursor.execute(query)
        type_length = np.array(cursor.fetchall())
        if(len(type_length) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
            return None
        return np.squeeze(type_length)[0], float(np.squeeze(type_length)[1]) * units.m

    def get_antenna_type(self, station_id, channel_id):
        cursor = self.__mysql.cursor()
        query = """
        SELECT antenna_type FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN antennas USING(antenna_uid)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id} AND ch.channel_id = {channel_id:d} ;
            """.format(time=self.__current_time, station_id=station_id, channel_id=channel_id)
        cursor.execute(query)
        antenna_type = np.array(cursor.fetchall())
        if(len(antenna_type) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        if(len(antenna_type[0]) != 1):
            logger.error("more than one antenna type return for station channel combination -> bug in detector description, only first element is returned")
        return antenna_type[0][0]

    def get_antenna_deployment_time(self, station_id, channel_id):
        cursor = self.__mysql.cursor()
        query = """
        SELECT deployment_time FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN antennas USING(antenna_uid)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id} AND ch.channel_id = {channel_id:d} ;
            """.format(time=self.__current_time, station_id=station_id, channel_id=channel_id)
        cursor.execute(query)
        deployment_time = np.array(cursor.fetchall())
        if(len(deployment_time) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        if(len(deployment_time[0]) != 1):
            logger.error("more than one antenna deployment_time return for station channel combination -> bug in detector description, only first element is returned")
        return deployment_time[0][0]

    def get_antenna_orientation(self, station_id, channel_id):
        """ returns the orientation of a specific antenna
        * orientation theta: boresight direction (zenith angle, 0deg is the zenith, 180deg is straight down)
        * orientation phi: boresight direction (azimuth angle counting from East counterclockwise)
        * rotation theta: rotation of the antenna, vector in plane of tines pointing away from connector
        * rotation phi: rotation of the antenna, vector in plane of tines pointing away from connector
        """
        cursor = self.__mysql.cursor()
        query = """
        SELECT orientation_theta, orientation_phi, rotation_theta, rotation_phi FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN antennas USING(antenna_uid)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id} AND ch.channel_id = {channel_id:d} ;
            """.format(time=self.__current_time, station_id=station_id, channel_id=channel_id)
        cursor.execute(query)
        result = np.array(cursor.fetchall())
        if(len(result) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        return np.deg2rad(result.flatten())

    def get_amplifier_type(self, station_id, channel_id):
        cursor = self.__mysql.cursor()
        query = """
        SELECT amp_type FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN amps USING(amp_uid)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id} AND ch.channel_id = {channel_id:d} ;
            """.format(time=self.__current_time, station_id=station_id, channel_id=channel_id)
        cursor.execute(query)
        amp_type = np.array(cursor.fetchall())
        if(len(amp_type) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        logger.debug("get_amplifier_type({},{}) returns {}".format(station_id, channel_id, amp_type.flatten()[0]))
        return amp_type.flatten()[0]

    def get_sampling_frequency(self, station_id, channel_id):
        cursor = self.__mysql.cursor()
        query = """
        SELECT sampling_frequency FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN adcs USING(adc_uid)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id} AND ch.channel_id = {channel_id:d} ;
            """.format(time=self.__current_time, station_id=station_id, channel_id=channel_id)
        cursor.execute(query)
        sampling_frequency = np.array(cursor.fetchall())
        if(len(sampling_frequency) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        return sampling_frequency.flatten()[0] * units.GHz

    def get_number_of_samples(self, station_id, channel_id):
        cursor = self.__mysql.cursor()
        query = """
        SELECT n_samples FROM stations AS st
            JOIN channels AS ch USING(station_uid)
            JOIN adcs USING(adc_uid)
        WHERE CAST('{time}' AS DATETIME) between ch.commission_time and ch.decommission_time
            AND CAST('{time}' AS DATETIME) between st.commission_time and st.decommission_time
            AND st.station_id = {station_id} AND ch.channel_id = {channel_id:d} ;
            """.format(time=self.__current_time, station_id=station_id, channel_id=channel_id)
        cursor.execute(query)
        n_samples = np.array(cursor.fetchall())
        if(len(n_samples) == 0):
            frame = inspect.currentframe()
            self.__error(frame)
        return n_samples.flatten()[0]

    def get_antenna_model(self, station_id, channel_id):
        """
        determine correct antenna model from antenna type, position and orientation of antenna

        so far only infinite firn and infinite air cases are differentiated

        """

        antenna_type = self.get_antenna_type(station_id, channel_id)
        antenna_relative_position = self.get_relative_position(station_id, channel_id)

        antenna_model = ""
        if(antenna_relative_position[2] > 0):
            antenna_model = "{}_infiniteair".format(antenna_type)
        else:
            antenna_model = "{}_infinitefirn".format(antenna_type)
        return antenna_model


relative_positions = {0: np.array([0, 3, 0]) * units.m, 1: np.array([3, 0, 0]) * units.m,
             2: np.array([0, -3, 0]) * units.m, 3: np.array([-3, 0, 0]) * units.m}

station_types = ['HRA_4', 'CR_4', 'HRA_8', 'dummy']

number_of_channels_for_station_type = {'HRA_4': 4, 'CR_4': 4, 'HRA_8': 8}

antenna_types_for_station_type = {'CR_4': ['UEW', 'UNS', 'UEW', 'UNS'],
                                  'HRA_4': ['DEW', 'DNS', 'DEW', 'DNS'],
                                  'dummy': ['DEW', 'DEW', 'DEW']}


def get_cable_delays(station_id):
    if station_id == 41:
        cable_delays = [19.95 * units.ns, 19.86 * units.ns, 18.82 * units.ns, 19.86 * units.ns]
    elif station_id == 51:
        cable_delays = [19.8 * units.ns, 19.8 * units.ns, 19.8 * units.ns, 19.7 * units.ns, 27.3 * units.ns, 27.5 * units.ns, 27.3 * units.ns, 19.6 * units.ns]
    else:
        logger.warning("Cable delays not implemented for other stations. Using defaults from 41.")
        cable_delays = [19.95 * units.ns, 19.86 * units.ns, 18.82 * units.ns, 19.86 * units.ns]
    return cable_delays


def get_antenna_model_file(station_type):
    # dummy module
    dir_path = os.path.dirname(os.path.realpath(__file__))  # get the directory of this file
    antenna_model = 'WIPLD_antennamodel_firn_v2.root'
    antenna_model_file = os.path.join(dir_path, 'AntennaModels', antenna_model)

    return antenna_model_file


def get_relative_position(station, channel):
    # default 4 antenna station
    if station == 41:
        relative_positions = {0: np.array([0, 4, 0]) * units.m, 1: np.array([4, 0, 0]) * units.m,
             2: np.array([0, -4, 0]) * units.m, 3: np.array([-4, 0, 0]) * units.m}
    else:
        logger.debug("Getting antenna positions for default station.")
        relative_positions = {0: np.array([0, 3, 0]) * units.m, 1: np.array([3, 0, 0]) * units.m,
             2: np.array([0, -3, 0]) * units.m, 3: np.array([-3, 0, 0]) * units.m}
    return relative_positions[channel]


def get_relative_positions(station):
    # default 4 antenna station
    if station == 41:
        relative_positions = {0: np.array([0, 4, 0]) * units.m, 1: np.array([4, 0, 0]) * units.m,
             2: np.array([0, -4, 0]) * units.m, 3: np.array([-4, 0, 0]) * units.m}
    else:
        logger.debug("Getting antenna positions for default station.")
        relative_positions = {0: np.array([0, 3, 0]) * units.m, 1: np.array([3, 0, 0]) * units.m,
             2: np.array([0, -3, 0]) * units.m, 3: np.array([-3, 0, 0]) * units.m}
    return relative_positions


def get_antenna_type(station_type, channel):

    if station_type not in antenna_types_for_station_type.keys():
        logger.error("Station type {} not known".format(station_type))
        antenna_type = None
    else:
        config = antenna_types_for_station_type[station_type]
        if len(config) < channel:
            logger.error("Requested channel {} not present in station".format(channel))
            antenna_type = None
        else:
            antenna_type = config[channel]

    return antenna_type


def get_amplifier_type(station_id, channel):
    # dummy module
    # currently a choice of 100, 200 and 300 series
    return '100'


def get_station_type(station_id, time):
    # dummy module
    if station_id == 41:
        type = station_types[1]
    elif station_id == 30:
        type = station_types[0]
    elif station_id == 52:
        type = station_types[2]
    elif station_id == 0:  # dummy station
        type = 'dummy'
    else:
        logger.error("Station id {} not assigned a station type".format(station_id))
        type = None
    return type


def get_number_of_channels(station_type):
    # dummy module
    if station_type not in number_of_channels_for_station_type.keys():
        logger.error("Station type {} not known".format(station_type))
        n_channels = None
    else:
        n_channels = number_of_channels_for_station_type[station_type]

    return n_channels
