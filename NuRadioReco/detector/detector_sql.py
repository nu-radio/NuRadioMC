import NuRadioReco.detector.detector_base
import numpy as np
from NuRadioReco.utilities import units
import NuRadioReco.detector.RNO_G.analog_components
import NuRadioReco.detector.ARIANNA.analog_components
from radiotools import helper as hp
import os
import logging
from tinydb import TinyDB, Query
from tinydb_serialization import SerializationMiddleware
from tinydb.storages import MemoryStorage
import astropy.time
from datetime import datetime
from tinydb_serialization import Serializer
import six  # # used for compatibility between py2 and py3
import warnings
from astropy.utils.exceptions import ErfaWarning
import NuRadioReco.utilities.metaclasses

logger = logging.getLogger('NuRadioReco.detector')
warnings.filterwarnings('ignore', category=ErfaWarning)


class SQLDetector(NuRadioReco.detector.detector_base.DetectorBase):
    
    def __init__(self, log_level=logging.WARNING, assume_inf=True, antenna_by_depth=False):
        """
        Initialize the stations detector properties.

        Parameters
        ----------
       
        assume_inf : Bool
            Default to True, if true forces antenna madels to have infinite boundary conditions, 
            otherwise the antenna madel will be determined by the station geometry.
        antenna_by_depth: bool (default True)
            if True the antenna model is determined automatically depending on the depth of the antenna.
            This is done by appending e.g. '_InfFirn' to the antenna model name.
            if False, the antenna model as specified in the database is used.
        """
        self.__logger = logging.getLogger('NuRadioReco.genericDetector')
        self.__logger.setLevel(log_level)
        super(SQLDetector, self).__init__(assume_inf=assume_inf, antenna_by_depth=antenna_by_depth)


    def buffer_detector_in_tiny_db(self):
        """
        buffers the complete SQL database into a TinyDB object (either in memory or into 
        a local JSON file)

        Parameters
        ----------
        in_memory: bool
            if True: the mysql database will be buffered as a tiny tb object that only 
            exists in memory
            if False: the mysql database will be buffered as a tiny tb object and saved 
            in a local json file
        filename: string
            only relevant if `in_memory = True`: the filename of the json file of the tiny db object
        """

        logger.info("buffering SQL database on-the-fly")
        if self._in_memory:
            db = TinyDB(storage=MemoryStorage)
        else:
            db = TinyDB(self._filename, storage=self._serialization, 
                        sort_keys=True, indent=4, separators=(',', ': '))
        db.truncate()

        from NuRadioReco.detector import db_sql
        sqldb = db_sql.DatabaseInterface()
        results = sqldb.get_everything_stations()
        table_stations = db.table('stations')
        table_stations.truncate()
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
        table_channels.truncate()
        results = sqldb.get_everything_channels()
        for channel in results:
            table_channels.insert({'station_id': channel['st.station_id'],
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

        results = sqldb.get_everything_positions()
        table_positions = db.table('positions')
        table_positions.truncate()
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