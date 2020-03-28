import os
import astropy.time
import logging
from tinydb import TinyDB, Query
from tinydb_serialization import SerializationMiddleware
from tinydb.storages import MemoryStorage
from tinydb_serialization import Serializer
import NuRadioReco.detector.detector
from NuRadioReco.utilities import units
from NuRadioReco.detector.detector import DateTimeSerializer
import copy
logger = logging.getLogger('NuRadioReco.DetectorSysUncertainties')


class DetectorSysUncertainties(NuRadioReco.detector.detector.Detector):
    """
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
        self.det = super(DetectorSysUncertainties, self)
        self.det.__init__(source, json_filename, dictionary, assume_inf)
        self._antenna_orientation_override = {}

    def set_antenna_orientation_offsets(self, ori_theta, ori_phi, rot_theta, rot_phi, station_id=None, channel_id=None):
        """
        sets a systematic offset for the antenna orientation
        
        Parameters
        ---------
        ori_theta: float
            boresight direction (zenith angle, 0deg is the zenith, 180deg is straight down)
        ori_phi: float
            boresight direction (azimuth angle counting from East counterclockwise)
        rot_theta: float
            rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector in plane of tines pointing away from connector
        rot_phi: float
            rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector in plane of tines pointing away from connector
        station_id: int or None
            the station id, if None offset will be applied to all stations/channels
        channel_id: int or None
            the channel id, if None offset will be applied to all channels
        """
        key = "any"
        if station_id is not None:
            if(channel_id is not None):
                key = (station_id, channel_id)
            else:
                key = station_id
        self._antenna_orientation_override[key] = [ori_theta, ori_phi, rot_theta, rot_phi]

    def get_antenna_orientation(self, station_id, channel_id):
        """
        returns the orientation of a specific antenna + a systematic offset

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
        ori = self.det.get_antenna_orientation(station_id, channel_id)
        if("any" in self._antenna_orientation_override):
            tmp = self._antenna_orientation_override["any"]
            ori += tmp
            logger.info(f"adding orientation theta = {tmp[0]/units.deg:.1f} deg, phi = {tmp[1]/units.deg:.1f} deg, rotation theta = {tmp[2]/units.deg:.1f} deg, phi = {tmp[3]/units.deg:.1f} deg to all channels of any station")
        if station_id in self._antenna_orientation_override:
            tmp = self._antenna_orientation_override[station_id]
            ori += tmp
            logger.info(f"adding orientation theta = {tmp[0]/units.deg:.1f} deg, phi = {tmp[1]/units.deg:.1f} deg, rotation theta = {tmp[2]/units.deg:.1f} deg, phi = {tmp[3]/units.deg:.1f} deg to all channels of station {station_id}")
        if((station_id, channel_id) in self._antenna_orientation_override):
            tmp = self._antenna_orientation_override[(station_id, channel_id)]
            logger.info(f"adding orientation theta = {tmp[0]/units.deg:.1f} deg, phi = {tmp[1]/units.deg:.1f} deg, rotation theta = {tmp[2]/units.deg:.1f} deg, phi = {tmp[3]/units.deg:.1f} deg to channel {channel_id} of station {station_id}")
            ori += tmp
        return ori

