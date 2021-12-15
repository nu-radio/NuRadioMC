import logging
import NuRadioReco.detector.detector
from NuRadioReco.utilities import units
logger = logging.getLogger('NuRadioReco.DetectorSysUncertainties')


class DetectorSysUncertainties(NuRadioReco.detector.detector.Detector):
    """
    """

    def __init__(self, source='json', json_filename='ARIANNA/arianna_detector_db.json',
                 dictionary=None, assume_inf=True, antenna_by_depth=True):
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
        antenna_by_depth: bool (default True)
            if True the antenna model is determined automatically depending on the depth of the antenna. This is done by
            appending e.g. '_InfFirn' to the antenna model name.
            if False, the antenna model as specified in the database is used.
        """
        NuRadioReco.detector.detector.Detector.__init__(self, source, json_filename, dictionary, assume_inf, antenna_by_depth)
        self._antenna_orientation_override = {}
        self._antenna_position_override = {}

    def set_antenna_orientation_offsets(self, ori_theta, ori_phi, rot_theta, rot_phi, station_id=None, channel_id=None):
        """
        sets a systematic offset for the antenna orientation

        Parameters
        ----------
        ori_theta: float
            orientation of the antenna, as a zenith angle (0deg is the zenith, 180deg is straight down); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
        ori_phi: float
            orientation of the antenna, as an azimuth angle (counting from East counterclockwise); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
        rot_theta: float
            rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
        rot_phi: float
            rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
        station_id: int or None
            the station id, if None offset will be applied to all stations/channels
        channel_id: int or None
            the channel id, if None offset will be applied to all channels
        """
        key = "any"
        if station_id is not None:
            if channel_id is not None:
                key = (station_id, channel_id)
            else:
                key = station_id
        self._antenna_orientation_override[key] = [ori_theta, ori_phi, rot_theta, rot_phi]

    def reset_antenna_orientation_offsets(self):
        """
        resets all previously set antenna orientation offsets
        """
        self._antenna_orientation_override = {}

    def get_antenna_orientation(self, station_id, channel_id):
        """
        returns the orientation of a specific antenna + a systematic offset

        Parameters
        ----------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns typle of floats
            * orientation theta: orientation of the antenna, as a zenith angle (0deg is the zenith, 180deg is straight down); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
            * orientation phi: orientation of the antenna, as an azimuth angle (counting from East counterclockwise); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
            * rotation theta: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
            * rotation phi: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
        """
        ori = super().get_antenna_orientation(station_id, channel_id)
        if "any" in self._antenna_orientation_override:
            tmp = self._antenna_orientation_override["any"]
            ori += tmp
            logger.info(f"adding orientation theta = {tmp[0]/units.deg:.1f} deg, phi = {tmp[1]/units.deg:.1f} deg, rotation theta = {tmp[2]/units.deg:.1f} deg, phi = {tmp[3]/units.deg:.1f} deg to all channels of any station")
        if station_id in self._antenna_orientation_override:
            tmp = self._antenna_orientation_override[station_id]
            ori += tmp
            logger.info(f"adding orientation theta = {tmp[0]/units.deg:.1f} deg, phi = {tmp[1]/units.deg:.1f} deg, rotation theta = {tmp[2]/units.deg:.1f} deg, phi = {tmp[3]/units.deg:.1f} deg to all channels of station {station_id}")
        if (station_id, channel_id) in self._antenna_orientation_override:
            tmp = self._antenna_orientation_override[(station_id, channel_id)]
            logger.info(f"adding orientation theta = {tmp[0]/units.deg:.1f} deg, phi = {tmp[1]/units.deg:.1f} deg, rotation theta = {tmp[2]/units.deg:.1f} deg, phi = {tmp[3]/units.deg:.1f} deg to channel {channel_id} of station {station_id}")
            ori += tmp
        return ori

    def set_antenna_position_offsets(self, x, y, z, station_id=None, channel_id=None):
        """
        sets a systematic offset for the antenna position

        Parameters
        ----------
        x: float
            x-position of antenna
        y: float
            y-position of antenna
        z: float
            z-position of antenna ( (-) is below the surface)
        station_id: int or None
            the station id, if None offset will be applied to all stations/channels
        channel_id: int or None
            the channel id, if None offset will be applied to all channels
        """
        key = "any"
        if station_id is not None:
            if channel_id is not None:
                key = (station_id, channel_id)
            else:
                key = station_id
        self._antenna_position_override[key] = [x, y, z]

    def reset_antenna_position_offsets(self):
        """
        resets all previously set antenna position offsets
        """
        self._antenna_position_override = {}

    def get_relative_position(self, station_id, channel_id):
        """
        returns the orientation of a specific antenna + a systematic offset

        Parameters
        ----------
        station_id: int
            the station id
        channel_id: int
            the channel id

        Returns tuple of floats
            * x-position of antenna
            * y-position of antenna
            * z-position of antenna
        """
        pos = super().get_relative_position(station_id, channel_id)
        if "any" in self._antenna_position_override:
            tmp = self._antenna_position_override["any"]
            pos += tmp
            logger.info(f"adding position x = {tmp[0]/units.m:.1f} m, y = {tmp[1]/units.m:.1f} m, z = {tmp[2]/units.m:.1f} m to all channels of any station")
        if station_id in self._antenna_position_override:
            tmp = self._antenna_position_override[station_id]
            pos += tmp
            logger.info(f"adding position x = {tmp[0]/units.m:.1f} m, y = {tmp[1]/units.m:.1f} m, z = {tmp[2]/units.m:.1f} m to all channels of station {station_id}")
        if (station_id, channel_id) in self._antenna_position_override:
            tmp = self._antenna_position_override[(station_id, channel_id)]
            logger.info(f"adding position x = {tmp[0]/units.m:.1f} m, y = {tmp[1]/units.m:.1f} m, z = {tmp[2]/units.m:.1f} m to channel {channel_id} of station {station_id}")
            pos += tmp
        return pos
