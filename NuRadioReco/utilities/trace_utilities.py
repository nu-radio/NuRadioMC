from NuRadioReco.utilities import units
import numpy as np
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.utilities import ice
from NuRadioReco.utilities import geometryUtilities as geo_utl
import logging
logger = logging.getLogger('trace_utilities')




def get_efield_antenna_factor(station, frequencies, channels, detector, zenith, azimuth, antenna_pattern_provider):
    """
    Returns the antenna response to a radio signal coming from a specific direction
    
    Parameters
    ---------------
    station: Station
    frequencies: array of complex
        frequencies of the radio signal for which the antenna response is needed
    channels: array of int
        IDs of the channels
    detector: Detector
    zenith, azimuth: float, float
        incoming direction of the signal. Note that refraction and reflection at the ice/air boundary are taken into account
    antenna_pattern_provider: AntennaPatternProvider
    -----------------
    """
    n_ice = ice.get_refractive_index(-0.01, detector.get_site(station.get_id()))
    efield_antenna_factor = np.zeros((len(channels), 2, len(frequencies)), dtype=np.complex)  # from antenna model in e_theta, e_phi
    
    for iCh, channel in enumerate(station.iter_channels(channels)):
        zenith_antenna = zenith
        t_theta = 1.
        t_phi = 1.
        # first check case if signal comes from above
        if(zenith <= 0.5 * np.pi):
            # is antenna below surface?
            position = detector.get_relative_position(station.get_id(), channel.get_id())
            if(position[2] <= 0):
                zenith_antenna = geo_utl.get_fresnel_angle(zenith, n_ice, 1)
                t_theta = geo_utl.get_fresnel_t_p(zenith, n_ice, 1)
                t_phi = geo_utl.get_fresnel_t_s(zenith, n_ice, 1)
                logger.info("channel {:d}: electric field is refracted into the firn. theta {:.0f} -> {:.0f}. Transmission coefficient p (eTheta) {:.2f} s (ePhi) {:.2f}".format(iCh, zenith / units.deg, zenith_antenna / units.deg, t_theta, t_phi))
        else:
            # now the signal is coming from below, do we have an antenna above the surface?
            position = det.get_relative_position(station_id, channel.get_id())
            if(position[2] > 0):
                zenith_antenna = geo_utl.get_fresnel_angle(zenith, 1., n_ice)
        if(zenith_antenna is None):
            logger.warning("fresnel reflection at air-firn boundary leads to unphysical results, no reconstruction possible")
            return None

        logger.debug("angles: zenith {0:.0f}, zenith antenna {1:.0f}, azimuth {2:.0f}".format(np.rad2deg(zenith), np.rad2deg(zenith_antenna), np.rad2deg(azimuth)))
        antenna_model = detector.get_antenna_model(station.get_id(), channel.get_id(), zenith_antenna)
        antenna_pattern = antenna_pattern_provider.load_antenna_pattern(antenna_model)
        ori = detector.get_antanna_orientation(station.get_id(), channel.get_id())
        VEL = antenna_pattern.get_antenna_response_vectorized(frequencies, zenith_antenna, azimuth, *ori)
        efield_antenna_factor[iCh] = np.array([VEL['theta'] * t_theta, VEL['phi'] * t_phi])
    return efield_antenna_factor
