from NuRadioReco.modules.base.module import register_run
import numpy as np
from scipy import constants

from radiotools import helper as hp

from NuRadioReco.utilities import units, ice
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp

import logging
logger = logging.getLogger('templateDirectionFitter')


class templateDirectionFitter:
    """
    Calculates basic signal parameters.
    """

    def __init__(self):
        pass

    def begin(self):
        pass

    @register_run()
    def run(self, evt, station, det, debug=True, channels_to_use=[0, 1, 2, 3], cosmic_ray=False):
        """
        Fits the direction using templates

        Parameters
        ----------
        evt: event

        station: station

        det: detector

        debug: bool
            set debug
        channels_to_use: list
            antenna to use for fit
        cosmic_ray: bool
            type to set correlation template

        """
        if(cosmic_ray):
            type_str = 'cr'
            xcorrelations = chp.cr_xcorrelations
        else:
            type_str = 'nu'
            xcorrelations = chp.nu_xcorrelations

        station_id = station.get_id()
        channels = station.iter_channels(channels_to_use)

        times = []
        positions = []

        for iCh, channel in enumerate(channels):
            channel_id = channel.get_id()
            times.append(channel[xcorrelations]['{}_ref_xcorr_time'.format(type_str)] + channel.get_trace_start_time())
            positions.append(det.get_relative_position(station_id, channel_id))

        times = np.array(times)
        positions = np.array(positions)
        site = det.get_site(station_id)
        n_ice = ice.get_refractive_index(-0.01, site)

        from scipy import optimize as opt

        def obj_plane(params, positions, t_measured):
            zenith, azimuth = params
            if cosmic_ray:
                if((zenith < 0) or (zenith > 0.5 * np.pi)):
                    return np.inf
            else:
                if((zenith < 0.5 * np.pi) or (zenith > np.pi)):
                    return np.inf
            v = hp.spherical_to_cartesian(zenith, azimuth)
            c = constants.c * units.m / units.s
            if not cosmic_ray:
                c = c / n_ice
                logger.debug("using speed of light = {:.4g}".format(c))
            t_expected = -(np.dot(v, positions.T) / c)
            sigma = 1 * units.ns
            chi2 = np.sum(((t_expected - t_expected.mean()) - (t_measured - t_measured.mean())) ** 2 / sigma ** 2)
            logger.debug("texp = {texp}, tm = {tmeas}, {chi2}".format(texp=t_expected, tmeas=t_measured, chi2=chi2))
            return chi2

        method = "Nelder-Mead"
        options = {'maxiter': 1000,
                   'disp': False}
        zenith_start = 135 * units.deg
        if cosmic_ray:
            zenith_start = 45 * units.deg
        starting_chi2 = {}
        for starting_az in np.array([0, 90, 180, 270]) * units.degree:
            starting_chi2[starting_az] = obj_plane((zenith_start, starting_az), positions, times)
        azimuth_start = min(starting_chi2, key=starting_chi2.get)
        res = opt.minimize(obj_plane, x0=[zenith_start, azimuth_start], args=(positions, times), method=method, options=options)

        output_str = "reconstucted angles theta = {:.1f}, phi = {:.1f}".format(res.x[0] / units.deg, hp.get_normalized_angle(res.x[1]) / units.deg)
        if station.has_sim_station():
            sim_zen = station.get_sim_station()[stnp.zenith]
            sim_az = station.get_sim_station()[stnp.azimuth]
            dOmega = hp.get_angle(hp.spherical_to_cartesian(sim_zen, sim_az), hp.spherical_to_cartesian(res.x[0], res.x[1]))
            output_str += "  MC theta = {:.1f}, phi = {:.1f},  dOmega = {:.2f}".format(sim_zen / units.deg, sim_az / units.deg, dOmega / units.deg)
        logger.info(output_str)
        station[stnp.zenith] = res.x[0]
        station[stnp.azimuth] = hp.get_normalized_angle(res.x[1])
        if(cosmic_ray):
            station[stnp.cr_zenith] = res.x[0]
            station[stnp.cr_azimuth] = hp.get_normalized_angle(res.x[1])
        else:
            station[stnp.nu_zenith] = res.x[0]
            station[stnp.nu_azimuth] = hp.get_normalized_angle(res.x[1])

    def end(self):
        pass
