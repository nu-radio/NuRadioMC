from NuRadioReco.modules.base.module import register_run
import numpy as np
from scipy import constants, stats

from radiotools import helper as hp

from NuRadioReco.utilities import units, ice
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp

import logging
logger = logging.getLogger('efieldTimeDirectionFitter')


class efieldTimeDirectionFitter:
    """
    Calculates basic signal parameters.
    """

    def __init__(self):
        self.__debug = None
        self.__time_uncertainty = None
        self.begin()
        pass

    def begin(self, debug=False, time_uncertainty=0.1 * units.ns):
        self.__debug = debug
        self.__time_uncertainty = time_uncertainty
        pass

    @register_run()
    def run(self, evt, station, det, channels_to_use=None, cosmic_ray=False):
        """
        Parameters
        ----------------
        evt: Event
            The event to run the module on
        station: Station
            The station to run the module on
        det: Detector
            The detector description
        channels_to_use: list of int (default: [0, 1, 2, 3])
            List with the IDs of channels to use for reconstruction
        cosmic_ray: Bool (default: False)
            Flag to mark event as cosmic ray

        """
        if channels_to_use is None:
            channels_to_use = [0, 1, 2, 3]
        station_id = station.get_id()

        times = []
        times_error = []
        positions = []
        for iCh, efield in enumerate(station.get_electric_fields()):
            if(len(efield.get_channel_ids()) > 1):
                # FIXME: this can be changed later if each efield has a position and absolute time
                raise AttributeError("found efield that is valid for more than one channel. Position can't be determined.")
            channel_id = efield.get_channel_ids()[0]
            if(channel_id not in channels_to_use):
                continue
            times.append(efield[efp.signal_time])
            if(efield.has_parameter_error(efp.signal_time)):
                times_error.append((efield.get_parameter_error(efp.signal_time) ** 2 + self.__time_uncertainty ** 2) ** 0.5)
            else:
                times_error.append(self.__time_uncertainty)
            positions.append(det.get_relative_position(station_id, channel_id))

        times = np.array(times)
        times_error = np.array(times_error)
        positions = np.array(positions)
        site = det.get_site(station_id)
        n_ice = ice.get_refractive_index(-0.01, site)

        from scipy import optimize as opt

        def get_expected_times(params, channel_positions):
            zenith, azimuth = params
            if cosmic_ray:
                if((zenith < 0) or (zenith > 0.5 * np.pi)):
                    return np.ones(len(channel_positions)) * np.inf
            else:
                if((zenith < 0.5 * np.pi) or (zenith > np.pi)):
                    return np.ones(len(channel_positions)) * np.inf
            v = hp.spherical_to_cartesian(zenith, azimuth)
            c = constants.c * units.m / units.s
            if not cosmic_ray:
                c = c / n_ice
                logger.debug("using speed of light = {:.4g}".format(c))
            t_expected = -(np.dot(v, channel_positions.T) / c)
            return t_expected

        def obj_plane(params, pos, t_measured):
            t_expected = get_expected_times(params, pos)
            chi_squared = np.sum(((t_expected - t_expected.mean()) - (t_measured - t_measured.mean())) ** 2 / times_error ** 2)
            logger.debug("texp = {texp}, tm = {tmeas}, {chi2}".format(texp=t_expected, tmeas=t_measured, chi2=chi_squared))
            return chi_squared

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

        chi2 = res.fun
        df = len(channels_to_use) - 3
        if(df == 0):
            chi2ndf = chi2
            chi2prob = stats.chi2.sf(chi2, 1)
        else:
            chi2ndf = chi2 / df
            chi2prob = stats.chi2.sf(chi2, df)

        output_str = "reconstucted angles theta = {:.1f}, phi = {:.1f}, chi2/ndf = {:.2g}/{:d} = {:.2g}, chi2prob = {:.3g}".format(res.x[0] / units.deg,
                                                                                                                                   hp.get_normalized_angle(res.x[1]) / units.deg,
                                                                                                                                   res.fun, df,
                                                                                                                                   chi2ndf,
                                                                                                                                   chi2prob)

        logger.info(output_str)
        station[stnp.zenith] = res.x[0]
        station[stnp.azimuth] = hp.get_normalized_angle(res.x[1])
        station[stnp.chi2_efield_time_direction_fit] = chi2
        station[stnp.ndf_efield_time_direction_fit] = df
        if(cosmic_ray):
            station[stnp.cr_zenith] = res.x[0]
            station[stnp.cr_azimuth] = hp.get_normalized_angle(res.x[1])

        if(self.__debug):
            # calculate residuals
            t_exp = get_expected_times(res.x, positions)
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1, 1)
            ax.errorbar(channels_to_use, ((times - times.mean()) - (t_exp - t_exp.mean())) / units.ns, fmt='o',
                        yerr=times_error / units.ns)
            ax.set_xlabel("channel id")
            ax.set_ylabel(r"$t_\mathrm{meas} - t_\mathrm{exp}$ [ns]")
            pass

    def end(self):
        pass
