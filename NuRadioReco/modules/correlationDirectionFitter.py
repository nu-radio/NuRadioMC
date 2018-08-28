from __future__ import absolute_import, division, print_function, unicode_literals
from scipy import signal, fftpack
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
import scipy.optimize as opt
from radiotools import helper as hp
import logging
logger = logging.getLogger('correlationDirectionFitter')


class correlationDirectionFitter:
    """
    Fits the direction using correlation of parallel channels.
    """

    def __init__(self):
        self.__zenith = []
        self.__azimuth = []
        self.__delta_zenith = []
        self.__delta_azimuth = []
        self.begin()

    def begin(self, debug=False):
        self.__debug = debug

    def run(self, evt, station, det, n_index=None, ZenLim=[0 * units.deg, 90 * units.deg],
            AziLim=[0 * units.deg, 360 * units.deg],
            channel_pairs=((0, 2), (1, 3))):
        """
        reconstruct signal arrival direction for all events

        Parameters
        ----------
        n_index: float
            the index of refraction

        ZenLim: 2-dim array/list of floats
            the zenith angle limits for the fit
            default if 0-90deg (upward coming signal)
        AziLim: 2-dim array/list of floats
            the azimuth angle limits for the fit
            default is 0-360deg
        channel_pairs: pair of pair of integers
            specify the two channel pairs to use, default ((0, 2), (1, 3))
        """

        use_correlation = True

        def ll_regular_station(angles, corr_02, corr_13, sampling_rate, positions):
            """
            Likelihood function for a four antenna ARIANNA station, using correction.
            Using correlation, has no built in wrap around, pulse needs to be in the middle
            """

            zenith = angles[0]
            azimuth = angles[1]
            times = []

            for pos in positions:
                tmp = []
                tmp.append(geo_utl.get_time_delay_from_direction(zenith, azimuth, pos[0], n=n_index) * sampling_rate)
                tmp.append(geo_utl.get_time_delay_from_direction(zenith, azimuth, pos[1], n=n_index) * sampling_rate)
                times.append(tmp)

            delta_t_02 = times[0][1] - times[0][0]
            delta_t_13 = times[1][1] - times[1][0]

            pos_02 = int(corr_02.shape[0] / 2 - delta_t_02)
            pos_13 = int(corr_13.shape[0] / 2 - delta_t_13)

#             weight_02 = np.sum(corr_02 ** 2)  # Normalize crosscorrelation
#             weight_13 = np.sum(corr_13 ** 2)
#
#             likelihood = -1 * (corr_02[pos_02] ** 2 / weight_02 + corr_13[pos_13] ** 2 / weight_13)
            # After deliberating a bit, I don't think we should use the square because anti-correlating
            # pulses would be wrong, given that it is not a continous waveform

            weight_02 = np.sum(np.abs(corr_02))  # Normalize crosscorrelation
            weight_13 = np.sum(np.abs(corr_13))

            likelihood = -1 * (corr_02[pos_02] / weight_02 + corr_13[pos_13] / weight_13)

            return likelihood

        def ll_regular_station_fft(angles, corr_02_fft, corr_13_fft, sampling_rate, positions):
            """
            Likelihood function for a four antenna ARIANNA station, using FFT convolution
            Using FFT convolution, has built-in wrap around, but ARIANNA signals are too short for it to be accurate
            will show problems at zero time delay
            """

            zenith = angles[0]
            azimuth = angles[1]
            times = []

            for pos in positions:
                tmp = []
                tmp.append(geo_utl.get_time_delay_from_direction(zenith, azimuth, pos[0], n=n_index) * sampling_rate)
                tmp.append(geo_utl.get_time_delay_from_direction(zenith, azimuth, pos[1], n=n_index) * sampling_rate)
                times.append(tmp)

            delta_t_02 = times[0][1] - times[0][0]
            delta_t_13 = times[1][1] - times[1][0]

            if delta_t_02 < 0:
                pos_02 = int(delta_t_02 + corr_02_fft.shape[0])
            else:
                pos_02 = int(delta_t_02)

            if delta_t_13 < 0:
                pos_13 = int(delta_t_13 + corr_13_fft.shape[0])
            else:
                pos_13 = int(delta_t_13)

            weight_02 = np.sum(np.abs(corr_02_fft))  # Normalize crosscorrelation
            weight_13 = np.sum(np.abs(corr_13_fft))

            likelihood = -1 * (np.abs(corr_02_fft[pos_02]) ** 2 / weight_02 + np.abs(corr_13[pos_13]) ** 2 / weight_13)
            return likelihood

        channels = station.get_channels()
        station_id = station.get_id()
        positions = det.get_relative_positions(station_id)
        positions_pairs = [[positions[channel_pairs[0][0]], positions[channel_pairs[0][1]]],
                           [positions[channel_pairs[1][0]], positions[channel_pairs[1][1]]]]
        sampling_rate = channels[0].get_sampling_rate()  # assume that channels have the same sampling rate

        # determine automatically if one channel has an inverted waveform with respect to the other
        signs = [1., 1.]
        for iPair, pair in enumerate(channel_pairs):
            antenna_type = det.get_antenna_type(station_id, pair[0])
            if("LPDA" in antenna_type):
                otheta, ophi, rot_theta, rot_azimuth = det.get_antanna_orientation(station_id, pair[0])
                otheta2, ophi2, rot_theta2, rot_azimuth2 = det.get_antanna_orientation(station_id, pair[1])
                if(np.isclose(np.abs(rot_azimuth - rot_azimuth2), 180 * units.deg, atol=1 * units.deg)):
                    signs[iPair] = -1

        if use_correlation:
            # Correlation
            corr_02 = signal.correlate(channels[channel_pairs[0][0]].get_trace(),
                                       signs[0] * channels[channel_pairs[0][1]].get_trace())
            corr_13 = signal.correlate(channels[channel_pairs[1][0]].get_trace(),
                                       signs[1] * channels[channel_pairs[1][1]].get_trace())

        else:
            # FFT convolution
            corr_02_fft = fftpack.ifft(-1 * fftpack.fft(channels[channel_pairs[0][0]].get_trace()).conjugate() * fftpack.fft(channels[channel_pairs[0][1]].get_trace()))
            corr_13_fft = fftpack.ifft(-1 * fftpack.fft(channels[channel_pairs[1][0]].get_trace()).conjugate() * fftpack.fft(channels[channel_pairs[1][1]].get_trace()))

        # Alternative fitters:
#             ll = opt.minimize(ll_regular_station,(zenith_orig-np.deg2rad(15) ,azimuth_orig-np.deg2rad(25)),
#             (corr_02, corr_13,sampling_rate),method='BFGS') # gets stuck in local minima
#
#             ll = opt.basinhopping(ll_regular_station,(zenith_orig-np.deg2rad(15) ,azimuth_orig-np.deg2rad(25)),
#                 minimizer_kwargs={'args':(corr_02, corr_13,sampling_rate)}) # does not accept range
#

        if use_correlation:
        # Using correlation
            ll = opt.brute(ll_regular_station, ranges=(slice(ZenLim[0], ZenLim[1], 0.05),
                                                       slice(AziLim[0], AziLim[1], 0.05)),
                            args=(corr_02, corr_13, sampling_rate, positions_pairs), full_output=True, finish=opt.fmin)  # slow but does the trick
        else:
            ll = opt.brute(ll_regular_station_fft, ranges=(slice(ZenLim[0], ZenLim[1], 0.05),
                                                           slice(AziLim[0], AziLim[1], 0.05)),
                           args=(corr_02_fft, corr_13_fft, sampling_rate, positions_pairs), full_output=True, finish=opt.fmin)  # slow but does the trick

        if self.__debug:
            import peakutils
            zenith = ll[0][0]
            azimuth = ll[0][1]
            times = []

            for pos in positions_pairs:
                tmp = []
                tmp.append(geo_utl.get_time_delay_from_direction(zenith, azimuth, pos[0], n=n_index) * sampling_rate)
                tmp.append(geo_utl.get_time_delay_from_direction(zenith, azimuth, pos[1], n=n_index) * sampling_rate)
                times.append(tmp)

            delta_t_02 = times[0][1] - times[0][0]
            delta_t_13 = times[1][1] - times[1][0]

            pos_02 = int(corr_02.shape[0] / 2 - delta_t_02)
            pos_13 = int(corr_13.shape[0] / 2 - delta_t_13)

            toffset = -(np.arange(0, corr_02.shape[0]) - corr_02.shape[0] / 2) / sampling_rate

            fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
            ax.plot(toffset, corr_02)
            ax.axvline(delta_t_02 / sampling_rate, label='time', c='k')
            indices = peakutils.indexes(corr_02, thres=0.8, min_dist=5)
            t02s = toffset[indices]
            ax.plot(toffset[indices], corr_02[indices], 'o')

            ax2.plot(toffset, corr_13)
            indices = peakutils.indexes(corr_13, thres=0.8, min_dist=5)
            ax2.plot(toffset[indices], corr_13[indices], 'o')

            ax2.axvline(delta_t_13 / sampling_rate, label='time', c='k')
            ax2.set_xlabel("time")
            ax2.set_ylabel("Correlation Ch 1/ Ch3", fontsize='small')
            ax.set_ylabel("Correlation Ch 0/ Ch2", fontsize='small')
            plt.tight_layout()

        station[stnp.zenith] = max(ZenLim[0], min(ZenLim[1], ll[0][0]))
        station[stnp.azimuth] = ll[0][1]
        output_str = "reconstucted angles theta = {:.1f}, phi = {:.1f}".format(station[stnp.zenith] / units.deg, station[stnp.azimuth] / units.deg)
        if station.has_sim_station():
            sim_zen = None
            sim_az = None
            if(station.get_sim_station().has_parameter(stnp.zenith)):
                sim_zen = station.get_sim_station()[stnp.zenith]
                sim_az = station.get_sim_station()[stnp.azimuth]
            elif(station.get_sim_station().has_channels()):  # in case of a neutrino simulation, each channel has a slightly different arrival direction -> compute the average
                sim_zen = []
                sim_az = []
                for sim_channels in station.get_sim_station().iter_channels(use_channels=np.array(channel_pairs).flatten()):
                    for sim_channel in sim_channels:
                        if(sim_channel[chp.ray_path_type] == 'direct' or sim_channel[chp.ray_path_type] == 'refracted'):
                            sim_zen.append(sim_channel[chp.zenith])
                            sim_az.append(sim_channel[chp.azimuth])
                sim_zen = np.mean(np.array(sim_zen))
                sim_az = np.mean(np.array(sim_az))

            if(sim_zen is not None):
                dOmega = hp.get_angle(hp.spherical_to_cartesian(sim_zen, sim_az), hp.spherical_to_cartesian(station[stnp.zenith], station[stnp.azimuth]))
                output_str += "  MC theta = {:.1f}, phi = {:.1f},  dOmega = {:.2f}, dZen = {:.01f}, dAz = {:.1f}".format(sim_zen / units.deg, sim_az / units.deg, dOmega / units.deg, (station[stnp.zenith] - sim_zen) / units.deg, (station[stnp.azimuth] - hp.get_normalized_angle(sim_az)) / units.deg)
                self.__zenith.append(sim_zen)
                self.__azimuth.append(sim_az)
                self.__delta_zenith.append(station[stnp.zenith] - sim_zen)
                self.__delta_azimuth.append(station[stnp.azimuth] - hp.get_normalized_angle(sim_az))

        logger.info(output_str)
        # Still have to add fit quality parameter to output

        if self.__debug:
            import peakutils
            # access simulated efield and high level parameters
            sim_present = False
            if(station.has_sim_station()):
                if(station.get_sim_station().has_parameter(stnp.zenith)):
                    sim_station = station.get_sim_station()
                    azimuth_orig = sim_station[stnp.azimuth]
                    zenith_orig = sim_station[stnp.zenith]
                    sim_present = True

            if sim_present:
                logger.debug("True CoREAS zenith {0}, azimuth {1}".format(zenith_orig, azimuth_orig))
            logger.debug("Result of direction fitting: [zenith, azimuth] {}".format(np.rad2deg(ll[0])))

            # Show fit space
            zen = np.arange(ZenLim[0], ZenLim[1], 1 * units.deg)
            az = np.arange(AziLim[0], AziLim[1], 2 * units.deg)

            x_plot = np.zeros(zen.shape[0] * az.shape[0])
            y_plot = np.zeros(zen.shape[0] * az.shape[0])
            z_plot = np.zeros(zen.shape[0] * az.shape[0])
            i = 0
            for a in az:
                for z in zen:
                    # Evaluate fit function for grid
                    if use_correlation:
                        z_plot[i] = ll_regular_station([z, a], corr_02, corr_13, sampling_rate, positions_pairs)
                    else:
                        z_plot[i] = ll_regular_station_fft([z, a], corr_02_fft, corr_13_fft, sampling_rate, positions_pairs)
                    x_plot[i] = a
                    y_plot[i] = z
                    i += 1

            fig, ax = plt.subplots(1, 1)
            ax.scatter(np.rad2deg(x_plot), np.rad2deg(y_plot), c=z_plot, cmap='gnuplot2_r', lw=0)
#             ax.imshow(z_plot, cmap='gnuplot2_r', extent=(0, 360, 90, 180))
            if sim_present:
                ax.plot(np.rad2deg(hp.get_normalized_angle(azimuth_orig)), np.rad2deg(zenith_orig), marker='d', c='g', label="True")
            ax.scatter(np.rad2deg(ll[0][1]), np.rad2deg(ll[0][0]), marker='o', c='k', label='Fit')
#             ax.colorbar(label='Fit parameter')
            ax.set_ylabel('Zenith [rad]')
            ax.set_xlabel('Azimuth [rad]')
            plt.tight_layout()

            # plot allowed solution separately for each pair of channels
            toffset = -(np.arange(0, corr_02.shape[0]) - corr_02.shape[0] / 2.) / sampling_rate
            indices = peakutils.indexes(corr_02, thres=0.8, min_dist=5)
            t02s = toffset[indices][np.argsort(corr_02[indices])[::-1]]
            toffset = -(np.arange(0, corr_13.shape[0]) - corr_13.shape[0] / 2.) / sampling_rate
            indices = peakutils.indexes(corr_13, thres=0.8, min_dist=5)
            t13s = toffset[indices][np.argsort(corr_13[indices])[::-1]]
            from scipy import constants
            c = constants.c * units.m / units.s
            dx = -6 * units.m

            def get_deltat13(dt, phi):
                t = -1. * dt * c / (dx * np.cos(phi) * n_index)
                t[t < 0] = np.nan
                return np.arcsin(t)

            def get_deltat02(dt, phi):
                t = -1 * dt * c / (dx * np.sin(phi) * n_index)
                t[t < 0] = np.nan
                return np.arcsin(t)

            def getDeltaTCone(r, dt):
                dist = np.linalg.norm(r)
                t0 = -dist * n_index / c
                Phic = np.arccos(dt / t0)  # cone angle for allowable solutions
                logger.debug('dist = {}, dt = {}, t0 = {}, phic = {}'.format(dist, dt, t0, Phic))
                nr = r / dist  # normalize
                p = np.cross([0, 0, 1], nr)  # create a perpendicular normal vector to r
                p = p / np.linalg.norm(p)
                q = np.cross(nr, p)  # nr, p, and q form an orthonormal basis
                logger.debug('nr = {}\np = {}\nq = {}\n'.format(nr, p, q))
                ThetaC = np.linspace(0, 2 * np.pi, 1000)
                Phis = np.zeros(len(ThetaC))
                Thetas = np.zeros(len(ThetaC))
                for i, thetac in enumerate(ThetaC):
                    # create a set of vectors that point along the cone defined by r and PhiC
                    rc = nr + np.tan(Phic) * (np.sin(thetac) * p + np.cos(thetac) * q)
                    nrc = rc / np.linalg.norm(rc)
                    theta = np.arccos(nrc[2])
                    phi = np.arctan2(nrc[1], nrc[0])
                    Phis[i] = phi
                    Thetas[i] = theta
                return Phis, Thetas

            # phis = np.deg2rad(np.linspace(0, 360, 10000))
            r0_2 = positions_pairs[0][1] - positions_pairs[0][0]  # vector pointing from Ch2 to Ch0
            r1_3 = positions_pairs[1][1] - positions_pairs[1][0]  # vector pointing from Ch3 to Ch1
            logger.debug('r02 {}\nr13 {}'.format(r0_2, r1_3))
            linestyles = ['-', '--', ':', '-.']
            for i, t02 in enumerate(t02s):
                # theta02 = get_deltat02(t02, phis)
                phi02, theta02 = getDeltaTCone(r0_2, t02)
                theta02[theta02 < 0] += np.pi
                phi02[phi02 < 0] += 2 * np.pi
                jumppos02 = np.where(np.abs(np.diff(phi02)) >= 5.0)
                for j, pos in enumerate(jumppos02):
                    phi02 = np.insert(phi02, pos + 1 + j, np.nan)
                    theta02 = np.insert(theta02, pos + 1 + j, np.nan)
                # mask02 = ~np.isnan(theta02)
                ax.plot(np.rad2deg(phi02), np.rad2deg(theta02), '{}C3'.format(linestyles[i % 4]), label='c 0+2 dt = {}'.format(t02))
                ax.plot(np.rad2deg(phi02), 180 - np.rad2deg(theta02), '{}C3'.format(linestyles[i % 4]))
            for i, t13 in enumerate(t13s):
                # theta13 = get_deltat13(t13, phis)
                phi13, theta13 = getDeltaTCone(r1_3, t13)
                theta13[theta13 < 0] += np.pi
                phi13[phi13 < 0] += 2 * np.pi
                jumppos13 = np.where(np.abs(np.diff(phi13)) >= 5.0)
                for j, pos in enumerate(jumppos13):
                    phi13 = np.insert(phi13, pos + 1 + j, np.nan)
                    theta13 = np.insert(theta13, pos + 1 + j, np.nan)
                # mask13 = ~np.isnan(theta13)
                ax.plot(np.rad2deg(phi13), np.rad2deg(theta13), '{}C2'.format(linestyles[i % 4]), label='c 1+3 dt = {}'.format(t13))
                ax.plot(np.rad2deg(phi13), 180 - np.rad2deg(theta13), '{}C2'.format(linestyles[i % 4]))
            ax.legend(fontsize='small')
            ax.set_ylim(ZenLim[0] / units.deg, ZenLim[1] / units.deg)
            ax.set_xlim(AziLim[0] / units.deg, AziLim[1] / units.deg)

            # plot expectation
            # import expectation as e
            # zenith_expected = np.pi - e.get_arrival_angle(time.mktime(station.get_station_time().timetuple()))
            # ax.plot(225.5, np.rad2deg(zenith_expected), 'xr', label='expectation')

#             plt.legend()

    def end(self):
        fig, ax = plt.subplots(1, 1)
        mask = np.abs(self.__delta_azimuth) < (1 * units.deg)
        ax.scatter(np.array(self.__zenith)[mask] / units.deg, np.array(self.__delta_zenith)[mask] / units.deg, s=20)
        ax.set_xlabel("zenith angle (MC) [deg]")
        ax.set_ylabel("(zenith_rec - zenith_MC) [deg]")
        fig.tight_layout()
        fig.savefig("zenith_bias.png")

        from radiotools import plthelpers as php
        bins = np.arange(-10, 10, .1)
        fig, ax = php.get_histogram(np.array(self.__delta_azimuth) / units.deg, bins=bins, xlabel="delta azimuth [deg]")
        fig.savefig("azimuth.png")
        plt.show()
        pass
