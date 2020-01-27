import scipy.optimize as opt
from scipy import signal, fftpack
import matplotlib.pyplot as plt
import numpy as np
import logging

from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.modules.base.module import register_run

from radiotools import helper as hp


class correlationDirectionFitter:
    """
    Fits the direction using correlation of parallel channels.
    """

    def __init__(self):
        self.__zenith = []
        self.__azimuth = []
        self.__delta_zenith = []
        self.__delta_azimuth = []
        self.logger = logging.getLogger('NuRadioReco.correlationDirectionFitter')
        self.begin()

    def begin(self, debug=False, log_level=None):
        if(log_level is not None):
            self.logger.setLevel(log_level)
        self.__debug = debug

    @register_run()
    def run(self, evt, station, det, n_index=None, ZenLim=[0 * units.deg, 90 * units.deg],
            AziLim=[0 * units.deg, 360 * units.deg],
            channel_pairs=((0, 2), (1, 3)),
            use_envelope=False):
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
        use_envelope: bool (default False)
            if True, the hilbert envelope of the traces is used
        """

        use_correlation = True

        def ll_regular_station(angles, corr_02, corr_13, sampling_rate, positions, trace_start_times):
            """
            Likelihood function for a four antenna ARIANNA station, using correction.
            Using correlation, has no built in wrap around, pulse needs to be in the middle
            """

            zenith = angles[0]
            azimuth = angles[1]
            times = []

            for pos in positions:
                tmp = []
                tmp.append(geo_utl.get_time_delay_from_direction(zenith, azimuth, pos[0], n=n_index))
                tmp.append(geo_utl.get_time_delay_from_direction(zenith, azimuth, pos[1], n=n_index))
                times.append(tmp)

            delta_t_02 = times[0][1] - times[0][0]
            delta_t_13 = times[1][1] - times[1][0]
            # take different trace start times into account
            delta_t_02 -= (trace_start_times[0][1] - trace_start_times[0][0])
            delta_t_13 -= (trace_start_times[1][1] - trace_start_times[1][0])
            delta_t_02 *= sampling_rate
            delta_t_13 *= sampling_rate
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

        def ll_regular_station_fft(angles, corr_02_fft, corr_13_fft, sampling_rate, positions, trace_start_times):
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

            delta_t_02 = (times[0][1] + trace_start_times[0][1] * sampling_rate) - (times[0][0] + trace_start_times[0][0] * sampling_rate)
            delta_t_13 = (times[1][1] + trace_start_times[1][1] * sampling_rate) - (times[1][0] + trace_start_times[1][0] * sampling_rate)

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

        station_id = station.get_id()
        positions = det.get_relative_positions(station_id)
        positions_pairs = [[positions[channel_pairs[0][0]], positions[channel_pairs[0][1]]],
                           [positions[channel_pairs[1][0]], positions[channel_pairs[1][1]]]]
        sampling_rate = station.get_channel(0).get_sampling_rate()  # assume that channels have the same sampling rate
        trace_start_time_pairs = [[station.get_channel(channel_pairs[0][0]).get_trace_start_time(), station.get_channel(channel_pairs[0][1]).get_trace_start_time()],
                                    [station.get_channel(channel_pairs[1][0]).get_trace_start_time(), station.get_channel(channel_pairs[1][1]).get_trace_start_time()]]
        # determine automatically if one channel has an inverted waveform with respect to the other
        signs = [1., 1.]
        for iPair, pair in enumerate(channel_pairs):
            antenna_type = det.get_antenna_type(station_id, pair[0])
            if("LPDA" in antenna_type):
                otheta, ophi, rot_theta, rot_azimuth = det.get_antenna_orientation(station_id, pair[0])
                otheta2, ophi2, rot_theta2, rot_azimuth2 = det.get_antenna_orientation(station_id, pair[1])
                if(np.isclose(np.abs(rot_azimuth - rot_azimuth2), 180 * units.deg, atol=1 * units.deg)):
                    signs[iPair] = -1

        if use_correlation:
            # Correlation
            if not use_envelope:
                corr_02 = signal.correlate(station.get_channel(channel_pairs[0][0]).get_trace(),
                                           signs[0] * station.get_channel(channel_pairs[0][1]).get_trace())
                corr_13 = signal.correlate(station.get_channel(channel_pairs[1][0]).get_trace(),
                                           signs[1] * station.get_channel(channel_pairs[1][1]).get_trace())
            else:
                corr_02 = signal.correlate(np.abs(signal.hilbert(station.get_channel(channel_pairs[0][0]).get_trace())),
                                           np.abs(signal.hilbert(station.get_channel(channel_pairs[0][1]).get_trace())))
                corr_13 = signal.correlate(np.abs(signal.hilbert(station.get_channel(channel_pairs[1][0]).get_trace())),
                                           np.abs(signal.hilbert(station.get_channel(channel_pairs[1][1]).get_trace())))

        else:
            # FFT convolution
            corr_02_fft = fftpack.ifft(-1 * fftpack.fft(station.get_channel(channel_pairs[0][0]).get_trace()).conjugate() * fftpack.fft(station.get_channel(channel_pairs[0][1]).get_trace()))
            corr_13_fft = fftpack.ifft(-1 * fftpack.fft(station.get_channel(channel_pairs[1][0]).get_trace()).conjugate() * fftpack.fft(station.get_channel(channel_pairs[1][1]).get_trace()))

        # Alternative fitters:
#             ll = opt.minimize(ll_regular_station,(zenith_orig-np.deg2rad(15) ,azimuth_orig-np.deg2rad(25)),
#             (corr_02, corr_13,sampling_rate),method='BFGS') # gets stuck in local minima
#
#             ll = opt.basinhopping(ll_regular_station,(zenith_orig-np.deg2rad(15) ,azimuth_orig-np.deg2rad(25)),
#                 minimizer_kwargs={'args':(corr_02, corr_13,sampling_rate)}) # does not accept range
#

        if use_correlation:
        # Using correlation
            ll = opt.brute(ll_regular_station, ranges=(slice(ZenLim[0], ZenLim[1], 0.01),
                                                       slice(AziLim[0], AziLim[1], 0.01)),
                            args=(corr_02, corr_13, sampling_rate, positions_pairs, trace_start_time_pairs),
                            full_output=True, finish=opt.fmin)  # slow but does the trick
#             print(ll)
        else:
            ll = opt.brute(ll_regular_station_fft, ranges=(slice(ZenLim[0], ZenLim[1], 0.05),
                                                           slice(AziLim[0], AziLim[1], 0.05)),
                           args=(corr_02_fft, corr_13_fft, sampling_rate, positions_pairs, trace_start_time_pairs), full_output=True, finish=opt.fmin)  # slow but does the trick

        if self.__debug:
            import peakutils
            zenith = ll[0][0]
            azimuth = ll[0][1]
            times = []

            for pos in positions_pairs:
                tmp = []
                tmp.append(geo_utl.get_time_delay_from_direction(zenith, azimuth, pos[0], n=n_index))
                tmp.append(geo_utl.get_time_delay_from_direction(zenith, azimuth, pos[1], n=n_index))
                times.append(tmp)

            delta_t_02 = times[0][1] - times[0][0]
            delta_t_13 = times[1][1] - times[1][0]
            # take different trace start times into account
            delta_t_02 -= (trace_start_time_pairs[0][1] - trace_start_time_pairs[0][0])
            delta_t_13 -= (trace_start_time_pairs[1][1] - trace_start_time_pairs[1][0])
            delta_t_02 *= sampling_rate
            delta_t_13 *= sampling_rate

            toffset = -(np.arange(0, corr_02.shape[0]) - corr_02.shape[0] / 2) / sampling_rate

            fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
            ax.plot(toffset, corr_02)
            ax.axvline(delta_t_02 / sampling_rate, label='time', c='k')
            indices = peakutils.indexes(corr_02, thres=0.8, min_dist=5)
            t02s = toffset[indices]
            ax.plot(toffset[indices], corr_02[indices], 'o')
            imax = np.argmax(corr_02[indices])
            self.logger.debug("offset 02= {:.3f}".format(toffset[indices[imax]] -  (delta_t_02 / sampling_rate)))

            ax2.plot(toffset, corr_13)
            indices = peakutils.indexes(corr_13, thres=0.8, min_dist=5)
            ax2.plot(toffset[indices], corr_13[indices], 'o')
            imax = np.argmax(corr_13[indices])
#             print("offset 13= {:.3f}".format(toffset[indices[imax]] -  (delta_t_13 / sampling_rate)))

            ax2.axvline(delta_t_13 / sampling_rate, label='time', c='k')

            ax2.set_xlabel("time")
            ax2.set_ylabel("Correlation Ch 1/ Ch3", fontsize='small')
            ax.set_ylabel("Correlation Ch 0/ Ch2", fontsize='small')
            plt.tight_layout()
#             plt.close("all")

        station[stnp.zenith] = max(ZenLim[0], min(ZenLim[1], ll[0][0]))
        station[stnp.azimuth] = ll[0][1]
        output_str = "reconstucted angles theta = {:.1f}, phi = {:.1f}".format(station[stnp.zenith] / units.deg, station[stnp.azimuth] / units.deg)
        if station.has_sim_station():
            sim_zen = None
            sim_az = None
            if(station.get_sim_station().is_cosmic_ray()):
                sim_zen = station.get_sim_station()[stnp.zenith]
                sim_az = station.get_sim_station()[stnp.azimuth]
            elif(station.get_sim_station().is_neutrino()):  # in case of a neutrino simulation, each channel has a slightly different arrival direction -> compute the average
                sim_zen = []
                sim_az = []
                for efield in station.get_sim_station().get_electric_fields_for_channels(ray_path_type='direct'):
                    sim_zen.append(efield[efp.zenith])
                    sim_az.append(efield[efp.azimuth])
                sim_zen = np.array(sim_zen)
                sim_az = hp.get_normalized_angle(np.array(sim_az))
                ops = "average incident zenith {:.1f} +- {:.1f}".format(np.mean(sim_zen) / units.deg, np.std(sim_zen) / units.deg)
                ops += " (individual: "
                for x in sim_zen:
                    ops += "{:.1f}, ".format(x / units.deg)
                ops += ")"
                self.logger.debug(ops)
                ops = "average incident azimuth {:.1f} +- {:.1f}".format(np.mean(sim_az) / units.deg, np.std(sim_az) / units.deg)
                ops += " (individual: "
                for x in sim_az:
                    ops += "{:.1f}, ".format(x / units.deg)
                ops += ")"

                self.logger.debug(ops)
                sim_zen = np.mean(np.array(sim_zen))
                sim_az = np.mean(np.array(sim_az))

            if(sim_zen is not None):
                dOmega = hp.get_angle(hp.spherical_to_cartesian(sim_zen, sim_az), hp.spherical_to_cartesian(station[stnp.zenith], station[stnp.azimuth]))
                output_str += "  MC theta = {:.2f}, phi = {:.2f},  dOmega = {:.2f}, dZen = {:.1f}, dAz = {:.1f}".format(sim_zen / units.deg, hp.get_normalized_angle(sim_az) / units.deg, dOmega / units.deg, (station[stnp.zenith] - sim_zen) / units.deg, (station[stnp.azimuth] - hp.get_normalized_angle(sim_az)) / units.deg)
                self.__zenith.append(sim_zen)
                self.__azimuth.append(sim_az)
                self.__delta_zenith.append(station[stnp.zenith] - sim_zen)
                self.__delta_azimuth.append(station[stnp.azimuth] - hp.get_normalized_angle(sim_az))

        self.logger.info(output_str)
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
                self.logger.debug("True CoREAS zenith {0}, azimuth {1}".format(zenith_orig, azimuth_orig))
            self.logger.debug("Result of direction fitting: [zenith, azimuth] {}".format(np.rad2deg(ll[0])))

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
                        z_plot[i] = ll_regular_station([z, a], corr_02, corr_13, sampling_rate, positions_pairs, trace_start_time_pairs)
                    else:
                        z_plot[i] = ll_regular_station_fft([z, a], corr_02_fft, corr_13_fft, sampling_rate, positions_pairs, trace_start_time_pairs)
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
            t02s = toffset[indices][np.argsort(corr_02[indices])[::-1]] + (trace_start_time_pairs[0][1] - trace_start_time_pairs[0][0])
            toffset = -(np.arange(0, corr_13.shape[0]) - corr_13.shape[0] / 2.) / sampling_rate
            indices = peakutils.indexes(corr_13, thres=0.8, min_dist=5)
            t13s = toffset[indices][np.argsort(corr_13[indices])[::-1]] + (trace_start_time_pairs[1][1] - trace_start_time_pairs[1][0])
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
                self.logger.debug('dist = {}, dt = {}, t0 = {}, phic = {}'.format(dist, dt, t0, Phic))
                nr = r / dist  # normalize
                p = np.cross([0, 0, 1], nr)  # create a perpendicular normal vector to r
                p = p / np.linalg.norm(p)
                q = np.cross(nr, p)  # nr, p, and q form an orthonormal basis
                self.logger.debug('nr = {}\np = {}\nq = {}\n'.format(nr, p, q))
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
            self.logger.debug('r02 {}\nr13 {}'.format(r0_2, r1_3))
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
