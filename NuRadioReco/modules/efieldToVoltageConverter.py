import numpy as np
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from NuRadioReco.utilities import ice
# from detector import antennamodel
from NuRadioReco.detector import antennapattern
from radiotools import coordinatesystems
import copy
import time
import logging
import NuRadioReco.framework.channel
logger = logging.getLogger('efieldToVoltageConverter')


class efieldToVoltageConverter:
    """
    Module that should be used to convert simulations to data.
    It assumes that an efield is given as input and creates the channels of a station.
    It will make the voltages according to the station id that has been set in the reader.
    The station id, defines antenna location and antenna type.
    It also shifts the traces according to an arrival direction. Note that the trace has to be upsampled significantly,
    in order for the arrival direction shift to be accurate. A sampling of more than 50 GHz is required to be reasonably
    accurate.
    """

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self, debug=False, uncertainty={}):
        self.__debug = debug
        self.__uncertainty = uncertainty
        # some uncertainties are systematic, fix them here
        if('sys_dx' in self.__uncertainty):
            self.__uncertainty['sys_dx'] = np.random.normal(0, self.__uncertainty['sys_dx'])
        if('sys_dy' in self.__uncertainty):
            self.__uncertainty['sys_dy'] = np.random.normal(0, self.__uncertainty['sys_dy'])
        if('sys_dz' in self.__uncertainty):
            self.__uncertainty['sys_dz'] = np.random.normal(0, self.__uncertainty['sys_dz'])
        if('sys_amp'in self.__uncertainty):
            for iCh in self.__uncertainty['sys_amp'].keys():
                self.__uncertainty['sys_amp'][iCh] = np.random.normal(1, self.__uncertainty['sys_amp'][iCh])
        self.antenna_provider = antennapattern.AntennaPatternProvider()

    def run(self, evt, station, det):
        t = time.time()

        # access simulated efield and high level parameters
        sim_station = station.get_sim_station()
        sim_station_id = sim_station.get_id()
        azimuth = sim_station['azimuth']
        zenith = sim_station['zenith']
        event_time = sim_station.get_station_time()

        nChannels = det.get_number_of_channels(sim_station_id)
        sampling = 1. / sim_station.get_sampling_rate()

        if(self.__debug):
            efield = sim_station.get_trace()  # in on-sky coordinates, times, e_r, e_phi, e_theta
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)
            ax.plot(efield[0], label='eR')
            ax.plot(efield[1], label='eTheta')
            ax.plot(efield[2], label='ePhi')
            ax.legend()
            ax.set_title("electric field before antenne response")
#             plt.show()

        ff = sim_station.get_frequencies()
        site = det.get_site(sim_station_id)
        n_ice = ice.get_refractive_index(-0.01, site)

        for iCh in range(nChannels):
            efield_fft = copy.copy(sim_station.get_frequency_spectrum())  # we make a copy to not alter the original efield if reflectios off the boundary are taken into account
            # we first check if we have a fresnel refraction at air-firn boundary
            zenith_antenna = zenith
            # first check case if signal comes from above
            if(zenith <= 0.5 * np.pi):
                # is antenna below surface?
                position = det.get_relative_position(sim_station_id, iCh)
                if(position[2] <= 0):
                    # signal comes from above and antenna is in the firn
                    zenith_antenna = geo_utl.get_fresnel_angle(zenith, n_ice, 1)
                    t_parallel = geo_utl.get_fresnel_t_parallel(zenith, n_ice, 1)
                    t_perpendicular = geo_utl.get_fresnel_t_perpendicular(zenith, n_ice, 1)
                    efield_fft[1] *= t_parallel  # eTheta is parallel to the incident plane
                    efield_fft[2] *= t_perpendicular  # ePhi is perpendicular to the incident plane
                    logger.info("channel {:d}: electric field is refracted into the firn. theta {:.0f} -> {:.0f}. Transmission coefficient parallel {:.2f} perpendicular {:.2f}".format(iCh, zenith / units.deg, zenith_antenna / units.deg, t_parallel, t_perpendicular))

                    # ##DEBUG
                    if 0:
                        # correct for reflected signal
                        cs = coordinatesystems.cstrafo(zenith, azimuth)
                        efield_fft[0] = np.zeros_like(efield_fft[0])
                        if(self.__debug):
                            fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
                            ax[0].plot(ff / units.MHz, np.abs(efield_fft[0]), label='eR')
                            ax[0].plot(ff / units.MHz, np.abs(efield_fft[1]), label='eTheta')
                            ax[0].plot(ff / units.MHz, np.abs(efield_fft[2]), label='ePhi')
                            ax[0].set_title("theta = {:.0f} -> {:.0f} , phi = {:.0f}".format(zenith / units.deg, zenith_antenna / units.deg, azimuth / units.deg))
    #                     efield_fft = cs.transform_from_onsky_to_ground(efield_fft)
                        if(self.__debug):
                            ax[1].plot(ff / units.MHz, np.abs(efield_fft[0]), 'C0-', label='x')
                            ax[1].plot(ff / units.MHz, np.abs(efield_fft[1]), 'C1-', label='y')
                            ax[1].plot(ff / units.MHz, np.abs(efield_fft[2]), 'C2-', label='z')
                            ax[1].set_xlim(0, 500)
                            ax[0].legend()
                            ax[1].legend()
                        t_parallel = geo_utl.get_fresnel_t_parallel(zenith, n_ice, 1)
                        t_perpendicular = geo_utl.get_fresnel_t_perpendicular(zenith, n_ice, 1)
    #                     efield_fft[0] *= t_parallel
    #                     efield_fft[1] *= t_parallel
    #                     efield_fft[2] *= t_perpendicular
                        # parallel and perpendicular are with respect to the plane of incident and NOT the surface!
                        efield_fft[1] *= t_parallel
                        efield_fft[2] *= t_perpendicular
                        if(self.__debug):
                            ax[1].plot(ff / units.MHz, np.abs(efield_fft[0]), 'C0--')
                            ax[1].plot(ff / units.MHz, np.abs(efield_fft[1]), 'C1--')
                            ax[1].plot(ff / units.MHz, np.abs(efield_fft[2]), 'C2--')
                            ax[1].set_xlabel("frequency [MHz]")
                        efield_fft = cs.transform_from_onsky_to_ground(efield_fft)
                        cs2 = coordinatesystems.cstrafo(zenith_antenna, azimuth)
                        efield_fft = cs2.transform_from_ground_to_onsky(efield_fft)  # backtransformation with refracted zenith angle
                        if(self.__debug):
                            ax[0].plot(ff / units.MHz, np.abs(efield_fft[0]), 'C0--')
                            ax[0].plot(ff / units.MHz, np.abs(efield_fft[1]), 'C1--')
                            ax[0].plot(ff / units.MHz, np.abs(efield_fft[2]), 'C2--')
                            plt.show()

            else:
                # now the signal is coming from below, do we have an antenna above the surface?
                position = det.get_relative_position(sim_station_id, iCh)
                if(position[2] > 0):
                    zenith_antenna = geo_utl.get_fresnel_angle(zenith, 1., n_ice)
                    if(zenith_antenna is not None):
                        logger.debug('refracting out of the ice {:.1f} -> {:.1f}'.format(zenith / units.deg, zenith_antenna / units.deg))
            if(zenith_antenna is None):
                logger.warning("fresnel reflection at air-firn boundary leads to unphysical results, setting channel {} to zero".format(iCh))
                channel = NuRadioReco.framework.channel.Channel(iCh)
                channel.set_trace(np.zeros(sim_station.get_trace().shape[-1]), 1. / sampling)
                station.add_channel(channel)
            else:
                # get antenna pattern for current channel
                antenna_model = det.get_antenna_model(sim_station_id, iCh, zenith)
                antenna_pattern = self.antenna_provider.load_antenna_pattern(antenna_model)
                ori = det.get_antanna_orientation(sim_station_id, iCh)
                VEL = antenna_pattern.get_antenna_response_vectorized(ff, zenith_antenna, azimuth, *ori)

                # window function
#                 b, a = scipy.signal.butter(10, 500 * units.MHz, 'low', analog=True)
#                 b, a = scipy.signal.butter(4, [50 * units.MHz, 500 * units.MHz], 'bandpass', analog=True)
#                 w, h = scipy.signal.freqs(b, a, ff)
#                 b, a = scipy.signal.cheby2(10, 60, 600 * units.MHz, 'low', analog=True)
#                 w, h = scipy.signal.freqs(b, a, ff)
                # Apply antenna response to electric field
                voltage_fft = efield_fft[2] * VEL['phi'] + efield_fft[1] * VEL['theta']
                if(self.__debug):
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(ff, np.abs(VEL['phi']), label='phi')
                    ax.plot(ff, np.abs(VEL['theta']), label='theta')

    #                 from detector import antennaTimedomain as aT
    #                 spec = aT.get_antenna_response(0, ff)
    #                 ax.plot(ff, np.abs(spec), label='Time domain measurement')

                    ax.set_title("antenna response channel {}".format(iCh))
                    ax.set_xlim(0, 0.6)
                    ax.legend()
                # Remove DC offset
                voltage_fft[np.where(ff < 5 * units.MHz)] = 0.
                if('amp' in self.__uncertainty):
                    voltage_fft *= np.random.normal(1, self.__uncertainty['amp'][iCh])
                if('sys_amp' in self.__uncertainty):
                    voltage_fft *= self.__uncertainty['sys_amp'][iCh]
                channel = NuRadioReco.framework.channel.Channel(iCh)
                channel.set_frequency_spectrum(voltage_fft, 1. / sampling)

                voltage = channel.get_trace()

                # calculate time shift from antenna position and arrival direction
                antenna_position = det.get_relative_position(sim_station_id, iCh)
                if('sys_dx' in self.__uncertainty):
                    antenna_position[0] += self.__uncertainty['sys_dx']
                if('sys_dy' in self.__uncertainty):
                    antenna_position[1] += self.__uncertainty['sys_dy']
                if('sys_dz' in self.__uncertainty):
                    antenna_position[2] += self.__uncertainty['sys_dz']
                # determine refractive index of signal propagation speed between antennas
                refractive_index = ice.get_refractive_index(1, site)  # if signal comes from above, in-air propagation speed
                if(zenith > 0.5 * np.pi):
                    # if signal comes from below, use refractivity at antenna position
                    # for antennas above the surface, the relevant index of refraction is the one for slightly below the surface
                    refractive_index = ice.get_refractive_index(min(-1, antenna_position[2]), site)
                time_shift = geo_utl.get_time_delay_from_direction(zenith, azimuth, antenna_position, n=refractive_index)
                if('dt' in self.__uncertainty):
                    time_shift += np.random.normal(0, self.__uncertainty['dt'])
                time_shift_samples = int(round(time_shift / sampling))  # Check ???
                logger.debug("Shifting channel {} by {:.3}ns = {:.2f}samples = {}samples (rounded) -> error {:.4f}ns, using n = {:.3f}".format(iCh, time_shift,
                      time_shift / sampling, time_shift_samples, (time_shift - time_shift_samples * sampling) / units.ns, refractive_index))
                voltage = np.roll(voltage, time_shift_samples)
                channel.set_trace(voltage, channel.get_sampling_rate())

                if(self.__debug):
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(voltage)
                    ax.set_title("voltage trace channel {}".format(iCh))

                station.add_channel(channel)
#         if(self.__debug):
#             plt.show()
        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
