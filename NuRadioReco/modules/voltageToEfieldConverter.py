from NuRadioReco.modules.base.module import register_run

from NuRadioReco.detector import antennapattern
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.electric_field
from NuRadioReco.framework.parameters import stationParameters as stnp, electricFieldParameters as efp
from NuRadioReco.utilities import geometryUtilities as geo_utl, units, ice

import numpy as np
import copy
import matplotlib.pyplot as plt
import logging
import functools

logger = logging.getLogger('NuRadioReco.voltageToEfieldConverter')


def stacked_lstsq(L, b, rcond=1e-10):
    """
    Solve L x = b, via SVD least squares cutting of small singular values
    L is an array of shape (..., M, N) and b of shape (..., M).
    Returns x of shape (..., N)

    Note that if L is symmetric, it is inverted analytically instead.
    """
    if L.shape[-2] == L.shape[-1]: # try analytic matrix inversion if possible

        if L.shape[-1] == 2: # use explicit formula for matrix inverse
            denom = (L[:,0,0] * L[:,1,1] - L[:,0,1] * L[:,1,0])
            e_theta = (b[:,0] * L[:,1,1] - b[:,1] * L[:,0,1]) / denom
            e_phi = (b[:,1] - L[:,1,0] * e_theta) / L[:,1,1]
            return np.stack((e_theta, e_phi), axis=-1)
        else:
            return np.sum(np.linalg.inv(L) * b[:, None], axis=-1)

    u, s, v = np.linalg.svd(L, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond * s_max
    inv_s = np.zeros_like(s)
    inv_s[s >= s_min] = 1 / s[s >= s_min]
    x = np.einsum('...ji,...j->...i', v,
                  inv_s * np.einsum('...ji,...j->...i', u, b.conj()))
    return np.conj(x, x)


class voltageToEfieldConverter:
    """
    Unfold the electric field from the channel voltages

    This module reconstructs the electric field by solving the system of equation
    that relate the incident electric field via the antenna response functions
    to the measured voltages (see Eq. 4 of the NuRadioReco paper
    https://link.springer.com/article/10.1140/epjc/s10052-019-6971-5).
    The module assumed that the electric field is identical at the antennas/channels
    that are used for the reconstruction. Furthermore, at least two antennas with
    orthogonal polarization response are needed to reconstruct the 3dim electric field.
    Alternatively, the polarization of the resulting efield could be forced to a
    single polarization component. In that case, a single antenna is sufficient.
    """

    def __init__(self):
        self.antenna_provider = None
        self.__freqs = None
        self.__caching = False
        self.begin()

    def begin(self, caching=False):
        self.antenna_provider = antennapattern.AntennaPatternProvider()
        self.__caching = caching
        pass


    @functools.lru_cache(maxsize=1024)
    def _get_cached_antenna_response(self, ant_pattern, zen, azi, *ant_orient):
        """
        Returns the cached antenna reponse for a given antenna pattern, antenna orientation
        and signal arrival direction. This wrapper is necessary as arrays and list are not
        hashable (i.e., can not be used as arguments in functions one wants to cache).
        This module ensures that the cache is clearied if the vector `self.__freqs` changes.
        """
        return ant_pattern.get_antenna_response_vectorized(self.__freqs, zen, azi, *ant_orient)


    @register_run()
    def run(self, evt, station, det, use_channels=None, use_MC_direction=False, force_Polarization=''):
        """
        run method. This function is executed for each event

        Parameters
        ----------
        evt : `NuRadioReco.framework.event.Event`
        station : `NuRadioReco.framework.base_station.BaseStation`
        det : Detector object
        use_channels: array of ints (default: [0, 1, 2, 3])
            the channel ids to use for the electric field reconstruction
        use_MC_direction: bool, default: False
            If True uses zenith and azimuth direction from simulated station.
            Otherwise, uses reconstructed direction from station parameters.
        force_Polarization: str, optional
            If eTheta or ePhi, then only reconstructs chosen polarization of electric field,
            assuming the other is 0. Otherwise (default), reconstructs electric field for both eTheta and ePhi
        """
        if use_channels is None:
            use_channels = [0, 1, 2, 3]

        station_id = station.get_id()

        if use_MC_direction:
            zenith = station.get_sim_station()[stnp.zenith]
            azimuth = station.get_sim_station()[stnp.azimuth]
        else:
            logger.info(
                "Using reconstructed (or starting) angles as no signal arrival angles are present")
            zenith = station[stnp.zenith]
            azimuth = station[stnp.azimuth]

        efield_antenna_factor, spectra = self.get_array_of_channels(
            station, use_channels, det, zenith, azimuth, self.antenna_provider)

        n_frequencies = len(spectra[0])
        denom = (efield_antenna_factor[0][0] * efield_antenna_factor[-1][1] - efield_antenna_factor[0][1] * efield_antenna_factor[-1][0])
        mask = np.abs(denom) != 0

        # solve it in a vectorized way
        efield3_f = np.zeros((3, n_frequencies), dtype=complex)
        if force_Polarization == 'eTheta':
            efield3_f[1:2, mask] = np.moveaxis(
                stacked_lstsq(np.moveaxis(
                    efield_antenna_factor[:, 0, mask], 1, 0)[:, :, np.newaxis], np.moveaxis(spectra[:, mask], 1, 0)), 0, 1)
        elif force_Polarization == 'ePhi':
            efield3_f[2:, mask] = np.moveaxis(
                stacked_lstsq(np.moveaxis(
                    efield_antenna_factor[:, 1, mask], 1, 0)[:, :, np.newaxis], np.moveaxis(spectra[:, mask], 1, 0)), 0, 1)
        else:
            efield3_f[1:, mask] = np.moveaxis(
                stacked_lstsq(np.moveaxis(
                    efield_antenna_factor[:, :, mask], 2, 0), np.moveaxis(spectra[:, mask], 1, 0)), 0, 1)

        efield_position = np.mean([
            det.get_relative_position(station.get_id(), channel_id)
            for channel_id in use_channels], axis=0)

        electric_field = NuRadioReco.framework.electric_field.ElectricField(use_channels, efield_position)
        electric_field.set_frequency_spectrum(
            efield3_f, station.get_channel(use_channels[0]).get_sampling_rate())

        electric_field.set_parameter(efp.zenith, zenith)
        electric_field.set_parameter(efp.azimuth, azimuth)

        # figure out the timing of the E-field
        t_shifts = np.zeros(spectra.shape[0])
        site = det.get_site(station_id)
        if zenith > 0.5 * np.pi:
            logger.warning("Module has not been optimized for neutrino reconstruction yet. Results may be nonsense.")
            refractive_index = ice.get_refractive_index(-1, site)  # if signal comes from below, use refractivity at antenna position
        else:
            refractive_index = ice.get_refractive_index(1, site)  # if signal comes from above, in-air propagation speed

        for i_ch, channel_id in enumerate(use_channels):
            antenna_position = det.get_relative_position(station.get_id(), channel_id)
            t_shifts[i_ch] = station.get_channel(channel_id).get_trace_start_time() - geo_utl.get_time_delay_from_direction(zenith, azimuth, antenna_position, n=refractive_index)

        electric_field.set_trace_start_time(t_shifts.max())
        station.add_electric_field(electric_field)

    def get_array_of_channels(
            self, station, use_channels, det, zenith, azimuth,
            antenna_pattern_provider, return_time_domain=False):

        time_shifts = np.zeros(len(use_channels))
        t_cables = np.zeros(len(use_channels))
        t_geos = np.zeros(len(use_channels))

        station_id = station.get_id()
        site = det.get_site(station_id)
        for iCh, channel in enumerate(station.iter_channels(use_channels)):
            channel_id = channel.get_id()

            antenna_position = det.get_relative_position(station_id, channel_id)
            # determine refractive index of signal propagation speed between antennas
            # if neutrino or signal comes from below, use refractivity at antenna position
            if station.is_neutrino() or zenith > 0.5 * np.pi:
                refractive_index = ice.get_refractive_index(antenna_position[2], site)
            else:
                refractive_index = ice.get_refractive_index(1, site)

            time_shift = -geo_utl.get_time_delay_from_direction(zenith, azimuth, antenna_position, n=refractive_index)
            t_geos[iCh] = time_shift
            t_cables[iCh] = channel.get_trace_start_time()
            logger.debug("time shift channel {}: {:.2f}ns (signal prop), {:.2f}ns (trace start time)".format(channel.get_id(), time_shift, channel.get_trace_start_time()))
            time_shift += channel.get_trace_start_time()
            time_shifts[iCh] = time_shift

        delta_t = time_shifts.max() - time_shifts.min()
        tmin = time_shifts.min()
        tmax = time_shifts.max()
        logger.debug("adding relative station time = {:.0f}ns".format((t_cables.min() + t_geos.max()) / units.ns))
        logger.debug("delta t is {:.2f}".format(delta_t / units.ns))
        trace_length = station.get_channel(use_channels[0]).get_times()[-1] - station.get_channel(use_channels[0]).get_times()[0]

        debug_cut = 0
        if debug_cut:
            fig, ax = plt.subplots(len(use_channels), 1)

        traces = []
        n_samples = None
        for iCh, channel in enumerate(station.iter_channels(use_channels)):
            tstart = delta_t - (time_shifts[iCh] - tmin)
            tstop = tmax - time_shifts[iCh] - delta_t + trace_length
            iStart = int(round(tstart * channel.get_sampling_rate()))
            iStop = int(round(tstop * channel.get_sampling_rate()))
            if n_samples is None:
                n_samples = iStop - iStart
                if n_samples % 2:
                    n_samples -= 1

            trace = copy.copy(channel.get_trace())  # copy to not modify data structure
            trace = trace[iStart:(iStart + n_samples)]
            if debug_cut:
                ax[iCh].plot(trace)

            base_trace = NuRadioReco.framework.base_trace.BaseTrace()  # create base trace class to do the fft with correct normalization etc.
            base_trace.set_trace(trace, channel.get_sampling_rate())
            traces.append(base_trace)

        times = traces[0].get_times()  # assumes that all channels have the same sampling rate
        if return_time_domain:  # save time domain traces first to avoid extra fft
            time_traces = np.zeros((len(use_channels), len(times)))
            for iCh, trace in enumerate(traces):
                time_traces[iCh] = trace.get_trace()

        frequencies = traces[0].get_frequencies()  # assumes that all channels have the same sampling rate
        spectra = np.zeros((len(use_channels), len(frequencies)), dtype=complex)
        for iCh, trace in enumerate(traces):
            spectra[iCh] = trace.get_frequency_spectrum()

        # If we cache the antenna pattern, we need to make sure that the frequencies have not changed
        # between stations. If they have, we need to clear the cache.
        if self.__caching:
            if self.__freqs is None:
                self.__freqs = frequencies
            else:
                if len(self.__freqs) != len(frequencies):
                    self.__freqs = frequencies
                    self._get_cached_antenna_response.cache_clear()
                    logger.warning(
                        "Frequencies have changed (array length). Clearing antenna response cache. "
                        "If you similate neutrinos/in-ice radio emission, this is not surprising. Please disable caching "
                        "By passing `caching==False` to the begin method. If you simulate air showers and this happens often, "
                        "something might be wrong...")
                elif not np.allclose(self.__freqs, frequencies, rtol=0, atol=0.01 * units.MHz):
                    self.__freqs = frequencies
                    self._get_cached_antenna_response.cache_clear()
                    logger.warning(
                        "Frequencies have changed (values). Clearing antenna response cache. "
                        "If you similate neutrinos/in-ice radio emission, this is not surprising. Please disable caching "
                        "By passing `caching==False` to the begin method. If you simulate air showers and this happens often, "
                        "something might be wrong...")

        # efield_antenna_factor = signal_processing.get_efield_antenna_factor(
        #     station, frequencies, use_channels, det, zenith, azimuth, antenna_pattern_provider)

        efield_antenna_factor = np.zeros((len(use_channels), 2, len(frequencies)), dtype=complex)  # from antenna model in e_theta, e_phi
        for iCh, channel_id in enumerate(use_channels):
            zenith_antenna, t_theta, t_phi = geo_utl.fresnel_factors_and_signal_zenith(
                det, station, channel_id, zenith)

            if zenith_antenna is None:
                logger.warning(
                    "Fresnel reflection at air-firn boundary leads to unphysical results, "
                    "no reconstruction possible")
                return None

            logger.debug("angles: zenith {0:.0f}, zenith antenna {1:.0f}, azimuth {2:.0f}".format(
                np.rad2deg(zenith), np.rad2deg(zenith_antenna), np.rad2deg(azimuth)))

            antenna_model = det.get_antenna_model(station.get_id(), channel_id, zenith_antenna)
            antenna_pattern = antenna_pattern_provider.load_antenna_pattern(antenna_model)
            ori = det.get_antenna_orientation(station.get_id(), channel_id)
            if self.__caching:
                vel = self._get_cached_antenna_response(
                        antenna_pattern, zenith_antenna, azimuth, *ori)
            else:
                vel = antenna_pattern.get_antenna_response_vectorized(frequencies, zenith_antenna, azimuth, *ori)

            efield_antenna_factor[iCh] = np.array([vel['theta'] * t_theta, vel['phi'] * t_phi])

        if debug_cut:
            plt.show()

        if return_time_domain:
            return efield_antenna_factor, spectra, time_traces

        return efield_antenna_factor, spectra

    def end(self):
        pass
