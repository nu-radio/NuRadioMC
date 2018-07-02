import numpy as np
import os
import copy
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from NuRadioReco.detector import antennapattern
import NuRadioReco.framework.base_trace
import matplotlib.pyplot as plt
import logging
from NuRadioReco.utilities import ice
logger = logging.getLogger('voltageToEfieldConverter')


def get_array_of_channels(station, use_channels, det, zenith, azimuth,
                          antenna_pattern_provider, time_domain=False):
    time_shifts = np.zeros(len(use_channels))
    t_cables = np.zeros(len(use_channels))
    t_geos = np.zeros(len(use_channels))

    station_id = station.get_id()
    site = det.get_site(station_id)
    n_ice = ice.get_refractive_index(-0.01, site)
    for iCh, channel in enumerate(station.iter_channels(use_channels)):
        channel_id = channel.get_id()

        antenna_position = det.get_relative_position(station_id, channel_id)
        # determine refractive index of signal propagation speed between antennas
        refractive_index = ice.get_refractive_index(1, site)  # if signal comes from above, in-air propagation speed
        if(zenith > 0.5 * np.pi):
            refractive_index = ice.get_refractive_index(antenna_position[2], site)  # if signal comes from below, use refractivity at antenna position
        time_shift = -geo_utl.get_time_delay_from_direction(zenith, azimuth, antenna_position, n=refractive_index)
        t_geos[iCh] = time_shift
        t_cables[iCh] = channel.get_trace_start_time()
        logger.debug("time shift channel {}: {:.2f}ns (signal prop), {:.2f}ns (trace start time)".format(channel.get_id(), time_shift, channel.get_trace_start_time()))
        time_shift -= channel.get_trace_start_time()
        time_shifts[iCh] = time_shift

    delta_t = time_shifts.max() - time_shifts.min()
    tmin = time_shifts.min()
    tmax = time_shifts.max()
#             print('cables ', t_cables)
#             print('geos', t_geos)
    station.add_trace_start_time(t_cables.min() + t_geos.max())
#             print(time_shifts)
    logger.debug("adding relative station time = {:.0f}ns".format((t_cables.min() + t_geos.max()) / units.ns))
    logger.debug("delta t is {:.2f}".format(delta_t / units.ns))
#             print(delta_t - (time_shifts - tmin))
    trace_length = station.get_channels()[0].get_times()[-1] - station.get_channels()[0].get_times()[0]
    debug_cut = 0
    if(debug_cut):
        fig, ax = plt.subplots(len(use_channels), 1)

    traces = []
    n_samples = None
    for iCh, channel in enumerate(station.iter_channels(use_channels)):
        tstart = delta_t - (time_shifts[iCh] - tmin)
        tstop = tmax - time_shifts[iCh] - delta_t + trace_length
        iStart = int(round(tstart * channel.get_sampling_rate()))
        iStop = int(round(tstop * channel.get_sampling_rate()))
        if(n_samples is None):
            n_samples = iStop - iStart
            if(n_samples % 2):
                n_samples -= 1

        trace = copy.copy(channel.get_trace())  # copy to not modify data structure
        trace = trace[iStart:(iStart + n_samples)]
        if(debug_cut):
            ax[iCh].plot(trace)
        base_trace = NuRadioReco.framework.base_trace.BaseTrace()  # create base trace class to do the fft with correct normalization etc.
        base_trace.set_trace(trace, channel.get_sampling_rate())
        traces.append(base_trace)
    times = traces[0].get_times()  # assumes that all channels have the same sampling rate
    if(time_domain):  # save time domain traces first to avoid extra fft
        V_timedomain = np.zeros((len(use_channels), len(times)))
        for iCh, trace in enumerate(traces):
            V_timedomain[iCh] = trace.get_trace()
    frequencies = traces[0].get_frequencies()  # assumes that all channels have the same sampling rate
    V = np.zeros((len(use_channels), len(frequencies)), dtype=np.complex)
    for iCh, trace in enumerate(traces):
        V[iCh] = trace.get_frequency_spectrum()

    efield_antenna_factor = np.zeros((len(use_channels), 2, len(frequencies)), dtype=np.complex)  # from antenna model in e_theta, e_phi
    for iCh, channel in enumerate(station.iter_channels(use_channels)):
        zenith_antenna = zenith
        transmission_parallel = 1.
        transmission_perpendicular = 1.
        # first check case if signal comes from above
        if(zenith <= 0.5 * np.pi):
            # is antenna below surface?
            position = det.get_relative_position(station_id, channel.get_id())
            if(position[2] <= 0):
                zenith_antenna = geo_utl.get_fresnel_angle(zenith, n_ice, 1)
                transmission_parallel = geo_utl.get_fresnel_t_parallel(zenith, n_ice, 1)
                transmission_perpendicular = geo_utl.get_fresnel_t_perpendicular(zenith, n_ice, 1)
                logger.info("channel {:d}: electric field is refracted into the firn. theta {:.0f} -> {:.0f}. Transmission coefficient parallel {:.2f} perpendicular {:.2f}".format(iCh, zenith / units.deg, zenith_antenna / units.deg, transmission_parallel, transmission_perpendicular))
        else:
            # now the signal is coming from below, do we have an antenna above the surface?
            position = det.get_relative_position(station_id, channel.get_id())
            if(position[2] > 0):
                zenith_antenna = geo_utl.get_fresnel_angle(zenith, 1., n_ice)
        if(zenith_antenna is None):
            logger.warning("fresnel reflection at air-firn boundary leads to unphysical results, no reconstruction possible")
            return

        logger.debug("angles: zenith {0:.0f}, zenith antenna {1:.0f}, azimuth {2:.0f}".format(np.rad2deg(zenith), np.rad2deg(zenith_antenna), np.rad2deg(azimuth)))
        antenna_model = det.get_antenna_model(station_id, channel.get_id(), zenith_antenna)
        antenna_pattern = antenna_pattern_provider.load_antenna_pattern(antenna_model)
        ori = det.get_antanna_orientation(station_id, channel.get_id())
        VEL = antenna_pattern.get_antenna_response_vectorized(frequencies, zenith_antenna, azimuth, *ori)
        efield_antenna_factor[iCh] = np.array([VEL['theta'] * transmission_parallel, VEL['phi'] * transmission_perpendicular])

    if(debug_cut):
        plt.show()

    if(time_domain):
        return efield_antenna_factor, V, V_timedomain

    return efield_antenna_factor, V


def stacked_lstsq(L, b, rcond=1e-10):
    """
    Solve L x = b, via SVD least squares cutting of small singular values
    L is an array of shape (..., M, N) and b of shape (..., M).
    Returns x of shape (..., N)
    """
    u, s, v = np.linalg.svd(L, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond * s_max
    inv_s = np.zeros_like(s)
    inv_s[s >= s_min] = 1 / s[s >= s_min]
    x = np.einsum('...ji,...j->...i', v,
                  inv_s * np.einsum('...ji,...j->...i', u, b.conj()))
    return np.conj(x, x)


class voltageToEfieldConverter:

    def __init__(self):
        self.begin()

    def begin(self):
        self.antenna_provider = antennapattern.AntennaPatternProvider()
        pass

    def run(self, evt, station, det, debug=False, debug_plotpath=None, use_channels=[0, 1, 2, 3]):
        """
        run method. This function is executed for each event

        Parameters
        ---------
        evt
        station
        det
        debug: bool
            if True debug plotting is enables
        debug_plotpath: string or None
            if not None plots will be saved to a file rather then shown. Plots will
            be save into the `debug_plotpath` directory
        use_channels: array of ints
            the channel ids to use for the electric field reconstruction
        """
        event_time = station.get_station_time()
        station_id = station.get_id()

        if station.get_sim_station() is not None:
            zenith = station.get_sim_station()['zenith']
            azimuth = station.get_sim_station()['azimuth']
            sim_present = True
        else:
            logger.warning("Using reconstructed angles as no simulation present")
            zenith = station['zenith']
            azimuth = station['azimuth']
            sim_present = False

        channels = station.get_channels()

#         if debug:
#             ax.plot(channel.get_times() / units.ns, trace / units.mV, label="Ch {}".format(channel_id))
#
#             axf.plot(ff[ff < 500 * units.MHz], np.abs(channel.get_frequency_spectrum())[ff < 500 * units.MHz] / units.mV, label="Ch {}".format(channel_id))
#             ax.legend()

        efield_antenna_factor, V = get_array_of_channels(station, use_channels, det, zenith, azimuth, self.antenna_provider)
        n_frequencies = len(V[0])
        denom = (efield_antenna_factor[0][0] * efield_antenna_factor[1][1] - efield_antenna_factor[0][1] * efield_antenna_factor[1][0])
        mask = np.abs(denom) != 0
        # solving for electric field using just two orthorgonal antennas
        E1 = np.zeros_like(V[0])
        E2 = np.zeros_like(V[0])
        E1[mask] = (V[0] * efield_antenna_factor[1][1] - V[1] * efield_antenna_factor[0][1])[mask] / denom[mask]
        E2[mask] = (V[1] - efield_antenna_factor[1][0] * E1)[mask] / efield_antenna_factor[1][1][mask]
        denom = (efield_antenna_factor[0][0] * efield_antenna_factor[-1][1] - efield_antenna_factor[0][1] * efield_antenna_factor[-1][0])
        mask = np.abs(denom) != 0
#         mask[:17] = False
        E1[mask] = (V[0] * efield_antenna_factor[-1][1] - V[-1] * efield_antenna_factor[0][1])[mask] / denom[mask]
        E2[mask] = (V[-1] - efield_antenna_factor[-1][0] * E1)[mask] / efield_antenna_factor[-1][1][mask]
#
#         E3 = np.zeros_like(V[2])
#         E4 = np.zeros_like(V[2])
#         denom = (efield_antenna_factor[0][0] * efield_antenna_factor[1][1] - efield_antenna_factor[0][1] * efield_antenna_factor[1][0])
#         mask = np.abs(denom) != 0
#         E3[mask] = (V[2] * efield_antenna_factor[1][1] - V[3] * efield_antenna_factor[0][1])[mask] / denom[mask]
#         E4[mask] = (V[3] - efield_antenna_factor[1][0] * E3)[mask] / efield_antenna_factor[1][1][mask]
#
#         efield_f = np.zeros((2, n_frequencies), dtype=np.complex)
#         efield_f2 = np.zeros((2, n_frequencies), dtype=np.complex)

        # solve it in a vectorized way
        efield3_f = np.zeros((2, n_frequencies), dtype=np.complex)
        efield3_f[:, mask] = np.moveaxis(stacked_lstsq(np.moveaxis(efield_antenna_factor[:, :, mask], 2, 0), np.moveaxis(V[:, mask], 1, 0)), 0, 1)
        # add eR direction
        efield3_f = np.array([np.zeros_like(efield3_f[0], dtype=np.complex),
                             efield3_f[0],
                             efield3_f[1]])

#         for iF, f in enumerate(frequencies):
#             A = efield_antenna_factor[:, :, iF]
#             b = V[:, iF]
#
#             Q, R = np.linalg.qr(A)  # qr decomposition of A
#             Qb = np.dot(Q.T, b)  # computing Q^T*b (project b onto the range of A)
#             x = np.linalg.solve(R, Qb)  # solving R*x = Q^T*b
#             efield_f[:, iF] = np.array(x)
#
#             solution = linalg.lstsq(A, b, cond=1e-2)
#             x_lstsq = solution[0]
#             efield_f2[:, iF] = np.array(x_lstsq)
#
#             if (f < 500 * units.MHz):
#                 print "%.0f MHz" % (f / units.MHz), solution

        station.set_frequency_spectrum(efield3_f, channels[0].get_sampling_rate())

        if debug:
            fig, (ax2, ax2f) = plt.subplots(2, 1, figsize=(10, 8))
            lw = 2

#             f1, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
#             a0.set_ylabel("VEL [m]")
#             a2.set_ylabel("VEL [m]")
#             a3.set_xlabel("Frequencies [GHz]")
#             a2.set_xlabel("Frequencies [GHz]")
#             f1.suptitle("Antenna model")

#             efield3 = np.fft.irfft(efield3_f, norm="ortho") / 2 ** 0.5
#             efield = np.fft.irfft(efield_f, norm="ortho") / 2 ** 0.5
#             efield2 = np.fft.irfft(efield_f2, norm="ortho") / 2 ** 0.5
#             efield21 = np.fft.irfft(E1, norm="ortho") / 2 ** 0.5
#             efield22 = np.fft.irfft(E2, norm="ortho") / 2 ** 0.5
#             efield31 = np.fft.irfft(E3, norm="ortho") / 2 ** 0.5

            times = station.get_times() / units.ns
#             ax2.plot(times, efield21 / units.mV * units.m, ":C2", label="exact solution Ch 0+1")
#             ax2.plot(times, efield31 / units.mV * units.m, ":C3", label="exact solution Ch 2+3")
            ax2.plot(times, station.get_trace()[1] / units.mV * units.m, "-C0", label="reconstructed eTheta", lw=lw)
            ax2.plot(times, station.get_trace()[2] / units.mV * units.m, "-C1", label="reconstructed ePhi", lw=lw)
            ax2.set_xlim(400, 600)
#             ax2.plot(times, efield21 / units.mV * units.m, "-C2", label="ch0+1 eTheta", lw=lw)
#             ax2.plot(times, efield22 / units.mV * units.m, "-C3", label="ch0+1 ePhi", lw=lw)
#             ax2.set_xlim(times.min(), times.max())

            ff = station.get_frequencies() / units.MHz
            ax2f.plot(ff[ff < 500], np.abs(station.get_frequency_spectrum()[1][ff < 500]) / units.mV * units.m, "-C0", label="4 stations lsqr eTheta", lw=lw)
            ax2f.plot(ff[ff < 500], np.abs(station.get_frequency_spectrum()[2][ff < 500]) / units.mV * units.m, "-C1", label="4 stations lsqr ePhi", lw=lw)

            if station.has_sim_station():
                sim_station = station.get_sim_station()
                logger.debug("station start time {:.1f}ns, relativ sim station time = {:.1f}".format(station.get_trace_start_time(), sim_station.get_trace_start_time()))
                # ax2.plot(times_sim / units.ns, efield_sim[0] / units.mV * units.m, "--", label="simulation eR")
                ax2.plot(sim_station.get_times() / units.ns, sim_station.get_trace()[1] / units.mV * units.m, "--C2", label="simulation eTheta", lw=lw)
                ax2.plot(sim_station.get_times() / units.ns, sim_station.get_trace()[2] / units.mV * units.m, "--C3", label="simulation ePhi", lw=lw)
                ax2f.plot(sim_station.get_frequencies() / units.MHz, np.abs(sim_station.get_frequency_spectrum()[1] / units.mV * units.m), "--C2", label="simulation eTheta", lw=lw)
                ax2f.plot(sim_station.get_frequencies() / units.MHz, np.abs(sim_station.get_frequency_spectrum()[2] / units.mV * units.m), "--C3", label="simulation ePhi", lw=lw)

            ax2.legend(fontsize="xx-small")
            ax2.set_xlabel("time [ns]")
            ax2.set_ylabel("electric-field [mV/m]")
#             axf.set_xlabel("Frequency [MHz]")
            ax2f.set_ylim(1e-3, 5)
            ax2f.set_xlabel("Frequency [MHz]")
            ax2f.set_xlim(100, 500)
            ax2f.semilogy(True)
            if sim_present:
                sim = station.get_sim_station()
                fig.suptitle("Simulation: Zenith {:.1f}, Azimuth {:.1f}".format(np.rad2deg(sim["zenith"]), np.rad2deg(sim["azimuth"])))
            else:
                fig.suptitle("Data: reconstructed zenith {:.1f}, azimuth {:.1f}".format(np.rad2deg(zenith), np.rad2deg(azimuth)))
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)
            if(debug_plotpath is not None):
                fig.savefig(os.path.join(debug_plotpath, 'run_{:05d}_event_{:06d}_efield.png'.format(evt.get_run_number(), evt.get_id())))
                plt.close(fig)

            # plot antenna response and channels
            fig, ax = plt.subplots(len(V), 2, sharex='col', sharey='col')
            for iCh in range(len(V)):
                ax[iCh, 0].plot(ff, np.abs(efield_antenna_factor[iCh][0]), label="theta, channel {}".format(use_channels[iCh]), lw=lw)
                ax[iCh, 0].plot(ff, np.abs(efield_antenna_factor[iCh][1]), label="phi, channel {}".format(use_channels[iCh]), lw=lw)
                ax[iCh, 0].legend(fontsize='xx-small')
                ax[iCh, 0].set_xlim(0, 500)
                ax[iCh, 1].set_xlim(400, 600)
                ax[iCh, 1].plot(times, np.fft.irfft(V[iCh], norm='ortho') / 2 ** 0.5 / units.micro / units.V, lw=lw)
                ax[iCh, 0].set_ylabel("H [m]")
                ax[iCh, 1].set_ylabel(r"V [$\mu$V]")
                RMS = det.get_noise_RMS(station.get_id(), 0)
                ax[iCh, 1].text(0.6, 0.8, 'S/N={:.1f}'.format(np.max(np.abs(np.fft.irfft(V[iCh], norm='ortho') / 2 ** 0.5)) / RMS), transform=ax[iCh, 1].transAxes)
            ax[-1, 1].set_xlabel("time [ns]")
            ax[-1, 0].set_xlabel("frequency [MHz]")
            fig.tight_layout()
            if(debug_plotpath is not None):
                fig.savefig(os.path.join(debug_plotpath, 'run_{:05d}_event_{:06d}_channels.png'.format(evt.get_run_number(), evt.get_id())))
                plt.close(fig)

#             f1.tight_layout()
#             f1.subplots_adjust(top=0.95)

    def end(self):
        pass
