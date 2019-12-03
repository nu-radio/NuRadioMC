from NuRadioReco.modules.base.module import register_run
import numpy as np
import os
import copy
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units, fft
from NuRadioReco.utilities import ice
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import trace_utilities
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.electric_field
import matplotlib.pyplot as plt
from scipy import signal

import logging
logger = logging.getLogger('voltageToEfieldConverter')

from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp


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
        time_shift += channel.get_trace_start_time()
        time_shifts[iCh] = time_shift

    delta_t = time_shifts.max() - time_shifts.min()
    tmin = time_shifts.min()
    tmax = time_shifts.max()
    logger.debug("adding relative station time = {:.0f}ns".format((t_cables.min() + t_geos.max()) / units.ns))
    logger.debug("delta t is {:.2f}".format(delta_t / units.ns))
    trace_length = station.get_channel(0).get_times()[-1] - station.get_channel(0).get_times()[0]
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

    efield_antenna_factor = trace_utilities.get_efield_antenna_factor(station, frequencies, use_channels, det, zenith, azimuth, antenna_pattern_provider)

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

    @register_run()
    def run(self, evt, station, det, debug=False, debug_plotpath=None, use_channels=[0, 1, 2, 3], use_MC_direction=False):
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

        if use_MC_direction:
            zenith = station.get_sim_station()[stnp.zenith]
            azimuth = station.get_sim_station()[stnp.azimuth]
            sim_present = True
        else:
            logger.info("Using reconstructed (or starting) angles as no signal arrival angles are present")
            zenith = station[stnp.zenith]
            azimuth = station[stnp.azimuth]
            sim_present = False

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
        E1[mask] = (V[0] * efield_antenna_factor[-1][1] - V[-1] * efield_antenna_factor[0][1])[mask] / denom[mask]
        E2[mask] = (V[-1] - efield_antenna_factor[-1][0] * E1)[mask] / efield_antenna_factor[-1][1][mask]
        # solve it in a vectorized way
        efield3_f = np.zeros((2, n_frequencies), dtype=np.complex)
        efield3_f[:, mask] = np.moveaxis(stacked_lstsq(np.moveaxis(efield_antenna_factor[:, :, mask], 2, 0), np.moveaxis(V[:, mask], 1, 0)), 0, 1)
        # add eR direction
        efield3_f = np.array([np.zeros_like(efield3_f[0], dtype=np.complex),
                             efield3_f[0],
                             efield3_f[1]])

        electric_field = NuRadioReco.framework.electric_field.ElectricField(use_channels, [0, 0, 0])
        electric_field.set_frequency_spectrum(efield3_f, station.get_channel(0).get_sampling_rate())
        electric_field.set_parameter(efp.zenith, zenith)
        electric_field.set_parameter(efp.azimuth, azimuth)
        # figure out the timing of the E-field
        t_shifts = np.zeros(V.shape[0])
        site = det.get_site(station_id)
        if(zenith > 0.5 * np.pi):
            logger.warning("Module has not been optimized for neutrino reconstruction yet. Results may be nonsense.")
            refractive_index = ice.get_refractive_index(-1, site)  # if signal comes from below, use refractivity at antenna position
        else:
            refractive_index = ice.get_refractive_index(1, site)  # if signal comes from above, in-air propagation speed
        for i_ch, channel_id in enumerate(use_channels):
            antenna_position = det.get_relative_position(station.get_id(), channel_id)
            t_shifts[i_ch] = station.get_channel(channel_id).get_trace_start_time() - geo_utl.get_time_delay_from_direction(zenith, azimuth, antenna_position, n=refractive_index)

        electric_field.set_trace_start_time(t_shifts.max())
        station.add_electric_field(electric_field)

    def end(self):
        pass
