from NuRadioReco.modules.base.module import register_run
import numpy as np
import copy
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units
from NuRadioReco.utilities import ice
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import signal_processing
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.electric_field
import matplotlib.pyplot as plt
import logging
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp

logger = logging.getLogger('NuRadioReco.voltageToEfieldConverter')


def get_array_of_channels(station, use_channels, det, zenith, azimuth,
                          antenna_pattern_provider, time_domain=False, efield_position=None):
    """ Get the voltage traces and antenna factors for the electric field reconstruction.

    Parameters
    ----------
    station : `NuRadioReco.framework.station.Station`
        The station object containing the channels and their parameters.
    use_channels : list of int
        The channel ids to use for the electric field reconstruction.
    det : `NuRadioReco.detector.Detector`
        The detector object containing the site and antenna information.
    zenith : float
        The zenith angle of the incoming signal in radians.
    azimuth : float
        The azimuth angle of the incoming signal in radians.
    antenna_pattern_provider : `NuRadioReco.detector.antennapattern.AntennaPatternProvider`
        The antenna pattern provider to get the antenna response functions.
    efield_position : numpy array, optional
        The position where the electric field is calculated - determines time shift of the voltage traces.
        If None, it raises an error.
    time_domain : bool, optional
        If True, returns the time domain traces as well. Default is False.

    Returns
    -------
    times : numpy array
        The time array of the traces.
    efield_antenna_factor : numpy array
        The antenna factor for the electric field at the given position.
    V : numpy array
        The frequency spectrum of the voltage traces for the selected channels.
    V_timedomain : numpy array, optional
        The time domain traces for the selected channels, if `time_domain` is True.
    """


    if efield_position is None:
        raise ValueError(
            "Function signiture changed. Please provide `efield_position`. "
            "To retain old behavior, use `efield_position=np.array([0, 0, 0])`. "
            "The return signature has also changed. Additionally the time vector "
            "of the deconvolved electric field traces is returned as first "
            "argument (see documentation)!")

    t_mins = []
    t_maxs = []
    t_shifts = []

    station_id = station.get_id()
    site = det.get_site(station_id)
    for iCh, channel in enumerate(station.iter_channels(use_channels)):
        channel_id = channel.get_id()

        antenna_position = det.get_relative_position(station_id, channel_id)
        # determine refractive index of signal propagation speed between antennas
        refractive_index = ice.get_refractive_index(1, site)  # if signal comes from above, in-air propagation speed
        if station.is_cosmic_ray():
            if zenith > 0.5 * np.pi:
                refractive_index = ice.get_refractive_index(antenna_position[2], site)  # if signal comes from below, use refractivity at antenna position

        if station.is_neutrino():
            refractive_index = ice.get_refractive_index(antenna_position[2], site)

        time_shift = -geo_utl.get_time_delay_from_direction(zenith, azimuth, antenna_position - efield_position, n=refractive_index)

        t_shifts.append(time_shift)
        t_min = channel.get_trace_start_time() + time_shift
        t_mins.append(t_min)
        t_max = t_min + channel.get_number_of_samples() / channel.get_sampling_rate()
        t_maxs.append(t_max)

    # take the intersection of all channels
    t_min = np.max(t_mins)
    t_max = np.min(t_maxs)

    n_samples = int((t_max - t_min) * channel.get_sampling_rate())
    if n_samples % 2:
        n_samples -= 1

    electric_field_window = NuRadioReco.framework.base_trace.BaseTrace()  # create base trace class to do the fft with correct normalization etc.
    electric_field_window.set_trace(np.zeros(n_samples), channel.get_sampling_rate(), t_min)

    traces = []
    for iCh, channel in enumerate(station.iter_channels(use_channels)):
        channel_copy = copy.copy(channel)  # copy to not modify data structure
        channel_copy.add_trace_start_time(t_shifts[iCh])

        channel_in_window = copy.copy(electric_field_window)
        channel_in_window.add_to_trace(channel_copy)
        traces.append(channel_in_window)


    times = traces[0].get_times()  # assumes that all channels have the same sampling rate
    if time_domain:  # save time domain traces first to avoid extra fft
        V_timedomain = np.zeros((len(use_channels), len(times)))
        for iCh, trace in enumerate(traces):
            V_timedomain[iCh] = trace.get_trace()

    frequencies = traces[0].get_frequencies()  # assumes that all channels have the same sampling rate
    V = np.zeros((len(use_channels), len(frequencies)), dtype=complex)
    for iCh, trace in enumerate(traces):
        V[iCh] = trace.get_frequency_spectrum()

    efield_antenna_factor = signal_processing.get_efield_antenna_factor(
        station, frequencies, use_channels, det, zenith, azimuth, antenna_pattern_provider)

    if time_domain:
        return times, efield_antenna_factor, V, V_timedomain

    return times, efield_antenna_factor, V


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
        self.begin()

    def begin(self):
        self.antenna_provider = antennapattern.AntennaPatternProvider()
        pass

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

        if use_MC_direction:
            zenith = station.get_sim_station()[stnp.zenith]
            azimuth = station.get_sim_station()[stnp.azimuth]
        else:
            logger.info("Using reconstructed (or starting) angles as no signal arrival angles are present")
            zenith = station[stnp.zenith]
            azimuth = station[stnp.azimuth]

        efield_position = np.mean([
            det.get_relative_position(station.get_id(), channel_id)
            for channel_id in use_channels], axis=0)

        times, efield_antenna_factor, V = get_array_of_channels(
            station, use_channels, det, zenith, azimuth, self.antenna_provider, efield_position=efield_position)

        n_frequencies = len(V[0])
        denom = (efield_antenna_factor[0][0] * efield_antenna_factor[-1][1] -
                 efield_antenna_factor[0][1] * efield_antenna_factor[-1][0])
        mask = np.abs(denom) != 0

        # solve it in a vectorized way
        efield3_f = np.zeros((3, n_frequencies), dtype=complex)
        if force_Polarization == 'eTheta':
            efield3_f[1:2, mask] = np.moveaxis(stacked_lstsq(np.moveaxis(efield_antenna_factor[:, 0, mask], 1, 0)[:, :, np.newaxis], np.moveaxis(V[:, mask], 1, 0)), 0, 1)
        elif force_Polarization == 'ePhi':
            efield3_f[2:, mask] = np.moveaxis(stacked_lstsq(np.moveaxis(efield_antenna_factor[:, 1, mask], 1, 0)[:, :, np.newaxis], np.moveaxis(V[:, mask], 1, 0)), 0, 1)
        else:
            efield3_f[1:, mask] = np.moveaxis(stacked_lstsq(np.moveaxis(efield_antenna_factor[:, :, mask], 2, 0), np.moveaxis(V[:, mask], 1, 0)), 0, 1)

        electric_field = NuRadioReco.framework.electric_field.ElectricField(use_channels, efield_position)
        electric_field.set_frequency_spectrum(efield3_f, station.get_channel(use_channels[0]).get_sampling_rate())
        electric_field.set_parameter(efp.zenith, zenith)
        electric_field.set_parameter(efp.azimuth, azimuth)
        electric_field.set_trace_start_time(times[0])
        station.add_electric_field(electric_field)

    def end(self):
        pass
