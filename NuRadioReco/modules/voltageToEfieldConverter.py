from NuRadioReco.modules.base.module import register_run
import numpy as np
import copy
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units, fft
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


def get_array_of_channels(station, use_channels, det, zenith, azimuth, antenna_pattern_provider,
                          time_domain=False, efield_position=None, propagator=None, travel_time_delays=None):
    """ Get the time-alligned voltage traces and antenna factors for electric field reconstruction using unfolding.

    If the efield_position is at the same position as the antennas, the travel time delays are zero. Otherwise, the travel time delays are calculated

    If station.is_cosmic_ray() is True, the travel time delays are calculated from the geometry of the station and
    the direction of the incoming signal using the refractive index of air or shallow ice assuming a plane wave.

    If station.is_neutrino() is True, a propagator must be provided to calculate the travel time delays, using
    station.get_parameter(stnp.nu_vertex) and a previously reconstructed (or MC true) absolute vertex position must
    hence be saved in stnp.nu_vertex.

    Alternatively, the travel time delays can be provided directly via the travel_time_delays argument, which overrides
    the automatic calculation of the travel time delays. This is useful for geometries not supported by the automatic
    calculation, e.g. plane waves in air observed with deep antennas.

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
    propagator : `NuRadioMC.SignalProp.propagation_base_class.ray_tracing_base`, optional
        Propagator module used to calculate the travel times for a neutrino event.
        An instance of ``ray_tracing_base`` or any class derived from it.
    travel_time_delays : numpy array, optional
        The travel time delays for each channel in use_channels. If not provided (default), the travel time
        delays are calculated from the geometry of the station and the direction of the incoming signal. Must
        have same length as use_channels if provided.

    Returns
    -------
    times : numpy array
        The time array of the traces.
    efield_antenna_factor : numpy array
        The antenna factor for the electric field at the given position.
    V_freq_domain : numpy array
        The frequency spectrum of the voltage traces for the selected channels.
    V_time_domain : numpy array, optional
        The time domain traces for the selected channels, if `time_domain` is True.
    """


    if efield_position is None:
        raise ValueError(
            "Function signature changed. Please provide `efield_position`. "
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

        if travel_time_delays is None:

            if np.linalg.norm(det.get_relative_position(station_id, channel_id) - efield_position) < 0.01 * units.mm:
                time_shift = 0.0

            elif station.is_cosmic_ray():
                antenna_position = det.get_relative_position(station_id, channel_id) # here, only the relative position is needed

                # determine refractive index of signal propagation speed between antennas
                if zenith <= 0.5 * np.pi:
                    refractive_index = ice.get_refractive_index(1, site)  # if signal comes from above, in-air propagation speed
                elif zenith > 0.5 * np.pi:
                    refractive_index = ice.get_refractive_index(antenna_position[2], site)  # if signal comes from below, use refractivity at antenna position

                time_shift = -geo_utl.get_time_delay_from_direction(zenith, azimuth, antenna_position - efield_position, n=refractive_index)

            elif station.is_neutrino():
                if propagator is None:
                    raise ValueError("For neutrino events, a propagator must be provided.")

                # Calculate the travel time delays using the propagator
                vertex_position = station.get_parameter(stnp.nu_vertex)
                antenna_position = det.get_relative_position(station_id, channel_id) + det.get_absolute_position(station.get_id()) # assuming that the vertex_position is absolute, the absolute position is needed here
                propagator.set_start_and_end_point(vertex_position, antenna_position)
                propagator.find_solutions()
                if propagator.get_number_of_solutions() > 0:
                    travel_time = propagator.get_travel_time(0) # using first solution
                else:
                    raise ValueError(f"No propagation solution found for channel {channel_id}. Please check the geometry. Vertex: {vertex_position}, Antenna: {antenna_position}.")

                # Warning if ray-traced arrival direction is not consistent with zenith and azimuth
                ray_traced_direction = propagator.get_receive_vector(0)
                ray_traced_zenith = np.arccos(ray_traced_direction[2])
                ray_traced_azimuth = np.arctan2(ray_traced_direction[1], ray_traced_direction[0])
                if not (np.isclose(ray_traced_zenith, zenith, atol=3 * units.deg) and np.isclose(ray_traced_azimuth, azimuth, atol=3 * units.deg)):
                    logger.warning(
                        f"Ray-traced arrival direction (zenith: {ray_traced_zenith}, azimuth: {ray_traced_azimuth}) is not "
                        f"consistent with the provided zenith ({zenith / units.deg}) and azimuth ({azimuth / units.deg}) angles. "
                        f"Please check the geometry and the reconstruction and provided angles. Vertex: {vertex_position}, Antenna: {antenna_position}."
                    )

                time_shift = -travel_time

            else:
                raise ValueError(
                    "Please set station.set_is_cosmic_ray(True) for cosmic ray events (plane wave air or shallow ice), "
                    "station.set_is_neutrino(True) for neutrino events (deep ice), or provide user-defined travel_time_delays."
                    )

        else:
            time_shift = -travel_time_delays[iCh]

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

    channels = []
    for iCh, channel in enumerate(station.iter_channels(use_channels)):
        channel_copy = copy.copy(channel)  # copy to not modify data structure
        channel_copy.add_trace_start_time(t_shifts[iCh])

        channel_in_window = copy.copy(electric_field_window)
        channel_in_window.add_to_trace(channel_copy)
        channels.append(channel_in_window)


    times = channels[0].get_times()  # assumes that all channels have the same sampling rate
    if time_domain:  # save time domain traces first to avoid extra fft
        V_time_domain = np.zeros((len(use_channels), len(times)))
        for iCh, trace in enumerate(channels):
            V_time_domain[iCh] = trace.get_trace()

    frequencies = channels[0].get_frequencies()  # assumes that all channels have the same sampling rate
    V_freq_domain = np.zeros((len(use_channels), len(frequencies)), dtype=complex)
    for iCh, trace in enumerate(channels):
        V_freq_domain[iCh] = trace.get_frequency_spectrum()

    efield_antenna_factor = signal_processing.get_efield_antenna_factor(
        station, frequencies, use_channels, det, zenith, azimuth, antenna_pattern_provider)

    if time_domain:
        return times, efield_antenna_factor, V_freq_domain, V_time_domain

    return times, efield_antenna_factor, V_freq_domain


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

    The module automatically calculates the relavant time delays for channels, if they are not at the same position.

    If station.is_cosmic_ray() is True, the travel time delays are calculated from the geometry of the station and
    the direction of the incoming signal using the refractive index of air or shallow ice assuming a plane wave.

    If station.is_neutrino() is True, a propagator must be provided to calculate the travel time delays, using
    station.get_parameter(stnp.nu_vertex) and a previously reconstructed (or MC true) absolute vertex position must
    hence be saved in stnp.nu_vertex.

    Alternatively, the travel time delays can be provided directly via the travel_time_delays argument, which overrides
    the automatic calculation of the travel time delays. This is useful for geometries not supported by the automatic
    calculation, e.g. plane waves in air observed with deep antennas.
    """

    def __init__(self):
        self.antenna_provider = None
        self.begin()

    def begin(self, debug=False, hann_window_duration=100 * units.ns, propagator=None):
        """
        Initialize the voltageToEfieldConverter.

        Parameters
        ----------
        debug : bool, optional
            If True, enables debug mode.
        hann_window_duration : float, optional
            A Hann window is applied to the edges of the voltage traces to reduce artifacts from the
            FFT and unfolding. The default duration is 100 ns at each edge, but can be set to 0 to disable the Hann window.
        propagator : `NuRadioMC.SignalProp.propagation_base_class.ray_tracing_base`, optional
            A propagator object to calculate the travel time delays for neutrino events.
            An instance of ``ray_tracing_base`` or any class derived from it.
        """
        self.antenna_provider = antennapattern.AntennaPatternProvider()
        self.debug = debug
        self.hann_window_duration = hann_window_duration
        self.propagator = propagator
        pass

    @register_run()
    def run(self, evt, station, det, use_channels=None, use_MC_direction=False, force_Polarization='', travel_time_delays=None):
        """
        run method. This function is executed for each event

        Parameters
        ----------
        evt : `NuRadioReco.framework.event.Event`
        station : `NuRadioReco.framework.base_station.BaseStation`
        det : Detector object
        use_channels: array of ints
            The channel ids to use for the electric field reconstruction
        use_MC_direction: bool, default: False
            If True uses zenith and azimuth direction from simulated station.
            Otherwise, uses reconstructed direction from station parameters.
        force_Polarization: str, optional
            If eTheta or ePhi, then only reconstructs chosen polarization of electric field,
            assuming the other is 0. Otherwise (default), reconstructs electric field for both eTheta and ePhi
        travel_time_delays: array of floats, optional
            If provided, the travel time delays for each channel in use_channels are used to shift the
            voltage traces to the same reference position. This is useful for situations where ray-tracing
            is needed to determine the travel time delays, e.g. for antennas in deep ice. If not provided,
            the travel time delays are calculated from the geometry of the station and the direction of the
            incoming signal using the refractive index of air or shallow ice.
        """
        if use_channels is None:
            msg = ("No channels specified for electric field reconstruction. "
                "Please provide `use_channels` argument to the voltageToEfieldConverter.")
            logger.error(msg)
            raise ValueError(msg)
        elif len(use_channels) < 2 and not force_Polarization:
            msg = ("At least two channels with orthogonal antenna response are needed for electric field reconstruction. "
                "Please provide at least two channels in `use_channels` or set `force_Polarization` to 'eTheta' or 'ePhi'.")
            logger.error(msg)
            raise ValueError(msg)

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

        times, efield_antenna_factor, traces_freq_domain = get_array_of_channels(
            station, use_channels, det, zenith, azimuth, self.antenna_provider, efield_position=efield_position, propagator=self.propagator, travel_time_delays=travel_time_delays)

        if self.hann_window_duration > 0:
            hann_window_samples = int(self.hann_window_duration * station.get_channel(use_channels[0]).get_sampling_rate())
            traces_time_domain = fft.freq2time(traces_freq_domain, station.get_channel(use_channels[0]).get_sampling_rate())
            hann_window = signal_processing.half_hann_window(traces_time_domain.shape[1], hann_window_length=hann_window_samples)
            traces_time_domain[:, :] *= hann_window[None, :]
            traces_freq_domain = fft.time2freq(traces_time_domain, station.get_channel(use_channels[0]).get_sampling_rate())

        if self.debug:
            fig, ax = plt.subplots(len(use_channels), 1, figsize=(10, 2*len(use_channels)))
            ax[0].set_title(f'Station {station.get_id()} - Event {evt.get_id()} - Run {evt.get_run_number()} - Shifted Voltage Traces')
            traces_time_domain = fft.freq2time(traces_freq_domain, station.get_channel(use_channels[0]).get_sampling_rate())
            for i_ch in range(len(traces_freq_domain)):
                ax[i_ch].plot(times, traces_time_domain[i_ch,:], label=f'Channel {use_channels[i_ch]}')
                ax[i_ch].set_ylabel('Voltage [V]')
                ax[i_ch].set_xlim(times[0], times[-1])
            ax[-1].set_xlabel('Time [ns]')
            ax[0].legend()
            fig.tight_layout()
            plt.savefig("debug_voltage_traces.png")

            fig, ax = plt.subplots(len(use_channels), 1, figsize=(10, 2*len(use_channels)))
            frequencies = np.fft.rfftfreq(len(times), d=1/station.get_channel(use_channels[0]).get_sampling_rate())
            ax[0].set_title(f'Station {station.get_id()} - Event {evt.get_id()} - Run {evt.get_run_number()} - Shifted Voltage Traces')
            for i_ch in range(len(traces_freq_domain)):
                ax[i_ch].plot(frequencies, abs(efield_antenna_factor[i_ch,0,:]), label=f'Channel {use_channels[i_ch]}, theta')
                ax[i_ch].plot(frequencies, abs(efield_antenna_factor[i_ch,1,:]), label=f'Channel {use_channels[i_ch]}, phi')
                ax[i_ch].set_ylabel('Voltage [V]')
                ax[i_ch].set_xlim(frequencies[0], frequencies[-1])
            ax[-1].set_xlabel('Frequency [GHz]')
            ax[0].legend()
            fig.tight_layout()
            plt.savefig("debug_antenna_factors.png")

        n_frequencies = len(traces_freq_domain[0])
        denom = (efield_antenna_factor[0][0] * efield_antenna_factor[-1][1] -
                 efield_antenna_factor[0][1] * efield_antenna_factor[-1][0])
        mask = np.abs(denom) != 0

        # solve it in a vectorized way
        efield3_f = np.zeros((3, n_frequencies), dtype=complex)
        if force_Polarization == 'eTheta':
            efield3_f[1:2, mask] = np.moveaxis(stacked_lstsq(np.moveaxis(efield_antenna_factor[:, 0, mask], 1, 0)[:, :, np.newaxis], np.moveaxis(traces_freq_domain[:, mask], 1, 0)), 0, 1)
        elif force_Polarization == 'ePhi':
            efield3_f[2:, mask] = np.moveaxis(stacked_lstsq(np.moveaxis(efield_antenna_factor[:, 1, mask], 1, 0)[:, :, np.newaxis], np.moveaxis(traces_freq_domain[:, mask], 1, 0)), 0, 1)
        else:
            efield3_f[1:, mask] = np.moveaxis(stacked_lstsq(np.moveaxis(efield_antenna_factor[:, :, mask], 2, 0), np.moveaxis(traces_freq_domain[:, mask], 1, 0)), 0, 1)

        electric_field = NuRadioReco.framework.electric_field.ElectricField(use_channels, efield_position)
        electric_field.set_frequency_spectrum(efield3_f, station.get_channel(use_channels[0]).get_sampling_rate())
        electric_field.set_parameter(efp.zenith, zenith)
        electric_field.set_parameter(efp.azimuth, azimuth)
        electric_field.set_trace_start_time(times[0])
        station.add_electric_field(electric_field)

        if self.debug:
            fig, ax = plt.subplots(3, 1, figsize=(10, 6))
            ax[0].set_title(f'Station {station.get_id()} - Event {evt.get_id()} - Run {evt.get_run_number()} - Reconstructed Electric Field')
            for i in range(3):
                ax[i].plot(times, fft.freq2time(efield3_f[i,:], station.get_channel(use_channels[0]).get_sampling_rate()), label=f'E-field component {i}')
                ax[i].set_ylabel('Electric Field [V/m]')
                ax[i].set_xlim(times[0], times[-1])
                ax[i].legend(loc=1)
            ax[-1].set_xlabel('Time [ns]')
            fig.tight_layout()
            plt.savefig("debug_efield_unfolded.png")

    def end(self):
        pass
