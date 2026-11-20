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
from radiotools import helper as hp

logger = logging.getLogger('NuRadioReco.voltageToEfieldConverter')


def _get_ray_traced_travel_time(propagator, source_position, receiver_position, zenith, azimuth,
                                label="receiver", warn=True, max_angle_deviation=3 * units.deg):
    """ Ray-trace from ``source_position`` to ``receiver_position`` (both absolute) and return the travel time
    of the solution whose arrival direction is closest to (zenith, azimuth). """
    propagator.set_start_and_end_point(source_position, receiver_position)
    propagator.find_solutions()
    n_solutions = propagator.get_number_of_solutions()
    if n_solutions == 0:
        raise ValueError(
            f"No propagation solution found for {label}. Please check the geometry. "
            f"Source: {source_position}, receiver: {receiver_position}.")

    receive_vector_provided = hp.spherical_to_cartesian(zenith, azimuth)
    space_angle_differences = [hp.get_angle(receive_vector_provided, propagator.get_receive_vector(iS))
                               for iS in range(n_solutions)]
    solution_index = int(np.argmin(space_angle_differences))

    if warn and space_angle_differences[solution_index] > max_angle_deviation:
        logger.warning(
            f"{label}: the closest ray-traced arrival direction deviates by "
            f"{space_angle_differences[solution_index] / units.deg:.1f} deg from the provided direction "
            f"(zenith {zenith / units.deg:.1f} deg, azimuth {azimuth / units.deg:.1f} deg). "
            f"Please check the geometry and the provided angles.")

    return propagator.get_travel_time(solution_index)


def get_array_of_channels(station, use_channels, det, zenith, azimuth, antenna_pattern_provider,
                          time_domain=False, efield_position=None, time_delay_mode="plane_wave",
                          propagator=None, travel_time_delays=None, ice_model=None, source_position=None):
    """ Get the time-aligned voltage traces and antenna factors for electric field reconstruction using unfolding.

    Each channel trace is shifted by the signal arrival time delay between ``efield_position`` and the channel,
    and all traces are cut to their common time window. Channels at the same position as ``efield_position``
    are not shifted. How the delays are determined is set by ``time_delay_mode``:

    * 'plane_wave' (default): plane wave arriving from (zenith, azimuth). For cosmic rays arriving from above
      (``station.is_cosmic_ray()`` and zenith <= 90 deg), the in-air propagation speed is used. Otherwise, the
      refractive index at the antenna depth is used: from ``ice_model`` if provided, else from
      ``NuRadioReco.utilities.ice``, which is only allowed for antennas shallower than 10 m.
      Suited for distant sources.
    * 'ray_tracing': ray tracing with ``propagator`` from ``source_position`` to each channel and to
      ``efield_position``. For each ray, the solution whose arrival direction is closest to (zenith, azimuth)
      is used. Suited for sources close to the station, e.g. in shallow ice.
    * 'user_defined': the delays are given directly via ``travel_time_delays``, for geometries not covered
      by the above (e.g. plane waves from air observed with deep antennas).

    All channels need to have the same sampling rate.

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
        Should be the position relative to the station center.
        If None, it raises an error.
    time_domain : bool, optional
        If True, returns the time domain traces as well. Default is False.
    time_delay_mode : {'plane_wave', 'ray_tracing', 'user_defined'}, default: 'plane_wave'
        How the signal arrival time delays between the channels are determined (see above).
    propagator : instance of ``NuRadioMC.SignalProp.propagation_base_class.ray_tracing_base``, optional
        Propagator used to calculate the travel times. Required for ``time_delay_mode='ray_tracing'``.
    travel_time_delays : array of floats, optional
        Signal arrival time at each channel in ``use_channels`` (same order) relative to ``efield_position``;
        positive values mean that the channel sees the signal later. Required for
        ``time_delay_mode='user_defined'``, ignored (with a warning) otherwise.
    ice_model : ``NuRadioMC.utilities.medium`` ice model, optional
        Used for ``time_delay_mode='plane_wave'`` to get the refractive index at the antenna depth.
        Required if any antenna is deeper than 10 m (unless the signal is a cosmic ray from above).
    source_position : numpy array, optional
        Absolute position of the signal source (e.g. neutrino vertex or pulser), in the same coordinate system
        as ``det.get_absolute_position``. Required for ``time_delay_mode='ray_tracing'``.

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

    if time_delay_mode not in ('plane_wave', 'ray_tracing', 'user_defined'):
        raise ValueError(f"Unknown time_delay_mode {time_delay_mode!r}.")

    if time_delay_mode == 'user_defined':
        if travel_time_delays is None or len(travel_time_delays) != len(use_channels):
            raise ValueError(
                "time_delay_mode='user_defined' requires `travel_time_delays` with one entry per channel in `use_channels`.")
    else:
        if travel_time_delays is not None:
            logger.warning(
                "travel_time_delays is provided but will be ignored for time_delay_mode != 'user_defined'."
            )

    if time_delay_mode == "ray_tracing":
        if propagator is None:
            raise ValueError(
                "For time_delay_mode='ray_tracing', a propagator must be provided."
            )
        if source_position is None:
            raise ValueError(
                "For time_delay_mode='ray_tracing', a source_position must be provided."
            )

    sampling_rates = [channel.get_sampling_rate() for channel in station.iter_channels(use_channels)]
    if np.unique(sampling_rates).size != 1:
        raise ValueError(
            "All channels used for unfolding must have the same sampling rate"
        )

    t_mins = []
    t_maxs = []
    t_shifts = []

    station_id = station.get_id()
    site = det.get_site(station_id)

    if time_delay_mode == "ray_tracing":
        # reference travel time from the source to the efield position (absolute positions)
        station_position = det.get_absolute_position(station_id)
        efield_travel_time = _get_ray_traced_travel_time(
            propagator, source_position, station_position + efield_position, zenith, azimuth,
            label="efield position")

    for iCh, channel in enumerate(station.iter_channels(use_channels)):
        channel_id = channel.get_id()

        if time_delay_mode == "user_defined":
            time_shift = -travel_time_delays[iCh]

        elif np.linalg.norm(det.get_relative_position(station_id, channel_id) - efield_position) < 0.01 * units.mm:
            time_shift = 0.0

        elif time_delay_mode == "ray_tracing":
            antenna_position = station_position + det.get_relative_position(station_id, channel_id)
            travel_time = _get_ray_traced_travel_time(
                propagator, source_position, antenna_position, zenith, azimuth,
                label=f"Channel {channel_id}", warn=False)
            time_shift = -(travel_time - efield_travel_time)

        elif time_delay_mode == "plane_wave":
            antenna_position = det.get_relative_position(station_id, channel_id) # here, only the relative position is needed

            # plane wave: in-air speed for cosmic rays from above, otherwise n at the antenna depth
            if station.is_cosmic_ray() and zenith <= 0.5 * np.pi:
                refractive_index = ice.get_refractive_index(1, site)
            else:
                if ice_model is not None:
                    refractive_index = ice_model.get_index_of_refraction(antenna_position)
                elif antenna_position[2] > -10 * units.m:
                    refractive_index = ice.get_refractive_index(antenna_position[2], site)
                else:
                    raise ValueError(
                        "For time_delay_mode='plane_wave', the ice_model must be provided for antennas below 10 m depth."
                    )
            time_shift = -geo_utl.get_time_delay_from_direction(
                zenith, azimuth, antenna_position - efield_position, n=refractive_index)

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

    The electric field is reconstructed at the mean position of the used channels. If the channels are not
    at the same position, their voltage traces are first aligned in time. The signal arrival time delays
    are determined according to ``time_delay_mode`` (set in `begin`):

    * 'plane_wave' (default): plane wave arriving from (zenith, azimuth). Suited for distant sources.
      For antennas deeper than 10 m, an ``ice_model`` is required.
    * 'ray_tracing': ray tracing from a source position (``source_position`` in `run`, or the station
      parameter ``stnp.nu_vertex``) to the channels. Suited for sources close to the station,
      e.g. in shallow ice. Requires an ``ice_model``.
    * 'user_defined': delays provided via ``travel_time_delays`` in `run`, for geometries not covered by
      the above.

    All channels in ``use_channels`` need to have the same sampling rate.
    """

    def __init__(self):
        self.antenna_provider = None
        self.begin()

    def begin(self, hann_window_duration=30 * units.ns, time_delay_mode='plane_wave', ice_model=None, propagation_module='analytic', debug=False):
        """
        Initialize the voltageToEfieldConverter.

        Parameters
        ----------
        hann_window_duration : float, optional
            A Hann window is applied to the edges of the voltage traces to reduce artifacts from the
            FFT and unfolding. The default duration is 30 ns at each edge, but can be set to 0 to disable
            the Hann window.
        time_delay_mode : {'plane_wave', 'ray_tracing', 'user_defined'}, default: 'plane_wave'
            How the signal arrival time delays between the channels are determined:

            * 'plane_wave': plane wave arriving from (zenith, azimuth), using the in-air propagation speed
              for cosmic rays from above and the refractive index at the antenna depth otherwise.
            * 'ray_tracing': ray tracing from the source position to the channels. Requires ``ice_model``,
              and ``source_position`` in `run` (or the station parameter ``stnp.nu_vertex``).
            * 'user_defined': delays given via ``travel_time_delays`` in `run`.

        ice_model : ``NuRadioMC.utilities.medium`` ice model, optional
            Required for ``time_delay_mode='ray_tracing'`` (used to set up the propagator). For 'plane_wave',
            it provides the refractive index at the antenna depth and is required for antennas deeper than
            10 m. If None (default), no propagator is created.
        propagation_module : str, default: 'analytic'
            Propagation module name passed to ``propagation.get_propagation_module``. Only used together with
            ``ice_model``. Options are: 'analytic', 'radiopropa', 'direct_ray'.
        debug : bool, optional
            If True, saves debug plots of the aligned voltage traces, antenna factors and unfolded
            electric field.
        """
        self.antenna_provider = antennapattern.AntennaPatternProvider()
        self.time_delay_mode = time_delay_mode
        self.ice_model = ice_model
        self.hann_window_duration = hann_window_duration
        self.debug = debug

        if ice_model is not None:
            from NuRadioMC.SignalProp import propagation  # lazy: only needed for ray-traced time delays
            self.propagator = propagation.get_propagation_module(propagation_module)(ice_model)
        else:
            self.propagator = None

    @register_run()
    def run(self, evt, station, det, use_channels=None, use_MC_direction=False, force_Polarization='', travel_time_delays=None, source_position=None):
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
            Signal arrival time at each channel in ``use_channels`` (same order) relative to the electric
            field position (the mean position of ``use_channels``); positive values mean that the channel
            sees the signal later. Required for ``time_delay_mode='user_defined'``, ignored (with a warning)
            otherwise.
        source_position: numpy array, optional
            Absolute position of the signal source (e.g. neutrino vertex or pulser), in the same coordinate
            system as ``det.get_absolute_position``. Used for ``time_delay_mode='ray_tracing'``. If None,
            the station parameter ``stnp.nu_vertex`` is used.
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

        if any(channel_id not in station.get_channel_ids() for channel_id in use_channels):
            raise ValueError("Some channels in `use_channels` are not present in the station. Please check the channel ids.")

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

        if self.time_delay_mode == "ray_tracing" and source_position is None:
            if not station.has_parameter(stnp.nu_vertex):
                raise ValueError("For time_delay_mode='ray_tracing', provide `source_position` or set stnp.nu_vertex.")
            source_position = station.get_parameter(stnp.nu_vertex)

        times, efield_antenna_factor, traces_freq_domain = get_array_of_channels(
            station, use_channels, det, zenith, azimuth, self.antenna_provider,
            efield_position=efield_position,
            ice_model=self.ice_model,
            source_position=source_position,
            time_delay_mode=self.time_delay_mode,
            propagator=self.propagator,
            travel_time_delays=travel_time_delays
        )

        if self.hann_window_duration > 0 * units.ns:
            sampling_rate = station.get_channel(use_channels[0]).get_sampling_rate()
            hann_window_samples = int(self.hann_window_duration * sampling_rate)
            traces_time_domain = fft.freq2time(traces_freq_domain, sampling_rate)
            hann_window = signal_processing.half_hann_window(traces_time_domain.shape[1], hann_window_length=hann_window_samples)
            traces_time_domain[:, :] *= hann_window[None, :]
            traces_freq_domain = fft.time2freq(traces_time_domain, sampling_rate)

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
            plt.close()

            fig, ax = plt.subplots(len(use_channels), 1, figsize=(10, 2*len(use_channels)))
            frequencies = fft.freqs(len(times), sampling_rate=station.get_channel(use_channels[0]).get_sampling_rate())
            ax[0].set_title(f'Station {station.get_id()} - Event {evt.get_id()} - Run {evt.get_run_number()} - Shifted Voltage Traces')
            for i_ch in range(len(traces_freq_domain)):
                ax[i_ch].plot(frequencies, abs(efield_antenna_factor[i_ch,0,:]), label=f'Channel {use_channels[i_ch]}, theta')
                ax[i_ch].plot(frequencies, abs(efield_antenna_factor[i_ch,1,:]), label=f'Channel {use_channels[i_ch]}, phi')
                ax[i_ch].set_ylabel('|Antenna factor| [m]')
                ax[i_ch].set_xlim(frequencies[0], frequencies[-1])
            ax[-1].set_xlabel('Frequency [GHz]')
            ax[0].legend()
            fig.tight_layout()
            plt.savefig("debug_antenna_factors.png")
            plt.close()

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
            plt.close()

    def end(self):
        pass
