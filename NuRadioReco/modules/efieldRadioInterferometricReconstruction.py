"""
This module hosts to classes
    - efieldInterferometricDepthReco
    - efieldInterferometricAxisReco
    - efieldInterferometricLateralReco

The radio-interferometric reconstruction (RIT) was proposed in [1].
The implementation here is based on work published in [2].
It is a rewrite of the original module for [3].

- [1]: H. Schoorlemmer, W. R. Carvalho Jr., arXiv:2006.10348
- [2]: F. Schlueter, T. Huege, doi:10.1088/1748-0221/16/07/P07048
- [3]: T. Wybouw, master thesis at iihe (VUB)

"""

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units, interferometry
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stp

from radiotools import helper as hp, coordinatesystems
from radiotools.atmosphere import models, refractivity

from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.base_shower import BaseShower
from NuRadioReco.detector.detector_base import DetectorBase

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, colorbar

import scipy
import scipy.stats
from scipy.optimize import curve_fit

from typing import Optional

from tqdm import tqdm, trange
from os import cpu_count

import logging
logger = logging.getLogger(
    'NuRadioReco.efieldRITReconstruction'
)


class efieldInterferometricDepthReco:
    """
    This class reconstructs the depth of the maximum of the longitudinal profile of the
    beam-formed radio emission: X_RIT along a given axis. X_RIT is found to correlate with X_max.
    This correlation may depend on the zenith angle, the frequency band, and detector layout
    (if the detector is very irregular).

    For the reconstruction, a shower axis is needed. The radio emission in the vxB polarisation is used
    (and thus the arrival direction is needed).

    In [3], consistent biases were observed for the beamformed fluence maxima, away from the MC axis.
    An XRIT reconstruction will be influenced by how close the axis is to these biased radio maxima.
    In particular, XRIT is likely to be found where the axis is closest the the curved longitundal fluence profile,
    instead of where the longitudinal radio profile actually peaks. This effect is stronger for vertical showers,
    where the charge-excess fraction is higher (the hypothesised bias culprit). With this in mind, use the module with caution.

    """

    def __init__(self):
        self._debug = None
        self._at = None
        self._tab = None
        self._use_channels = None
        self._signal_kind = None
        self._signal_threshold = None
        self._traces = None
        self._times = None
        self._positions = None
        self._axis = None
        self._core = None
        self._depths = None
        self._binsize = None
        self._tstep = None
        self._mc_jitter = None
        self._cs = None
        self._shower = None
        self._zenith = None
        self._window = None
        self._use_sim_pulses = None
        self._data = {}

    def set_geometry(self,
                     shower: BaseShower,
                     core: Optional[np.ndarray] = None,
                     axis: Optional[np.ndarray] = None,
                     smear_angle_radians: float = 0,
                     smear_core_meter: float = 0):
        """
        Set the initial geometry (core, zenith, azimuth) used in the RIT reconstruction.

        When provided, the values in ``core`` and ``axis`` are given priority to those coming from within the
        ``shower``. The latter is always stored as attribute to use ``showerParameters.observation_level``,
        ``showerParameters.magnetic_field_vector``, ``showerParameters.atmospheric_model`` later on.
        If ``core`` and ``axis`` are not passed, they will be extracted from ``shower``.
        When MC (i.e. when setting ``use_sim_pulses = True`` in `begin`) is used, the extracted core and axis
        are given a random error which is controlled by the parameters ``smear_core_meter`` and ``smear_angle_radians``.

        Note that this function is called from within the `begin` function, so there is no need to call it manually.

        Parameters
        ----------
        shower:  BaseShower
            Shower object for the RIT reconstruction.
        core: np.ndarray, optional
            Overwrite the core found in ``shower`` with this value. This is incompatible with ``smear_core_meter != 0``.
        axis: np.ndarray, optional
            Overwrite the axis found in ``shower`` with this value. This is incompatible with ``smear_angle_radians != 0``.
        smear_angle_radians: float, default: 0.
            The error used to smear the axis found in an MC shower.
            This is used as the mode of the opening angle distribution of the Vonmises-Fisher distribution
            (think Gaussian on a unit sphere).
        smear_core_meter: float, default: 0.
            Errors used to smear the core found in an MC shower.
            It is used as the standard deviation of a Gaussian centred on zero, holds for both east and north error.
        """

        assert not (core is not None and smear_core_meter != 0)
        assert not (axis is not None and smear_angle_radians != 0)

        # set mc core if no core is given
        if core is None:
            core = np.array(shower[shp.core])

        # set mc axis if no axis is given
        if axis is None:
            axis = shower.get_axis()

        # smearing
        if smear_core_meter:
            cs = coordinatesystems.cstrafo(
                *hp.cartesian_to_spherical(*axis), shower[shp.magnetic_field_vector])
            core_vxB = np.random.normal((0, 0), smear_core_meter)
            core = cs.transform_from_vxB_vxvxB_2D(
                core_vxB, core=core).flatten()

        if smear_angle_radians:
            concentration_parameter = 1 / smear_angle_radians**2
            axis = scipy.stats.vonmises_fisher.rvs(axis, int(concentration_parameter)).flatten()

        self._core = core
        self._axis = axis
        self._zenith = hp.get_angle(np.array([0, 0, 1]), self._axis)
        self._shower = shower
        self._cs = coordinatesystems.cstrafo(
            *hp.cartesian_to_spherical(*self._axis), shower[shp.magnetic_field_vector])

    def begin(self,
              shower: BaseShower,
              core: Optional[np.ndarray] = None,
              axis: Optional[np.ndarray] = None,
              window: float = 150*units.ns,
              use_sim_pulses: bool = False,
              use_channels: bool = False,
              debug: bool = False):
        """
        Set module configuration.

        Parameters
        ----------
        shower: BaseShower
            shower object to extract geometry (zenith, azimuth, core) and observation information
            (magnetic_field_vector, observation_level, atmospheric_model).
            Conventional choice: Event.get_first_shower(), Event.get_first_sim_shower().
        core, axis: np.ndarray, optional
            core and axis which overwrite those found in ``shower``
        window: float, default: 150 * units.ns
            Width of the trace section (centered on the trace maximum) which will be used.
        use_sim_pulses: bool, default: False
            If True, the processed event will be handled as simulated, using SimStation instead of Station,
            and assuming the geometry of `shower` is the truth.
        use_channels: bool, default: False
            If True, use the dominant polarization of channel traces, which are usually undefined for
            simulated events. This will give very unexpected results, not recommended!
        debug: bool, default: False
            If True, show some debug plots while running.
        """
        self._debug = debug
        self._window = window
        self._use_channels = use_channels
        self._use_sim_pulses = use_sim_pulses

        self.set_geometry(shower, core, axis)
        self.update_atmospheric_model_and_refractivity_table()

    def sample_longitudinal_profile(self, depths: np.ndarray):
        """
        Calculate the longitudinal profile of the interferometric signal sampled along the shower axis.

        Parameters
        ----------
        depths: np.ndarray
            Atmospheric depths at which to sample with RIT.

        Returns
        -------
        signals : np.ndarray
            Interferometric signals sampled along the given axis
        """

        signals = np.zeros(len(depths))
        for idx, depth in enumerate(depths):
            try:
                dist = self._at.get_distance_xmax_geometric(
                    self._zenith, depth, observation_level=self._shower[shp.observation_level])
            except ValueError:
                logger.info(
                    "ValueError in get_distance_xmax_geometric, setting signal to 0")
                signals[idx] = 0
                continue

            if dist < 0:
                signals[idx] = 0
                continue

            point_on_axis = self._axis * dist + self._core
            sum_trace = interferometry.interfere_traces_rit(
                point_on_axis, self._positions, self._traces, self._times, tab=self._tab)

            signal = interferometry.get_signal(
                sum_trace, self._tstep, kind=self._signal_kind)
            signals[idx] = signal

        return signals

    def reconstruct_interferometric_depth(self, return_profile=False):
        """
        Returns Gauss-parameters fitted to the "peak" of the interferometric
        longitudinal profile along the shower axis.

        If the beamformed fluence maximum is found at an edge the sampling range is extended,
        (with a min/max depth of 0/2000 g/cm^2).
        The Gaussian is fitted around the found peak with a refined sampling (use 20 samples in this narrow range).

        Parameters
        ----------
        return_profile : bool, default: False
            If ``True``, return the sampled profile in addition to the Gauss parameter.

        Returns
        -------
        If return_profile is True
            depths_fine : np.array
                Depths along shower axis finely sampled (used in fitting)

            signals : np.array
                Beamformed fluence along shower axis coarsely sampled.

            signals_fine : np.array
                Beamformed fluence along shower axis finely sampled (used in fitting)

            popt : list
                List of fitted Gauss parameters (amplitude, position, width)

        If return_profile is False:
            XRIT: float
                Atmospheric depth of beamformed fluence max.

        """
        signals = self.sample_longitudinal_profile(self._depths)

        # if max signal is at the upper edge add points there
        if np.argmax(signals) == len(self._depths) - 1:
            while True:
                depth_add = np.amax(self._depths) + self._binsize
                signal_add = self.sample_longitudinal_profile([depth_add])
                self._depths = np.hstack((self._depths, depth_add))
                signals = np.append(signals, signal_add)

                if not np.argmax(signals) == len(
                        self._depths) - 1 or depth_add > 2000:
                    break

        # if max signal is at the lower edge add points there
        elif np.argmax(signals) == 0:
            while True:
                depth_add = np.amin(self._depths) - self._binsize
                signal_add = self.sample_longitudinal_profile([depth_add])
                self._depths = np.hstack((depth_add, self. _depths))
                signals = np.append(signal_add, signals)

                if not np.argmax(signals) == 0 or depth_add <= 0:
                    break

        idx_max = np.argmax(signals)
        dtmp_max = self._depths[idx_max]

        depths_fine = np.linspace(
            dtmp_max - 30,
            dtmp_max + 30,
            20)  # 3 g/cm2 bins
        signals_fine = self.sample_longitudinal_profile(depths_fine)

        def normal(x, A, x0, sigma):
            """ Gauss curve """
            return A / np.sqrt(2 * np.pi * sigma ** 2) \
                * np.exp(-1 / 2 * ((x - x0) / sigma) ** 2)

        popt, _ = curve_fit(
            normal, depths_fine, signals_fine,
            p0=[np.amax(signals_fine), depths_fine[np.argmax(signals_fine)], 100],
            maxfev=1000
        )
        xrit = popt[1]

        if return_profile:
            return depths_fine, signals, signals_fine, popt

        return xrit

    @register_run()
    def run(self,
            evt: Event,
            det: Optional[DetectorBase] = None,
            station_ids: Optional[list] = None,
            signal_kind="power",
            relative_signal_treshold: float = 0.,
            depths: np.ndarray = np.arange(400, 800, 10) * units.g / units.cm2,
            mc_jitter: float = 0 * units.ns,
            ):
        """
        Run interferometric reconstruction to find the interferometric depth of shower maximum.

        Parameters
        ----------
        evt : Event
            Event to run the module on, containing (Sim)Station objects with the data.
        det : Detector, optional
            Positions can be obtained from channel positions in a Detector or those stored in
            the ``ElectricField`` objects' attributes.
        station_ids: list, default: None
            station_ids which will be read out. `None` implies using all available stations (`Event.get_station_ids()`)
        signal_kind: {'power', 'hilbert_sum', 'amplitude'}
            Quantity to time-integrate. Using 'power' (the default) squares the summed trace,
            which is recommended (square of sum if larger than sum of squares)
        relative_signal_treshold: float, default: 0.
            Fraction of the strongest measured fluence that should be contained
            in a trace for it to be used in the reconstruction.
        depths: np.ndarray, default: np.arange(400, 800, 10) * units.g / units.cm2
            Atmospheric depths along which to perform a first coarse sampling.
        mc_jitter: float, default: None
            Standard deviation of Gaussian noise added to timings, in nanoseconds (internal NRR units).
            None implies no jitter.
        """
        self._signal_kind = signal_kind
        self._signal_threshold = relative_signal_treshold

        self._depths = depths / units.g * units.cm2
        self._binsize = self._depths[1] - self._depths[0]
        self._mc_jitter = mc_jitter / units.ns

        self.set_station_data(evt, det, station_ids=station_ids)

        if not self._debug:
            xrit = self.reconstruct_interferometric_depth()
        else:
            depths_final, signals_tmp, signals_final, rit_parameters = \
                self.reconstruct_interferometric_depth(return_profile=True)
            xrit = rit_parameters[1]
            ax = plt.figure().add_subplot()
            ax.scatter(self._depths, signals_tmp, color="blue",
                       label="Coarse sampling", s=2, zorder=1.1)
            ax.scatter(depths_final, signals_final, color="red",
                       label="Fine sampling", s=2, zorder=1)
            ax.plot(depths_final, normal(depths_final, *rit_parameters),
                    label="Gaussian fit", color="black", ls="--")
            ax.axvline(rit_parameters[1])
            ax.set_xlabel(r"$X [\mathrm{g}/\mathrm{cm}^2]$")
            ax.set_ylabel(self._signal_kind)
            ax.legend()
            plt.show()

        self._shower.set_parameter(
            shp.interferometric_shower_maximum,
            xrit * units.g / units.cm2)

    def end(self):
        """
        End the module

        Returns
        -------
        self._data: dict
            Dictionary containing some additional information which can be useful for studies.
        """
        return self._data

    def update_atmospheric_model_and_refractivity_table(self):
        """
        Updates model of the atmosphere and tabulated, integrated refractive index according to shower properties.
        Default atmospheric model is 17, if it is not found in `self._shower`.
        """

        logger.warning(
            "flat earth geometry assumed. default was curved. If issue has been fixed, consider moving back to curved")
        curved = False

        try:
            atmospheric_model = self._shower[shp.atmospheric_model]
        except KeyError:
            logger.warning(
                "Shower doesn't contain showerParameters.atmospheric_model (likely not a sim shower). Setting model 17 (US standard, Keilhauer)")
            atmospheric_model = 17
        if self._at is None:
            self._at = models.Atmosphere(
                atmospheric_model, curved=curved)
            self._tab = refractivity.RefractivityTable(
                self._at.model, refractivity_at_sea_level=self._shower[shp.refractive_index_at_ground] - 1, curved=curved)

        elif self._at.model != atmospheric_model:
            self._at = models.Atmosphere(
                atmospheric_model, curved=curved)
            self._tab = refractivity.RefractivityTable(
                self._at.model, refractivity_at_sea_level=self._shower[shp.refractive_index_at_ground] - 1, curved=curved)

        elif self._tab._refractivity_at_sea_level != self._shower[shp.refractive_index_at_ground] - 1:
            self._tab = refractivity.RefractivityTable(
                self._at.model, refractivity_at_sea_level=self._shower[shp.refractive_index_at_ground] - 1, curved=curved)

    def set_station_data(self, evt: Event, det: Optional[DetectorBase], station_ids: Optional[list] = None):
        """
        Set station data (positions, traces and timing of traces) available to the module.
        Can handle both electric fields and channels (voltages); for the latter a detector is required.

        Parameters
        ----------
        evt : Event
            Event containing the data.
        det: Detector, optional
        station_ids: list, default: None
            List of requested station_ids. If None, all stations are used.
        """
        traces = []
        times = []
        pos = []
        if station_ids is None:
            station_ids = evt.get_station_ids()

        if self._use_channels:
            dominant_polarisations = [evt.get_station(
                sid)[stp.cr_dominant_polarisation] for sid in station_ids if self._use_channels]
            unique, counts = np.unique(
                dominant_polarisations, axis=0, return_counts=True)
            strongest_pol_overall = unique[np.argmax(counts)]

            positions_and_times_and_traces = []
            for sid in station_ids:
                station: Station = evt.get_station(sid)
                station_position = det.get_absolute_position(sid)
                positions_and_times_and_traces += [((station_position
                                                     + det.get_relative_position(sid, cid)),
                                                    station.get_channel(
                                                        cid).get_times(),
                                                    station.get_channel(
                                                        cid).get_trace(),
                                                    sid,
                                                    )
                                                   for cid in station.get_channel_ids()
                                                   if np.all(np.abs(det.get_antenna_orientation(sid, cid) - strongest_pol_overall) < 1e-6)]
        else:
            positions_and_times_and_traces = []
            for sid in station_ids:
                station: Station = evt.get_station(sid)

                if self._use_sim_pulses:
                    station = station.get_sim_station()

                # if det is not None:
                #     station_position = det.get_absolute_position(sid)

                #     positions_and_times_and_traces += [(station_position + np.mean([det.get_relative_position(sid, cid) for cid in electric_field.get_channel_ids()], axis=0),
                #                                         electric_field.get_times(),
                #                                         self._cs.transform_to_vxB_vxvxB(electric_field.get_trace())[0])
                #                                        for electric_field in station.get_electric_fields()]
                positions_and_times_and_traces += [(electric_field.get_position(),
                                                    electric_field.get_times(),
                                                    self._cs.transform_to_vxB_vxvxB(
                                                        electric_field.get_trace())[0],
                                                    sid,
                                                    )
                                                   for electric_field in station.get_electric_fields()]

        warned_early = False
        warned_late = False
        sids_rit = []
        for position, time, trace, sid in positions_and_times_and_traces:
            if self._use_sim_pulses and self._mc_jitter > 0:
                time += np.random.normal(scale=self._mc_jitter)

            # select window around argmax of trace
            nsampling = int(self._window / self._tstep)
            hw = nsampling // 2
            m = np.argmax(np.abs(trace))

            if m < hw:
                if not warned_early:
                    logger.warning(
                        "Trace max close to early edge. This warning is printed only once.")
                    warned_early = True
                m = hw
            if m > len(trace) - hw:
                if not warned_late:
                    logger.warning(
                        "Trace max close to late edge. This warning is printed only once.")
                    warned_late = True
                m = len(trace) - hw

            trace = trace[m - hw:m + hw]
            time = time[m - hw:m + hw]

            traces.append(trace)
            times.append(time)
            pos.append(position)
            sids_rit.append(sid)

        traces = np.array(traces)
        times = np.array(times)
        pos = np.array(pos)
        sids_rit = np.array(sids_rit)

        # efield positions are set to [0, 0, 0] in voltageToEfieldConverter. This should protect against such behaviour.
        assert not np.all(pos[:, :2] == 0)

        # Mask traces with too low fluence
        power = traces**2
        flu = np.sum(power, axis=-1)
        mask = (flu >= self._signal_threshold * np.max(flu))
        logger.info(
            f"{np.sum(mask) / len(mask) * 100: .3f}% of trace_vector used for RIT with relative fluence above {self._signal_threshold}"
        )

        debug_sids = [None]
        if self._debug:
            fig = plt.figure()
            gs = gridspec.GridSpec(len(debug_sids), 2, figure=fig, width_ratios=[
                                   .05, 1], height_ratios=np.ones_like(debug_sids))
            ax_footprint = fig.add_subplot(gs[:, 1])
            cmap = plt.cm.viridis
            sm = ax_footprint.scatter(
                *(pos[mask].T[:2, :]), c=flu[mask], cmap=cmap, s=1)
            cax = fig.add_subplot(gs[:, 0])
            cb = colorbar.Colorbar(cax, sm, label="Fluence")
            cb.ax.set_yticklabels([])
            ax_footprint.scatter(
                *(pos[~mask].T[:2, :]), c="red", s=1, marker="x", label="Excluded")
            ax_footprint.set_xlabel(r"$x_\mathrm{ground}~[\mathrm{m}]$")
            ax_footprint.set_ylabel(r"$y_\mathrm{ground}~[\mathrm{m}]$")
            ax_footprint.spines[["top", "right"]].set_visible(True)
            ax_footprint.tick_params(top=True, right=True)
            ax_footprint.set_aspect("equal")
            ax_footprint.legend()
            plt.show()

        traces = traces[mask]
        times = times[mask]
        pos = pos[mask]
        flu = flu[mask]
        power = power[mask]
        sids_rit = np.unique(sids_rit[mask])
        # sorted integesrs seperated by spaces, with the brackets [] removed
        self._data["station_set"] = str(sids_rit)[1:-1]

        self._traces = traces
        self._times = times
        self._positions = pos

        # Get the baselines, with MC geometry if available
        if self._use_sim_pulses:
            cs_shower = coordinatesystems.cstrafo(
                self._shower[shp.zenith], self._shower[shp.azimuth], magnetic_field_vector=self._shower[shp.magnetic_field_vector])
            logger.debug(f"self._positions shape: {self._positions.shape}")
            mc_core = self._shower[shp.core]
            if len(mc_core) == 2:
                mc_core = np.hstack([mc_core, 0])
            mc_core[2] = 0
            pos_showerplane = cs_shower.transform_to_vxB_vxvxB(
                self._positions, core=mc_core)
        else:
            pos_showerplane = self._cs.transform_to_vxB_vxvxB(
                self._positions, core=self._core)

        def get_baselines(positions):
            n = len(positions)
            baselines = np.empty((n*(n-1)//2, 3))
            counter = 0
            for i in range(n-1):
                for j in range(i+1, n):
                    baselines[counter] = (positions[i] - positions[j])
                    counter += 1
            assert counter == n * (n-1) / 2
            return np.abs(baselines)

        showerplane_baselines = get_baselines(pos_showerplane)
        ground_baselines = get_baselines(pos)

        self._data["max_vxB_baseline"] = max(
            showerplane_baselines[:, 0]) * units.m
        self._data["max_vxvxB_baseline"] = max(
            showerplane_baselines[:, 1]) * units.m

        self._data["max_eastwest_baseline"] = max(
            ground_baselines[:, 0]) * units.m
        self._data["max_northsouth_baseline"] = max(
            ground_baselines[:, 1]) * units.m

        self._data["n_observers"] = len(self._positions)
        self._data["max_fluence_fraction"] = np.max(flu) / np.sum(flu)


class efieldInterferometricAxisReco(efieldInterferometricDepthReco):
    """
    Class to reconstruct the shower axis with the radio-interferometric technique.
    The reconstructed axis is a straight line fitted through the beamformed fluence map.

    This is plagued by biases stronger for vertical showers, as observed in [3].
    This implies also a biased shower axis (on the order of 0.1-1 degree, depending on the layout irregularity)
    """

    def __init__(self):
        super().__init__()
        self._multiprocessing = None
        self._core0 = None
        self._axis0 = None
        self._core_spread = None
        self._axis_spread = None
        self._iterations = None

    def begin(self,
              shower: BaseShower,
              core: Optional[np.ndarray] = None,
              axis: Optional[np.ndarray] = None,
              window: float = 150*units.ns,
              use_sim_pulses: bool = False,
              use_channels: bool = False,
              core_spread: float = 10*units.m,
              axis_spread: float = 1*units.deg,
              multiprocessing: bool = False,
              sample_angular_resolution: float = 0.005*units.deg,
              initial_grid_spacing: float = 60*units.m,
              cross_section_width: float = 1000*units.m,
              refine_axis: bool = False,
              iterations: int = 1,
              debug: bool = False
              ):
        """
        Set module config.

        Parameters
        ----------
        shower: BaseShower
            Shower object to extract geometry (zenith, azimuth, core) and observation information
            (magnetic_field_vector, observation_level, atmospheric_model).
            Conventional choice: Event.get_first_shower(), Event.get_first_sim_shower().
        core: np.ndarray, optional
            Overwrite the core found in ``shower`` with this value. This is incompatible with ``core_spread != 0``.
        axis: np.ndarray, optional
            Overwrite the axis found in ``shower`` with this value. This is incompatible with ``axis_spread != 0``.
        window: float, default: 150 * units.ns
            Width of the trace section (centered on the trace maximum) which will be used.
        use_sim_pulses: bool, default: False
            If True, the processed event will be handled as simulated, using SimStation instead of Station,
            and assuming the geometry of ``shower`` is the truth.
        use_channels: bool (default: False)
            If True, use the dominant polarization of channel traces, which are usually undefined for simulated
            events. This will give very unexpected results, not recommended!
        core_spread: float, default: 10 * units.m
            Value used to smear the core found in an MC shower.
            It is taken as the standard deviation of a Gaussian centred on zero, holds for both east and north error.
        axis_spread: float, default: 1 * units.deg
            Error used to smear the axis found in an MC shower.
            The angular error is the mode of the opening angle distribution of the Vonmises-Fisher distribution
            (think Gaussian on a unit sphere).
        multiprocessing: bool, default: False
            Flag to enable multiprocessing. Not to be used for jobs, but useful in debugging.
        sample_angular_resolution: float, default: 0.005 * units.deg
            Final angular resolution of RIT method.
        initial_grid_spacing: float, default: 60 * units.m
        cross_section_width: float, default: 1000 * units.m
        refine_axis: bool, default: False
            Sample at more depths lying between those passed to `run` to better map the RIT axis
            (strongest beamforming correlation is not a straight line)
        iterations: int, default: 1
           Iterations to find lateral maxima, through which the axis is fit. For >1, median weights are
           associated to these median maxima, which is not clean: fix!
        debug: bool, default: False
        """
        self._debug = debug
        self._refine_axis = refine_axis
        self._window = window
        self._use_sim_pulses = use_sim_pulses
        self._use_channels = use_channels
        self._multiprocessing = multiprocessing
        self._angres = sample_angular_resolution / units.radian
        self._initial_grid_spacing = initial_grid_spacing / units.m
        self._cross_section_width = cross_section_width / units.m

        self._axis_spread = axis_spread / units.radian
        self.set_geometry(shower, core=core, axis=axis,
                          smear_angle_radians=0, smear_core_meter=0)
        self._core0 = np.copy(self._core)
        self._axis0 = np.copy(self._axis)
        self._axis_spread = axis_spread / units.radian
        self._core_spread = core_spread / units.m
        self._iterations = iterations

        self.update_atmospheric_model_and_refractivity_table()

    def find_maximum_in_plane(self,
                              xs_showerplane: np.ndarray,
                              ys_showerplane: np.ndarray,
                              p_axis: np.ndarray,
                              cs: coordinatesystems.cstrafo):
        """
        Sample interferometric signals in 2-d plane (vxB-vxvxB) perpendicular to a given axis
        on a rectangular/quadratic grid.

        Note that the orientation of the plane is defined by the `radiotools.coordinatesystems.cstrafo` object ``cs``.

        Parameters
        ----------
        xs_showerplane: np.ndarray
            x-coordinates defining the sampling positions.
        ys_showerplane: np.ndarray
            y-coordinates defining the sampling positions.
        p_axis : np.ndarray
            Origin of the plane along the axis.
        cs : radiotools.coordinatesytems.cstrafo
            Current coordinate system

        Returns
        -------
        idx : int
            Index of the entry with the largest signal (np.argmax(signals))
        signals : array(len(xs), len(ys))
            Interferometric signal
        """
        def yiteration(xdx, x):
            signals = np.zeros(len(ys_showerplane))
            for ydx, y in enumerate(ys_showerplane):
                p = p_axis + cs.transform_from_vxB_vxvxB(np.array([x, y, 0]))

                sum_trace = interferometry.interfere_traces_rit(
                    p, self._positions, self._traces, self._times, tab=self._tab)

                signal = interferometry.get_signal(
                    sum_trace, self._tstep, kind=self._signal_kind)
                signals[ydx] = signal
            return signals

        if self._multiprocessing:
            try:
                from joblib import Parallel, delayed
                signals = Parallel(n_jobs=max(min(cpu_count() // 2, len(xs_showerplane)), 2))(
                    delayed(yiteration)(xdx, x) for xdx, x in enumerate(xs_showerplane))
            except ImportError:
                logger.warning(
                    "Could not import joblib, single process instead")
                self._multiprocessing = False

        if not self._multiprocessing:
            signals = []
            for xdx, x in enumerate(xs_showerplane):
                signals.append(yiteration(xdx, x))

        signals = np.vstack(signals)
        idx = np.argmax(signals)
        return idx, signals

    def get_paxis_dr_target(self,
                            depth: float,
                            core: np.ndarray,
                            axis: np.ndarray):
        """
        Determine the origin of the plane perpendicular to the shower axis at the desired atmospheric depth,
        and the angular resolution expressed in distance at that depth.

        Parameters
        ----------
        depth: float
        core, axis: np.ndarray

        Returns
        -------
        p_axis: np.ndarray
        dr_ref_target: float
        """
        zenith, azimuth = hp.cartesian_to_spherical(*axis)

        dist = self._at.get_distance_xmax_geometric(
            zenith, depth, observation_level=self._shower[shp.observation_level])
        dr_ref_target = np.tan(self._angres) * dist
        p_axis = axis * dist + core
        return p_axis, dr_ref_target

    def sample_lateral_cross_section(self,
                                     depth: float,
                                     core: np.ndarray,
                                     axis: np.ndarray,
                                     cross_section_width: float,
                                     initial_grid_spacing: float):
        """
        Sample the lateral cross-section plane at a given atmospheric depth.

        Parameters
        ----------
        depth: float
        core, axis: np.ndarray
            Geometry for the sampling.
        cross_section_width: float
            Width of the plane in internal units.
        initial_grid_spacing: float
            Spacing to start sampling at, which will shrink together with the sampling region.

        Returns
        -------
        point_found: np.ndarray
        weight: float
            Weight (signal strength) as defined by `signal_kind`
        ground_grid_uncertainty: float
            Spatial uncertainty on the lateral max (Uniform distribution in a cell uncertainty)
        """
        zenith, azimuth = hp.cartesian_to_spherical(*axis)
        cs = coordinatesystems.cstrafo(
            zenith, azimuth, magnetic_field_vector=self._shower[shp.magnetic_field_vector])

        p_axis, dr_ref_target = self.get_paxis_dr_target(depth, core, axis)
        max_dist = cross_section_width / 2 + initial_grid_spacing

        if self._use_sim_pulses:
            # we use the true core to make sure that it is within the inital search gri
            shower_axis = hp.spherical_to_cartesian(
                self._shower[shp.zenith], self._shower[shp.azimuth])
            mc_core = self._shower[shp.core]
            if len(mc_core) == 2:
                mc_core = np.hstack([mc_core, 0])
            mc_core[2] = 0
            mc_at_plane = interferometry.get_intersection_between_line_and_plane(
                axis, p_axis, shower_axis, mc_core)
            # gives interserction between a plane normal to the shower axis initial guess (shower_axis_inital)
            # anchored at a point in this vB plane at the requested height/depth along the initial axis (p_axis),
            # with the true/montecarlo shower axis anchored at the true/mc core
            cs_shower = coordinatesystems.cstrafo(
                self._shower[shp.zenith], self._shower[shp.azimuth], self._shower[shp.magnetic_field_vector])
            # could instead use p_axis if no mc available?
            mc_vB = cs_shower.transform_to_vxB_vxvxB(mc_at_plane, core=p_axis)

            max_mc_vB_coordinate = np.max(np.abs(mc_vB))
            if max_dist < max_mc_vB_coordinate:
                logger.warning(f"MC axis does not intersect plane to be sampled around p_axis at {depth} g/cm2! " +
                               "Extending the plane to include MC axis. " +
                               f"Consider increasing cross section size by at least a factor {max_mc_vB_coordinate / max_dist: .2f}, since this warning will not appear for real data;)")
                max_dist = np.max(np.abs(mc_vB)) + initial_grid_spacing

        xlims = np.array([-max_dist, max_dist])
        ylims = np.array([-max_dist, max_dist])

        if self._use_sim_pulses:
            xlims += np.random.uniform(-0.1 * initial_grid_spacing,
                                       0.1 * initial_grid_spacing, 2)
            ylims += np.random.uniform(-0.1 * initial_grid_spacing,
                                       0.1 * initial_grid_spacing, 2)

        xs = np.arange(xlims[0], xlims[1] +
                       initial_grid_spacing, initial_grid_spacing)
        ys = np.arange(ylims[0], ylims[1] +
                       initial_grid_spacing, initial_grid_spacing)

        iloop = 0
        while True:
            idx, signals = self.find_maximum_in_plane(xs, ys, p_axis, cs)
            if self._debug:
                if self._use_sim_pulses:
                    axis0_intersect_vB = mc_vB
                else:
                    axis0_intersect_vB = np.zeros(2)

                plot_lateral_cross_section(
                    xs, ys, signals, axis0_intersect_vB, title=r"%.1f$\,$g$\,$cm$^{-2}$" % depth, is_mc=self._use_sim_pulses)
            iloop += 1

            # maximum
            x_max = xs[int(idx // len(ys))]
            y_max = ys[int(idx % len(ys))]

            # update range / grid
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]

            dr = np.sqrt(dx ** 2 + dy ** 2)
            if iloop == 10 or dr < dr_ref_target:
                break

            if iloop >= 2:
                dx /= 2
                dy /= 2
            xs = np.linspace(x_max - dx, x_max + dx, 5)
            ys = np.linspace(y_max - dy, y_max + dy, 5)

        weight = np.amax(signals)

        xfound = xs[int(idx // len(ys))]
        yfound = ys[int(idx % len(ys))]

        point_found = p_axis + \
            cs.transform_from_vxB_vxvxB(np.array([xfound, yfound, 0]))

        ground_grid_uncertainty = cs.transform_from_vxB_vxvxB_2D(
            np.array([dx, dy])/np.sqrt(12))
        return point_found, weight, ground_grid_uncertainty

    def reconstruct_shower_axis(self):
        """
        Run interferometric reconstruction of the shower axis.

        Find the maxima of the interferometric signals within 2-d plane (slices) along a given axis (initial guess).
        Through those maxima (their position in the atmosphere) a straight line is fitted to reconstruct the shower axis.

        Returns
        -------
        direction_rec, core_rec: np.ndarray
        """
        maxima = []
        w_maxima = []
        for i in trange(self._iterations):
            p = []
            w = []
            # only take random core and axis if efields are simulated (in which case MC geometry is likely set)
            # or if more than one iteration has passed, whicch for data implies then passed geometry (e.g. particle arra) is actually used
            if self._use_sim_pulses or i > 1:
                self.set_geometry(
                    self._shower, self._core0, self._axis0, self._axis_spread, self._core_spread)
            for depth in tqdm(self._depths):
                found_point, weight, ground_grid_uncertainty = self.sample_lateral_cross_section(
                    depth, self._core, self._axis, self._cross_section_width, self._initial_grid_spacing)

                p.append(found_point)
                w.append(weight)

            # extend to new depths if max is found at edges of self._depths
            counter = 0
            while True:
                if np.argmax(w) != 0 or counter >= 10:
                    break

                new_depth = self._depths[0] - self._binsize
                logger.info("extend to", new_depth)
                found_point, weight, ground_grid_uncertainty = self.sample_lateral_cross_section(
                    new_depth, self._core, self._axis, self._cross_section_width, self._initial_grid_spacing)

                self._depths = np.hstack(([new_depth], self._depths))
                found_points = [found_point] + p
                w = [weight] + w
                counter += 1

            counter = 0
            while True:
                if np.argmax(w) != len(w) or counter >= 10:
                    break

                new_depth = self._depths[-1] + self._binsize
                logger.info("extend to", new_depth)
                found_point, weight, ground_grid_uncertainty = self.sample_lateral_cross_section(
                    new_depth, self._core, self._axis, self._cross_section_width, self._initial_grid_spacing)

                self._depths = np.hstack((self._depths, [new_depth]))
                p.append(found_point)
                w.append(weight)
                counter += 1

            direction_rec, core_rec, opening_angle_sph, opening_angle_sph_std, core_std = self.fit_axis(
                p, w, self._axis, self._core, full_output=True
            )
            logger.info(f"core: {list(np.round(core_rec, 3))} +- {list(np.round(core_std, 3))} m")

            if self._use_sim_pulses:
                logger.info(
                    f"Opening angle with MC: {np.round(opening_angle_sph / units.deg, 3)} +- {np.round(opening_angle_sph_std / units.deg, 3)} deg"
                )

            # add smaller planes sampled along inital rit axis to increase amount of points to fit final rit axis
            depths2 = np.array([])
            if self._refine_axis:
                old_debug = self._debug
                self._debug = False
                refinement = 2
                depths2 = np.linspace(
                    self._depths[0], self._depths[-1], refinement*len(self._depths))
                for depth in tqdm([d for d in depths2 if d not in self._depths]):
                    found_point, weight, ground_grid_uncertainty = self.sample_lateral_cross_section(
                        depth, core_rec, direction_rec, self._cross_section_width / 4, self._cross_section_width / 20)
                    p.append(found_point)
                    w.append(weight)

                self._debug = old_debug

            maxima.append(np.array(p))
            w_maxima.append(np.array(w))

        # taking the median of the found points is ok, but associating the median weights to them is not!
        found_points = np.median(np.array(maxima), axis=0)
        weights = np.median(np.array(w_maxima), axis=0)
        zenith0 = hp.get_angle(np.array([0, 0, 1]), self._axis0)

        distances = []
        for i, depth in enumerate(np.hstack((self._depths, [d for d in depths2 if d not in self._depths]))):
            distances.append(self._at.get_distance_xmax_geometric(
                zenith0, depth, observation_level=self._shower[shp.observation_level]))
        distances = np.asarray(distances)

        direction_rec, core_rec, opening_angle_sph, opening_angle_sph_std, core_std = self.fit_axis(
            found_points, weights, self._axis0, self._core0, full_output=True
        )
        logger.info(f"core (refined): {list(np.round(core_rec, 3))} +- {list(np.round(core_std, 3))} m")

        if self._refine_axis:
            logger.info(
                f"Opening angle with axis0 (refined): {opening_angle_sph / units.deg: .3f} +- {opening_angle_sph_std / units.deg: .3f} deg"
            )

        if self._refine_axis and self._debug:
            plot_shower_axis_points(
                np.array(found_points), np.array(weights), self._shower
            )

        self._data["core"] = {"opt": core_rec * units.m, "std": core_std * units.m}
        z, a = hp.cartesian_to_spherical(*direction_rec)
        self._data["zenith"] = z * units.rad
        self._data["azimuth"] = a * units.rad
        self._data["opening_angle_v0"] = {
            "opt": opening_angle_sph * units.rad, "std": opening_angle_sph_std * units.rad}
        self._data["found_points"] = np.array(found_points)
        self._data["weights"] = np.array(weights)

        return direction_rec, core_rec

    def fit_axis(self, points, weights, axis0, core0, full_output: bool = False):
        """
        Fit the shower axis through a set of weighted points, helped by an initial guess.

        Parameters
        ----------
        points, weights: np.ndarray
        axis0, core0: np.ndarray
            Initial guess
        full_output: bool, default: False

        Returns
        -------
        if full_output is False:
            direction_rec, core_rec: np.ndarray

        if full_output is True:
            direction_rec, core_rec: np.ndarray
            opening_angle: float
                Opening angle of the reconstructed axis with the initial geometry
            np.sqrt(opening_angle_var): float
                Opening angle uncertainty
            np.sqrt([corex_var, corey_var]): np.ndarray
                Core uncertainty
        """
        points = np.array(points)
        weights = np.array(weights)

        popt, pcov = curve_fit(
            interferometry.fit_axis, points[:, -1], points.flatten(),
            sigma=np.amax(weights) / np.repeat(weights, 3),
            p0=[*hp.cartesian_to_spherical(*axis0), core0[0], core0[1]]
        )
        direction_rec = hp.spherical_to_cartesian(*popt[:2])
        core_rec = interferometry.fit_axis(np.zeros(1), *popt)  # 0 == GROUND

        if not full_output:
            return direction_rec, core_rec

        thetavar, phivar, corex_var, corey_var = np.diag(pcov)
        opening_angle, opening_angle_var = opening_angle_spherical(*hp.cartesian_to_spherical(
            *direction_rec), *hp.cartesian_to_spherical(*axis0), thetavar, phivar)

        return direction_rec, core_rec, opening_angle, np.sqrt(opening_angle_var), np.sqrt([corex_var, corey_var])

    @register_run()
    def run(self,
            evt: Event,
            det: Optional[DetectorBase] = None,
            station_ids: Optional[list] = None,
            signal_kind="power",
            relative_signal_treshold: float = 0.,
            depths: np.ndarray = np.arange(
                400, 900, 100) * units.g / units.cm2,
            mc_jitter: float = 0 * units.ns):
        """
        Run interferometric reconstruction of depth of coherent signal.

        Parameters
        ----------
        evt : Event
            Event to run the module on.
        det : Detector, optional
            Detector description
        station_ids: list, default: None
            station_ids which will be read out. `None` implies using all available stations (`Event.get_station_ids()`)
        signal_kind: {'power', 'hilbert_sum', 'amplitude'}
            Quantity to time-integrate. Using 'power' (the default) squares the summed trace,
            which is recommended (square of sum if larger than sum of squares)
        relative_signal_treshold: float, default: 0.
            Fraction of the strongest measured fluence that should be contained
            in a trace for it to be used in the reconstruction.
        depths: list, default: [500, 600, 700, 800, 900, 1000]
            Slant depths in g/cm^2  at which to sample lateral profiles.
        mc_jitter: float, default: None
            Standard deviation of Gaussian noise added to timings, if set.
        """
        self._signal_kind = signal_kind
        self._signal_threshold = relative_signal_treshold

        self._depths = depths / units.g * units.cm2
        self._binsize = self._depths[1] - self._depths[0]
        self._mc_jitter = mc_jitter / units.ns

        self.set_station_data(evt, det, station_ids=station_ids)

        direction_rec, core_rec = self.reconstruct_shower_axis()

        self._shower.set_parameter(
            shp.interferometric_shower_axis, direction_rec)
        self._shower.set_parameter(shp.interferometric_core, core_rec)

    def end(self):
        """
        End the module

        Returns
        -------
        self._data: dict:
            Dictionary containing some additional information which can be useful for studies.
        """
        return self._data


class efieldInterferometricLateralReco(efieldInterferometricAxisReco):
    """
    A module to determine the lateral width (vxB and vxvxB FWHM/FW80M) of the RIT beamformed map.
    """

    def __init__(self):
        super().__init__()
        self._lateral_sample_count = None
        self._lateral_vxB_width = None

    def begin(self,
              shower: BaseShower,
              core: Optional[np.ndarray] = None,
              axis: Optional[np.ndarray] = None,
              window: float = 150*units.ns,
              use_sim_pulses: bool = False,
              use_channels: bool = False,
              multiprocessing: bool = False,
              sample_angular_resolution: float = 0.005*units.deg,
              initial_grid_spacing: float = 40*units.m,
              cross_section_width: float = 200*units.m,
              relative_signal_treshold: float = 0.,
              lateral_vxB_sample_count: int = 80,
              lateral_vxB_width: float = 2000*units.m,
              debug: bool = False
              ):
        """
        Set module config.

        Parameters
        ----------
        shower: BaseShower
            shower object to extract geometry (zenith, azimuth, core) and observation information
            (magnetic_field_vector, observation_level, atmospheric_model).
            Conventional choice: Event.get_first_shower(), Event.get_first_sim_shower().
        core: np.ndarray, optional
            Overwrite the core found in ``shower`` with this value. This is incompatible with ``core_spread != 0``.
        axis: np.ndarray, optional
            Overwrite the axis found in ``shower`` with this value. This is incompatible with ``axis_spread != 0``.
        window: float, default: 150 * units.ns
            Width of the trace section (centered on the trace maximum) which will be used.
        use_sim_pulses: bool, default: False
            If True, the processed event will be handled as simulated, using SimStation instead of Station,
            and assuming the geometry of ``shower`` is the truth.
        use_channels: bool, default: False
            If True, use the dominant polarization of channel traces, which are usually undefined for simualated events. This will give very unexpected results, not recommended!
        multiprocessing: bool, default: False
            Flag to enable multiprocessing. Not to be used for jobs, but useful in debugging.
        sample_angular_resolution: float, default: 0.005*units.deg
            Final angular resolution of RIT method.
        initial_grid_spacing: float, default: 60*units.m
        cross_section_width: float, default: 1000*units.m
        relative_signal_treshold: float, default = 0.0
            Fraction of strongest signal necessary for a trace to be used for RIT. Default of 0 includes all channels.
        lateral_vxB_sample_count: int, default: 80
            Amount of RIT samples logarithmically spaces along the vxB and vxvxB directions in the shower plane.
        lateral_vxB_width: float, default: 2000*units.m
            Total length spanned by the samples in the vxB and vxvxB directions in the shower plane.
        debug : bool, default: False
            If True, show some debug plots.
        """
        self._debug = debug
        self._window = window
        self._use_sim_pulses = use_sim_pulses
        self._use_channels = use_channels
        self._multiprocessing = multiprocessing
        self._angres = sample_angular_resolution / units.radian
        self._cross_section_width = cross_section_width / units.m
        self._initial_grid_spacing = initial_grid_spacing / units.m
        self._lateral_vxB_sample_count = lateral_vxB_sample_count
        self._lateral_vxB_width = lateral_vxB_width
        self._signal_threshold = relative_signal_treshold

        self.set_geometry(shower, core=core, axis=axis)
        self.update_atmospheric_model_and_refractivity_table()

    def get_lateral_profile(self, p_axis: np.ndarray, xs: np.ndarray, orientation: str = "vxB"):
        """
        Obtain the lateral profile in the shower plane through a point p_axis.

        Parameters
        ----------
        p_axis: np.ndarray
            Shower plane anchor and origin.
        xs: np.ndarray
            Sample array in the shower plane (anchored at p_axis) coordinates.
        orientation: {"vxB", "vxvxB"}
            Orientation of the lateral profile

        Returns
        -------
        signals: np.ndarray
            Array of beamformed traces at the sample locations. Same shape as xs.
        """
        assert orientation in ["vxB", "vxvxB"]
        signals = np.zeros(len(xs))
        for xdx, x in enumerate(tqdm(xs)):
            if orientation == "vxB":
                p = p_axis + \
                    self._cs.transform_from_vxB_vxvxB(np.array([x, 0, 0]))
            else:
                p = p_axis + \
                    self._cs.transform_from_vxB_vxvxB(np.array([0, x, 0]))

            sum_trace = interferometry.interfere_traces_rit(
                p, self._positions, self._traces, self._times, tab=self._tab)

            signal = interferometry.get_signal(
                sum_trace, self._tstep, kind=self._signal_kind)
            signals[xdx] = signal
        return signals

    def find_fwhm(self, depth: float):
        """
        Find the width of the RIT beamformed profile at a certain atmospheric depth.

        Parameters
        ----------
        depth: float
            Atmospheric depth in internal units.

        Returns
        -------
        fwhm: dict[str, float]
            Dictionary containing different width measures.
            keys:   fwhm_vxB
                    fwhm_vxvxB
                    fw80m_vxB
                    fw80m_vxvxB

        """
        old_debug = self._debug
        self._debug = False
        p_max, weight, ground_grid_uncertainty = self.sample_lateral_cross_section(
            depth, self._core, self._axis, self._cross_section_width, self._initial_grid_spacing)
        self._debug = old_debug

        _, dr_ref_target = self.get_paxis_dr_target(
            depth, self._core, self._axis)
        max_dist = self._lateral_vxB_width / 2
        xlims = np.array([-max_dist, max_dist])

        n = self._lateral_vxB_sample_count // 2
        xs = np.hstack([np.geomspace(xlims[0], -dr_ref_target / 2, n),
                       np.geomspace(dr_ref_target / 2, xlims[1], n)])

        lateral_vxB_profile = self.get_lateral_profile(p_max, xs)
        lateral_vxvxB_profile = self.get_lateral_profile(
            p_max, xs, orientation="vxvxB")

        fwhm = {}
        for orientation, profile in zip(["vxB", "vxvxB"], [lateral_vxB_profile, lateral_vxvxB_profile]):
            pk_half = scipy.signal.peak_widths(
                profile, [np.argmax(profile)], rel_height=.5)

            lips = pk_half[2][0]
            rips = pk_half[3][0]
            fwhm[f"fwhm_{orientation}"] = xs[round(rips)] - xs[round(lips)]

            pk_80 = scipy.signal.peak_widths(
                profile, [np.argmax(profile)], rel_height=.2)

            lips = pk_80[2][0]
            rips = pk_80[3][0]
            fwhm[f"fw80m_{orientation}"] = xs[round(rips)] - xs[round(lips)]

            orientation_labels = {"vxB": r"$\mathbf{v}\times\mathbf{B}$",
                                  "vxvxB": r"$\mathbf{v}\times\mathbf{v}\times\mathbf{B}$"}
            if self._debug:
                # pk_full = scipy.signal.peak_widths(profile,
                #                                    [np.argmax(profile)],
                #                                    rel_height=1)
                # fwfm = xs[round(pk_full[3][0])] - xs[round(pk_full[2][0])]
                profile_normalized = profile / max(profile)
                ax = plt.figure(figsize=(3.4, 3.4)).add_subplot()

                ax.scatter(xs, profile_normalized, marker="v", s=1.)
                if orientation == "vxB":
                    ax.set_xlabel(r"$x_{\mathbf{v}\times\mathbf{B}}$ [m]")
                elif orientation == "vxvxB":
                    ax.set_xlabel(
                        r"$y_{\mathbf{v}\times\mathbf{v}\times\mathbf{B}}$ [m]")
                ax.set_ylabel("Fluence (normalized)")
                ax.hlines(pk_half[1][0] / max(profile), xs[round(pk_half[2][0])], xs[round(pk_half[3][0])],
                          color="red", label=f"FWHM {fwhm[f'fwhm_{orientation}']: .2f} m", lw=0.8, ls="-")
                ax.hlines(pk_80[1][0] / max(profile), xs[round(pk_80[2][0])], xs[round(pk_80[3][0])],
                          color="green", label=f"FW80M {fwhm[f'fw80m_{orientation}']: .2f} m", lw=0.8, ls="-")
                # ax.hlines(pk_full[1][0], xs[round(pk_full[2][0])], xs[round(pk_full[3][0])], color="green", label=f"FWFM {fwfm: .2f} m", lw=1, ls=":")
                ax.set_title("Orientation: " +
                             orientation_labels[orientation], loc="left")
                ax.legend(fontsize="small", loc="lower right")
                ax.set_xlim(-500, 500)
                plt.show()

        return fwhm

    @register_run()
    def run(self,
            evt: Event,
            det: Optional[DetectorBase] = None,
            depth: Optional[float] = None,
            station_ids: Optional[list] = None, signal_kind="power",
            relative_signal_treshold: float = 0.,
            mc_jitter: float = 0 * units.ns,
            ):
        """
        Run lateral RIT reco.

        Parameters
        ----------
        evt : Event
            Event to run the module on.
        det : Detector, optional
            Detector description
        depth: Optional[float] (default: None)
            slant depth in g/cm^2  at which to sample lateral profile.
            If None looks for showerParameters.interferometric_shower_maximum in the active shower object.
            If True, take electric field trace from sim_station
        station_ids: list, default: None
            station_ids which will be read out. `None` implies using all available stations (`Event.get_station_ids()`)
        relative_signal_treshold: float, default: 0.0
           Fraction of strongest signal necessary for a trace to be used for RIT. Default of 0 includes all channels.
        mc_jitter: float, default: None
            Standard deviation of Gaussian noise added to timings, if set.
        """
        self._signal_kind = signal_kind
        self._signal_threshold = relative_signal_treshold

        self._mc_jitter = mc_jitter / units.ns

        self.set_station_data(evt, det, station_ids=station_ids)

        if depth is None:
            depth = self._shower[shp.interferometric_shower_maximum]

        fwhm = self.find_fwhm(depth / units.g * units.cm2)
        self._shower.set_parameter(
            shp.interferometric_fwhm, fwhm["fwhm_vxB"] * units.m)
        self._shower.set_parameter(
            shp.interferometric_fw80m_vxB, fwhm["fw80m_vxB"] * units.m)
        self._shower.set_parameter(
            shp.interferometric_fw80m_vxvxB, fwhm["fw80m_vxvxB"] * units.m)
        self._data.update(fwhm)
        logger.info(f"Interferometric vxB FWHM at {depth / units.g * units.cm2: .2f} g/cm2: {fwhm['fwhm_vxB']: .2f} m")

    def end(self):
        """
        End the module

        Returns
        -------
        self._data: dict
            Dictionary containing some additional information which can be useful for studies.
        """
        return self._data


def plot_shower_axis_points(points: np.ndarray, weights: np.ndarray, shower: Optional[BaseShower] = None):
    if shower is not None:
        cs = coordinatesystems.cstrafo(
            shower[shp.zenith], shower[shp.azimuth], shower[shp.magnetic_field_vector])
    else:
        logger.warning(
            "plot_shower_axis: no shower passed, assuming showerplane points")
        cs = None

    logger.debug(f"points shape: {points.shape}")
    assert points.shape[:-1] == weights.shape
    if len(points.shape) == 3:
        permutation = np.argsort(points[0, :, -1])
        points_showerplane = []
        for axispoints in points:
            if shower is not None:
                axispoints_showerplane = cs.transform_to_vxB_vxvxB(
                    axispoints, shower[shp.core])
            else:
                axispoints_showerplane = axispoints
            points_showerplane.append(axispoints_showerplane)
        points_showerplane = np.array(points_showerplane)
        points_showerplane[..., -1] *= -1  # v-> -v
        quantiles = np.quantile(points_showerplane, [.16, .5, .84], axis=0)[
            :, permutation]
        assert np.all(quantiles[0] <= quantiles[1]) and np.all(
            quantiles[1] <= quantiles[2])
        median_showerplane = quantiles[1]
        delta_showerplane = quantiles[[0, 2]]
        logger.debug(f"permutation_shape {permutation.shape}")
        logger.debug(f"permuted shape: {median_showerplane.shape}")
        weights_median = np.median(weights, axis=0)[permutation]
        weights_median /= np.max(weights_median)

    elif len(points.shape) == 2:
        permutation = np.argsort(points[:, -1])
        if shower is not None:
            median_showerplane = cs.transform_to_vxB_vxvxB(
                points[permutation], shower[shp.core])
        else:
            median_showerplane = np.copy(points)[permutation]
        median_showerplane[..., -1] *= -1
        delta_showerplane = None
        weights_median = np.copy(weights)[permutation]
        weights_median /= np.max(weights_median)

    else:
        raise ValueError("points must be 2d or 3d.")

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 15], hspace=0.1)
    x, y, z = median_showerplane.T
    ax_z = fig.add_subplot(gs[1, 0])
    ax_w = fig.add_subplot(gs[1, 1])
    weight_modifier = 10
    sm_z = ax_z.scatter(x, y, c=z, s=weight_modifier * weights_median,
                        cmap=plt.cm.viridis, marker="v", zorder=1.1)
    if delta_showerplane is not None:
        assert np.all(delta_showerplane[0] <= median_showerplane) and np.all(
            median_showerplane <= delta_showerplane[1])
        delta_showerplane[0] = median_showerplane - delta_showerplane[0]
        delta_showerplane[1] -= median_showerplane
        dx, dy, dz = np.moveaxis(delta_showerplane, -1, 0)

        eb_kwargs = {"capsize": 1.5, "zorder": .99,
                     "elinewidth": .1, "capthick": .1, "lw": .1, "ecolor": "k"}

        ax_z.errorbar(x, y, dy, dx, **eb_kwargs)
    cb = colorbar.Colorbar(ax=fig.add_subplot(
        gs[0, 0]), orientation="horizontal", label="RIT -v [m]", ticklocation="top", mappable=sm_z)
    cb.ax.tick_params(labelsize="x-small")
    ax_z.set_xlabel("RIT vxB [m]", fontsize="small")
    ax_z.set_ylabel("RIT vxvxB [m]", fontsize="small")

    sm_w = ax_w.scatter(x, z, c=y, s=weight_modifier *
                        weights_median, cmap=plt.cm.plasma, marker="v", zorder=1.1)
    if delta_showerplane is not None:
        ax_w.errorbar(x, z, dz, dx, **eb_kwargs)
    cb = colorbar.Colorbar(ax=fig.add_subplot(
        gs[0, 1]), orientation="horizontal", label="RIT vxvxB [m]", ticklocation="top", mappable=sm_w)
    cb.ax.tick_params(labelsize="x-small")
    ax_w.set_xlabel("RIT vxB [m]", fontsize="small")
    ax_w.set_ylabel("RIT -v [m]", fontsize="small")
    for ax in [ax_z, ax_w]:
        ax.spines[["top", "right"]].set_visible(True)
        ax.tick_params(labelsize="x-small")
        ax.grid(visible=True, lw=.2)

    if shower is not None:
        fig.suptitle(f"zenith {shower[shp.zenith] / units.deg: .1f} deg, azimuth {shower[shp.azimuth] / units.deg: .1f} deg")
    plt.show()


def plot_lateral_cross_section(
        xs, ys, signals, mc_pos=None, fname=None, title=None, is_mc: bool = False):
    """
    Plot the lateral distribution of the beamformed singal (in the vxB, vxvxB directions).

    Parameters
    ----------
    xs : np.ndarray
        Positions on x-axis (vxB) at which the signal is sampled (on a 2d grid)
    ys : np.ndarray
        Positions on y-axis (vxvxB) at which the signal is sampled (on a 2d grid)
    signals : np.ndarray
        Signals sampled on the 2d grid defined by xs and ys.
    mc_pos : np.ndarray, default: None
        Intersection of the (MC-)axis with the "slice" of the lateral distribution plotted.
    fname : str, default: None
        Name of the figure. If given the figure is saved, if fname is None the figure is shown.
    title : str, default: None
        Title of the figure (Default: None)
    is_mc: bool, default: False
        Modifes label of mc_pos to mean intersect of initial guess with lateral section, necessarily zero.
    """

    yy, xx = np.meshgrid(ys, xs)

    ax = plt.figure().add_subplot()
    pcm = ax.pcolormesh(xx, yy, signals, shading='gouraud')
    ax.scatter(xx, yy, s=2, color="k", facecolors="none", lw=0.5)
    cbi = plt.colorbar(pcm, pad=0.02)
    cbi.set_label(r"$f_{B_{j}}$ [eV$\,$m$^{-2}$]")

    idx = np.argmax(signals)
    xfound = xs[int(idx // len(ys))]
    yfound = ys[int(idx % len(ys))]

    ax.scatter(xfound, yfound, s=2, color="blue", label="Max")
    if mc_pos is not None:
        if is_mc:
            label = r"Intersection $\hat{a}_\mathrm{MC}$"
        else:
            label = r"Intersection $\hat{a}_0$"
        ax.scatter(*mc_pos[:2], marker="*", color="red", label=label, s=2)

    ax.legend()
    ax.set_ylabel(
        r"$\mathbf{v} \times \mathbf{v} \times \mathbf{B}$ [m]")
    ax.set_xlabel(r"$\mathbf{v} \times \mathbf{B}$ [m]")
    if title is not None:
        ax.set_title(title)  # r"slant depth = %d g / cm$^2$" % depth)

    plt.tight_layout()
    if fname is not None:
        # "rit_xy_%d_%d_%s.png" % (depth, iloop, args.label))
        plt.savefig(fname)
    else:
        plt.show()


def normal(x, A, x0, sigma):
    """ Gauss curve """
    return A / np.sqrt(2 * np.pi * sigma ** 2) \
        * np.exp(-1 / 2 * ((x - x0) / sigma) ** 2)


def gaussian_2d(xy, A, mux, muy, sigmax, sigmay):
    mu = np.array([mux, muy])
    sigma = np.array([sigmax, sigmay])
    return A * np.exp(np.sum(-np.square(xy -
                      mu[:, np.newaxis] / (sigma[:, np.newaxis])) / 2, axis=0))

def opening_angle_spherical(theta1, phi1, theta2, phi2, theta1_var, phi1_var):
    """Give the the opening angle and variance on the opening angle between two vectors with spherical coordinates (1, theta, phi), asuming the second vector is known, such that theta2_var, phi2_var are not asked"""
    c1 = np.cos(theta1)
    c2 = np.cos(theta2)
    s1 = np.sin(theta1)
    s2 = np.sin(theta2)
    s12 = np.sin(phi1 - phi2)
    c12 = np.cos(phi1 - phi2)
    arg = s1*s2*c12 + c1*c2
    opening_angle_opt = np.arccos(arg)
    opening_angle_var = (1/(1-arg**2)) * ((c1*s2*c12 - s1*c2)
                                          ** 2 * theta1_var + (s1*s2*s12)**2 * phi1_var)
    return opening_angle_opt, opening_angle_var
