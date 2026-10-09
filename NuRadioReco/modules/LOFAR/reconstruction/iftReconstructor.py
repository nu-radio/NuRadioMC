"""
LOFAR IFT reconstruction module for NuRadioReco.

.. moduleauthor:: Karen Terveer <karen.terveer@fau.de>
"""

import logging
import os

import numpy as np
import radiotools.helper as hp
from NuRadioReco.framework.parameters import (
    electricFieldParameters,
    showerParameters,
    stationParameters,
)
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.utilities.LOFAR import iftDataHelpers
from NuRadioReco.utilities.LOFAR import qualityCuts

# Module-level constants used by the IFT reconstructor.
# Timing observables reach the likelihood in SI seconds; the conversion from
# NuRadioReco's internal units happens in iftDataHelpers. Hence the plain SI
# speed of light here rather than a unit-system constant.
_C_LIGHT = 299792458.0

_FLUENCE_RELATIVE_SYSTEMATIC_ERROR = 0.1
# Below this many usable noise windows an antenna's own mean and std are too
# noisy to be worth preferring over the global ones.
_MIN_WINDOWS_FOR_PER_ANTENNA_SIGMA = 4
_LDF_ENERGY_SCALE_STD = 0.04
_FAR_STATION_FLUENCE_ONLY_SNR = 15.0
_FAR_STATION_MIN_HIGH_SNR_ANTENNAS = 12
_FLUENCE_MIN_NOISE_FACTOR = 0.01
_STATION_SNR_THRESHOLD = 3.0
_STATION_MIN_ANTENNAS = 10
_MIN_STATIONS_REQUIRED = 1
# Reject events without a qualifying station unless fallback is enabled.
_STATION_FILTER_FALLBACK = False
_FALLBACK_ANTENNA_THRESHOLD = 48
_FALLBACK_SNR_THRESHOLD = 6.0
_CAUSALITY_CUT_DISTANCE_M = 600.0 * units.m
_TIMING_UNCERTAINTY_S = 1.5 * units.ns / units.s
_TIMING_SYST_INFLATION = 1.6
_TIMING_UNCERT_THRESHOLD = 25.0 * units.ns / units.s
_MIN_TIMING_POINTS = 24
_MIN_TIMING_POINTS_PRUNE = 12
_MIN_GOOD_TIMING_POINTS_RECO = 0
_MIN_ABSOLUTE_NEIGHBORS = 8
_DEFAULT_CORE_PRIOR_STD_M = 120.0
# Widest Xmax prior considered; the actual prior is this intersected with the
# range where the LDF shape splines are defined (see _xmax_prior_bounds).
_XMAX_PRIOR_MIN_GCM2 = 400.0
_XMAX_PRIOR_MAX_GCM2 = 1200.0
_XMAX_PRIOR_MIN_WIDTH_GCM2 = 150.0
_FLUENCE_DXMAX_PRECISION_GPCM2 = 16.3
_TIMING_DXMAX_PRECISION_GPCM2 = 35.0
_DEFAULT_N_VI_ITERATIONS = 10
_DEFAULT_N_SAMPLES = 80
_DEFAULT_SEED = 27
_SAMPLING_MODE = "linear_sample"
_RESAMPLING_MODE = "nonlinear_resample"
#: VI iterations drawn with the linear sampler before switching to the nonlinear
#: one. Raising it past ``n_iterations`` keeps the whole fit linear, which is much
#: cheaper and is enough for testing or quick initial reconstruction of new events.
_N_LINEAR_ITERATIONS = 2
#: Radiation-energy prior scale [eV] and logarithmic width.
_ERAD_PRIOR_SCALE = 10**6.5
_ERAD_PRIOR_STD = 3.5
# The fluence and timing models share one core prior; the wavefront core is that
# core plus this offset. Fixed prior width [m] of that offset.
_CORE_TIMING_OFFSET_STD_M = 10.0
_CONV_PATIENCE = 3
_CONV_MIN_ITERS = 3
_CONV_TOL_REDCHISQ = 0.01
_BROAD_SCAN_RANGE_DEG = 10.0
_BROAD_STEP_DEG = 1.0
_BROAD_SMOOTHING_NS = 5.0
_RECO_STAGES = [{"window": 300.0}, {"window": 100.0}]


def _stack_noise_blocks(blocks):
    """Vertically stack per-station (n_antenna, n_window) noise blocks.

    Stations can end up with different window counts -- their traces differ in
    length -- so the narrower blocks are NaN-padded to the widest before
    stacking. Every consumer of the result uses nan-aware reductions.
    """
    mats = [np.asarray(b, dtype=float) for b in blocks if np.size(b)]
    mats = [m if m.ndim == 2 else m.reshape(1, -1) for m in mats]
    if not mats:
        return np.empty((0, 0), dtype=float)
    width = max(m.shape[1] for m in mats)
    padded = [
        m if m.shape[1] == width else
        np.hstack([m, np.full((m.shape[0], width - m.shape[1]), np.nan)])
        for m in mats
    ]
    return np.vstack(padded)


class iftReconstructor:
    """
    LOFAR IFT reconstruction module.

    Run timing and station selection, variational inference, posterior filtering,
    and shower-parameter export. Set ``dry_run=True`` to stop after selection.
    """

    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.LOFAR.iftReconstructor")
        self.__settings = {}

    def begin(
        self,
        fluence_window_ns=100.0,
        snr_cut=0.0,
        timing_uncertainty_s=1.5e-9,
        min_timing_points=0,
        n_iterations=_DEFAULT_N_VI_ITERATIONS,
        n_samples=_DEFAULT_N_SAMPLES,
        random_seed=_DEFAULT_SEED,
        enable_fluence_correlated_field=True,
        enable_timing_correlated_field=True,
        atmosphere_dir=None,
        gdas_cache_dir=None,
        use_fallback_atmosphere=False,
        export_posterior_samples=False,
        output_directory=None,
        logger_level=logging.NOTSET,
        scan_range_deg=5.0,
        step_deg=0.5,
        sigma_ns=5.0,
        core_prior_std_m=_DEFAULT_CORE_PRIOR_STD_M,
        max_signal_fallback=True,
        station_filter_fallback=_STATION_FILTER_FALLBACK,
        max_signal_snr_threshold=iftDataHelpers.MAX_SIGNAL_SNR_THRESHOLD,
        n_linear_iterations=_N_LINEAR_ITERATIONS,
        dry_run=False,
    ):
        """
        Configure the reconstruction.

        ``max_signal_fallback`` only takes effect where the reconstruction would
        otherwise abort outright: if both the standard and the broad direction
        search return no timing data at all, every channel is searched blind for
        its largest excursion instead (see
        :func:`~NuRadioReco.utilities.LOFAR.iftDataHelpers.extract_max_signal_timing_data`).
        It never changes the result of an event where the normal search found
        something. Events reconstructed this way are flagged with
        ``used_max_signal_fallback`` in the exported ``.npz``.

        ``station_filter_fallback`` defaults to False. Events without a station
        meeting ``_STATION_MIN_ANTENNAS`` at ``_STATION_SNR_THRESHOLD`` are
        rejected with reason ``no_station_above_snr_threshold``. Set it to True
        to retain the loudest station when none meets the threshold.

        ``dry_run`` runs every stage that can still reject an event — pulse
        search, direction fits, RECO_STAGES, timing quality, antenna positions — and
        then stops right before the atmosphere lookup and the VI fit. The verdict
        is logged as a single ``RECO_GATE:`` line (``PASS`` or ``FAIL reason=...``),
        so a cheap pass over a whole event set can collect the events that are
        worth spending the full reconstruction on.
        """
        self.__settings = {
            "fluence_window_ns": fluence_window_ns,
            "snr_cut": snr_cut,
            "timing_uncertainty_s": timing_uncertainty_s,
            "min_timing_points": min_timing_points,
            "n_iterations": n_iterations,
            "n_samples": n_samples,
            "random_seed": random_seed,
            "enable_fluence_correlated_field": enable_fluence_correlated_field,
            "enable_timing_correlated_field": enable_timing_correlated_field,
            "atmosphere_dir": atmosphere_dir,
            "gdas_cache_dir": gdas_cache_dir,
            "use_fallback_atmosphere": use_fallback_atmosphere,
            "export_posterior_samples": export_posterior_samples,
            "output_directory": output_directory,
            "scan_range_deg": scan_range_deg,
            "step_deg": step_deg,
            "sigma_ns": sigma_ns,
            "core_prior_std_m": core_prior_std_m,
            "max_signal_fallback": max_signal_fallback,
            "station_filter_fallback": station_filter_fallback,
            "max_signal_snr_threshold": max_signal_snr_threshold,
            "n_linear_iterations": n_linear_iterations,
            "dry_run": dry_run,
        }
        self.logger.setLevel(logger_level)

    def _reco_gate_fail(self, reason):
        """Tag an abort with a machine-readable verdict for the job scripts."""
        self.logger.status("RECO_GATE: FAIL reason=%s", reason)

    def _select_stations(self, fluences, station_ids, noise_mean, fallback=False):
        """Apply the station gate, optionally retaining the loudest station."""
        counts = {
            int(sid): int(
                np.sum(
                    fluences[station_ids == sid] / noise_mean >= _STATION_SNR_THRESHOLD
                )
            )
            for sid in np.unique(station_ids)
        }
        selected = {
            sid for sid, count in counts.items() if count >= _STATION_MIN_ANTENNAS
        }
        if len(selected) < _MIN_STATIONS_REQUIRED and fallback and counts:
            ranked = sorted(
                counts,
                key=lambda sid: np.max(fluences[station_ids == sid]),
                reverse=True,
            )
            selected = set(ranked[:_MIN_STATIONS_REQUIRED])
            self.logger.warning("Station filter fallback: keeping loudest station(s).")
        if len(selected) < _MIN_STATIONS_REQUIRED:

            with np.errstate(divide="ignore", invalid="ignore"):
                _snr = np.asarray(fluences, dtype=float) / noise_mean
            _rank = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
            self.logger.status(
                "STATION_GATE: need %d stations with >=%d antennas at SNR>=%.1f; "
                "noise_mean=%.3e n_ant=%d SNR p50=%.2f p90=%.2f max=%.2f; "
                "best stations (id:n_above) %s",
                _MIN_STATIONS_REQUIRED,
                _STATION_MIN_ANTENNAS,
                _STATION_SNR_THRESHOLD,
                noise_mean,
                len(_snr),
                float(np.nanpercentile(_snr, 50)) if _snr.size else float("nan"),
                float(np.nanpercentile(_snr, 90)) if _snr.size else float("nan"),
                float(np.nanmax(_snr)) if _snr.size else float("nan"),
                ", ".join(f"{sid}:{n}" for sid, n in _rank) or "none",
            )
            self._reco_gate_fail("no_station_above_snr_threshold")
            return None
        return selected

    def _timing_count_passes(self, mask):
        """Enforce the final-fit minimum, independently of preliminary pruning."""
        if np.count_nonzero(mask) < _MIN_TIMING_POINTS:
            self._reco_gate_fail("timing_quality_check_failed")
            return False
        return True

    def _retained_samples(self, samples, indices):
        """Require enough retained samples for bias-corrected uncertainties."""
        if len(indices) < 2:
            self._reco_gate_fail("insufficient_posterior_samples")
            return None
        return [samples[i] for i in indices]

    @staticmethod
    def _get_lora_prior(event):
        try:
            shower = event.get_hybrid_information().get_hybrid_shower("LORA")
        except ValueError:
            return None

        prior = {}
        for key, parameter in (
            ("zenith", showerParameters.zenith),
            ("azimuth", showerParameters.azimuth),
            ("energy", showerParameters.energy),
        ):
            if shower.has_parameter(parameter):
                value = float(shower.get_parameter(parameter))
                if np.isfinite(value):
                    prior[key] = value
        if shower.has_parameter(showerParameters.core):
            core = np.asarray(shower.get_parameter(showerParameters.core), dtype=float)
            if core.shape == (3,) and np.all(np.isfinite(core)):
                prior["core"] = core
        return prior

    @staticmethod
    def _valid_direction(zenith, azimuth):
        return (
            zenith is not None and azimuth is not None
            and np.isfinite(zenith) and np.isfinite(azimuth)
            and 0.0 <= zenith <= np.pi
        )

    def _get_initial_geometry(self, event):
        """Use radio direction and an independently available particle core."""
        lora = self._get_lora_prior(event) or {}
        core = lora.get("core", np.array([0.0, 0.0, 7.6 * units.m])).copy()
        radio = self._get_radio_direction(event)
        if radio is not None:
            direction, source = radio, "radio"
        elif self._valid_direction(lora.get("zenith"), lora.get("azimuth")):
            direction, source = (lora["zenith"], lora["azimuth"]), "LORA"
        else:
            direction, source = (0.3, 0.0), "default"
        zenith, azimuth = direction
        return lora, core, float(zenith), float(azimuth) % (2 * np.pi), source

    def _fit_radio_direction(self, positions, times, weights, initial_direction):
        """Refine the radio direction, retaining the input estimate on failure."""
        zenith, azimuth, _ = iftDataHelpers.fit_plane_wave(
            positions, times, weights=weights
        )
        if self._valid_direction(zenith, azimuth):
            self.logger.info(
                "Radio plane-wave fit: zen=%.1f°, az=%.1f°",
                np.rad2deg(zenith), np.rad2deg(azimuth),
            )
            return float(zenith), float(azimuth) % (2 * np.pi)
        self.logger.warning("Plane-wave fit failed; retaining the initial direction.")
        return initial_direction

    @staticmethod
    def _get_radio_direction(event):
        """Return the antenna-count-weighted mean station plane-wave direction.

        Station plane-wave fits take precedence over an event radio shower. The
        remaining unflagged antenna pairs at a station are its statistical
        weight, so a well-populated station contributes more timing
        information than a sparsely populated one.  Averaging unit vectors
        avoids the discontinuity of azimuth angles around 0 and 2 pi.
        """
        directions, weights = [], []
        for station in event.get_stations():
            if station.has_parameter(
                stationParameters.cr_zenith
            ) and station.has_parameter(stationParameters.cr_azimuth):
                zenith = station.get_parameter(stationParameters.cr_zenith)
                azimuth = station.get_parameter(stationParameters.cr_azimuth)
                if not iftReconstructor._valid_direction(zenith, azimuth):
                    continue

                flagged = (
                    station.get_parameter(stationParameters.flagged_channels)
                    if station.has_parameter(stationParameters.flagged_channels)
                    else {}
                )
                group_sizes = {}
                for channel in station.iter_channels():
                    if channel.get_id() in flagged:
                        continue
                    group_id = channel.get_group_id()
                    group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
                usable_groups = [
                    group_id for group_id, size in group_sizes.items() if size >= 2
                ]
                if not usable_groups:
                    continue
                directions.append(hp.spherical_to_cartesian(zenith, azimuth))
                weights.append(len(usable_groups))

        if not directions:
            for shower in event.get_showers():
                if shower.has_parameter(showerParameters.zenith) and shower.has_parameter(
                    showerParameters.azimuth
                ):
                    zenith = shower.get_parameter(showerParameters.zenith)
                    azimuth = shower.get_parameter(showerParameters.azimuth)
                    if iftReconstructor._valid_direction(zenith, azimuth):
                        return float(zenith), float(azimuth) % (2 * np.pi)
            return None

        v = np.average(np.asarray(directions), axis=0, weights=weights)
        norm = np.linalg.norm(v)
        if norm <= 0:
            return None
        v /= norm
        zenith = np.arccos(np.clip(v[2], -1.0, 1.0))
        azimuth = np.arctan2(v[1], v[0]) % (2 * np.pi)
        return zenith, azimuth

    @staticmethod
    def _flatten_station_results(station_results):
        """Concatenate per-station extraction results into flat observable lists.

        ``station_results`` maps a station id onto the 8-tuple returned by
        :func:`~NuRadioReco.utilities.LOFAR.iftDataHelpers.extract_fluence_and_timing_data`.
        Stations without any antenna are skipped. Returns
        ``(posx, posy, fluences, times, is_signal, snrs, noise_fluences, station_ids)``.
        """
        posx, posy, fluences, times = [], [], [], []
        is_signal, snrs, noise_fluences, station_ids = [], [], [], []
        for station_id, result in station_results.items():
            px, py, fl, tm, sig, noise, snr, _ = result
            if len(fl) == 0:
                continue
            posx.extend(px)
            posy.extend(py)
            fluences.extend(fl)
            times.extend(tm)
            is_signal.extend(sig)
            snrs.extend(snr)
            noise_fluences.append(np.asarray(noise, dtype=float))
            station_ids.extend([station_id] * len(fl))
        return (posx, posy, fluences, times, is_signal, snrs,
                _stack_noise_blocks(noise_fluences), station_ids)

    @staticmethod
    def _get_or_create_radio_shower(event):
        showers = list(event.get_showers())
        if showers:
            return showers[0]
        import NuRadioReco.framework.radio_shower

        shower = NuRadioReco.framework.radio_shower.RadioShower(
            shower_id=event.get_id(), station_ids=event.get_station_ids()
        )
        event.add_shower(shower)
        return shower

    def _distance_to_shower_maximum(self, xmax_gcm2, zenith_rad, gdas_file=None):
        """Return geometric distance to shower maximum in meters."""
        import jax.numpy as jnp
        from NuRadioReco.utilities.LOFAR.atmosphere import Atmosphere

        if gdas_file is not None:
            atm = Atmosphere(gdas_file=gdas_file, observation_level=7.6)
        else:
            atm = Atmosphere(model=17, observation_level=7.6)
        return float(
            atm.get_geometric_distance_grammage(
                jnp.array(float(xmax_gcm2)), jnp.array(float(zenith_rad))
            )
        )

    def _xmax_prior_bounds(self, atm_path, zenith_rad):
        """Xmax prior range restricted to where the LDF shape splines are defined.

        ``jaxHelpers.evaluate_bspline`` clips its argument into the knot domain
        instead of extrapolating, so for ``dxmax`` outside
        ``ldf_dxmax_valid_range()`` the footprint shape stops responding to Xmax
        and the fit is free to run to the prior edge.

        Since ``dxmax = X_atm/cos(zenith) - Xmax``, the dxmax window maps to Xmax
        reversed.  Falls back to the full range if anything cannot be evaluated.
        """
        lo, hi = _XMAX_PRIOR_MIN_GCM2, _XMAX_PRIOR_MAX_GCM2
        try:
            from NuRadioReco.utilities.LOFAR.atmosphere import Atmosphere
            from NuRadioReco.utilities.LOFAR.jaxHelpers import ldf_dxmax_valid_range

            dx_lo, dx_hi = ldf_dxmax_valid_range()
            atm = (
                Atmosphere(gdas_file=atm_path, observation_level=7.6)
                if atm_path is not None
                else Atmosphere(model=17, observation_level=7.6)
            )
            # get_atmosphere() is g/m^2; the LDF works in g/cm^2.
            slant = float(atm.get_atmosphere(0.0)) * 1e-4 / np.cos(float(zenith_rad))
            lo = max(lo, slant - dx_hi)
            hi = min(hi, slant - dx_lo)
        except Exception as exc:
            self.logger.warning(
                "Could not restrict the Xmax prior to the LDF spline domain (%s) — "
                "using the full [%.0f, %.0f] g/cm2 range.",
                exc,
                _XMAX_PRIOR_MIN_GCM2,
                _XMAX_PRIOR_MAX_GCM2,
            )
            return _XMAX_PRIOR_MIN_GCM2, _XMAX_PRIOR_MAX_GCM2

        if (
            not np.isfinite(lo)
            or not np.isfinite(hi)
            or (hi - lo) < _XMAX_PRIOR_MIN_WIDTH_GCM2
        ):
            self.logger.warning(
                "LDF-spline-restricted Xmax prior [%.0f, %.0f] is degenerate — "
                "using the full [%.0f, %.0f] g/cm2 range.",
                lo,
                hi,
                _XMAX_PRIOR_MIN_GCM2,
                _XMAX_PRIOR_MAX_GCM2,
            )
            return _XMAX_PRIOR_MIN_GCM2, _XMAX_PRIOR_MAX_GCM2

        self.logger.status(
            "XMAX_PRIOR: [%.0f, %.0f] g/cm2 (LDF splines valid for dxmax in "
            "[%.0f, %.0f]; full range would be [%.0f, %.0f])",
            lo,
            hi,
            dx_lo,
            dx_hi,
            _XMAX_PRIOR_MIN_GCM2,
            _XMAX_PRIOR_MAX_GCM2,
        )
        return lo, hi

    def _set_efield_parameters(self, event, detector, zenith_rad, azimuth_rad):
        """Set per-E-field parameters from the reconstructed shower direction."""
        from scipy.signal import hilbert as scipy_hilbert

        for station in event.get_stations():
            for efield in station.get_electric_fields():
                if not efield.get_channel_ids():
                    continue
                ef_trace = efield.get_trace()
                ef_times = efield.get_times()

                efield.set_parameter(
                    electricFieldParameters.signal_energy_fluence,
                    iftDataHelpers.get_electric_field_energy_fluence(
                        ef_trace, ef_times
                    ),
                )
                efield.set_parameter(electricFieldParameters.zenith, zenith_rad)
                efield.set_parameter(electricFieldParameters.azimuth, azimuth_rad)

                envelope = np.abs(scipy_hilbert(np.linalg.norm(ef_trace, axis=0)))
                efield.set_parameter(
                    electricFieldParameters.signal_time,
                    ef_times[int(np.argmax(envelope))],
                )

    def _reconvert_electric_fields(
        self, event, detector, zenith, azimuth, converter=None
    ):
        """Replace fields atomically; preserve the previous set on failure."""
        if converter is None:
            from NuRadioReco.modules.voltageToEfieldConverterPerChannelGroup import (
                voltageToEfieldConverterPerChannelGroup,
            )

            converter = voltageToEfieldConverterPerChannelGroup()
            converter.begin(use_MC_direction=False)
        for station in event.get_stations():
            station.set_parameter(stationParameters.zenith, zenith)
            station.set_parameter(stationParameters.azimuth, azimuth)
            previous = list(station.get_electric_fields())
            station.set_electric_fields([])
            try:
                converter.run(event, station, detector)
                fields = list(station.get_electric_fields())
                if not fields:
                    raise ValueError("conversion produced no electric fields")
                groups = [tuple(sorted(field.get_channel_ids())) for field in fields]
                if len(set(groups)) != len(groups):
                    raise ValueError("conversion produced duplicate antenna groups")
            except Exception as exc:
                station.set_electric_fields(previous)
                self.logger.warning(
                    "Station %s: restoring %d previous electric fields after "
                    "failed re-conversion: %s",
                    station.get_id(),
                    len(previous),
                    exc,
                )

    @staticmethod
    def _calculate_ecr(model, x):
        """Return calibrated energy using effective Xmax and corrected E_rad."""
        return model.calculate_ecr_jax(x)

    def _reconstruct(self, event, detector):
        # --- Optional dependency imports ---
        try:
            import jax.numpy as jnp
            import nifty.re as jft
            import NuRadioReco.utilities.LOFAR.jaxHelpers  # noqa: F401
            from jax import random, vmap
            from NuRadioReco.utilities.LOFAR.iftModel import (
                SYST_MULT_MAX,
                SYST_MULT_MIN,
                footprintModel,
            )
        except ImportError as exc:
            raise ImportError(
                "Full IFT reconstruction requires 'jax', 'jaxlib', and 'nifty'. "
                "Install these packages to run IFT reconstruction."
            ) from exc

        pf = iftDataHelpers
        s = self.__settings
        n_vi = int(s.get("n_iterations", _DEFAULT_N_VI_ITERATIONS))
        n_samples = int(s.get("n_samples", _DEFAULT_N_SAMPLES))
        seed = int(s.get("random_seed", _DEFAULT_SEED))
        fluence_cf = bool(s.get("enable_fluence_correlated_field", True))
        timing_cf = bool(s.get("enable_timing_correlated_field", True))
        atm_dir = s.get("atmosphere_dir")
        scan_range = float(s.get("scan_range_deg", 5.0))
        step = float(s.get("step_deg", 0.5))
        sigma = float(s.get("sigma_ns", 5.0))
        core_prior_std = float(s.get("core_prior_std_m", _DEFAULT_CORE_PRIOR_STD_M))
        max_signal_fallback = bool(s.get("max_signal_fallback", True))
        station_filter_fallback = bool(
            s.get("station_filter_fallback", _STATION_FILTER_FALLBACK)
        )
        max_signal_snr = float(
            s.get("max_signal_snr_threshold", iftDataHelpers.MAX_SIGNAL_SNR_THRESHOLD)
        )
        used_max_signal_fallback = False
        dry_run = bool(s.get("dry_run", False))
        n_linear_iters = int(s.get("n_linear_iterations", _N_LINEAR_ITERATIONS))
        lora, lora_core, seed_zenith, seed_azimuth, direction_source = (
            self._get_initial_geometry(event)
        )
        self.logger.info("Initial direction source: %s", direction_source)
        shower_dir_guess = (seed_azimuth, seed_zenith)

        # PRE-SCAN: global reference pulse from 30 closest antennas
        all_ants = []
        for station in event.get_stations():
            sid = station.get_id()
            abs_pos = detector.get_absolute_position(sid)
            for ef in station.get_electric_fields():
                ch_ids = ef.get_channel_ids()
                if len(ch_ids) < 2:
                    continue
                try:
                    rel_pos = detector.get_relative_position(sid, ch_ids[0])
                except Exception:
                    rel_pos = np.zeros(3)
                pos = rel_pos + abs_pos
                dist = np.linalg.norm(pos[:2] - lora_core[:2])
                all_ants.append({"sid": sid, "dist": dist, "ef": ef, "pos": pos})

        all_ants.sort(key=lambda a: a["dist"])
        global_ref_time = None
        global_ref_pos = None
        best_prescan_snr = 0.0
        for ant in all_ants[:30]:
            try:
                ef_trace = ant["ef"].get_trace()
                # Use the raw trace times. Data traces start at t=0,
                # simulated traces start near -1300 ns, which moves the ROI clean off the pulse.
                ef_times = ant["ef"].get_times()
                t, snr = pf.simple_hilbert_finder(ef_trace, ef_times)
                if t is not None and snr > best_prescan_snr:
                    best_prescan_snr = snr
                    global_ref_time = t
                    global_ref_pos = ant["pos"]
            except Exception:
                continue

        if global_ref_time is not None:
            self.logger.info(
                "Pre-scan: E-field pulse SNR=%.1f at t=%.1f ns",
                best_prescan_snr,
                global_ref_time / units.ns,
            )

        # TWO-PASS DIRECTION SEARCH
        search_attempts = [
            {
                "name": "standard",
                "scan_range": scan_range,
                "step": step,
                "sigma": sigma,
            },
            {
                "name": "broad",
                "scan_range": _BROAD_SCAN_RANGE_DEG,
                "step": _BROAD_STEP_DEG,
                "sigma": _BROAD_SMOOTHING_NS,
            },
        ]

        posx_p1 = posy_p1 = tm_p1 = snr_p1 = sid_p1 = fl_p1 = nf_p1 = None
        sig_p1 = use_timing_p1 = None

        for attempt in search_attempts:
            self.logger.info("Timing search: %s", attempt["name"].upper())
            station_results = {}
            for station in event.get_stations():
                sid = station.get_id()
                try:
                    station_results[sid] = pf.extract_fluence_and_timing_data(
                        event,
                        sid,
                        detector,
                        shower_dir_guess,
                        _RECO_STAGES[0]["window"],
                        global_ref_time,
                        global_ref_pos,
                        scan_range_deg=attempt["scan_range"],
                        step_deg=attempt["step"],
                        sigma_ns=attempt["sigma"],
                    )
                except Exception as exc:
                    self.logger.debug("Station %s extract failed: %s", sid, exc)

            # Find reference station by highest mean SNR
            best_sid, best_mean_snr = None, -1.0
            for sid, res in station_results.items():
                if len(res[6]) > 0:
                    m = float(np.mean(res[6]))
                    if m > best_mean_snr:
                        best_mean_snr = m
                        best_sid = sid

            # Per-station direction consistency check
            if best_sid is not None:
                ref_dir = station_results[best_sid][7]
                best_ant = (
                    int(np.argmax(station_results[best_sid][6]))
                    if len(station_results[best_sid][6]) > 0
                    else 0
                )
                ref_time = (
                    float(station_results[best_sid][3][best_ant])
                    if len(station_results[best_sid][3]) > 0
                    else 0.0
                )
                ref_pos = (
                    np.array(
                        [
                            station_results[best_sid][0][best_ant],
                            station_results[best_sid][1][best_ant],
                        ]
                    )
                    if len(station_results[best_sid][0]) > 0
                    else np.zeros(2)
                )
                for sid, res in station_results.items():
                    if sid == best_sid:
                        continue
                    dev = np.rad2deg(
                        pf.angular_distance(
                            res[7][1], res[7][0], ref_dir[1], ref_dir[0]
                        )
                    )
                    if dev > 3.0:
                        try:
                            station_results[sid] = pf.extract_fluence_and_timing_data(
                                event,
                                sid,
                                detector,
                                ref_dir,
                                _RECO_STAGES[0]["window"],
                                ref_time * units.s,
                                ref_pos,
                                scan_range_deg=0.0,
                                step_deg=attempt["step"],
                                sigma_ns=attempt["sigma"],
                            )
                        except Exception:
                            pass

            _px, _py, _fl, _tm, _sig, _snr, _nf, _sid = self._flatten_station_results(
                station_results
            )

            if not _tm:
                if attempt["name"] == "standard":
                    self.logger.warning(
                        "Standard search: no timing data — retrying with broad scan."
                    )
                    continue
                if not max_signal_fallback:
                    self.logger.error(
                        "Both direction searches failed: no timing data extracted."
                    )
                    self._reco_gate_fail("no_timing_data")
                    return event

                # Last resort for the faintest events: drop the beamformer and the
                # region of interest entirely and take the largest excursion in
                # every channel, keeping only antennas that are both well above the
                # noise and consistent with one plane-wave arrival plane.
                self.logger.warning(
                    "No timing data from either direction search — falling back "
                    "to blind max-signal search (SNR >= %.1f).",
                    max_signal_snr,
                )
                for station in event.get_stations():
                    sid = station.get_id()
                    try:
                        station_results[sid] = pf.extract_max_signal_timing_data(
                            event,
                            sid,
                            detector,
                            shower_dir_guess,
                            _RECO_STAGES[0]["window"],
                            snr_threshold=max_signal_snr,
                        )
                    except Exception as exc:
                        self.logger.debug(
                            "Station %s max-signal extract failed: %s", sid, exc
                        )

                _px, _py, _fl, _tm, _sig, _snr, _nf, _sid = (
                    self._flatten_station_results(station_results)
                )

                if not _tm:
                    self.logger.error(
                        "Both direction searches failed, and the blind max-signal "
                        "fallback found nothing either: no timing data extracted."
                    )
                    self._reco_gate_fail("no_timing_data")
                    return event

                used_max_signal_fallback = True
                self.logger.warning(
                    "Blind max-signal fallback recovered %d timing point(s) from "
                    "%d station(s). Treat this event's reconstruction with care.",
                    len(_tm),
                    len(set(_sid)),
                )

            posx_p1 = np.array(_px)
            posy_p1 = np.array(_py)
            fl_p1 = np.array(_fl)
            tm_p1 = np.array(_tm)
            sig_p1 = np.array(_sig, dtype=bool)
            snr_p1 = np.array(_snr)
            nf_p1 = np.array(_nf)
            sid_p1 = np.array(_sid)

            # Causality cut + outlier removal + iterative pruning
            kx_g = np.sin(seed_zenith) * np.cos(seed_azimuth)
            ky_g = np.sin(seed_zenith) * np.sin(seed_azimuth)
            geom_d = -(1.0 / _C_LIGHT) * (kx_g * posx_p1 + ky_g * posy_p1)
            t0_g = (
                float(np.median(tm_p1[sig_p1] - geom_d[sig_p1]))
                if np.any(sig_p1)
                else 0.0
            )
            causality = np.abs(tm_p1 - t0_g) <= (_CAUSALITY_CUT_DISTANCE_M / _C_LIGHT)
            use_timing_p1 = sig_p1 & causality

            nm_p1 = float(np.mean(nf_p1)) if nf_p1.size > 0 else 1.0
            snr_fl_p1 = fl_p1 / nm_p1 if nm_p1 > 0 else np.zeros_like(fl_p1)

            if np.sum(use_timing_p1) > _MIN_ABSOLUTE_NEIGHBORS:
                keep = pf.detect_timing_outliers(
                    np.array([posx_p1[use_timing_p1], posy_p1[use_timing_p1]]),
                    tm_p1[use_timing_p1],
                    sid_p1[use_timing_p1],
                    timing_snrs=snr_fl_p1[use_timing_p1],
                )
                idx = np.where(use_timing_p1)[0]
                use_timing_p1[idx[~keep]] = False

            if np.sum(use_timing_p1) > _MIN_TIMING_POINTS_PRUNE:
                use_timing_p1, _, _, _, _, _ = pf.iterative_timing_pruning(
                    shower_dir_guess,
                    np.array([posx_p1, posy_p1]),
                    tm_p1,
                    use_timing_p1,
                    station_ids=sid_p1,
                    timing_snrs=snr_fl_p1,
                )

            n_good = int(np.sum(use_timing_p1))
            if n_good >= _MIN_GOOD_TIMING_POINTS_RECO:
                self.logger.info(
                    "Direction search (%s): %d good timing points.",
                    attempt["name"],
                    n_good,
                )
                break
            elif attempt["name"] == "standard":
                self.logger.warning(
                    "Standard search: %d points — retrying broad.", n_good
                )
            else:
                self.logger.warning(
                    "Broad search: %d points — proceeding with best available.", n_good
                )

        global_fit_zen, global_fit_az = self._fit_radio_direction(
            np.array([posx_p1[use_timing_p1], posy_p1[use_timing_p1]]),
            tm_p1[use_timing_p1], snr_p1[use_timing_p1],
            (seed_zenith, seed_azimuth),
        )

        shower_direction = (global_fit_az, global_fit_zen)

        self._reconvert_electric_fields(event, detector, global_fit_zen, global_fit_az)

        # RECO_STAGES: select best fluence extraction window
        best_res = None
        sel_win = 0.0
        noise_mean = 1.0
        noise_level = 1.0
        nf_by_station = {}

        for i_stage, stg in enumerate(_RECO_STAGES):
            self.logger.info("RECO stage %d: window=%.0f ns", i_stage, stg["window"])
            px_l, py_l, fl_l, tm_l, sig_l, snr_l, nf_l, sid_l = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            _nf_by_station = {}
            for station in event.get_stations():
                sid = station.get_id()
                try:
                    px, py, fl_v, tm_v, is_sig_v, nf, snr_v, _ = (
                        pf.extract_fluence_and_timing_data(
                            event,
                            sid,
                            detector,
                            shower_direction,
                            stg["window"],
                            global_ref_time,
                            global_ref_pos,
                            scan_range_deg=0.0,
                        )
                    )
                except Exception as exc:
                    self.logger.debug(
                        "Stage %d extract failed for station %s: %s", i_stage, sid, exc
                    )
                    continue
                if len(fl_v) > 0:
                    px_l.extend(px)
                    py_l.extend(py)
                    fl_l.extend(fl_v)
                    tm_l.extend(tm_v)
                    sig_l.extend(is_sig_v)
                    snr_l.extend(snr_v)
                    nf_l.append(np.asarray(nf, dtype=float))
                    sid_l.extend([int(sid)] * len(fl_v))
                    if np.size(nf) > 0:
                        _nf_by_station[int(sid)] = np.asarray(nf, dtype=float)

            if not fl_l:
                continue

            # nf_l is one (n_antenna, n_window) block per station, padded with
            # NaN where an antenna yielded fewer windows than its neighbours.
            _nf_stage = _stack_noise_blocks(nf_l)
            _nf_finite = _nf_stage[np.isfinite(_nf_stage)]
            stage_nm = float(_nf_finite.mean()) if _nf_finite.size else 1.0
            stage_ns = float(_nf_finite.std()) if _nf_finite.size > 1 else stage_nm
            curr_snr = (
                np.array(fl_l) / stage_nm if stage_nm > 0 else np.zeros(len(fl_l))
            )
            high_snr_count = int(np.sum(curr_snr > _FALLBACK_SNR_THRESHOLD))
            self.logger.info(
                "  Stage %d: %d antennas with SNR>%.1f",
                i_stage,
                high_snr_count,
                _FALLBACK_SNR_THRESHOLD,
            )

            if (
                high_snr_count >= _FALLBACK_ANTENNA_THRESHOLD
                or i_stage == len(_RECO_STAGES) - 1
            ):
                best_res = (
                    np.array(px_l),
                    np.array(py_l),
                    np.array(fl_l),
                    np.array(tm_l),
                    np.array(sig_l, dtype=bool),
                    np.array(snr_l),
                    _nf_stage,
                    np.array(sid_l),
                )
                sel_win = stg["window"]
                noise_mean = stage_nm
                noise_level = stage_ns
                nf_by_station = _nf_by_station
                self.logger.info("  Selected window: %.0f ns", sel_win)
                break

        if best_res is None:
            self.logger.error("RECO_STAGES: no data found in any stage — aborting.")
            self._reco_gate_fail("no_data_in_any_stage")
            return event

        posx, posy, fl, tm, is_sig, snr, _nfl, sids = best_res

        _blocks = np.asarray(_nfl, dtype=float)
        if _blocks.ndim != 2:
            _blocks = _blocks.reshape(len(fl), -1) if _blocks.size else _blocks
        nfloor = np.full(len(fl), noise_mean, dtype=float)
        nsigma = np.full(len(fl), noise_level, dtype=float)
        _n_ant = len(fl)
        if _blocks.ndim == 2 and _blocks.shape[0] == _n_ant and _blocks.shape[1] > 1:
            with np.errstate(invalid="ignore"):
                _fm = np.nanmean(_blocks, axis=1)
                _fs = np.nanstd(_blocks, axis=1)
                _nwin = np.sum(np.isfinite(_blocks), axis=1)
            # An antenna needs enough windows for its own spread to mean
            # anything; below that it keeps the global figure rather than a
            # std built from two or three numbers.
            _ok = (np.isfinite(_fm) & np.isfinite(_fs) & (_fm > 0)
                   & (_nwin >= _MIN_WINDOWS_FOR_PER_ANTENNA_SIGMA))
            nfloor[_ok] = _fm[_ok]
            nsigma[_ok] = _fs[_ok]
            self.logger.status(
                "NOISE_PER_ANT: n_ant=%d n_win=%d (median %d usable, %d antennas "
                "on the global floor) floor med=%.3e spread=%.1f%% (global %.3e); "
                "sigma med=%.3e (global %.3e)",
                _n_ant,
                _blocks.shape[1],
                int(np.median(_nwin)),
                int(_n_ant - _ok.sum()),
                float(np.median(nfloor)),
                100.0 * float(np.std(nfloor) / max(np.mean(nfloor), 1e-30)),
                noise_mean,
                float(np.median(nsigma)),
                noise_level,
            )
        else:
            self.logger.warning(
                "NOISE_PER_ANT: noise block has shape %s, expected (%d, >1) "
                "-- falling back to the global floor.",
                getattr(_blocks, "shape", None),
                len(fl),
            )

        # FLUENCE OUTLIER REMOVAL
        if len(fl) > 5:
            out_thresh = np.mean(fl) + 8.0 * np.std(fl)
            bad_mask = (fl > out_thresh) | (fl < _FLUENCE_MIN_NOISE_FACTOR * noise_mean)
            if np.any(bad_mask):
                self.logger.info("Removing %d fluence outliers.", int(np.sum(bad_mask)))
                keep = ~bad_mask
                posx, posy, fl, tm, is_sig, snr, sids, nfloor, nsigma = (
                    arr[keep]
                    for arr in (posx, posy, fl, tm, is_sig, snr, sids, nfloor, nsigma)
                )

        # NEIGHBOUR AUGMENTATION
        if np.sum(is_sig) > 0:
            try:
                kx_a = np.sin(global_fit_zen) * np.cos(global_fit_az)
                ky_a = np.sin(global_fit_zen) * np.sin(global_fit_az)
                geom_a = -(1.0 / _C_LIGHT) * (kx_a * posx[is_sig] + ky_a * posy[is_sig])
                t0_aug = float(np.median(tm[is_sig] - geom_a))
                known_set = {
                    (round(float(posx[i]), 1), round(float(posy[i]), 1))
                    for i in range(len(posx))
                }
                station_ids_list = [
                    int(station.get_id()) for station in event.get_stations()
                ]
                aug = pf.augment_neighbor_fluences(
                    event,
                    detector,
                    station_ids_list,
                    posx[is_sig],
                    posy[is_sig],
                    t0_aug,
                    shower_direction,
                    sel_win,
                    known_set,
                )
                if aug["fluences"]:
                    self.logger.info(
                        "Neighbour augmentation: added %d antenna(s).",
                        len(aug["fluences"]),
                    )
                    posx = np.append(posx, aug["posx"])
                    posy = np.append(posy, aug["posy"])
                    fl = np.append(fl, aug["fluences"])
                    tm = np.append(tm, aug["times"])
                    is_sig = np.append(is_sig, np.array(aug["is_signal"], dtype=bool))
                    snr = np.append(snr, aug["snrs"])
                    sids = np.append(
                        sids, np.array(aug["station_ids"], dtype=sids.dtype)
                    )
                    nfloor = np.append(
                        nfloor, np.full(len(aug["fluences"]), noise_mean)
                    )
                    nsigma = np.append(
                        nsigma, np.full(len(aug["fluences"]), noise_level)
                    )
            except Exception as exc:
                self.logger.warning("Neighbour augmentation failed: %s", exc)

        extra_fl_only = {k: np.array([]) for k in ("posx", "posy", "fl", "tm", "snr")}
        extra_fl_only["sids"] = np.array([], dtype=int)

        # STATION-LEVEL NOISE-FLOOR FILTER
        if noise_mean > 0:
            good_stns = self._select_stations(
                fl, sids, noise_mean, station_filter_fallback
            )
            if good_stns is None:
                return event
            bad_stns = set(np.unique(sids)) - good_stns

            if bad_stns:
                (
                    extra_px,
                    extra_py,
                    extra_fl_v,
                    extra_tm_v,
                    extra_snr_v,
                    extra_sids_v,
                ) = [], [], [], [], [], []
                for _sid in sorted(bad_stns):
                    try:
                        _px, _py, _fl_v, _tm_v, _sig_v, _nf_v, _snr_v, _ = (
                            pf.extract_fluence_and_timing_data(
                                event,
                                _sid,
                                detector,
                                shower_direction,
                                sel_win,
                                global_ref_time=None,
                                global_ref_pos=None,
                            )
                        )
                        _snr_arr = np.asarray(_snr_v, dtype=float)
                        _sig_arr = np.asarray(_sig_v, dtype=bool)
                        _high_msk = _sig_arr & (
                            _snr_arr >= _FAR_STATION_FLUENCE_ONLY_SNR
                        )
                        _n_high = int(np.sum(_high_msk))
                        for i in range(len(_fl_v)):
                            if _sig_v[i] and _snr_v[i] >= _FAR_STATION_FLUENCE_ONLY_SNR:
                                n_other = _n_high - (1 if _high_msk[i] else 0)
                                if n_other < _FAR_STATION_MIN_HIGH_SNR_ANTENNAS:
                                    continue
                                extra_px.append(_px[i])
                                extra_py.append(_py[i])
                                extra_fl_v.append(_fl_v[i])
                                extra_tm_v.append(_tm_v[i])
                                extra_snr_v.append(_snr_v[i])
                                extra_sids_v.append(int(_sid))
                    except Exception:
                        pass
                if extra_fl_v:
                    extra_fl_only = {
                        "posx": np.array(extra_px),
                        "posy": np.array(extra_py),
                        "fl": np.array(extra_fl_v),
                        "tm": np.array(extra_tm_v),
                        "snr": np.array(extra_snr_v),
                        "sids": np.array(extra_sids_v, dtype=int),
                    }
                    self.logger.info(
                        "Added %d fluence-only from %d low-signal station(s).",
                        len(extra_fl_v),
                        len(bad_stns),
                    )
                keep = ~np.isin(sids, list(bad_stns))
                posx, posy, fl, tm, is_sig, snr, sids, nfloor, nsigma = (
                    arr[keep]
                    for arr in (posx, posy, fl, tm, is_sig, snr, sids, nfloor, nsigma)
                )

        # TIMING PRUNING & UNCERTAINTY
        kx = np.sin(global_fit_zen) * np.cos(global_fit_az)
        ky = np.sin(global_fit_zen) * np.sin(global_fit_az)
        geom_delays = -(1.0 / _C_LIGHT) * (kx * posx + ky * posy)
        t0_guess = (
            float(np.median(tm[is_sig] - geom_delays[is_sig]))
            if np.any(is_sig)
            else 0.0
        )

        causality_mask = np.abs(tm - t0_guess) <= (_CAUSALITY_CUT_DISTANCE_M / _C_LIGHT)
        use_timing = is_sig & causality_mask
        snr_fl = fl / noise_mean if noise_mean > 0 else np.zeros_like(fl)

        if np.sum(use_timing) > _MIN_ABSOLUTE_NEIGHBORS:
            keep_sub = pf.detect_timing_outliers(
                np.array([posx[use_timing], posy[use_timing]]),
                tm[use_timing],
                sids[use_timing],
                timing_snrs=snr_fl[use_timing],
            )
            idx = np.where(use_timing)[0]
            use_timing[idx[~keep_sub]] = False

        t0_prior = 20.0 * units.ns / units.s
        fit_coeffs = fit_mean_pos = fit_scale = inv_AtA = None
        final_std = _TIMING_UNCERT_THRESHOLD

        if not self._timing_count_passes(use_timing):
            return event

        use_timing, final_std, fit_coeffs, fit_mean_pos, fit_scale, inv_AtA = (
            pf.iterative_timing_pruning(
                shower_direction,
                np.array([posx, posy]),
                tm,
                use_timing,
                station_ids=sids,
                timing_snrs=snr_fl,
            )
        )
        if (
            final_std > _TIMING_UNCERT_THRESHOLD
            or np.sum(use_timing) < _MIN_TIMING_POINTS
        ):
            self.logger.error(
                "Timing quality check failed (std=%.2f ns, n=%d) — aborting.",
                final_std * units.s / units.ns,
                int(np.sum(use_timing)),
            )
            self._reco_gate_fail("timing_quality_check_failed")
            return event

        self.logger.info(
            "Timing pruning: %d points, residual std=%.2f ns.",
            int(np.sum(use_timing)),
            final_std * units.s / units.ns,
        )

        # Refine t0 at LORA core
        core_prior_x = float(lora_core[0])
        core_prior_y = float(lora_core[1])
        if fit_coeffs is not None and fit_mean_pos is not None:
            dx = core_prior_x - fit_mean_pos[0]
            dy = core_prior_y - fit_mean_pos[1]
            xn = dx / fit_scale
            yn = dy / fit_scale
            r2n = xn**2 + yn**2
            t0_guess = float(
                fit_coeffs[0]
                + fit_coeffs[1] * xn
                + fit_coeffs[2] * yn
                + fit_coeffs[3] * r2n
            )
            if inv_AtA is not None:
                p = np.array([1.0, xn, yn, r2n])
                extrap_var = float(p.T @ inv_AtA @ p) * (final_std**2)
                dist_near = float(
                    np.min(
                        np.sqrt((posx - core_prior_x) ** 2 + (posy - core_prior_y) ** 2)
                    )
                )
                dist_unc = (dist_near / _C_LIGHT) * 0.3
                t0_prior = max(
                    float(np.sqrt(extrap_var)),
                    final_std,
                    dist_unc,
                    2.0 * units.ns / units.s,
                )

        # Local timing uncertainties
        timing_idx = np.where(use_timing)[0]
        per_point_timing_std = np.full(len(posx), 5.0 * units.ns / units.s)
        if len(timing_idx) >= 8:
            local_std = pf.get_local_timing_uncertainties(
                np.array([posx[timing_idx], posy[timing_idx]]),
                tm[timing_idx],
                sids[timing_idx],
                shower_direction,
            )
            per_point_timing_std[timing_idx] = local_std
        per_point_timing_std *= _TIMING_SYST_INFLATION
        inv_var_time = 1.0 / per_point_timing_std**2
        inv_var_time[~use_timing] = 0.0

        # Station multiplicity cut (≥ 2 timing points per station)
        stns_with_timing = {
            int(s_) for s_ in np.unique(sids) if np.sum((sids == s_) & use_timing) >= 2
        }
        if stns_with_timing:
            _keep = np.isin(sids, list(stns_with_timing))
            if np.sum(_keep) < len(fl):
                self.logger.info(
                    "Station multiplicity cut: %d points dropped.",
                    int(len(fl) - np.sum(_keep)),
                )
                posx = posx[_keep]
                posy = posy[_keep]
                fl = fl[_keep]
                tm = tm[_keep]
                is_sig = is_sig[_keep]
                snr = snr[_keep]
                sids = sids[_keep]
                use_timing = use_timing[_keep]
                per_point_timing_std = per_point_timing_std[_keep]
                inv_var_time = inv_var_time[_keep]
                nfloor = nfloor[_keep]
                nsigma = nsigma[_keep]

        # Recompute noise from timing-valid stations only
        if stns_with_timing and nf_by_station:
            valid_nf = [
                w
                for sid_k, w in nf_by_station.items()
                if sid_k in stns_with_timing and np.size(w) > 0
            ]
            if valid_nf:
                flat = np.concatenate(
                    [np.asarray(w, dtype=float).reshape(-1) for w in valid_nf])
                if flat.size > 0:
                    _flat_f = flat[np.isfinite(flat)]
                    if _flat_f.size:
                        noise_mean = float(_flat_f.mean())
                        noise_level = (
                            float(_flat_f.std()) if _flat_f.size > 1 else noise_mean
                        )
                    self.logger.info(
                        "Noise recomputed: mean=%.3e, std=%.3e", noise_mean, noise_level
                    )

        # Append fluence-only from bad stations
        n_extra = len(extra_fl_only["fl"])
        if n_extra > 0:
            posx = np.append(posx, extra_fl_only["posx"])
            posy = np.append(posy, extra_fl_only["posy"])
            fl = np.append(fl, extra_fl_only["fl"])
            tm = np.append(tm, extra_fl_only["tm"])
            is_sig = np.append(is_sig, np.zeros(n_extra, dtype=bool))
            snr = np.append(snr, extra_fl_only["snr"])
            sids = np.append(sids, extra_fl_only["sids"])
            use_timing = np.append(use_timing, np.zeros(n_extra, dtype=bool))
            per_point_timing_std = np.append(
                per_point_timing_std,
                np.full(n_extra, 5.0 * units.ns / units.s * _TIMING_SYST_INFLATION),
            )
            inv_var_time = np.append(inv_var_time, np.zeros(n_extra))
            nfloor = np.append(nfloor, np.full(n_extra, noise_mean))
            nsigma = np.append(nsigma, np.full(n_extra, noise_level))

        noise_level = max(noise_level, 1e-3 * noise_mean)
        self.logger.info(
            "Global noise: mean=%.3e, std=%.3e (%d stations)",
            noise_mean,
            noise_level,
            len(np.unique(sids)),
        )

        positions = np.column_stack((posx, posy))
        if np.unique(positions, axis=0).shape[0] != len(positions):
            self._reco_gate_fail("duplicate_antenna_positions")
            return event

        # DRY RUN STOP: every check that can still reject this event before the
        # fit has now passed, so the event would enter the reconstruction proper.

        if dry_run:
            self.logger.status(
                "RECO_GATE: PASS n_points=%d n_timing=%d n_stations=%d window=%.0f",
                fl.size,
                int(np.sum(use_timing)),
                len(np.unique(sids)),
                sel_win,
            )
            return event

        # ATMOSPHERE PATH
        atm_path = None
        if atm_dir:
            try:
                from NuRadioReco.utilities.LOFAR.gdas_tool import (
                    find_or_generate_atmosphere,
                )

                atm_path = find_or_generate_atmosphere(
                    event.get_id(), atm_dir, gdas_cache_dir=s.get("gdas_cache_dir")
                )
            except Exception as exc:
                self.logger.warning(
                    "GDAS atmosphere lookup/generation failed: %s — "
                    "falling back to standard US atmosphere (model 17).",
                    exc,
                )
        elif not s.get("use_fallback_atmosphere", False):
            self.logger.warning(
                "No atmosphere_dir provided to iftReconstructor.begin(). "
                "Pass atmosphere_dir for an event-specific GDAS atmosphere. "
                "Using US standard atmosphere (model 17), with reduced Xmax precision. "
                "Set use_fallback_atmosphere=True to silence this warning."
            )

        # MODEL CONSTRUCTION
        lora_core_x = float(lora_core[0])
        lora_core_y = float(lora_core[1])

        phi_prior = {"a_min": np.pi, "a_max": 3.0 * np.pi}
        theta_prior = {"a_min": 0.0, "a_max": np.radians(60.0)}
        t0_prior_inflated = t0_prior * 5.0

        xmax_lo, xmax_hi = self._xmax_prior_bounds(atm_path, global_fit_zen)

        model_kw = {
            "params_Erad": {"mean": float(np.log(_ERAD_PRIOR_SCALE)), "std": _ERAD_PRIOR_STD},
            "params_phi": phi_prior,
            "params_theta": theta_prior,
            "params_X_max": {"a_min": xmax_lo, "a_max": xmax_hi},
            "params_X_max_timing": {
                "a_min": _XMAX_PRIOR_MIN_GCM2,
                "a_max": _XMAX_PRIOR_MAX_GCM2,
            },
            "params_X": {"mean": 0.0, "std": core_prior_std},
            "params_Y": {"mean": 0.0, "std": core_prior_std},
            "params_core_timing_offset": {
                "mean": 0.0, "std": _CORE_TIMING_OFFSET_STD_M
            },
            "noise_floor_per_antenna": nfloor,
            "atmosphere_path": atm_path,
            "params_t0": {"mean": t0_guess, "std": t0_prior_inflated},
            "timing_std_s": _TIMING_UNCERTAINTY_S,
            "enable_syst_cf": fluence_cf,
            "enable_timing_cf": timing_cf,
            "enable_ldf_energy_scale_uncertainty": True,
            "ldf_energy_scale_fractional_std": _LDF_ENERGY_SCALE_STD,
            "enable_fluence_dxmax_precision": True,
            "fluence_dxmax_precision_gpcm2": _FLUENCE_DXMAX_PRECISION_GPCM2,
            "enable_timing_dxmax_precision": True,
            "timing_dxmax_precision_gpcm2": _TIMING_DXMAX_PRECISION_GPCM2,
            "syst_mult_min": SYST_MULT_MIN,
            "syst_mult_max": SYST_MULT_MAX,
        }

        b_field = hp.get_magnetic_field_vector("lofar")
        _fl_denom = np.maximum(
            nsigma**2 + (fl * _FLUENCE_RELATIVE_SYSTEMATIC_ERROR) ** 2, 1e-60
        )
        inv_fl = 1.0 / _fl_denom
        noise_cov_inv = jnp.stack([jnp.array(inv_fl), jnp.array(inv_var_time)])

        model = footprintModel(posx, posy, b_field, **model_kw)
        lh = jft.Gaussian(
            data=jnp.stack([jnp.array(fl), jnp.array(tm)]),
            noise_cov_inv=noise_cov_inv,
        ).amend(model)

        # INIT VALUES FROM LORA/RADIO PRIOR
        key = random.PRNGKey(seed)
        key, k_init, k_opt = random.split(key, 3)

        init_state = lh.init(k_init)
        init_values = dict(
            init_state.tree if hasattr(init_state, "tree") else init_state
        )

        az_wrapped = np.pi + (global_fit_az - np.pi) % (2.0 * np.pi)
        init_values["phi"] = pf.uniform_prior_latent(az_wrapped, phi_prior)
        init_values["theta"] = pf.uniform_prior_latent(global_fit_zen, theta_prior)
        init_values["X_max"] = jnp.array([0.0])
        init_values["X"] = jnp.array([lora_core_x / core_prior_std])
        init_values["Y"] = jnp.array([lora_core_y / core_prior_std])
        init_values["t0"] = jnp.array([0.0])
        pos_init = jft.Vector(init_values)

        # OptimizeVI LOOP WITH CONVERGENCE CHECK
        n_dof_chisq = max(
            int(np.asarray(jnp.stack([jnp.array(fl), jnp.array(tm)])).size), 1
        )
        delta = 1e-7

        def n_samples_sched(i_iter):
            """Halve the sample count over the first half of the schedule."""
            return n_samples // 2 if i_iter < n_vi // 2 else n_samples

        def sample_mode_sched(i_iter):
            """Start with linear sampling, then resample nonlinearly."""
            return _SAMPLING_MODE if i_iter < n_linear_iters else _RESAMPLING_MODE

        opt_vi = jft.OptimizeVI(lh, n_total_iterations=n_vi, kl_map=vmap)
        opt_state = opt_vi.init_state(
            k_opt,
            n_samples=n_samples_sched,
            draw_linear_kwargs=dict(
                cg=jft.static_cg,
                cg_name="SL",
                cg_kwargs=dict(
                    absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=800
                ),
            ),
            nonlinearly_update_kwargs=dict(
                minimize_kwargs=dict(
                    name="SN", xtol=delta, cg_kwargs=dict(name=None), maxiter=25
                ),
            ),
            kl_kwargs=dict(
                minimize_kwargs=dict(
                    name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=40
                ),
            ),
            sample_mode=sample_mode_sched,
        )
        samples_liquid = jft.Samples(pos=pos_init, samples=None, keys=None)

        prev_redchisq = None
        n_stable = 0
        is_converged = False
        for i_iter in range(n_vi):
            samples_liquid, opt_state = opt_vi.update(samples_liquid, opt_state)
            try:
                self.logger.info(opt_vi.get_status_message(samples_liquid, opt_state))
            except Exception:
                pass

            chisq = np.array([2.0 * float(lh(s_)) for s_ in samples_liquid])
            red_chisq = float(chisq.mean()) / n_dof_chisq
            if prev_redchisq is not None:
                # A stable chi2 counts towards convergence in every iteration,
                # including the linearly sampled ones at the start.
                d_chisq = abs(red_chisq - prev_redchisq)
                n_stable = n_stable + 1 if d_chisq < _CONV_TOL_REDCHISQ else 0
                self.logger.info(
                    "VI iter %d/%d: chi2/dof=%.3f (delta=%.4f, stable=%d/%d)",
                    i_iter + 1,
                    n_vi,
                    red_chisq,
                    d_chisq,
                    n_stable,
                    _CONV_PATIENCE,
                )
                if n_stable >= _CONV_PATIENCE and (i_iter + 1) >= _CONV_MIN_ITERS:
                    self.logger.info("VI converged after %d iterations.", i_iter + 1)
                    is_converged = True
                    break
            else:
                self.logger.info(
                    "VI iter %d/%d: chi2/dof=%.3f", i_iter + 1, n_vi, red_chisq
                )
            prev_redchisq = red_chisq

            # Monitor posterior width
            if (i_iter + 1) == 2:
                try:
                    xmax_now = [
                        float(np.asarray(model.X_max_combined(s_)).squeeze())
                        for s_ in samples_liquid
                    ]
                    xmax_eff_now = [
                        float(np.asarray(model.X_max_effective(s_)).squeeze())
                        for s_ in samples_liquid
                    ]
                except Exception as exc:
                    self.logger.debug("Iteration-two Xmax diagnostic skipped: %s", exc)
                    xmax_now, xmax_eff_now = [], []
                if len(xmax_now) >= 2:
                    xmax_std_now = float(
                        jft.mean_and_std(xmax_now, correct_bias=True)[1]
                    )
                    # Record both the reported and the clipped effective width.
                    eff_std_now = (
                        float(jft.mean_and_std(xmax_eff_now, correct_bias=True)[1])
                        if len(xmax_eff_now) >= 2
                        else float("nan")
                    )
                    self.logger.status(
                        "XMAX_ITER: iter=%d std_combined=%.1f std_eff=%.1f g/cm2",
                        i_iter + 1,
                        xmax_std_now,
                        eff_std_now,
                    )

        n_vi_iters_run = i_iter + 1

        samples_list = list(samples_liquid)
        # Two samples minimum: the posterior summary uses mean_and_std with
        # correct_bias=True, whose 1/(n-1) divides by zero on a single sample.
        filtered_samples = self._retained_samples(
            samples_list, list(range(len(samples_list)))
        )
        if filtered_samples is None:
            return event

        # POSTERIOR SUMMARY
        def _s(v):
            return float(np.asarray(v).squeeze())

        zen_s = [_s(np.rad2deg(model.zen_and_az(s_)[0])) for s_ in filtered_samples]
        # Wrap model phi (prior lives in [π, 3π]) to [0, 2π] before averaging
        az_s = [_s(model.zen_and_az(s_)[1] % (2 * np.pi)) for s_ in filtered_samples]
        cx_s = [_s(model.core(s_)[0]) for s_ in filtered_samples]
        cy_s = [_s(model.core(s_)[1]) for s_ in filtered_samples]
        xmax_s = [_s(model.X_max_combined(s_)) for s_ in filtered_samples]
        # Diagnostics for the X_max <-> fluence-offset degeneracy: the clipped
        # Xmax the LDF is actually evaluated at, and the offset separating the two.
        xmax_eff_s = [_s(model.X_max_effective(s_)) for s_ in filtered_samples]
        dxmax_off_s = [_s(model.fluence_dxmax_offset(s_)) for s_ in filtered_samples]
        ct_off_s = [
            np.asarray(model.core_timing_offset(s_)).reshape(-1)
            for s_ in filtered_samples
        ]
        ecr_s = [_s(self._calculate_ecr(model, s_)) for s_ in filtered_samples]
        erad_s = [
            _s(model.Erad(s_) / model.get_energy_correction_factor(s_))
            for s_ in filtered_samples
        ]

        zen_mean, zen_std = (
            float(v) for v in jft.mean_and_std(zen_s, correct_bias=True)
        )
        cx_mean, cx_std = (float(v) for v in jft.mean_and_std(cx_s, correct_bias=True))
        cy_mean, cy_std = (float(v) for v in jft.mean_and_std(cy_s, correct_bias=True))
        xmax_mean, xmax_std = (
            float(v) for v in jft.mean_and_std(xmax_s, correct_bias=True)
        )
        xmax_eff_mean, xmax_eff_std = (
            float(v) for v in jft.mean_and_std(xmax_eff_s, correct_bias=True)
        )
        dxmax_off_mean, dxmax_off_std = (
            float(v) for v in jft.mean_and_std(dxmax_off_s, correct_bias=True)
        )
        ct_off_mean = np.mean(np.asarray(ct_off_s), axis=0)
        ecr_mean, ecr_std = (
            float(v) for v in jft.mean_and_std(ecr_s, correct_bias=True)
        )
        erad_mean, erad_std = (
            float(v) for v in jft.mean_and_std(erad_s, correct_bias=True)
        )
        # Azimuth needs circular mean/std — arithmetic mean fails near the 0/2π wrap
        _sin_mean = np.mean(np.sin(az_s))
        _cos_mean = np.mean(np.cos(az_s))
        _R = float(np.sqrt(_sin_mean**2 + _cos_mean**2))
        az_mean_rad = float(np.arctan2(_sin_mean, _cos_mean)) % (2 * np.pi)
        az_mean = float(np.rad2deg(az_mean_rad))
        az_std = float(np.rad2deg(np.sqrt(-2.0 * np.log(max(_R, 1e-10)))))

        self.logger.info(
            "IFT result: zen=%.2f±%.2f°, az=%.2f±%.2f°, "
            "core=(%.1f±%.1f, %.1f±%.1f) m, Xmax=%.0f±%.0f g/cm², "
            "E_CR=%.3e±%.3e eV, E_rad=%.3e±%.3e eV",
            zen_mean,
            zen_std,
            az_mean,
            az_std,
            cx_mean,
            cx_std,
            cy_mean,
            cy_std,
            xmax_mean,
            xmax_std,
            ecr_mean,
            ecr_std,
            erad_mean,
            erad_std,
        )

        # Grep-able: how far the degenerate X_max/offset pair drifted, and whether
        # the posterior ended up against either end of the (spline-restricted) prior.
        self.logger.status(
            "XMAX_DIAG: xmax=%.0f±%.0f eff=%.0f offset=%+.1f prior=[%.0f,%.0f] rail=%s",
            xmax_mean,
            xmax_std,
            xmax_eff_mean,
            dxmax_off_mean,
            xmax_lo,
            xmax_hi,
            "LOW"
            if xmax_mean - xmax_lo < 30.0
            else ("HIGH" if xmax_hi - xmax_mean < 30.0 else "no"),
        )
        self.logger.status(
            "CORE_DIAG: fluence core=(%.0f, %.0f) timing offset=(%+.0f, %+.0f) "
            "= %.0f m (prior sigma %.0f)",
            cx_mean,
            cy_mean,
            ct_off_mean[0],
            ct_off_mean[1],
            float(np.hypot(*ct_off_mean)),
            _CORE_TIMING_OFFSET_STD_M,
        )

        # QUALITY CUTS
        # Applied to the finished fit. Report whether event passes a
        # reduced chi² < 2, core uncertainty < 50m and Xmax not hitting prior bounds
        #
        # Verdict is stored in the NPZ and printed; 
        # in the future a showerParameter could be added so the verdict
        # is also stored in the .nur

        xmax_all_s = [_s(model.X_max_combined(s_)) for s_ in samples_list]
        quality = qualityCuts.evaluate(
            red_chisq=red_chisq,
            core_x_std=cx_std,
            core_y_std=cy_std,
            xmax_samples=xmax_all_s,
            xmax_prior_min=xmax_lo,
            xmax_prior_max=xmax_hi,
        )
        for _line in qualityCuts.format_report(quality, event.get_id()).split("\n"):
            self.logger.status(_line)

        # WRITE TO RADIO SHOWER
        reco_zen = np.radians(zen_mean)
        reco_az = az_mean_rad
        reco_core = np.array([cx_mean * units.m, cy_mean * units.m, 7.6 * units.m])
        reco_energy = ecr_mean * units.eV
        reco_xmax = xmax_mean * units.g / units.cm**2
        reco_erad = erad_mean * units.eV

        shower = self._get_or_create_radio_shower(event)
        shower.set_parameter(showerParameters.zenith, reco_zen)
        shower.set_parameter(showerParameters.azimuth, reco_az)
        shower.set_parameter(showerParameters.core, reco_core)
        shower.set_parameter(showerParameters.shower_maximum, reco_xmax)
        shower.set_parameter(showerParameters.energy, reco_energy)
        shower.set_parameter(showerParameters.radiation_energy, reco_erad)
        shower.set_parameter(showerParameters.magnetic_field_vector, b_field)
        shower.set_parameter(showerParameters.observation_level, 760 * units.cm)

        for station in event.get_stations():
            station.set_parameter(stationParameters.cr_zenith, reco_zen)
            station.set_parameter(stationParameters.cr_azimuth, reco_az)
            station.set_parameter(stationParameters.cr_energy, reco_energy)
            station.set_parameter(stationParameters.cr_xmax, reco_xmax)

        dist_m = self._distance_to_shower_maximum(
            xmax_mean, reco_zen, gdas_file=atm_path
        )
        shower.set_parameter(
            showerParameters.distance_shower_maximum_geometric, dist_m * units.m
        )
        self.logger.info("Distance to shower maximum: %.1f m", dist_m)

        self._set_efield_parameters(event, detector, reco_zen, reco_az)

        self.logger.info(
            "IFT reconstruction (event %s):\n"
            "  Zenith  : %.2f +/- %.2f deg\n"
            "  Azimuth : %.2f +/- %.2f deg\n"
            "  Core X  : %.1f +/- %.1f m\n"
            "  Core Y  : %.1f +/- %.1f m\n"
            "  Xmax    : %.1f +/- %.1f g/cm2\n"
            "  E_CR    : %.3e +/- %.3e eV\n"
            "  E_rad   : %.3e +/- %.3e eV",
            event.get_id(),
            zen_mean,
            zen_std,
            az_mean,
            az_std,
            cx_mean,
            cx_std,
            cy_mean,
            cy_std,
            xmax_mean,
            xmax_std,
            ecr_mean,
            ecr_std,
            erad_mean,
            erad_std,
        )

        # RECONSTRUCTION PLOT
        _out_dir = s.get("output_directory") or os.getcwd()
        os.makedirs(_out_dir, exist_ok=True)

        try:
            cf_stats = None
            if model.enable_syst_cf or model.enable_timing_cf:
                _syst = [np.asarray(model.syst_cf(s_)) for s_ in filtered_samples]
                _tmcf = [
                    np.clip(
                        np.asarray(model.timing_cf_op_2(s_)) * units.s / units.ns,
                        -15.0,
                        15.0,
                    )
                    for s_ in filtered_samples
                ]
                cf_stats = {
                    "syst_mean": np.mean([np.exp(f) for f in _syst], axis=0),
                    "syst_std": np.std([np.exp(f) for f in _syst], axis=0),
                    "timing_mean": np.mean(_tmcf, axis=0),
                    "timing_std": np.std(_tmcf, axis=0),
                }

            from NuRadioReco.utilities.LOFAR import iftOutput

            all_data_plot = {
                "pos_x": posx,
                "pos_y": posy,
                "fluences": fl,
                "times": tm,
                "is_signal": use_timing,
            }
            ref_params_plot = {
                "zenith": lora.get("zenith", global_fit_zen),
                "azimuth": lora.get("azimuth", global_fit_az),
                "core": np.asarray(lora_core[:2], dtype=float),
                "energy": float(lora.get("energy", 0.0)) if lora is not None else 0.0,
            }
            # CoREAS truth, for a simulated event only. Its presence is what
            # distinguishes a simulation from real data here, so no branch on
            # "is this a sim" is needed and the data plots are untouched.
            mc_truth_plot = None
            try:
                _sim = event.get_first_sim_shower()
            except Exception:
                _sim = None
            if _sim is not None:
                def _p(param, scale=1.0):
                    try:
                        return float(_sim.get_parameter(param)) / scale
                    except Exception:
                        return None

                _core = None
                try:
                    _core = np.asarray(
                        _sim.get_parameter(showerParameters.core), dtype=float)
                except Exception:
                    pass
                mc_truth_plot = {
                    "xmax_gpcm2": _p(showerParameters.shower_maximum,
                                     units.g / units.cm**2),
                    "energy_ev": _p(showerParameters.energy, units.eV),
                    "zenith_rad": _p(showerParameters.zenith),
                    "azimuth_rad": _p(showerParameters.azimuth),
                    "core_x_m": None if _core is None else float(_core[0] / units.m),
                    "core_y_m": None if _core is None else float(_core[1] / units.m),
                }

            iftOutput.generate_reco_plot(
                filtered_samples,
                ecr_s,
                all_data_plot,
                _out_dir,
                event.get_id(),
                ref_params=ref_params_plot,
                signal_response=model,
                noise_mean=noise_mean,
                model_kw=model_kw,
                noise_level=noise_level,
                cf_stats=cf_stats,
                b_field=b_field,
                direction_reference=(global_fit_zen, global_fit_az),
                direction_reference_label="Radio plane wave",
                rejection_reason=None if quality["passed"] else quality["reason"],
                mc_truth=mc_truth_plot,
            )
        except Exception as _plot_exc:
            self.logger.warning(
                "Reconstruction plot failed: %s", _plot_exc, exc_info=True
            )

        if s.get("export_posterior_samples", False):
            out_dir = s.get("output_directory") or os.getcwd()
            os.makedirs(out_dir, exist_ok=True)
            posterior_data = {
                "posterior_sample_indices": np.arange(len(samples_list), dtype=int),
            }
            try:
                # Samples are JAX pytrees. Store every leaf stacked over samples,
                # together with its path, so every sample can be inspected
                # without pickle/object-array loading.
                from jax import tree_util as jtu

                stacked_samples = jtu.tree_map(
                    lambda *parts: np.stack([np.asarray(part) for part in parts]),
                    *samples_list,
                )
                path_leaves, tree_def = jtu.tree_flatten_with_path(stacked_samples)
                posterior_data["posterior_sample_tree"] = np.asarray(str(tree_def))
                posterior_data["posterior_leaf_paths"] = np.asarray(
                    [str(path) for path, _ in path_leaves]
                )
                posterior_data.update(
                    {
                        f"posterior_leaf_{i}": np.asarray(leaf)
                        for i, (_, leaf) in enumerate(path_leaves)
                    }
                )
            except Exception as exc:
                self.logger.warning(
                    "Could not serialize raw posterior samples: %s", exc
                )
            np.savez(
                os.path.join(out_dir, f"{event.get_id()}.npz"),
                event_id=event.get_id(),
                zenith=reco_zen,
                azimuth=reco_az,
                core=reco_core,
                energy=reco_energy,
                xmax=reco_xmax,
                radiation_energy=reco_erad,
                zen_mean=zen_mean,
                zen_std=zen_std,
                az_mean=az_mean,
                az_std=az_std,
                core_x_mean=cx_mean,
                core_x_std=cx_std,
                core_y_mean=cy_mean,
                core_y_std=cy_std,
                xmax_mean=xmax_mean,
                xmax_std=xmax_std,
                xmax_samples=np.asarray(xmax_s, dtype=float),
                xmax_effective_mean=xmax_eff_mean,
                xmax_effective_std=xmax_eff_std,
                dxmax_offset_mean=dxmax_off_mean,
                dxmax_offset_std=dxmax_off_std,
                xmax_prior_min=xmax_lo,
                xmax_prior_max=xmax_hi,
                core_timing_offset_mean=ct_off_mean,
                core_timing_offset_std_m=_CORE_TIMING_OFFSET_STD_M,
                ecr_mean=ecr_mean,
                ecr_std=ecr_std,
                erad_mean=erad_mean,
                erad_std=erad_std,
                red_chisq=red_chisq,
                n_dof=n_dof_chisq,
                quality_passed=bool(quality["passed"]),
                quality_rejection_reason=quality["reason"],
                quality_core_unc_m=quality["core_unc_m"],
                quality_xmax_rail_low=quality["xmax_rail_low"],
                quality_xmax_rail_high=quality["xmax_rail_high"],
                quality_cut_red_chisq_max=quality["cuts"]["red_chisq_max"],
                quality_cut_core_unc_max_m=quality["cuts"]["core_unc_max_m"],
                quality_cut_xmax_rail_edge=quality["cuts"]["xmax_rail_edge"],
                quality_cut_xmax_rail_max=quality["cuts"]["xmax_rail_max"],
                is_converged=is_converged,
                n_vi_iters=n_vi_iters_run,
                n_vi_max=n_vi,
                n_samples_total=len(samples_list),
                n_antennas=len(fl),
                n_timing=int(np.sum(use_timing)),
                used_max_signal_fallback=used_max_signal_fallback,
                noise_mean=noise_mean,
                noise_floor_per_antenna=nfloor,
                noise_std_per_antenna=nsigma,
                sel_win=sel_win,
                fluences=fl,
                pos_x=posx,
                pos_y=posy,
                times=tm,
                is_signal=is_sig,
                use_timing=use_timing,
                sids=sids,
                **posterior_data,
            )

        return event

    @register_run()
    def run(self, event, detector):
        return self._reconstruct(event, detector)

    def end(self):
        pass
