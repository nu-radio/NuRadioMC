"""
LOFAR IFT reconstruction module for NuRadioReco.

.. moduleauthor:: Karen Terveer <karen.terveer@fau.de>
"""
import logging
import os

import numpy as np
import radiotools.helper as hp

from NuRadioReco.framework.parameters import showerParameters, stationParameters, electricFieldParameters
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.LOFAR.utilities import iftDataHelpers
from NuRadioReco.utilities import units

# Module-level constants used by the IFT reconstructor
_C_LIGHT = 299792458.0
_FLUENCE_RELATIVE_SYSTEMATIC_ERROR = 0.10
_LDF_ENERGY_SCALE_STD = 0.04
_FAR_STATION_FLUENCE_ONLY_SNR = 12.0
_FAR_STATION_HIGH_SNR_THRESHOLD = 10.0
_FAR_STATION_MIN_HIGH_SNR_ANTENNAS = 12
_FLUENCE_MIN_NOISE_FACTOR = 0.05
_STATION_SNR_THRESHOLD = 3.0
_STATION_MIN_ANTENNAS = 10
_MIN_STATIONS_REQUIRED = 1
_FALLBACK_ANTENNA_THRESHOLD = 48
_FALLBACK_SNR_THRESHOLD = 6.0
_CAUSALITY_CUT_DISTANCE_M = 600.0
_TIMING_UNCERTAINTY_S = 1.5e-9
_TIMING_SYST_INFLATION = 1.5
_TIMING_UNCERT_THRESHOLD = 25e-9
_MIN_TIMING_POINTS = 12
_MIN_GOOD_TIMING_POINTS_RECO = 0
_MIN_ABSOLUTE_NEIGHBORS = 8
_DEFAULT_CORE_PRIOR_STD_M = 120.0
_FLUENCE_DXMAX_PRECISION_GPCM2 = 16.3
_TIMING_DXMAX_PRECISION_GPCM2 = 30.0
_DEFAULT_N_VI_ITERATIONS = 8
_DEFAULT_N_SAMPLES = 60
_DEFAULT_SEED = 27
_SAMPLING_MODE = "linear_sample"
_RESAMPLING_MODE = "nonlinear_resample"
_CONV_PATIENCE = 3
_CONV_MIN_ITERS = 5
_CONV_TOL_REDCHISQ = 0.01
_TRIGGER_SNR_BUFFER_FRAC = 0.05
_BROAD_SCAN_RANGE_DEG = 10.0
_BROAD_STEP_DEG = 1.0
_BROAD_SMOOTHING_NS = 5.0
_RECO_STAGES = [{"window": 300.0}, {"window": 200.0}, {"window": 100.0}]

# E_rad → E_CR calibration coefficients (zenith-corrected formula, N=3991 events).
# Formula: E_rad [MeV] = A * (sin²α + a²) * (B/B_ref)² * (E_CR/E_scale)^B * (1 + c·cos²θ)
# Source: NuRadioReco/utilities/data/LOFAR/ift/erad_calibration.npz, key prefix 'zen_'
_ERAD_CAL_A      = 16.72578725   # zen_A
_ERAD_CAL_B      =  2.02195864   # zen_B
_ERAD_CAL_a_CE   =  0.15889307   # zen_a_CE  (charge-excess contribution)
_ERAD_CAL_c_zen  = -0.62162961   # zen_c_zen (zenith-angle correction)
_ERAD_CAL_B_ref  =  0.24         # reference B-field magnitude [Gauss]
_ERAD_CAL_E_scale = 1e18         # energy scale [eV]


class iftReconstructor:
    """
    LOFAR IFT reconstruction module.

    With ``run_nifty=False`` (the default) this runs a fast smoke reconstruction
    that extracts fluence observables and writes them to the event RadioShower.

    With ``run_nifty=True`` the full JAX/NIFTy variational-inference pipeline is
    run: two-pass timing search, RECO_STAGES window selection, ``OptimizeVI`` with
    convergence check, trigger-based posterior filtering, and final shower-parameter
    export.
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
            debug_plots=False,
            debug_plot_dir=None,
            logger_level=logging.NOTSET,
            run_nifty=False,
            scan_range_deg=5.0,
            step_deg=0.5,
            sigma_ns=5.0,
            core_prior_std_m=_DEFAULT_CORE_PRIOR_STD_M,
            skip_lba_inner=False,
    ):
        """
        Configure the reconstruction.

        ``run_nifty`` is False by default; set to True for the full VI pipeline.
        In smoke mode the correlated fields default to True to match the production
        pipeline; set them False for fast CI smoke tests.
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
            "debug_plots": debug_plots,
            "debug_plot_dir": debug_plot_dir,
            "run_nifty": run_nifty,
            "scan_range_deg": scan_range_deg,
            "step_deg": step_deg,
            "sigma_ns": sigma_ns,
            "core_prior_std_m": core_prior_std_m,
            "skip_lba_inner": skip_lba_inner,
        }
        self.logger.setLevel(logger_level)

    @staticmethod
    def _simple_energy_fluence(electric_field_trace, times):
        return iftDataHelpers.get_electric_field_energy_fluence(electric_field_trace, times)

    @staticmethod
    def _get_lora_prior(event):
        try:
            shower = event.get_hybrid_information().get_hybrid_shower("LORA")
        except ValueError:
            return None

        required = (showerParameters.zenith, showerParameters.azimuth, showerParameters.core)
        if not all(shower.has_parameter(p) for p in required):
            return None

        prior = {
            "zenith": shower.get_parameter(showerParameters.zenith),
            "azimuth": shower.get_parameter(showerParameters.azimuth),
            "core": np.asarray(shower.get_parameter(showerParameters.core), dtype=float),
        }
        if shower.has_parameter(showerParameters.energy):
            prior["energy"] = shower.get_parameter(showerParameters.energy)
        return prior

    @staticmethod
    def _get_radio_direction(event):
        zeniths, azimuths = [], []
        for station in event.get_stations():
            if station.has_parameter(stationParameters.cr_zenith) and \
                    station.has_parameter(stationParameters.cr_azimuth):
                zeniths.append(station.get_parameter(stationParameters.cr_zenith))
                azimuths.append(station.get_parameter(stationParameters.cr_azimuth))
            elif station.has_parameter(stationParameters.zenith) and \
                    station.has_parameter(stationParameters.azimuth):
                zeniths.append(station.get_parameter(stationParameters.zenith))
                azimuths.append(station.get_parameter(stationParameters.azimuth))

        if not zeniths:
            return None

        vectors = [
            hp.spherical_to_cartesian(zen, az)
            for zen, az in zip(zeniths, azimuths)
            if np.isfinite(zen) and np.isfinite(az)
        ]
        if not vectors:
            return None
        v = np.mean(vectors, axis=0)
        norm = np.linalg.norm(v)
        if norm <= 0:
            return None
        v /= norm
        zenith = np.arccos(np.clip(v[2], -1.0, 1.0))
        azimuth = np.arctan2(v[1], v[0]) % (2 * np.pi)
        return zenith, azimuth

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
        from NuRadioReco.modules.LOFAR.utilities.atmosphere import Atmosphere
        import jax.numpy as jnp
        if gdas_file is not None:
            atm = Atmosphere(gdas_file=gdas_file, observation_level=7.6)
        else:
            atm = Atmosphere(model=17, observation_level=7.6)
        return float(atm.get_geometric_distance_grammage(
            jnp.array(float(xmax_gcm2)), jnp.array(float(zenith_rad))
        ))

    def _set_efield_parameters(self, event, detector, zenith_rad, azimuth_rad):
        """Set per-E-field parameters from the reconstructed shower direction."""
        from scipy.signal import hilbert as scipy_hilbert

        for station in event.get_stations():
            sid = station.get_id()
            for efield in station.get_electric_fields():
                if not efield.get_channel_ids():
                    continue
                ef_trace = efield.get_trace()
                ef_times = efield.get_times()

                efield.set_parameter(
                    electricFieldParameters.signal_energy_fluence,
                    iftDataHelpers.get_electric_field_energy_fluence(ef_trace, ef_times),
                )
                efield.set_parameter(electricFieldParameters.zenith, zenith_rad)
                efield.set_parameter(electricFieldParameters.azimuth, azimuth_rad)

                envelope = np.abs(scipy_hilbert(np.linalg.norm(ef_trace, axis=0)))
                efield.set_parameter(
                    electricFieldParameters.signal_time,
                    ef_times[int(np.argmax(envelope))],
                )

                pol_angle = iftDataHelpers.compute_polarization_angle(
                    efield, station, float(zenith_rad), float(azimuth_rad)
                )
                if pol_angle is not None:
                    efield.set_parameter(electricFieldParameters.polarization_angle, pol_angle)

    def _collect_observables(self, event, detector):
        positions, fluences, station_ids = [], [], []
        for station in event.get_stations():
            sid = station.get_id()
            stn_pos = detector.get_absolute_position(sid)
            for efield in station.get_electric_fields():
                ch_ids = efield.get_channel_ids()
                if not ch_ids:
                    continue
                pos = stn_pos + efield.get_position()
                fluence = np.sum(self._simple_energy_fluence(efield.get_trace(), efield.get_times()))
                if not np.isfinite(fluence):
                    continue
                positions.append(pos)
                fluences.append(fluence)
                station_ids.append(sid)
        return np.asarray(positions), np.asarray(fluences), np.asarray(station_ids, dtype=int)

    def _plot_reconstruction_debug(self, event, positions, fluences, station_ids,
                                    core, zenith, azimuth):
        if not self.__settings.get("debug_plots", False):
            return

        import matplotlib.pyplot as plt

        output_dir = self.__settings.get("debug_plot_dir") or \
            self.__settings.get("output_directory") or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        finite = np.isfinite(fluences) & (fluences > 0)
        fig, axd = plt.subplot_mosaic(
            [["footprint", "hist"], ["station", "summary"]],
            figsize=(11, 8), constrained_layout=True,
        )

        if np.any(finite):
            sizes = 20 + 160 * np.sqrt(fluences[finite] / np.max(fluences[finite]))
            sc = axd["footprint"].scatter(
                positions[finite, 0] / units.m, positions[finite, 1] / units.m,
                c=fluences[finite], s=sizes, cmap="viridis", alpha=0.85,
            )
            fig.colorbar(sc, ax=axd["footprint"], label="Fluence")
            axd["hist"].hist(np.log10(fluences[finite]), bins=30)
            axd["hist"].set_xlabel("log10 fluence")
            axd["hist"].set_ylabel("Antenna count")
        else:
            for k in ("footprint", "hist"):
                axd[k].text(0.5, 0.5, "No finite fluences", ha="center", va="center")

        axd["footprint"].scatter(core[0] / units.m, core[1] / units.m,
                                  marker="x", s=100, color="tab:red", label="reco core")
        axd["footprint"].set_aspect("equal", adjustable="box")
        axd["footprint"].set_xlabel("East [m]")
        axd["footprint"].set_ylabel("North [m]")
        axd["footprint"].legend(loc="best")

        if len(station_ids):
            unique_ids, counts = np.unique(station_ids, return_counts=True)
            axd["station"].bar([f"CS{sid:03d}" for sid in unique_ids], counts)
            axd["station"].set_ylabel("E-fields")
            axd["station"].tick_params(axis="x", rotation=45)
        else:
            axd["station"].text(0.5, 0.5, "No stations", ha="center", va="center")

        axd["summary"].axis("off")
        summary = (
            f"Event {event.get_id()}\n"
            f"zenith: {zenith / units.deg:.2f} deg\n"
            f"azimuth: {azimuth / units.deg:.2f} deg\n"
            f"core: ({core[0] / units.m:.1f}, {core[1] / units.m:.1f}) m\n"
            f"finite fluences: {np.sum(finite)} / {len(fluences)}\n"
            f"IFT iterations: {self.__settings.get('n_iterations')}\n"
            f"IFT samples: {self.__settings.get('n_samples')}\n"
            f"fluence CF: {self.__settings.get('enable_fluence_correlated_field')}\n"
            f"timing CF: {self.__settings.get('enable_timing_correlated_field')}"
        )
        axd["summary"].text(0.02, 0.98, summary, va="top", family="monospace")
        fig.suptitle("LOFAR IFT reconstruction debug summary")
        fig.savefig(
            os.path.join(output_dir, f"ift_reconstruction_summary_{event.get_id()}.png"),
            dpi=160, bbox_inches="tight",
        )
        plt.close(fig)

    def _run_nifty(self, event, detector):
        # --- Optional dependency imports ---
        try:
            import jax.numpy as jnp
            from jax import random, vmap
            import nifty.re as jft
            from NuRadioReco.modules.LOFAR.utilities.iftModel import (
                footprintModel, SYST_MULT_MIN, SYST_MULT_MAX,
            )
            import NuRadioReco.modules.LOFAR.utilities.jaxHelpers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LOFAR full IFT reconstruction requires optional packages 'jax' and 'nifty8'. "
                "Install them or run with run_nifty=False for the fast smoke reconstruction."
            ) from exc

        def _calculate_ecr(model, x):
            """E_rad → E_CR using the zenith-corrected calibration (module-level constants)."""
            zenith  = model.zen_and_az(x)[0]
            azimuth = model.zen_and_az(x)[1]
            Erad_eV = model.Erad(x) / model.get_energy_correction_factor(x)
            B_vect  = jnp.array(model.magnetic_field_vector).squeeze()
            B_mag   = jnp.linalg.norm(B_vect)
            s_vec   = jnp.stack([
                jnp.sin(zenith) * jnp.cos(azimuth),
                jnp.sin(zenith) * jnp.sin(azimuth),
                jnp.cos(zenith),
            ]).squeeze()
            sin_alpha = jnp.clip(jnp.linalg.norm(jnp.cross(s_vec, B_vect / B_mag)), 0.05, 1.0)
            bc_term   = (B_mag / _ERAD_CAL_B_ref) ** 2
            ce_term   = sin_alpha ** 2 + _ERAD_CAL_a_CE ** 2
            zen_term  = 1.0 + _ERAD_CAL_c_zen * jnp.cos(zenith) ** 2
            denom     = 1e6 * _ERAD_CAL_A * bc_term * ce_term * zen_term
            return _ERAD_CAL_E_scale * jnp.power(Erad_eV / denom, 1.0 / _ERAD_CAL_B)

        pf = iftDataHelpers
        s = self.__settings
        n_vi         = int(s.get("n_iterations",                  _DEFAULT_N_VI_ITERATIONS))
        n_samples    = int(s.get("n_samples",                     _DEFAULT_N_SAMPLES))
        seed         = int(s.get("random_seed",                   _DEFAULT_SEED))
        fluence_cf   = bool(s.get("enable_fluence_correlated_field", True))
        timing_cf    = bool(s.get("enable_timing_correlated_field",  True))
        atm_dir      = s.get("atmosphere_dir")
        skip_lba     = bool(s.get("skip_lba_inner",               False))
        scan_range   = float(s.get("scan_range_deg",              5.0))
        step         = float(s.get("step_deg",                    0.5))
        sigma        = float(s.get("sigma_ns",                    5.0))
        core_prior_std = float(s.get("core_prior_std_m",          _DEFAULT_CORE_PRIOR_STD_M))

        # ---- Priors from LORA or radio direction ----
        lora      = self._get_lora_prior(event)
        radio_dir = self._get_radio_direction(event)
        if lora is not None:
            lora_zen  = float(lora["zenith"])
            lora_az   = float(lora["azimuth"])
            lora_core = lora["core"].copy()
        elif radio_dir is not None:
            lora_zen, lora_az = float(radio_dir[0]), float(radio_dir[1])
            lora_core = np.array([0.0, 0.0, 7.6 * units.m])
        else:
            lora_zen, lora_az = 0.3, 0.0
            lora_core = np.array([0.0, 0.0, 7.6 * units.m])

        lora_az_norm   = lora_az % (2.0 * np.pi)
        shower_dir_guess = (lora_az_norm, lora_zen)

        # PRE-SCAN: global reference pulse from 30 closest antennas
        all_ants = []
        for station in event.get_stations():
            sid     = station.get_id()
            abs_pos = detector.get_absolute_position(sid)
            for ef in station.get_electric_fields():
                ch_ids = ef.get_channel_ids()
                if len(ch_ids) < 2:
                    continue
                try:
                    rel_pos = detector.get_relative_position(sid, ch_ids[0])
                except Exception:
                    rel_pos = np.zeros(3)
                pos  = rel_pos + abs_pos
                dist = np.linalg.norm(pos[:2] - lora_core[:2])
                all_ants.append({"sid": sid, "dist": dist, "ef": ef, "pos": pos})

        all_ants.sort(key=lambda a: a["dist"])
        global_ref_time = None
        global_ref_pos  = None
        best_prescan_snr = 0.0
        for ant in all_ants[:30]:
            try:
                ef_trace = ant["ef"].get_trace()
                ef_times = ant["ef"].get_times() - ant["ef"].get_trace_start_time()
                t, snr   = pf.simple_hilbert_finder(ef_trace, ef_times)
                if t is not None and snr > best_prescan_snr:
                    best_prescan_snr = snr
                    global_ref_time  = t
                    global_ref_pos   = ant["pos"]
            except Exception:
                continue

        if global_ref_time is not None:
            self.logger.info("Pre-scan: E-field pulse SNR=%.1f at t=%.1f ns",
                             best_prescan_snr, global_ref_time)

        # TWO-PASS DIRECTION SEARCH
        search_attempts = [
            {"name": "standard", "scan_range": scan_range,            "step": step,          "sigma": sigma},
            {"name": "broad",    "scan_range": _BROAD_SCAN_RANGE_DEG, "step": _BROAD_STEP_DEG, "sigma": _BROAD_SMOOTHING_NS},
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
                        event, sid, detector, shower_dir_guess,
                        _RECO_STAGES[0]["window"],
                        global_ref_time, global_ref_pos,
                        scan_range_deg=attempt["scan_range"],
                        step_deg=attempt["step"],
                        sigma_ns=attempt["sigma"],
                        skip_lba_inner=skip_lba,
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
                ref_dir  = station_results[best_sid][7]
                best_ant = int(np.argmax(station_results[best_sid][6])) if len(station_results[best_sid][6]) > 0 else 0
                ref_time = float(station_results[best_sid][3][best_ant]) if len(station_results[best_sid][3]) > 0 else 0.0
                ref_pos  = (np.array([station_results[best_sid][0][best_ant],
                                      station_results[best_sid][1][best_ant]])
                            if len(station_results[best_sid][0]) > 0 else np.zeros(2))
                for sid, res in station_results.items():
                    if sid == best_sid:
                        continue
                    dev = np.rad2deg(pf.angular_distance(res[7][1], res[7][0], ref_dir[1], ref_dir[0]))
                    if dev > 3.0:
                        try:
                            station_results[sid] = pf.extract_fluence_and_timing_data(
                                event, sid, detector, ref_dir,
                                _RECO_STAGES[0]["window"],
                                ref_time * 1e9, ref_pos,
                                scan_range_deg=0.0,
                                step_deg=attempt["step"],
                                sigma_ns=attempt["sigma"],
                                skip_lba_inner=skip_lba,
                            )
                        except Exception:
                            pass

            # Flatten
            _px, _py, _fl, _tm, _sig, _snr, _nf, _sid = [], [], [], [], [], [], [], []
            for sid, res in station_results.items():
                px, py, fl_v, tm_v, is_sig_v, nf, snr_v, _ = res
                if len(fl_v) > 0:
                    _px.extend(px);  _py.extend(py);  _fl.extend(fl_v); _tm.extend(tm_v)
                    _sig.extend(is_sig_v); _snr.extend(snr_v); _nf.extend(nf)
                    _sid.extend([sid] * len(fl_v))

            if not _tm:
                if attempt["name"] == "standard":
                    self.logger.warning("Standard search: no timing data — retrying with broad scan.")
                    continue
                else:
                    self.logger.error("Both direction searches failed: no timing data extracted.")
                    return event

            posx_p1 = np.array(_px);  posy_p1 = np.array(_py)
            fl_p1   = np.array(_fl);  tm_p1   = np.array(_tm)
            sig_p1  = np.array(_sig, dtype=bool)
            snr_p1  = np.array(_snr); nf_p1  = np.array(_nf)
            sid_p1  = np.array(_sid)

            # Causality cut + outlier removal + iterative pruning
            kx_g   = np.sin(lora_zen) * np.cos(lora_az_norm)
            ky_g   = np.sin(lora_zen) * np.sin(lora_az_norm)
            geom_d = -(1.0 / _C_LIGHT) * (kx_g * posx_p1 + ky_g * posy_p1)
            t0_g   = float(np.median(tm_p1[sig_p1] - geom_d[sig_p1])) if np.any(sig_p1) else 0.0
            causality    = np.abs(tm_p1 - t0_g) <= (_CAUSALITY_CUT_DISTANCE_M / _C_LIGHT)
            use_timing_p1 = sig_p1 & causality

            nm_p1   = float(np.mean(nf_p1)) if nf_p1.size > 0 else 1.0
            snr_fl_p1 = fl_p1 / nm_p1 if nm_p1 > 0 else np.zeros_like(fl_p1)

            if np.sum(use_timing_p1) > _MIN_ABSOLUTE_NEIGHBORS:
                keep = pf.detect_timing_outliers(
                    np.array([posx_p1[use_timing_p1], posy_p1[use_timing_p1]]),
                    tm_p1[use_timing_p1], sid_p1[use_timing_p1],
                    timing_snrs=snr_fl_p1[use_timing_p1],
                )
                idx = np.where(use_timing_p1)[0]
                use_timing_p1[idx[~keep]] = False

            if np.sum(use_timing_p1) > _MIN_TIMING_POINTS:
                use_timing_p1, _, _, _, _, _ = pf.iterative_timing_pruning(
                    shower_dir_guess,
                    np.array([posx_p1, posy_p1]),
                    tm_p1, use_timing_p1,
                    station_ids=sid_p1, timing_snrs=snr_fl_p1,
                )

            n_good = int(np.sum(use_timing_p1))
            if n_good >= _MIN_GOOD_TIMING_POINTS_RECO:
                self.logger.info("Direction search (%s): %d good timing points.", attempt["name"], n_good)
                break
            elif attempt["name"] == "standard":
                self.logger.warning("Standard search: %d points — retrying broad.", n_good)
            else:
                self.logger.warning("Broad search: %d points — proceeding with best available.", n_good)

        # ---- Plane-wave fit → global direction ----
        radio_zen, radio_az, _ = pf.fit_plane_wave(
            np.array([posx_p1[use_timing_p1], posy_p1[use_timing_p1]]),
            tm_p1[use_timing_p1],
            weights=snr_p1[use_timing_p1],
        )
        if radio_zen is not None:
            global_fit_zen = radio_zen
            global_fit_az  = radio_az
            self.logger.info("Radio plane-wave fit: zen=%.1f°, az=%.1f°",
                             np.rad2deg(radio_zen), np.rad2deg(radio_az))
        else:
            self.logger.warning("Plane-wave fit failed — falling back to LORA direction.")
            global_fit_zen = lora_zen
            global_fit_az  = lora_az_norm

        shower_direction = (global_fit_az, global_fit_zen)

        # ---- E-field re-conversion at the fitted direction ----
        try:
            from NuRadioReco.modules.voltageToEfieldConverterPerChannelGroup import (
                voltageToEfieldConverterPerChannelGroup,
            )
            v2e = voltageToEfieldConverterPerChannelGroup()
            v2e.begin(use_MC_direction=False)
            for station in event.get_stations():
                station.set_parameter(stationParameters.zenith,  global_fit_zen)
                station.set_parameter(stationParameters.azimuth, global_fit_az)
                try:
                    v2e.run(event, station, detector)
                except Exception as exc:
                    self.logger.debug("V2E re-conversion failed for station %s: %s",
                                      station.get_id(), exc)
            self.logger.info("E-field re-conversion complete.")
        except ImportError:
            self.logger.warning("voltageToEfieldConverterPerChannelGroup not available — skipping re-conversion.")

        # RECO_STAGES: select best fluence extraction window
        best_res   = None
        sel_win    = 0.0
        noise_mean = 1.0
        noise_level = 1.0
        nf_by_station = {}

        for i_stage, stg in enumerate(_RECO_STAGES):
            self.logger.info("RECO stage %d: window=%.0f ns", i_stage, stg["window"])
            px_l, py_l, fl_l, tm_l, sig_l, snr_l, nf_l, sid_l = [], [], [], [], [], [], [], []
            _nf_by_station = {}
            for station in event.get_stations():
                sid = station.get_id()
                try:
                    px, py, fl_v, tm_v, is_sig_v, nf, snr_v, _ = pf.extract_fluence_and_timing_data(
                        event, sid, detector, shower_direction,
                        stg["window"],
                        global_ref_time, global_ref_pos,
                        scan_range_deg=0.0,
                        skip_lba_inner=skip_lba,
                    )
                except Exception as exc:
                    self.logger.debug("Stage %d extract failed for station %s: %s", i_stage, sid, exc)
                    continue
                if len(fl_v) > 0:
                    px_l.extend(px);  py_l.extend(py);  fl_l.extend(fl_v); tm_l.extend(tm_v)
                    sig_l.extend(is_sig_v); snr_l.extend(snr_v); nf_l.extend(nf)
                    sid_l.extend([int(sid)] * len(fl_v))
                    if len(nf) > 0:
                        _nf_by_station[int(sid)] = list(nf)

            if not fl_l:
                continue

            stage_nm = float(np.mean(nf_l)) if nf_l else 1.0
            stage_ns = float(np.std(nf_l)) if len(nf_l) > 1 else stage_nm
            curr_snr = np.array(fl_l) / stage_nm if stage_nm > 0 else np.zeros(len(fl_l))
            high_snr_count = int(np.sum(curr_snr > _FALLBACK_SNR_THRESHOLD))
            self.logger.info("  Stage %d: %d antennas with SNR>%.1f",
                             i_stage, high_snr_count, _FALLBACK_SNR_THRESHOLD)

            if high_snr_count >= _FALLBACK_ANTENNA_THRESHOLD or i_stage == len(_RECO_STAGES) - 1:
                best_res = (
                    np.array(px_l), np.array(py_l),
                    np.array(fl_l), np.array(tm_l),
                    np.array(sig_l, dtype=bool), np.array(snr_l),
                    np.array(nf_l), np.array(sid_l),
                )
                sel_win    = stg["window"]
                noise_mean = stage_nm
                noise_level = stage_ns
                nf_by_station = _nf_by_station
                self.logger.info("  Selected window: %.0f ns", sel_win)
                break

        if best_res is None:
            self.logger.error("RECO_STAGES: no data found in any stage — aborting.")
            return event

        posx, posy, fl, tm, is_sig, snr, _nfl, sids = best_res

        # FLUENCE OUTLIER REMOVAL
        if len(fl) > 5:
            out_thresh = np.mean(fl) + 8.0 * np.std(fl)
            bad_mask = (fl > out_thresh) | (fl < _FLUENCE_MIN_NOISE_FACTOR * noise_mean)
            if np.any(bad_mask):
                self.logger.info("Removing %d fluence outliers.", int(np.sum(bad_mask)))
                keep = ~bad_mask
                posx, posy, fl, tm, is_sig, snr, sids = (
                    arr[keep] for arr in (posx, posy, fl, tm, is_sig, snr, sids))

        # NEIGHBOUR AUGMENTATION
        if np.sum(is_sig) > 0:
            try:
                kx_a   = np.sin(global_fit_zen) * np.cos(global_fit_az)
                ky_a   = np.sin(global_fit_zen) * np.sin(global_fit_az)
                geom_a = -(1.0 / _C_LIGHT) * (kx_a * posx[is_sig] + ky_a * posy[is_sig])
                t0_aug = float(np.median(tm[is_sig] - geom_a))
                known_set = {(round(float(posx[i]), 1), round(float(posy[i]), 1)) for i in range(len(posx))}
                station_ids_list = [int(station.get_id()) for station in event.get_stations()]
                aug = pf.augment_neighbor_fluences(
                    event, detector, station_ids_list,
                    posx[is_sig], posy[is_sig], t0_aug,
                    shower_direction, sel_win, known_set,
                    skip_lba_inner=skip_lba,
                )
                if aug["fluences"]:
                    self.logger.info("Neighbour augmentation: added %d antenna(s).", len(aug["fluences"]))
                    posx   = np.append(posx,   aug["posx"])
                    posy   = np.append(posy,   aug["posy"])
                    fl     = np.append(fl,     aug["fluences"])
                    tm     = np.append(tm,     aug["times"])
                    is_sig = np.append(is_sig, np.array(aug["is_signal"], dtype=bool))
                    snr    = np.append(snr,    aug["snrs"])
                    sids   = np.append(sids,   np.array(aug["station_ids"], dtype=sids.dtype))
            except Exception as exc:
                self.logger.warning("Neighbour augmentation failed: %s", exc)

        extra_fl_only = {k: np.array([]) for k in ("posx", "posy", "fl", "tm", "snr")}
        extra_fl_only["sids"] = np.array([], dtype=int)

        # STATION-LEVEL NOISE-FLOOR FILTER
        if noise_mean > 0:
            unique_sids = np.unique(sids)
            stn_n_above = {int(s_): int(np.sum(fl[sids == s_] / noise_mean >= _STATION_SNR_THRESHOLD))
                           for s_ in unique_sids}
            stn_max_snr = {int(s_): float(np.max(fl[sids == s_]) / noise_mean)
                           for s_ in unique_sids}
            good_stns = {s_ for s_, n in stn_n_above.items() if n >= _STATION_MIN_ANTENNAS}
            bad_stns  = set(stn_n_above.keys()) - good_stns

            if len(good_stns) < _MIN_STATIONS_REQUIRED:
                top_n = sorted(stn_max_snr, key=stn_max_snr.get, reverse=True)[:_MIN_STATIONS_REQUIRED]
                self.logger.warning("Station filter fallback: keeping top-%d stations by SNR.", _MIN_STATIONS_REQUIRED)
                good_stns = set(top_n)
                bad_stns  = set(stn_n_above.keys()) - good_stns

            if len(good_stns) < _MIN_STATIONS_REQUIRED:
                self.logger.error(
                    "Only %d station(s) with signal (need %d) — IFT reconstruction requires more data. Proceeding anyways.",
                    len(good_stns), _MIN_STATIONS_REQUIRED,
                )

            if bad_stns:
                extra_px, extra_py, extra_fl_v, extra_tm_v, extra_snr_v, extra_sids_v = [], [], [], [], [], []
                for _sid in sorted(bad_stns):
                    try:
                        _px, _py, _fl_v, _tm_v, _sig_v, _nf_v, _snr_v, _ = pf.extract_fluence_and_timing_data(
                            event, _sid, detector, shower_direction, sel_win,
                            global_ref_time=None, global_ref_pos=None,
                            skip_lba_inner=skip_lba,
                        )
                        _snr_arr  = np.asarray(_snr_v, dtype=float)
                        _sig_arr  = np.asarray(_sig_v, dtype=bool)
                        _high_msk = _sig_arr & (_snr_arr >= _FAR_STATION_FLUENCE_ONLY_SNR)
                        _n_high   = int(np.sum(_high_msk))
                        for i in range(len(_fl_v)):
                            if _sig_v[i] and _snr_v[i] >= _FAR_STATION_FLUENCE_ONLY_SNR:
                                n_other = _n_high - (1 if _high_msk[i] else 0)
                                if n_other < _FAR_STATION_MIN_HIGH_SNR_ANTENNAS:
                                    continue
                                extra_px.append(_px[i]);    extra_py.append(_py[i])
                                extra_fl_v.append(_fl_v[i]); extra_tm_v.append(_tm_v[i])
                                extra_snr_v.append(_snr_v[i]); extra_sids_v.append(int(_sid))
                    except Exception:
                        pass
                if extra_fl_v:
                    extra_fl_only = {
                        "posx": np.array(extra_px), "posy": np.array(extra_py),
                        "fl":   np.array(extra_fl_v), "tm":  np.array(extra_tm_v),
                        "snr":  np.array(extra_snr_v),
                        "sids": np.array(extra_sids_v, dtype=int),
                    }
                    self.logger.info("Added %d fluence-only from %d low-signal station(s).",
                                     len(extra_fl_v), len(bad_stns))
                keep = ~np.isin(sids, list(bad_stns))
                posx, posy, fl, tm, is_sig, snr, sids = (
                    arr[keep] for arr in (posx, posy, fl, tm, is_sig, snr, sids))

        # TIMING PRUNING & UNCERTAINTY
        kx = np.sin(global_fit_zen) * np.cos(global_fit_az)
        ky = np.sin(global_fit_zen) * np.sin(global_fit_az)
        geom_delays = -(1.0 / _C_LIGHT) * (kx * posx + ky * posy)
        t0_guess    = float(np.median(tm[is_sig] - geom_delays[is_sig])) if np.any(is_sig) else 0.0

        causality_mask = np.abs(tm - t0_guess) <= (_CAUSALITY_CUT_DISTANCE_M / _C_LIGHT)
        use_timing     = is_sig & causality_mask
        snr_fl         = fl / noise_mean if noise_mean > 0 else np.zeros_like(fl)

        if np.sum(use_timing) > _MIN_ABSOLUTE_NEIGHBORS:
            keep_sub = pf.detect_timing_outliers(
                np.array([posx[use_timing], posy[use_timing]]),
                tm[use_timing], sids[use_timing], timing_snrs=snr_fl[use_timing],
            )
            idx = np.where(use_timing)[0]
            use_timing[idx[~keep_sub]] = False

        t0_prior   = 20e-9
        fit_coeffs = fit_mean_pos = fit_scale = inv_AtA = None
        final_std  = _TIMING_UNCERT_THRESHOLD
        if np.sum(use_timing) > _MIN_TIMING_POINTS:
            use_timing, final_std, fit_coeffs, fit_mean_pos, fit_scale, inv_AtA = \
                pf.iterative_timing_pruning(
                    shower_direction, np.array([posx, posy]), tm, use_timing,
                    station_ids=sids, timing_snrs=snr_fl,
                )
            if final_std > _TIMING_UNCERT_THRESHOLD or np.sum(use_timing) < _MIN_TIMING_POINTS:
                self.logger.error(
                    "Timing quality check failed (std=%.2f ns, n=%d) — aborting.",
                    final_std * 1e9, int(np.sum(use_timing)))
                return event

            self.logger.info("Timing pruning: %d points, residual std=%.2f ns.",
                             int(np.sum(use_timing)), final_std * 1e9)

            # Refine t0 at LORA core
            core_prior_x = float(lora_core[0])
            core_prior_y = float(lora_core[1])
            if fit_coeffs is not None and fit_mean_pos is not None:
                dx  = core_prior_x - fit_mean_pos[0]
                dy  = core_prior_y - fit_mean_pos[1]
                xn  = dx / fit_scale; yn = dy / fit_scale
                r2n = xn ** 2 + yn ** 2
                t0_guess = float(fit_coeffs[0] + fit_coeffs[1] * xn +
                                 fit_coeffs[2] * yn + fit_coeffs[3] * r2n)
                if inv_AtA is not None:
                    p = np.array([1.0, xn, yn, r2n])
                    extrap_var = float(p.T @ inv_AtA @ p) * (final_std ** 2)
                    dist_near  = float(np.min(np.sqrt((posx - core_prior_x) ** 2 +
                                                      (posy - core_prior_y) ** 2)))
                    dist_unc   = (dist_near / _C_LIGHT) * 0.3
                    t0_prior   = max(float(np.sqrt(extrap_var)), final_std, dist_unc, 2e-9)
        else:
            self.logger.warning("Not enough timing points — using fallback t0 prior.")

        # Local timing uncertainties
        timing_idx           = np.where(use_timing)[0]
        per_point_timing_std = np.full(len(posx), 5.0e-9)
        if len(timing_idx) >= 8:
            local_std = pf.get_local_timing_uncertainties(
                np.array([posx[timing_idx], posy[timing_idx]]),
                tm[timing_idx], sids[timing_idx], shower_direction,
            )
            per_point_timing_std[timing_idx] = local_std
        per_point_timing_std *= _TIMING_SYST_INFLATION
        inv_var_time            = 1.0 / per_point_timing_std ** 2
        inv_var_time[~use_timing] = 0.0

        # Station multiplicity cut (≥ 2 timing points per station)
        stns_with_timing = {int(s_) for s_ in np.unique(sids)
                            if np.sum((sids == s_) & use_timing) >= 2}
        if stns_with_timing:
            _keep = np.isin(sids, list(stns_with_timing))
            if np.sum(_keep) < len(fl):
                self.logger.info("Station multiplicity cut: %d points dropped.", int(len(fl) - np.sum(_keep)))
                posx = posx[_keep]; posy = posy[_keep]; fl   = fl[_keep]
                tm   = tm[_keep];   is_sig = is_sig[_keep]; snr = snr[_keep]
                sids = sids[_keep]; use_timing = use_timing[_keep]
                per_point_timing_std = per_point_timing_std[_keep]
                inv_var_time         = inv_var_time[_keep]

        # Recompute noise from timing-valid stations only
        if stns_with_timing and nf_by_station:
            valid_nf = [w for sid_k, w in nf_by_station.items()
                        if sid_k in stns_with_timing and len(w) > 0]
            if valid_nf:
                flat = np.concatenate([np.asarray(w) for w in valid_nf])
                if flat.size > 0:
                    noise_mean  = float(np.mean(flat))
                    noise_level = float(np.std(flat)) if flat.size > 1 else noise_mean
                    self.logger.info("Noise recomputed: mean=%.3e, std=%.3e", noise_mean, noise_level)

        # Append fluence-only from bad stations
        n_extra = len(extra_fl_only["fl"])
        if n_extra > 0:
            posx   = np.append(posx,   extra_fl_only["posx"])
            posy   = np.append(posy,   extra_fl_only["posy"])
            fl     = np.append(fl,     extra_fl_only["fl"])
            tm     = np.append(tm,     extra_fl_only["tm"])
            is_sig = np.append(is_sig, np.zeros(n_extra, dtype=bool))
            snr    = np.append(snr,    extra_fl_only["snr"])
            sids   = np.append(sids,   extra_fl_only["sids"])
            use_timing           = np.append(use_timing,
                                             np.zeros(n_extra, dtype=bool))
            per_point_timing_std = np.append(per_point_timing_std,
                                             np.full(n_extra, 5.0e-9 * _TIMING_SYST_INFLATION))
            inv_var_time         = np.append(inv_var_time, np.zeros(n_extra))

        noise_level = max(noise_level, 1e-3 * noise_mean)
        self.logger.info("Global noise: mean=%.3e, std=%.3e (%d stations)",
                         noise_mean, noise_level, len(np.unique(sids)))

        # ATMOSPHERE PATH
        atm_path = None
        if atm_dir:
            try:
                from NuRadioReco.modules.LOFAR.utilities.gdas_tool import find_or_generate_atmosphere
                atm_path = find_or_generate_atmosphere(
                    event.get_id(), atm_dir, gdas_cache_dir=s.get("gdas_cache_dir")
                )
            except Exception as exc:
                self.logger.warning(
                    "GDAS atmosphere lookup/generation failed: %s — "
                    "falling back to standard US atmosphere (model 17).", exc
                )
        elif not s.get("use_fallback_atmosphere", False):
            self.logger.warning(
                "No atmosphere_dir provided to iftReconstructor.begin(). "
                "Pass atmosphere_dir for an event-specific GDAS atmosphere. "
                "Using standard US atmosphere (model 17) — Xmax precision will be reduced. "
                "Set use_fallback_atmosphere=True to silence this warning."
            )

        # MODEL CONSTRUCTION
        lora_core_x = float(lora_core[0])
        lora_core_y = float(lora_core[1])

        phi_prior     = {"a_min": np.pi,  "a_max": 3.0 * np.pi}
        theta_prior   = {"a_min": 0.0,    "a_max": np.radians(60.0)}
        t0_prior_inflated = t0_prior * 5.0

        model_kw = {
            "params_phi":   phi_prior,
            "params_theta": theta_prior,
            "params_X_max": {"a_min": 400.0, "a_max": 1200.0},
            "params_X":     {"mean": 0.0, "std": core_prior_std},
            "params_Y":     {"mean": 0.0, "std": core_prior_std},
            "noise_mean":   noise_mean,
            "atmosphere_path": atm_path,
            "params_t0":    {"mean": t0_guess, "std": t0_prior_inflated},
            "timing_std_s": _TIMING_UNCERTAINTY_S,
            "enable_syst_cf":   fluence_cf,
            "enable_timing_cf": timing_cf,
            "enable_ldf_energy_scale_uncertainty": True,
            "ldf_energy_scale_fractional_std": _LDF_ENERGY_SCALE_STD,
            "enable_fluence_dxmax_precision": True,
            "fluence_dxmax_precision_gpcm2":  _FLUENCE_DXMAX_PRECISION_GPCM2,
            "enable_timing_dxmax_precision":  True,
            "timing_dxmax_precision_gpcm2":   _TIMING_DXMAX_PRECISION_GPCM2,
            "syst_mult_min": SYST_MULT_MIN,
            "syst_mult_max": SYST_MULT_MAX,
        }

        b_field       = hp.get_magnetic_field_vector("lofar")
        _fl_denom = np.maximum(noise_level ** 2 + (fl * _FLUENCE_RELATIVE_SYSTEMATIC_ERROR) ** 2, 1e-60)
        inv_fl        = 1.0 / _fl_denom
        noise_cov_inv = jnp.stack([jnp.array(inv_fl), jnp.array(inv_var_time)])

        model = footprintModel(posx, posy, b_field, **model_kw)
        lh    = jft.Gaussian(
            data=jnp.stack([jnp.array(fl), jnp.array(tm)]),
            noise_cov_inv=noise_cov_inv,
        ).amend(model)

        # INIT VALUES FROM LORA/RADIO PRIOR
        key          = random.PRNGKey(seed)
        key, k_init, k_opt = random.split(key, 3)

        init_state  = lh.init(k_init)
        init_values = dict(init_state.tree if hasattr(init_state, "tree") else init_state)

        az_wrapped = np.pi + (global_fit_az - np.pi) % (2.0 * np.pi)
        init_values["phi"]   = pf.uniform_prior_latent(az_wrapped, phi_prior)
        init_values["theta"] = pf.uniform_prior_latent(global_fit_zen, theta_prior)
        init_values["X_max"] = jnp.array([0.0])
        init_values["X"]     = jnp.array([lora_core_x / core_prior_std])
        init_values["Y"]     = jnp.array([lora_core_y / core_prior_std])
        init_values["t0"]    = jnp.array([0.0])
        pos_init = jft.Vector(init_values)

        # OptimizeVI LOOP WITH CONVERGENCE CHECK
        n_dof_chisq = max(int(np.asarray(jnp.stack([jnp.array(fl), jnp.array(tm)])).size), 1)
        delta = 1e-7
        n_samples_sched   = lambda i: n_samples // 2 if i < n_vi // 2 else n_samples
        sample_mode_sched = lambda iiter: _SAMPLING_MODE if iiter < 2 else _RESAMPLING_MODE

        opt_vi    = jft.OptimizeVI(lh, n_total_iterations=n_vi, kl_map=vmap)
        opt_state = opt_vi.init_state(
            k_opt,
            n_samples=n_samples_sched,
            draw_linear_kwargs=dict(
                cg=jft.static_cg,
                cg_name="SL",
                cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=800),
            ),
            nonlinearly_update_kwargs=dict(
                minimize_kwargs=dict(name="SN", xtol=delta, cg_kwargs=dict(name=None), maxiter=25),
            ),
            kl_kwargs=dict(
                minimize_kwargs=dict(name="M",  xtol=delta, cg_kwargs=dict(name=None), maxiter=40),
            ),
            sample_mode=sample_mode_sched,
        )
        samples_liquid = jft.Samples(pos=pos_init, samples=None, keys=None)

        prev_redchisq = None
        n_stable      = 0
        is_converged  = False
        for i_iter in range(n_vi):
            samples_liquid, opt_state = opt_vi.update(samples_liquid, opt_state)
            try:
                self.logger.info(opt_vi.get_status_message(samples_liquid, opt_state))
            except Exception:
                pass

            chisq     = np.array([2.0 * float(lh(s_)) for s_ in samples_liquid])
            red_chisq = float(chisq.mean()) / n_dof_chisq
            mode_now  = sample_mode_sched(opt_state.nit)
            is_nonlinear = mode_now.startswith("nonlinear")

            if prev_redchisq is not None:
                d_chisq  = abs(red_chisq - prev_redchisq)
                n_stable = n_stable + 1 if (d_chisq < _CONV_TOL_REDCHISQ and is_nonlinear) else 0
                self.logger.info(
                    "VI iter %d/%d: chi2/dof=%.3f (delta=%.4f, stable=%d/%d)",
                    i_iter + 1, n_vi, red_chisq, d_chisq, n_stable, _CONV_PATIENCE)
                if n_stable >= _CONV_PATIENCE and (i_iter + 1) >= _CONV_MIN_ITERS:
                    self.logger.info("VI converged after %d iterations.", i_iter + 1)
                    is_converged = True
                    break
            else:
                self.logger.info("VI iter %d/%d: chi2/dof=%.3f", i_iter + 1, n_vi, red_chisq)
            prev_redchisq = red_chisq

        n_vi_iters_run = i_iter + 1

        # TRIGGER FILTER
        samples_list = list(samples_liquid)
        num_efield_triggered = pf.count_efield_triggered_stations(fl, sids, noise_mean, sel_win)
        unique_s_count   = len(np.unique(sids))
        min_trigger_stns = min(max(num_efield_triggered, 1), unique_s_count)
        kept_idx, _ = pf.filter_samples_by_trigger(
            model, samples_list, sids, noise_mean, sel_win, min_trigger_stns,
            snr_buffer_frac=_TRIGGER_SNR_BUFFER_FRAC,
        )
        n_kept = len(kept_idx)
        n_discarded = len(samples_list) - n_kept
        self.logger.info("Trigger filter: %d/%d samples passed (min_stations=%d)",
                         n_kept, len(samples_list), min_trigger_stns)
        filtered_samples = [samples_list[i] for i in kept_idx] if n_kept > 1 else samples_list

        # POSTERIOR SUMMARY
        def _s(v):
            return float(np.asarray(v).squeeze())

        zen_s  = [_s(np.rad2deg(model.zen_and_az(s_)[0])) for s_ in filtered_samples]
        # Wrap model phi (prior lives in [π, 3π]) to [0, 2π] before averaging
        az_s   = [_s(model.zen_and_az(s_)[1] % (2 * np.pi)) for s_ in filtered_samples]
        cx_s   = [_s(model.core(s_)[0])           for s_ in filtered_samples]
        cy_s   = [_s(model.core(s_)[1])           for s_ in filtered_samples]
        xmax_s = [_s(model.X_max_combined(s_))    for s_ in filtered_samples]
        ecr_s  = [_s(_calculate_ecr(model, s_))                                   for s_ in filtered_samples]
        erad_s = [_s(model.Erad(s_) / model.get_energy_correction_factor(s_))  for s_ in filtered_samples]

        zen_mean,  zen_std  = (float(v) for v in jft.mean_and_std(zen_s,  correct_bias=True))
        cx_mean,   cx_std   = (float(v) for v in jft.mean_and_std(cx_s,   correct_bias=True))
        cy_mean,   cy_std   = (float(v) for v in jft.mean_and_std(cy_s,   correct_bias=True))
        xmax_mean, xmax_std = (float(v) for v in jft.mean_and_std(xmax_s, correct_bias=True))
        ecr_mean,  ecr_std  = (float(v) for v in jft.mean_and_std(ecr_s,  correct_bias=True))
        erad_mean, erad_std = (float(v) for v in jft.mean_and_std(erad_s, correct_bias=True))
        # Azimuth needs circular mean/std — arithmetic mean fails near the 0/2π wrap
        _sin_mean   = np.mean(np.sin(az_s))
        _cos_mean   = np.mean(np.cos(az_s))
        _R          = float(np.sqrt(_sin_mean**2 + _cos_mean**2))
        az_mean_rad = float(np.arctan2(_sin_mean, _cos_mean)) % (2 * np.pi)
        az_mean     = float(np.rad2deg(az_mean_rad))
        az_std      = float(np.rad2deg(np.sqrt(-2.0 * np.log(max(_R, 1e-10)))))

        self.logger.info(
            "IFT result: zen=%.2f±%.2f°, az=%.2f±%.2f°, "
            "core=(%.1f±%.1f, %.1f±%.1f) m, Xmax=%.0f±%.0f g/cm², "
            "E_CR=%.3e±%.3e eV, E_rad=%.3e±%.3e eV",
            zen_mean, zen_std, az_mean, az_std,
            cx_mean, cx_std, cy_mean, cy_std,
            xmax_mean, xmax_std, ecr_mean, ecr_std,
            erad_mean, erad_std,
        )

        # WRITE TO RADIO SHOWER
        reco_zen  = np.radians(zen_mean)
        reco_az   = az_mean_rad
        reco_core = np.array([cx_mean * units.m, cy_mean * units.m, 7.6 * units.m])
        reco_energy = ecr_mean * units.eV
        reco_xmax   = xmax_mean * units.g / units.cm ** 2
        reco_erad   = erad_mean * units.eV

        shower = self._get_or_create_radio_shower(event)
        shower.set_parameter(showerParameters.zenith,           reco_zen)
        shower.set_parameter(showerParameters.azimuth,          reco_az)
        shower.set_parameter(showerParameters.core,             reco_core)
        shower.set_parameter(showerParameters.shower_maximum,   reco_xmax)
        shower.set_parameter(showerParameters.energy,           reco_energy)
        shower.set_parameter(showerParameters.radiation_energy, reco_erad)
        shower.set_parameter(showerParameters.magnetic_field_vector, b_field)
        shower.set_parameter(showerParameters.observation_level, 760 * units.cm)

        for station in event.get_stations():
            station.set_parameter(stationParameters.cr_zenith, reco_zen)
            station.set_parameter(stationParameters.cr_azimuth, reco_az)
            station.set_parameter(stationParameters.cr_energy,  reco_energy)
            station.set_parameter(stationParameters.cr_xmax,    reco_xmax)

        dist_m = self._distance_to_shower_maximum(xmax_mean, reco_zen, gdas_file=atm_path)
        shower.set_parameter(showerParameters.distance_shower_maximum_geometric, dist_m * units.m)
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
            zen_mean, zen_std, az_mean, az_std,
            cx_mean, cx_std, cy_mean, cy_std,
            xmax_mean, xmax_std, ecr_mean, ecr_std, erad_mean, erad_std,
        )

        # RECONSTRUCTION PLOT
        _out_dir = s.get("output_directory") or os.getcwd()
        os.makedirs(_out_dir, exist_ok=True)

        try:
            cf_stats = None
            if model.enable_syst_cf or model.enable_timing_cf:
                _syst  = [np.asarray(model.syst_cf(s_)) for s_ in filtered_samples]
                _tmcf  = [np.clip(np.asarray(model.timing_cf_op_2(s_)) * 1e9,
                                  -15.0, 15.0) for s_ in filtered_samples]
                cf_stats = {
                    "syst_mean":   np.mean([np.exp(f) for f in _syst], axis=0),
                    "syst_std":    np.std([np.exp(f)  for f in _syst], axis=0),
                    "timing_mean": np.mean(_tmcf, axis=0),
                    "timing_std":  np.std(_tmcf,  axis=0),
                }

            from NuRadioReco.modules.LOFAR.utilities import iftOutput
            all_data_plot = {
                'pos_x': posx, 'pos_y': posy,
                'fluences': fl, 'times': tm,
                'is_signal': use_timing,
            }
            ref_params_plot = {
                'zenith':  lora_zen,
                'azimuth': lora_az,
                'core':    np.asarray(lora_core[:2], dtype=float),
                'energy':  float(lora.get('energy', 0.0)) if lora is not None else 0.0,
            }
            iftOutput.generate_reco_plot(
                filtered_samples, ecr_s, all_data_plot, _out_dir, event.get_id(),
                ref_params=ref_params_plot,
                signal_response=model,
                noise_mean=noise_mean,
                model_kw=model_kw,
                noise_level=noise_level,
                cf_stats=cf_stats,
                b_field=b_field,
            )
        except Exception as _plot_exc:
            self.logger.warning("Reconstruction plot failed: %s", _plot_exc, exc_info=True)

        if s.get("export_posterior_samples", False):
            out_dir = s.get("output_directory") or os.getcwd()
            os.makedirs(out_dir, exist_ok=True)
            np.savez(
                os.path.join(out_dir, f"{event.get_id()}.npz"),
                event_id=event.get_id(),
                zenith=reco_zen, azimuth=reco_az, core=reco_core,
                energy=reco_energy, xmax=reco_xmax, radiation_energy=reco_erad,
                zen_mean=zen_mean, zen_std=zen_std,
                az_mean=az_mean,   az_std=az_std,
                core_x_mean=cx_mean, core_x_std=cx_std,
                core_y_mean=cy_mean, core_y_std=cy_std,
                xmax_mean=xmax_mean, xmax_std=xmax_std,
                ecr_mean=ecr_mean,   ecr_std=ecr_std,
                erad_mean=erad_mean, erad_std=erad_std,
                red_chisq=red_chisq, n_dof=n_dof_chisq,
                is_converged=is_converged,
                n_vi_iters=n_vi_iters_run, n_vi_max=n_vi,
                n_samples_total=len(samples_list), n_samples_kept=n_kept,
                n_samples_discarded=n_discarded,
                n_efield_triggered=num_efield_triggered,
                min_trigger_stations=min_trigger_stns,
                n_antennas=len(fl), n_timing=int(np.sum(use_timing)),
                noise_mean=noise_mean, sel_win=sel_win,
                fluences=fl, pos_x=posx, pos_y=posy, times=tm,
                is_signal=is_sig, use_timing=use_timing, sids=sids,
            )

        return event

    @register_run()
    def run(self, event, detector):
        if self.__settings.get("run_nifty", False):
            return self._run_nifty(event, detector)

        prior           = self._get_lora_prior(event)
        radio_direction = self._get_radio_direction(event)

        if prior is not None:
            zenith  = prior["zenith"]
            azimuth = prior["azimuth"]
            core    = prior["core"].copy()
            energy  = prior.get("energy", 1e17 * units.eV)
        elif radio_direction is not None:
            zenith, azimuth = radio_direction
            core   = np.array([0.0, 0.0, 7.6 * units.m])
            energy = 1e17 * units.eV
        else:
            zenith  = 0.0 * units.radian
            azimuth = 0.0 * units.radian
            core    = np.array([0.0, 0.0, 7.6 * units.m])
            energy  = 1e17 * units.eV

        positions, fluences, station_ids = self._collect_observables(event, detector)
        finite = np.isfinite(fluences) & (fluences > 0)

        if prior is None and np.any(finite):
            weights      = fluences[finite]
            weighted_core = np.average(positions[finite, :2], weights=weights, axis=0)
            core[:2]     = weighted_core

        shower = self._get_or_create_radio_shower(event)
        shower.set_parameter(showerParameters.zenith,           zenith)
        shower.set_parameter(showerParameters.azimuth,          azimuth)
        shower.set_parameter(showerParameters.core,             core)
        shower.set_parameter(showerParameters.energy,           energy)
        shower.set_parameter(showerParameters.magnetic_field_vector, hp.get_magnetic_field_vector("lofar"))
        shower.set_parameter(showerParameters.observation_level, 760 * units.cm)

        for station in event.get_stations():
            station.set_parameter(stationParameters.cr_zenith, zenith)
            station.set_parameter(stationParameters.cr_azimuth, azimuth)
            station.set_parameter(stationParameters.cr_energy,  energy)

        self._set_efield_parameters(event, detector, float(zenith), float(azimuth))

        self._plot_reconstruction_debug(
            event, positions, fluences, station_ids, core, zenith, azimuth
        )

        if self.__settings.get("export_posterior_samples", False):
            output_directory = self.__settings.get("output_directory") or os.getcwd()
            os.makedirs(output_directory, exist_ok=True)
            np.savez(
                os.path.join(output_directory, f"{event.get_id()}.npz"),
                event_id=event.get_id(),
                zenith=zenith, azimuth=azimuth, core=core,
                energy=energy,
                positions=positions, fluences=fluences, station_ids=station_ids,
                n_iterations=self.__settings.get("n_iterations"),
                n_samples=self.__settings.get("n_samples"),
                enable_fluence_correlated_field=self.__settings.get("enable_fluence_correlated_field"),
                enable_timing_correlated_field=self.__settings.get("enable_timing_correlated_field"),
            )

        self.logger.info(
            "LOFAR IFT smoke reconstruction event %s: zenith %.2f deg, azimuth %.2f deg, "
            "core=(%.1f, %.1f)",
            event.get_id(), zenith / units.deg, azimuth / units.deg,
            core[0] / units.m, core[1] / units.m,
        )
        return event

    def end(self):
        pass
