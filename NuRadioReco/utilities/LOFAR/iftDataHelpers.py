"""
Pulse, fluence, and timing helpers for the LOFAR IFT reconstruction.

.. moduleauthor:: Karen Terveer <karen.terveer@fau.de>
"""

from collections import defaultdict
import logging
import os

import numpy as np
import scipy.constants
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, correlate
from scipy.spatial.distance import cdist
from scipy.stats import norm as _scipy_norm
try:
    import jax
    import jax.numpy as jnp
    import jax.tree_util as jtu
except ImportError:
    jax = None
    jnp = None
    jtu = None

from NuRadioReco.utilities import units


logger = logging.getLogger("NuRadioReco.LOFAR.iftDataHelpers")

CORE_STATIONS_ALWAYS_KEEP = set()
POSITION_CUT = 350 * units.m
MIN_TIMING_POINTS = 24
TIMING_UNCERT_THRESHOLD = 15.0 * units.ns / units.s
SNR_CUT = 2.5
ROI_START_NS = 0.0
ROI_END_NS = 400000.0
DEFAULT_FLUENCE_WINDOW_NS = 400.0
# Trace and detector values use NuRadioReco's internal unit system. This is
# therefore the speed of light in internal metres per internal time unit.
C_LIGHT_NURADIO = scipy.constants.c * units.m / units.s
OUTLIER_NEIGHBORS_K = 10
OUTLIER_STD_FACTOR = 2.5
MIN_ABSOLUTE_NEIGHBORS = 8
TRIGGER_SNR_THRESHOLD = 6.0
TRIGGER_WINDOW_NS = 55.0
TRIGGER_ANTENNA_FRACTION = 0.5
TRIGGER_MIN_STATIONS = 3
TRIGGER_SNR_BUFFER_FRAC = 0.05
# SNR above which a timing point is shielded from global pruning unless it is
# also a severe local outlier. Wrong-peak locks reach SNR 8-15, so a lower gate
# shields them and stalls the pruning.
HIGH_SNR_TIMING_PROTECTION_SNR = 15.0
HIGH_SNR_TIMING_OUTLIER_SIGMA = 10.0
HIGH_SNR_TIMING_MIN_LOCAL_SIGMA_NS = 1.0
LOCAL_FIT_NEIGHBORS_K = 20
DEFAULT_SCAN_RANGE_DEG = 5.0
DEFAULT_STEP_DEG = 0.5
DEFAULT_SMOOTHING_NS = 5.0

# --- Blind "largest signal in every channel" fallback (extract_max_signal_timing_data)
# Calibrated on real LBA_OUTER data. P(noise>=thr) / P(signal>=thr):
# 5.0: 0.157/0.83, 6.0: 0.015/0.73, 8.0: 0.009/0.49, 12.0: 0.007/0.18. Above ~6
# the noise curve flattens at ~1% (RFI, not thermal), so a higher cut only costs
# signal.
MAX_SIGNAL_SNR_THRESHOLD = 6.0
# Trace fraction skipped at each end, where the band-pass and taper suppress power.
MAX_SIGNAL_EDGE_FRACTION = 0.125
# Half-width (ns) of the arrival-time consensus window.
MAX_SIGNAL_CONSENSUS_WINDOW_NS = 150.0
# Antennas that must share one arrival plane before a station reports anything.
# The per-antenna cut alone is not enough (~7 noise antennas survive in a
# 450-antenna event); the consensus window is what rejects them.
MAX_SIGNAL_MIN_CONSENSUS_ANTENNAS = MIN_ABSOLUTE_NEIGHBORS

conversion_factor_integrated_signal = (
    scipy.constants.c
    * scipy.constants.epsilon_0
    * units.joule
    / units.s
    / units.volt ** 2
)


def get_electric_field_energy_fluence(
        electric_field_trace, times, signal_window_mask=None, noise_window_mask=None):
    """Compute simple integrated electric-field energy fluence without Rice subtraction."""
    electric_field_trace = np.asarray(electric_field_trace)
    times = np.asarray(times)
    if electric_field_trace.size == 0 or times.size < 2:
        return np.zeros(electric_field_trace.shape[0] if electric_field_trace.ndim > 1 else 1)

    if signal_window_mask is None:
        f_signal = np.sum(np.abs(electric_field_trace) ** 2, axis=1)
    else:
        f_signal = np.sum(np.abs(electric_field_trace[:, signal_window_mask]) ** 2, axis=1)

    if noise_window_mask is not None and np.sum(noise_window_mask) > 0:
        f_noise = np.sum(electric_field_trace[:, noise_window_mask] ** 2, axis=1)
        f_signal -= f_noise * np.sum(signal_window_mask) / np.sum(noise_window_mask)

    return f_signal * (times[1] - times[0]) * conversion_factor_integrated_signal


def compute_polarization_angle(efield, station, zenith_rad, azimuth_rad):
    """Compute the linear polarization angle in the vxB/vxvxB shower plane.

    Uses Stokes parameters evaluated at the sample of peak Stokes-I intensity
    within the signal window stored on the first channel of the E-field group.
    Returns the angle in radians (0 = vxB direction, pi/2 = vxvxB direction),
    or None if the computation cannot be completed.
    """
    import radiotools.coordinatesystems
    from NuRadioReco.framework.electric_field import get_stokes
    from NuRadioReco.framework.parameters import channelParameters as _ch_p

    ch_ids = efield.get_channel_ids()
    if not ch_ids:
        return None
    try:
        signal_window = station.get_channel(ch_ids[0]).get_parameter(_ch_p.signal_regions)
        pulse_start, pulse_end = int(signal_window[0]), int(signal_window[1])
    except Exception:
        return None

    trace = efield.get_trace()[:, pulse_start:pulse_end]
    if trace.shape[1] < 8:
        return None
    try:
        cs = radiotools.coordinatesystems.cstrafo(
            zenith_rad, azimuth_rad, magnetic_field_vector=None, site="lofar"
        )
        trace_shower = cs.transform_to_vxB_vxvxB(cs.transform_from_onsky_to_ground(trace))
        stokes = get_stokes(*trace_shower[:2], window_samples=min(64, trace.shape[1]))
        idx = int(np.argmax(stokes[0]))
        return 0.5 * np.arctan2(float(stokes[2, idx]), float(stokes[1, idx]))
    except Exception:
        return None


def get_spatial_clusters(positions, n_neighbors=6, overlap_stride=3):
    """Generate overlapping nearest-neighbor antenna clusters."""
    positions = np.asarray(positions)
    n_antennas = len(positions)
    if n_antennas <= n_neighbors:
        return [np.arange(n_antennas)]

    dist_matrix = cdist(positions, positions)
    clusters = []
    for index in range(0, n_antennas, overlap_stride):
        clusters.append(np.argsort(dist_matrix[index])[:n_neighbors])
    clusters.append(np.arange(n_antennas))
    return clusters


def station_beamform_pulse_finder(
        efield_list, times_list, positions_2d, shower_zen, shower_az,
        scan_range_deg=DEFAULT_SCAN_RANGE_DEG, step_deg=DEFAULT_STEP_DEG,
        sigma_ns=DEFAULT_SMOOTHING_NS):
    """Find a station pulse time with clustered incoherent envelope beamforming.

    Returns (best_time_nuradio, best_snr, best_direction), where the time is
    in NuRadioReco internal units and the direction is (azimuth, zenith).
    """
    if len(efield_list) < 2:
        return None, 0.0, (shower_az, shower_zen)

    ref_times = np.asarray(times_list[0])
    if len(ref_times) < 2:
        return None, 0.0, (shower_az, shower_zen)
    dt = ref_times[1] - ref_times[0]
    n_samples = len(ref_times)
    sigma_samples = sigma_ns * units.ns / dt

    envelopes = []
    valid_indices = []
    for index, efield in enumerate(efield_list):
        trace = np.asarray(efield)
        trace_power = np.sum(trace ** 2, axis=0) if trace.ndim > 1 else trace ** 2
        if len(trace_power) != n_samples:
            continue
        envelopes.append(gaussian_filter1d(np.sqrt(trace_power), sigma=sigma_samples))
        valid_indices.append(index)

    if len(valid_indices) < 2:
        return None, 0.0, (shower_az, shower_zen)

    envelopes = np.asarray(envelopes)
    positions = np.asarray(positions_2d)[valid_indices]
    clusters = get_spatial_clusters(positions, n_neighbors=8, overlap_stride=4)

    az_grid = shower_az + np.deg2rad(np.arange(-scan_range_deg, scan_range_deg + 0.1, step_deg))
    zen_grid = shower_zen + np.deg2rad(np.arange(-scan_range_deg, scan_range_deg + 0.1, step_deg))

    best_snr = 0.0
    best_time_nuradio = None
    best_zen, best_az = shower_zen, shower_az
    rel_pos_x = positions[:, 0] - np.mean(positions[:, 0])
    rel_pos_y = positions[:, 1] - np.mean(positions[:, 1])

    # Precompute all per-antenna delays for every grid direction in one vectorised batch.
    # kx/ky: (n_zen, n_az); delays_all: (n_zen, n_az, n_ant)
    kx_grid = np.outer(np.sin(zen_grid), np.cos(az_grid))
    ky_grid = np.outer(np.sin(zen_grid), np.sin(az_grid))
    delays_all = (1.0 / C_LIGHT_NURADIO) * (
        kx_grid[:, :, np.newaxis] * rel_pos_x[np.newaxis, np.newaxis, :]
        + ky_grid[:, :, np.newaxis] * rel_pos_y[np.newaxis, np.newaxis, :]
    )
    delay_samples_all = (delays_all / dt).astype(np.int32)  # (n_zen, n_az, n_ant)

    for i_zen, zenith in enumerate(zen_grid):
        for i_az, azimuth in enumerate(az_grid):
            delay_samples = delay_samples_all[i_zen, i_az]  # (n_ant,)

            for cluster_indices in clusters:
                if len(cluster_indices) < 3:
                    continue
                beamformed_sum = np.zeros(n_samples)
                for cluster_index in cluster_indices:
                    shift = -delay_samples[cluster_index]
                    shifted_trace = np.roll(envelopes[cluster_index], shift)
                    if shift > 0:
                        shifted_trace[:shift] = 0
                    elif shift < 0:
                        shifted_trace[shift:] = 0
                    beamformed_sum += shifted_trace
                beamformed_sum /= len(cluster_indices)

                peak_idx = np.argmax(beamformed_sum)
                noise_section = (
                    beamformed_sum[:n_samples // 4]
                    if peak_idx > n_samples // 2 else beamformed_sum[-n_samples // 4:]
                )
                noise_std = np.std(noise_section)
                current_snr = 0.0 if noise_std < 1e-12 else (
                    beamformed_sum[peak_idx] - np.mean(noise_section)) / noise_std
                if current_snr > best_snr:
                    best_snr = current_snr
                    best_time_nuradio = ref_times[peak_idx]
                    best_zen, best_az = zenith, azimuth

    return best_time_nuradio, best_snr, (best_az, best_zen)


def simple_hilbert_finder(voltage_traces, voltage_times):
    power_trace = np.sum(np.asarray(voltage_traces) ** 2, axis=0)
    if len(power_trace) < 10:
        return None, 0.0
    envelope = np.abs(hilbert(power_trace))
    q25, _, q75 = np.percentile(envelope, [25, 50, 75])
    noise_est = q25 if q75 > 2 * q25 else np.median(envelope)
    peak_idx = np.argmax(envelope)
    snr = envelope[peak_idx] / noise_est if noise_est > 1e-12 else float("inf")
    return voltage_times[peak_idx], snr


def power_peak_finder(voltage_traces, voltage_times):
    power_trace = np.sum(np.asarray(voltage_traces) ** 2, axis=0)
    if len(power_trace) < 10:
        return None, 0.0
    smoothed_power = gaussian_filter1d(power_trace, sigma=5)
    peak_idx = np.argmax(smoothed_power)
    noise_est = np.std(power_trace[:len(power_trace) // 5]) if len(power_trace) > 5 else np.std(power_trace)
    snr = smoothed_power[peak_idx] / noise_est if noise_est > 1e-12 else float("inf")
    return voltage_times[peak_idx], snr


_TEMPLATE_LIBRARY = None
TEMPLATE_CORR_THRESHOLD = 0.05


def set_template_library(path):
    """Load the template library (.npz) for the template-match timing fallback."""
    global _TEMPLATE_LIBRARY
    _TEMPLATE_LIBRARY = None
    if path and os.path.exists(path):
        try:
            with np.load(path) as data:
                _TEMPLATE_LIBRARY = {k: data[k] for k in data}
            logger.info("Loaded timing template library from %s (%d arrays).", path, len(_TEMPLATE_LIBRARY))
        except Exception as exc:
            logger.warning("Could not load template library %s: %s", path, exc)
    elif path:
        logger.warning("Template library not found at %s; template-match timing disabled.", path)


def template_match_finder(voltage_traces, voltage_times, template_library):
    """Cross-correlate two polarisations against a template library; return (peak_time_ns, score)."""
    if not template_library or not isinstance(template_library, dict):
        return None, 0.0
    data_pol1, data_pol2 = voltage_traces[0], voltage_traces[1]
    std1, std2 = np.std(data_pol1), np.std(data_pol2)
    if std1 < 1e-9 or std2 < 1e-9:
        return None, 0.0
    data_pol1_norm = (data_pol1 - np.mean(data_pol1)) / (std1 * np.linalg.norm(data_pol1))
    data_pol2_norm = (data_pol2 - np.mean(data_pol2)) / (std2 * np.linalg.norm(data_pol2))
    best_corr_val, best_lag, best_name = -np.inf, -1, None
    template_names = sorted(set(k.split('_template')[0] for k in template_library))
    for name in template_names:
        try:
            tmpl1 = template_library[f'{name}_template_pol1']
            tmpl2 = template_library[f'{name}_template_pol2']
        except KeyError:
            continue
        if np.std(tmpl1) < 1e-9 or np.std(tmpl2) < 1e-9:
            continue
        tmpl1_n = (tmpl1 - np.mean(tmpl1)) / (np.std(tmpl1) * np.linalg.norm(tmpl1))
        tmpl2_n = (tmpl2 - np.mean(tmpl2)) / (np.std(tmpl2) * np.linalg.norm(tmpl2))
        if len(tmpl1_n) > len(data_pol1_norm):
            continue
        combined = correlate(data_pol1_norm, tmpl1_n, mode='valid', method='fft') + \
            correlate(data_pol2_norm, tmpl2_n, mode='valid', method='fft')
        if len(combined) == 0:
            continue
        peak_val = np.max(combined)
        if peak_val > best_corr_val:
            best_corr_val, best_lag, best_name = peak_val, int(np.argmax(combined)), name
    if best_lag == -1:
        return None, 0.0
    score = best_corr_val / (len(template_library[f'{best_name}_template_pol1']) * 2)
    tmpl_power = (template_library[f'{best_name}_template_pol1'] ** 2
                  + template_library[f'{best_name}_template_pol2'] ** 2)
    peak_index = best_lag + int(np.argmax(tmpl_power))
    if score >= TEMPLATE_CORR_THRESHOLD and peak_index < len(voltage_times):
        return voltage_times[peak_index], score
    return None, score


def _neighbor_timing_outlier_test(positions, times, point_index, n_neighbors=OUTLIER_NEIGHBORS_K):
    n_points = times.size
    if n_points <= 1:
        return 0.0, np.inf, False
    k_for_point = min(n_neighbors, n_points - 1)
    if k_for_point < MIN_ABSOLUTE_NEIGHBORS:
        return 0.0, np.inf, False
    dist_matrix = cdist(positions.T, positions.T)
    neighbor_indices = np.argsort(dist_matrix[point_index, :])[1:k_for_point + 1]
    neighbor_times_ns = times[neighbor_indices] * units.s / units.ns
    neighbor_median_ns = np.median(neighbor_times_ns)
    diff_ns = np.abs(times[point_index] * units.s / units.ns - neighbor_median_ns)
    neighbor_residuals_ns = neighbor_times_ns - neighbor_median_ns
    local_sigma_ns = 1.4826 * np.median(np.abs(neighbor_residuals_ns))
    if local_sigma_ns <= 0 or not np.isfinite(local_sigma_ns):
        local_sigma_ns = np.std(neighbor_residuals_ns)
    if local_sigma_ns <= 0 or not np.isfinite(local_sigma_ns):
        local_sigma_ns = HIGH_SNR_TIMING_MIN_LOCAL_SIGMA_NS
    local_sigma_ns = max(local_sigma_ns, HIGH_SNR_TIMING_MIN_LOCAL_SIGMA_NS)
    is_outlier = diff_ns > HIGH_SNR_TIMING_OUTLIER_SIGMA * local_sigma_ns
    return diff_ns, local_sigma_ns, is_outlier


def _high_snr_timing_point_can_be_demoted(global_index, positions, times, candidate_mask, station_ids, timing_snrs):
    if timing_snrs is None or timing_snrs.size == 0:
        return True, None, None
    snr = timing_snrs[global_index]
    if not np.isfinite(snr) or snr < HIGH_SNR_TIMING_PROTECTION_SNR:
        return True, None, None
    local_mask = candidate_mask.copy()
    if station_ids is not None:
        station_mask = local_mask & (station_ids == station_ids[global_index])
        if np.sum(station_mask) > MIN_ABSOLUTE_NEIGHBORS:
            local_mask = station_mask
    local_indices = np.where(local_mask)[0]
    if local_indices.size <= MIN_ABSOLUTE_NEIGHBORS:
        return False, 0.0, np.inf
    local_matches = np.where(local_indices == global_index)[0]
    if local_matches.size == 0:
        return False, 0.0, np.inf
    diff_ns, sigma_ns, is_severe_outlier = _neighbor_timing_outlier_test(
        positions[:, local_indices], times[local_indices], int(local_matches[0]))
    return is_severe_outlier, diff_ns, sigma_ns


def detect_timing_outliers(positions, times, station_ids, timing_snrs=None):
    """Demote timing points that disagree with nearest neighbors inside each station.

    High-SNR points (>= HIGH_SNR_TIMING_PROTECTION_SNR) are only removed if they
    are severe outliers in a stricter local test (timing_snrs controls this).
    """
    positions = np.asarray(positions)
    times = np.asarray(times)
    station_ids = np.asarray(station_ids)
    keep_mask = np.ones(times.size, dtype=bool)

    for station_id in np.unique(station_ids):
        station_indices = np.where(station_ids == station_id)[0]
        if len(station_indices) <= 1:
            continue
        station_positions = positions[:, station_indices]
        station_times = times[station_indices]
        k_for_station = min(OUTLIER_NEIGHBORS_K, len(station_indices) - 1)
        if k_for_station < MIN_ABSOLUTE_NEIGHBORS:
            continue
        time_std_ns = np.std((station_times - np.mean(station_times)) * units.s / units.ns)
        threshold_ns = max(OUTLIER_STD_FACTOR * time_std_ns, 15.0)
        dist_matrix = cdist(station_positions.T, station_positions.T)
        station_is_outlier = np.zeros(len(station_indices), dtype=bool)
        for index in range(len(station_indices)):
            neighbor_indices = np.argsort(dist_matrix[index, :])[1:k_for_station + 1]
            diff_ns = np.abs(
                (station_times[index] - np.median(station_times[neighbor_indices]))
                * units.s / units.ns
            )
            is_outlier = diff_ns > threshold_ns
            if is_outlier and timing_snrs is not None:
                global_idx = station_indices[index]
                point_snr = timing_snrs[global_idx]
                if np.isfinite(point_snr) and point_snr >= HIGH_SNR_TIMING_PROTECTION_SNR:
                    _, _, is_outlier = _neighbor_timing_outlier_test(station_positions, station_times, index)
            if is_outlier:
                station_is_outlier[index] = True
        keep_mask[station_indices] = ~station_is_outlier

    logger.info("Identified %d timing outliers for demotion.", times.size - np.sum(keep_mask))
    return keep_mask


def iterative_timing_pruning(shower_direction, positions, times, initial_mask,
                             station_ids=None, timing_snrs=None):
    """Iteratively prune timing residual outliers with a local polynomial time model.

    High-SNR points are skipped unless they are severe outliers in a local neighbor test.
    """
    del shower_direction
    current_mask = np.asarray(initial_mask, dtype=bool).copy()
    max_removals = max(int(np.sum(initial_mask)) - MIN_TIMING_POINTS + 1, 0)
    logger.info("Starting iterative timing pruning. Initial points: %d", int(np.sum(current_mask)))
    protected_skips = 0

    for _ in range(max_removals):
        active_indices = np.where(current_mask)[0]
        if len(active_indices) < MIN_TIMING_POINTS:
            break
        result = _fit_timing_polynomial(positions[:, active_indices], times[active_indices])
        residuals = result["residuals"]
        std_residual = np.std(residuals)
        if std_residual < TIMING_UNCERT_THRESHOLD:
            logger.info("Timing pruning converged. Std: %.2f ns. Points remaining: %d",
                        std_residual * units.s / units.ns, len(active_indices))
            return (current_mask, std_residual, result["coeffs"], result["mean_pos_2d"],
                    result["scale"], result["inv_AtA"])
        removed_point = False
        for worst_local_idx in np.argsort(np.abs(residuals))[::-1]:
            worst_global_idx = active_indices[worst_local_idx]
            can_demote, _, _ = _high_snr_timing_point_can_be_demoted(
                worst_global_idx, positions, times, current_mask, station_ids, timing_snrs)
            if can_demote:
                current_mask[worst_global_idx] = False
                removed_point = True
                break
            protected_skips += 1
        if not removed_point:
            logger.warning("Timing pruning stopped: all remaining candidates are protected high-SNR points.")
            break

    if protected_skips > 0:
        logger.info("High-SNR timing protection skipped %d global pruning removals.", protected_skips)

    active_indices = np.where(current_mask)[0]
    if len(active_indices) >= MIN_TIMING_POINTS:
        result = _fit_timing_polynomial(positions[:, active_indices], times[active_indices])
        return (current_mask, np.std(result["residuals"]), result["coeffs"], result["mean_pos_2d"],
                result["scale"], result["inv_AtA"])
    return current_mask, 15.0 * units.ns / units.s, None, None, 1.0, None


def _fit_timing_polynomial(active_pos, active_times):
    mean_pos_2d = np.mean(active_pos[:2, :], axis=1)
    x = active_pos[0, :] - mean_pos_2d[0]
    y = active_pos[1, :] - mean_pos_2d[1]
    scale = np.std(np.sqrt(x ** 2 + y ** 2))
    if scale < 1e-6:
        scale = 1.0
    x_norm = x / scale
    y_norm = y / scale
    matrix = np.column_stack([np.ones(len(active_times)), x_norm, y_norm, x_norm ** 2 + y_norm ** 2])
    coeffs, _, _, _ = np.linalg.lstsq(matrix, active_times, rcond=None)
    try:
        inv_AtA = np.linalg.inv(matrix.T @ matrix)
    except np.linalg.LinAlgError:
        inv_AtA = None
    return {
        "coeffs": coeffs,
        "residuals": active_times - matrix @ coeffs,
        "mean_pos_2d": mean_pos_2d,
        "scale": scale,
        "inv_AtA": inv_AtA,
    }


def get_local_timing_uncertainties(
        positions, times, station_ids, shower_direction,
        min_uncertainty_s=1.5 * units.ns / units.s, cluster_size=20):
    """Estimate per-antenna timing uncertainty from local nearest-neighbor residuals."""
    del station_ids, shower_direction
    from scipy.spatial import KDTree

    positions = np.asarray(positions)
    times = np.asarray(times)
    n_antennas = positions.shape[1]
    uncertainties = np.zeros(n_antennas)
    pos_2d = positions[:2, :].T
    tree = KDTree(pos_2d)

    for index in range(n_antennas):
        _, neighbor_indices = tree.query(pos_2d[index], k=min(cluster_size, n_antennas))
        if np.size(neighbor_indices) < 8:
            uncertainties[index] = 5.0 * units.ns / units.s
            continue
        result = _fit_timing_polynomial(positions[:, neighbor_indices], times[neighbor_indices])
        uncertainties[index] = max(np.std(result["residuals"]), min_uncertainty_s)
    return uncertainties


def extract_fluence_and_timing_data(
        event, station_id, detector, shower_direction, fluence_window_ns=None,
        global_ref_time=None, global_ref_pos=None,
        scan_range_deg=DEFAULT_SCAN_RANGE_DEG, step_deg=DEFAULT_STEP_DEG,
        sigma_ns=DEFAULT_SMOOTHING_NS, skip_lba_inner=False):
    """Extract antenna fluences and pulse times for one station.

    Each antenna's region of interest is centred on the plane-wave-propagated
    global reference pulse (``global_ref_time``), or on the station's beamformed
    pulse time when no global reference is available, with a fixed +/- 500 ns
    window. Antennas for which no pulse is found are simply not reported.

    Returns: posx, posy, fluences, max_times, is_signal, noise_fluences, snr_values, best_direction
    """
    if fluence_window_ns is None:
        fluence_window_ns = DEFAULT_FLUENCE_WINDOW_NS

    posx, posy, fluences, max_times, is_signal, noise_fluences, snr_values = [], [], [], [], [], [], []
    best_direction = shower_direction

    try:
        station = event.get_station(station_id)
    except ValueError:
        return (*_observable_arrays(posx, posy, fluences, max_times, is_signal, noise_fluences, snr_values),
                shower_direction)

    station_abs_pos = detector.get_absolute_position(station_id)

    # Trace times and ROIs use NuRadioReco's internal time unit. The IFT
    # likelihood, by contrast, receives SI-second timing observables below.
    fluence_window_nuradio = fluence_window_ns * units.ns

    # 1. Station-level beamforming to find best direction and ROI centre
    beamformed_time_nuradio = None
    try:
        bf_traces, bf_times_list, bf_pos_list = [], [], []
        for ch in station.iter_channels():
            ch_id = ch.get_id()
            if ch_id % 2 != 0:
                continue
            if skip_lba_inner:
                try:
                    if detector.get_antenna_mode(station_id, ch_id) == "LBA_inner":
                        continue
                except Exception:
                    pass
            try:
                ch_odd = station.get_channel(ch_id + 1)
                bf_traces.append(np.array([ch.get_trace(), ch_odd.get_trace()]))
                bf_times_list.append(ch.get_times())
                bf_pos_list.append((detector.get_relative_position(station_id, ch_id) + station_abs_pos)[:2])
            except Exception:
                continue
        if bf_traces:
            az_guess, zen_guess = shower_direction
            bf_t, _, bf_dir = station_beamform_pulse_finder(
                bf_traces, bf_times_list, np.array(bf_pos_list), zen_guess, az_guess,
                scan_range_deg=scan_range_deg, step_deg=step_deg, sigma_ns=sigma_ns)
            if bf_t is not None:
                beamformed_time_nuradio = bf_t
                best_direction = bf_dir
    except Exception as exc:
        logger.debug("BF failed for station %s: %s", station_id, exc)

    az, zen = shower_direction
    kx = np.sin(zen) * np.cos(az)
    ky = np.sin(zen) * np.sin(az)
    ref_delay_nuradio = (
        -(1.0 / C_LIGHT_NURADIO) * (kx * global_ref_pos[0] + ky * global_ref_pos[1])
        if global_ref_pos is not None else 0.0
    )

    # 2. Per-antenna processing
    antenna_data = []
    for ef in station.get_electric_fields():
        try:
            ch_ids = ef.get_channel_ids()
            if len(ch_ids) < 2:
                continue
            ch_id_even = ch_ids[0]
            if skip_lba_inner:
                try:
                    if detector.get_antenna_mode(station_id, ch_id_even) == "LBA_inner":
                        continue
                except Exception:
                    pass
            position = detector.get_relative_position(station_id, ch_id_even) + station_abs_pos
            try:
                ch_even_obj = station.get_channel(ch_id_even)
                ch_odd_obj = station.get_channel(ch_id_even + 1)
                full_voltage_traces = np.array([ch_even_obj.get_trace(), ch_odd_obj.get_trace()])
                full_voltage_times = ch_even_obj.get_times()
            except Exception:
                continue

            ant_x, ant_y = position[0], position[1]
            ant_delay_nuradio = -(1.0 / C_LIGHT_NURADIO) * (kx * ant_x + ky * ant_y)
            if global_ref_time is not None:
                ant_roi_center = global_ref_time + (ant_delay_nuradio - ref_delay_nuradio)
            else:
                ant_roi_center = beamformed_time_nuradio

            if ant_roi_center is not None:
                ant_roi_start = ant_roi_center - 500.0 * units.ns
                ant_roi_end = ant_roi_center + 500.0 * units.ns
            else:
                ant_roi_start = ROI_START_NS * units.ns
                ant_roi_end = ROI_END_NS * units.ns

            time_mask = (full_voltage_times >= ant_roi_start) & (full_voltage_times <= ant_roi_end)
            if not np.any(time_mask):
                continue
            cut_v_traces = full_voltage_traces[:, time_mask]
            cut_v_times = full_voltage_times[time_mask]

            efield_trace = ef.get_trace()
            efield_times = ef.get_times()

            max_time_nuradio = None
            current_max_snr = 0.0
            h_time, h_snr = simple_hilbert_finder(cut_v_traces, cut_v_times)
            p_time, p_snr = power_peak_finder(cut_v_traces, cut_v_times)
            current_max_snr = max(h_snr, p_snr)
            if h_time is not None and h_snr >= SNR_CUT:
                max_time_nuradio = h_time
            elif p_time is not None and p_snr >= SNR_CUT:
                max_time_nuradio = p_time
            else:
                t_time, t_score = template_match_finder(cut_v_traces, cut_v_times, _TEMPLATE_LIBRARY)
                if t_time is not None and t_score >= TEMPLATE_CORR_THRESHOLD:
                    max_time_nuradio = t_time
                    current_max_snr = max(current_max_snr, t_score)

            antenna_data.append({
                'position': position, 'max_time_nuradio': max_time_nuradio,
                'current_max_snr': current_max_snr, 'efield_trace': efield_trace,
                'efield_times': efield_times,
            })
        except Exception as exc:
            logger.warning("Could not process E-field for station %s: %s", station_id, exc)

    # Pass 2: compute fluences
    for ant in antenna_data:
        position = ant['position']
        max_time_nuradio = ant['max_time_nuradio']
        current_max_snr = ant['current_max_snr']
        efield_trace = ant['efield_trace']
        efield_times = ant['efield_times']

        if max_time_nuradio is not None:
            sig_mask = ((efield_times >= max_time_nuradio - fluence_window_nuradio / 2.0)
                        & (efield_times <= max_time_nuradio + fluence_window_nuradio / 2.0))
            fl = (np.sum(get_electric_field_energy_fluence(efield_trace[:, sig_mask], efield_times[sig_mask]))
                  if np.any(sig_mask) else 0.0)
            posx.append(position[0])
            posy.append(position[1])
            fluences.append(fl)
            max_times.append(max_time_nuradio / units.s)
            is_signal.append(True)
            snr_values.append(current_max_snr)
            noise_fluences.extend(_noise_fluence_windows(
                efield_trace, efield_times, max_time_nuradio, fluence_window_ns
            ))

    return (*_observable_arrays(posx, posy, fluences, max_times, is_signal, noise_fluences, snr_values),
            best_direction)


def _station_roi_from_beamforming(station, detector, station_id, station_abs_pos, shower_direction):
    bf_traces, bf_times_list, bf_pos_list = [], [], []
    for channel in station.iter_channels():
        channel_id = channel.get_id()
        if channel_id % 2 != 0:
            continue
        try:
            channel_odd = station.get_channel(channel_id + 1)
            if detector.get_antenna_mode(station_id, channel_id) == "LBA_inner":
                continue
            bf_traces.append(np.array([channel.get_trace(), channel_odd.get_trace()]))
            bf_times_list.append(channel.get_times())
            bf_pos_list.append((detector.get_relative_position(station_id, channel_id) + station_abs_pos)[:2])
        except Exception:
            continue

    if not bf_traces:
        return ROI_START_NS * units.ns, ROI_END_NS * units.ns

    azimuth_guess, zenith_guess = shower_direction
    beamformed_time_nuradio, beamformed_snr, _ = station_beamform_pulse_finder(
        bf_traces, bf_times_list, np.array(bf_pos_list), zenith_guess, azimuth_guess)
    if beamformed_time_nuradio is None or beamformed_snr <= 3.0:
        return ROI_START_NS * units.ns, ROI_END_NS * units.ns

    positions = np.array(bf_pos_list)
    station_span = np.max(np.linalg.norm(positions - np.mean(positions, axis=0), axis=1)) * 2
    half_window_nuradio = max(
        100.0 * units.ns,
        station_span * np.sin(zenith_guess) / C_LIGHT_NURADIO / 2 + 50.0 * units.ns,
    )
    return (beamformed_time_nuradio - half_window_nuradio,
            beamformed_time_nuradio + half_window_nuradio)


_BANDPASS_EDGE_SAMPLES_START = 20000
_BANDPASS_EDGE_SAMPLES_END = 45536


def _noise_fluence_windows(efield_trace, efield_times, max_time_nuradio, fluence_window_ns):
    if len(efield_times) < 2:
        return []
    time_step_nuradio = efield_times[1] - efield_times[0]
    if time_step_nuradio <= 0:
        return []
    fluence_window_nuradio = fluence_window_ns * units.ns
    window_length_samples = int(fluence_window_nuradio / time_step_nuradio)
    if window_length_samples <= 0:
        return []

    n_total = len(efield_times)
    edge_end = min(_BANDPASS_EDGE_SAMPLES_END, n_total)

    noise_fluences = []
    for start_index in range(0, n_total, window_length_samples):
        # Skip bandpass roll-off edges where filter suppresses power
        if start_index < _BANDPASS_EDGE_SAMPLES_START:
            continue
        end_index = start_index + window_length_samples
        if end_index > edge_end:
            continue
        window_start_time = efield_times[start_index]
        in_signal = not (
            window_start_time + fluence_window_nuradio < max_time_nuradio - fluence_window_nuradio / 2.0
            or window_start_time > max_time_nuradio + fluence_window_nuradio / 2.0
        )
        if in_signal:
            continue
        slice_trace = efield_trace[:, start_index:end_index]
        slice_times = efield_times[start_index:end_index]
        if len(slice_times) == window_length_samples:
            noise_fluences.append(np.sum(get_electric_field_energy_fluence(slice_trace, slice_times)))
    return noise_fluences


def blind_max_signal_finder(voltage_traces, voltage_times,
                            edge_fraction=MAX_SIGNAL_EDGE_FRACTION):
    """Find the largest envelope excursion anywhere in an antenna's trace.

    Unlike :func:`simple_hilbert_finder` and :func:`power_peak_finder`, which need
    a region of interest, this searches the whole trace and normalises with a
    robust median/MAD estimator, so the SNR is not spoiled by the pulse itself.

    Returns ``(max_time_nuradio, snr)``, or ``(None, 0.0)`` for an unusable trace.
    The SNR is ``(peak - median) / (1.4826 * MAD)``; see
    :data:`MAX_SIGNAL_SNR_THRESHOLD` for calibrated values.
    """
    traces = np.asarray(voltage_traces, dtype=float)
    times = np.asarray(voltage_times, dtype=float)
    if traces.ndim == 1:
        traces = traces[np.newaxis, :]
    n_samples = traces.shape[-1]
    if n_samples < 64 or len(times) != n_samples:
        return None, 0.0

    edge = int(edge_fraction * n_samples)
    lo, hi = edge, n_samples - edge
    if hi - lo < 64:
        lo, hi = 0, n_samples

    analytic = hilbert(traces[:, lo:hi], axis=-1)
    envelope = np.sqrt(np.sum(np.abs(analytic) ** 2, axis=0))

    median = np.median(envelope)
    mad = np.median(np.abs(envelope - median)) * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        return None, 0.0

    peak_index = int(np.argmax(envelope))
    snr = float((envelope[peak_index] - median) / mad)
    return float(times[lo + peak_index]), snr


def plane_wave_consensus_mask(posx, posy, times, shower_direction,
                              window_ns=MAX_SIGNAL_CONSENSUS_WINDOW_NS):
    """Keep only arrival times that share one plane-wave arrival plane.

    Subtracts the geometric delay for `shower_direction` (given as
    ``(azimuth, zenith)`` in radians) from every arrival time and keeps the
    antennas in the densest ``2 * window_ns`` interval of the residuals. Real
    pulses pile up in one interval; blind-search noise peaks do not.

    Positions are in internal length units, `times` in seconds. Returns a boolean
    mask over the input.
    """
    posx = np.asarray(posx, dtype=float)
    posy = np.asarray(posy, dtype=float)
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return np.zeros(0, dtype=bool)

    azimuth, zenith = shower_direction
    kx = np.sin(zenith) * np.cos(azimuth)
    ky = np.sin(zenith) * np.sin(azimuth)
    # Same convention as the causality cut in iftReconstructor: positions in
    # internal metres, arrival times in SI seconds.
    geometric_delay = -(1.0 / scipy.constants.c) * (kx * posx + ky * posy)
    residual = times - geometric_delay

    half_window = window_ns * units.ns / units.s
    # Densest interval: for each candidate centre count the residuals within it.
    counts = np.array([
        np.sum(np.abs(residual - centre) <= half_window) for centre in residual
    ])
    best_centre = residual[int(np.argmax(counts))]
    return np.abs(residual - best_centre) <= half_window


def extract_max_signal_timing_data(
        event, station_id, detector, shower_direction, fluence_window_ns=None,
        snr_threshold=MAX_SIGNAL_SNR_THRESHOLD, skip_lba_inner=False):
    """Blind per-antenna maximum-signal extraction, for events with no beamformer lock.

    A fallback with the same return signature as
    :func:`extract_fluence_and_timing_data`, for the faintest events where the
    normal extraction returns nothing. It uses no region of interest and no
    beamformer: every antenna is searched over its whole trace
    (:func:`blind_max_signal_finder`), antennas below `snr_threshold` are dropped,
    and the survivors must agree on one arrival plane
    (:func:`plane_wave_consensus_mask`). Fewer than
    :data:`MAX_SIGNAL_MIN_CONSENSUS_ANTENNAS` agreeing gives an empty result.
    """
    if fluence_window_ns is None:
        fluence_window_ns = DEFAULT_FLUENCE_WINDOW_NS
    fluence_window_nuradio = fluence_window_ns * units.ns

    posx, posy, fluences, max_times, is_signal, noise_fluences, snr_values = \
        [], [], [], [], [], [], []
    empty = (*_observable_arrays(posx, posy, fluences, max_times, is_signal,
                                 noise_fluences, snr_values), shower_direction)

    try:
        station = event.get_station(station_id)
    except ValueError:
        return empty

    station_abs_pos = detector.get_absolute_position(station_id)

    candidates = []
    for efield in station.get_electric_fields():
        channel_ids = efield.get_channel_ids()
        if len(channel_ids) < 2:
            continue
        channel_id_even = channel_ids[0]
        if skip_lba_inner:
            try:
                if detector.get_antenna_mode(station_id, channel_id_even) == "LBA_inner":
                    continue
            except Exception:
                pass
        try:
            channel_even = station.get_channel(channel_id_even)
            channel_odd = station.get_channel(channel_id_even + 1)
            voltage_traces = np.array([channel_even.get_trace(), channel_odd.get_trace()])
            voltage_times = channel_even.get_times()
        except Exception:
            continue

        max_time_nuradio, snr = blind_max_signal_finder(voltage_traces, voltage_times)
        if max_time_nuradio is None or snr < snr_threshold:
            continue

        position = detector.get_relative_position(station_id, channel_id_even) + station_abs_pos
        candidates.append({
            'position': position,
            'max_time_nuradio': max_time_nuradio,
            'snr': snr,
            'efield_trace': efield.get_trace(),
            'efield_times': efield.get_times(),
        })

    if len(candidates) < MAX_SIGNAL_MIN_CONSENSUS_ANTENNAS:
        logger.info(
            "Station %s: blind max-signal search found %d antenna(s) above SNR %.1f, "
            "need %d for a consensus — treating as no signal.",
            station_id, len(candidates), snr_threshold,
            MAX_SIGNAL_MIN_CONSENSUS_ANTENNAS)
        return empty

    keep = plane_wave_consensus_mask(
        [c['position'][0] for c in candidates],
        [c['position'][1] for c in candidates],
        [c['max_time_nuradio'] / units.s for c in candidates],
        shower_direction,
    )
    if int(np.sum(keep)) < MAX_SIGNAL_MIN_CONSENSUS_ANTENNAS:
        logger.info(
            "Station %s: blind max-signal search — only %d of %d antennas share an "
            "arrival plane, need %d — treating as no signal.",
            station_id, int(np.sum(keep)), len(candidates),
            MAX_SIGNAL_MIN_CONSENSUS_ANTENNAS)
        return empty

    for candidate, in_plane in zip(candidates, keep):
        if not in_plane:
            continue
        max_time_nuradio = candidate['max_time_nuradio']
        efield_trace = candidate['efield_trace']
        efield_times = candidate['efield_times']

        signal_mask = ((efield_times >= max_time_nuradio - fluence_window_nuradio / 2.0)
                       & (efield_times <= max_time_nuradio + fluence_window_nuradio / 2.0))
        fluence = (np.sum(get_electric_field_energy_fluence(
            efield_trace[:, signal_mask], efield_times[signal_mask]))
            if np.any(signal_mask) else 0.0)

        posx.append(candidate['position'][0])
        posy.append(candidate['position'][1])
        fluences.append(fluence)
        max_times.append(max_time_nuradio / units.s)
        is_signal.append(True)
        snr_values.append(candidate['snr'])
        noise_fluences.extend(_noise_fluence_windows(
            efield_trace, efield_times, max_time_nuradio, fluence_window_ns))

    logger.info(
        "Station %s: blind max-signal search kept %d antenna(s) (of %d above SNR %.1f).",
        station_id, len(fluences), len(candidates), snr_threshold)

    return (*_observable_arrays(posx, posy, fluences, max_times, is_signal,
                                noise_fluences, snr_values), shower_direction)


def augment_neighbor_fluences(
        event, detector, station_ids_to_check, reference_positions_x, reference_positions_y,
        reference_t0, shower_direction, fluence_window_ns, known_positions_set,
        skip_lba_inner=False, max_neighbor_dist_m=100.0):
    """Conservative outer-LDF augmentation with fluence-only antennas near existing signal antennas.

    Antennas within max_neighbor_dist_m of a signal antenna but not yet extracted are added
    with fluence integrated at the plane-wave-predicted arrival time. Mutates known_positions_set.
    """
    new_data = {'posx': [], 'posy': [], 'fluences': [], 'times': [],
                'station_ids': [], 'is_signal': [], 'snrs': []}
    if len(reference_positions_x) == 0:
        return new_data

    ref_pos_arr = np.array([reference_positions_x, reference_positions_y]).T
    az, zen = shower_direction
    kx = np.sin(zen) * np.cos(az)
    ky = np.sin(zen) * np.sin(az)

    for station_id in station_ids_to_check:
        try:
            if station_id not in event.get_station_ids():
                continue
            station = event.get_station(station_id)
            station_abs_pos = detector.get_absolute_position(station_id)
        except Exception:
            continue
        for ef in station.get_electric_fields():
            try:
                ch_ids = ef.get_channel_ids()
                if len(ch_ids) < 2:
                    continue
                ch_even = ch_ids[0]
                if skip_lba_inner:
                    try:
                        if detector.get_antenna_mode(station_id, ch_even) == "LBA_inner":
                            continue
                    except Exception:
                        pass
                abs_pos = detector.get_relative_position(station_id, ch_even) + station_abs_pos
                pos_xy = abs_pos[:2]
                pos_key = (round(float(pos_xy[0]), 1), round(float(pos_xy[1]), 1))
                if pos_key in known_positions_set:
                    continue
                if np.min(cdist([pos_xy], ref_pos_arr)[0]) > max_neighbor_dist_m:
                    continue

                geom_delay_s = -(1.0 / scipy.constants.c) * (kx * pos_xy[0] + ky * pos_xy[1])
                expected_time_s = reference_t0 + geom_delay_s
                efield_trace = ef.get_trace()
                efield_times = ef.get_times()
                expected_time_nuradio = expected_time_s * units.s
                fluence_window_nuradio = fluence_window_ns * units.ns
                mask = ((efield_times >= expected_time_nuradio - fluence_window_nuradio / 2.0)
                        & (efield_times <= expected_time_nuradio + fluence_window_nuradio / 2.0))
                fl = (np.sum(get_electric_field_energy_fluence(efield_trace[:, mask], efield_times[mask]))
                      if np.sum(mask) >= 2 else 0.0)
                new_data['posx'].append(float(pos_xy[0]))
                new_data['posy'].append(float(pos_xy[1]))
                new_data['fluences'].append(float(fl))
                new_data['times'].append(float(expected_time_s))
                new_data['station_ids'].append(int(station_id))
                new_data['is_signal'].append(False)
                new_data['snrs'].append(0.0)
                known_positions_set.add(pos_key)
            except Exception:
                continue
    return new_data


def extract_neighbor_fluences(
        event, detector, station_ids_to_check, reference_positions_x, reference_positions_y,
        reference_t0, shower_direction, fluence_window_ns, known_positions_mask_set):
    """Deprecated alias for augment_neighbor_fluences."""
    result = augment_neighbor_fluences(
        event, detector, station_ids_to_check, reference_positions_x, reference_positions_y,
        reference_t0, shower_direction, fluence_window_ns, known_positions_mask_set)
    return defaultdict(list, result)


def angular_distance(zen1, az1, zen2, az2):
    """Great-circle angular distance between two directions (radians)."""
    n1 = [np.sin(zen1) * np.cos(az1), np.sin(zen1) * np.sin(az1), np.cos(zen1)]
    n2 = [np.sin(zen2) * np.cos(az2), np.sin(zen2) * np.sin(az2), np.cos(zen2)]
    return np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))


def fit_global_plane_wave(positions, times):
    """Least-squares plane-wave timing helper returning zenith and azimuth."""
    positions = np.asarray(positions)
    times = np.asarray(times)
    matrix = np.column_stack([positions[0], positions[1], np.ones(times.size)])
    coeffs, _, _, _ = np.linalg.lstsq(matrix, times, rcond=None)
    kx = -coeffs[0] * scipy.constants.c
    ky = -coeffs[1] * scipy.constants.c
    horizontal = np.sqrt(kx ** 2 + ky ** 2)
    zenith = np.arcsin(np.clip(horizontal, 0.0, 1.0))
    azimuth = np.arctan2(ky, kx) % (2 * np.pi)
    return zenith, azimuth, coeffs


def fit_plane_wave(positions, times, weights=None):
    """Weighted least-squares plane wave fit returning (zenith, azimuth, t0)."""
    n_points = len(times)
    if n_points < 3:
        return None, None, None
    A = np.column_stack([positions[0], positions[1], np.ones(n_points)])
    if weights is not None:
        W = np.diag(weights)
        try:
            res = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ times
        except np.linalg.LinAlgError:
            return None, None, None
    else:
        res, _, _, _ = np.linalg.lstsq(A, times, rcond=None)
    a, b, t0 = res
    kx = -a * scipy.constants.c
    ky = -b * scipy.constants.c
    k_perp_sq = kx ** 2 + ky ** 2
    if k_perp_sq > 1.0:
        scale = 1.0 / np.sqrt(k_perp_sq)
        kx *= scale
        ky *= scale
        k_perp_sq = 1.0
    kz = np.sqrt(1.0 - k_perp_sq)
    return np.arccos(kz), np.arctan2(ky, kx) % (2 * np.pi), t0


def count_efield_triggered_stations(fluences, station_ids, noise_mean, reconstruction_window_ns,
                                    snr_threshold=TRIGGER_SNR_THRESHOLD,
                                    window_ns=TRIGGER_WINDOW_NS,
                                    fraction=TRIGGER_ANTENNA_FRACTION):
    """Count how many stations trigger in electric-field (fluence) space."""
    fluences = np.asarray(fluences, dtype=float)
    station_ids = np.asarray(station_ids)
    if noise_mean <= 0 or fluences.size == 0:
        return 0
    noise_win = noise_mean * (window_ns / reconstruction_window_ns)
    threshold = snr_threshold * noise_win
    n_trig = 0
    for sid in np.unique(station_ids):
        st = fluences[station_ids == sid]
        if st.size == 0:
            continue
        if (np.sum(st > threshold) / st.size) >= fraction:
            n_trig += 1
    return n_trig


def filter_samples_by_trigger(model, samples_list, station_ids, noise_mean, reconstruction_window_ns,
                              min_stations=TRIGGER_MIN_STATIONS, snr_buffer_frac=TRIGGER_SNR_BUFFER_FRAC,
                              return_trigger_counts=False):
    """Retain posterior samples satisfying the LOFAR local trigger criterion.

    Returns ``(kept_indices, discarded_indices)`` — lists of integer indices
    into ``samples_list``. With ``return_trigger_counts=True``, also returns a
    per-sample array with the number of stations satisfying the criterion.
    """
    noise_55ns = noise_mean * (TRIGGER_WINDOW_NS / reconstruction_window_ns)
    threshold_fluence = TRIGGER_SNR_THRESHOLD * (1.0 - snr_buffer_frac) * noise_55ns
    logger.info(
        "Trigger filter: win=%g ns, noise(55ns)=%.2f, thresh=%.2f (buffer=%.0f%%), min_stations=%d",
        reconstruction_window_ns, noise_55ns, threshold_fluence, snr_buffer_frac * 100, min_stations)

    def _get_fluence_only(s):
        return model(s)[0]

    try:
        get_fluences = jax.vmap(_get_fluence_only)
        stacked = jtu.tree_map(lambda *parts: jnp.stack(parts), *samples_list)
        all_fluences = np.array(get_fluences(stacked))
    except Exception as exc:
        logger.warning("Vmap filtering failed (%s), falling back to loop.", exc)
        all_fluences = np.array([model(s)[0] for s in samples_list])

    unique_stations = np.unique(station_ids)
    kept_indices, discarded_indices, trigger_counts = [], [], []
    for i in range(len(samples_list)):
        sample_fluences = all_fluences[i]
        stations_triggered = 0
        for sid in unique_stations:
            st_mask = station_ids == sid
            st_fluences = sample_fluences[st_mask]
            if len(st_fluences) == 0:
                continue
            if (np.sum(st_fluences > threshold_fluence) / len(st_fluences)) >= TRIGGER_ANTENNA_FRACTION:
                stations_triggered += 1
        trigger_counts.append(stations_triggered)
        if stations_triggered >= min_stations:
            kept_indices.append(i)
        else:
            discarded_indices.append(i)
    if return_trigger_counts:
        return kept_indices, discarded_indices, np.asarray(trigger_counts, dtype=int)
    return kept_indices, discarded_indices


def make_centered_uniform_prior(center, half_width, lower=None, upper=None):
    """Return {"a_min": ..., "a_max": ...} for a UniformPrior centred on `center`."""
    a_min = float(center) - float(half_width)
    a_max = float(center) + float(half_width)
    if lower is not None:
        a_min = max(float(lower), a_min)
    if upper is not None:
        a_max = min(float(upper), a_max)
    if not a_min < a_max:
        eps = max(float(half_width), 1e-3)
        a_min = float(center) - eps
        a_max = float(center) + eps
    return {"a_min": a_min, "a_max": a_max}


def uniform_prior_latent(value, prior):
    """Convert a physical value to the latent (standard-normal) space of a UniformPrior.

    Inverse of jft.prior.UniformPrior's forward map. Use this to set VI init_values
    from a LORA/plane-wave estimate so the optimizer starts at the right point.
    """
    frac = (float(value) - float(prior["a_min"])) / (float(prior["a_max"]) - float(prior["a_min"]))
    frac = np.clip(frac, 1e-6, 1.0 - 1e-6)
    return jnp.array([_scipy_norm.ppf(frac)])


def _observable_arrays(posx, posy, fluences, max_times, is_signal, noise_fluences, snr_values):
    return (
        np.asarray(posx),
        np.asarray(posy),
        np.asarray(fluences),
        np.asarray(max_times),
        np.asarray(is_signal, dtype=bool),
        list(noise_fluences),
        np.asarray(snr_values),
    )
