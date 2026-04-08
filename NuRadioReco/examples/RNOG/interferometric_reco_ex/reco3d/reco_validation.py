"""Reconstruction validation metrics for 3D interferometric reco.

Computes per-channel SNR, surface correlation, peak isolation, and
derived per-string visibility metrics. Used by the reco driver (inline
during reconstruction) and by build_validation_df.py (post-hoc).
"""

import numpy as np
import pandas as pd

PA_CHANNELS = [0, 1, 2, 3]
HELPER_B_CHANNELS = [9, 10]
HELPER_C_CHANNELS = [22, 23]
HELPER_CHANNELS = [9, 10, 22, 23]
SHALLOW_CHANNELS = [5, 6, 7]
ALL_RECO_CHANNELS = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]

DEFAULT_HELPER_SNR_THRESHOLD = 5.0
DEFAULT_SURF_Z_MAX = -10.0
DEFAULT_SURF_ZEN_MAX = 65.0


def compute_channel_snrs(volt_arrays, channels):
    """Per-channel SNR from preprocessed voltage traces.

    Uses split-trace noise RMS (lowest segments) to avoid including
    signal in the noise estimate, and peak-to-peak amplitude within a
    coincidence window for the signal estimate. Matches the definition
    in feature_extraction/gather_variables.py.

    Args:
        volt_arrays: List of voltage trace arrays, one per channel.
        channels: List of channel IDs (same order as volt_arrays).

    Returns:
        Dict mapping channel_id to SNR value.
    """
    from NuRadioReco.utilities.trace_utilities import (
        get_split_trace_noise_RMS, get_signal_to_noise_ratio)

    snrs = {}
    for v, ch in zip(volt_arrays, channels):
        noise_rms = get_split_trace_noise_RMS(v)
        snrs[ch] = float(get_signal_to_noise_ratio(v, noise_rms)
                         if noise_rms > 0 else 0.0)
    return snrs


def compute_surf_corr(mean_corr, rho_vec, phi_vec, z_vec, pa_center_z, config):
    """Compute surface correlation metrics from the coarse grid.

    Args:
        mean_corr: 3D array of mean correlation values (rho x phi x z).
        rho_vec: 1D array of rho grid points (meters).
        phi_vec: 1D array of phi grid points (radians).
        z_vec: 1D array of z grid points (meters, absolute).
        pa_center_z: PA center z position (meters, relative to station).
        config: Dict with optional surf_corr_z_max and surf_corr_zen_max.

    Returns:
        Dict with surf_corr_z and surf_corr_zen.
    """
    result = {}

    surf_z_max = config.get('surf_corr_z_max', DEFAULT_SURF_Z_MAX)
    z_mask = z_vec >= surf_z_max
    if z_mask.any():
        result['surf_corr_z'] = float(np.nanmax(mean_corr[:, :, z_mask]))
    else:
        result['surf_corr_z'] = np.nan

    surf_zen_max = config.get('surf_corr_zen_max', DEFAULT_SURF_ZEN_MAX)
    rho_grid, _, z_grid = np.meshgrid(rho_vec, phi_vec, z_vec, indexing='ij')
    zenith_grid = np.degrees(np.arctan2(rho_grid, z_grid - pa_center_z))
    zen_mask = zenith_grid <= surf_zen_max
    if zen_mask.any():
        result['surf_corr_zen'] = float(np.nanmax(mean_corr[zen_mask]))
    else:
        result['surf_corr_zen'] = np.nan

    return result


def compute_peak_isolation(coarse_peaks, n_top=5):
    """Ratio of best peak correlation to mean of top-N peaks.

    A high ratio (>>1) means the best peak is well-separated from
    competitors. A ratio near 1 means many peaks with similar correlation
    (noisy landscape, less reliable reco).

    Args:
        coarse_peaks: List of (rho, phi, z, corr) tuples.
        n_top: Number of top peaks to average.

    Returns:
        Float ratio, or NaN if fewer than 2 peaks.
    """
    if len(coarse_peaks) < 2:
        return np.nan
    corrs = sorted([p[3] for p in coarse_peaks], reverse=True)
    top_mean = np.mean(corrs[:n_top])
    return float(corrs[0] / top_mean) if top_mean > 0 else np.nan


def compute_snr_derived(channel_snrs, threshold=DEFAULT_HELPER_SNR_THRESHOLD):
    """Compute derived per-string SNR metrics from a channel_snrs dict.

    Args:
        channel_snrs: Dict mapping channel_id to SNR value.
        threshold: SNR threshold for "signal present" classification.

    Returns:
        Dict with pa_avg_snr, pa_max_snr, helper_b_max_snr, etc.
    """
    def _safe_stat(chs, func):
        vals = [channel_snrs.get(ch, 0.0) for ch in chs]
        return float(func(vals)) if vals else 0.0

    result = {
        'pa_avg_snr': _safe_stat(PA_CHANNELS, np.mean),
        'pa_max_snr': _safe_stat(PA_CHANNELS, np.max),
        'helper_b_max_snr': _safe_stat(HELPER_B_CHANNELS, np.max),
        'helper_b_min_snr': _safe_stat(HELPER_B_CHANNELS, np.min),
        'helper_c_max_snr': _safe_stat(HELPER_C_CHANNELS, np.max),
        'helper_c_min_snr': _safe_stat(HELPER_C_CHANNELS, np.min),
    }

    helper_snrs = [channel_snrs.get(ch, 0.0) for ch in HELPER_CHANNELS]
    all_snrs = [channel_snrs.get(ch, 0.0) for ch in ALL_RECO_CHANNELS]

    result['n_helpers_above'] = int(sum(1 for s in helper_snrs if s > threshold))
    result['n_channels_above'] = int(sum(1 for s in all_snrs if s > threshold))
    result['has_helper_signal'] = result['n_helpers_above'] > 0

    return result


def extract_metrics(mean_corr_c, rho_vec_c, phi_vec_c, z_vec_c,
                    channel_snrs, coarse_peaks, pa_center_z, config):
    """Compute all validation metrics from the coarse grid and channel SNRs.

    Called by run_hierarchical when config['validation'] is True.

    Args:
        mean_corr_c: 3D coarse grid correlation array.
        rho_vec_c: Coarse rho grid points.
        phi_vec_c: Coarse phi grid points (radians).
        z_vec_c: Coarse z grid points (meters).
        channel_snrs: Dict from compute_channel_snrs.
        coarse_peaks: List of (rho, phi, z, corr) tuples.
        pa_center_z: PA center z position.
        config: Reco config dict.

    Returns:
        Dict with all validation metrics to merge into the reco result.
    """
    result = {}

    for ch, snr in channel_snrs.items():
        result[f'ch{ch}_snr'] = snr

    result.update(compute_snr_derived(channel_snrs))
    result.update(compute_surf_corr(
        mean_corr_c, rho_vec_c, phi_vec_c, z_vec_c, pa_center_z, config))
    result['peak_isolation_ratio'] = compute_peak_isolation(coarse_peaks)

    return result


def compute_derived_metrics(df, threshold=DEFAULT_HELPER_SNR_THRESHOLD):
    """Add derived per-string columns to a DataFrame with ch*_snr columns.

    Used by build_validation_df.py for post-hoc processing of reco H5 files
    that already have per-channel SNR columns.

    Args:
        df: DataFrame with columns like ch0_snr, ch9_snr, etc.
        threshold: SNR threshold for helper visibility.

    Returns:
        DataFrame with added columns.
    """
    snr_cols = {ch: f'ch{ch}_snr' for ch in ALL_RECO_CHANNELS}
    available = {ch: col for ch, col in snr_cols.items() if col in df.columns}

    if not available:
        return df

    pa_cols = [available[ch] for ch in PA_CHANNELS if ch in available]
    hb_cols = [available[ch] for ch in HELPER_B_CHANNELS if ch in available]
    hc_cols = [available[ch] for ch in HELPER_C_CHANNELS if ch in available]
    helper_cols = [available[ch] for ch in HELPER_CHANNELS if ch in available]
    all_cols = list(available.values())

    if pa_cols:
        df['pa_avg_snr'] = df[pa_cols].mean(axis=1)
        df['pa_max_snr'] = df[pa_cols].max(axis=1)
    if hb_cols:
        df['helper_b_max_snr'] = df[hb_cols].max(axis=1)
        df['helper_b_min_snr'] = df[hb_cols].min(axis=1)
    if hc_cols:
        df['helper_c_max_snr'] = df[hc_cols].max(axis=1)
        df['helper_c_min_snr'] = df[hc_cols].min(axis=1)
    if helper_cols:
        df['n_helpers_above'] = (df[helper_cols] > threshold).sum(axis=1)
        df['has_helper_signal'] = df['n_helpers_above'] > 0
    if all_cols:
        df['n_channels_above'] = (df[all_cols] > threshold).sum(axis=1)

    return df


def compute_shower_metrics(df, truth_dir):
    """Classify events by shower dominance pattern from sim truth HDF5.

    For each event, determines how many showers are visible and whether
    different channels see different showers as dominant (mixed dominance).
    Mixed dominance events have ~2.5x worse reco accuracy.

    Requires columns: source_file, runNum (or run_number).

    Args:
        df: DataFrame with event identifiers.
        truth_dir: Base directory of simulation production.

    Returns:
        DataFrame with added columns: n_showers, shower_energy_ratio,
        dominance_pattern, dominant_shower_type.
    """
    import h5py
    import os

    run_col = 'run_number' if 'run_number' in df.columns else 'runNum'

    df['n_showers'] = 1
    df['shower_energy_ratio'] = 0.0
    df['dominance_pattern'] = 'unknown'
    df['dominant_shower_type'] = 'unknown'

    reco_channels = ALL_RECO_CHANNELS

    hdf5_lookup = {}
    for root, _, files in os.walk(truth_dir):
        if 'final_set' in root:
            continue
        for fname in files:
            if fname.endswith('.hdf5'):
                hdf5_lookup[fname] = os.path.join(root, fname)

    for source_file in df['source_file'].unique():
        hdf5_name = source_file.replace('.nur', '.hdf5')
        hdf5_path = hdf5_lookup.get(hdf5_name)
        if hdf5_path is None:
            continue

        with h5py.File(hdf5_path, 'r') as f:
            egids = f['event_group_ids'][:]
            triggered = f['triggered'][:]
            shower_e = f['shower_energies'][:]
            shower_type = np.array([
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f['shower_type'][:]])

            has_station = 'station_23' in f
            if has_station:
                st = f['station_23']
                max_amps = st['max_amp_shower_and_ray'][:] if 'max_amp_shower_and_ray' in st else None
                egid_per_shower = st['event_group_id_per_shower'][:] if 'event_group_id_per_shower' in st else egids
            else:
                max_amps = None
                egid_per_shower = egids

        trig_mask = triggered
        trig_egids = egids[trig_mask]
        trig_e = shower_e[trig_mask]
        trig_type = shower_type[trig_mask]
        trig_station_egids = egid_per_shower[trig_mask] if len(egid_per_shower) == len(triggered) else trig_egids
        trig_max_amps = max_amps[trig_mask] if max_amps is not None else None

        unique_egids = np.unique(trig_egids)

        for eid in unique_egids:
            evt_mask = trig_egids == eid
            energies = trig_e[evt_mask]
            types = trig_type[evt_mask]
            n_shw = len(energies)

            sorted_idx = np.argsort(energies)[::-1]
            e_ratio = energies[sorted_idx[1]] / energies[sorted_idx[0]] if n_shw > 1 else 0.0
            dom_type = types[sorted_idx[0]]

            pattern = 'uniform'
            if n_shw > 1 and trig_max_amps is not None:
                station_mask = trig_station_egids[trig_mask] == eid if len(trig_station_egids) == trig_mask.sum() else evt_mask
                if station_mask.sum() >= 2:
                    amps = trig_max_amps[station_mask]
                    dominant_per_ch = np.zeros(amps.shape[1], dtype=int)
                    for ch_idx in range(amps.shape[1]):
                        ch_amps = []
                        for shw_idx in range(amps.shape[0]):
                            a = np.nanmax(amps[shw_idx, ch_idx, :])
                            ch_amps.append(a if np.isfinite(a) else 0.0)
                        dominant_per_ch[ch_idx] = np.argmax(ch_amps)
                    reco_ch_indices = [i for i, ch in enumerate(range(24)) if ch in reco_channels and i < len(dominant_per_ch)]
                    if reco_ch_indices:
                        dom_vals = dominant_per_ch[reco_ch_indices]
                        if len(set(dom_vals[dom_vals >= 0])) > 1:
                            pattern = 'mixed'

            df_mask = (df['source_file'] == source_file) & (df[run_col] == eid)
            df.loc[df_mask, 'n_showers'] = n_shw
            df.loc[df_mask, 'shower_energy_ratio'] = e_ratio
            df.loc[df_mask, 'dominance_pattern'] = pattern
            df.loc[df_mask, 'dominant_shower_type'] = dom_type

    return df


def compute_truth_metrics(df, pa_center):
    """Compute angular separation, zenith error, azimuth error from truth.

    Requires columns: rho, phi, z, true_xx, true_yy, true_zz.

    Args:
        df: DataFrame with reco and truth columns.
        pa_center: Array [x, y, z] of PA center absolute position.

    Returns:
        DataFrame with added truth comparison columns.
    """
    phi_rad = np.radians(df['phi'].values)
    reco_dx = df['rho'].values * np.cos(phi_rad)
    reco_dy = df['rho'].values * np.sin(phi_rad)
    reco_dz = df['z'].values - pa_center[2]
    reco_r = np.sqrt(reco_dx**2 + reco_dy**2 + reco_dz**2)
    reco_r = np.where(reco_r > 0, reco_r, 1.0)

    true_dx = df['true_xx'].values - pa_center[0]
    true_dy = df['true_yy'].values - pa_center[1]
    true_dz = df['true_zz'].values - pa_center[2]
    true_r = np.sqrt(true_dx**2 + true_dy**2 + true_dz**2)
    true_r = np.where(true_r > 0, true_r, 1.0)

    cos_angle = np.clip(
        (reco_dx * true_dx + reco_dy * true_dy + reco_dz * true_dz)
        / (reco_r * true_r), -1, 1)
    df['angular_separation'] = np.degrees(np.arccos(cos_angle))

    reco_rho = np.sqrt(reco_dx**2 + reco_dy**2)
    true_rho = np.sqrt(true_dx**2 + true_dy**2)
    df['reco_zenith'] = np.degrees(np.arctan2(reco_rho, reco_dz))
    df['true_zenith'] = np.degrees(np.arctan2(true_rho, true_dz))
    df['zenith_error'] = np.abs(df['reco_zenith'] - df['true_zenith'])

    reco_az = np.degrees(np.arctan2(reco_dy, reco_dx)) % 360
    true_az = np.degrees(np.arctan2(true_dy, true_dx)) % 360
    az_diff = np.abs(reco_az - true_az)
    df['azimuth_error'] = np.minimum(az_diff, 360 - az_diff)

    return df
