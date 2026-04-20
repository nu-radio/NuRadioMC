#!/usr/bin/env python3
"""Driver script for RNO-G per-event feature extraction.

Reads RNO-G data (ROOT via ``dataProviderRNOG`` or NUR via
``dataProviderNuRadio``), runs the shared
``NuRadioReco.modules.channelFeatureExtractor`` on every event to get
per-channel features, then applies RNO-G-specific aggregation:

- Phased-array (PA) coherent sum and VPOL coherent sum
- Coherent-sum features (SNR, impulsivity, kurtosis, entropy,
  spectral descriptors, impulse-template correlations)
- Group-level aggregates over the PA, VPOL, and deep (VPOL + HPOL)
  channel groups (mean of per-channel kurtosis, entropy, spectral
  features, impulse correlations; mean SNR for PA and VPOL)

Results for every processed event are flattened into a single row and
written as a pandas DataFrame to HDF5 with the original config saved as
metadata.

A minimal reference example of channelFeatureExtractor alone (no RNO-G
aggregation) lives in the module's docstring; use this driver as the
starting point for a production RNO-G feature-extraction job.

Usage
-----
    python feature_extraction.py --config features_config.yaml \\
        --input /path/to/station21_run1000.root \\
        --station_id 21 --run_chunk 0
"""

import argparse
import json
import logging
import os
import re
import warnings

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from NuRadioReco.modules.channelFeatureExtractor import channelFeatureExtractor
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
from NuRadioReco.modules.RNO_G.dataProviderNuRadio import dataProviderNuRadio
from NuRadioReco.modules.RNO_G.stationHitFilter import stationHitFilter
from NuRadioReco.detector import detector
from NuRadioReco.detector.RNO_G import rnog_detector
import NuRadioReco.utilities.trace_utilities as trace_utils


logger = logging.getLogger("NuRadioReco.examples.RNOG.feature_extraction")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


PA_CHS = (0, 1, 2, 3)
VPOL_CHS = (0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23)
HPOL_CHS = (4, 8, 11, 21)
DEEP_CHS = tuple(sorted(set(VPOL_CHS) | set(HPOL_CHS)))

SPECTRAL_KEYS = (
    "spectral_centroid", "spectral_bandwidth", "spectral_skewness",
    "spectral_kurtosis", "spectral_entropy", "spectral_slope",
    "spectral_rolloff_90", "spectral_flatness", "low_band_fraction",
)
IMPULSE_TEMPLATE_NAMES = (
    "delta", "bipolar", "gaussian", "bipolar_wide", "sinc")


def init_detector(config):
    """Build a detector from a config entry.

    Config keys:
        detector_source: "json" (NuRadioReco JSON) or "rnog_file"
            (exported MongoDB snapshot, ``.json.xz``).
        detector_file: path to the file (required).
    """
    source = config.get("detector_source", "json")
    path = config["detector_file"]
    if source == "rnog_file":
        return rnog_detector.Detector(detector_file=path)
    return detector.Detector(json_filename=path)


def select_provider(config, input_files, det):
    """Instantiate the right data provider for the input type."""
    data_type = config.get("data_type", "root")
    preproc_cfg = config.get("preprocessor", None)
    reader_kwargs = config.get("reader_kwargs", {})
    if data_type == "nur":
        provider = dataProviderNuRadio()
        provider.begin(input_files, det,
                       reader_kwargs=reader_kwargs,
                       preprocessor_config=preproc_cfg)
    else:
        provider = dataProviderRNOG()
        provider.begin(input_files, det,
                       reader_kwargs=reader_kwargs,
                       preprocessor_config=preproc_cfg)
    return provider


def _coherent_sum(traces, ref_ch, chs):
    """PA/VPOL coherent sum aligned against ``ref_ch``."""
    if ref_ch not in traces:
        return None
    ref = traces[ref_ch]
    others = [traces[c] for c in chs if c != ref_ch and c in traces]
    if not others:
        return ref
    return trace_utils.get_coherent_sum(others, ref)


def _mean_over(per_ch, chs):
    """Mean of per-channel values, NaN if no channel has the key."""
    vals = [per_ch[ch] for ch in chs if ch in per_ch]
    return float(np.mean(vals)) if vals else float("nan")


def _coherent_sum_features(trace, sampling_rate, n_entropy_bins,
                            spec_fmin, spec_fmax, spec_low):
    """Scalar features on a coherent-sum trace."""
    row = {}
    noise_rms = trace_utils.get_split_trace_noise_RMS(trace)
    row["coherent_snr"] = float(
        trace_utils.get_signal_to_noise_ratio(trace, noise_rms))
    row["coherent_impulsivity_nrmc"] = float(trace_utils.get_impulsivity(trace))
    ext = trace_utils.get_extended_impulsivity(trace)
    row["coherent_impulsivity"] = ext["impulsivity_custom"]
    row["coherent_impulsivity_r_squared"] = ext["impulsivity_r_squared"]
    row["coherent_impulsivity_slope"] = ext["impulsivity_slope"]
    row["coherent_impulsivity_intercept"] = ext["impulsivity_intercept"]
    row["coherent_impulsivity_ks_statistic"] = ext["impulsivity_ks_statistic"]
    row["coherent_kurtosis"] = float(stats.kurtosis(trace))
    row["coherent_entropy"] = float(
        trace_utils.get_entropy(trace, n_hist_bins=n_entropy_bins))
    spec = trace_utils.get_spectral_features(
        trace, sampling_rate,
        fmin=spec_fmin, fmax=spec_fmax, low_band_boundary=spec_low,
    )
    for key, val in spec.items():
        row[f"coherent_{key}"] = float(val)
    corrs = trace_utils.get_impulse_template_correlations(
        trace, sampling_rate)
    for name, val in corrs.items():
        row[f"coherent_impulse_corr_{name}"] = float(val)
    return row


def build_feature_row(per_ch, traces, sampling_rate, config):
    """Flatten per-channel features + RNO-G aggregates into one row."""
    row = {}

    for ch, feats in per_ch.items():
        for name, val in feats.items():
            row[f"ch{ch}_{name}"] = val

    row["pa_avg_snr"] = _mean_over({c: per_ch[c]["snr"] for c in per_ch if "snr" in per_ch[c]}, PA_CHS)
    row["vpol_avg_snr"] = _mean_over({c: per_ch[c]["snr"] for c in per_ch if "snr" in per_ch[c]}, VPOL_CHS)

    for key in ("kurtosis", "entropy"):
        vals_per_ch = {c: per_ch[c][key] for c in per_ch if key in per_ch[c]}
        row[f"{key}_avg_pa"] = _mean_over(vals_per_ch, PA_CHS)
        row[f"{key}_avg_vpol"] = _mean_over(vals_per_ch, VPOL_CHS)
        row[f"{key}_avg_deep"] = _mean_over(vals_per_ch, DEEP_CHS)

    for key in SPECTRAL_KEYS:
        vals_per_ch = {c: per_ch[c][key] for c in per_ch if key in per_ch[c]}
        row[f"{key}_avg_pa"] = _mean_over(vals_per_ch, PA_CHS)
        row[f"{key}_avg_vpol"] = _mean_over(vals_per_ch, VPOL_CHS)
        row[f"{key}_avg_deep"] = _mean_over(vals_per_ch, DEEP_CHS)

    for tmpl in IMPULSE_TEMPLATE_NAMES:
        col = f"impulse_corr_{tmpl}"
        vals_per_ch = {c: per_ch[c][col] for c in per_ch if col in per_ch[c]}
        row[f"{col}_avg_pa"] = _mean_over(vals_per_ch, PA_CHS)
        row[f"{col}_avg_vpol"] = _mean_over(vals_per_ch, VPOL_CHS)
        row[f"{col}_avg_deep"] = _mean_over(vals_per_ch, DEEP_CHS)

    if config.get("build_coherent_sums", True):
        pa_avail = [c for c in PA_CHS if c in traces]
        vpol_avail = [c for c in VPOL_CHS if c in traces]
        pa_ref = pa_avail[0] if pa_avail else None
        vpol_ref = vpol_avail[0] if vpol_avail else None

        cs_pa = _coherent_sum(traces, pa_ref, pa_avail) if pa_ref is not None else None
        cs_vpol = _coherent_sum(traces, vpol_ref, vpol_avail) if vpol_ref is not None else None

        cfg = config.get("features", {})
        n_bins = cfg.get("n_entropy_bins", 50)
        spec_fmin = cfg.get("spectral_fmin", 0.08)
        spec_fmax = cfg.get("spectral_fmax", 0.6)
        spec_low = cfg.get("spectral_low_band_boundary", 0.1)

        if cs_pa is not None:
            pa_feats = _coherent_sum_features(
                cs_pa, sampling_rate, n_bins, spec_fmin, spec_fmax, spec_low)
            row.update(pa_feats)
        if cs_vpol is not None:
            vpol_feats = _coherent_sum_features(
                cs_vpol, sampling_rate, n_bins, spec_fmin, spec_fmax, spec_low)
            for k, v in vpol_feats.items():
                row[k + "_vpol"] = v

    return row


def compute_hit_filter_features(event, station, det, hit_filter):
    """Run the station hit filter and return summary features for one event.

    Columns:
        passed_hit_filter: 1 if event passes the filter, else 0.
        n_coincident_pairs_pa: coincident channel pairs inside the PA group.
        n_high_hits_pa: PA channels with Hilbert max above threshold.
        n_coincident_pairs_in_ice: coincident pairs across all in-ice groups.
        n_high_hits_in_ice: in-ice channels with Hilbert max above threshold.
    """
    passed = hit_filter.run(event, station, det)

    in_time_window = hit_filter.is_in_time_window()
    over_hit_threshold = hit_filter.is_over_hit_threshold()

    n_pairs_pa = int(sum(in_time_window[0]))
    n_high_pa = int(sum(over_hit_threshold[:4]))

    n_pairs_in_ice = n_pairs_pa + int(
        sum(in_time_window[grp][0] for grp in range(1, 4))
    )
    n_high_in_ice = int(sum(over_hit_threshold[:15]))

    return {
        "passed_hit_filter": int(passed),
        "n_coincident_pairs_pa": n_pairs_pa,
        "n_high_hits_pa": n_high_pa,
        "n_coincident_pairs_in_ice": n_pairs_in_ice,
        "n_high_hits_in_ice": n_high_in_ice,
    }


def extract_station_traces(station, channel_ids):
    """Read current traces off the station into ``{ch: np.ndarray}``."""
    out = {}
    for ch in station.iter_channels():
        ch_id = ch.get_id()
        if channel_ids is None or ch_id in channel_ids:
            out[ch_id] = np.asarray(ch.get_trace())
    return out


def main(config, input_files, run_chunk, event_filter=None):
    """Run the RNO-G feature extraction pipeline over ``input_files``."""
    det = init_detector(config)
    provider = select_provider(config, input_files, det)

    channels = tuple(config.get("channels", DEEP_CHS))
    extractor = channelFeatureExtractor()
    extractor.begin(config=config.get("features", {}))

    hf_cfg = config.get("hit_filter", {})
    hit_filter = None
    if hf_cfg.get("enabled", False):
        hit_filter = stationHitFilter(
            complete_time_check=hf_cfg.get("complete_time_check", True),
            complete_hit_check=hf_cfg.get("complete_hit_check", True),
        )
        hit_filter.begin()

    results = []
    for event in provider.run():
        station = event.get_station()

        run_num = event.get_run_number()
        evt_num = event.get_id()
        if event_filter is not None and evt_num not in event_filter:
            continue

        hf_features = {}
        if hit_filter is not None:
            hf_features = compute_hit_filter_features(event, station, det, hit_filter)
            if hf_cfg.get("require_pass", False) and not hf_features["passed_hit_filter"]:
                continue

        per_ch = extractor.run(event, station, det, channel_ids=channels)

        sampling_rate = None
        for ch in station.iter_channels():
            sampling_rate = ch.get_sampling_rate()
            break

        traces = extract_station_traces(station, channels)
        row = build_feature_row(per_ch, traces, sampling_rate, config)

        if hit_filter is not None and hf_cfg.get("add_features", True):
            row.update(hf_features)

        row["run_number"] = run_num
        row["event_number"] = evt_num
        row["source_file"] = _source_for_event(event, input_files)

        if config.get("data_type", "root") == "nur":
            lge = _extract_log10_energy(row["source_file"])
            if lge is not None:
                row["log10_energy"] = lge

        results.append(row)

    if hit_filter is not None and hf_cfg.get("log_summary", True):
        hit_filter.end()

    provider.end()

    df = pd.DataFrame(results)
    _save(df, config, run_chunk)


def _source_for_event(event, input_files):
    """Best-effort event-to-source mapping. Falls back to first input."""
    try:
        return event.get_parameter("source_file")
    except Exception:
        return input_files[0] if input_files else ""


def _save(df, config, run_chunk):
    """Write feature DataFrame to HDF5 (+ config metadata group)."""
    if df is None or df.empty:
        logger.warning("No feature rows to save.")
        return

    output_root = config.get(
        "output_root_dir",
        os.environ.get("FEATURE_OUTPUT_ROOT",
                       os.path.join(os.getcwd(), "feature_extraction")),
    )
    experiment = config.get("experiment_id", "default")
    category = "sim_data" if config.get("data_type", "root") == "nur" else "real_data"
    save_dir = os.path.join(
        output_root, "results", category, experiment,
        f"station{config.get('station_id', 0)}",
        str(config.get("year", 0)),
    )
    os.makedirs(save_dir, exist_ok=True)
    filename = (f"station{config.get('station_id', 0)}_features_df"
                f"_chunk{run_chunk}_{experiment}.h5")
    path = os.path.join(save_dir, filename)

    if "run_number" in df.columns and "event_number" in df.columns:
        df = df.sort_values(["run_number", "event_number"])

    df.to_hdf(path, key="data", mode="w", format="table", complevel=5)
    with h5py.File(path, "a") as f:
        _save_config(f, config)
    print(f"Saved {len(df)} events to {path}", flush=True)


def _save_config(h5file, config, group_name="config"):
    """Store a config dict as attributes on an HDF5 group."""
    grp = h5file.require_group(group_name)
    for key, val in config.items():
        try:
            if isinstance(val, (dict, list, tuple)):
                grp.attrs[key] = json.dumps(val, default=str)
            elif val is None:
                grp.attrs[key] = "None"
            else:
                grp.attrs[key] = val
        except Exception:
            grp.attrs[key] = str(val)


def _extract_log10_energy(filepath):
    """Parse log10(E/eV) from a ``lgE_<val>`` token in the filename."""
    m = re.search(r"lgE_?([0-9.]+)", os.path.basename(filepath))
    return float(m.group(1)) if m else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RNO-G per-event feature extraction driver")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config path")
    parser.add_argument("-i", "--input", type=str, nargs="+", required=True,
                        help="Input data file(s) (ROOT or NUR)")
    parser.add_argument("--station_id", type=int, default=None,
                        help="Override config['station_id']")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--data_type", type=str, default=None,
                        choices=[None, "root", "nur"])
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--run_chunk", type=str, default="0",
                        help="Chunk identifier for output filename")
    parser.add_argument("--events", type=int, nargs="+", default=None,
                        help="Only process these event numbers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(name)s - %(levelname)s - %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in ("station_id", "year", "data_type", "experiment_id"):
        cli_val = getattr(args, key)
        if cli_val is not None:
            config[key] = cli_val

    for key, val in list(config.items()):
        if isinstance(val, str) and "$" in val:
            config[key] = os.path.expandvars(val)

    event_filter = set(args.events) if args.events else None
    print(f"Processing {len(args.input)} input file(s)", flush=True)
    main(config, args.input, args.run_chunk, event_filter=event_filter)
