#!/usr/bin/env python3
"""Driver script for RNO-G per-event feature extraction.

Reads RNO-G data (ROOT via ``dataProviderRNOG`` or NUR via
``dataProviderNuRadio``), runs the shared
``NuRadioReco.modules.channelFeatureExtractor`` on every event to get
per-channel features, then applies RNO-G-specific aggregation. Which
antenna groups receive aggregate and coherent-sum columns is controlled
by the top-level ``antenna_groups`` config key; valid names are defined
by ``ANTENNA_GROUPS`` (``pa``, ``vpol``, ``hpol``, ``deep``) and the
default is all four. Coherent-sum features are built only for groups
that also appear in ``COHERENT_SUM_GROUPS``.

Aggregates produced per enabled antenna group:

- ``{feature}_avg_{group}`` for every per-channel feature (SNR,
  kurtosis, entropy, max amplitude, impulsivity, spectral descriptors,
  impulse-template correlations).
- For groups in ``COHERENT_SUM_GROUPS``: a coherent sum across that
  group's available channels plus ``coherent_{feature}_{group}``
  columns (SNR, impulsivity, kurtosis, entropy, spectral features,
  impulse-template correlations).

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
# `deep` = all VPOLs and HPOLs (string-deployed antennas on power +
# helper strings). Does NOT include the surface LPDAs (ch12-20).
DEEP_CHS = tuple(sorted(set(VPOL_CHS) | set(HPOL_CHS)))

ANTENNA_GROUPS = {
    "pa": PA_CHS,
    "vpol": VPOL_CHS,
    "hpol": HPOL_CHS,
    "deep": DEEP_CHS,
}
# Subset of ANTENNA_GROUPS for which coherent-sum features are defined.
# Extending to hpol or deep requires picking a reference channel + the
# aligned-shift logic in _coherent_sum, which is out of scope for the
# initial config-ification.
COHERENT_SUM_GROUPS = ("pa", "vpol")

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
        detector_source: "rnog_mongo" (live MongoDB query, default),
            "rnog_file" (exported MongoDB snapshot, ``.json.xz``),
            or "json" (NuRadioReco JSON).
        detector_file: path to the file. Required for "rnog_file" and
            "json"; ignored for "rnog_mongo".
    """
    source = config.get("detector_source", "rnog_mongo")
    if source == "rnog_mongo":
        return detector.Detector(source="rnog_mongo")
    path = config["detector_file"]
    if source == "rnog_file":
        return rnog_detector.Detector(detector_file=path)
    return detector.Detector(json_filename=path)


def select_provider(config, input_files, det):
    """Instantiate the right data provider for the input type.

    Picks between ``dataProviderNuRadio`` (NUR simulation) and
    ``dataProviderRNOG`` (ROOT data) from the first input file's
    extension. All inputs must share the same extension.
    """
    exts = {os.path.splitext(f)[1].lower() for f in input_files}
    if len(exts) != 1 or next(iter(exts)) not in {".nur", ".root"}:
        raise ValueError(
            f"Inputs must share a single extension (.nur or .root); got {exts}"
        )
    is_nur = next(iter(exts)) == ".nur"

    preproc_cfg = config.get("preprocessor", None)
    reader_kwargs = config.get("reader_kwargs", {})
    provider = dataProviderNuRadio() if is_nur else dataProviderRNOG()
    provider.begin(input_files, det,
                   reader_kwargs=reader_kwargs,
                   preprocessor_config=preproc_cfg)
    return provider, is_nur


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
    """Flatten per-channel features + RNO-G aggregates into one row.

    ``antenna_groups`` in the config controls which RNO-G groups get
    per-channel-average columns (``{feature}_avg_{group}``) and, for
    groups in ``COHERENT_SUM_GROUPS``, coherent-sum feature columns
    (``coherent_{feature}_{group}``). Default is every group in
    ``ANTENNA_GROUPS`` so the column set matches historical output.
    """
    row = {}

    for ch, feats in per_ch.items():
        for name, val in feats.items():
            row[f"ch{ch}_{name}"] = val

    groups = _resolve_antenna_groups(config)

    for key in ("snr", "kurtosis", "entropy", "max_amplitude", "impulsivity"):
        vals_per_ch = {c: per_ch[c][key] for c in per_ch if key in per_ch[c]}
        for group_name in groups:
            row[f"{key}_avg_{group_name}"] = _mean_over(
                vals_per_ch, ANTENNA_GROUPS[group_name])

    for key in SPECTRAL_KEYS:
        vals_per_ch = {c: per_ch[c][key] for c in per_ch if key in per_ch[c]}
        for group_name in groups:
            row[f"{key}_avg_{group_name}"] = _mean_over(
                vals_per_ch, ANTENNA_GROUPS[group_name])

    for tmpl in IMPULSE_TEMPLATE_NAMES:
        col = f"impulse_corr_{tmpl}"
        vals_per_ch = {c: per_ch[c][col] for c in per_ch if col in per_ch[c]}
        for group_name in groups:
            row[f"{col}_avg_{group_name}"] = _mean_over(
                vals_per_ch, ANTENNA_GROUPS[group_name])

    if config.get("build_coherent_sums", True):
        cfg = config.get("features", {})
        n_bins = cfg.get("n_entropy_bins", 50)
        spec_fmin = cfg.get("spectral_fmin", 0.08)
        spec_fmax = cfg.get("spectral_fmax", 0.6)
        spec_low = cfg.get("spectral_low_band_boundary", 0.1)

        for group_name in groups:
            if group_name not in COHERENT_SUM_GROUPS:
                continue
            group_chs = ANTENNA_GROUPS[group_name]
            avail = [c for c in group_chs if c in traces]
            ref = avail[0] if avail else None
            if ref is None:
                continue
            cs = _coherent_sum(traces, ref, avail)
            if cs is None:
                continue
            cs_feats = _coherent_sum_features(
                cs, sampling_rate, n_bins, spec_fmin, spec_fmax, spec_low)
            for k, v in cs_feats.items():
                row[f"{k}_{group_name}"] = v

    return row


def _resolve_antenna_groups(config):
    """Return the ordered list of antenna-group names to build aggregates for.

    Reads ``config['antenna_groups']``. If absent, defaults to every
    group in ``ANTENNA_GROUPS`` so column layout matches historical output.
    Unknown group names raise ValueError so typos fail loudly instead of
    silently dropping expected columns.
    """
    requested = config.get("antenna_groups", list(ANTENNA_GROUPS.keys()))
    unknown = [g for g in requested if g not in ANTENNA_GROUPS]
    if unknown:
        raise ValueError(
            f"antenna_groups contains unknown group(s) {unknown}; "
            f"valid options are {list(ANTENNA_GROUPS.keys())}."
        )
    return list(requested)


def compute_hit_filter_features(event, station, det, hit_filter):
    """Run the station hit filter and return summary features for one event.

    Column names follow the driver's antenna-group naming convention:
    ``_pa`` = scoped to the PA group, ``_deep`` = scoped to string-deployed
    antennas (VPOL ∪ HPOL). See ``ANTENNA_GROUPS``.

    Columns:
        passed_hit_filter: 1 if event passes the filter, else 0.
        n_coincident_pairs_pa: coincident channel pairs inside the PA group.
        n_high_hits_pa: PA channels with Hilbert max above threshold.
        n_coincident_pairs_deep: coincident pairs across PA + string helpers.
        n_high_hits_deep: string-deployed channels with Hilbert max above threshold.
    """
    passed = hit_filter.run(event, station, det)

    in_time_window = hit_filter.is_in_time_window()
    over_hit_threshold = hit_filter.is_over_hit_threshold()

    n_pairs_pa = int(sum(in_time_window[0]))
    n_high_pa = int(sum(over_hit_threshold[:4]))

    n_pairs_deep = n_pairs_pa + int(
        sum(in_time_window[grp][0] for grp in range(1, 4))
    )
    n_high_deep = int(sum(over_hit_threshold[:15]))

    return {
        "passed_hit_filter": int(passed),
        "n_coincident_pairs_pa": n_pairs_pa,
        "n_high_hits_pa": n_high_pa,
        "n_coincident_pairs_deep": n_pairs_deep,
        "n_high_hits_deep": n_high_deep,
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

    channels = tuple(config.get("channels", DEEP_CHS))
    extractor = channelFeatureExtractor()
    extractor.begin(config=config.get("features", {}))

    hf_cfg = config.get("hit_filter", {})
    hit_filter = None
    if hf_cfg.get("enabled", False):
        time_check = hf_cfg.get("complete_time_check", True)
        hit_check = hf_cfg.get("complete_hit_check", True)
        if hf_cfg.get("add_features", True) and not (time_check and hit_check):
            raise ValueError(
                "hit_filter.add_features requires complete_time_check=True "
                "and complete_hit_check=True. With either disabled the "
                "stationHitFilter skips populating is_in_time_window / "
                "is_over_hit_threshold, and the n_coincident_pairs_* / "
                "n_high_hits_* feature columns would silently be None. "
                f"Got complete_time_check={time_check}, "
                f"complete_hit_check={hit_check}."
            )
        hit_filter = stationHitFilter(
            complete_time_check=time_check,
            complete_hit_check=hit_check,
        )
        hit_filter.begin()

    # Iterate one input file at a time so each event's source_file is
    # unambiguous. eventReader does not tag events with their originating
    # file, and feeding a multi-file list would require guessing.
    results = []
    is_nur = False
    for src_file in input_files:
        src_basename = os.path.basename(src_file)
        if event_filter is not None and 'by_file' in event_filter \
                and src_basename not in event_filter['by_file']:
            continue
        provider, is_nur = select_provider(config, [src_file], det)
        for event in provider.run():
            station = event.get_station()

            run_num = event.get_run_number()
            evt_num = event.get_id()
            if event_filter is not None:
                if 'by_file' in event_filter:
                    if (run_num, evt_num) not in event_filter['by_file'][src_basename]:
                        continue
                elif 'by_run' in event_filter:
                    if run_num not in event_filter['by_run'] \
                            or evt_num not in event_filter['by_run'][run_num]:
                        continue
                elif 'by_event' in event_filter:
                    if evt_num not in event_filter['by_event']:
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
            row["source_file"] = src_file

            if is_nur:
                lge = _extract_log10_energy(src_file)
                if lge is not None:
                    row["log10_energy"] = lge

            results.append(row)

        provider.end()

    if hit_filter is not None and hf_cfg.get("log_summary", True):
        hit_filter.end()

    df = pd.DataFrame(results)
    _save(df, config, run_chunk, is_nur)


def _save(df, config, run_chunk, is_nur):
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
    category = "sim_data" if is_nur else "real_data"
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
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--run_chunk", type=str, default="0",
                        help="Chunk identifier for output filename")
    parser.add_argument("--events", type=str, nargs="+", default=None,
                        help=(
                            "Filter to a subset of events. Accepts either a "
                            "space-separated list of integer event numbers, or "
                            "a path to a JSON file, auto-detected: run-keyed "
                            "{run: [events]} if keys parse as integers, else "
                            "file-aware {src: [[run, evt], ...]}. See "
                            "NuRadioReco.utilities.io_utilities.parse_event_ids."
                        ))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(name)s - %(levelname)s - %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in ("station_id", "year", "experiment_id"):
        cli_val = getattr(args, key)
        if cli_val is not None:
            config[key] = cli_val

    for key, val in list(config.items()):
        if isinstance(val, str) and "$" in val:
            config[key] = os.path.expandvars(val)

    event_filter = None
    if args.events:
        from NuRadioReco.utilities.io_utilities import parse_event_ids
        event_filter = parse_event_ids(args.events)
    print(f"Processing {len(args.input)} input file(s)", flush=True)
    main(config, args.input, args.run_chunk, event_filter=event_filter)
