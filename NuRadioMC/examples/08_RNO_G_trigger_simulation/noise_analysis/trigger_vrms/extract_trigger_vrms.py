"""Extract trigger-path Vrms from FT data and save as YAML.

Reads FT noise events, applies the readout-to-trigger transfer function,
and measures the per-channel Vrms. The output YAML is passed to simulate.py
via --trigger_vrms.

The number of FT events to use (--n_events) should be informed by the
convergence study (vrms_convergence_study.py).
"""

import argparse
import numpy as np
import os
import yaml
import logging
import datetime as dt

from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator
from NuRadioReco.modules.measured_noise.RNO_G.noiseImporter import noiseImporter
from NuRadioReco.utilities import units

logger = logging.getLogger("extract_trigger_vrms")
logging.basicConfig(level=logging.INFO)

TRIGGER_CHANNELS = [0, 1, 2, 3]


def extract_vrms(noise_imp, hw_resp, det, station_id, n_events, seed=42):
    """Draw n_events FT events and measure trigger-path Vrms per channel.

    Returns dict mapping channel_id to Vrms in V.
    """
    rng = np.random.default_rng(seed)
    event_indices = noise_imp._noiseImporter__event_index_list
    station_ids = noise_imp._noiseImporter__station_id_list
    mask = station_ids == station_id

    sum_sq = {ch: 0.0 for ch in TRIGGER_CHANNELS}
    n_samples_total = {ch: 0 for ch in TRIGGER_CHANNELS}

    for i in range(n_events):
        idx = int(rng.choice(event_indices[mask]))
        noise_event = noise_imp._noise_reader.get_event_by_index(idx)
        if noise_event is None:
            continue
        sid = noise_event.get_station_ids()[0]
        noise_station = noise_event.get_station(sid)

        for ch_id in TRIGGER_CHANNELS:
            noise_ch = noise_station.get_channel(ch_id)
            noise_trace = noise_ch.get_trace()
            noise_sr = noise_ch.get_sampling_rate()

            trig_sr = 5.0 * units.GHz
            n_up = int(round(len(noise_trace) * trig_sr / noise_sr))
            noise_up = noiseImporter._upsample(noise_trace, n_up)

            transfer = noise_imp._get_readout_to_trigger_transfer(
                ch_id, n_up, det, station_id, trig_sr)
            noise_fft = np.fft.rfft(noise_up)
            trig_noise = np.fft.irfft(noise_fft * transfer, n=n_up)

            sum_sq[ch_id] += np.sum(trig_noise ** 2)
            n_samples_total[ch_id] += len(trig_noise)

    vrms = {}
    for ch_id in TRIGGER_CHANNELS:
        if n_samples_total[ch_id] > 0:
            vrms[ch_id] = np.sqrt(sum_sq[ch_id] / n_samples_total[ch_id])
        else:
            vrms[ch_id] = np.nan
    return vrms


def main():
    """Extract per-channel trigger-path Vrms and save as YAML."""
    parser = argparse.ArgumentParser(
        description="Extract trigger-path Vrms from FT data")
    parser.add_argument("--ft_noise_dir", type=str, required=True)
    parser.add_argument("--station_id", type=int, default=23)
    parser.add_argument("--event_time", type=str, default="2022-10-01")
    parser.add_argument("--detector_file", type=str, default=None)
    parser.add_argument("--n_events", type=int, default=100,
                        help="Number of FT events to measure (100 gives <1%% spread)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output YAML path (default: trigger_vrms_station{id}.yaml)")
    args = parser.parse_args()

    det = rnog_detector.Detector(
        detector_file=args.detector_file, log_level=logging.WARNING,
        always_query_entire_description=False,
        select_stations=args.station_id)
    det.update(dt.datetime.fromisoformat(args.event_time))

    hw_resp = hardwareResponseIncorporator.hardwareResponseIncorporator()
    hw_resp.begin(trigger_channels=TRIGGER_CHANNELS)

    import glob as _glob
    ft_files = sorted(_glob.glob(os.path.join(args.ft_noise_dir, "station*_run*.root")))
    if not ft_files:
        ft_files = sorted(_glob.glob(os.path.join(args.ft_noise_dir, "run*/waveforms.root")))
    if not ft_files:
        raise FileNotFoundError(f"No FT ROOT files in {args.ft_noise_dir}")

    noise_imp = noiseImporter()
    noise_imp.begin(
        noise_files=ft_files,
        match_station_id=True,
        scramble_noise_file_order=True,
        random_seed=args.seed,
        inject_trigger_copies=True,
        trigger_channels=TRIGGER_CHANNELS,
        hardware_response_incorporator=hw_resp,
        reader_kwargs={
            "select_runs": False,
            "convert_to_voltage": True,
            "apply_baseline_correction": "median",
        },
    )

    if args.output is None:
        args.output = f"trigger_vrms_station{args.station_id}.yaml"

    vrms = extract_vrms(
        noise_imp, hw_resp, det, args.station_id, args.n_events, args.seed)

    output = {
        "trigger_vrms_V": {ch: float(v) for ch, v in vrms.items()},
        "metadata": {
            "station_id": args.station_id,
            "event_time": args.event_time,
            "n_events": args.n_events,
            "seed": args.seed,
            "ft_noise_dir": args.ft_noise_dir,
        },
    }
    with open(args.output, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    print(f"Saved to {args.output}")
    for ch_id in TRIGGER_CHANNELS:
        print(f"  ch{ch_id}: {vrms[ch_id]/units.mV:.3f} mV")


if __name__ == "__main__":
    main()
