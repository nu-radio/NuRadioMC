#!/usr/bin/env python3
"""Measure FT-noise trigger-channel Vrms from a sample of tiled noise realizations.

Builds noise-only trigger-channel traces the way the simulation does: draw FT events from
the pool, upsample and Hann-tile them to the internal trace length, apply the hardware
response (trigger path), and measure the per-channel Vrms. The recommended fixed value per
channel is the median of the trigger-copy Vrms over the sampled realizations.

This is the original (sampled) measurement; measure_trigger_vrms_full.py processes the whole
FT pool. Both use the branch simulate.py tiling and upsampling. Re-measurement depends on the
FT pool, clean mask, and detector response vintage, so it will not reproduce the historical
hardcoded dicts to the digit; the shipped trigger_vrms_station{NN}.yaml pin what production used.

Usage: python measure_trigger_vrms.py --station 23 --ft_noise_dir <dir> [--clean_mask <npz>]
"""
import argparse
import os
import sys
import math
import logging
import numpy as np

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', '..'))

from astropy.time import Time
from NuRadioReco.utilities import units
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
from simulate import (FTNoisePool, get_vrms_from_temperature_for_trigger_channels,
                      tile_noise_overlap_add, upsample_trace, TILE_OVERLAP)

TRIGGER_CHS = [0, 1, 2, 3]
N_UP = int(round(2048 * (5.0 / 3.2)))  # FT event length at 5 GHz


def main():
    """Sample tiled FT noise, measure trigger-copy Vrms per channel, print the medians."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type=int, required=True)
    parser.add_argument("--ft_noise_dir", required=True)
    parser.add_argument("--clean_mask", default=None)
    parser.add_argument("--detector_file", default=None,
                        help="detector description file; omit to query MongoDB at --event_time")
    parser.add_argument("--event_time", default="2022-10-01")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_internal", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sid = args.station

    det = rnog_detector.Detector(detector_file=args.detector_file, log_level=logging.ERROR,
                                 select_stations=sid)
    det.update(Time(args.event_time))

    thermal_vrms = get_vrms_from_temperature_for_trigger_channels(det, sid, TRIGGER_CHS, 300)
    ft_pool = FTNoisePool(ft_dir=args.ft_noise_dir, clean_mask_path=args.clean_mask,
                          station_id=sid, seed=args.seed)
    hw_resp = hardwareResponseIncorporator.hardwareResponseIncorporator()
    hw_resp.begin(trigger_channels=np.array(TRIGGER_CHS))
    trig_board = triggerBoardResponse.triggerBoardResponse()
    trig_board.begin(clock_offset=0.0, adc_output="counts")

    stride = N_UP - TILE_OVERLAP
    n_tiles = max(1, math.ceil(args.n_internal / stride))
    print(f"Station {sid}: internal {args.n_internal} samples, {n_tiles} tiles/event, "
          f"{args.n_samples} realizations")

    ft_vrms_after_hw = {ch: [] for ch in TRIGGER_CHS}
    for i in range(args.n_samples):
        ft_events = [ft_pool.get_noise_event() for _ in range(n_tiles)]
        station = NuRadioReco.framework.station.Station(sid)
        for ch_id in TRIGGER_CHS:
            channel = NuRadioReco.framework.channel.Channel(ch_id)
            tiles = [upsample_trace(e.get(ch_id), N_UP) for e in ft_events if e.get(ch_id) is not None]
            noise = tile_noise_overlap_add(tiles, args.n_internal) if tiles else np.zeros(args.n_internal)
            channel.set_trace(noise, 5.0 * units.GHz)
            station.add_channel(channel)
        evt = NuRadioReco.framework.event.Event(0, i)
        evt.set_station(station)
        hw_resp.run(evt, station, det, sim_to_data=True)
        for ch_id in TRIGGER_CHS:
            ch = station.get_channel(ch_id)
            trace = (ch.get_trigger_channel().get_trace() if ch.has_extra_trigger_channel()
                     else ch.get_trace())
            ft_vrms_after_hw[ch_id].append(np.std(trace))

    print(f"{'ch':>4} {'thermal':>10} {'FT+HW median':>14} {'ratio':>8}")
    recommended = {}
    for ch_id in TRIGGER_CHS:
        med = np.median(ft_vrms_after_hw[ch_id])
        recommended[ch_id] = med
        therm = thermal_vrms[TRIGGER_CHS.index(ch_id)]
        print(f"{ch_id:>4} {therm/units.mV:>9.4f}m {med/units.mV:>13.4f}m {med/therm:>7.3f}x")

    print("\nTRIGGER_VRMS_FT = {")
    for ch_id in TRIGGER_CHS:
        print(f"    {ch_id}: {recommended[ch_id]:.6e},")
    print("}")


if __name__ == "__main__":
    main()
