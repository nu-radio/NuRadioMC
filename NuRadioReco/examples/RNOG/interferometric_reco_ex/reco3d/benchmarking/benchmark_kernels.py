"""Benchmark pair-major vs point-major Numba correlation kernels.

Loads one event, builds coarse and refine grids, runs both kernel variants,
and compares timing and correctness. Use this as a quick sanity check that
the pair-major kernels produce identical results and to measure the speedup
on your hardware.
"""

import argparse
import datetime
import itertools
import logging
import os
import sys
import time

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECO3D_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, RECO3D_DIR)


def run_benchmark(config_path, nur_file, event_index, n_runs, detector_file):
    """Load one event, benchmark both kernel variants on coarse and refine grids.

    Args:
        config_path: path to reco YAML config
        nur_file: path to input NUR file
        event_index: which event in the file to process
        n_runs: number of timed iterations per kernel (takes median)
        detector_file: path to detector JSON.xz (optional)
    """
    from NuRadioReco.detector.RNO_G import rnog_detector
    from NuRadioReco.modules.channelResampler import channelResampler
    from NuRadioReco.modules.channelAddCableDelay import channelAddCableDelay
    from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import (
        hardwareResponseIncorporator)
    from NuRadioReco.modules.io.eventReader import eventReader
    from NuRadioReco.utilities import units
    from NuRadioReco.modules.interferometricDirectionReconstruction3D import (
        InterferometricReco3D)
    from fast_grouped_multiray import (
        pack_tt_grids, pack_tt_grids_transposed, pack_corr_data,
        build_combo_table,
        _perpair_multiray_kernel_t,
        _perpair_multiray_kernel_pairmajor,
        _grouped_multiray_kernel, _grouped_multiray_kernel_pairmajor,
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key, val in config.items():
        if isinstance(val, str) and '$' in val:
            config[key] = os.path.expandvars(val)

    station_id = config['station_id']
    channels = config['channels']

    det_path = detector_file or config.get('detector_file')
    if det_path:
        det_path = os.path.expandvars(det_path)
    det = rnog_detector.Detector(
        detector_file=det_path,
        log_level=logging.WARNING, select_stations=station_id)
    det_date = config.get('detector_date', '2022-10-01')
    det.update(datetime.datetime.fromisoformat(det_date))

    reader = eventReader()
    reader.begin(nur_file)
    event_ids = reader._eventReader__fin.get_event_ids()
    eid = event_ids[event_index]
    evt = reader._eventReader__fin.get_event(event_id=eid)
    stn = evt.get_station(station_id)

    cable_delay = channelAddCableDelay(); cable_delay.begin()
    hw_response = hardwareResponseIncorporator(); hw_response.begin()
    resampler = channelResampler(); resampler.begin()

    cable_delay.run(evt, stn, det, mode='subtract')
    hw_response.run(evt, stn, det, sim_to_data=False, mode='phase_only')
    resampler.run(evt, stn, det, sampling_rate=10 * units.GHz)

    reco = InterferometricReco3D()
    reco.begin(station_id, config, det)

    volt_arrays, time_arrays = [], []
    for ch in channels:
        channel = stn.get_channel(ch)
        volt_arrays.append(channel.get_trace())
        time_arrays.append(channel.get_times())

    hilbert_mode = config.get('hilbert_envelope_mode', None)
    apply_hann = config.get('apply_hann_window', False)
    corr_norm = config.get('correlation_normalization', 'normalized')
    v_pairs = list(itertools.combinations(volt_arrays, 2))
    corr_data = reco._prepare_corr_funcs(
        time_arrays, v_pairs,
        hilbert_envelope_mode=hilbert_mode,
        apply_hann_window=apply_hann,
        correlation_normalization=corr_norm)

    coarse_limits = config.get('coarse_limits', [1, 1500, 0, 360, -1500, 0])
    n_rho_coarse = config.get('coarse_n_rho', 50)
    coarse_steps = config.get('coarse_step_sizes', [30, 5, 30])

    rho_vec = np.geomspace(max(coarse_limits[0], 1.0), coarse_limits[1],
                           n_rho_coarse)
    phi_vec = np.arange(coarse_limits[2], coarse_limits[3],
                        coarse_steps[1]) * (np.pi / 180.0)
    z_vec = np.arange(coarse_limits[4],
                      coarse_limits[5] + coarse_steps[2], coarse_steps[2])

    src_enu = reco._build_source_enu_matrix(rho_vec, phi_vec, z_vec)
    tt_all = reco._compute_tt_multiray(src_enu, channels)

    grid_shape = None
    for ch in channels:
        for rt in tt_all.get(ch, {}):
            grid_shape = tt_all[ch][rt].shape
            break
        if grid_shape:
            break

    n_points = int(np.prod(grid_shape))
    ch_pairs = list(itertools.combinations(range(len(channels)), 2))
    n_pairs = len(ch_pairs)
    n_ch = len(channels)

    tt_packed, ch_avail_rts = pack_tt_grids(tt_all, channels, grid_shape)
    tt_t, _ = pack_tt_grids_transposed(tt_all, channels, grid_shape)
    corr_packed, corr_lengths, dts, offsets = pack_corr_data(
        corr_data, n_pairs)

    ch_rt_mask = np.zeros((n_ch, tt_packed.shape[1]), dtype=np.bool_)
    for ci in range(n_ch):
        for rt_idx in ch_avail_rts[ci]:
            ch_rt_mask[ci, rt_idx] = True

    pair_ch1 = np.array([p[0] for p in ch_pairs], dtype=np.int64)
    pair_ch2 = np.array([p[1] for p in ch_pairs], dtype=np.int64)
    pw = np.ones(n_pairs, dtype=np.float64)

    print("Warming up Numba kernels...")
    _perpair_multiray_kernel_t(tt_t, corr_packed, corr_lengths, dts, offsets,
                               pair_ch1, pair_ch2, pw, ch_rt_mask, n_points)
    _perpair_multiray_kernel_pairmajor(tt_packed, corr_packed, corr_lengths,
                                       dts, offsets, pair_ch1, pair_ch2, pw,
                                       ch_rt_mask, n_points)

    print(f"\nCoarse grid: {grid_shape}, {n_points} points, {n_pairs} pairs")
    print(f"Benchmarking ({n_runs} runs each)...\n")

    times_pm = []
    for _ in range(n_runs):
        t0 = time.time()
        result_pm = _perpair_multiray_kernel_t(
            tt_t, corr_packed, corr_lengths, dts, offsets,
            pair_ch1, pair_ch2, pw, ch_rt_mask, n_points)
        times_pm.append(time.time() - t0)

    times_pair = []
    for _ in range(n_runs):
        t0 = time.time()
        result_pair = _perpair_multiray_kernel_pairmajor(
            tt_packed, corr_packed, corr_lengths, dts, offsets,
            pair_ch1, pair_ch2, pw, ch_rt_mask, n_points)
        times_pair.append(time.time() - t0)

    t_pm = np.median(times_pm)
    t_pair = np.median(times_pair)

    max_diff = np.max(np.abs(result_pm - result_pair))
    rel_diff = max_diff / (np.max(np.abs(result_pm)) + 1e-15)
    peak_pm = np.unravel_index(np.argmax(result_pm), grid_shape)
    peak_pair = np.unravel_index(np.argmax(result_pair), grid_shape)

    print("PER-PAIR KERNEL (coarse scan)")
    print(f"  Point-major: {t_pm:.3f}s")
    print(f"  Pair-major:  {t_pair:.3f}s")
    print(f"  Speedup:     {t_pm / t_pair:.2f}x")
    print(f"  Max abs diff:  {max_diff:.2e}")
    print(f"  Max rel diff:  {rel_diff:.2e}")
    print(f"  Peak match:    {'YES' if peak_pm == peak_pair else 'NO'}")

    peak_idx = np.unravel_index(np.argmax(result_pm), grid_shape)
    rho_peak = rho_vec[peak_idx[0]]
    phi_peak = phi_vec[peak_idx[1]] * 180 / np.pi
    z_peak = z_vec[peak_idx[2]]

    refine_window = config.get('refine_window', [30, 5, 30])
    refine_steps = config.get('refine_step_sizes', [2, 0.5, 2])

    rho_r = np.arange(max(1, rho_peak - refine_window[0]),
                      rho_peak + refine_window[0] + refine_steps[0],
                      refine_steps[0])
    phi_r = np.arange(phi_peak - refine_window[1],
                      phi_peak + refine_window[1] + refine_steps[1],
                      refine_steps[1]) * (np.pi / 180.0)
    z_r = np.arange(max(-1500, z_peak - refine_window[2]),
                    min(0, z_peak + refine_window[2]) + refine_steps[2],
                    refine_steps[2])

    src_enu_r = reco._build_source_enu_matrix(rho_r, phi_r, z_r)
    tt_all_r = reco._compute_tt_multiray(src_enu_r, channels)

    grid_shape_r = None
    for ch in channels:
        for rt in tt_all_r.get(ch, {}):
            grid_shape_r = tt_all_r[ch][rt].shape
            break
        if grid_shape_r:
            break

    n_points_r = int(np.prod(grid_shape_r))
    tt_packed_r, ch_avail_rts_r = pack_tt_grids(tt_all_r, channels,
                                                 grid_shape_r)
    tt_t_r, _ = pack_tt_grids_transposed(tt_all_r, channels, grid_shape_r)
    corr_packed_r, corr_lengths_r, dts_r, offsets_r = pack_corr_data(
        corr_data, n_pairs)

    ch_to_group = {}
    depth_groups = {'pa': [0, 1, 2, 3], 'helper_b': [9, 10],
                    'helper_c': [22, 23]}
    gidx = 0
    for name, members in depth_groups.items():
        active = [ch for ch in members if ch in channels]
        if active:
            for ch_id in active:
                ci = channels.index(ch_id)
                ch_to_group[ci] = gidx
            gidx += 1
    for ci in range(n_ch):
        if ci not in ch_to_group:
            ch_to_group[ci] = gidx
            gidx += 1
    n_groups = gidx

    combo_table = build_combo_table(
        list(range(n_ch)),
        {i: ch_to_group.get(i, i) for i in range(n_ch)},
        n_groups, ch_avail_rts_r)

    _grouped_multiray_kernel(tt_packed_r, corr_packed_r, corr_lengths_r,
                             dts_r, offsets_r, pair_ch1, pair_ch2, pw,
                             combo_table, n_points_r)
    _grouped_multiray_kernel_pairmajor(tt_packed_r, corr_packed_r,
                                       corr_lengths_r, dts_r, offsets_r,
                                       pair_ch1, pair_ch2, pw,
                                       combo_table, n_points_r)

    print(f"\nRefine grid: {grid_shape_r}, {n_points_r} points, "
          f"{combo_table.shape[0]} combos")
    print(f"Benchmarking ({n_runs} runs each)...\n")

    times_g_pm = []
    for _ in range(n_runs):
        t0 = time.time()
        result_g_pm = _grouped_multiray_kernel(
            tt_packed_r, corr_packed_r, corr_lengths_r, dts_r, offsets_r,
            pair_ch1, pair_ch2, pw, combo_table, n_points_r)
        times_g_pm.append(time.time() - t0)

    times_g_pair = []
    for _ in range(n_runs):
        t0 = time.time()
        result_g_pair = _grouped_multiray_kernel_pairmajor(
            tt_packed_r, corr_packed_r, corr_lengths_r, dts_r, offsets_r,
            pair_ch1, pair_ch2, pw, combo_table, n_points_r)
        times_g_pair.append(time.time() - t0)

    t_g_pm = np.median(times_g_pm)
    t_g_pair = np.median(times_g_pair)

    max_diff_g = np.max(np.abs(result_g_pm - result_g_pair))
    rel_diff_g = max_diff_g / (np.max(np.abs(result_g_pm)) + 1e-15)
    peak_g_pm = np.argmax(result_g_pm)
    peak_g_pair = np.argmax(result_g_pair)

    print("GROUPED KERNEL (refine scan)")
    print(f"  Point-major: {t_g_pm:.3f}s")
    print(f"  Pair-major:  {t_g_pair:.3f}s")
    print(f"  Speedup:     {t_g_pm / t_g_pair:.2f}x")
    print(f"  Max abs diff:  {max_diff_g:.2e}")
    print(f"  Max rel diff:  {rel_diff_g:.2e}")
    print(f"  Peak match:    {'YES' if peak_g_pm == peak_g_pair else 'NO'}")

    reader.end()
    reco.end()


def main():
    """Parse arguments and run kernel benchmark."""
    parser = argparse.ArgumentParser(
        description='Benchmark pair-major vs point-major correlation kernels')
    parser.add_argument('--config', required=True, help='Reco config YAML')
    parser.add_argument('--nur-file', required=True, help='Input NUR file')
    parser.add_argument('--event-index', type=int, default=0,
                        help='Event index in file (default: 0)')
    parser.add_argument('--n-runs', type=int, default=5,
                        help='Timed iterations per kernel (default: 5)')
    parser.add_argument('--detector-file', default=None,
                        help='Detector JSON.xz file (overrides config)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')
    run_benchmark(args.config, args.nur_file, args.event_index, args.n_runs,
                  args.detector_file)


if __name__ == '__main__':
    main()
