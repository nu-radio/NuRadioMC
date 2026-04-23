#!/usr/bin/env python3
"""Driver script for 3D interferometric direction reconstruction on RNO-G data.

Pass 1: hierarchical 3D grid search (coarse scan, peak extraction, refine,
L-BFGS-B optimization) using multiray travel time tables.
Pass 2 (optional): antenna dedispersion at pass-1 estimated angles, then
local re-search. In rx mode, only the Rx antenna phase is removed. In rxtx
mode, both Rx and Tx antenna phases are removed (requires known emitter
position). Use hw mode to skip pass 2 entirely.

For a minimal reference example, see
``interferometric_reco_3d_simple.py`` in the same directory.
"""

import argparse
import datetime
import os
import yaml
import time
import numpy as np
import logging
import h5py

import NuRadioReco.detector.detector as detector
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.modules.channelAntennaDedispersion import channelAntennaDedispersion
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
from NuRadioReco.modules.RNO_G.dataProviderNuRadio import dataProviderNuRadio
from NuRadioReco.utilities import units
from NuRadioReco.detector.antennapattern import AntennaPatternProvider
from NuRadioMC.SignalProp.analyticraytracing import ray_tracing
from NuRadioMC.utilities.medium import greenland_simple

from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from NuRadioReco.framework.channel import Channel

logger = logging.getLogger("reco3d.iterative")

ice = greenland_simple()


def init_detector(config):
    """Build and update a Detector from config."""
    det_file = config.get('detector_file', None)
    det_date_str = config.get('detector_date', '2022-10-01')
    det_date = datetime.datetime.fromisoformat(det_date_str)
    station_id = config['station_id']

    if det_file:
        det = rnog_detector.Detector(
            detector_file=det_file,
            log_level=logging.WARNING,
            select_stations=station_id,
        )
    else:
        det = detector.Detector(source="rnog_mongo")

    det.update(det_date)
    return det


def compute_arrival_angles(rho, phi_deg, z, station_id, det, channels):
    """Compute per-channel arrival angles by ray tracing from a cylindrical position.

    Parameters
    ----------
    rho, phi_deg, z : float
        Cylindrical coordinates relative to station center (z absolute).
    station_id : int
    det : Detector
    channels : list of int

    Returns
    -------
    dict
        Maps channel_id -> (zenith_rad, azimuth_rad).
    """
    phi_rad = np.radians(phi_deg)
    stn_abs = np.array(det.get_absolute_position(station_id))
    source_abs = stn_abs + np.array([
        rho * np.cos(phi_rad), rho * np.sin(phi_rad), z - stn_abs[2]
    ])

    angles = {}
    rt = ray_tracing(ice, log_level=logging.WARNING)
    for ch_id in channels:
        ch_abs = stn_abs + np.array(det.get_relative_position(station_id, ch_id))
        rt.set_start_and_end_point(source_abs, ch_abs)
        rt.find_solutions()
        for iS in range(rt.get_number_of_solutions()):
            if rt.get_solution_type(iS) == 1:
                rv = rt.get_receive_vector(iS)
                zen = np.arccos(rv[2] / np.linalg.norm(rv))
                az = np.arctan2(rv[1], rv[0])
                angles[ch_id] = (zen, az)
                break
    return angles


def compute_launch_angles(emitter_abs, station_id, det, channels):
    """Compute per-channel launch angles from a known emitter position.

    Parameters
    ----------
    emitter_abs : array-like
        Absolute (x, y, z) position of the emitter.

    Returns
    -------
    dict
        Maps channel_id -> (zenith_rad, azimuth_rad).
    """
    launch_angles = {}
    stn_abs = np.array(det.get_absolute_position(station_id))
    rt = ray_tracing(ice, log_level=logging.WARNING)
    for ch_id in channels:
        ch_abs = stn_abs + np.array(det.get_relative_position(station_id, ch_id))
        rt.set_start_and_end_point(emitter_abs, ch_abs)
        rt.find_solutions()
        for iS in range(rt.get_number_of_solutions()):
            if rt.get_solution_type(iS) == 1:
                lv = rt.get_launch_vector(iS)
                zen = np.arccos(lv[2] / np.linalg.norm(lv))
                az = np.arctan2(lv[1], lv[0])
                launch_angles[ch_id] = (zen, az)
                break
    return launch_angles


def apply_rx_dedispersion(stn, det, station_id, channels, arrival_angles,
                          provider):
    """Remove Rx antenna phase response at estimated arrival angles."""
    for ch_id in channels:
        if ch_id not in arrival_angles:
            continue
        zen_arrival, az_arrival = arrival_angles[ch_id]
        channel = stn.get_channel(ch_id)
        ff = channel.get_frequencies()
        antenna_name = det.get_antenna_model(station_id, ch_id)
        antenna = provider.load_antenna_pattern(antenna_name)
        zen_ori, az_ori, zen_rot, az_rot = det.get_antenna_orientation(
            station_id, ch_id)
        VEL = antenna.get_antenna_response_vectorized(
            ff, zen_arrival, az_arrival, zen_ori, az_ori, zen_rot, az_rot)
        pol = "theta" if np.sum(np.abs(VEL['theta'])) > np.sum(np.abs(VEL['phi'])) else "phi"
        response = np.exp(1j * np.angle(VEL[pol]))
        spec = channel.get_frequency_spectrum() / response
        spec[0] = 0
        channel.set_frequency_spectrum(spec, channel.get_sampling_rate())


def apply_tx_dedispersion(stn, channels, launch_angles, provider,
                          tx_model, tx_ori):
    """Remove Tx antenna phase response at known launch angles."""
    tx_antenna = provider.load_antenna_pattern(tx_model)
    for ch_id in channels:
        if ch_id not in launch_angles:
            continue
        zen_launch, az_launch = launch_angles[ch_id]
        channel = stn.get_channel(ch_id)
        ff = channel.get_frequencies()
        VEL = tx_antenna.get_antenna_response_vectorized(
            ff, zen_launch, az_launch, *tx_ori)
        pol = "theta" if np.sum(np.abs(VEL['theta'])) > np.sum(np.abs(VEL['phi'])) else "phi"
        response = np.exp(1j * np.angle(VEL[pol]))
        spec = channel.get_frequency_spectrum() / response
        spec[0] = 0
        channel.set_frequency_spectrum(spec, channel.get_sampling_rate())


def parse_pulser_position(filename, pa_abs):
    """Extract emitter position from pulser scan filename.

    Parameters
    ----------
    filename : str
        Filename like 'output_r50.0_zen100.0_az90.0.nur'.
    pa_abs : array
        Absolute PA center position.

    Returns
    -------
    np.ndarray or None
        Absolute emitter position, or None if filename doesn't match pattern.
    """
    import re
    m = re.search(r'output_r([\d.]+)_zen([\d.]+)_az([\d.]+)\.', filename)
    if not m:
        return None
    r = float(m.group(1))
    zen = np.radians(float(m.group(2)))
    az = np.radians(float(m.group(3)))
    return pa_abs + r * np.array([
        np.sin(zen) * np.cos(az),
        np.sin(zen) * np.sin(az),
        np.cos(zen)
    ])


def _z_profile_pass2(reco2, evt, stn, det, config, config_p2_template,
                      r1, p2_window, z_step):
    """Profile-likelihood pass2: scan z, optimize (rho, phi) at each z.

    For sources at shallow zenith (near ice surface), the correlation landscape
    is nearly flat in z but has meaningful structure in (rho, phi). By
    optimizing (rho, phi) independently at each z candidate and comparing
    the resulting correlations, we break the rho-z degeneracy that causes
    the standard 3D search to converge to the wrong z.

    Parameters
    ----------
    reco2 : InterferometricReco3D
        Pre-initialized pass2 reco object (tables loaded).
    evt, stn, det : NuRadioReco objects
        Event, station, detector (already preprocessed with dedispersion).
    config : dict
        Full config with 'limits' defining the z search range.
    config_p2_template : dict
        Template for pass2 config.
    r1 : dict
        Pass1 result with 'rho', 'phi', 'z'.
    p2_window : list
        [rho_halfwidth, phi_halfwidth, z_halfwidth_unused].
    z_step : float
        Step size in meters for the z profile scan.

    Returns
    -------
    dict
        Best result across all z slices.
    """
    z_min = config['limits'][4]
    z_max = config['limits'][5]
    z_values = np.arange(z_min, z_max + z_step, z_step)

    rho_lo = max(1, r1['rho'] - p2_window[0])
    rho_hi = r1['rho'] + p2_window[0]
    phi_lo = r1['phi'] - p2_window[1]
    phi_hi = r1['phi'] + p2_window[1]

    best = None
    z_profile_log = []
    for z_c in z_values:
        z_lo = max(z_min, z_c - z_step / 2)
        z_hi = min(z_max, z_c + z_step / 2)
        config_p2 = dict(config_p2_template)
        config_p2['limits'] = [rho_lo, rho_hi, phi_lo, phi_hi, z_lo, z_hi]
        r2 = reco2.run(evt, stn, det, config_p2)
        z_profile_log.append((z_c, r2['rho'], r2['phi'], r2['z'],
                              r2['max_corr']))
        if best is None or r2['max_corr'] > best['max_corr']:
            best = r2

    if z_profile_log:
        logger.info("  z-profile: %s",
                     "  ".join(f"z={e[0]:.0f}:corr={e[4]:.3f},rho={e[1]:.1f}"
                               for e in z_profile_log))

    return best


from reco_validation import (
    PA_CHANNELS, SHALLOW_CHANNELS, HELPER_CHANNELS,
    compute_channel_snrs,
)


def check_helper_snr(station, threshold=5.0):
    """Check whether any helper channel exceeds SNR threshold.

    Args:
        station: NuRadioReco Station with preprocessed traces.
        threshold: Minimum SNR (max|V|/noise_rms) to count as signal.

    Returns:
        True if at least one helper channel is above threshold.
    """
    traces = []
    ch_ids = []
    for ch_id in HELPER_CHANNELS:
        try:
            traces.append(station.get_channel(ch_id).get_trace())
            ch_ids.append(ch_id)
        except (KeyError, AttributeError):
            continue
    if not traces:
        return False
    snrs = compute_channel_snrs(traces, ch_ids)
    return any(s >= threshold for s in snrs.values())


def make_fallback_config(config):
    """Create a plane-wave fallback config for PA-only zenith reconstruction.

    Uses only PA channels with a coarse phi grid (unconstrained dimension).

    Args:
        config: Original reco config dict.

    Returns:
        Modified config dict for plane-wave fallback.
    """
    fb = dict(config)
    fb['channels'] = PA_CHANNELS + SHALLOW_CHANNELS
    fb['coarse_step_sizes'] = [0, 60, 0]
    return fb


def main():
    """Run iterative 3D interferometric reconstruction on input events."""
    parser = argparse.ArgumentParser(
        description="3D interferometric direction reconstruction")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, nargs="+", required=True)
    parser.add_argument("-o", "--outputfile", type=str, required=True)
    parser.add_argument("--mode", type=str, default="rx",
                        choices=["hw", "rx", "rxtx"],
                        help="hw=pass1 only, rx=iterative Rx, rxtx=iterative Rx+Tx")
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--skip_events", type=int, default=0,
                        help="Skip this many events at the start of each file")
    parser.add_argument("--events", type=str, nargs="+", default=None,
                        help=(
                            "Filter to a subset of events. Accepts either a "
                            "space-separated list of integer event numbers, or "
                            "a path to a JSON file, auto-detected: run-keyed "
                            "{run: [events]} if keys parse as integers, else "
                            "file-aware {src: [[run, evt], ...]}. See "
                            "NuRadioReco.utilities.io_utilities.parse_event_ids."
                        ))
    parser.add_argument("--validation", action="store_true",
                        help="Record per-channel SNR and correlation quality metrics")
    parser.add_argument("--save-nur", type=str, default=None,
                        help="Write events with coherent WF channels to NUR file")
    parser.add_argument("--auto-gpu", action="store_true",
                        help="Detect an available GPU and enable the GPU "
                             "reco backend (overrides use_gpu in config).")
    parser.add_argument("--station_id", type=int, default=None,
                        help="Override config['station_id']. Useful when real "
                             "and sim inputs were produced for different "
                             "stations (e.g. running station-21 data against "
                             "station-23 sim NURs).")
    args = parser.parse_args()

    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or None
    host_cpus = os.cpu_count()
    has_gpu = False
    try:
        import cupy as _cp
        has_gpu = _cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        pass
    print(f"[reco3d] detected: cpus={slurm_cpus or host_cpus} "
          f"(SLURM={slurm_cpus}, host={host_cpus}), "
          f"cupy_device_count={_cp.cuda.runtime.getDeviceCount() if has_gpu else 0}")

    event_filter = None
    if args.events:
        from NuRadioReco.utilities.io_utilities import parse_event_ids
        event_filter = parse_event_ids(args.events)

    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key, val in config.items():
        if isinstance(val, str) and '$' in val:
            config[key] = os.path.expandvars(val)

    if args.validation:
        config['validation'] = True

    if args.auto_gpu:
        if has_gpu:
            config['use_gpu'] = True
            print("[reco3d] --auto-gpu: enabling GPU backend")
        else:
            print("[reco3d] --auto-gpu: no GPU detected, staying on CPU")

    if args.station_id is not None:
        config['station_id'] = args.station_id

    det = init_detector(config)
    station_id = config['station_id']
    channels = config['channels']

    station_pos = np.array(det.get_absolute_position(station_id))
    ch1_rel = np.array(det.get_relative_position(station_id, 1))
    ch2_rel = np.array(det.get_relative_position(station_id, 2))
    pa_abs = station_pos + 0.5 * (ch1_rel + ch2_rel)

    # Shared preprocessing chain (block-offset, glitch, cable delay,
    # hw phase removal, CW, bandpass) runs inside the dataProvider
    # instantiated per input file below. Upsampling and antenna
    # dedispersion stay driver-owned because reco's two-pass flow orders
    # them around rx/tx dedispersion; always force apply_upsampling=False
    # in the preprocessor config so the driver controls upsampling
    # externally via the explicit resampler calls.
    preproc_config = dict(config.get('preprocessor', {}))
    preproc_config['apply_upsampling'] = False

    resampler = channelResampler(); resampler.begin()
    antenna_dedispersion = channelAntennaDedispersion()

    provider = AntennaPatternProvider()
    tx_model = config.get('tx_antenna_model', 'RNOG_vpol_v3_5inch_center_n1.74')
    tx_ori = config.get('tx_antenna_orientation', [0, 0, 90 * units.deg, 0])

    reco1 = InterferometricReco3D()
    reco1.begin(station_id, config, det)

    # Enforce pass 2 steps no coarser than refine grid
    p2_window = config.get('pass2_window', [50, 20, 50])
    p2_steps = config.get('pass2_step_sizes', [2, 1, 2])
    refine_steps = config.get('refine_step_sizes', [5, 1, 5])
    p2_steps = [min(p, r) for p, r in zip(p2_steps, refine_steps)]

    # Reusable pass 2 reco object (tables are the same, only limits change)
    reco2 = None
    if args.mode != "hw":
        p2_hierarchical = config.get('pass2_hierarchical', False)
        config_p2_template = {
            'station_id': station_id, 'channels': channels,
            'coord_system': 'cylindrical',
            'hierarchical': p2_hierarchical,
            'multi_ray_types': config.get('multi_ray_types', False),
            'multiray_combo_mode': config.get('multiray_combo_mode', 'grouped'),
            'time_delay_tables': config['time_delay_tables'],
            'interp_method': config.get('interp_method', 'linear'),
            'snr_pair_weighting': config.get('snr_pair_weighting', True),
            'hilbert_envelope_mode': config.get('hilbert_envelope_mode', 'traces'),
            'correlation_normalization': config.get('correlation_normalization', 'normalized'),
            'apply_hann_window': config.get('apply_hann_window', False),
            'limits': [1, 100, 0, 360, -100, 0],
            'step_sizes': p2_steps,
            'n_rho': 0,
            'n_z': config.get('pass2_n_z', config.get('n_z', 0)),
            'z_spacing': config.get('z_spacing', 'linear'),
            'z_surface_offset': config.get('z_surface_offset', 0.1),
        }
        if p2_hierarchical:
            config_p2_template.update({
                'coarse_limits': [1, 100, 0, 360, -100, 0],
                'coarse_n_rho': 0,
                'coarse_n_z': config.get('pass2_coarse_n_z', 0),
                'coarse_step_sizes': config.get(
                    'pass2_coarse_step_sizes', [2, 1, 2]),
                'coarse_n_peaks': config.get('pass2_coarse_n_peaks', 3),
                'coarse_peak_separation': config.get(
                    'pass2_coarse_peak_separation', [5, 3, 5]),
                'refine_window': config.get(
                    'pass2_refine_window', [5, 3, 5]),
                'refine_step_sizes': p2_steps,
            })
        if 'table_name_pattern' in config:
            config_p2_template['table_name_pattern'] = config['table_name_pattern']
        if 'multiray_table_name_pattern' in config:
            config_p2_template['multiray_table_name_pattern'] = config['multiray_table_name_pattern']
        reco2 = InterferometricReco3D()
        reco2.begin(station_id, config_p2_template, det)

    COH_WF_CHANNEL_BASE = 100

    nur_writer = None
    if args.save_nur:
        from NuRadioReco.modules.io.eventWriter import eventWriter
        from NuRadioReco.framework.event import Event as NREvent
        from NuRadioReco.framework.station import Station as NRStation
        nur_writer = eventWriter()
        nur_writer.begin(args.save_nur)

    results = []
    n_processed = 0
    t_total = 0

    is_nur = args.input[0].endswith('.nur')

    for input_file in args.input:
        file_basename = os.path.basename(input_file)

        if event_filter is not None and 'by_file' in event_filter \
                and file_basename not in event_filter['by_file']:
            continue

        if is_nur:
            data_provider = dataProviderNuRadio()
            data_provider.begin(
                input_file, det, preprocessor_config=preproc_config)
        else:
            data_provider = dataProviderRNOG()
            data_provider.begin(
                input_file, det,
                reader_kwargs={'mattak_kwargs': {
                    'read_daq_status': False, 'backend': 'uproot'}},
                preprocessor_config=preproc_config,
            )
        event_ids = data_provider.get_event_ids()

        emitter_pos = None
        launch_angles = {}
        if args.mode == "rxtx":
            emitter_pos = parse_pulser_position(
                file_basename, pa_abs)
            if emitter_pos is not None:
                launch_angles = compute_launch_angles(
                    emitter_pos, station_id, det, channels)

        event_ids = event_ids[args.skip_events:]

        for eid in event_ids:
            run_nr = int(eid[0])
            evt_nr = int(eid[1])
            if event_filter is not None:
                if 'by_file' in event_filter:
                    if (run_nr, evt_nr) not in event_filter['by_file'][file_basename]:
                        continue
                elif 'by_run' in event_filter:
                    if run_nr not in event_filter['by_run'] \
                            or evt_nr not in event_filter['by_run'][run_nr]:
                        continue
                elif 'by_event' in event_filter:
                    if evt_nr not in event_filter['by_event']:
                        continue

            t0 = time.time()

            t_preproc_start = time.time()
            evt1 = data_provider.get_event(int(eid[0]), int(eid[1]))
            stn1 = evt1.get_station(station_id)
            # channelPreprocessor already ran inside data_provider;
            # apply the driver-owned pass-1 steps (upsample + optional
            # generic dedispersion).
            if config.get('apply_upsampling', True):
                resampler.run(evt1, stn1, det,
                              sampling_rate=10 * units.GHz)
            if config.get('apply_dedispersion', False):
                antenna_dedispersion.run(evt1, stn1, det)
            t_preproc = time.time() - t_preproc_start

            use_fallback = False
            pw_threshold = config.get('plane_wave_snr_threshold', 5.0)
            if config.get('plane_wave_fallback', False):
                if not check_helper_snr(stn1, threshold=pw_threshold):
                    use_fallback = True

            p1_config = make_fallback_config(config) if use_fallback else config
            t_p1_start = time.time()
            r1 = reco1.run(evt1, stn1, det, p1_config)
            t_p1 = time.time() - t_p1_start
            tag = " [FALLBACK]" if use_fallback else ""
            logger.info("  pass1%s: %.2fs (preproc: %.2fs)",
                        tag, t_p1, t_preproc)
            r1['preproc_time'] = t_preproc

            if use_fallback:
                r1['phi'] = np.nan
                r1['plane_wave_fallback'] = 1
                result = r1
            elif args.mode == "hw":
                result = r1
            else:
                # Re-read the event for pass 2 so we start from fresh
                # traces with only channelPreprocessor applied (no
                # upsampling, no generic dedispersion). rx/tx dedispersion
                # below uses arrival angles from pass 1.
                evt2 = data_provider.get_event(int(eid[0]), int(eid[1]))
                stn2 = evt2.get_station(station_id)

                t_p2_pre = time.time()
                rx_angles = compute_arrival_angles(
                    r1['rho'], r1['phi'], r1['z'],
                    station_id, det, channels)
                apply_rx_dedispersion(stn2, det, station_id, channels,
                                      rx_angles, provider)

                if args.mode == "rxtx" and launch_angles:
                    apply_tx_dedispersion(stn2, channels, launch_angles,
                                          provider, tx_model, tx_ori)

                resampler.run(evt2, stn2, det, sampling_rate=10 * units.GHz)
                t_p2_dedisp = time.time() - t_p2_pre

                z_profile_step = config.get('z_profile_step', None)
                if z_profile_step is not None:
                    r2 = _z_profile_pass2(
                        reco2, evt2, stn2, det, config, config_p2_template,
                        r1, p2_window, z_profile_step)
                else:
                    config_p2 = dict(config_p2_template)
                    p2_limits = [
                        max(1, r1['rho'] - p2_window[0]),
                        r1['rho'] + p2_window[0],
                        r1['phi'] - p2_window[1],
                        r1['phi'] + p2_window[1],
                        r1['z'] - p2_window[2],
                        r1['z'] + p2_window[2],
                    ]
                    config_p2['limits'] = p2_limits
                    if p2_hierarchical:
                        config_p2['coarse_limits'] = p2_limits
                    t_p2_reco_start = time.time()
                    r2 = reco2.run(evt2, stn2, det, config_p2)
                    t_p2_reco = time.time() - t_p2_reco_start

                logger.info("  pass2: dedisp=%.2fs, reco=%.2fs",
                            t_p2_dedisp, t_p2_reco)
                result = r2
                result['pass1_rho'] = r1['rho']
                result['pass1_phi'] = r1['phi']
                result['pass1_z'] = r1['z']
                result['pass1_corr'] = r1['max_corr']
                result['p1_preproc_time'] = t_preproc
                result['p1_total_time'] = t_p1
                for k in ['coarse_time', 'refine_time', 'opt_time']:
                    if k in r1:
                        result['p1_' + k] = r1[k]
                result['p2_dedisp_time'] = t_p2_dedisp
                result['p2_reco_time'] = t_p2_reco
                for k in ['coarse_time', 'refine_time', 'opt_time',
                           'grid_time']:
                    if k in r2:
                        result['p2_' + k] = r2[k]

            dt_evt = time.time() - t0
            t_total += dt_evt

            result['run_number'] = int(eid[0])
            result['event_number'] = int(eid[1])
            result['source_file'] = os.path.basename(input_file)

            if nur_writer is not None:
                n_coh = config.get('n_coherent_waveforms', 1)
                coh_ch_ids = [COH_WF_CHANNEL_BASE + i for i in range(n_coh)]
                stn1_out = evt1.get_station(station_id)
                for wf_key in sorted(k for k in result
                                     if k.startswith('coherent_wf_')):
                    pk_idx = int(wf_key.split('_')[-1])
                    ch_id = COH_WF_CHANNEL_BASE + pk_idx
                    ch = Channel(channel_id=ch_id)
                    wf_times = result.get('coherent_times')
                    sr = 1.0 / (wf_times[1] - wf_times[0]) * 1e9 if wf_times is not None else 10e9
                    ch.set_trace(result[wf_key], sr)
                    stn1_out.add_channel(ch)
                evt_out = NREvent(evt1.get_run_number(), evt1.get_id())
                stn_out = NRStation(station_id)
                for ch_id in coh_ch_ids:
                    if stn1_out.has_channel(ch_id):
                        stn_out.add_channel(stn1_out.get_channel(ch_id))
                evt_out.set_station(stn_out)
                nur_writer.run(evt_out)

            results.append(result)

            n_processed += 1
            if n_processed % 50 == 0:
                logger.info("  %d events, %.1fs elapsed, %.2fs/evt",
                            n_processed, t_total, t_total / n_processed)

            if args.max_events and n_processed >= args.max_events:
                break

        data_provider.end()
        if args.max_events and n_processed >= args.max_events:
            break

    reco1.end()
    if reco2 is not None:
        reco2.end()

    if nur_writer is not None:
        nur_writer.end()

    if results:
        outdir = os.path.dirname(args.outputfile)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        numeric_keys = ['rho', 'phi', 'z', 'max_corr']

        if args.validation:
            channels = config['channels']
            for ch in channels:
                numeric_keys.append(f'ch{ch}_snr')
            numeric_keys.extend([
                'surf_corr_z', 'surf_corr_zen', 'peak_isolation_ratio',
                'pa_avg_snr', 'pa_max_snr',
                'helper_b_max_snr', 'helper_b_min_snr',
                'helper_c_max_snr', 'helper_c_min_snr',
            ])

        optional_keys = ['pass1_rho', 'pass1_phi', 'pass1_z', 'pass1_corr',
                         'grid_time', 'opt_time', 'coarse_time', 'refine_time',
                         'preproc_time', 'post_time', 'raw_refine_time',
                         'peak_time',
                         'p1_preproc_time', 'p1_total_time',
                         'p1_coarse_time', 'p1_refine_time', 'p1_opt_time',
                         'p2_dedisp_time', 'p2_reco_time',
                         'p2_coarse_time', 'p2_refine_time', 'p2_opt_time',
                         'p2_grid_time',
                         'plane_wave_fallback', 'n_saved_peaks']

        # Discover multi-peak and per-polarization keys from first result
        if results:
            existing = set(numeric_keys) | set(optional_keys)
            skip = {'run_number', 'event_number', 'source_file',
                    'coarse_peaks', 'coherent_times'}
            for k in sorted(results[0]):
                if k in existing or k in skip:
                    continue
                if k.startswith('peak_') or k.endswith(('_vpol', '_hpol')):
                    optional_keys.append(k)
                elif isinstance(results[0][k], (int, float, np.floating)):
                    optional_keys.append(k)

        for k in optional_keys:
            if k in results[0] and k not in numeric_keys:
                numeric_keys.append(k)

        with h5py.File(args.outputfile, 'w') as f:
            grp = f.create_group('results')
            for key in numeric_keys:
                grp.create_dataset(
                    key,
                    data=np.array([r.get(key, np.nan) for r in results]))
            grp.create_dataset(
                'run_number',
                data=np.array([r['run_number'] for r in results], dtype=int))
            grp.create_dataset(
                'event_number',
                data=np.array([r['event_number'] for r in results], dtype=int))

            filenames = [r.get('source_file', '') for r in results]
            dt = h5py.special_dtype(vlen=str)
            grp.create_dataset('source_file', data=filenames, dtype=dt)

            if results and 'coherent_times' in results[0]:
                wf_grp = f.create_group('coherent_waveforms')
                wf_grp.create_dataset(
                    'times', data=results[0]['coherent_times'])
                for wf_key in sorted(k for k in results[0]
                                     if k.startswith('coherent_wf_')):
                    peak_idx = wf_key.split('_')[-1]
                    wfs = np.array([r.get(wf_key, np.zeros_like(
                        results[0][wf_key])) for r in results])
                    wf_grp.create_dataset(f'peak_{peak_idx}', data=wfs)

            if args.validation:
                for val_key, val_dtype, val_default in [
                    ('n_helpers_above', int, 0),
                    ('n_channels_above', int, 0),
                    ('has_helper_signal', bool, False),
                ]:
                    if val_key not in grp:
                        grp.create_dataset(
                            val_key,
                            data=np.array([r.get(val_key, val_default)
                                           for r in results], dtype=val_dtype))

            f.attrs['mode'] = args.mode
            f.attrs['n_events'] = n_processed
            f.attrs['validation'] = args.validation

        logger.info("Saved %d results to %s", len(results), args.outputfile)

    print(f"\nProcessed {n_processed} events in {t_total:.2f}s "
          f"({t_total/max(n_processed,1):.3f}s/evt)")


if __name__ == "__main__":
    main()
