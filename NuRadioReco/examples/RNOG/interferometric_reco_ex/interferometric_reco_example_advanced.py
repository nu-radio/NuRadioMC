#!/usr/bin/env python3
"""
This script provides a command-line interface for running interferometric direction 
reconstruction using pre-calculated time delay tables.

Usage:
    python interferometric_reco_example.py --config config.yaml --inputfile data.root --outputfile output.h5

Example:
    python interferometric_reco_example.py \
        --config example_config.yaml \
        --inputfile /path/to/station21_run476.root \
        --outputfile reconstruction_results.h5 \
"""

import argparse
from datetime import datetime
import os, sys
import yaml
import gc
import time
import numpy as np
import logging

from NuRadioReco.utilities.logging import set_general_log_level
# Set NuRadioReco logging level (INFO=20, DEBUG=10)
set_general_log_level(logging.WARNING)
logger = logging.getLogger("NuRadioReco.modules.interferometricDirectionReconstruction")

from NuRadioReco.utilities import units
from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.channelSinewaveSubtraction import channelSinewaveSubtraction
from NuRadioReco.modules.channelAddCableDelay import channelAddCableDelay
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.detector import detector
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters
import NuRadioReco.utilities.trace_utilities as trace_utils
from NuRadioReco.modules.interferometricDirectionReconstruction import interferometricDirectionReconstruction
from NuRadioReco.utilities.interferometry_io_utilities import (
    save_interferometric_results_hdf5 as save_results_to_hdf5,
    save_interferometric_results_nur as save_results_to_nur,
    save_correlation_map,
    create_organized_paths
)

def get_PA_position(station_id, det):
    """
    Calculate phased array (PA) center position.
    
    Returns both relative to station and absolute coordinates.
    This should be called once per station, not per event.
    
    Parameters
    ----------
    station_id : int
        Station ID
    det : Detector
        Detector object
    
    Returns
    -------
    tuple : (pa_pos_rel_station, pa_pos_abs)
        PA center relative to station [x, y, z] and absolute position [x, y, z]
    """
    station_pos_abs = np.array(det.get_absolute_position(station_id))
    ch1_pos_rel = np.array(det.get_relative_position(station_id, 1))
    ch2_pos_rel = np.array(det.get_relative_position(station_id, 2))
    pa_pos_rel_station = 0.5 * (ch1_pos_rel + ch2_pos_rel)  # PA center relative to station
    pa_pos_abs = station_pos_abs + pa_pos_rel_station  # PA absolute position
    
    return pa_pos_rel_station, pa_pos_abs

def get_sim_vertex_rel_PA(event_object, pa_pos_abs):
    """
    Get simulation vertex position relative to PA center.
    
    Parameters
    ----------
    event_object : Event
        NuRadioReco Event object
    pa_pos_abs : np.ndarray
        Absolute position of PA center [x, y, z]
    
    Returns
    -------
    np.ndarray
        Vertex position relative to PA center [x, y, z]
    """
    interaction_vertex_abs = list(event_object.get_sim_showers())[0].get_parameter(showerParameters.vertex)
    interaction_vertex_rel_PA = np.array(interaction_vertex_abs) - pa_pos_abs
    
    return interaction_vertex_rel_PA

def get_sim_truth_fixed_coord(config, event_object, pa_pos_abs):
    """
    Get simulation truth value for fixed coordinate.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    event_object : Event
        NuRadioReco Event object
    pa_pos_abs : np.ndarray
        Absolute position of PA center [x, y, z]
    
    Returns
    -------
    float
        Fixed coordinate value in appropriate units
    """
    interaction_vertex_rel_PA = get_sim_vertex_rel_PA(event_object, pa_pos_abs)
    x_rel_PA, y_rel_PA, z_rel_PA = interaction_vertex_rel_PA
    
    rho_rel_PA = np.sqrt(x_rel_PA**2 + y_rel_PA**2)
    r_rel_PA = np.sqrt(x_rel_PA**2 + y_rel_PA**2 + z_rel_PA**2)
    
    zenith_rel_PA = np.degrees(np.arccos(z_rel_PA / r_rel_PA))
    azimuth_rel_PA = np.degrees(np.arctan2(y_rel_PA, x_rel_PA)) % 360
    
    if config['coord_system'] == 'cylindrical':
        if config['rec_type'] == 'rhoz':
            return azimuth_rel_PA
        elif config['rec_type'] == 'phiz':
            return rho_rel_PA
    elif config['coord_system'] == 'spherical':
        if config['rec_type'] == 'phitheta':
            return r_rel_PA    

def calculate_channel_snr(trace):
    """
    Calculate SNR for a channel using split-trace RMS method
    
    Parameters
    ----------
    trace : np.ndarray
        Voltage trace for the channel
    
    Returns
    -------
    float
        Signal-to-noise ratio
    """
    
    noise_rms = trace_utils.get_split_trace_noise_RMS(trace)
    snr = trace_utils.get_signal_to_noise_ratio(trace, noise_rms)
    
    return snr

def detect_edge_signal(trace, n_chunks=10, edge_threshold_sigma=3.0):
    """
    Detect if signal is cut off at edge of trace window.
    
    Divides trace into chunks and compares edge chunk power to middle chunks.
    Flags as edge signal if edge power exceeds median + N*std of middle chunks.
    
    Parameters
    ----------
    trace : np.ndarray
        Voltage trace for the channel
    n_chunks : int, optional
        Number of chunks to divide trace into (default: 10)
    edge_threshold_sigma : float, optional
        Number of standard deviations above median to flag as edge (default: 3.0)
    
    Returns
    -------
    tuple : (is_edge, debug_info)
        is_edge : bool - True if edge signal detected
        debug_info : dict - Information about detection for logging
    """
    # Divide trace into chunks
    chunk_size = len(trace) // n_chunks
    if chunk_size < 10:  # Need reasonable chunk size
        return False, {}
    
    # Calculate RMS for each chunk
    chunk_rms = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else len(trace)
        chunk = trace[start:end]
        chunk_rms.append(np.std(chunk))
    
    chunk_rms = np.array(chunk_rms)
    
    # Get statistics from middle chunks (exclude first and last)
    middle_chunks = chunk_rms[1:-1]
    median_middle = np.median(middle_chunks)
    std_middle = np.std(middle_chunks)
    
    # Check if either edge exceeds threshold
    threshold = median_middle + edge_threshold_sigma * std_middle
    
    first_edge_high = chunk_rms[0] > threshold
    last_edge_high = chunk_rms[-1] > threshold
    
    is_edge = first_edge_high or last_edge_high
    
    debug_info = {
        'first_rms': chunk_rms[0],
        'last_rms': chunk_rms[-1],
        'median_middle': median_middle,
        'std_middle': std_middle,
        'threshold': threshold,
        'first_edge_high': first_edge_high,
        'last_edge_high': last_edge_high
    }
    
    return is_edge, debug_info

def filter_channels_by_edge_signal(event_station, channels, edge_threshold_sigma=3.0, n_chunks=10, verbose=False):
    """
    Filter out channels with signals at trace edges.
    
    Parameters
    ----------
    event_station : Station
        Station object containing channels
    channels : list
        List of channel IDs to check
    edge_threshold_sigma : float, optional
        Number of standard deviations above median to flag as edge (default: 3.0)
    n_chunks : int, optional
        Number of chunks to divide trace into (default: 10)
    verbose : bool, optional
        Print edge detection information (default: False)
    
    Returns
    -------
    list
        List of channel IDs without edge signals
    """
    passing_channels = []
    edge_channels = []
    
    for ch in channels:
        try:
            channel = event_station.get_channel(ch)
            trace = channel.get_trace()
            
            is_edge, debug_info = detect_edge_signal(trace, n_chunks=n_chunks, edge_threshold_sigma=edge_threshold_sigma)
            
            if not is_edge:
                passing_channels.append(ch)
            else:
                edge_channels.append(ch)
                # Log which edge triggered and the values
                edge_location = []
                if debug_info.get('first_edge_high', False):
                    edge_location.append(f"START (RMS={debug_info['first_rms']:.2f} > {debug_info['threshold']:.2f})")
                if debug_info.get('last_edge_high', False):
                    edge_location.append(f"END (RMS={debug_info['last_rms']:.2f} > {debug_info['threshold']:.2f})")
                
                print(f"    Channel {ch} DROPPED: Edge signal detected at {' and '.join(edge_location)}")
                
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not check edge signal for channel {ch}: {e}")
            # If we can't check, assume it's okay
            passing_channels.append(ch)
    
    if verbose and len(edge_channels) > 0:
        print(f"  Summary: {len(edge_channels)} channel(s) dropped due to edge signals: {edge_channels}")
        print(f"  Summary: {len(passing_channels)} channel(s) passed edge detection: {passing_channels}")
    
    return passing_channels

def filter_channels_by_snr(event_station, channels, snr_threshold, helper_channels=[9, 10, 22, 23], verbose=False):
    """
    Filter channels based on SNR threshold and check if helper channels pass.
    
    Parameters
    ----------
    event_station : Station
        Station object containing channels
    channels : list
        List of channel IDs to check
    snr_threshold : float
        Minimum SNR required
    helper_channels : list, optional
        List of helper channel IDs that must have at least one passing (default: [9, 10, 22, 23])
    verbose : bool, optional
        Print SNR information (default: False)
    
    Returns
    -------
    tuple : (passing_channels, should_skip_event)
        passing_channels : list of channel IDs that passed threshold
        should_skip_event : bool, True if no helper channels passed threshold
    """
    passing_channels = []
    failed_channels = []
    channel_snrs = {}
    
    for ch in channels:
        try:
            channel = event_station.get_channel(ch)
            trace = channel.get_trace()
            snr = calculate_channel_snr(trace)
            channel_snrs[ch] = snr
            
            if snr >= snr_threshold:
                passing_channels.append(ch)
            else:
                failed_channels.append(ch)
                print(f"    Channel {ch} DROPPED: SNR too low (SNR={snr:.2f} < threshold={snr_threshold:.2f})")
        except Exception as e:
            print(f"    Channel {ch} DROPPED: Could not calculate SNR ({e})")
            continue
    
    # Check if at least one helper channel passed
    helper_channels_passing = [ch for ch in helper_channels if ch in passing_channels]
    should_skip_event = len(helper_channels_passing) == 0
    
    if verbose:
        print(f"  Channel SNRs: {channel_snrs}")
        if len(failed_channels) > 0:
            print(f"  Summary: {len(failed_channels)} channel(s) dropped due to low SNR: {failed_channels}")
        print(f"  Summary: {len(passing_channels)} channel(s) passed SNR threshold: {passing_channels}")
        print(f"  Helper channels passing: {helper_channels_passing}")
        if should_skip_event:
            print(f"  WARNING: No helper channels [{','.join(map(str, helper_channels))}] passed SNR threshold")
    
    return passing_channels, should_skip_event

def run_two_stage_reconstruction(reco, event, event_station, det, config, pa_pos_abs, save_maps, save_pair_maps, save_maps_to):
    """
    Run two-stage automatic reconstruction:
    1. Stage 1 (rhoz): Find optimal radial distance and depth
    2. Stage 2 (spherical): Use that distance to find azimuth and zenith
    
    Parameters
    ----------
    reco : interferometricDirectionReconstruction
        Reconstruction module instance
    event : Event
        NuRadioReco Event object
    event_station : Station
        Station object from event
    det : Detector
        Detector object
    config : dict
        Configuration dictionary (should have mode='auto')
    pa_pos_abs : np.ndarray
        Absolute position of PA center [x, y, z]
    save_maps : bool
        Whether to save correlation maps
    save_pair_maps : bool
        Whether to save pair correlation maps
    save_maps_to : str
        Directory to save maps
        
    Returns
    -------
    corr_map_path : str
        Path to saved correlation map (from final stage)
    """
    
    # Stage 1: rhoz reconstruction to find optimal distance and depth
    # Keep existing limits and step_sizes for stage 1 (should be in config)
    stage1_config = config.copy()
    stage1_config['limits'] = [0, 200, -200, 0]
    stage1_config['channels'] = [0, 1, 2, 3, 5, 6, 7] # [0, 1, 2, 3, 5, 6, 7]
    stage1_config['fixed_coord'] = 0
    stage1_config['coord_system'] = 'cylindrical'
    stage1_config['rec_type'] = 'rhoz'
    stage1_config['find_alternate_reco'] = False
    
    logger.info(f"[AUTO MODE] Stage 1: Running rhoz reconstruction to find optimal distance...")
    
    # Run stage 1 reconstruction (don't save maps from this stage)
    reco.run(event, event_station, det, stage1_config, 
             save_maps=False, save_pair_maps=False, save_maps_to=None)
    
    # Extract stage 1 results
    rho_best = event_station.get_parameter(stnp.rec_coord0)  # In internal units (meters)
    z_best_abs = event_station.get_parameter(stnp.rec_coord1)  # Absolute depth in meters
    stage1_max_corr = event_station.get_parameter(stnp.rec_max_correlation)
    
    rho_best_m = rho_best / units.m
    z_best_abs_m = z_best_abs / units.m
    
    # Convert z from absolute depth to PA-relative depth
    # z_best_abs is absolute depth (negative below surface)
    # pa_pos_abs[2] is PA absolute z position
    # z_rel_PA is depth relative to PA
    z_rel_PA_m = z_best_abs_m - pa_pos_abs[2]
    
    # Calculate radial distance from PA center using PA-relative coordinates
    r_best_m = np.sqrt(rho_best_m**2 + z_rel_PA_m**2)
    
    logger.info(f"[AUTO MODE] Stage 1 results: rho={rho_best_m:.1f}m, z_abs={z_best_abs_m:.1f}m, z_rel_PA={z_rel_PA_m:.1f}m, r={r_best_m:.1f}m, maxCorr={stage1_max_corr:.3f}")
    
    # Stage 2: Spherical reconstruction with fixed radial distance
    stage2_config = config.copy()
    stage2_config['coord_system'] = 'spherical'
    stage2_config['rec_type'] = 'phitheta'  # Not strictly needed for spherical but for consistency
    stage2_config['fixed_coord'] = r_best_m  # Use calculated radial distance
    stage2_config['limits'] = [0, 360, 0, 180]  # Full sky coverage: phi [0, 360°], theta [0, 180°]
    # Keep step_sizes from original config
    
    logger.info(f"[AUTO MODE] Stage 2: Running spherical reconstruction with fixed r={r_best_m:.1f}m...")
    
    # Run stage 2 reconstruction (save maps from this final stage if requested)
    corr_map_path = reco.run(event, event_station, det, stage2_config,
                             save_maps=save_maps, save_pair_maps=save_pair_maps, 
                             save_maps_to=save_maps_to)
    
    # Final results are now in station parameters (overwritten by stage 2)
    final_max_corr = event_station.get_parameter(stnp.rec_max_correlation)
    phi_best = event_station.get_parameter(stnp.rec_coord0) / units.deg
    theta_best = event_station.get_parameter(stnp.rec_coord1) / units.deg
    
    #print(f"[AUTO MODE] Stage 2 results: phi={phi_best:.1f}°, theta={theta_best:.1f}°, maxCorr={final_max_corr:.3f}")
    
    return corr_map_path

def main():    
    parser = argparse.ArgumentParser(
        prog="interferometric_reco_example.py", 
        description="Run interferometric direction reconstruction on RNO-G data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic reconstruction (saves to ./results/station{ID}/run{NUM}/)
            python interferometric_reco_example.py --config example_config.yaml --inputfile data.root

            # Process specific events with map data saving
            python interferometric_reco_example.py --config example_config.yaml --inputfile data.root --events 1 5 10 --save_maps --verbose

            # Process specific runs from NUR simulation file
            python interferometric_reco_example.py --config example_config.yaml --inputfile simulation.nur --runs 0 5 10 --save_maps

            # Use NUR output format instead of HDF5
            python interferometric_reco_example.py --config example_config.yaml --inputfile file1.root --output_type nur
        """
    )
    
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to YAML configuration file")
    parser.add_argument("--inputfile", type=str, nargs="+", required=True,
                       help="Path(s) to input data file(s) (ROOT or NUR). Can specify multiple files for same station.")
    parser.add_argument("--output_type", type=str, choices=['hdf5', 'nur'], default='hdf5',
                       help="Output file format: 'hdf5' for HDF5 tables or 'nur' for NuRadioReco format (default: hdf5)")
    parser.add_argument("--outputfile", type=str, default=None,
                       help="Optional: manually specify output file path. If not set, uses organized default path.")
    parser.add_argument("--events", type=int, nargs="*", default=None, 
                       help="Specific event IDs to process. If not provided, processes all events in given file.")
    parser.add_argument("--runs", type=int, nargs="*", default=None,
                       help="Specific run numbers to process.")

    parser.add_argument("--sim-truth-fixed-coord", action="store_true")
    parser.add_argument("--use-cache", required=False, default=False)
    parser.add_argument("--save-maps", action="store_true",
                       help="Save correlation map data to pickle files for later plotting")
    parser.add_argument("--save-pair-maps", action="store_true",
                       help="Save individual channel pair correlation maps (requires --save-maps)")
    parser.add_argument("--snr-threshold", type=float, default=None,
                       help="SNR threshold for channel filtering. Channels below threshold are dropped. If no helper channels [9,10,22,23] pass threshold, event is skipped.")
    parser.add_argument("--edge-sigma", type=float, default=None,
                       help="Edge signal detection threshold in standard deviations. Channels with signals at trace edges exceeding this threshold are dropped. If no helper channels [9,10,22,23] remain, event is skipped.")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print reconstruction results for each event")

    args = parser.parse_args()
        
    if args.events is not None and args.runs is not None:
        raise ValueError("Cannot specify both --events and --runs. Use --events for ROOT files and --runs for NUR files.")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    reco_mode = config.get('mode', 'manual')
    if reco_mode not in ['manual', 'auto']:
        raise ValueError(f"Invalid mode '{reco_mode}'. Must be 'manual' or 'auto'")

    required_params = ['coord_system', 'channels', 'limits', 'step_sizes', 'time_delay_tables']
    for param in required_params:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' not found in config file")
    
    if reco_mode == 'manual' and 'fixed_coord' not in config and not args.sim_truth_fixed_coord:
        raise ValueError("In 'manual' mode, 'fixed_coord' must be specified in config (or use --sim_truth_fixed_coord for simulations)")
    
    if reco_mode == 'auto':
        if param not in config:
            raise ValueError(f"Required parameter '{param}' not found in config file")

    input_files = args.inputfile if isinstance(args.inputfile, list) else [args.inputfile]
    is_nur_file = input_files[0].endswith('.nur')
    
    if args.sim_truth_fixed_coord and not is_nur_file:
        logging.error("Wrong file type! Expected a .nur file when sim_truth_fixed_coord passed.")
        sys.exit(1)
    
    # Warn if --runs is used with ROOT files
    if args.runs is not None and not is_nur_file:
        print("WARNING: --runs flag is not supported for ROOT files (real data).")
        print("    ROOT files contain data from a single run. Use --events instead.")
        print("    Ignoring --runs flag and processing all events.")
        args.runs = None
        
    channel_add_cable_delay = channelAddCableDelay()
    channel_add_cable_delay.begin()

    channel_resampler = channelResampler()
    channel_resampler.begin()
    
    channel_bandpass_filter = channelBandPassFilter()
    channel_bandpass_filter.begin()
    
    cw_filter = channelSinewaveSubtraction()
    cw_filter.begin(
        save_filtered_freqs=False,
        freq_band=tuple(config.get('cw_freq_band', [0.1, 0.7]))
    )

    reco = interferometricDirectionReconstruction()
    
    # Initialize detector description
    # Option 1: Load from JSON file (uncomment if using local detector file)
    #det = detector.Detector(json_filename=config['detector_json'])
    # Option 2: Load from MongoDB (default for RNO-G)
    det = detector.Detector(source="rnog_mongo")
    det.update(datetime(2022, 10, 1))
    station_id = config.get('station_id')
    
    # Pre-calculate PA position once (it's constant per station)
    # This avoids redundant per-event calculations
    pa_pos_rel_station, pa_pos_abs = get_PA_position(station_id, det)
    
    # Pre-load cable delays for all channels
    # This triggers the detector's internal buffering/caching mechanism
    print("Pre-loading cable delays for station...", station_id)
    for ch in range(24):  # RNO-G has 24 channels per station
        try:
            _ = det.get_cable_delay(station_id, ch)
        except:
            pass  # Some channels may not exist in detector description
    print("Done!\n")    
    
    reco.begin(station_id=station_id, config=config, det=det, use_cache=args.use_cache)
    
    # Handle both --events and --runs flags
    # For NUR files with --runs, we'll check run_number
    # For ROOT files with --events (or NUR files with --events), we'll check event_id
    events_to_process = set(args.events) if args.events is not None else None
    runs_to_process = set(args.runs) if args.runs is not None else None
    
    results = [] if args.output_type == 'hdf5' else None
    events_for_nur = [] if args.output_type == 'nur' else None
    
    results_path = None
    maps_dir = None
    corr_map_path = None
    
    n_processed, n_skipped = 0, 0
    found_event_ids, found_run_numbers = [], []
    
    t_total = 0
    
    reader = eventReader() if is_nur_file else readRNOGData()
    
    for file_idx, input_file in enumerate(input_files, 1):
        file_events_processed = 0
        
        print(f"Processing file {file_idx}/{len(input_files)}: {input_file}", flush=True)
        
        if is_nur_file:
            reader.begin(input_file, read_detector=True)
        else:
            reader.begin(input_file, mattak_kwargs={'backend': 'uproot'})
                
        for event in reader.run():
            
            t0 = time.time()
            
            event_id = event.get_id()
            run_number = event.get_run_number()
            
            found_event_ids.append(event_id)
            found_run_numbers.append(run_number)
            
            if runs_to_process is not None:
                if run_number not in runs_to_process:
                    n_skipped += 1
                    continue
            
            if events_to_process is not None:
                if event_id not in events_to_process:
                    n_skipped += 1
                    continue

            if results_path is None:
                if args.outputfile is not None:
                    results_path = args.outputfile
                    maps_dir = None
                    print(f"Will save reconstruction results to: {results_path}")
                else:
                    results_path, maps_dir = create_organized_paths(
                        config, run_number, args.output_type, 
                        event_number=event_id
                    )
                    if results is not None or events_for_nur is not None:
                        print(f"Will save reconstruction results to: {results_path}")
                    if args.save_maps:
                        print(f"Will save correlation map data to: {maps_dir}")
            
            event_station = event.get_station(station_id)
            # Apply optional preprocessing steps based on config
            
            # Apply cable delay correction (subtract cable delays)
            if config.get('apply_cable_delay', True):
                channel_add_cable_delay.run(event, event_station, det, mode='subtract')
            
            # Upsampling improves time resolution for correlation analysis
            if config.get('apply_upsampling', False):
                channel_resampler.run(event, event_station, det, sampling_rate=5 * units.GHz)
            
            # Bandpass filter reduces noise outside antenna sensitivity range
            if config.get('apply_bandpass', False):
                channel_bandpass_filter.run(event, event_station, det, 
                    passband=[0.1 * units.GHz, 0.6 * units.GHz],
                    filter_type='butter', order=10)
                            
            # Remove continuous wave interference
            if config.get('apply_cw_removal', False):
                peak_prominence = config.get('cw_peak_prominence', 4.0)
                cw_filter.run(event, event_station, det, peak_prominence=peak_prominence)

            # Start with all configured channels
            channels_to_use = config['channels'].copy()
            
            # Apply edge signal filtering if threshold is specified
            if args.edge_sigma is not None:
                if args.verbose:
                    print(f"\n  Applying edge signal detection (threshold: {args.edge_sigma} sigma)")
                
                channels_to_use = filter_channels_by_edge_signal(
                    event_station,
                    channels_to_use,
                    edge_threshold_sigma=args.edge_sigma,
                    n_chunks=10,
                    verbose=args.verbose
                )
                
                if len(channels_to_use) < 2:
                    print(f"  Skipping event {event_id} (run {run_number}): Fewer than 2 channels after edge filtering ({len(channels_to_use)} channels)")
                    n_skipped += 1
                    continue

            # Apply SNR filtering if threshold is specified
            if args.snr_threshold is not None:
                if args.verbose:
                    print(f"\n  Applying SNR threshold: {args.snr_threshold}")
                
                passing_channels, should_skip = filter_channels_by_snr(
                    event_station, 
                    channels_to_use,  # Use channels that already passed edge filter
                    args.snr_threshold,
                    helper_channels=[9, 10, 22, 23],
                    verbose=args.verbose
                )
                
                if should_skip:
                    print(f"  Skipping event {event_id} (run {run_number}): No helper channels passed SNR threshold")
                    n_skipped += 1
                    continue
                
                if len(passing_channels) < 2:
                    print(f"  Skipping event {event_id} (run {run_number}): Fewer than 2 channels passed SNR threshold ({len(passing_channels)} channels)")
                    n_skipped += 1
                    continue
                
                channels_to_use = passing_channels
            
            # Check if we have at least one helper channel after all filtering
            if args.edge_sigma is not None or args.snr_threshold is not None:
                helper_channels = [9, 10, 22, 23]
                helper_channels_remaining = [ch for ch in helper_channels if ch in channels_to_use]
                
                if len(helper_channels_remaining) == 0:
                    print(f"  Skipping event {event_id} (run {run_number}): No helper channels [{','.join(map(str, helper_channels))}] remaining after filtering")
                    n_skipped += 1
                    continue
                
                if args.verbose:
                    print(f"  Using {len(channels_to_use)} channels for reconstruction: {channels_to_use}")
            
            # Create event-specific config with filtered channels
            event_config = config.copy()
            event_config['channels'] = channels_to_use

            # Get event-specific fixed_coord if using per-event values
            if args.sim_truth_fixed_coord:
                event_config['fixed_coord'] = get_sim_truth_fixed_coord(event_config, event, pa_pos_abs)
                    
            # Run interferometric direction reconstruction
            # For auto mode: two-stage reconstruction (rhoz then spherical)
            # For manual mode: single-stage with config parameters
            if reco_mode == 'auto':
                corr_map_path = run_two_stage_reconstruction(
                    reco, event, event_station, det, event_config, pa_pos_abs,
                    save_maps=args.save_maps,
                    save_pair_maps=args.save_pair_maps,
                    save_maps_to=maps_dir if (args.save_maps or args.save_pair_maps) else None
                )
            else:
                # Standard single-stage reconstruction
                corr_map_path = reco.run(event, event_station, det, event_config, 
                        save_maps=args.save_maps, 
                        save_pair_maps=args.save_pair_maps,
                        save_maps_to=maps_dir if (args.save_maps or args.save_pair_maps) else None)
            
            # Extract reconstruction results from station parameters
            # These were set by the reconstruction module's run() method
            max_corr = event_station.get_parameter(stnp.rec_max_correlation)
            surf_corr = event_station.get_parameter(stnp.rec_surf_corr)
            rec_coord0 = event_station.get_parameter(stnp.rec_coord0)  # Generic first coordinate
            rec_coord1 = event_station.get_parameter(stnp.rec_coord1)  # Generic second coordinate
            
            # Try to get alternate reconstruction (second-best correlation peak)
            try:
                rec_coord0_alt = event_station.get_parameter(stnp.rec_coord0_alt)
                rec_coord1_alt = event_station.get_parameter(stnp.rec_coord1_alt)
            except:
                rec_coord0_alt = np.nan
                rec_coord1_alt = np.nan
            
            # Build results dictionary for HDF5 output
            # Convert generic coordinates to physically meaningful values based on coord_system
            # For auto mode, the final results are always spherical (from stage 2)
            if results is not None:
                result_row = {
                    "filename": input_file,  # Critical for simulation files with duplicate IDs
                    "runNum": run_number,
                    "eventNum": event_id,
                    "maxCorr": max_corr,
                    "surfCorr": surf_corr,
                }
                
                # Determine which coordinate system the final results are in
                # Auto mode always produces spherical results
                final_coord_system = 'spherical' if reco_mode == 'auto' else config["coord_system"]
                
                # Map generic coordinates to physical quantities based on coordinate system
                if final_coord_system == "cylindrical":
                    if config["rec_type"] == "phiz":
                        # φ-z reconstruction: azimuth and depth (radius is fixed)
                        result_row["phi"] = rec_coord0 / units.deg
                        result_row["z"] = rec_coord1 / units.m

                        if not np.isnan(rec_coord0_alt) and not np.isnan(rec_coord1_alt):
                            result_row["phi_alt"] = rec_coord0_alt / units.deg
                            result_row["z_alt"] = rec_coord1_alt / units.m
                        else:
                            result_row["phi_alt"] = np.nan
                            result_row["z_alt"] = np.nan
                    elif config["rec_type"] == "rhoz":
                        # ρ-z reconstruction: radius and depth (azimuth is fixed)
                        result_row["rho"] = rec_coord0 / units.m
                        result_row["z"] = rec_coord1 / units.m

                        if not np.isnan(rec_coord0_alt) and not np.isnan(rec_coord1_alt):
                            result_row["rho_alt"] = rec_coord0_alt / units.m
                            result_row["z_alt"] = rec_coord1_alt / units.m
                        else:
                            result_row["rho_alt"] = np.nan
                            result_row["z_alt"] = np.nan
                elif final_coord_system == "spherical":
                    # Spherical reconstruction: azimuth and zenith (radius is fixed)
                    result_row["phi"] = rec_coord0 / units.deg
                    result_row["theta"] = rec_coord1 / units.deg

                    if not np.isnan(rec_coord0_alt) and not np.isnan(rec_coord1_alt):
                        result_row["phi_alt"] = rec_coord0_alt / units.deg
                        result_row["theta_alt"] = rec_coord1_alt / units.deg
                    else:
                        result_row["phi_alt"] = np.nan
                        result_row["theta_alt"] = np.nan
                
                results.append(result_row)
            
            if events_for_nur is not None:
                events_for_nur.append(event)
            
            if args.verbose:
                print(f"\n=== Reconstruction Results ===")
                print(f"Station: {station_id}")
                if results is not None:
                    # Print the result row which already has properly named columns
                    for key, value in result_row.items():
                        if key not in ['filename', 'runNum', 'eventNum']:  # Skip metadata
                            print(f"{key}: {value:.3f}" if not np.isnan(value) else f"{key}: nan")
                else:
                    # For NUR output, print basic info
                    print(f"Max correlation: {max_corr:.3f}")
                    print(f"Surface correlation: {surf_corr:.3f}")
                print(f"===============================\n")
            
            n_processed += 1
            file_events_processed += 1
            
            t_total += time.time() - t0
            
            if events_to_process is not None and n_processed == len(events_to_process):
                break
        
        reader.end()
        
        gc.collect()
        
        if events_to_process is not None and n_processed == len(events_to_process):
            break

    reco.end()
    
    # Check if any requested events/runs were not found
    for requested, found_list, label, id_label in [
        (events_to_process, found_event_ids, "event(s)", "event IDs"),
        (runs_to_process, found_run_numbers, "run(s)", "run numbers")
    ]:
        if requested is not None:
            missing = requested - set(found_list)
            if missing:
                print(f"\nWARNING: {len(missing)} requested {label} were not found in input files:")
                print(f"  Missing {id_label}: {sorted(missing)}")
                
                # If none were processed, show available IDs to help user
                if n_processed == 0:
                    unique_ids = sorted(set(found_list))
                    print(f"  Available {id_label} in file: {unique_ids[:20]}")
                    if len(unique_ids) > 20:
                        print(f"  ... and {len(unique_ids) - 20} more")
    
    if results_path:
        if results:
            reco_results_path = save_results_to_hdf5(results, results_path, config)
        elif events_for_nur:
            reco_results_path = save_results_to_nur(events_for_nur, results_path)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Processed: {n_processed} events")
    if n_processed > 0:
        print(f"Total time: {t_total:.2f}s")
        print(f"Time per event: {t_total / n_processed:.2f}s")
    if events_to_process is not None or runs_to_process is not None:
        print(f"Skipped: {n_skipped} events")
    print(f"{'='*50}\n")

    if corr_map_path is not None:
        print(f"To plot, do:\npython correlation_map_plotter.py --input {corr_map_path} --comprehensive {reco_results_path}") if is_nur_file else print(f"To plot, do:\npython correlation_map_plotter.py --input {corr_map_path}")

if __name__ == "__main__":
    main()