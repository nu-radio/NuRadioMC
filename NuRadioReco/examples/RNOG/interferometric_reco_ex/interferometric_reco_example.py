#!/usr/bin/env python3
"""
Simple interferometric direction reconstruction script.

This is a streamlined version with just the core reconstruction functionality.
For advanced features (SNR filtering, edge detection, two-stage reconstruction),
see interferometric_reco_example_advanced.py

Usage:
    python interferometric_reco_simple_example.py --config config.yaml --input data.root

Example:
    python interferometric_reco_simple_example.py \
        --config example_config.yaml \
        --input /path/to/station21_run476.root \
        --outputfile reconstruction_results.h5
"""

import logging
import argparse
from datetime import datetime
import os, sys
import yaml
import gc
import time
import numpy as np

from NuRadioReco.utilities import units
from NuRadioReco.utilities.logging import set_general_log_level
# Set NuRadioReco logging level (INFO=20, DEBUG=10)
set_general_log_level(logging.ERROR)
logger = logging.getLogger("NuRadioReco.modules.interferometricDirectionReconstruction")
logger.setLevel(logging.WARNING)

from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.channelSinewaveSubtraction import channelSinewaveSubtraction
from NuRadioReco.modules.channelAddCableDelay import channelAddCableDelay
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.detector import detector
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters
from NuRadioReco.modules.interferometricDirectionReconstruction import interferometricDirectionReconstruction
from NuRadioReco.utilities.interferometry_io_utilities import (
    save_interferometric_results_hdf5 as save_results_to_hdf5,
    save_interferometric_results_nur as save_results_to_nur,
    save_correlation_map,
    create_organized_paths
)

def main():    
    parser = argparse.ArgumentParser(
        prog="interferometric_reco_simple_example.py", 
        description="Run simple interferometric direction reconstruction on RNO-G data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic reconstruction
            python interferometric_reco_simple_example.py --config example_config.yaml --input data.root

            # Save correlation maps
            python interferometric_reco_simple_example.py --config example_config.yaml --input data.root --save_maps
        """
    )
    
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to YAML configuration file")
    parser.add_argument("-i", "--input", type=str, nargs="+", required=True,
                       help="Path(s) to input data file(s) (ROOT or NUR). Can specify multiple files for same station.")
    parser.add_argument("--output_type", type=str, choices=['hdf5', 'nur'], default='hdf5',
                       help="Output file format: 'hdf5' for HDF5 tables or 'nur' for NuRadioReco format (default: hdf5)")
    parser.add_argument("-o", "--outputfile", type=str, default=None,
                       help="Optional: manually specify output file path. If not set, uses organized default path.")
    parser.add_argument("--events", type=int, nargs="*", default=None, 
                       help="Specific event IDs to process. If not provided, processes all events in given file.")
    parser.add_argument("--runs", type=int, nargs="*", default=None,
                       help="Specific run numbers to process.")

    parser.add_argument("--save-maps", action="store_true",
                       help="Save correlation map data to pickle files for later plotting")
    parser.add_argument("--save-pair-maps", action="store_true",
                       help="Save individual channel pair correlation maps (requires --save_maps)")

    args = parser.parse_args()
        
    if args.events is not None and args.runs is not None:
        raise ValueError("Cannot specify both --events and --runs. Use --events for ROOT files and --runs for NUR files.")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    required_params = ['coord_system', 'channels', 'limits', 'step_sizes', 'time_delay_tables']
    for param in required_params:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' not found in config file")

    input_files = args.input if isinstance(args.input, list) else [args.input]
    is_nur_file = input_files[0].endswith('.nur')
    
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
    
    # Initialize detector description from MongoDB
    det = detector.Detector(source="rnog_mongo")
    det.update(datetime(2022, 10, 1))
    station_id = config.get('station_id')
    
    # Pre-load cable delays for all channels
    # This triggers the detector's internal buffering/caching mechanism
    print(f"Pre-loading cable delays for station {station_id}...")
    for ch in range(24):  # RNO-G has 24 channels per station
        try:
            _ = det.get_cable_delay(station_id, ch)
        except:
            pass  # Some channels may not exist in detector description
    print("Done!\n")
    
    reco.begin(station_id=station_id, config=config, det=det)
    
    events_to_process = set(args.events) if args.events is not None else None
    runs_to_process = set(args.runs) if args.runs is not None else None
    
    results = [] if args.output_type == 'hdf5' else None
    events_for_nur = [] if args.output_type == 'nur' else None
    
    # Define coordinate mapping once (used for saving to HDF5)
    # Format: (coord_system, rec_type) -> [(name0, unit0), (name1, unit1)]
    coordinate_mapping = {
        ("cylindrical", "phiz"): [("phi", units.deg), ("z", units.m)],
        ("cylindrical", "rhoz"): [("rho", units.m), ("z", units.m)],
        ("spherical", "phitheta"): [("phi", units.deg), ("theta", units.deg)]
    }
    coord_key = (config["coord_system"], config.get("rec_type", "phitheta"))
    
    results_path = None
    maps_dir = None
    corr_map_path = None
    
    n_processed, n_skipped = 0, 0
    found_event_numbers, found_run_numbers = [], [] 
    
    t_total = 0
    
    reader = eventReader() if is_nur_file else readRNOGData()
    
    for file_idx, input_file in enumerate(input_files, 1):
        file_events_processed = 0
        
        print(f"Processing file {file_idx}/{len(input_files)}: {input_file}", flush=True)
        
        reader.begin(input_file)

        # Get event IDs differently based on file type
        if is_nur_file:
            # For NUR files, use the underlying NuRadioRecoio object to get event IDs
            event_ids = reader._eventReader__fin.get_event_ids()
        else:
            # For ROOT files, readRNOGData has get_event_ids() method
            event_ids = reader.get_event_ids()
        
        # Store all available run/event numbers for error reporting
        all_runs = [eid[0] for eid in event_ids]
        all_events = [eid[1] for eid in event_ids]
        
        # Filter event IDs based on user request
        if runs_to_process or events_to_process:
            filtered_ids = []
            for event_id in event_ids:
                run_num, evt_num = event_id
                if runs_to_process and run_num not in runs_to_process:
                    continue
                if events_to_process and evt_num not in events_to_process:
                    continue
                filtered_ids.append(event_id)
            event_ids = filtered_ids

        print(f"Found {len(event_ids)} event(s) to process")
        if len(event_ids) == 0 and (runs_to_process or events_to_process):
            print(f"WARNING: No events match the requested criteria!")
            if runs_to_process:
                available_runs = sorted(set(all_runs))
                print(f"  Requested run(s): {sorted(runs_to_process)}")
                print(f"  Available run(s): {available_runs[:20]}")
                if len(available_runs) > 20:
                    print(f"  ... and {len(available_runs) - 20} more")
            if events_to_process:
                available_events = sorted(set(all_events))
                print(f"  Requested event(s): {sorted(events_to_process)}")
                print(f"  Available event(s): {available_events[:20]}")
                if len(available_events) > 20:
                    print(f"  ... and {len(available_events) - 20} more")
            
        for event_id in event_ids:
            
            t0 = time.time()
            
            # Get event differently based on file type
            if is_nur_file:
                run_number, event_number = event_id
                event = reader._eventReader__fin.get_event(event_id)
            else:
                run_number, event_number = event_id
                event = reader.get_event(run_number, event_number)
            
            found_event_numbers.append(event_number)
            found_run_numbers.append(run_number)

            if results_path is None:
                if args.outputfile is not None:
                    results_path = args.outputfile
                    maps_dir = None
                    print(f"Will save reconstruction results to: {results_path}")
                else:
                    results_path, maps_dir = create_organized_paths(
                        config, run_number, args.output_type, 
                        event_number=event_number
                    )
                    if results is not None or events_for_nur is not None:
                        print(f"Will save reconstruction results to: {results_path}")
                    if args.save_maps:
                        print(f"Will save correlation map data to: {maps_dir}")
            
            event_station = event.get_station(station_id)
            
            # Apply cable delay correction (subtract cable delays)
            if config.get('apply_cable_delay', True):
                channel_add_cable_delay.run(event, event_station, det, mode='subtract')

            # Apply optional preprocessing steps based on config
            # Upsampling improves time resolution for correlation analysis
            if config.get('apply_upsampling', False):
                channel_resampler.run(event, event_station, det, sampling_rate=10 * units.GHz)

            # Bandpass filter reduces noise outside antenna sensitivity range
            if config.get('apply_bandpass', False):
                channel_bandpass_filter.run(event, event_station, det, 
                    passband=[0.1 * units.GHz, 0.6 * units.GHz],
                    filter_type='butter', order=10)
                            
            # Remove continuous wave interference
            if config.get('apply_cw_removal', False):
                peak_prominence = config.get('cw_peak_prominence', 4.0)
                cw_filter.run(event, event_station, det, peak_prominence=peak_prominence)
            
            # Run interferometric direction reconstruction
            corr_map_path = reco.run(event, event_station, det, config, 
                    save_maps=args.save_maps, 
                    save_pair_maps=args.save_pair_maps,
                    save_maps_to=maps_dir if (args.save_maps or args.save_pair_maps) else None)
                        
            # Extract reconstruction results from station parameters
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
            if results is not None:
                result_row = {
                    "filename": input_file,
                    "runNum": run_number,
                    "eventNum": event_number,
                    "maxCorr": max_corr,
                    "surfCorr": surf_corr,
                }
                
                # Map generic coordinates to physical quantities using pre-defined mapping
                if coord_key in coordinate_mapping:
                    coord_info = coordinate_mapping[coord_key]
                    name0, unit0 = coord_info[0][0], coord_info[0][1]
                    name1, unit1 = coord_info[1][0], coord_info[1][1]
                    result_row[name0] = rec_coord0 / unit0
                    result_row[name1] = rec_coord1 / unit1
                    result_row[f"{name0}_alt"] = rec_coord0_alt / unit0 if not np.isnan(rec_coord0_alt) else np.nan
                    result_row[f"{name1}_alt"] = rec_coord1_alt / unit1 if not np.isnan(rec_coord1_alt) else np.nan
                
                results.append(result_row)
                            
            if events_for_nur is not None:
                events_for_nur.append(event)
            
            summary = [
                "\n=== Reconstruction Results ===",
                f"Station: {station_id}",
            ]
            if results is not None:
                summary.extend(
                    f"{key}: {value:.3f}" if not np.isnan(value) else f"{key}: nan"
                    for key, value in result_row.items()
                    if key not in ['filename', 'runNum', 'eventNum']
                )
            else:
                summary.extend([
                    f"Max correlation: {max_corr:.3f}",
                    f"Surface correlation: {surf_corr:.3f}",
                ])
            summary.append("===============================")
            logger.info("\n".join(summary))
            
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
    
    # Save results if we processed any events
    if results_path and n_processed > 0:
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
    print(f"{'='*50}\n")

    if n_processed > 0 and corr_map_path is not None:
        print(f"To plot, do:\npython correlation_map_plotter.py --input {corr_map_path} --comprehensive {reco_results_path}") if is_nur_file else print(f"To plot, do:\npython correlation_map_plotter.py --input {corr_map_path}")

if __name__ == "__main__":
    main()
