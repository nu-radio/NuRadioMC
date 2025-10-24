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
import os
import yaml
import gc
import time
import numpy as np
import logging
logging.getLogger('NuRadioReco').setLevel(logging.ERROR)

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

from NuRadioReco.modules.interferometricDirectionReconstruction import interferometricDirectionReconstruction
from NuRadioReco.utilities.interferometry_io_utilities import (
    save_interferometric_results_hdf5 as save_results_to_hdf5,
    save_interferometric_results_nur as save_results_to_nur,
    save_correlation_map,
    create_organized_paths
)

def get_sim_vertex_rel_PA(station_id, event_object, det):
    
    interaction_vertex_abs = list(event_object.get_sim_showers())[0].get_parameter(showerParameters.vertex)

    station_pos_abs = np.array(det.get_absolute_position(station_id))
    ch1_pos_rel = np.array(det.get_relative_position(station_id, 1))
    ch2_pos_rel = np.array(det.get_relative_position(station_id, 2))
    pa_pos_rel = 0.5 * (ch1_pos_rel + ch2_pos_rel) # PA center relative to station

    interaction_vertex_rel_PA = np.array(interaction_vertex_abs) - station_pos_abs - pa_pos_rel
    
    return interaction_vertex_rel_PA

def get_sim_truth_fixed_coord(config, event_object, det):
    
    interaction_vertex_rel_PA = get_sim_vertex_rel_PA(config['station_id'], event_object, det)
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

    parser.add_argument("--sim_truth_fixed_coord", action="store_true")
    parser.add_argument("--save-maps", action="store_true",
                       help="Save correlation map data to pickle files for later plotting")
    parser.add_argument("--save-pair-maps", action="store_true",
                       help="Save individual channel pair correlation maps (requires --save-maps)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print reconstruction results for each event")

    args = parser.parse_args()
        
    # Validate event/run selection arguments
    if args.events is not None and args.runs is not None:
        raise ValueError("Cannot specify both --events and --runs. Use --events for ROOT files and --runs for NUR files.")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    required_params = ['coord_system', 'channels', 'limits', 'step_sizes', 'time_delay_tables']
    for param in required_params:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' not found in config file")

    input_files = args.inputfile if isinstance(args.inputfile, list) else [args.inputfile]
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
    
    # Initialize detector description
    # Option 1: Load from JSON file (uncomment if using local detector file)
    #det = detector.Detector(json_filename=config['detector_json'])
    # Option 2: Load from MongoDB (default for RNO-G)
    det = detector.Detector(source="rnog_mongo")
    det.update(datetime(2022, 10, 1))
    station_id = config.get('station_id')
        
    # Pre-compute and cache delay matrices for this station/config combination
    # This significantly speeds up processing by loading cached data if available
    # Skip precomputation if using per-event fixed_coord (since it varies per event)
    if not args.sim_truth_fixed_coord:
        reco.begin(station_id=station_id, config=config, det=det)
    else:
        print("Skipping delay matrix precomputation (using per-event fixed_coord)")
    
    # Handle both --events and --runs flags
    # For NUR files with --runs, we'll check run_number
    # For ROOT files with --events (or NUR files with --events), we'll check event_id
    events_to_process = set(args.events) if args.events is not None else None
    runs_to_process = set(args.runs) if args.runs is not None else None
    
    results = [] if args.output_type == 'hdf5' else None
    events_for_nur = [] if args.output_type == 'nur' else None
    
    results_path = None
    maps_dir = None
    
    n_processed = 0
    n_skipped = 0
    found_event_ids = [] 
    found_run_numbers = []
    
    t_total = 0
    
    if is_nur_file:
        reader = eventReader()
    else:
        reader = readRNOGData()
    
    for file_idx, input_file in enumerate(input_files, 1):
        file_events_processed = 0
        
        print(f"Processing file {file_idx}/{len(input_files)}: {input_file}", flush=True)
        
        if is_nur_file:
            reader.begin(input_file, read_detector=True)
        else:
            reader.begin(input_file, mattak_kwargs={'backend': 'uproot'})
            #reader.begin(input_file)
        
        event_generator = reader.run()
        
        for event in event_generator:
            
            t0 = time.time()
            
            event_id = event.get_id()
            run_number = event.get_run_number()
            
            found_event_ids.append(event_id)
            found_run_numbers.append(run_number)
            
            # Handle --runs flag (for NUR files)
            if runs_to_process is not None:
                if run_number not in runs_to_process:
                    n_skipped += 1
                    continue
            
            # Handle --events flag (for both ROOT and NUR files)
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
            
            # Remove channels not used in reconstruction to save memory and processing time
            for ch_id in [ch for ch in event_station.get_channel_ids() if ch not in config['channels']]:
                event_station.remove_channel(ch_id)
            
            # Apply cable delay correction (subtract cable delays)
            channel_add_cable_delay.run(event, event_station, det, mode='subtract')
            
            # # Apply optional preprocessing steps based on config
            # # Upsampling improves time resolution for correlation analysis
            if config.get('apply_upsampling', False):
                channel_resampler.run(event, event_station, det, sampling_rate=5 * units.GHz)
            
            # # Bandpass filter reduces noise outside antenna sensitivity range
            if config.get('apply_bandpass', False):
                channel_bandpass_filter.run(event, event_station, det, 
                    passband=[0.1 * units.GHz, 0.6 * units.GHz],
                    filter_type='butter', order=10)
                            
            # Remove continuous wave interference
            if config.get('apply_cw_removal', False):
                peak_prominence = config.get('cw_peak_prominence', 4.0)
                cw_filter.run(event, event_station, det, peak_prominence=peak_prominence)

            # Get event-specific fixed_coord if using per-event values
            event_config = config.copy()  
            if args.sim_truth_fixed_coord:
                event_config['fixed_coord'] = get_sim_truth_fixed_coord(event_config, event, det)

            # Run interferometric direction reconstruction
            # Results are stored in station parameters (accessed below via get_parameter)
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
            if results is not None:
                result_row = {
                    "filename": input_file,  # Critical for simulation files with duplicate IDs
                    "runNum": run_number,
                    "eventNum": event_id,
                    "maxCorr": max_corr,
                    "surfCorr": surf_corr,
                }
                
                # Map generic coordinates to physical quantities based on coordinate system
                if config["coord_system"] == "cylindrical":
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
                elif config["coord_system"] == "spherical":
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
                print(f"Max correlation: {max_corr:.3f}")
                print(f"Surface correlation: {surf_corr:.3f}")
                
                if config["coord_system"] == "cylindrical":
                    if config["rec_type"] == "phiz":
                        print(f"Reconstructed azimuth (φ): {rec_coord0/units.deg:.1f}°")
                        print(f"Reconstructed depth (z): {rec_coord1/units.m:.1f} m")
                        print(f"Fixed radius (ρ): {config['fixed_coord']} m")
                        if not np.isnan(rec_coord0_alt) and not np.isnan(rec_coord1_alt):
                            print(f"Alternate azimuth (φ): {rec_coord0_alt/units.deg:.1f}°")
                            print(f"Alternate depth (z): {rec_coord1_alt/units.m:.1f} m")
                    elif config["rec_type"] == "rhoz":
                        print(f"Reconstructed radius (ρ): {rec_coord0/units.m:.1f} m")
                        print(f"Reconstructed depth (z): {rec_coord1/units.m:.1f} m")
                        print(f"Fixed azimuth (φ): {config['fixed_coord']}°")
                        if not np.isnan(rec_coord0_alt) and not np.isnan(rec_coord1_alt):
                            print(f"Alternate radius (ρ): {rec_coord0_alt/units.m:.1f} m")
                            print(f"Alternate depth (z): {rec_coord1_alt/units.m:.1f} m")
                elif config["coord_system"] == "spherical":
                    print(f"Reconstructed azimuth (φ): {rec_coord0/units.deg:.1f}°")
                    print(f"Reconstructed zenith (θ): {rec_coord1/units.deg:.1f}°")
                    print(f"Fixed radius (r): {config['fixed_coord']} m")
                    if not np.isnan(rec_coord0_alt) and not np.isnan(rec_coord1_alt):
                        print(f"Alternate azimuth (φ): {rec_coord0_alt/units.deg:.1f}°")
                        print(f"Alternate zenith (θ): {rec_coord1_alt/units.deg:.1f}°")
                
                print(f"Coordinate system: {config['coord_system']}")
                print(f"Reconstruction type: {config['rec_type']}")
                print("===============================\n")
            
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
        if n_processed == 0:
            print(f"\nWARNING: None of the requested events were found!")
            if runs_to_process is not None:
                print(f"Requested run numbers: {sorted(runs_to_process)}")
            else:
                print(f"Requested event indices: {sorted(events_to_process)}")
            if is_nur_file:
                unique_indices = sorted(set(found_run_numbers))
                print(f"Available run numbers in file: {unique_indices[:20]}")
                if len(unique_indices) > 20:
                    print(f"... and {len(unique_indices) - 20} more events")
                print(f"   Example: --runs {unique_indices[0]} {unique_indices[1] if len(unique_indices) > 1 else ''}")
            else:
                unique_events = sorted(set(found_event_ids))
                print(f"Available event IDs in file: {unique_events[:20]}")
                if len(unique_events) > 20:
                    print(f"... and {len(unique_events) - 20} more events")
    print(f"{'='*50}\n")

    if corr_map_path is not None:
        print(f"To plot, do:\npython correlation_map_plotter.py --input {corr_map_path} --comprehensive {reco_results_path}")

if __name__ == "__main__":
    main()