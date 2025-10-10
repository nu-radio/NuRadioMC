#!/usr/bin/env python3
"""
Interferometric Direction Reconstruction Runner

This script provides a command-line interface for running interferometric direction 
reconstruction using pre-calculated time delay tables.

Usage:
    python interferometric_reco_runner.py --config config.yaml --inputfile data.root --outputfile output.h5

Example:
    python interferometric_reco_runner.py \
        --config example_config.yaml \
        --inputfile /path/to/station21_run476.root \
        --outputfile reconstruction_results.h5 \
        --save_plots --verbose --events 1 2 3
"""

import argparse
from datetime import datetime
import os
import yaml
import gc
import numpy as np

from NuRadioReco.utilities import units
from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.channelSinewaveSubtraction import channelSinewaveSubtraction
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.detector import detector
from NuRadioReco.framework.parameters import stationParameters as stnp

from NuRadioReco.modules.interferometricDirectionReconstruction import (
    interferometricDirectionReconstruction, 
    save_results_to_hdf5
)


def main():
    parser = argparse.ArgumentParser(
        prog="interferometric_reco_runner.py", 
        description="Run interferometric direction reconstruction on RNO-G data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reconstruction
  python interferometric_reco_runner.py --config example_config.yaml --inputfile data.root --outputfile results.h5

  # Process specific events with plots
  python interferometric_reco_runner.py --config example_config.yaml --inputfile data.root --events 1 5 10 --save_plots --verbose

  # Multiple input files
  python interferometric_reco_runner.py --config example_config.yaml --inputfile file1.root file2.root --outputfile merged.h5
        """
    )
    
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to YAML configuration file")
    parser.add_argument("--inputfile", type=str, nargs="+", required=True,
                       help="Path(s) to input data file(s) (ROOT or NUR). Can specify multiple files for same station.")
    parser.add_argument("--outputfile", type=str, default=None, 
                       help="Path to output file (.h5 for HDF5 table, .nur for NuRadioReco format)")
    parser.add_argument("--events", type=int, nargs="*", default=None, 
                       help="Specific event IDs to process (optional). If not provided, processes all events")
    parser.add_argument("--save_plots", action="store_true", 
                       help="Save correlation map plots for processed events")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print reconstruction results for each event")

    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    required_params = ['coord_system', 'channels', 'limits', 'step_sizes', 'fixed_coord', 'time_delay_tables']
    for param in required_params:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' not found in config file")

    input_files = args.inputfile if isinstance(args.inputfile, list) else [args.inputfile]
    
    is_nur_file = input_files[0].endswith('.nur')
    
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
    
    det = detector.Detector(json_filename=config['detector_json'])
    det.update(datetime(2022, 10, 1))
    
    station_id = config.get('station_id')
    
    reco.begin(preload_cache_for_station=station_id, config=config, det=det)
    
    events_to_process = set(args.events) if args.events is not None else None
    
    results = [] if args.outputfile and args.outputfile.endswith('.h5') else None
    if results is not None:
        print(f"Will save reconstruction results to HDF5 file: {args.outputfile}")
    
    n_processed = 0
    n_skipped = 0
    found_event_ids = [] 
    found_run_numbers = []
    
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
        
        event_generator = reader.run()
        
        for event in event_generator:
            
            event_id = event.get_id()
            run_number = event.get_run_number()
            
            found_event_ids.append(event_id)
            found_run_numbers.append(run_number)
            
            # Event selection logic
            # NOTE: NuRadioMC simulation files (.nur) can have a quirk where all events share
            # the same event_id (typically 0), but each event has a unique run_number. 
            # The run_number effectively serves as the event index for simulations.
            # For real data (.root files), event_id is the correct identifier.
            if events_to_process is not None:
                if is_nur_file:
                    # For NUR files (simulations), check run_number as the event index
                    if run_number not in events_to_process:
                        n_skipped += 1
                        continue
                else:
                    # For ROOT files (data), use event_id
                    if event_id not in events_to_process:
                        n_skipped += 1
                        continue
            
            station = event.get_station()
            station_id = station.get_id()
            
            for ch_id in [ch for ch in station.get_channel_ids() if ch not in config['channels']]:
                station.remove_channel(ch_id)
            
            if config.get('apply_upsampling', False):
                channel_resampler.run(event, station, det, sampling_rate=5 * units.GHz)
            
            if config.get('apply_bandpass', False):
                channel_bandpass_filter.run(event, station, det, 
                    passband=[0.1 * units.GHz, 0.6 * units.GHz],
                    filter_type='butter', order=10)
            
            if config.get('apply_cw_removal', False):
                peak_prominence = config.get('cw_peak_prominence', 4.0)
                cw_filter.run(event, station, det, peak_prominence=peak_prominence)

            reco.run(event, station, det, config, corr_map=args.save_plots)
            
            max_corr = station.get_parameter(stnp.rec_max_correlation)
            surf_corr = station.get_parameter(stnp.rec_surf_corr)
            rec_coord0 = station.get_parameter(stnp.rec_coord0)
            rec_coord1 = station.get_parameter(stnp.rec_coord1)
            
            try:
                rec_coord0_alt = station.get_parameter(stnp.rec_coord0_alt)
                rec_coord1_alt = station.get_parameter(stnp.rec_coord1_alt)
            except:
                rec_coord0_alt = np.nan
                rec_coord1_alt = np.nan
            
            if results is not None:
                result_row = {
                    "runNum": run_number,
                    "eventNum": event_id,
                    "maxCorr": max_corr,
                    "surfCorr": surf_corr,
                }
                
                if config["coord_system"] == "cylindrical":
                    if config["rec_type"] == "phiz":
                        result_row["phi"] = rec_coord0 / units.deg
                        result_row["z"] = rec_coord1 / units.m

                        if not np.isnan(rec_coord0_alt) and not np.isnan(rec_coord1_alt):
                            result_row["phi_alt"] = rec_coord0_alt / units.deg
                            result_row["z_alt"] = rec_coord1_alt / units.m
                        else:
                            result_row["phi_alt"] = np.nan
                            result_row["z_alt"] = np.nan
                    elif config["rec_type"] == "rhoz":
                        result_row["rho"] = rec_coord0 / units.m
                        result_row["z"] = rec_coord1 / units.m

                        if not np.isnan(rec_coord0_alt) and not np.isnan(rec_coord1_alt):
                            result_row["rho_alt"] = rec_coord0_alt / units.m
                            result_row["z_alt"] = rec_coord1_alt / units.m
                        else:
                            result_row["rho_alt"] = np.nan
                            result_row["z_alt"] = np.nan
                elif config["coord_system"] == "spherical":
                    result_row["phi"] = rec_coord0 / units.deg
                    result_row["theta"] = rec_coord1 / units.deg

                    if not np.isnan(rec_coord0_alt) and not np.isnan(rec_coord1_alt):
                        result_row["phi_alt"] = rec_coord0_alt / units.deg
                        result_row["theta_alt"] = rec_coord1_alt / units.deg
                    else:
                        result_row["phi_alt"] = np.nan
                        result_row["theta_alt"] = np.nan
                
                results.append(result_row)
            
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
            
            if events_to_process is not None and n_processed == len(events_to_process):
                break
        
        reader.end()
        
        gc.collect()
        
        if events_to_process is not None and n_processed == len(events_to_process):
            break

    reco.end()
    
    if args.outputfile and args.outputfile.endswith('.h5') and results:
        save_results_to_hdf5(results, args.outputfile, config)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Processed: {n_processed} events")
    if events_to_process is not None:
        print(f"Skipped: {n_skipped} events")
        if n_processed == 0:
            print(f"\nWARNING: None of the requested events were found!")
            print(f"Requested event indices: {sorted(events_to_process)}")
            if is_nur_file:
                unique_indices = sorted(set(found_run_numbers))
                print(f"Available event indices in file: {unique_indices[:20]}")
                if len(unique_indices) > 20:
                    print(f"... and {len(unique_indices) - 20} more events")
                print(f"   Example: --events {unique_indices[0]} {unique_indices[1] if len(unique_indices) > 1 else ''}")
            else:
                unique_events = sorted(set(found_event_ids))
                print(f"Available event IDs in file: {unique_events[:20]}")
                if len(unique_events) > 20:
                    print(f"... and {len(unique_events) - 20} more events")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()