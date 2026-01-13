"""
LOFAR Beamformer Sky Mapper Example
====================================

This script demonstrates how to use the beamformerSkyMapper module to find 
cosmic ray arrival directions via full-sky interferometric imaging.

The module performs frequency-domain beamforming over a grid of sky directions
and identifies the direction of maximum power. It also performs time-windowed
analysis to detect transient signals (cosmic rays) vs continuous sources (e.g. Sun).

Usage
-----
    python beamformer_sky_mapper_example.py <path_to_nur_file>
    
    # Or with optional arguments:
    python beamformer_sky_mapper_example.py <path_to_nur_file> --output-dir ./output --debug

-------

.. moduleauthor:: Karen Terveer
"""

import argparse
import numpy as np

import NuRadioReco.modules.io.eventReader as eventReader
from NuRadioReco.modules.io.LOFAR.readLOFARData import LOFAR_event_id_to_unix
from NuRadioReco.modules.LOFAR.beamformerSkyMapper_LOFAR import beamformerSkyMapper
from NuRadioReco.framework.parameters import stationParameters
from NuRadioReco.utilities import units

from astropy.time import Time


def main():
    # ============================================================
    # Parse command line arguments
    # ============================================================
    parser = argparse.ArgumentParser(
        description='LOFAR Beamformer Sky Mapper - Find CR direction via full-sky beamforming'
    )
    parser.add_argument(
        'input_file', type=str,
        help='Path to LOFAR .nur file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='beamformer_output',
        help='Output directory for debug plots (default: beamformer_output)'
    )
    parser.add_argument(
        '--n-zenith', type=int, default=45,
        help='Number of zenith angle bins (default: 45 = 2 deg resolution)'
    )
    parser.add_argument(
        '--n-azimuth', type=int, default=180,
        help='Number of azimuth bins (default: 180 = 2 deg resolution)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Generate debug plots (sky maps, beamformed traces)'
    )
    parser.add_argument(
        '--station-id', type=int, default=None,
        help='Process only this station ID (default: process all stations)'
    )
    
    args = parser.parse_args()
    
    # ============================================================
    # Initialize modules
    # ============================================================
    
    # Event reader
    evtReader = eventReader.eventReader()
    evtReader.begin(filename=[args.input_file], read_detector=True)
    detector = evtReader.get_detector()
    
    # Beamformer sky mapper
    beamformer = beamformerSkyMapper()
    beamformer.begin(
        n_zenith=args.n_zenith,
        n_azimuth=args.n_azimuth,
        n_time_windows=10,
        window_overlap=0.5,
        freq_range_mhz=(30, 80),
        use_efields=False,  # Use raw voltages (recommended for direction finding)
        mark_sun=True,      # Mark Sun position on sky maps
        debug=args.debug,
        output_dir=args.output_dir
    )
    
    # ============================================================
    # Process events
    # ============================================================
    
    for event in evtReader.run():
        event_id = event.get_id()
        
        # Update detector to event time
        event_time_unix = LOFAR_event_id_to_unix(int(event_id))
        obs_time = Time(event_time_unix, format='unix')
        detector.update(obs_time)
        
        print(f"\n{'='*60}")
        print(f"Event {event_id}")
        print(f"Time (UTC): {obs_time.iso}")
        print(f"{'='*60}")
        
        # Process each station
        for station in event.get_stations():
            station_id = station.get_id()
            
            # Skip if specific station requested and this isn't it
            if args.station_id is not None and station_id != args.station_id:
                continue
            
            print(f"\nProcessing station {station_id}...")
            
            # Run the beamformer sky mapper
            beamformer.run(event, station, detector)
            
            # Retrieve results from station parameters
            if station.has_parameter(stationParameters.cr_zenith):
                cr_zenith = station.get_parameter(stationParameters.cr_zenith)
                cr_azimuth = station.get_parameter(stationParameters.cr_azimuth)
                
                print(f"  CR direction found:")
                print(f"    Zenith:  {cr_zenith / units.deg:.1f} deg")
                print(f"    Azimuth: {cr_azimuth / units.deg:.1f} deg")
            else:
                print(f"  No direction found (insufficient data)")
    
    # ============================================================
    # Cleanup
    # ============================================================
    
    beamformer.end()
    
    print(f"\n{'='*60}")
    print("Processing complete")
    if args.debug:
        print(f"Debug output saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
