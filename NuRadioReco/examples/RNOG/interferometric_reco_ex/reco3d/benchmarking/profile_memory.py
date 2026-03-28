"""Profile peak memory usage through each stage of 3D reconstruction.

Runs one event through the full reconstruction pipeline and reports peak RSS
at each stage boundary using resource.getrusage. Optionally uses tracemalloc
for Python-side allocation detail.
"""

import argparse
import datetime
import logging
import os
import resource
import sys
import time

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECO3D_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, RECO3D_DIR)


def get_peak_rss_kb():
    """Return peak RSS in KB (Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def checkpoint(name, checkpoints, t0, tracemalloc_on=False):
    """Record a memory/time checkpoint.

    Args:
        name: label for this checkpoint
        checkpoints: list to append to
        t0: start time
        tracemalloc_on: whether tracemalloc is active
    """
    rss = get_peak_rss_kb()
    elapsed = time.time() - t0
    entry = {'name': name, 'rss_kb': rss, 'elapsed': elapsed}
    if tracemalloc_on:
        import tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        entry['tracemalloc_current_mb'] = current / 1e6
        entry['tracemalloc_peak_mb'] = peak / 1e6
    checkpoints.append(entry)


def profile(config_path, nur_file, mode, event_index, detector_file,
            use_tracemalloc):
    """Run one event through the pipeline with memory checkpoints.

    Args:
        config_path: path to reco YAML config
        nur_file: path to input NUR file
        mode: "hw", "rx", or "rxtx"
        event_index: which event in the file to process
        detector_file: path to detector JSON.xz (optional)
        use_tracemalloc: enable tracemalloc for Python allocation tracking
    """
    if use_tracemalloc:
        import tracemalloc
        tracemalloc.start()

    checkpoints = []
    t0 = time.time()

    checkpoint('start', checkpoints, t0, use_tracemalloc)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key, val in config.items():
        if isinstance(val, str) and '$' in val:
            config[key] = os.path.expandvars(val)

    from NuRadioReco.detector.RNO_G import rnog_detector
    from NuRadioReco.modules.channelResampler import channelResampler
    from NuRadioReco.modules.channelAddCableDelay import channelAddCableDelay
    from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import (
        hardwareResponseIncorporator)
    from NuRadioReco.modules.io.eventReader import eventReader
    from NuRadioReco.utilities import units
    from NuRadioReco.modules.interferometricDirectionReconstruction3D import (
        InterferometricReco3D)

    station_id = config['station_id']

    det_path = detector_file or config.get('detector_file')
    if det_path:
        det_path = os.path.expandvars(det_path)
    det = rnog_detector.Detector(
        detector_file=det_path,
        log_level=logging.WARNING, select_stations=station_id)
    det_date = config.get('detector_date', '2022-10-01')
    det.update(datetime.datetime.fromisoformat(det_date))

    checkpoint('detector_init', checkpoints, t0, use_tracemalloc)

    reco = InterferometricReco3D()
    reco.begin(station_id, config, det)

    checkpoint('table_load', checkpoints, t0, use_tracemalloc)

    reader = eventReader()
    reader.begin(nur_file)
    event_ids = reader._eventReader__fin.get_event_ids()
    eid = event_ids[event_index]
    evt = reader._eventReader__fin.get_event(event_id=eid)
    stn = evt.get_station(station_id)

    cable_delay = channelAddCableDelay()
    cable_delay.begin()
    hw_response = hardwareResponseIncorporator()
    hw_response.begin()
    resampler = channelResampler()
    resampler.begin()

    if config.get('apply_cable_delay', True):
        cable_delay.run(evt, stn, det, mode='subtract')
    if config.get('apply_hw_phase_removal', True):
        hw_response.run(evt, stn, det, sim_to_data=False, mode='phase_only')
    if config.get('apply_upsampling', True):
        resampler.run(evt, stn, det, sampling_rate=10 * units.GHz)

    checkpoint('preprocessing', checkpoints, t0, use_tracemalloc)

    result = reco.run(evt, stn, det, config)

    checkpoint('reco_complete', checkpoints, t0, use_tracemalloc)

    if mode in ('rx', 'rxtx'):
        # Pass 2 would go here, but requires re-reading the event
        # and applying dedispersion. For memory profiling purposes,
        # the table load and pass 1 dominate memory.
        pass

    reader.end()
    reco.end()

    checkpoint('cleanup', checkpoints, t0, use_tracemalloc)

    if use_tracemalloc:
        import tracemalloc
        tracemalloc.stop()

    # Print results
    print(f"File: {nur_file}")
    print(f"Event index: {event_index}, Mode: {mode}")
    print(f"Config: {config_path}")
    print()

    header = "| Stage | Peak RSS (MB) | Delta (MB) | Wall time (s) |"
    if use_tracemalloc:
        header = ("| Stage | Peak RSS (MB) | Delta (MB) | Wall time (s) "
                  "| tracemalloc current (MB) | tracemalloc peak (MB) |")
    sep = "|" + "|".join(["-------"] * header.count("|")) + "|"

    print(header)
    print(sep)

    prev_rss = 0
    for cp in checkpoints:
        rss_mb = cp['rss_kb'] / 1024
        delta_mb = (cp['rss_kb'] - prev_rss) / 1024
        row = (f"| {cp['name']} | {rss_mb:.1f} | "
               f"{'+' if delta_mb >= 0 else ''}{delta_mb:.1f} | "
               f"{cp['elapsed']:.2f} |")
        if use_tracemalloc:
            row += (f" {cp.get('tracemalloc_current_mb', 0):.1f} |"
                    f" {cp.get('tracemalloc_peak_mb', 0):.1f} |")
        print(row)
        prev_rss = cp['rss_kb']

    print()
    final_mb = checkpoints[-1]['rss_kb'] / 1024
    print(f"Final peak RSS: {final_mb:.1f} MB ({checkpoints[-1]['rss_kb']} KB)")


def main():
    """Parse arguments and run memory profiling."""
    parser = argparse.ArgumentParser(
        description='Profile peak memory usage through 3D reconstruction stages')
    parser.add_argument('--config', required=True, help='Reco config YAML')
    parser.add_argument('--nur-file', required=True, help='Input NUR file')
    parser.add_argument('--mode', default='hw', choices=['hw', 'rx', 'rxtx'],
                        help='Reconstruction mode (default: hw)')
    parser.add_argument('--event-index', type=int, default=0,
                        help='Event index in file (default: 0)')
    parser.add_argument('--detector-file', default=None,
                        help='Detector JSON.xz file (overrides config)')
    parser.add_argument('--tracemalloc', action='store_true',
                        help='Enable tracemalloc for Python allocation detail')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    profile(args.config, args.nur_file, args.mode, args.event_index,
            args.detector_file, args.tracemalloc)


if __name__ == '__main__':
    main()
