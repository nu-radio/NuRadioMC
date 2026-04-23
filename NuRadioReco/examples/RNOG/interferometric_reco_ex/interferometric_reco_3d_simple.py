#!/usr/bin/env python3
"""Minimal reference example for 3D interferometric direction reconstruction.

Demonstrates the core usage pattern: load detector, read RNO-G data,
preprocess (cable delay, hardware response, resample, bandpass), run
InterferometricReco3D, print per-event results.

A richer driver (mode switching, pass-2 dedispersion, validation
outputs, chunked HDF5 writing) lives next to this file as
``interferometric_reco_3d_advanced.py``. Start there if you are running
a production analysis; start here if you just want to see how the
engine is wired up.

Usage:
    python interferometric_reco_3d_simple.py \\
        --config reco_config.yaml --input path/to/file.nur
"""

import argparse
import datetime
import logging

import yaml
import numpy as np

import NuRadioReco.detector.detector as detector
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.modules.channelAddCableDelay import channelAddCableDelay
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from NuRadioReco.utilities import units


def init_detector(config):
    """Build and update a Detector from config."""
    det_file = config.get('detector_file', None)
    det_date = datetime.datetime.fromisoformat(
        config.get('detector_date', '2022-10-01'))
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


def preprocess(event, station, det, modules, sampling_rate, passband):
    """Apply cable delay, hardware response, resampling, and bandpass filter."""
    modules['cable_delay'].run(event, station, det, mode='subtract')
    modules['hw_response'].run(event, station, det, sim_to_data=False)
    modules['resampler'].run(event, station, det, sampling_rate=sampling_rate)
    modules['bandpass_filter'].run(
        event, station, det, passband=passband,
        filter_type='butter', order=8)


def iter_events(input_path):
    """Yield (event, station) from a NUR or RNO-G ROOT input."""
    if input_path.endswith('.nur'):
        reader = eventReader()
        reader.begin([input_path])
        for evt in reader.run():
            yield evt, evt.get_station(evt.get_station_ids()[0])
    else:
        reader = readRNOGData()
        reader.begin([input_path])
        for evt in reader.run():
            yield evt, evt.get_station(evt.get_station_ids()[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Reco config YAML.')
    p.add_argument('--input', required=True, help='NUR file or ROOT run dir.')
    p.add_argument('--n-events', type=int, default=5,
                   help='Max events to process.')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')

    with open(args.config) as f:
        config = yaml.safe_load(f)

    det = init_detector(config)
    station_id = config['station_id']

    modules = {
        'cable_delay': channelAddCableDelay(),
        'hw_response': hardwareResponseIncorporator(),
        'resampler': channelResampler(),
        'bandpass_filter': channelBandPassFilter(),
    }
    for m in modules.values():
        m.begin()

    reco = InterferometricReco3D()
    reco.begin(station_id, config, det)

    sampling_rate = config.get('sampling_rate', 3.2) * units.GHz
    passband = config.get('passband', [0.1, 0.6]) * units.GHz

    for i, (evt, stn) in enumerate(iter_events(args.input)):
        if i >= args.n_events:
            break
        preprocess(evt, stn, det, modules, sampling_rate, passband)
        result = reco.run(evt, stn, det)
        zen_deg = np.degrees(result.get('zenith', np.nan))
        az_deg = np.degrees(result.get('azimuth', np.nan))
        print(f"evt {evt.get_run_number()}:{evt.get_id()}  "
              f"rho={result.get('rho', np.nan):.1f} m  "
              f"phi={result.get('phi', np.nan):.1f} deg  "
              f"z={result.get('z', np.nan):.1f} m  "
              f"corr={result.get('max_corr', np.nan):.3f}  "
              f"zen={zen_deg:.1f} deg  az={az_deg:.1f} deg")


if __name__ == "__main__":
    main()
