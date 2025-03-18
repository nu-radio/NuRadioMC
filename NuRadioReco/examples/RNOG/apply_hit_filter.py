"""
Applying RNO-G Hit Filter (this script was modified from data_analysis_example_advanced.py).

Use the standard RNO-G data processing and apply the Hit Filter,
exclude forced triggers,
then save RF events that passed the filter.
"""

import argparse
import logging
import numpy as np
import NuRadioReco.modules.RNO_G.dataProviderRNOG
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelCWNotchFilter
import NuRadioReco.detector.RNO_G.rnog_detector
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.RNO_G.stationHitFilter
from NuRadioReco.utilities import units, logging as nulogging


logger = logging.getLogger("NuRadioReco.example.RNOG.apply_hit_filter")
logger.setLevel(nulogging.LOGGING_STATUS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply the Hit Filter')

    parser.add_argument('filenames', type=str, nargs="*",
                        help='Specify root data files if not specified in the config file')
    parser.add_argument('--outputfile', type=str, required=True, help='Specify the output file')
    parser.add_argument('--detectorfile', type=str, nargs=1, default=None,
                        help="Specify detector file. If you do not specified a file. "
                        "the description is queried from the database.")

    args = parser.parse_args()
    nulogging.set_general_log_level(logging.ERROR)
    args.outputfile = args.outputfile

    logger.status(f"writing output to {args.outputfile}")

    # Initialize detector class
    det = NuRadioReco.detector.RNO_G.rnog_detector.Detector(detector_file=args.detectorfile)

    # Initialize io modules
    dataProviderRNOG = NuRadioReco.modules.RNO_G.dataProviderRNOG.dataProviderRNOG()
    dataProviderRNOG.begin(files=args.filenames, det=det)
    info = dataProviderRNOG.reader.get_events_information(keys=["station", "run", "eventNumber", "triggerType"])

    eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
    eventWriter.begin(filename=args.outputfile)

    # Initialize additional modules
    channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
    channelResampler.begin()

    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
    channelBandPassFilter.begin()

    channelCWNotchFilter = NuRadioReco.modules.channelCWNotchFilter.channelCWNotchFilter()
    channelCWNotchFilter.begin()

    hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
    hardwareResponseIncorporator.begin()

    # Initialize Hit Filter
    stationHitFilter = NuRadioReco.modules.RNO_G.stationHitFilter.stationHitFilter()
    stationHitFilter.begin()

    # Count events
    n_events_FT = 0
    n_events_passed = 0

    # Loop over all events
    for idx, evt in enumerate(dataProviderRNOG.run()):

        if (idx + 1) % 50 == 0:
            print(f'Processed events: {idx + 1}')

        is_FT = info[idx].get('triggerType') == "FORCE"
        n_events_FT += int(is_FT)

        if not is_FT:
            station = evt.get_station()

            det.update(station.get_station_time())

            channelResampler.run(evt, station, det, sampling_rate=5 * units.GHz)

            channelBandPassFilter.run(
                evt, station, det,
                passband=[0.1 * units.GHz, 0.6 * units.GHz],
                filter_type='butter', order=10)

            hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False, mode='phase_only')

            channelCWNotchFilter.run(evt, station, det)

            # Hit Filter
            is_passed_HF = stationHitFilter.run(evt, station, det)
            n_events_passed += int(is_passed_HF)

            # Down sample before saving
            channelResampler.run(evt, station, det, sampling_rate=2 * units.GHz)

            # Write out events
            if is_passed_HF:
                eventWriter.run(evt, det=None, mode={'Channels':True, "ElectricFields":True})

    dataProviderRNOG.end()
    eventWriter.end()

    logger.status(
        f"\nTotal: {idx + 1} events"
        f"\nForced Triggers: {n_events_FT} events"
        f"\nRF Triggers: {idx + 1 - n_events_FT} events"
        f"\nRF Passed Hit Filter: {n_events_passed} events")
