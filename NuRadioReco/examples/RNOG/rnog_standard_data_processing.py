import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelCWNotchFilter

import NuRadioReco.modules.RNO_G.dataProviderRNOG
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.io.eventWriter

import NuRadioReco.detector.RNO_G.rnog_detector
from NuRadioReco.utilities import units, logging as nulogging

import argparse
import logging
import time
import os

logger = logging.getLogger("NuRadioReco.example.RNOG.rnog_standard_data_processing")
logger.setLevel(logging.INFO)

channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

channelCWNotchFilter = NuRadioReco.modules.channelCWNotchFilter.channelCWNotchFilter()
channelCWNotchFilter.begin()

hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponseIncorporator.begin()


def process_event(evt, det):
    """
    Process a single event

    Parameters
    ----------
    evt : NuRadioReco.event.Event
        Event to process
    det : NuRadioReco.detector.detector.Detector
        Detector object

    Returns
    -------
    evt : NuRadioReco.event.Event
        Processed event
    """

    # Get the station. This will throw an error if more than one station is in the event.
    station = evt.get_station()
    det.update(station.get_station_time())

    # Resample
    channelResampler.run(evt, station, det, sampling_rate=5 * units.GHz)

    # Hardware response
    hardwareResponseIncorporator.run(
        evt, station, det, sim_to_data=False, mode='phase_only')

    # Bandpass filter
    channelBandPassFilter.run(
        evt, station, det,
        passband=[0.1 * units.GHz, 0.6 * units.GHz],
        filter_type='butter', order=10)

    # CW notch filter
    channelCWNotchFilter.run(evt, station, det)

    return evt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run standard RNO-G data processing')

    parser.add_argument('--filenames', type=str, nargs="*",
                        help='Specify root data files if not specified in the config file')
    parser.add_argument('--outputfile', type=str, nargs=1, default=None)

    args = parser.parse_args()
    nulogging.set_general_log_level(logging.INFO)

    if args.outputfile is None:
        if len(args.filenames) > 1:
            raise ValueError("Please specify an output file")

        path = args.filenames[0]

        if path.endswith(".root"):
            args.outputfile = path.replace(".root", ".nur")
        elif os.path.isdir(path):
            args.outputfile = os.path.join(path, "output.nur")

    # Initialize detector class
    det = NuRadioReco.detector.RNO_G.rnog_detector.Detector(

    )

    # Initialize io modules
    dataProviderRNOG = NuRadioReco.modules.RNO_G.dataProviderRNOG.dataProvideRNOG()
    dataProviderRNOG.begin(files=args.filenames, det=det)

    eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
    eventWriter.begin(filename=args.outputfile)

    # For time logging
    t_total = 0

    # Loop over all events (the reader module has options to select events -
    # see class documentation or module arguements in config file)
    for idx, evt in enumerate(dataProviderRNOG.run()):

        if (idx + 1) % 50 == 0:
            logger.info(f'"Processing events: {idx + 1}\r', end="")

        t0 = time.time()
        process_event(evt, det)

        # Write event - the RNO-G detector class is not stored within the nur files.
        eventWriter.run(evt, det=None)

        logger.debug("Time for event: %f", time.time() - t0)
        t_total += time.time() - t0

    dataProviderRNOG.end()

    logger.info(
        f"Processed {idx + 1} events:"
        f"\n\tTotal time: {t_total:.2f}s"
        f"\n\tTime per event: {t_total / (idx + 1):.2f}s")
