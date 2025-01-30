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
import yaml
import time

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


def use_module(name, config):
    """
    Simple helper function returns True if module is supposed to be used

    Parameters
    ----------
    name : str
        Name of the module
    config : dict
        (Yaml) configuration

    Returns
    -------
    use : bool
        True if module is supposed to be used
    """
    return name in config and config[name]["use"]


def process_event(evt, det, config):
    """
    Process a single event

    Parameters
    ----------
    evt : NuRadioReco.event.Event
        Event to process
    det : NuRadioReco.detector.detector.Detector
        Detector object
    config : dict
        (Yaml) configuration
    """

    # Get the station. This will throw an error if more than one station is in the event.
    station = evt.get_station()
    det.update(station.get_station_time())

    # Resample
    if use_module("channelResampler", config):
        channelResampler.run(evt, station, det, **config["channelResampler"]["kwargs"])

    # Hardware response
    if use_module("hardwareResponseIncorporator", config):
        hardwareResponseIncorporator.run(
            evt, station, det, sim_to_data=False, **config["hardwareResponseIncorporator"]["kwargs"])

    # CW notch filter
    if use_module("channelCWNotchFilter", config):
        channelCWNotchFilter.run(evt, station, det)

    # Bandpass filter
    if use_module("channelBandPassFilter", config):
        channelBandPassFilter.run(evt, station, det, **config["channelBandPassFilter"]["kwargs"])


def process_data(config):
    """
    Processing of RNO-G data.

    Parameters
    ----------
    config : dict
        (Yaml) configuration
    """

    # Initialize detector class
    det = NuRadioReco.detector.RNO_G.rnog_detector.Detector(
        **config["detector"]["kwargs"]
    )

    # Initialize modules
    paths = config["readRNOGDataMattak"]["kwargs"].pop("filenames")
    dataProviderRNOG = NuRadioReco.modules.RNO_G.dataProviderRNOG.dataProvideRNOG()
    dataProviderRNOG.begin(
        files=paths, reader_kwargs=config["readRNOGDataMattak"]["kwargs"], det=det)

    eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
    eventWriter.begin(filename=config["eventWriter"]["kwargs"]['filename'])

    # For time logging
    t_total = 0

    # Loop over all events (the reader module has options to select events -
    # see class documentation or module arguements in config file)
    for idx, evt in enumerate(dataProviderRNOG.run()):

        if (idx + 1) % 50 == 0:
            print(f'"Processing events: {idx + 1}\r', end="")

        t0 = time.time()
        process_event(evt, det, config)

        # Write event - the RNO-G detector class is not stored within the nur files.
        eventWriter.run(evt, det=None, mode=config["eventWriter"]["kwargs"]['mode'])

        logger.debug("Time for event: %f", time.time() - t0)
        t_total += time.time() - t0

    dataProviderRNOG.end()

    logger.info(
        f"Processed {idx + 1} events:"
        f"\n\tTotal time: {t_total:.2f}s"
        f"\n\tTime per event: {t_total / (idx + 1):.2f}s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run standard RNO-G data processing')

    parser.add_argument('config', type=str, help='Yaml config file to steer the data procession and reconstruction')
    parser.add_argument('--filenames', type=str, nargs="*", help='Specify root data files if not specified in the config file')

    args = parser.parse_args()
    # nulogging.set_general_log_level(logging.INFO)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if config["readRNOGDataMattak"]["kwargs"]["filenames"] is None:
        config["readRNOGDataMattak"]["kwargs"]["filenames"] = args.filenames

    process_data(config)
