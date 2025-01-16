import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelCWNotchFilter

import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
import NuRadioReco.modules.io.eventWriter

import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator

import NuRadioReco.detector.RNO_G.rnog_detector

import argparse
import yaml


def use_module(name, config):
    return name in config and config[name]["use"]


def process_data(config):

    # Initialize modules

    channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()

    channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
    channelResampler.begin()

    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
    channelBandPassFilter.begin()

    channelCWNotchFilter = NuRadioReco.modules.channelCWNotchFilter.channelCWNotchFilter()
    channelCWNotchFilter.begin()

    eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
    eventWriter.begin(filename=config["eventWriter"]["args"]['filename'])

    hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
    hardwareResponseIncorporator.begin()

    def process_event(evt, det, config):

        for station in evt.get_stations():

            detector.update(station.get_station_time())

            # Correcting for block offsets is already performed in the readRNOGDataMattak module

            # Add cable delay
            if use_module("channelAddCableDelay", config):
                channelAddCableDelay.run(evt, station, det, mode='subtract')

            # Resample
            if use_module("channelResampler", config):
                channelResampler.run(evt, station, det, **config["channelResampler"]["args"])

            # Hardware response
            if use_module("hardwareResponseIncorporator", config):
                hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False)

            # CW notch filter
            if use_module("channelCWNotchFilter", config):
                channelCWNotchFilter.run(evt, station, det)

            # Bandpass filter
            if use_module("channelBandPassFilter", config):
                channelBandPassFilter.run(evt, station, det, **config["channelBandPassFilter"]["args"])

            # Write event
            eventWriter.run(evt, det=None, mode=config["eventWriter"]["args"]['mode'])


    detector = NuRadioReco.detector.RNO_G.rnog_detector.Detector(
        **config["detector"]["args"]
    )

    paths = config["readRNOGDataMattak"]["args"].pop("filenames")
    readRNOGDataMattak = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData()
    readRNOGDataMattak.begin(
        paths, **config["readRNOGDataMattak"]["args"]
    )

    for event in readRNOGDataMattak.run():
        process_event(event, detector, config)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run standard RNO-G data processing')

    parser.add_argument('config', type=str, help='Yaml config file')
    parser.add_argument('--rootfiles', type=str, nargs="*", help='Specify to root files to read & process')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if config["readRNOGDataMattak"]["args"]["filenames"] is None:
        config["readRNOGDataMattak"]["args"]["filenames"] = args.rootfiles

    process_data(config)
