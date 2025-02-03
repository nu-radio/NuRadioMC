"""
Script that tests the `channelGalacticNoiseAdder` module
"""
from NuRadioReco.modules import channelGalacticNoiseAdder
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
from NuRadioReco.detector.detector import Detector
from NuRadioReco.utilities import units
import numpy as np
import astropy.time
import logging
import argparse
import time

logger = logging.getLogger('NuRadioReco.channelGalacticNoiseAdder')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Script to test the channelGalacticNoiseAdder module.")
    argparser.add_argument('--skymodel', default='gsm2008', help='Which skymodel to test')
    argparser.add_argument('--n_trials', '-n', default=10, help='Number of runs')
    args = argparser.parse_args()


    noiseadder = channelGalacticNoiseAdder.channelGalacticNoiseAdder()
    noiseadder.begin(skymodel=args.skymodel, seed=1234)
    det = Detector(json_filename='RNO_G/RNO_season_2023.json', antenna_by_depth=False)
    det.update(astropy.time.Time.now())

    evt = NuRadioReco.framework.event.Event(0, 0)
    evt.set_station(NuRadioReco.framework.station.Station(11))
    station = evt.get_station(11)
    station.set_station_time(astropy.time.Time.now())

    n_trials = args.n_trials
    channel_ids = [13,16,19] # only look at upward-facing LPDAs

    for i in channel_ids: # only look at upward-facing LPDAs
        channel = NuRadioReco.framework.channel.Channel(i)
        channel.set_trace(np.zeros(2048), sampling_rate=2.4)
        station.add_channel(channel)

    logger.status(f"Simulating galactic noise for {len(channel_ids)} channels, {n_trials} trials...")
    vrms = np.zeros((len(channel_ids), n_trials))
    for i in range(n_trials):
        if i==1:
            t0 = time.time() # we start after the first trial to exclude the antenna loading time

        # reset channel trace
        for channel_id in channel_ids:
            channel = station.get_channel(channel_id)
            channel.set_frequency_spectrum(0*channel.get_frequency_spectrum(), sampling_rate='same')

        noiseadder.run(evt, station, det)
        for ichannel, channel_id in enumerate(channel_ids):
            vrms[ichannel, i] = np.std(station.get_channel(channel_id).get_trace())

    dt_per_trial = (time.time() - t0) / (len(channel_ids) * (n_trials - 1)) * units.s

    logger.status(
        "median RMS noise voltage at antenna outputs: "
        f"{np.round(np.median(vrms, axis=1) / units.microvolt, 3)} muV "
        "(Expected ~6 muV)"
        )
    logger.status(
        f"Simulating galactic noise took {dt_per_trial/units.ms:.0f} ms per channel and trial."
    )