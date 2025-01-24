import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelCWNotchFilter
from NuRadioReco.modules.RNO_G import channelBlockOffsetFitter
import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.RNO_G.channelGlitchDetector 
import NuRadioReco.detector.RNO_G.rnog_detector

from NuRadioReco.utilities import units, logging as nulogging
#import glitch_removal

import argparse
import logging
import yaml
import time
import pandas as pd
import csv

logger = logging.getLogger("NuRadioReco.example.RNOG.rnog_standard_data_processing")
logger.setLevel(logging.INFO)



class Preprocessor:
    def begin(self):
        pass

    def run(self, evt, station, det, glitch_tag = True, block_offset = True, resample = True, cable_delay = True, dedisperse = True, cw_filt = True, bandpass = True):

        channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()

        channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
        channelResampler.begin()

        channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
        channelBandPassFilter.begin()

        channelCWNotchFilter = NuRadioReco.modules.channelCWNotchFilter.channelCWNotchFilter()
        channelCWNotchFilter.begin()

        hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
        hardwareResponseIncorporator.begin()

        channelGlitchDetector = NuRadioReco.modules.RNO_G.channelGlitchDetector.channelGlitchDetector()

        block_offsets = channelBlockOffsetFitter.channelBlockOffsets()
        #glitch = glitch_removal.GlitchFinder()
        
        det.update(station.get_station_time())

        #Glitch Tagging
        if (glitch_tag == True):
            channelGlitchDetector.run(evt, station)
            

        # Correcting for block offsets
        if (block_offset == True):
            block_offsets.remove_offsets(evt, station)

        # Add cable delay
        if (cable_delay == True):
            channelAddCableDelay.run(evt, station, det, mode='subtract')

        # Resample
        if (resample == True):
            channelResampler.run(evt, station, det, sampling_rate = 5)

        # Hardware response
        if (dedisperse == True):
            hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False, mode='phase_only')

        # CW notch filter
        if (cw_filt == True):
            channelCWNotchFilter.run(evt, station, det)

        # Bandpass filter
        if (bandpass == True):
            channelBandPassFilter.run(evt, station, det, passband = [0.1,0.6], filter_type = "butter", order = 10)
        
        #if (glitch_tag == True):
            #return is_bad


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run standard RNO-G data processing')

    parser.add_argument('--filenames', type=str, nargs="*", help='Specify root data files if not specified in the config file')
    parser.add_argument('--outfile', type=str, help='Output filename (.nur)')

    args = parser.parse_args()
    # nulogging.set_general_log_level(logging.INFO)

    filename = args.filenames
    outfile = args.outfile
    
    preprocessor = Preprocessor()
    preprocessor.begin()

    readRNOGDataMattak = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData(log_level=logging.INFO)
    readRNOGDataMattak.begin(
            filename, read_calibrated_data = False, convert_to_voltage = True, overwrite_sampling_rate = 3.2)
    
    det = NuRadioReco.detector.RNO_G.rnog_detector.Detector()

    eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
    eventWriter.begin(filename=outfile)

    for idx, evt in enumerate(readRNOGDataMattak.run()):
        for station in evt.get_stations():
            preprocessor.run(evt, station, det)
            eventWriter.run(evt, det=None)

    readRNOGDataMattak.end()

