import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
import NuRadioReco.modules.RNO_G.channelGlitchDetector
import NuRadioReco.modules.RNO_G.channelBlockOffsetFitter

import NuRadioReco.modules.channelAddCableDelay

import logging
logger = logging.getLogger('NuRadioReco.RNO_G.dataProviderRNOG')

class dataProvideRNOG:

    def __init__(self):

        self.channelGlitchDetector = NuRadioReco.modules.RNO_G.channelGlitchDetector.channelGlitchDetector()
        self.channelBlockOffsetFitter = NuRadioReco.modules.RNO_G.channelBlockOffsetFitter.channelBlockOffsetFitter()
        self.reader = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData()

        self.channelCableDelayAdder = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()


    def begin(self, files, reader_kwargs={}, det=None):
        self.files = files

        self.channelGlitchDetector.begin()
        self.channelBlockOffsetFitter.begin()
        self.reader.begin(self.files, apply_baseline_correction=None, **reader_kwargs)
        self.channelCableDelayAdder.begin()

        assert det is not None, "Detector object is None, please provide a detector object."
        self.detector = det

    def end(self):
        pass

    def run(self):

        for event in self.reader.run():

            # This will throw an error if the event has more than one station
            station = event.get_station()

            if self.channelGlitchDetector.run(event, station, self.detector):
                logger.warning(f"Glitch found in run.event {event.get_run_number()}{event.get_id()}. "
                               "Skipping this event.")
                continue  # skip the rest of the loop if a glitch is found in the station

            self.channelBlockOffsetFitter.run(event, station, self.detector)

            self.channelCableDelayAdder.run(evt, station, self.detector, mode='subtract')

            yield event
