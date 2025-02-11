import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
import NuRadioReco.modules.RNO_G.channelGlitchDetector
import NuRadioReco.modules.RNO_G.channelBlockOffsetFitter

import NuRadioReco.modules.channelAddCableDelay

import logging
logger = logging.getLogger('NuRadioReco.RNO_G.dataProviderRNOG')

class dataProvideRNOG:

    def __init__(self):

        self.channelGlitchDetector = NuRadioReco.modules.RNO_G.channelGlitchDetector.channelGlitchDetector()
        self.channelBlockOffsetFitter = NuRadioReco.modules.RNO_G.channelBlockOffsetFitter.channelBlockOffsets()
        self.reader = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData()

        self.channelCableDelayAdder = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()


    def begin(self, files, reader_kwargs={}, det=None):
        self.files = files

        self.channelGlitchDetector.begin()
        self.channelBlockOffsetFitter.begin()
        self.reader.begin(self.files, **reader_kwargs)
        self.channelCableDelayAdder.begin()

        assert det is not None, "Detector object is None, please provide a detector object."
        self.detector = det

    def end(self):
        self.reader.end()
        self.channelGlitchDetector.end()

    def run(self):

        for event in self.reader.run():

            # This will throw an error if the event has more than one station
            station = event.get_station()
            self.detector.update(station.get_station_time())

            self.channelGlitchDetector.run(event, station, self.detector)

            self.channelBlockOffsetFitter.run(event, station, self.detector)

            self.channelCableDelayAdder.run(event, station, self.detector, mode='subtract')

            yield event
