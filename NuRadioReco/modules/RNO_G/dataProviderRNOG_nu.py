import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
import NuRadioReco.modules.RNO_G.channelGlitchDetector
import NuRadioReco.modules.RNO_G.channelBlockOffsetFitter
import uproot
import NuRadioReco.modules.channelAddCableDelay

import logging
logger = logging.getLogger('NuRadioReco.RNO_G.dataProviderRNOG')

class dataProviderRNOG:

    def __init__(self):

        self.channelGlitchDetector = NuRadioReco.modules.RNO_G.channelGlitchDetector.channelGlitchDetector()
        self.channelBlockOffsetFitter = NuRadioReco.modules.RNO_G.channelBlockOffsetFitter.channelBlockOffsets()
        self.reader = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData()

        self.channelCableDelayAdder = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()


    def begin(self, files, reader_kwargs={}, det=None):
        self.files = files

        self.channelGlitchDetector.begin()
        self.channelBlockOffsetFitter.begin()
        self.reader.begin(self.files, mattak_kwargs={"backend": "uproot"})
        self.channelCableDelayAdder.begin()

        assert det is not None, "Detector object is None, please provide a detector object."
        self.detector = det

    def end(self):
        self.reader.end()
        self.channelGlitchDetector.end()

    def run(self):
        header_path = f"{self.files}/headers.root"
        f = uproot.open(header_path)
        _HEADER_TREES = ("hdr", "header", "hd", "hds", "headers")
        tree = None
        for name in _HEADER_TREES:
            if name in f:
                tree = f[name]
                times = tree["trigger_time"].array(library="np")
                readout_times = tree["readout_time"].array(library="np")
        if tree is None:
            times = None
            readout_times = None
        count = 0 
        print(len(times))
        print(sum(1 for _ in self.reader.run()))
        
        for event in self.reader.run():
            station = event.get_station()
            self.channelGlitchDetector.run(event, station, self.detector)

            self.channelBlockOffsetFitter.run(event, station, self.detector)

            self.channelCableDelayAdder.run(event, station, self.detector, mode='subtract')
            time = None 
            readout_time = None

            if (times is not None):
                time = times[count]
            if (readout_times is not None):
                readout_time = readout_times[count]
            
            count += 1 

            yield event, time, readout_time 

