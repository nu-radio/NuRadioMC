import NuRadioReco.detector.detector_mongo
import datetime

class RNOG_detector(NuRadioReco.detector.detector_mongo.Detector):
    def __init__(self, time=datetime.datetime.now()):
        super(RNOG_detector, self).__init__("test")
        self.update(time)

det = RNOG_detector()
