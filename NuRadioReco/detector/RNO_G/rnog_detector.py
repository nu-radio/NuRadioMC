import NuRadioReco.detector.detector_base

import astropy
import datetime

class RNOG_Detector(NuRadioReco.detector.detector_base.DetectorBase):
    def __init__(self, time=datetime.datetime.now()):
        super(RNOG_Detector, self).__init__(source="mongo")
        self.update(time)

det = RNOG_Detector()
