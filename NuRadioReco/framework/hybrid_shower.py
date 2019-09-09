import NuRadioReco.framework.base_shower

class HybridShower(NuRadioReco.framework.base_shower.BaseShower):
    def __init__(self, name):
        self.__name = name
        self.__hybrid_detector = None
        super().__init__()

    def get_name(self):
        return self.__name

    def set_hybrid_detector(self, hybrid_detector):
        self.__hybrid_detector = hybrid_detector

    def get_hybrid_detector(self):
        return self.__hybrid_detector
