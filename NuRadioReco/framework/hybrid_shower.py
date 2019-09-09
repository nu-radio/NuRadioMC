import NuRadioReco.framework.base_shower
import pickle

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

    def serialize(self):
        base_shower_pickle = NuRadioReco.framework.base_shower.BaseShower.serialize(self)
        data = {
            'base_shower': base_shower_pickle,
            'name': self.__name
        }
        return pickle.dumps(data, protocol=2)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.base_shower.BaseShower.deserialize(self, data['base_shower'])
        self.__name = data['name']
