import pickle
import NuRadioReco.framework.hybrid_shower

class HybridInformation():
    def __init__(self):
        self.__hybrid_showers = []
        self.__hybrid_shower_names = []

    def add_hybrid_shower(self, hybrid_shower):
        if hybrid_shower.get_name() in self.__hybrid_shower_names:
            raise ValueError('A hybrid shower with the name {} already exists'.format(hybrid_shower.get_name()))
        self.__hybrid_showers.append(hybrid_shower)
        self.__hybrid_shower_names.append(hybrid_shower.get_name())

    def get_hybrid_showers(self):
        for shower in self.__hybrid_showers:
            yield shower

    def get_hybrid_shower(self, name):
        if name not in self.__hybrid_shower_names:
            raise ValueError('Could not find hybrid shower with name {}'.format(name))
        return self.__hybrid_showers[self.__hybrid_shower_names.index(name)]

    def serialize(self):
        shower_pickles = []
        for shower in self.get_hybrid_showers():
            shower_pickles.append(shower.serialize())
        data = {
            'shower_pickles': shower_pickles
        }
        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        for shower_pkl in pickle.loads(data_pkl)['shower_pickles']:
            shower = NuRadioReco.framework.hybrid_shower.HybridShower('')
            shower.deserialize(shower_pkl)
            self.add_hybrid_shower(shower)
