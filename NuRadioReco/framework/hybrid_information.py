
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
