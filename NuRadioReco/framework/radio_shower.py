import NuRadioReco.framework.base_shower
import pickle


class RadioShower(NuRadioReco.framework.base_shower.BaseShower):

    def __init__(self, shower_id=0, station_ids=None):
        self.__station_ids = station_ids
        super().__init__(shower_id=shower_id)

    def get_station_ids(self):
        return self.__station_ids

    def has_station_ids(self, ids):
        for station_id in ids:
            if station_id not in self.__station_ids:
                return False
        return True

    def serialize(self):
        base_shower_pickle = NuRadioReco.framework.base_shower.BaseShower.serialize(self)
        data = {
            'station_ids': self.__station_ids,
            'base_shower': base_shower_pickle
        }
        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.base_shower.BaseShower.deserialize(self, data['base_shower'])
        self.__station_ids = data['station_ids']
