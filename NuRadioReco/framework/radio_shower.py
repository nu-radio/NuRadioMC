import NuRadioReco.framework.base_shower


class RadioShower(NuRadioReco.framework.base_shower.BaseShower):
    def __init__(self, station_ids=None):
        self.__station_ids = station_ids
        super().__init__(self)

    def get_station_ids(self):
        return self.__station_ids

    def has_station_ids(self, ids):
        for id in ids:
            if id not in self.__station_ids:
                return False
        return True
