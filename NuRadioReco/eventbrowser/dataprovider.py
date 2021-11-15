import NuRadioReco.eventbrowser.dataprovider_root
import NuRadioReco.eventbrowser.dataprovider_nur


class DataProvider(object):
    __instance = None

    def __new__(cls):
        if DataProvider.__instance is None:
            DataProvider.__instance = object.__new__(cls)
        return DataProvider.__instance

    def __init__(self):
        self.__data_provider = None

    def set_filetype(self, use_root):
        if use_root:
            self.__data_provider = NuRadioReco.eventbrowser.dataprovider_root.DataProviderRoot()
        else:
            self.__data_provider = NuRadioReco.eventbrowser.dataprovider_nur.DataProvider()

    def get_file_handler(self, user_id, filename):
        return self.__data_provider.get_file_handler(user_id, filename)
