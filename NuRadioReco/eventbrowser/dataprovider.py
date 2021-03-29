from __future__ import absolute_import, division, print_function  # , unicode_literals
from NuRadioReco.modules.io import NuRadioRecoio


class DataProvider(object):
    __instance = None

    def __new__(cls):
        if DataProvider.__instance is None:
            DataProvider.__instance = object.__new__(cls)
        return DataProvider.__instance

    def __init__(self):
        self.__user_instances = {}

    def get_arianna_io(self, user_id, filename):
        if filename is None:
            return
        if user_id not in self.__user_instances:
            self.__user_instances[user_id] = NuRadioRecoio.NuRadioRecoio(filename)
        if filename != self.__user_instances[user_id].get_filenames()[0]:
            # user is requesting new file -> close current file and open new one
            self.__user_instances[user_id].close_files()
            self.__user_instances[user_id] = NuRadioRecoio.NuRadioRecoio(filename)
        return self.__user_instances[user_id]
