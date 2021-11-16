import NuRadioReco.modules.io.rno_g.rnogDataReader


class DataProviderRoot(object):
    __instance = None

    def __new__(cls):
        if DataProviderRoot.__instance is None:
            DataProviderRoot.__instance = object.__new__(cls)
        return DataProviderRoot.__instance

    def __init__(self):
        self.__user_instances = {}

    def get_file_handler(self, user_id, filename):
        if filename is None:
            return
        if user_id not in self.__user_instances:
            self.__user_instances[user_id] = NuRadioReco.modules.io.rno_g.rnogDataReader.RNOGDataReader([filename])

        if filename != self.__user_instances[user_id].get_filenames()[0]:
            # user is requesting new file -> close current file and open new one
            self.__user_instances[user_id] = NuRadioReco.modules.io.rno_g.rnogDataReader.RNOGDataReader([filename])
            #TODO begin method does not exist in RNOGDataReader
            #self.__user_instances[user_id].begin(filename)
        return self.__user_instances[user_id]
