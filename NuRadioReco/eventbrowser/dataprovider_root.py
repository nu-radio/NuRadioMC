from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
import time
import logging
from NuRadioReco.modules.base import module
logger = module.setup_logger(level=logging.INFO)

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
        
        logger.debug(f"Call get_file_handler with user_id {user_id} and file {filename}")
        
        # Apparently at some point the initial user_id None is replaced by a hash although the file did not change. 
        # Do not read the file again but update the user_id.
        # if user_id is not None and len(list(self.__user_instances.keys())) == 1 and list(self.__user_instances.keys())[0] == None:
        #     reader = self.__user_instances.pop(None)
        #     self.__user_instances[user_id] = reader
        if user_id is not None:
            user_id = None
        
        if user_id not in self.__user_instances:
            logger.debug(f"Creater user instance for {user_id} with file {filename}.")

            self.__user_instances[user_id] = None  # This is important! It tells parallel processes that the first file is being read! 
            reader = readRNOGData(try_loading_runtable=False)
            reader.begin(filename, overwrite_sampling_rate=3.2)
            self.__user_instances[user_id] = reader
            
        # While the current file is being read wait for it!
        i = 0
        while i < 15:
            if self.__user_instances[user_id] is not None:
                break
            i += 1
            logger.debug(f"Wait {i} seconds for user instance ...")
            time.sleep(1)
            
        if self.__user_instances[user_id] is None:
            raise ValueError(f"User instance for {user_id} is still None..")

        if filename != self.__user_instances[user_id].get_filenames()[0]:
            logger.debug(f"Creater user instance for {user_id} with new file {filename}.")
            # user is requesting new file -> close current file and open new one
            self.__user_instances[user_id] = readRNOGData(try_loading_runtable=False)
            self.__user_instances[user_id].begin(filename, overwrite_sampling_rate=3.2)

        return self.__user_instances[user_id]
