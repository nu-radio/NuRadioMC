from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

import time
    

class DataProvider(object):
    __instance = None

    def __new__(cls):
        if DataProvider.__instance is None:
            DataProvider.__instance = object.__new__(cls)
        return DataProvider.__instance

    def __init__(self):
        self.__user_instances = {}

    def set_filetype(self, use_root=False):
        self._use_root = use_root
        
    def set_logger(self, logger):
        self.logger = logger
        
    def _initialize_nur_io(self, filename):
        self.logger.info(f"Create NuRadioRecoio object with {filename}")
        return NuRadioRecoio.NuRadioRecoio(filename)
    
    def _initalize_rnog_io(self, filename):
        self.logger.info(f"Create RNO-G reader with {filename}")
        reader = readRNOGData(try_loading_runtable=False)
        reader.begin(filename, overwrite_sampling_rate=3.2)
        return reader
    
    def waiting(self, key, dt=0.05 , t_tot=15, logger=None):
        # While the current file is being read wait for it!
        time_counter = 0
        while time_counter < int(t_tot / dt): 
            if self.__user_instances[key] is not None:
                break
            time_counter += 1
            if time_counter % int(3 / dt) == 0 and logger is not None:
                logger.debug(f"Wait {time_counter * dt} seconds for user instance ...")
            time.sleep(dt)

    def get_file_handler(self, user_id, filename):
        self.logger.debug(f"Call get_file_handler with user_id {user_id} and file {filename}")

        if filename is None:
            return
        
        # Apparently at some point the initial user_id None is replaced by a hash although the file did not change. 
        # Do not read the file again but update the user_id.
        if user_id is not None and user_id not in list(self.__user_instances.keys()) and None in list(self.__user_instances.keys()):
            self.logger.info(f"Replace user_id None by {user_id}")
            # In principle we want to remove the entry for "None".
            # However, parallel processes might be in the while loop a
            # few lines below which would cause an key error when we drop "None"
            # here. Hence, we just copy the object/reference. That should be no
            # problem, it is just ugly AF.
            self.__user_instances[user_id] = None
            
            # If we do not wait here we might copy a None which is not getting updated anymore.
            self.waiting(key=None, logger=self.logger)
            self.__user_instances[user_id] = self.__user_instances[None]

        
        if user_id not in self.__user_instances:
            self.logger.debug(f"Creater user instance for {user_id} with file {filename}.")

            self.__user_instances[user_id] = None  # This is important! It tells parallel processes that the first file is being read! 
            if self._use_root:
                io_reader = self._initalize_rnog_io(filename)
            else:
                io_reader = self._initialize_nur_io(filename)
            
            self.__user_instances[user_id] = io_reader
            
        # Wait until the file has been read
        self.waiting(key=user_id, logger=self.logger)
            
        if self.__user_instances[user_id] is None:
            raise ValueError(f"User instance for {user_id} is still None..")

        if filename != self.__user_instances[user_id].get_filenames()[0]:
            self.logger.debug(f"Create new io instance with file {filename} for user {user_id}.")

            if not self._use_root:
                self.__user_instances[user_id].close_files()
                self.__user_instances[user_id] = self._initialize_nur_io(filename)
            else: 
                self.__user_instances[user_id] = self._initalize_rnog_io(filename)

        return self.__user_instances[user_id]