from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
import os
import time
import numpy as np

### we create a wrapper for readRNOGData to mirror the interface of the .nur reader
class readRNOG_wrapper(readRNOGData):
    
    def get_event_ids(self):
        event_infos = self.get_events_information()
        return np.array([(i['run'], i['eventNumber']) for i in event_infos.values()])
    
    def get_event_i(self, i):
        return self.get_event_by_index(i)
    
    def get_event(self, event_id):
        return super().get_event(*event_id)

    def get_n_events(self):
        return self._n_events_total


class DataProviderRoot(object):
    __instance = None

    def __new__(cls):
        if DataProviderRoot.__instance is None:
            DataProviderRoot.__instance = object.__new__(cls)
        return DataProviderRoot.__instance

    def __init__(self, max_user_instances=12):
        """"
        Convenience wrapper for the root-based RNO-G data reader

        Parameters
        ----------
        max_user_instances: int, default=12
            Each unique session id gets its own reader, up to a maximum
            of ``max_user_instances`` concurrent readers. Subsequent
            requests for new readers drop older readers.
                
        """
        self.__max_user_instances = max_user_instances
        self.__user_instances = {}

    def get_file_handler(self, user_id, filename):
        if filename is None:
            return
        if user_id not in self.__user_instances:
            # create new reader for the new user
            reader = readRNOG_wrapper()
            self.__user_instances[user_id] = dict(
                reader=reader, filename=None,
            )
        if filename != self.__user_instances[user_id]['filename']:
            # user is requesting new file -> close current file and open new one
            reader = self.__user_instances[user_id]['reader']
            reader.begin([os.path.dirname(filename)], overwrite_sampling_rate=3.2) #TODO - remove hardcoded sampling rate
            self.__user_instances[user_id] = dict(
                reader=reader, filename=filename,
            )
        
        # update last access time
        self.__user_instances[user_id]['last_access_time'] = time.time()

        # check if we exceed maximum number of concurrent sessions
        if len(self.__user_instances) > self.__max_user_instances:
            users = {
                self.__user_instances[k]['last_access_time']:k
                for k in self.__user_instances.keys()
            }
            users_by_access_time = sorted(users)
            # drop oldest session
            self.__user_instances.pop(users_by_access_time[0])

        return self.__user_instances[user_id]['reader']
