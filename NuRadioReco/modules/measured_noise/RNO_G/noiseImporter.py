import numpy as np
import glob
import os
import random
import collections

from NuRadioReco.modules.io.rno_g.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units

import logging


class noiseImporter:
    """
    Imports recorded traces from RNOG stations. 
    
    """


    def begin(self, noise_folders, file_pattern="*",
              match_station_id=False, station_ids=None,
              channel_mapping=None, scramble_noise_file_order=True,
              log_level=logging.INFO, reader_kwargs={}):
        """
        
        Parameters
        ----------
        noise_folders: str or list(str)
            Folder(s) containing noise file(s). Search in any subfolder as well.
            
        file_patters: str
            File patters used to search for directories, (Default: "*", other examples might be "combined")
            
        match_station_id: bool
            If True, add only noise from stations with the same id. (Default: False)
        
        station_ids: list(int)
            Only add noise from those station ids. If None, use any station. (Default: None)
        
        channel_mapping: dict or None
            option relevant for MC studies of new station designs where we do not
            have forced triggers for. The channel_mapping dictionary maps the channel
            ids of the MC station to the channel ids of the noise data
            Default is None which is 1-to-1 mapping
            
        scramble_noise_file_order: bool
            If True, randomize the order of noise files before reading them. (Default: True)
        
        log_level: loggging log level
            the log level, default logging.INFO

        reader_kwargs: dict
            Optional arguements passed to readRNOGDataMattak
        """
        
        self.logger = logging.getLogger('NuRadioReco.RNOG.noiseImporter')
        self.logger.setLevel(log_level)
        
        self._match_station_id = match_station_id
        self.__station_ids = station_ids
        
        self.__channel_mapping = channel_mapping
        
        self.logger.info(f"\n\tMatch station id: {match_station_id}"
                    f"\n\tUse noise from only those stations: {station_ids}"
                    f"\n\tUse the following channel mapping: {channel_mapping}"
                    f"\n\tRandomize sequence of noise files: {scramble_noise_file_order}")
        
        if not isinstance(noise_folders, list):
            noise_folders = [noise_folders]
        
        # find all subfolders
        noise_files = []
        for noise_folder in noise_folders:
            noise_files += glob.glob(f"{noise_folder}/**/{file_pattern}root", recursive=True)
        self.__noise_folders = np.unique([os.path.dirname(e) for e in noise_files])       
        
        self.logger.info(f"Found {len(self.__noise_folders)}")
        if not len(self.__noise_folders):
            raise ValueError
                
        if scramble_noise_file_order:
            random.shuffle(self.__noise_folders)
        
        if "log_level" in reader_kwargs:
            log_level_reader = reader_kwargs.pop("log_level")
        else:
            log_level_reader = log_level

        self._noise_reader = readRNOGData()                
        selectors = [lambda einfo: einfo.triggerType == "FORCE"]
        self._noise_reader.begin(self.__noise_folders, selectors=selectors, 
                                 log_level=log_level_reader,
                                 **reader_kwargs)


        self.logger.info("Get event informations ...")
        # instead of reading all noise events into memory we only get certain information here and read all data in run()
        
        noise_information = self._noise_reader.get_event_informations(keys=["station"])
        self.__event_index_list = np.array(list(noise_information.keys()))
        self.__station_id_list = np.array([ele["station"] for ele in noise_information.values()])
        
        self._n_use_event = collections.defaultdict(int)
       

    def __get_noise_channel(self, channel_id):
        if self.__channel_mapping is None:
            return channel_id
        else:
            return self.__channel_mapping[channel_id]
        

    @register_run()
    def run(self, evt, station, det):

        if self._match_station_id:
            # select only noise events from simulated station id
            station_mask = self.__station_id_list == station.get_id()
            if not np.any(station_mask):
                raise ValueError(f"No station with id {station.get_id()} in noise data.")
                            
        else:
            # select all noise events
            station_mask = np.full_like(self.__event_index_list, True)

        # int(..) necessary because pyroot can not handle np.int64
        i_noise = int(np.random.choice(self.__event_index_list[station_mask]))
        self._n_use_event[i_noise] += 1
        noise_event = self._noise_reader.read_event(i_noise)
        
        station_id = noise_event.get_station_ids()[0]
        noise_station = noise_event.get_station(station_id)
        
        if self.__station_ids is not None and not station_id in self.__station_ids:
            raise ValueError(f"Station id {station_id} not in list of allowed ids: {self.__station_ids}")

        self.logger.debug("Selected noise event {} ({}, run {}, event {})".format(
            i_noise, noise_station.get_station_time(), noise_event.get_run_number(),
            noise_event.get_id()))
        
        for channel in station.iter_channels():
            channel_id = channel.get_id()

            trace = channel.get_trace()
            noise_channel = noise_station.get_channel(self.__get_noise_channel(channel_id))
            noise_trace = noise_channel.get_trace()

            if len(trace) > 2048:
                self.logger.warn("Simulated trace is longer than 2048 bins... trim with :2048")
                trace = trace[:2048]
            
            # sanity checks
            if len(trace) != len(noise_trace):
                erg_msg = f"Mismatch in trace lenght: Noise has {len(noise_trace)} " + \
                    "and simulation has {len(trace)} samples"
                self.logger.error(erg_msg)
                raise ValueError(erg_msg)

            if channel.get_sampling_rate() != noise_channel.get_sampling_rate():
                erg_msg = "Mismatch in sampling rate: Noise has {} and simulation has {} GHz".format(
                    noise_channel.get_sampling_rate() / units.GHz, channel.get_sampling_rate() / units.GHz)
                self.logger.error(erg_msg)
                raise ValueError(erg_msg)
            
            trace = trace + noise_trace
            channel.set_trace(trace, channel.get_sampling_rate())

    def end(self):
        self._noise_reader.end()
        n_use = np.array(list(self._n_use_event.values()))
        sort = np.flip(np.argsort(n_use))
        self.logger.info("\n\tThe five most used noise events have been used: {}"
                         .format(", ".join([str(ele) for ele in n_use[sort][:5]])))
        pass
