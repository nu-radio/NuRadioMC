import numpy as np
import glob
import os
import sys

from NuRadioReco.modules.io.rno_g.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units

from NuRadioReco.modules import channelResampler

import logging


class noiseImporterRNOG:
    """
    Imports recorded traces from RNOG stations. 
    
    """


    def begin(self, noise_folder, station_ids=None,
              channel_mapping=None, log_level=logging.INFO,
              convert_noise_to_voltage=True, 
              match_station_ids=False):
        """
        
        Parameters
        ----------
        noise_folder: string
            the folder containing the noise file or subfolders containing noise files
        
        station_ids: list(int)
            the station ids from which to add noise ()
        
        channel_mapping: dict or None
            option relevant for MC studies of new station designs where we do not
            have forced triggers for. The channel_mapping dictionary maps the channel
            ids of the MC station to the channel ids of the noise data
            Default is None which is 1-to-1 mapping
        
        log_level: loggging log level
            the log level, default logging.INFO
            
        """
        
        self.__channel_mapping = channel_mapping
        self.__station_ids = station_ids
        self._convert_noise_to_voltage = convert_noise_to_voltage
        self._match_station_ids = match_station_ids
        
        self._channel_respampler = channelResampler.channelResampler()

        self.logger = logging.getLogger('noiseImporter')

        self.logger.setLevel(log_level)
        self.__channel_mapping = channel_mapping
        
        noise_files = glob.glob(f"{noise_folder}/**/*root", recursive=True)
        self.__noise_folders = np.unique([os.path.dirname(e) for e in noise_files])
        
        self.logger.info(f"Found {len(self.__noise_folders)} folders in {noise_folder}")
        if not len(self.__noise_folders):
            raise ValueError
        
        noise_reader = readRNOGData()
        selectors = [lambda einfo: einfo.triggerType == "FORCE"]
        noise_reader.begin(self.__noise_folders, selectors=selectors)
        self._noise_events = [evt for evt in noise_reader.run()]
        noise_reader.end()
        
    
    def _buffer_station_id_list(self):
        if self.__station_id_list is None:
            self.__station_id_list = np.squeeze([evt.get_station_ids() for evt in self._noise_events])
        
        return self.__station_id_list
        
        
    def __get_noise_channel(self, channel_id):
        if self.__channel_mapping is None:
            return channel_id
        else:
            return self.__channel_mapping[channel_id]
        

    @register_run()
    def run(self, evt, station, det):

        if self._match_station_ids:
            
            station_ids = self._buffer_station_id_list()
            mask = station_ids == station.get_id()
            if not np.any(mask):
                raise ValueError(f"No station with id {station.get_id()} in noise data.")
            
            i_noise = np.random.choice(np.arange(len(mask))[mask])
                
        else:
            i_noise = np.random.randint(0, len(self._noise_events))
        
        noise_event = self._noise_events[i_noise]
        
        station_id = noise_event.get_station_ids()[0]
        noise_station = noise_event.get_station(station_id)
        
        if self.__station_ids is not None and not station_id in self.__station_ids:
            raise KeyError()

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
        pass
