import numpy as np

import logging
import os
import time
import datetime

from NuRadioReco.modules.base.module import register_run

import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import NuRadioReco.framework.trigger

from NuRadioReco.utilities import units

import mattak.Dataset


def baseline_correction_128(wfs):
    
    # Get baseline in chunks of 128 bins
    # np.split -> (16, n_events, n_channels, 128)
    # np.mean -> (16, n_events, n_channels)
    means = np.mean(np.split(wfs, 2048 // 128, axis=-1), axis=-1)
    
    # Get baseline traces
    # np.repeat -> (2048, n_events, n_channels)
    baseline_traces = np.repeat(means, 128 % 2048, axis=0)
    
    # np.moveaxis -> (n_events, n_channels, 2048)
    baseline_traces = np.moveaxis(baseline_traces, 0, -1)
    
    return wfs - baseline_traces


class readRNOGData:

   def begin(self, 
             data_dirs, log_level=logging.INFO, selectors=None, 
             read_calibrated_data=False,
             apply_baseline_correction=True,
             convert_to_voltage=True,
             select_triggers=None):
      
      """
      
      Parameters
      ----------
      
      data_dirs: list of strings / string
         Path to run directories (i.e. ".../stationXX/runXXX/")
         
      log_level: enum
         Set verbosity level of logger
         
      selectors: list of lambdas
         List of lambda(eventInfo) -> bool to pass to mattak.Dataset.iterate to select events.
         Example:
         
         trigger_selector = lambda eventInfo: eventInfo.triggerType == "FORCE"
         
      read_calibrated_data: bool
         If True, read calibrated waveforms from Mattak.Dataset. If False, read "raw" ADC traces.
         (temp. Default: False)
      
      apply_baseline_correction: bool
         Only applies when non-calibrated data are read. If true, correct for DC offset.
         (Default: True)

      convert_to_voltage: bool
         Only applies when non-calibrated data are read. If true, convert ADC to voltage.
         (Default: True)   
      """
      
      t0 = time.time()
      
      self.logger = logging.getLogger('NuRadioReco.readRNOGData')
      self.logger.setLevel(log_level)
      
      self._read_calibrated_data = read_calibrated_data
      self._apply_baseline_correction = apply_baseline_correction
      self._convert_to_voltage = convert_to_voltage
      
      # Temporary solution hard-coded values from Cosmin. Only used when uncalibrated data
      # is read and convert_to_voltage is True.
      self._adc_ref_voltage_range = 2.5 * units.volt
      self._adc_n_bits = 12

      if not isinstance(data_dirs, (list, np.ndarray)):
         data_dirs = [data_dirs]

      if selectors is not None:
         if not isinstance(selectors, (list, np.ndarray)):
            selectors = [selectors]
            
      if select_triggers is not None:
         if isinstance(select_triggers, str):
            selectors.append(lambda eventInfo: eventInfo.triggerType == select_triggers)
         else:
            for select_trigger in select_triggers:
               selectors.append(lambda eventInfo: eventInfo.triggerType == select_trigger)

      self._selectors = selectors
      self.logger.info(f"Found {len(self._selectors)} selectors")
      
      self._time_begin = 0
      self._time_run = 0
      self.__counter = 0
      self.__skipped = 0
      
      self._datasets = []
      self.__n_events_per_dataset = []
      
      self.logger.info(f"Parse through {len(data_dirs)} directories.")
      
      for data_dir in data_dirs:
         
         if not os.path.exists(data_dir):
            self.logger.error(f"The directory {data_dir} does not exist")
      
         dataset = mattak.Dataset.Dataset(station=0, run=0, data_dir=data_dir)
         self._datasets.append(dataset)
         self.__n_events_per_dataset.append(dataset.N())

      # keeps track which event index is in which dataset
      self._event_idxs_datasets = np.cumsum(self.__n_events_per_dataset)
      self._n_events_total = np.sum(self.__n_events_per_dataset)
      
      self._time_begin = time.time() - t0


   def get_n_events_of_prev_datasets(self, dataset_idx):
      dataset_idx_prev = dataset_idx - 1
      return int(self._event_idxs_datasets[dataset_idx_prev]) if dataset_idx_prev >= 0 else 0


   @register_run()
   def run(self):

      for event_idx in range(self._n_events_total):
         self.logger.debug(f"Processing event number {event_idx} out of total {self._n_events_total}")
         t0 = time.time()

         # find correct dataset
         dataset_idx = np.digitize(event_idx, self._event_idxs_datasets)
         dataset = self._datasets[dataset_idx]

         event_idx_in_dataset = event_idx - self.get_n_events_of_prev_datasets(dataset_idx)
         dataset.setEntries(event_idx_in_dataset)  # increment iterator -> point to new event
         
         event_info = dataset.eventInfo()

         skip = False
         if self._selectors is not None:
            for selector in self._selectors:
               if not selector(event_info):
                  skip = True
         
         if skip:
            self.__skipped += 1
            continue
                     
         evt = NuRadioReco.framework.event.Event(event_info.run, event_info.eventNumber)
         station = NuRadioReco.framework.station.Station(event_info.station)
         station.set_station_time(datetime.datetime.fromtimestamp(event_info.readoutTime))

         trigger = NuRadioReco.framework.trigger.Trigger(event_info.triggerType)
         trigger.set_triggered()
         trigger.set_trigger_time(event_info.triggerTime)
         station.set_trigger(trigger)

         waveforms = dataset.wfs()
         
         for channel_id, wf in enumerate(waveforms):
            channel = NuRadioReco.framework.channel.Channel(channel_id) 
            if self._read_calibrated_data:   
               channel.set_trace(wf * units.mV, event_info.sampleRate * units.GHz)
            else:
               # wf stores ADC counts
               
               if self._apply_baseline_correction:
                  # correct baseline
                  wf = baseline_correction_128(wf)
               
               if self._convert_to_voltage:
                  # convert adc to voltage
                  wf *= (self._adc_ref_voltage_range / (2 ** (self._adc_n_bits) - 1))
                
               channel.set_trace(wf, event_info.sampleRate * units.GHz)
                            
            station.add_channel(channel)
         
         evt.set_station(station)
         
         self._time_run += time.time() - t0
         self.__counter += 1
         yield evt


   def end(self):
      self.logger.info(f"Read {self.__counter} events (skipped {self.__skipped} events)"
         f"\n\tTime to initialize data sets  : {self._time_begin:.2f}"
         f"\n\tTime to initialize all events : {self._time_run:.2f}"
         f"\n\tTime to per event             : {self._time_run / self.__counter:.2f}")