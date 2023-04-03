import numpy as np

import logging
import os
import time
import astropy.time

from NuRadioReco.modules.base.module import register_run

import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import NuRadioReco.framework.trigger

from NuRadioReco.utilities import units
import mattak.Dataset

try:
   from rnog_data.runtable import RunTable
   import pandas
   imported_runtable = True
except ImportError:
   print("Import of run table failed. You will not be able to select runs! \n" 
         "You can get the interface from GitHub: git@github.com:RNO-G/rnog-data-analysis-and-issues.git")
   imported_runtable = False


def baseline_correction(wfs, n_bins=128):
    
   # Get baseline in chunks of 128 bins
   # np.split -> (16, n_events, n_channels, 128)
   # np.mean -> (16, n_events, n_channels)
   if n_bins is not None:
      medians = np.median(np.split(wfs, 2048 // n_bins, axis=-1), axis=-1)
    
      # Get baseline traces
      # np.repeat -> (2048, n_events, n_channels)
      baseline_traces = np.repeat(medians, n_bins % 2048, axis=0)
   else:
      medians = np.median(wfs, axis=-1)
    
      # Get baseline traces
      # np.repeat -> (2048, n_events, n_channels)
      baseline_traces = np.repeat(medians, 2048, axis=0)
          
   # np.moveaxis -> (n_events, n_channels, 2048)
   baseline_traces = np.moveaxis(baseline_traces, 0, -1)
    
   return wfs - baseline_traces


def get_time_offset(trigger_type):
   """ 
   Mapping the offset between trace start time and trigger time (~ signal time). 
   Temporary use hard-coded values for each trigger type. In the future this
   information might be time, station, and channel dependent and should come 
   from a database (or is already calibrated in mattak)
   
   Parameters
   ----------
   
   trigger_type: str
      Trigger type encoded as string from Mattak
      
   Returns
   -------
   
   time_offset: float
      trace_start_time = trigger_time - time_offset 
   
   """
   
   time_offsets = {
      "FORCE": 0,
      "LT": 213 * units.ns,  # ~ 1 / 3 of trace @ 2048 sample with 3.2 GSa/s
      "RADIANT": 320 * units.ns  # ~ 1 / 2 of trace @ 2048 sample with 3.2 GSa/s
   }
   
   if trigger_type.startswith("RADIANT"):
      trigger_type = "RADIANT"
   
   if trigger_type in time_offsets:
      return time_offsets[trigger_type]
   else:
      raise KeyError(f"Unknown trigger type: {trigger_type}. Known are: FORCE, LT, RADIANT. Abort ....")


class readRNOGData:

   def begin(self, 
             data_dirs, log_level=logging.INFO, selectors=None, 
             read_calibrated_data=False,
             apply_baseline_correction=True,
             convert_to_voltage=True,
             select_triggers=None,
             select_runs=True,
             run_types=["physics"],
             max_trigger_rate=1 * units.Hz):
      
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
         
      select_runs: bool
         Select runs
         
      run_types: list
         Used to select/reject runs from information in the RNO-G RunTable. List of run_types to be used. (Default: ['physics'])
         
      max_trigger_rate: float
         Used to select/reject runs from information in the RNO-G RunTable. Maximum allowed trigger rate (per run) in Hz.
         If 0, no cut is applied. (Default: 1 Hz)
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
      
      if select_runs:
         self.logger.info("\n\tSelect runs with type: {}".format(", ".join(run_types)) +
                         f"\n\tSelect runs with max. trigger rate of {max_trigger_rate / units.Hz} Hz")
      
      self.__max_trigger_rate = max_trigger_rate
      self.__run_types = run_types
      
      global imported_runtable
      if imported_runtable:
         self.logger.debug("Access RunTable database ...")
         try:
            self.__run_table = RunTable().get_table()
         except:
            self.logger.error("No connect to RunTable database could be established. "
                             "Runs will not be filtered.")
         imported_runtable = False

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
      
      self.__skipped_runs = 0
      self.__n_runs = 0
      
      for data_dir in data_dirs:
         
         if not os.path.exists(data_dir):
            self.logger.error(f"The directory {data_dir} does not exist")
      
         dataset = mattak.Dataset.Dataset(station=0, run=0, data_dir=data_dir)
         
         # filter runs/datasets based on 
         if select_runs and imported_runtable and not self.__select_run(dataset):
            self.__skipped_runs += 1
            continue
         
         self.__n_runs += 1
         self._datasets.append(dataset)
         self.__n_events_per_dataset.append(dataset.N())

      # keeps track which event index is in which dataset
      self._event_idxs_datasets = np.cumsum(self.__n_events_per_dataset)
      self._n_events_total = np.sum(self.__n_events_per_dataset)
      self._time_begin = time.time() - t0
      
      if not self._n_events_total:
         err = "No runs have been selected. Abort ..."
         self.logger.error(err)
         raise ValueError(err)

      
   def __select_run(self, dataset):
      """ Filter/select runs/datasets. Return True to select an dataset, return False to skip it """
     
      # get first eventInfo
      dataset.setEntries(0)
      event_info = dataset.eventInfo()
      
      run_id = event_info.run
      station_id = event_info.station
      
      run_info = self.__run_table.query(f"station == {station_id:d} & run == {run_id:d}")
      run_type = run_info["run_type"].values[0]
      
      if not run_type in self.__run_types:
         self.logger.info(f"Reject station {station_id} run {run_id} because of run type {run_type}")
         return False
      
      trigger_rate = run_info["trigger_rate"].values[0] * units.Hz 
      if self.__max_trigger_rate and trigger_rate > self.__max_trigger_rate:
         self.logger.info(f"Reject station {station_id} run {run_id} because trigger rate is to high ({trigger_rate / units.Hz} Hz)")
         return False
      
      return True


   def __get_n_events_of_prev_datasets(self, dataset_idx):
      """ Get number of events from previous dataset to correctly set pointer """
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

         event_idx_in_dataset = event_idx - self.__get_n_events_of_prev_datasets(dataset_idx)
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
         station.set_station_time(astropy.time.Time(event_info.triggerTime, format='unix'))

         trigger = NuRadioReco.framework.trigger.Trigger(event_info.triggerType)
         trigger.set_triggered()
         trigger.set_trigger_time(event_info.triggerTime)
         station.set_trigger(trigger)

         # access data
         waveforms = dataset.wfs()
         
         for channel_id, wf in enumerate(waveforms):
            channel = NuRadioReco.framework.channel.Channel(channel_id) 
            if self._read_calibrated_data:   
               channel.set_trace(wf * units.mV, event_info.sampleRate * units.GHz)
            else:
               # wf stores ADC counts
               
               if self._apply_baseline_correction:
                  # correct baseline
                  wf = baseline_correction(wf)
               
               if self._convert_to_voltage:
                  # convert adc to voltage
                  wf *= (self._adc_ref_voltage_range / (2 ** (self._adc_n_bits) - 1))
                
               channel.set_trace(wf, event_info.sampleRate * units.GHz)
            
            time_offset = get_time_offset(event_info.triggerType)
            channel.set_trace_start_time(-time_offset)  # relative to event/trigger time
                            
            station.add_channel(channel)
         
         evt.set_station(station)
         
         self._time_run += time.time() - t0
         self.__counter += 1
         yield evt


   def end(self):
      self.logger.info(
         f"\n\tRead {self.__counter} events (skipped {self.__skipped} events)"
         f"\n\tTime to initialize data sets  : {self._time_begin:.2f}s"
         f"\n\tTime to initialize all events : {self._time_run:.2f}s"
         f"\n\tTime to per event             : {self._time_run / self.__counter:.2f}s"
         f"\n\tRead {self.__n_runs} runs, skipped {self.__skipped_runs} runs.")