import numpy as np
import pandas
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
   imported_runtable = True
except ImportError:
   print("Import of run table failed. You will not be able to select runs! \n" 
         "You can get the interface from GitHub: git@github.com:RNO-G/rnog-data-analysis-and-issues.git")
   imported_runtable = False


def baseline_correction(wfs, n_bins=128, func=np.median):
   """
   Simple baseline correction function. Determines baseline in discrete chuncks of "n_bins" with
   the function specified (i.e., mean or median).
   
   Parameters
   ----------
   
   wfs: np.array(n_events, n_channels, n_samples)
      Waveforms of several events/channels.
      
   n_bins: int
      Number of samples/bins in one "chunck". If None, calculate median/mean over entire trace. (Default: 128)
      
   func: np.mean or np.median
      Function to calculate pedestal
   
   Returns
   -------
   
   wfs_corrected: np.array(n_events, n_channels, n_samples)
      Baseline/pedestal corrected waveforms
   """
    
   # Example: Get baselines in chunks of 128 bins
   # wfs in (n_events, n_channels, 2048)
   # np.split -> (16, n_events, n_channels, 128) each waveform split in 16 chuncks
   # func -> (16, n_events, n_channels) pedestal for each chunck
   if n_bins is not None:
      baseline_values = func(np.split(wfs, 2048 // n_bins, axis=-1), axis=-1)

      # np.repeat -> (2048, n_events, n_channels) concatenate the 16 chuncks to one baseline
      baseline_traces = np.repeat(baseline_values, n_bins % 2048, axis=0)
   else:
      baseline_values = func(wfs, axis=-1)
      # np.repeat -> (2048, n_events, n_channels) concatenate the 16 chuncks to one baseline
      baseline_traces = np.repeat(baseline_values, 2048, axis=0)
          
   # np.moveaxis -> (n_events, n_channels, 2048)
   baseline_traces = np.moveaxis(baseline_traces, 0, -1)
    
   return wfs - baseline_traces


def get_time_offset(trigger_type):
   """ 
   Mapping the offset between trace start time and trigger time (~ signal time). 
   Temporary use hard-coded values for each trigger type. In the future this
   information might be time, station, and channel dependent and should come 
   from a database (or is already calibrated in mattak)
   
   Current values motivated by figures posted in PR https://github.com/nu-radio/NuRadioMC/pull/519
   
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
      "LT": 250 * units.ns,  
      "RADIANT": 475 * units.ns  
   }
   
   if trigger_type.startswith("RADIANT"):
      trigger_type = "RADIANT"
   
   if trigger_type in time_offsets:
      return time_offsets[trigger_type]
   else:
      raise KeyError(f"Unknown trigger type: {trigger_type}. Known are: FORCE, LT, RADIANT. Abort ....")


class readRNOGData:

   def begin(self, 
             data_dirs,  
             read_calibrated_data=False,
             select_triggers=None,
             select_runs=True,
             apply_baseline_correction=True,
             convert_to_voltage=True,
             selectors=None,
             run_table_path=None,
             run_types=["physics"],
             run_time_range=None,
             max_trigger_rate=0 * units.Hz,
             mattak_backend="auto",
             log_level=logging.INFO):
      """

      Parameters
      ----------

      data_dirs: list of strings / string
         Path to run directories (i.e. ".../stationXX/runXXX/")
         
      read_calibrated_data: bool
         If True, read calibrated waveforms from Mattak.Dataset. If False, read "raw" ADC traces.
         (temp. Default: False)
         
      select_triggers: str or list(str)
         Names of triggers which should be selected. Convinence interface instead of passing a selector
         (see "selectors" below. (Default: None) 
         
      select_runs: bool
         If True, use information in run_table to select runs (based on run_type, run_time, trigger_rate, ...).
         If the run_table is not available no selection is performed (and the programm is not interrupted, 
         only an error message is raised). See parameters to configure run selection. (Default: True)
         
      Other Parameters
      ----------------
      
      apply_baseline_correction: bool
         Only applies when non-calibrated data are read. If true, correct for DC offset.
         (Default: True)

      convert_to_voltage: bool
         Only applies when non-calibrated data are read. If true, convert ADC to voltage.
         (Default: True)

      selectors: list of lambdas
         List of lambda(eventInfo) -> bool to pass to mattak.Dataset.iterate to select events.
         Example: trigger_selector = lambda eventInfo: eventInfo.triggerType == "FORCE"
         
      run_table_path: str
         Path to a run_table.cvs file. If None, the run table is queried from the DB. (Default: None)
         
      run_types: list
         Used to select/reject runs from information in the RNO-G RunTable. List of run_types to be used. (Default: ['physics'])
         
      run_time_range: tuple
         Specify a time range to select runs (it is sufficient that runs cover the time range partially). 
         Each value of the tuple has to be in a format which astropy.time.Time understands. A value can be None 
         which means that the lower or upper bound is unconstrained. If run_time_range is None no time selection is
         applied. (Default: None)
         
      max_trigger_rate: float
         Used to select/reject runs from information in the RNO-G RunTable. Maximum allowed trigger rate (per run) in Hz.
         If 0, no cut is applied. (Default: 1 Hz)
         
      mattak_backend: str
         Select a mattak backend. Options are "auto", "pyroot", "uproot". If "auto" is selected, pyroot is used if available otherwise
         a "fallback" to uproot is used. (Default: "auto") 

      log_level: enum
         Set verbosity level of logger
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
      
      self.__max_trigger_rate = max_trigger_rate
      self.__run_types = run_types
      
      if run_time_range is not None:
         convert_time = lambda t: None if t is None else astropy.time.Time(t)
         self._time_low = convert_time(run_time_range[0])
         self._time_high = convert_time(run_time_range[1])
      else:
         self._time_low = None
         self._time_high = None         
      
      if select_runs:
         if run_table_path is None:
            global imported_runtable
            if imported_runtable:
               self.logger.debug("Access RunTable database ...")
               try:
                  self.__run_table = RunTable().get_table()
               except:
                  self.logger.error("No connect to RunTable database could be established. "
                                    "Runs will not be filtered.")
                  imported_runtable = False
         else:
            self.__run_table = pandas.read_csv(run_table_path)
            imported_runtable = True
            
      if select_runs:
         self.logger.info("\n\tSelect runs with type: {}".format(", ".join(run_types)) +
                         f"\n\tSelect runs with max. trigger rate of {max_trigger_rate / units.Hz} Hz"
                         f"\n\tSelect runs which are between {self._time_low} - {self._time_high}")
      
      if not isinstance(data_dirs, (list, np.ndarray)):
         data_dirs = [data_dirs]

      if selectors is not None:
         if not isinstance(selectors, (list, np.ndarray)):
            selectors = [selectors]
         
         self.logger.info(f"Found {len(selectors)} selector(s)")

      self._selectors = selectors
            
      if select_triggers is not None:
         if isinstance(select_triggers, str):
            selectors.append(lambda eventInfo: eventInfo.triggerType == select_triggers)
         else:
            for select_trigger in select_triggers:
               selectors.append(lambda eventInfo: eventInfo.triggerType == select_trigger)
      
      self._time_begin = 0
      self._time_run = 0
      self.__counter = 0
      self.__skipped = 0
      
      self._event_informations = None
      self._datasets = []
      self.__n_events_per_dataset = []
      
      self.logger.info(f"Parse through {len(data_dirs)} directory/ies.")
      
      self.__skipped_runs = 0
      self.__n_runs = 0
      
      for data_dir in data_dirs:
         
         if not os.path.exists(data_dir):
            self.logger.error(f"The directory {data_dir} does not exist")
      
         dataset = mattak.Dataset.Dataset(station=0, run=0, data_dir=data_dir, backend=mattak_backend)

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
      
      self.logger.info(f"{self._n_events_total} events in {len(self._datasets)} runs/datasets have been found.")
      
      # Variable not yet implemented in mattak
      # self.logger.info(f"Using the {self._datasets[0].backend} Mattak backend.")
      
      if not self._n_events_total:
         err = "No runs have been selected. Abort ..."
         self.logger.error(err)
         raise ValueError(err)

      
   def __select_run(self, dataset):
      """ Filter/select runs/datasets. 
      
      Parameters
      ----------
      
      dataset: mattak.Dataset.Dataset
      
      select: bool
         Return True to select an dataset, return False to reject/skip it.
      """
     
      # get first eventInfo
      dataset.setEntries(0)
      event_info = dataset.eventInfo()
      
      run_id = event_info.run
      station_id = event_info.station
      
      run_info = self.__run_table.query(f"station == {station_id:d} & run == {run_id:d}")
      
      # "time_start/end" is stored in the isot format. datetime is much faster than astropy (~85ns vs 55 mus).
      # But using datetime would mean to stip decimals because datetime can only handle mu sec precision and can not cope 
      # with the additional decimals for ns.
      if self._time_low is not None:
         time_end = astropy.time.Time(run_info["time_end"].values[0])
         if time_end < self._time_low:
            self.logger.info(f"Reject station {station_id} run {run_id} because run ended before {self._time_low}")
            return False
      
      if self._time_high is not None:
         time_start = astropy.time.Time(run_info["time_start"].values[0])
         if time_start > self._time_high:
            self.logger.info(f"Reject station {station_id} run {run_id} because run started time after {self._time_high}")
            return False
      
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
      """ Get accumulated number of events from previous datasets """
      dataset_idx_prev = dataset_idx - 1
      return int(self._event_idxs_datasets[dataset_idx_prev]) if dataset_idx_prev >= 0 else 0
      
   
   def __get_dataset_for_event(self, event_idx):
      """ Get correct dataset and set entry accordingly to event index
      
      Parameters
      ----------
      
      event_index: int
         Same as in read_event().
      
      Returns
      -------
      
      dataset: mattak.Dataset.Dataset
      """
      # find correct dataset
      dataset_idx = np.digitize(event_idx, self._event_idxs_datasets)
      dataset = self._datasets[dataset_idx]

      event_idx_in_dataset = event_idx - self.__get_n_events_of_prev_datasets(dataset_idx)
      dataset.setEntries(event_idx_in_dataset)  # increment iterator -> point to new event
            
      return dataset
   
   
   def filter_event(self, evtinfo, event_idx=None):
      """ Filter an event base on its EventInfo and the configured selectors.

      Parameters
      ----------
      
      event_info: mattak.Dataset.EventInfo
         The event info object for one event.
      
      event_index: int
         Same as in read_event(). Only use for logger.info(). (Default: None)
         
      Returns
      -------
      
      skip: bool
         Returns True to skip/reject event, return False to keep/read event
      """
      if self._selectors is not None:
         for selector in self._selectors:
            if not selector(evtinfo):
               self.logger.debug(f"Event {event_idx} (station {evtinfo.station}, run {evtinfo.run}, "
                                 f"event number {evtinfo.eventNumber}) is skipped.")
               self.__skipped += 1
               return True
      
      return False
   
   
   def get_event_informations(self, keys=["station", "run", "eventNumber"]):
      """ Return information of all events from the EventInfo
      
      This function is useful to make a pre-selection of events before actually reading them in combination with 
      self.read_event().
      
      Parameters
      ----------
      
      keys: list(str)
         List of the information to receive from each event. Have to match the attributes (member variables)
         of the mattak.Dataset.EventInfo class (examples are "station", "run", "triggerTime", "triggerType", "eventNumber", ...).
         (Default: ["station", "run", "eventNumber"])
         
      Returns
      -------
      
      data: dict
         Keys of the dict are the event indecies (as used in self.read_event(event_index)). The values are dictinaries 
         them self containing the information specified with "keys" parameter.
      """
      
      # Read if dict is None ...
      do_read = self._event_informations is None
      
      if not do_read:
         # ... or when it does not have the desired information
         first_event_info = next(iter(self._event_informations))
         print(first_event_info)
         for key in keys:
            if key not in list(first_event_info.keys()):
               do_read = True
      
      if do_read:
      
         self._event_informations = {}
         n_prev = 0
         for dataset in self._datasets:
            dataset.setEntries((0, dataset.N()))
            
            for idx, evtinfo in enumerate(dataset.eventInfo()):  # returns a list
         
               event_idx = idx + n_prev  # event index accross all datasets combined 
            
               if self.filter_event(evtinfo, event_idx):
                  continue
            
               self._event_informations[event_idx] = {key: getattr(evtinfo, key) for key in keys}
            
            n_prev += dataset.N()

      return self._event_informations
   
   
   def _get_event(self, event_info, waveforms):
      """ Return a NuRadioReco event
      
      Parameters
      ----------
      
      event_info: mattak.Dataset.EventInfo
         The event info object for one event.
         
      waveforms: np.array(n_channel, n_samples)
         Typically what dataset.wfs() returns (for one event!)

      Returns
      -------
      
      evt: NuRadioReco.framework.event
      """
      
      evt = NuRadioReco.framework.event.Event(event_info.run, event_info.eventNumber)
      station = NuRadioReco.framework.station.Station(event_info.station)
      station.set_station_time(astropy.time.Time(event_info.triggerTime, format='unix'))

      trigger = NuRadioReco.framework.trigger.Trigger(event_info.triggerType)
      trigger.set_triggered()
      trigger.set_trigger_time(event_info.triggerTime)
      station.set_trigger(trigger)
      
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
      
      return evt     


   @register_run()
   def run(self):
      """ 
      Loop over all events.
      
      Returns
      -------
      
      evt: generator(NuRadioReco.framework.event)
      """
      event_idx = -1
      for dataset in self._datasets:
         dataset.setEntries((0, dataset.N()))
                  
         # read all event infos of the entier dataset (= run)
         event_infos = dataset.eventInfo()
         wfs = None

         for idx, evtinfo in enumerate(event_infos):  # returns a list
            event_idx += 1
            
            self.logger.debug(f"Processing event number {event_idx} out of total {self._n_events_total}")
            t0 = time.time()
                  
            if self.filter_event(evtinfo, event_idx):
               continue
            
            # Just read wfs if necessary
            if wfs is None:
               wfs = dataset.wfs()
               
            waveforms_of_event = wfs[idx]
            
            evt = self._get_event(evtinfo, waveforms_of_event)
            
            self._time_run += time.time() - t0
            self.__counter += 1
                           
            yield evt



   def read_event(self, event_index):
      """ Allows to read a specific event identifed by its index 
      
      Parameters
      ----------
      
      event_index: int
         The index of a particluar event. The index is the chronological number from 0 to 
         number of total events (across all datasets).
         
      Returns
      -------
      
      evt: NuRadioReco.framework.event
      """
      
      self.logger.debug(f"Processing event number {event_index} out of total {self._n_events_total}")
      t0 = time.time()

      dataset = self.__get_dataset_for_event(event_index)
      event_info = dataset.eventInfo()  # returns a single eventInfo

      if self.filter_event(event_info, event_index):
         return None
            
      # access data
      waveforms = dataset.wfs()
      
      evt = self._get_event(event_info, waveforms)
      
      self._time_run += time.time() - t0
      self.__counter += 1
   
      return evt
   
      
   def get_event(self, event_id):
      """ Allows to read a specific event identifed by its id

      Parameters
      ----------

      event_id: int
         Event Id
         
      Returns
      -------

      evt: NuRadioReco.framework.event
      """

      self.logger.debug(f"Processing event {event_id}")
      t0 = time.time()

      event_infos = self.get_event_informations(keys=["eventNumber"])
      event_idx_ids = np.array([[index, ele["eventNumber"]] for index, ele in event_infos.items()])
      mask = event_idx_ids[:, 1] == event_id

      if not np.any(mask):
         self.logger.info(f"Could not find event with id: {event_id}.")
         return None
      elif np.sum(mask) > 1:
         self.logger.error(f"Found several events with the same id: {event_id}.")
         raise ValueError(f"Found several events with the same id: {event_id}.")
      else:
         pass

      event_index = event_idx_ids[mask, 0][0]

      dataset = self.__get_dataset_for_event(event_index)
      event_info = dataset.eventInfo()  # returns a single eventInfo

      if self.filter_event(event_info, event_index):
         return None
            
      # access data
      waveforms = dataset.wfs()

      evt = self._get_event(event_info, waveforms)

      self._time_run += time.time() - t0
      self.__counter += 1

      return evt


   def end(self):
      self.logger.info(
         f"\n\tRead {self.__counter} events (skipped {self.__skipped} events)"
         f"\n\tTime to initialize data sets  : {self._time_begin:.2f}s"
         f"\n\tTime to initialize all events : {self._time_run:.2f}s"
         f"\n\tTime to per event             : {self._time_run / self.__counter:.2f}s"
         f"\n\tRead {self.__n_runs} runs, skipped {self.__skipped_runs} runs.")
