import numpy as np
import logging
import os
import time
import astropy.time
import math
from functools import lru_cache
from inspect import signature

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.RNO_G.channelBlockOffsetFitter import \
    channelBlockOffsets, fit_block_offsets, _calculate_block_offsets

import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import NuRadioReco.framework.trigger
from NuRadioReco.framework.parameters import channelParameters

from NuRadioReco.utilities import units

logger = logging.getLogger('NuRadioReco.RNOG.readRNOGData')

try:
    import mattak.Dataset
except ImportError as e:
    logger.error(
        "To use the readRNOGDataMattak module, 'mattak' needs to be installed:\n\n"
        "\t pip install git+https://github.com/RNO-G/mattak"
        )
    raise(e)


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
        "RADIANT": 475 * units.ns,
        "UNKNOWN": 0  # Due to a firmware issue at the beginning of data taking the trigger types were not properly set.
    }

    # Should have the same time offset ?!
    if trigger_type.startswith("RADIANT"):
        trigger_type = "RADIANT"

    if trigger_type in time_offsets:
        return time_offsets[trigger_type]
    else:
        known_trigger_types = ", ".join(time_offsets.keys())
        raise KeyError(f"Unknown trigger type: {trigger_type}. Known are: {known_trigger_types}. Abort ....")


def _all_files_in_directory(mattak_dir):
    """
    Checks if all Mattak root files are in a directory.
    Ignoring runinfo.root because (asaik) not all runs have those and information is currently not read by Mattak.
    There are mattak directories which produce a ReferenceError when reading. They have a "combined.root" which is
    apparently empty but are missing the daqstatus, pedestal, and header file.

    Parameters
    ----------
    mattak_dir: str
        Path to a mattak directory

    Returns
    -------
    all_there: bool
        True, if all "req_files" are there and waveforms.root or combined.root. Otherwise returns False.
    """
    # one or the other has to be present
    if not os.path.exists(os.path.join(mattak_dir, "waveforms.root")) and \
            not os.path.exists(os.path.join(mattak_dir, "combined.root")):
        return False

    req_files = ["daqstatus.root", "headers.root", "pedestal.root"]
    for file in req_files:
        if not os.path.exists(os.path.join(mattak_dir, file)):
            logging.error(f"File {file} could not be found in {mattak_dir}")
            return False

    return True


def _convert_to_astropy_time(t):
    """ Convert to astropy.time.Time """
    return None if t is None else astropy.time.Time(t)


class readRNOGData:

    def __init__(self, run_table_path=None, load_run_table=False, log_level=logging.NOTSET):
        """
        Reader for RNO-G ``.root`` files

        This class provides read access to RNO-G ``.root`` files and converts them
        to NuRadioMC :class:`Events <NuRadioReco.framework.event.Event>`. Requires ``mattak``
        (https://github.com/RNO-G/mattak) to be installed.

        Parameters
        ----------
        run_table_path: str | None
            Path to a run_table.csv file. If None, the run table is queried from the DB. (Default: None)

        load_run_table: bool
            If True, try to load the run_table from run_table_path. Otherwise, skip this. (Default: False)

        log_level: enum
            Set verbosity level of logger. If logging.DEBUG, set mattak to verbose (unless specified in mattak_kwargs).
            (Default: logging.NOTSET, ie adhere to general log level)

        Examples
        --------
        .. code-block::

            reader = readRNOGDataMattak.readRNOGData() # initialize reader
            reader.begin('/path/to/root_file_or_folder')

            evt = reader.get_event_by_index(0) # returns the first event in the file
            # OR
            evt = reader.get_event(run_nr=1100, event_id=679) # returns the event with run_number 1100 and event_id 679
            # OR
            for evt in reader.run(): # loop over all events in file
                # perform some analysis
                pass
        """
        self.logger = logging.getLogger('NuRadioReco.RNOG.readRNOGData')
        self.logger.setLevel(log_level)

        self._blockoffsetfitter = channelBlockOffsets()

        # Initialize run table for run selection
        self.__run_table = None

        if load_run_table:
            if run_table_path is None:
                try:
                    from rnog_data.runtable import RunTable
                    self.logger.debug("Access RunTable database ...")
                    try:
                        self.__run_table = RunTable().get_table()
                    except:
                        self.logger.warn("No connect to RunTable database could be established. "
                                        "Runs can not be filtered.")
                except ImportError:
                    self.logger.warn(
                        "\nImport `from rnog_data.runtable import RunTable` failed. You can still use readRNOGData, "
                        "but runs can not be filtered. To install the run table, run:\n\n"
                        "\tpip install git+ssh://git@github.com/RNO-G/rnog-runtable.git\n"
                    )
            else:
                # some users may mistakenly try to pass the .root files to __init__
                # we check for this and raise a (hopefully) helpful error message
                user_passed_root_file_msg = (
                    "The optional argument run_table_path expects a csv file, "
                    "but you passed a list of files or a .root file. Note that "
                    "the .root files to read in should be passed to the `begin` method of this class"
                )
                if isinstance(run_table_path, (list, np.ndarray)):
                    raise TypeError(user_passed_root_file_msg)
                elif os.path.isdir(run_table_path) or run_table_path.endswith('.root'):
                    raise ValueError(user_passed_root_file_msg)

                import pandas
                self.__run_table = pandas.read_csv(run_table_path)


    def begin(self,
            dirs_files,
            read_calibrated_data=False,
            select_triggers=None,
            select_runs=False,
            apply_baseline_correction='auto',
            convert_to_voltage=True,
            selectors=[],
            run_types=["physics"],
            run_time_range=None,
            max_trigger_rate=0 * units.Hz,
            mattak_kwargs={},
            overwrite_sampling_rate=None,
            max_in_mem=256,
            use_fallback_time=True):
        """
        Parameters
        ----------
        dirs_files: str, list(str)
            Path to run directories (i.e. ".../stationXX/runXXX/") or path to root files (have to be "combined" mattak files).

        read_calibrated_data: bool
            If True, read calibrated waveforms from Mattak.Dataset. If False, read "raw" ADC traces.
            (temp. Default: False, this can/should be switched once the calibration in incorp. into Mattak)

        select_triggers: str or list(str)
            Names of triggers which should be selected. Convinence interface instead of passing a selector
            (see "selectors" below). (Default: None)

        select_runs: bool
            If True, use information in run_table to select runs (based on run_type, run_time, trigger_rate, ...).
            If the run_table is not available no selection is performed (and the programm is not interrupted,
            only an error message is raised). See parameters to configure run selection. (Default: False)

        Other Parameters
        ----------------
        apply_baseline_correction: str {'auto', 'fit', 'approximate', 'median', 'none'}, optional
            Removes the DC (baseline) block offsets (pedestals).
            Options are, in order of decreasing precision and increasing performance:

            * 'fit' : do a full out-of-band fit to determine the block offsets; for more details,
              see :mod:`NuRadioReco.modules.RNO_G.channelBlockOffsetFitter` (slow)
            * 'approximate' : estimate block offsets by looking at the low-pass filtered trace
            * 'median' : subtract the median of each block (faster)
            * 'none' : do not apply a baseline correction (fastest)

            The default ('auto') first performs the 'approximate' block offset removal, then
            automatically decides whether to continue with the full 'fit' depending on the estimated
            block offset size.

        convert_to_voltage: bool
            Only applies when non-calibrated data are read. If true, convert ADC to voltage.
            (Default: True)

        selectors: list of lambdas
            List of lambda(eventInfo) -> bool to pass to mattak.Dataset.iterate to select events.
            Example: trigger_selector = lambda eventInfo: eventInfo.triggerType == "FORCE"
            (Default: [])

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

        mattak_kwargs: dict
            Dictionary of arguments for mattak.Dataset.Dataset. (Default: {})
            Example: Select a mattak "backend". Options are "auto", "pyroot", "uproot". If "auto" is selected,
            pyroot is used if available otherwise a "fallback" to uproot is used. (Default: "auto")

        overwrite_sampling_rate: float
            Set sampling rate of the imported waveforms. This overwrites what is read out from runinfo
            (i.e., stored in the mattak files) only when the stored sampling rate is invalid (i.e. 0 or None).
            If None, nothing is overwritten and the sampling rate from the mattak file is used. (Default: None)
            NOTE: This option might be necessary when old mattak files are read which have this not set.

        max_in_mem: int
            Set the maximum number of events that can be stored in memory. The datareader will divide
            the data in batches based on this number.
            NOTE: This is only relevant for the mattak uproot backend
        use_fallback_time: bool
            If True and if the trigger time is infinity, the readout time will be used instead of the trigger time.
            Otherwise, the event will be skipped in the case of infinite trigger times.
            (Default=True)
        """
        t0 = time.time()

        self._read_calibrated_data = read_calibrated_data

        baseline_correction_valid_options = ['auto', 'approximate', 'fit', 'median', 'none']
        if apply_baseline_correction is None:
            apply_baseline_correction = 'none'

        if apply_baseline_correction.lower() not in baseline_correction_valid_options:
            raise ValueError(
                f"Value for apply_baseline_correction ({apply_baseline_correction}) not recognized. "
                f"Valid options are {baseline_correction_valid_options}"
            )
        self._apply_baseline_correction = apply_baseline_correction.lower()
        self._convert_to_voltage = convert_to_voltage

        # Temporary solution hard-coded values from Cosmin. Only used when uncalibrated data
        # is read and convert_to_voltage is True.
        self._adc_ref_voltage_range = 2.5 * units.volt
        self._adc_n_bits = 12

        self._overwrite_sampling_rate = overwrite_sampling_rate

        # set max wavform array size that can be loaded in memory
        self._max_in_mem = max_in_mem

        self._use_fallback_time = use_fallback_time

        # Set parameter for run selection
        self.__max_trigger_rate = max_trigger_rate
        self.__run_types = run_types

        if run_time_range is not None:
            self._time_low = _convert_to_astropy_time(run_time_range[0])
            self._time_high = _convert_to_astropy_time(run_time_range[1])
        else:
            self._time_low = None
            self._time_high = None

        if select_runs and self.__run_table is not None:
            self.logger.info("\n\tSelect runs with type: {}".format(", ".join(run_types)) +
                            f"\n\tSelect runs with max. trigger rate of {max_trigger_rate / units.Hz} Hz"
                            f"\n\tSelect runs which are between {self._time_low} - {self._time_high}")

        self._selectors = []
        self.add_selectors(self._check_for_valid_information_in_event_info)
        self.add_selectors(selectors, select_triggers)

        # Read data
        self._time_begin = 0
        self._time_run = 0
        self._event_idx = -1 # only for logging
        self.__counter = 0
        self.__skipped = 0
        self.__invalid = 0

        self._events_information = None
        self._datasets = []
        self.__n_events_per_dataset = []

        if not isinstance(dirs_files, (list, np.ndarray)):
            dirs_files = [dirs_files]

        self.logger.info(f"Parse through / read-in {len(dirs_files)} directory(ies) / file(s).")

        self.__skipped_runs = 0
        self.__n_runs = 0

        # Set verbose for mattak
        verbose = mattak_kwargs.pop("verbose", self.logger.level <= logging.DEBUG)

        for dir_file in dirs_files:

            if not os.path.exists(dir_file):
                self.logger.error(f"The directory/file {dir_file} does not exist")
                continue

            if os.path.isdir(dir_file):

                if not _all_files_in_directory(dir_file):
                    self.logger.error(f"Incomplete directory: {dir_file}. Skip ...")
                    continue

            try:
                dataset = mattak.Dataset.Dataset(station=0, run=0, data_path=dir_file, verbose=verbose, **mattak_kwargs)
            except (ReferenceError, KeyError) as e:
                self.logger.error(f"The following exeption was raised reading in the run: {dir_file}. Skip that run ...:\n", exc_info=e)
                continue

            # filter runs/datasets based on
            if select_runs and self.__run_table is not None and not self.__select_run(dataset):
                self.__skipped_runs += 1
                continue

            self.__n_runs += 1
            self._datasets.append(dataset)
            self.__n_events_per_dataset.append(dataset.N())

        if not len(self._datasets):
            err = "Found no valid datasets. Stop!"
            self.logger.error(err)
            raise FileNotFoundError(err)

        # keeps track which event index is in which dataset
        self._event_idxs_datasets = np.cumsum(self.__n_events_per_dataset)
        self._n_events_total = np.sum(self.__n_events_per_dataset)
        self._time_begin = time.time() - t0

        self.logger.info(f"{self._n_events_total} events in {len(self._datasets)} runs/datasets "
                         f"have been found using the {self._datasets[0].backend} Mattak backend.")

        if not self._n_events_total:
            err = "No runs have been selected. Abort ..."
            self.logger.error(err)
            raise ValueError(err)


    def add_selectors(self, selectors, select_triggers=None):
        """
        Add selectors (Callable(eventInfo) -> bool) to an internal list of selectors.
        They are used for event filtering.

        Parameters
        ----------
        selectors: list of Callables
            List of Callable(eventInfo) -> bool to pass to mattak.Dataset.iterate to select events.
            Example: trigger_selector = lambda eventInfo: eventInfo.triggerType == "FORCE"

        select_triggers: str or list(str) (Default: None)
            Names of triggers which should be selected. Convenience interface instead of passing a selector.
        """

        # Initialize selectors for event filtering
        if selectors is None:
            selectors = []

        if not isinstance(selectors, (list, np.ndarray)):
            selectors = [selectors]

        if select_triggers is not None:
            if isinstance(select_triggers, str):
                selectors.append(lambda eventInfo: eventInfo.triggerType == select_triggers)
            else:
                for select_trigger in select_triggers:
                    selectors.append(lambda eventInfo: eventInfo.triggerType == select_trigger)

        self.logger.info(f"Add {len(selectors)} selector(s)")
        self._selectors += selectors
        self.get_waveforms.cache_clear() # reset cached waveforms


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

        if not len(run_info):
            self.logger.error(f"Run {run_id:d} (station {station_id:d}) not in run table. Reject...")
            return False

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
        if run_type not in self.__run_types:
            self.logger.info(f"Reject station {station_id} run {run_id} because of run type {run_type}")
            return False

        trigger_rate = run_info["trigger_rate"].values[0] * units.Hz
        if self.__max_trigger_rate and trigger_rate > self.__max_trigger_rate:
            self.logger.info(f"Reject station {station_id} run {run_id} because trigger rate is to high ({trigger_rate / units.Hz:.2f} Hz)")
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


    def _select_events(self, evtinfo):
        """ Filter an event base on its EventInfo and the configured selectors.

        Parameters
        ----------
        event_info: mattak.Dataset.EventInfo
            The event info object for one event.

        event_index: int (Default: None)
            Same as in read_event(). Only use for logger.info().
        Returns
        -------
        skip: bool
            Returns False to skip/reject event, return True to keep/read event
        """
        self.logger.debug(
            f"(_select_events) Processing event number {self.__counter} out of total {self._n_events_total}")

        self.__counter += 1  # for logging
        if self._selectors is not None:
            for selector in self._selectors:
                if not selector(evtinfo):
                    self.logger.debug(f"Event {self.__counter - 1} (station {evtinfo.station}, run {evtinfo.run}, "
                                      f"event number {evtinfo.eventNumber}) did not pass a filter. Skip it ...")
                    self.__skipped += 1
                    return False

        return True


    def get_events_information(self, keys=["station", "run", "eventNumber"]):
        """ Return information of events from the EventInfo. Only information of events passing the
        selectors, which may have been specified, are returned.

        This function is useful to make a pre-selection of events before actually reading them in combination with
        self.read_event().

        Parameters
        ----------
        keys : str or list(str) or None (Default: ["station", "run", "eventNumber"])
            List of the information to receive from each event. Have to match the attributes (member variables)
            of the mattak.Dataset.EventInfo class (examples are "station", "run", "triggerTime", "triggerType", "eventNumber", ...).
            If None, read in all keys present in the EventInfo class.

        Returns
        -------
        data: dict
            Keys of the dict are the event indecies (as used in self.read_event(event_index)). The values are dictinaries
            them self containing the information specified with "keys" parameter.
        """

        if keys is None:
            keys = [k for k in signature(mattak.Dataset.EventInfo).parameters]

        if isinstance(keys, str):
            keys = [keys]

        # Read if dict is None ...
        do_read = self._events_information is None

        if not do_read:
            # ... or when it does not have the desired information
            first_event_info = self._events_information[list(self._events_information.keys())[0]]

            for key in keys:
                if key not in list(first_event_info.keys()):
                    do_read = True

        if do_read:

            self._events_information = {}
            n_prev = 0
            for dataset in self._datasets:
                dataset.setEntries((0, dataset.N()))

                for idx, evtinfo in enumerate(dataset.eventInfo()):  # returns a list

                    event_idx = idx + n_prev  # event index accross all datasets combined
                    self._event_idx = event_idx
                    if not self._select_events(evtinfo):
                        continue

                    self._events_information[event_idx] = {key: getattr(evtinfo, key) for key in keys}

                n_prev += dataset.N()

        return self._events_information

    @lru_cache(maxsize=1)
    def get_waveforms(self, apply_baseline_correction=None, max_events=1000):
        """ Return waveforms of events passing the selectors which may have been specified

        Parameters
        ----------
        apply_baseline_correction: str | None
            If not None, apply a different baseline correction algorithm than specified in the
            `begin` method. Otherwise (default), use the same setting as specified there.

        max_events : int | None (default: 1000)
            The maximum number of waveforms to return.
            If None, return all waveforms in all datasets. Note that this may cause a crash
            due to memory overflow if too many waveforms are selected.

        Returns
        -------
        wfs: np.array
            Waveforms of all "selected" events. The wavefroms are either calibrated or not
            based on the class config.
        """

        if apply_baseline_correction is None:
            apply_baseline_correction = self._apply_baseline_correction

        events_waveforms = []

        for dataset in self._datasets:
            dataset.setEntries((0, dataset.N()))
            if apply_baseline_correction in ['auto', 'fit', 'approximate']: # we need the sampling rate
                try:
                    sampling_rate = dataset.eventInfo()[0].sampleRate
                except AttributeError:
                    sampling_rate = None
                if not sampling_rate: # invalid sampling rate - overwrite
                    sampling_rate = self._overwrite_sampling_rate

            for idx, (_, wfs) in enumerate(dataset.iterate(
                calibrated=self._read_calibrated_data,
                selectors=self._select_events)):

                if self._read_calibrated_data:
                    wfs = wfs * units.V
                else:
                    # wf stores ADC counts
                    if self._convert_to_voltage:
                        # convert adc to voltage
                        wfs = wfs * (self._adc_ref_voltage_range / (2 ** (self._adc_n_bits) - 1))

                if apply_baseline_correction == 'median':
                    wfs = _calculate_block_offsets(wfs)
                elif apply_baseline_correction in ['auto', 'fit', 'approximate']:
                    wfs = np.vstack([
                        fit_block_offsets(
                            wf, mode=self._apply_baseline_correction,
                            sampling_rate=sampling_rate, return_trace=True)[1]
                        for wf in wfs])

                events_waveforms.append(wfs)
                if (max_events is not None) and (len(events_waveforms) >= max_events):
                    self.logger.warning(
                        f"Number of waveforms {len(events_waveforms)} exceeds max_events. Returning first {max_events} waveforms only."
                        )
                    return np.array(events_waveforms)

        return np.array(events_waveforms)


    def _check_for_valid_information_in_event_info(self, event_info):
        """
        Checks if certain information (sampling rate, trigger time) in mattak.Dataset.EventInfo are valid

        Parameters
        ----------
        event_info: mattak.Dataset.EventInfo

        Returns
        -------
        is_valid: bool
            Returns True if all information valid, false otherwise
        """

        if math.isinf(event_info.triggerTime) and not self._use_fallback_time:
            self.logger.error(f"Event {event_info.eventNumber} (st {event_info.station}, run {event_info.run}) "
                              "has inf trigger time. Event is skipped ..."
                              f"You can avoid this by setting 'use_fallback_time' in the begin() method.")
            self.__invalid += 1
            return False


        if (event_info.sampleRate == 0 or event_info.sampleRate is None) and self._overwrite_sampling_rate is None:
            self.logger.error(f"Event {event_info.eventNumber} (st {event_info.station}, run {event_info.run}) "
                              f"has a sampling rate of {event_info.sampleRate} GHz. Event is skipped ... "
                              f"You can avoid this by setting 'overwrite_sampling_rate' in the begin() method.")
            self.__invalid += 1
            return False

        return True


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
        # use the readout time if the trigger time is infinity
        if self._use_fallback_time and math.isinf(event_info.triggerTime):
            self.logger.warning(f"Event {event_info.eventNumber} (st {event_info.station}, run {event_info.run}) "
                                 "has inf trigger time, readout time will be used instead.")
            trigger_time = event_info.readoutTime
        else:
            trigger_time = event_info.triggerTime

        # only overwrite sampling rate if the stored value is invalid
        if self._overwrite_sampling_rate is not None and event_info.sampleRate in [0, None]:
            sampling_rate = self._overwrite_sampling_rate
        else:
            sampling_rate = event_info.sampleRate

        evt = NuRadioReco.framework.event.Event(event_info.run, event_info.eventNumber)
        station = NuRadioReco.framework.station.Station(event_info.station)
        station.set_station_time(astropy.time.Time(trigger_time, format='unix'))

        trigger = NuRadioReco.framework.trigger.Trigger(event_info.triggerType)
        trigger.set_triggered()
        trigger.set_trigger_time(0)  # The trigger time is relative to the event/station time
        station.set_trigger(trigger)

        readout_delays = event_info.readoutDelay
        for channel_id, wf in enumerate(waveforms):
            channel = NuRadioReco.framework.channel.Channel(channel_id)

            if self._read_calibrated_data:
                channel.set_trace(wf * units.V, sampling_rate * units.GHz)
            else:
                # wf stores ADC counts
                if self._convert_to_voltage:
                    # convert adc to voltage
                    wf = wf * (self._adc_ref_voltage_range / (2 ** (self._adc_n_bits) - 1))

                channel.set_trace(wf, sampling_rate * units.GHz)

            time_offset = get_time_offset(event_info.triggerType) + readout_delays[channel_id]
            channel.set_trace_start_time(-time_offset)  # relative to event/trigger time

            station.add_channel(channel)

        evt.set_station(station)
        if self._apply_baseline_correction in ['auto', 'fit', 'approximate', 'median']:
            self._blockoffsetfitter.remove_offsets(evt, station, mode=self._apply_baseline_correction)

        return evt


    @register_run()
    def run(self):
        """
        Loop over all events.

        Yields
        ------
        evt: `NuRadioReco.framework.event.Event`
        """

        for dataset in self._datasets:
            dataset.setEntries((0, dataset.N()))

            # read all event infos of the entire dataset (= run)
            for evtinfo, wf in dataset.iterate(
                    calibrated=self._read_calibrated_data, selectors=self._select_events,
                    max_entries_in_mem=self._max_in_mem):

                t0 = time.time()
                evt = self._get_event(evtinfo, wf)
                self._time_run += time.time() - t0
                yield evt


    def get_event_by_index(self, event_index):
        """ Allows to read a specific event identifed by its index

        Parameters
        ----------
        event_index: int
            The index of a particluar event. The index is the chronological number from 0 to
            number of total events (across all datasets).

        Returns
        -------
        evt: `NuRadioReco.framework.event.Event`
        """

        self.logger.debug(f"Processing event number {event_index} out of total {self._n_events_total}")
        t0 = time.time()

        dataset = self.__get_dataset_for_event(event_index)
        event_info = dataset.eventInfo()  # returns a single eventInfo

        if not self._select_events(event_info):
            return None

        # access data
        waveforms = dataset.wfs()

        evt = self._get_event(event_info, waveforms)

        self._time_run += time.time() - t0
        self.__counter += 1

        return evt


    def get_event(self, run_nr, event_id):
        """ Allows to read a specific event identifed by run number and event id

        Parameters
        ----------
        run_nr: int
            Run number

        event_id: int
            Event Id

        Returns
        -------
        evt: `NuRadioReco.framework.event.Event`
        """

        self.logger.debug(f"Getting event {event_id}")
        t0 = time.time()

        event_infos = self.get_events_information(keys=["eventNumber", "run"])
        event_idx_ids = np.array([[index, ele["eventNumber"], ele["run"]] for index, ele in event_infos.items()])
        mask = np.all([event_idx_ids[:, 1] == event_id, event_idx_ids[:, 2] == run_nr], axis=0)

        if not np.any(mask):
            self.logger.info(f"Could not find event with id: {event_id}.")
            return None
        elif np.sum(mask) > 1:
            self.logger.error(f"Found several events with the same id: {event_id}.")
            raise ValueError(f"Found several events with the same id: {event_id}.")
        else:
            pass

        # int(...) necessary to pass it to mattak
        event_index = int(event_idx_ids[mask, 0][0])
        dataset = self.__get_dataset_for_event(event_index)
        event_info = dataset.eventInfo()  # returns a single eventInfo

        self._event_idx = event_index
        if not self._select_events(event_info):
            return None

        # access data
        waveforms = dataset.wfs()

        evt = self._get_event(event_info, waveforms)

        self._time_run += time.time() - t0
        self.__counter += 1

        return evt

    def end(self):
        if self.__counter:
            self.logger.info(
                f"\n\tRead {self.__counter} events ({self.__skipped} events are skipped (filtered), {self.__invalid} invalid events)"
                f"\n\tTime to initialize data sets  : {self._time_begin:.2f}s"
                f"\n\tTime to read all events       : {self._time_run:.4f}s"
                f"\n\tTime to per event             : {self._time_run / self.__counter:.4f}s"
                f"\n\tRead {self.__n_runs} runs, skipped {self.__skipped_runs} runs.")
        else:
            self.logger.warning(
                f"\n\tRead {self.__counter} events   (skipped {self.__skipped} events, {self.__invalid} invalid events)"
                f"\n\tTime to initialize data sets  : {self._time_begin:.2f}s"
                f"\n\tTime to read all events       : {self._time_run:.2f}s")

    def get_n_events(self):
        return self._n_events_total

### we create a wrapper for readRNOGData to mirror the interface of the .nur reader
class _readRNOGData_eventbrowser(readRNOGData):
    """
    Wrapper for readRNOGData for use in the eventbrowser

    This wrapper mirrors the interface of the .nur reader for use in the eventbrowser,
    and uses the lru_cache to speed up IO for parallel plots of the same event.
    It should probably not be used outside of the eventbrowser.

    """
    def begin(self, *args, **kwargs):
        # We reduce the required amount of IO by caching individual events.
        # However, we need to clear the cache every time we change the file
        # we're reading. This is implemented here.
        self.get_event_i.cache_clear()
        self.get_event.cache_clear()
        return super().begin(*args, **kwargs)

    def get_event_ids(self):
        event_infos = self.get_events_information()
        return np.array([(i['run'], i['eventNumber']) for i in event_infos.values()])

    @lru_cache(maxsize=1)
    def get_event_i(self, i):
        return self.get_event_by_index(i)

    @lru_cache(maxsize=1)
    def get_event(self, event_id):
        return super().get_event(*event_id)

    def get_detector(self):
        """Not implemented in mattak reader"""
        return None

    def get_header(self):
        """Not implemented in mattak reader"""
        return None
