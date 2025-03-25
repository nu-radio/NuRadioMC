"""
Reader module for LOFAR data

This module contains the reader class `readLOFARData` for LOFAR data (similar to the `eventReader <NuRadioReco.modules.io.eventReader>`).
This class converts raw TBB (.h5) data to the NuRadioReco `Event <NuRadioReco.framework.event.Event>` structure.
"""
import re
import os
import glob
import json
import math
import logging
from collections import defaultdict

import numpy as np
import radiotools.helper as hp

from astropy.time import Time
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units

import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import NuRadioReco.framework.hybrid_shower
import NuRadioReco.framework.radio_shower
from NuRadioReco.framework.parameters import stationParameters, showerParameters

import NuRadioReco.modules.io.LOFAR._rawTBBio as rawTBBio
import NuRadioReco.modules.io.LOFAR._rawTBBio_metadata as rawTBBio_metadata


logger = logging.getLogger('NuRadioReco.LOFAR.readLOFARData')


def get_metadata(filenames, metadata_dir):
    """
    Get metadata from TBB file.

    Parameters
    ----------
    filenames : list[str]
        List of TBB file paths to read in
    metadata_dir : str
        Path to the TBB metadata directory

    Returns
    -------
    station_name : str
    antenna_set : str
    time_unix : int
    time_ns : float
    clock_frequency : float
    positions : ndarray
    dipole_ids : list
    calibration_delays : ndarray
    """
    logger.info("Getting metadata from filename: %s" % filenames)
    tbb_file = rawTBBio.MultiFile_Dal1(filenames, metadata_dir=metadata_dir)
    station_name = tbb_file.get_station_name()
    antenna_set = tbb_file.get_antenna_set()
    clock_frequency = tbb_file.SampleFrequency * units.Hz # rawTBBio uses s and Hz as base units

    ns_per_sample = units.ns / clock_frequency
    logger.info("The file contains %3.2f ns per sample" % ns_per_sample)  # test
    time_unix = tbb_file.get_timestamp()
    time_ns = ns_per_sample * tbb_file.get_nominal_sample_number()

    positions = tbb_file.get_LOFAR_centered_positions()
    dipole_ids = tbb_file.get_antenna_names()

    # Try to extract calibration delays from TBB metadata
    calibration_delays = tbb_file.get_timing_callibration_delays(force_file_delays=True) * units.s # rawTBBio uses s and Hz as base units

    tbb_file.close_file()

    return (
        station_name,
        antenna_set,
        time_unix,
        time_ns,
        clock_frequency,
        positions,
        dipole_ids,
        calibration_delays,
    )  # switch to dict? But have to choose keys and read the m out again...


def lora_timestamp_to_blocknumber(
        lora_seconds,
        lora_nanoseconds,
        start_time,
        sample_number,
        clock_offset=1e4 * units.ns,
        block_size=2 ** 16,
        sampling_frequency=200 * units.MHz,
):
    """
    Calculates block number corresponding to LORA timestamp and the sample number within that block

    Parameters
    ----------
    lora_seconds : int
        LORA timestamp in seconds (UNIX timestamp, second after 1st January 1970)
    lora_nanoseconds : int
        The number of nanoseconds after `lora_seconds` at which LORA triggered
    start_time : int
        LOFAR TBB timestamp
    sample_number : int
        Sample number in the block where the trace starts
    clock_offset : float, default=1e4 * units.ns
        Clock offset between LORA and LOFAR
    block_size : int, default=2**16
        Block size of the LOFAR data
    sampling_frequency : float, default=200 * units.MHz
        Sampling frequency of LOFAR

    Returns
    -------
    blocknumber : int
        Index of block corresponding to LORA timestamp
    samplenumber : int
        Index of sample corresponding to LORA timestamp, within block ``blocknumber``
    """

    lora_samplenumber = (
            (lora_nanoseconds - clock_offset / units.ns) * sampling_frequency / units.MHz * 1e-3
    )  # MHz to nanoseconds

    value = (lora_samplenumber - sample_number) + (lora_seconds - start_time) * (sampling_frequency / units.Hz)

    if value < 0:
        raise ValueError("Event not in file.")

    return int(value / block_size), int(value % block_size)


def LOFAR_event_id_to_unix(event_id):
    # 1262304000 = Unix timestamp on Jan 1, 2010 (date -u --date "Jan 1, 2010 00:00:00" +"%s")
    return event_id + 1262304000


def tbb_filetag_from_unix(timestamp):
    """
    Returns TBB filename based on UNIX timestamp of an event.

    Parameters
    ----------
    timestamp: int
        UNIX timestamp from GPS

    Returns
    -------
    filename: str
        The tag in the TBB filename identifying the files of the event.

    Notes
    -----
    Technically speaking there is no such thing as an "UTC timestamp". The Coordinated Universal Time (UTC)
    is a time standard , which defines a reference for current time. It uses hours, minutes and seconds to
    divide a day. Crucially, it allows for the introduction of leap seconds.

    The UNIX timestamp on the other hand is defined as the number of **non-leap** seconds passed since
    00:00:00 UTC on 1 January 1970. However, when a leap second occurs the UNIX timestamp is actually reset
    (i.e. that same timestamp refers to two moments in time). As such, the UNIX timestamp remains
    synchronised with the UTC time (except for the one second that is a leap second).

    Note that astropy has support for leap seconds. During the day a leap seconds occurs, UNIX timestamps
    are reported as floating point values (instead of the usual integers). Conversely, converting a UNIX
    timestamp to a datetime will give a millisecond contribution. Though the seconds are still accounted
    for as one would expect.
    """

    dt_object = Time(timestamp, format='unix').to_datetime()
    year = dt_object.year
    month = dt_object.month
    day = dt_object.day
    hour = dt_object.hour
    minute = dt_object.minute
    sec = dt_object.second
    radio_file_tag = "D" + str(year) + str(month).zfill(2) + str(day).zfill(2)
    radio_file_tag += "T" + str(hour).zfill(2) + str(minute).zfill(2)
    radio_file_tag += str(sec).zfill(2)

    return radio_file_tag


def tbbID_to_nrrID(channel_id, mode):
    """
    Converts a TBB channel ID to the corresponding NRR channel ID given the antenna mode.

    This simply adds a "9" as the fourth element of the channelID, if the antenna mode is "LBA_inner".
    The function :func:`nrrID_to_tbbID` can be used to do the opposite.

    As of February 2024, this function only supports "LBA_INNER" and "LBA_OUTER" as possible antenna modes.
    Note that the antenna mode is always converted to lowercase, so the comparison is case-insensitive (i.e.
    "LBA_inner" and "LBA_INNER" are both recognised as the same antenna set).

    Parameters
    ----------
    channel_id: str or int
        TBB channel ID
    mode: {"LBA_inner", "LBA_outer"}
        The antenna set for which to convert (case-insensitive).

    Returns
    -------
    nrr_channel_id: str
        The NuRadioReco channelID corresponding to the TBB channelID depending on whether the antenna mode
        is "lba_inner" or "lba_outer"

    Notes
    -----
    This function encodes the convention used in the `LOFAR.json` detector description. For the inner LBA antennas,
    a "9" was added as the fourth element of the channelID. However, the TBB files always use the same set of
    channel IDs (i.e. the same channel ID refers to different physical antennas depending on the antenna set).
    Given a channel ID from the TBB file and the antenna set, this function returns the channel ID of the
    corresponding channel in the NRR Detector description.
    """

    # if channelID is integer, convert to string and fill it with zeroes at the beginning to get a string length of 9
    if type(channel_id) == int:
        channel_id = str(channel_id).zfill(9)

    if mode.lower() == "lba_outer":  # for LBA_outer, keep the zero on 4th position of string
        # For safety, the string is overwritten here. But it should not be necessary
        nrr_channel_id = channel_id[:3] + '0' + channel_id[4:]
    elif mode.lower() == "lba_inner":  # for LBA_inner, replace the fourth digit (zero) in the string with a 9
        nrr_channel_id = channel_id[:3] + '9' + channel_id[4:]
    else:
        logger.warning("%s is not a valid antenna mode - valid modes are LBA_inner and LBA_outer" % mode)
        nrr_channel_id = channel_id  # return the input channel ID if mode is invalid.

    return nrr_channel_id


def nrrID_to_tbbID(channel_id):
    """
    This function does the opposite of :func:`tbbID_to_nrrID`.

    It returns the TBB channel ID given a NRR channel ID.
    Following the convention used in the LOFAR detector description as of February 2024, this simply replaces the
    fourth element of the channelID with a "0".

    Parameters
    ----------
    channel_id: str or int
        Channel ID

    Returns
    -------
    tbb_channel_id: str
        The TBB channelID corresponding to the NuRadioReco channelID

    See Also
    --------
    tbbID_to_nrrID : convert TBB channel ID to NRR channel ID
    """

    # if channelID is integer, convert to string and fill it with zeroes at the beginning to get a string length of 9
    if type(channel_id) == int:
        channel_id = str(channel_id).zfill(9)

    tbb_channel_id = channel_id[:3] + '0' + channel_id[4:]  # replace fourth element with a 0

    return tbb_channel_id


class getLOFARtraces:
    def __init__(
            self, tbb_h5_filename, metadata_dir, time_unix, time_ns, trace_length_nbins
    ):
        """
        A class to facilitate getting traces from LOFAR TBB HDF5 Files.

        This class is used internally by `readLOFARData` to read in LOFAR traces.
        Most users will want to use the `readLOFARData` to be able to use the

        Parameters
        ----------
        tbb_h5_filename : str
            The TBB (.h5) file to read in.
        metadata_dir : str
            The path where the metadata for the desired event are stored.
        time_unix: int
            Event trigger timestamp in (UNIX) seconds
        time_ns: int
            Event trigger timestamp in ns past UTC second
        trace_length_nbins : int
            Desired length of trace to be loaded from TBB HDF5 files.
            This does not affect trace size read-in for RFI cleaning

        """
        self.metadata_dir = metadata_dir
        self.data_filename = tbb_h5_filename
        self.trace_length_nbins = trace_length_nbins
        self.block_number = None
        self.sample_number_in_block = None
        self.tbb_file = None
        self.time_unix = time_unix
        self.time_ns = time_ns
        self.alignment_shift = None

        self.setup_trace_loading()

    def setup_trace_loading(self):
        """
        Opens the file and sets some variables.

        This enables `get_trace` to be called repeatedly for different dipoles.
        """
        self.tbb_file = rawTBBio.MultiFile_Dal1(self.data_filename, metadata_dir=self.metadata_dir)
        sample_number = self.tbb_file.get_nominal_sample_number()
        timestamp = self.tbb_file.get_timestamp()
        station_clock_offsets = rawTBBio_metadata.getClockCorrections(
            metadata_dir=self.metadata_dir, time=timestamp
        )
        this_station_name = self.tbb_file.get_station_name()

        logger.info("Getting clock offset for station %s" % this_station_name)
        this_clock_offset = station_clock_offsets[this_station_name] * units.s  # kept constant at 1e4 in PyCRTools
        logger.info("Clock offset is %1.4e ns" % (this_clock_offset / units.ns))

        packet = lora_timestamp_to_blocknumber(self.time_unix, self.time_ns, timestamp, sample_number,
                                               clock_offset=this_clock_offset, block_size=self.trace_length_nbins)

        self.block_number, self.sample_number_in_block = packet

        self.alignment_shift = -(
                self.trace_length_nbins // 2 - self.sample_number_in_block
        )  # minus sign, apparently...

        logger.info(
            "Block number = %d, sample number in block = %d, alignment shift = %d"
            % (self.block_number, self.sample_number_in_block, self.alignment_shift)
        )

    def check_trace_quality(self):
        """
        Check all traces recorded from the TBB against quality requirements.

        Returns two sets. The first is a list of dipole ids failing any of the following
        three requirements:

        #. The starting sample number deviates by more than 25% (of the trace length)
           from the median starting sample number;
        #. The starting sample number is later than the median starting sample number;
        #. The length of the recorded trace deviates by more than 10% from the median
           trace length.

        The second set corresponds to all dipole ids for which the matching dipole
        (with the other polarization) is missing.

        Returns
        -------
        deviating_dipoles : set of str
        dipoles_missing_counterpart : set of str
        """
        dipole_names = np.array(self.tbb_file.get_antenna_names())

        # Find the dipoles whose starting sample number and/or number of samples recorded deviates from the median
        sample_number_per_antenna = self.tbb_file.get_all_sample_numbers()
        data_length_per_antenna = self.tbb_file.get_full_data_lengths()

        median_sample_number = np.median(sample_number_per_antenna)
        median_data_length = np.median(data_length_per_antenna)

        deviating_dipole_sample_number = np.where(
            np.abs(
                sample_number_per_antenna - median_sample_number
            ) > median_data_length / 4
        )[0]

        deviating_dipole_starting_later = np.where(
            sample_number_per_antenna > median_sample_number
        )[0]

        deviating_dipole_data_length = np.where(
            np.abs(
                data_length_per_antenna - median_data_length
            ) > median_data_length / 10
        )[0]

        deviating_dipoles = np.unique(
            np.concatenate(
                (
                    deviating_dipole_sample_number,
                    deviating_dipole_starting_later,
                    deviating_dipole_data_length
                )
            )
        )

        # Also check if some dipoles are missing their counterpart
        all_dipoles = [int(x) % 100 for x in self.tbb_file.get_antenna_names()]
        dipoles_missing_counterpart = [x for x in all_dipoles if (x + (1 - 2 * (x % 2))) not in all_dipoles]

        # Use sets for superior search performance
        # Index with lists to make it work for empty arrays
        return set(dipole_names[list(deviating_dipoles)]), set(dipole_names[dipoles_missing_counterpart])

    def get_trace(self, dipole_id):
        """
        Return the trace for antenna ``dipole_id``

        Parameters
        ----------
        dipole_id: str
            The dipole id

        Returns
        -------
        trace : np.ndarray
            The trace for antenna ``dipole_id``
        """

        start_sample = self.trace_length_nbins * self.block_number
        start_sample += self.alignment_shift

        trace = self.tbb_file.get_data(
            start_sample, self.trace_length_nbins, antenna_ID=dipole_id
        )

        return trace

    def close_file(self):
        """
        Closes the currently opened TBB file.
        """
        self.tbb_file.close_file()
        return


class readLOFARData:
    """
    This class reads in the data from the TBB files and puts them into an Event structure.

    This class uses the raw TBB file, (processed) LORA json and LOFAR metadata.
    If the directory paths are not provided, they default to the ones on COMA.

    Parameters
    ----------
    restricted_station_set : list, optional
        Only read in data for stations in ``restricted_station_set``. If not provided,
        read in all stations for which TBB files can be found.
    tbb_directory: Path-like str, default="/vol/astro5/lofar/astro3/vhecr/lora_triggered/data/"
        The path to the directory containing the TBB files.
    json_directory: Path-like str, default="/vol/astro7/lofar/kratos_files/json"
        The path to the directory containing the JSON files from LORA.
    metadata_directory: Path-like str, default="/vol/astro7/lofar/vhecr/kratos/data/"
        The path to the directory containing the LOFAR metadata (antenna positions and timing calibrations).
    """

    def __init__(self, restricted_station_set=None, tbb_directory=None, json_directory=None, metadata_directory=None):
        self.logger = logger  # logging.getLogger('NuRadioReco.readLOFARData')

        self.tbb_dir = '/vol/astro5/lofar/astro3/vhecr/lora_triggered/data/' if tbb_directory is None else tbb_directory
        self.json_dir = '/vol/astro7/lofar/kratos_files/json' if json_directory is None else json_directory
        self.meta_dir = '/vol/astro7/lofar/vhecr/kratos/data/' if metadata_directory is None else metadata_directory

        self.__event_id = None
        self.__stations = None

        self.__lora_timestamp = None
        self.__lora_timestamp_ns = None
        self.__hybrid_shower = None

        self.__restricted_station_set = restricted_station_set

    @staticmethod
    def __new_stations():
        """
        Create a dictionary to contain all the LOFAR stations' file paths and metadata. The 'files' component has
        been initialised with lists in order to make it easier to append multiple files for a given station.

        Returns
        -------
        station_dict: dict
            Dictionary with the station names as keys.
        """
        return {
            'CS001': {'files': []},
            'CS002': {'files': []},
            'CS003': {'files': []},
            'CS004': {'files': []},
            'CS005': {'files': []},
            'CS006': {'files': []},
            'CS007': {'files': []},
            'CS011': {'files': []},
            'CS013': {'files': []},
            'CS017': {'files': []},
            'CS021': {'files': []},
            'CS024': {'files': []},
            'CS026': {'files': []},
            'CS028': {'files': []},
            'CS030': {'files': []},
            'CS031': {'files': []},
            'CS032': {'files': []},
            'CS101': {'files': []},
            'CS103': {'files': []},
            'CS201': {'files': []},
            'CS301': {'files': []},
            'CS302': {'files': []},
            'CS401': {'files': []},
            'CS501': {'files': []},
        }  # Dictionary containing list of TBB files for every station in event

    def get_stations(self):
        """
        Return the internal dictionary which contains the paths to the TBB event files and the extracted metadata
        per stations.

        Returns
        -------
        stations : dict
            Dictionary with station names as keys and dictionaries as values, who have a ``files`` key with as
            value a list of TBB filepaths and a ``metadata`` key which has a list with metadata as value.

        Notes
        -----
        The metadata key is only set in the `readLOFARData.begin` function, to avoid setting it multiple times if
        there is more than 1 TBB file for a given station.

        Metadata is a list containing (in this order):
            1. station name
            2. antenna set
            3. tbb timestamp (unix)
            4. tbb timestamp (nanoseconds after last second)
            5. station clock frequency
            6. positions of antennas
            7. dipole IDs
            8. calibration delays per dipole
        """
        return self.__stations.copy()

    def _get_station_calibration_delays(self, station_id):
        """
        Make a dictionary of channel ids and their corresponding calibration delays.

        This is done to avoid misapplying the delays to the wrong channel. Also converts the list
        of channel IDs pulled from the TBB metadata to their NRR channel ID counterpart.

        Parameters
        ----------
        station_id : int
            The station ID for which to get the calibration delays

        Returns
        -------
        station_calibration_delays : dict
            Dictionary containing the NRR channel IDs as keys and the calibration delays as values
        """
        station_name = f"CS{station_id:03}"
        station_calibration_delays = dict(
            zip(
                map(
                    int,
                    [tbbID_to_nrrID(channel_id, self.__stations[station_name]['metadata'][1])
                     for channel_id in self.__stations[station_name]['metadata'][-2]]
                ),
                self.__stations[station_name]['metadata'][-1]
            )
        )

        return station_calibration_delays

    def begin(self, event_id, logger_level=logging.NOTSET):
        """
        Prepare the reader to ingest the event with ID ``event_id``.

        This resets the internal representation of the stations as well as the event ID.
        The timestamps are read from the LORA JSON file corresponding to the event.
        The function then globs through the TBB directory to find all files corresponding to the event and adds them to
        the corresponding station file list. It also loads the metadata for every station.

        Parameters
        ----------
        event_id: int
            The ID of the event to load.
        logger_level : int, default=logging.NOTSET
            Use this parameter to override the logging level for this module.
        """
        self.logger.setLevel(logger_level)

        # Set the internal variables
        self.__event_id = int(event_id)  # ID might be provided as str
        self.__stations = self.__new_stations()

        # Check LORA file for parameters
        with open(os.path.join(self.json_dir, f'{self.__event_id}.json')) as file:
            lora_dict = json.load(file)

        self.__lora_timestamp = lora_dict["LORA"]["utc_time_stamp"]
        self.__lora_timestamp_ns = lora_dict["LORA"]["time_stamp_ns"]

        if self.__lora_timestamp != LOFAR_event_id_to_unix(self.__event_id):
            self.logger.error(f"LORA timestamp {self.__lora_timestamp} does not match event ID {self.__event_id}")

        # Read in data from LORA file and save it in a HybridShower
        self.__hybrid_shower = NuRadioReco.framework.hybrid_shower.HybridShower("LORA")

        # Read in zenith and azimuth -> make sure they are in range [-pi, pi]
        zenith = math.remainder(lora_dict["LORA"]["zenith_rad"], 2 * np.pi)
        azimuth = math.remainder(lora_dict["LORA"]["azimuth_rad"], 2 * np.pi)

        # Read in core position reconstruction from LORA
        core_pos_x = lora_dict["LORA"]["core_x_m"]
        core_pos_y = lora_dict["LORA"]["core_y_m"]

        # Read in energy estimate from LORA
        energy = lora_dict["LORA"]["energy_GeV"]

        # The LORA coordinate system has x pointing East -> set this through magnetic field vector (values from 2015)
        self.__hybrid_shower.set_parameter(showerParameters.magnetic_field_vector,
                                           np.array([0.004675, 0.186270, -0.456412]))
        self.__hybrid_shower.set_parameter(showerParameters.zenith, zenith * units.radian)
        self.__hybrid_shower.set_parameter(showerParameters.azimuth, azimuth * units.radian)

        # Add LORA core and energy to parameters. The z-Position of the core is always at 7.6m for LOFAR
        self.__hybrid_shower.set_parameter(showerParameters.core, np.array([core_pos_x * units.m, core_pos_y * units.m, 7.6 * units.m]))
        self.__hybrid_shower.set_parameter(showerParameters.energy, energy * units.GeV)

        # Go through TBB directory and identify all files for this event
        tbb_filename_pattern = tbb_filetag_from_unix(self.__lora_timestamp)

        tbb_filename_pattern = self.tbb_dir + "/*" + tbb_filename_pattern + "*.h5"
        self.logger.debug(f'Looking for files with {tbb_filename_pattern}...')
        all_tbb_files = glob.glob(
            tbb_filename_pattern
        )  # this is expensive in a big NFS-mounted directory...
        # TODO: save paths of files per event in some kind of database

        for tbb_filename in all_tbb_files:
            station_name = re.findall(r"CS\d\d\d", tbb_filename)
            station_name = next(iter(station_name), None)  # Get the first entry, if the list is not empty -> defaults to None
            if station_name is None:
                logger.status(f'TBB file {tbb_filename} is for remote station, skipping...')
                continue
            if (self.__restricted_station_set is not None) and (station_name not in self.__restricted_station_set):
                continue  # only process stations in the given set

            self.__stations[station_name]['files'].append(tbb_filename)

        # Save the metadata after all files for a station have been found
        # TODO: make metadata a dictionary
        for station_name in self.__stations:
            station_files = self.__stations[station_name]['files']
            if len(station_files) > 0:
                self.logger.info(f'Found files {station_files} for station {station_name}...')
                self.__stations[station_name]['metadata'] = get_metadata(station_files, self.meta_dir)

    @register_run()
    def run(self, detector, trace_length=65536):
        """
        Runs the reader with the provided detector.

        For every station that has files associated with it, a Station
        object is created together with its channels (pulled from the detector description, depending on the antenna
        set (LBA_OUTER/INNER)). Every channel also gets a group ID which is retrieved from the Detector description.
        For LOFAR we use the integer value of the even dipole number, so channels '001000000' and '001000001',
        which are the two dipoles composing one physical antenna, both get group ID 1000000.

        Parameters
        ----------
        detector: Detector object
            The detector description to be used for this event.
        trace_length: int
            Desired length of the trace to be loaded from TBB files.

        Yields
        ------
        evt: Event object
            The event containing all the loaded traces.

        Notes
        -----
        For each LOFAR station, one `Station <NuRadioReco.framework.station.Station>`
        with the corresponding station_id will be created, which contains the voltage traces.
        Note that these voltage traces are already corrected for the calibration delays,
        using the corresponding delays stored in the input TBB file.

        Additionally, the LORA reconstruction data is stored in
        the `HybridShower <NuRadioReco.framework.hybrid_shower.HybridShower>`, and an
        (empty) `RadioShower <NuRadioReco.framework.radio_shower.RadioShower>` is created
        to store the output of additional reconstruction modules.

        """
        # Create an empty with 1 run, as only 1 shower per event
        evt = NuRadioReco.framework.event.Event(1, self.__event_id)

        # Add HybridShower to HybridInformation
        evt.get_hybrid_information().add_hybrid_shower(self.__hybrid_shower)

        # update the detector to the event time
        time = Time(self.__lora_timestamp, format='unix')
        detector.update(time)

        # Add all Detector stations to Event
        for station_name, station_dict in self.__stations.items():
            station_id = int(station_name[2:])
            station_files = station_dict['files']

            if len(station_files) == 0:
                continue

            # The metadata is only defined if there are files in the station
            antenna_set = station_dict['metadata'][1]
            station_calibration_delays = self._get_station_calibration_delays(station_id)

            station = NuRadioReco.framework.station.Station(station_id)
            station.set_station_time(time)

            # Use KRATOS io functions to access trace
            lofar_trace_access = getLOFARtraces(
                station_files,
                self.meta_dir,
                self.__lora_timestamp,
                self.__lora_timestamp_ns,
                trace_length
            )
            channels_deviating, channels_missing_counterpart = lofar_trace_access.check_trace_quality()

            self.logger.debug("Channels deviating: %s" % channels_deviating)
            self.logger.debug("Channels no counterpart: %s" % channels_missing_counterpart)

            # empty set to add the NRR flagged channel IDs to
            flagged_nrr_channel_ids: dict = defaultdict(list)
            flagged_nrr_channel_group_ids: set[int] = set()  # keep track of channel group IDs to remove

            # Get the list of all dipole names which are present in the TTB file
            # This avoids issues when a TBB file would not contain all channels
            channel_tbb_ids: list[str] = station_dict['metadata'][6]

            self.logger.debug(f"These channels are present in the TBB file: {channel_tbb_ids}")

            for TBB_channel_id in channel_tbb_ids:
                # convert TBB ID to NRR equivalent based on antenna set to be able to access trace
                channel_id = int(tbbID_to_nrrID(TBB_channel_id, antenna_set))

                if TBB_channel_id in channels_deviating:
                    self.logger.status(f"Channel {channel_id} was flagged at read-in, "
                                       f"not adding to station {station_name}")
                    flagged_nrr_channel_ids[channel_id].append("reader_deviating_channel")
                    flagged_nrr_channel_group_ids.add(detector.get_channel_group_id(station_id, channel_id))
                    continue
                elif TBB_channel_id in channels_missing_counterpart:
                    self.logger.status(f"Channel {channel_id} was flagged at read-in, "
                                       f"not adding to station {station_name}")
                    flagged_nrr_channel_ids[channel_id].append("reader_channel_missing_counterpart")
                    flagged_nrr_channel_group_ids.add(detector.get_channel_group_id(station_id, channel_id))
                    continue

                # read in trace, see if that works. Needed or overly careful?
                try:
                    this_trace = lofar_trace_access.get_trace(TBB_channel_id)  # TBB_channel_id is str of 9 digits
                except IndexError:
                    self.logger.warning(f"Could not read data for channel id {channel_id}")
                    flagged_nrr_channel_ids[channel_id].append("reader_trace_error")
                    flagged_nrr_channel_group_ids.add(detector.get_channel_group_id(station_id, channel_id))
                    continue

                # The channel_group_id should be interpreted as an antenna index (e.g. like 'a1000000' which
                # was used in PyCRTools). The group ID is pulled from the Detector description.
                # Example: dipoles '001000000' (NRR ID 1000000) and '001000001' (NRR ID 1000001)
                # both get group ID 1000000
                channel_group: int = detector.get_channel_group_id(station_id, channel_id)

                channel = NuRadioReco.framework.channel.Channel(channel_id, channel_group_id=channel_group)
                channel.set_trace(this_trace, station_dict['metadata'][4])
                channel.apply_time_shift(-1 * station_calibration_delays[channel_id]) # apply the calibration delays
                station.add_channel(channel)

            # Check both channels from the flagged group IDs are removed from station
            # This is needed because when a trace read in fails, the counterpart is not automatically removed
            self.logger.debug(f"Flagged channel group IDs: {flagged_nrr_channel_group_ids}")
            channels_to_remove = []  # cannot remove channel in loop, so store them and delete after
            for channel_group_id in flagged_nrr_channel_group_ids:
                try:
                    for channel in station.iter_channel_group(channel_group_id):
                        self.logger.status(f"Removing channel {channel.get_id()} with group ID {channel_group_id} "
                                           f"from station {station_name}")
                        channels_to_remove.append(channel)
                except ValueError:
                    # The channel_group_id not longer present in the station
                    self.logger.debug(f"Both channels of group ID {channel_group_id} were already removed "
                                      f"from station {station_name}")

            for channel in channels_to_remove:
                station.remove_channel(channel)
                flagged_nrr_channel_ids[channel.get_id()].append("reader_removed_group_id")

            # store set of flagged nrr channel ids as station parameter
            station.set_parameter(stationParameters.flagged_channels, flagged_nrr_channel_ids)

            # Add station to Event
            evt.set_station(station)

            lofar_trace_access.close_file()

        # Add general event radio shower to event to store reconstruction values later
        radio_shower = NuRadioReco.framework.radio_shower.RadioShower(
            shower_id=evt.get_id(), station_ids=evt.get_station_ids()
        )
        radio_shower.set_parameter(showerParameters.observation_level, 760*units.cm)
        radio_shower.set_parameter(showerParameters.magnetic_field_vector, hp.get_magnetic_field_vector("lofar"))
        evt.add_shower(radio_shower)
        yield evt

    def end(self):
        pass
