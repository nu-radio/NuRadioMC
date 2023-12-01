import re
import os
import glob
import json
import logging
import numpy as np

from datetime import datetime
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.modules.base import module

import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import NuRadioReco.framework.hybrid_shower
import NuRadioReco.framework.radio_shower
from NuRadioReco.framework.parameters import stationParameters, showerParameters

import NuRadioReco.modules.io.LOFAR.rawTBBio as rawTBBio
import NuRadioReco.modules.io.LOFAR.rawTBBio_metadata as rawTBBio_metadata


logger = module.setup_logger(level=logging.WARNING)


def get_metadata(filenames, metadata_dir):
    """
    Get metadata from TBB file.

    Parameters
    ----------
    filenames : list[str]
        List of TBB file paths to read in
    metadata_dir : str
        Path to the TBB metadata directory

    """
    logger.info("Getting metadata from filename: %s" % filenames)
    tbb_file = rawTBBio.MultiFile_Dal1(filenames, metadata_dir=metadata_dir)
    station_name = tbb_file.get_station_name()
    antenna_set = tbb_file.get_antenna_set()
    clock_frequency = tbb_file.SampleFrequency

    ns_per_sample = 1.0e9 / clock_frequency
    logger.info("The file contains %3.2f ns per sample" % ns_per_sample)  # test
    time_s = tbb_file.get_timestamp()
    time_ns = ns_per_sample * tbb_file.get_nominal_sample_number()

    positions = tbb_file.get_LOFAR_centered_positions()
    dipole_ids = tbb_file.get_antenna_names()

    # Try to extract calibration delays from TBB metadata
    calibration_delays = tbb_file.get_timing_callibration_delays(force_file_delays=True)

    tbb_file.close_file()

    return (
        station_name,
        antenna_set,
        time_s,
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
    block_size=2**16,
    sampling_frequency=200 * units.MHz,
):
    """
    Calculates block number corresponding to LORA timestamp and the sample number within that block

    Parameters
    ----------
    lora_seconds : int
        LORA timestamp in seconds (UTC timestamp, second after 1st January 1970)
    lora_nanoseconds : int
        LORA timestamp in nanoseconds
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
    A tuple containing `blocknumber` and `samplenumber`, in this order.
    """

    lora_samplenumber = (
            (lora_nanoseconds - clock_offset / units.ns) * sampling_frequency / units.MHz * 1e-3
    )  # MHz to nanoseconds

    value = (lora_samplenumber - sample_number) + (lora_seconds - start_time) * (sampling_frequency / units.Hz)

    if value < 0:
        raise ValueError("Event not in file.")

    return int(value / block_size), int(value % block_size)


def tbb_filetag_from_utc(timestamp):
    """
    Returns TBB filename based on UTC timestamp of an event.

    Parameters
    ----------
    timestamp: int
        UTC timestamp from GPS

    Returns
    -------
    filename: str
        The tag in the TBB filename identifying the files of the event.
    """
    # utc_timestamp_base = 1262304000  # Unix timestamp on Jan 1, 2010 (date -u --date "Jan 1, 2010 00:00:00" +"%s")

    dt_object = datetime.utcfromtimestamp(timestamp)
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


class getLOFARtraces:
    def __init__(
        self, tbb_h5_filename, metadata_dir, time_s, time_ns, trace_length_nbins
    ):
        """
        A Class to facilitate getting traces from LOFAR TBB HDF5 Files

        Parameters
        ----------
        time_s: int
            Event trigger timestamp in UTC seconds
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
        self.time_s = time_s
        self.time_ns = time_ns
        self.alignment_shift = None

        self.setup_trace_loading()

    def setup_trace_loading(self):
        """
        Opens the file and sets some variables.
        so that get_trace() can be called repeatedly for different dipoles.
        """
        self.tbb_file = rawTBBio.MultiFile_Dal1(self.data_filename, metadata_dir=self.metadata_dir)
        sample_number = self.tbb_file.get_nominal_sample_number()
        timestamp = self.tbb_file.get_timestamp()
        station_clock_offsets = rawTBBio_metadata.getClockCorrections(metadata_dir=self.metadata_dir)
        this_station_name = self.tbb_file.get_station_name()

        logger.info("Getting clock offset for station %s" % this_station_name)
        this_clock_offset = station_clock_offsets[this_station_name] * units.s  # kept constant at 1e4 in PyCRTools
        logger.info("Clock offset is %1.4e ns" % (this_clock_offset / units.ns))

        packet = lora_timestamp_to_blocknumber(self.time_s, self.time_ns, timestamp, sample_number,
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
        Parameters
        ----------
        dipole_id: str
            The dipole id
        """

        start_sample = self.trace_length_nbins * self.block_number
        start_sample += self.alignment_shift

        trace = self.tbb_file.get_data(
            start_sample, self.trace_length_nbins, antenna_ID=dipole_id
        )

        return trace

    def close_file(self):
        self.tbb_file.close_file()
        return


# TODO: make reader only read certain stations
class readLOFARData:
    """
    This class reads in the data from the TBB files and puts them into an Event structure. It relies on the KRATOS
    package. If the directory paths are not provided, they default to the ones on COMA.

    Parameters
    ----------
    tbb_directory: Path-like str, default="/vol/astro3/lofar/vhecr/lora_triggered/data/"
        The path to the directory containing the TBB files.
    json_directory: Path-like str, default="/vol/astro7/lofar/kratos_files/json"
        The path to the directory containing the JSON files from LORA.
    metadata_directory: Path-like str, default="/vol/astro7/lofar/vhecr/kratos/data/"
        The path to the directory containing the LOFAR metadata (antenna positions and timing calibrations).
    """
    def __init__(self, restricted_station_set=None, tbb_directory=None, json_directory=None, metadata_directory=None):
        self.logger = logging.getLogger('NuRadioReco.readLOFARData')

        self.tbb_dir = '/vol/astro3/lofar/vhecr/lora_triggered/data/' if tbb_directory is None else tbb_directory
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
                Dictionary with station names as keys and dictionaries as values, who have a `files` key with as
                value a list of TBB filepaths and a `metadata` key which has a list with metadat as value.

        Notes
        -----
        The metadata key is only set in the `readLOFARData.begin()` function, to avoid setting it multiple times if
        there is more than 1 TBB file for a given station.

        Metadata is a list containing (in this order):
            1. station name
            2. antenna set
            3. tbb timestamp (seconds)
            4. tbb timestamp (nanoseconds)
            5. station clock frequency (Hz)
            6. positions of antennas
            7. dipole IDs
            8. calibration delays per dipole
        """
        return np.copy(self.__stations)

    def begin(self, event_id, logger_level=logging.WARNING):
        """
        Prepare the reader to ingest the event with ID `event_id`. This resets the internal representation of the
        stations as well as the event ID. The timestamps are read from the LORA JSON file corresponding to the event.
        The function then globs through the TBB directory to find all files corresponding to the event and adds them to
        the corresponding station file list. It also loads the metadata for every station.

        Parameters
        ----------
        event_id: int
            The ID of the event to load.
        logger_level : int, default=logging.WARNING
            The logging level to use for the module.
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

        # Read in data from LORA file and save it in a HybridShower
        self.__hybrid_shower = NuRadioReco.framework.hybrid_shower.HybridShower("LORA")

        # The LORA coordinate system has x pointing East -> set this through magnetic field vector (values from 2015)
        self.__hybrid_shower.set_parameter(showerParameters.magnetic_field_vector,
                                           np.array([0.004675, 0.186270, -0.456412]))
        self.__hybrid_shower.set_parameter(showerParameters.zenith, lora_dict["LORA"]["zenith_rad"] * units.radian)
        self.__hybrid_shower.set_parameter(showerParameters.azimuth, lora_dict["LORA"]["azimuth_rad"] * units.radian)

        # Go through TBB directory and identify all files for this event
        tbb_filename_pattern = tbb_filetag_from_utc(self.__event_id + 1262304000)  # event id is based on timestamp

        tbb_filename_pattern = self.tbb_dir + "/*" + tbb_filename_pattern + "*.h5"
        self.logger.debug(f'Looking for files with {tbb_filename_pattern}...')
        all_tbb_files = glob.glob(
            tbb_filename_pattern
        )  # this is expensive in a big NFS-mounted directory...
        # TODO: save paths of files per event in some kind of database

        for tbb_filename in all_tbb_files:
            station_name = re.findall(r"CS\d\d\d", tbb_filename)[0]
            if (self.__restricted_station_set is not None) and (station_name not in self.__restricted_station_set):
                continue # only process stations in the given set
            self.logger.info(f'Found file {tbb_filename} for station {station_name}...')
            self.__stations[station_name]['files'].append(tbb_filename)

            # Save the metadata only once (in case there are multiple files for a station)
            if 'metadata' not in self.__stations[station_name]:
                self.__stations[station_name]['metadata'] = get_metadata([tbb_filename], self.meta_dir)

    @register_run()
    def run(self, detector, trace_length=65536):
        """
        Runs the reader with the provided detector. For every station that has files associated with it, a Station
        object is created together with its channels (pulled from the detector description). Every channel also gets
        a group ID which corresponds to the polarisation (i.e. 0 for even and 1 for odd), as to be able to retrieve
        all channels per polarisation during processing.

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
        """
        # Create an empty with 1 run, as only 1 shower per event
        evt = NuRadioReco.framework.event.Event(1, self.__event_id)

        # Add HybridShower to HybridInformation
        evt.get_hybrid_information().add_hybrid_shower(self.__hybrid_shower)

        # Add all Detector stations to Event
        for station_name, station_dict in self.__stations.items():
            station_id = int(station_name[2:])
            station_files = station_dict['files']

            if len(station_files) == 0:
                continue

            station = NuRadioReco.framework.station.Station(station_id)
            radio_shower = NuRadioReco.framework.radio_shower.RadioShower(shower_id=station_id,
                                                                          station_ids=[station_id])

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

            # done here as it needs median timing values over all traces in the station
            flagged_channel_ids = channels_deviating.union(channels_missing_counterpart)
            for channel_id in detector.get_channel_ids(station_id):
                if channel_id in flagged_channel_ids:
                    continue                 
                if detector.get_channel(station_id, channel_id)['ant_orientation_phi'] == 225.0:
                    channel_group = 0
                elif detector.get_channel(station_id, channel_id)['ant_orientation_phi'] == 135.0:
                    channel_group = 1
                else:
                    raise ValueError('Orientation not implemented')
                # read in trace, see if that works. Needed or overly careful?
                try:
                    this_trace = lofar_trace_access.get_trace(str(channel_id).zfill(9))  # channel ID is 9 digits
                except:  # FIXME: Too general except statement
                    flagged_channel_ids.add(channel_id)                    
                    logger.warning("Could not read data for channel id %s" % channel_id)
                    continue 
                
                channel = NuRadioReco.framework.channel.Channel(channel_id, channel_group_id=channel_group)
                channel.set_trace(this_trace, station_dict['metadata'][4] * units.Hz)
                station.add_channel(channel)

            # store set of flagged channel ids as station parameter
            station.set_parameter(stationParameters.flagged_channels, flagged_channel_ids)

            # Add station to Event, together with RadioShower to store reconstruction values later on
            evt.set_station(station)
            evt.add_shower(radio_shower)

            lofar_trace_access.close_file()

        yield evt

    def end(self):
        pass
