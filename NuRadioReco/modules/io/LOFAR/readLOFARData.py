import re
import os
import glob
import json
import logging

from datetime import datetime
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.modules.base import module

import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel

from kratos.data_io import lofar_io


logger = module.setup_logger(level=logging.WARNING)


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


# TODO: make reader only read certain stations
class readLOFARData:
    """
    This class reads in the data from the TBB files and puts them into an Event structure. It relies on the KRATOS
    package.

    Parameters
    ----------
    tbb_directory: Path-like str
        The path to the directory containing the TBB files.
    json_directory: Path-like str
        The path to the directory containing the JSON files from LORA.
    metadata_directory: Path-like str
        The path to the directory containing the LOFAR metadata (antenna positions and timing calibrations).
    """
    def __init__(self, tbb_directory, json_directory, metadata_directory):
        self.logger = logging.getLogger('NuRadioReco.readLOFARData')

        self.tbb_dir = tbb_directory
        self.json_dir = json_directory
        self.meta_dir = metadata_directory

        self.__event_id = None
        self.__stations = None
        self.__lora_timestamp = None
        self.__lora_timestamp_ns = None

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
            'CS301': {'files': []},
            'CS302': {'files': []},
            'CS401': {'files': []},
            'CS501': {'files': []},
        }  # Dictionary containing list of TBB files for every station in event

    def get_stations(self):
        return self.__stations

    def begin(self, event_id):
        """
        Prepare the reader to ingest the event with ID `event_id`. This resets the internal representation of the
        stations as well as the event ID. The timestamps are read from the LORA JSON file corresponding to the event.
        The function then globs through the TBB directory to find all files corresponding to the event and adds them to
        the corresponding station file list. It also loads the metadata for every station.

        Parameters
        ----------
        event_id: int
            The ID of the event to load.
        """
        # Set the internal variables
        self.__event_id = int(event_id)  # ID might be provided as str
        self.__stations = self.__new_stations()

        # Check LORA file for parameters
        with open(os.path.join(self.json_dir, f'{self.__event_id}.json')) as file:
            lora_dict = json.load(file)

        self.__lora_timestamp = lora_dict["LORA"]["utc_time_stamp"]
        self.__lora_timestamp_ns = lora_dict["LORA"]["time_stamp_ns"]

        # Go through TBB directory and identify all files for this event
        tbb_filename_pattern = tbb_filetag_from_utc(self.__event_id + 1262304000)  # event id is based on timestamp

        tbb_filename_pattern = self.tbb_dir + "/*" + tbb_filename_pattern + "*.h5"
        self.logger.debug(f'Looking for files with {tbb_filename_pattern}...')
        all_tbb_files = glob.glob(
            tbb_filename_pattern
        )  # this is expensive in a big NFS-mounted directory...
        # TODO: save paths of files per event in some kind of database

        for tbb_filename in all_tbb_files:
            station_name = re.findall("CS\d\d\d", tbb_filename)[0]
            self.logger.info(f'Found file {tbb_filename} for station {station_name}...')
            self.__stations[station_name]['files'].append(tbb_filename)

            # Save the metadata only once (in case there are multiple files for a station)
            if 'metadata' not in self.__stations[station_name]:
                self.__stations[station_name]['metadata'] = lofar_io.get_metadata(
                    [tbb_filename],  # get_metadata() makes use of MultiFile, which expects a list of filenames
                    self.meta_dir
                )
                # Metadata is a list containing:
                # station name, antenna set, tbb timestamp (seconds), tbb timestamp (nanoseconds),
                # station clock frequency (Hz), positions of antennas, dipole IDs and calibration delays per dipole

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

        # Add all Detector stations to Event
        for station_name, station_dict in self.__stations.items():
            station_id = int(station_name[2:])
            station_files = station_dict['files']

            if len(station_files) == 0:
                continue
            station = NuRadioReco.framework.station.Station(station_id)

            # Use KRATOS io functions to access trace (TODO: import these into NRR)
            lofar_trace_access = lofar_io.GetLOFARTraces(
                station_files,
                self.meta_dir,
                self.__lora_timestamp,
                self.__lora_timestamp_ns,
                trace_length
            )
            for channel_id in detector.get_channel_ids(station_id):
                if detector.get_channel(station_id, channel_id)['ant_orientation_phi'] == 225.0:
                    channel_group = 0
                elif detector.get_channel(station_id, channel_id)['ant_orientation_phi'] == 135.0:
                    channel_group = 1
                else:
                    raise ValueError('Orientation not implemented')
                # TODO: check trace quality
                channel = NuRadioReco.framework.channel.Channel(channel_id, channel_group_id=channel_group)
                channel.set_trace(
                    lofar_trace_access.get_trace(str(channel_id).zfill(9)),  # channel ID is 9 digits
                    station_dict['metadata'][4] * units.Hz
                )
                station.add_channel(channel)

            evt.set_station(station)

            lofar_trace_access.close_file()

        yield evt

    def end(self):
        pass
