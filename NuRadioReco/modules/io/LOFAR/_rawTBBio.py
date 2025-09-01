"""
This module implements an interface for reading LOFAR TBB data.

Adapted from https://github.com/Bhare8972/LOFAR-LIM/blob/master/LoLIM/IO/raw_tbb_IO.py;
original module description is included below.
Most users will not want to use this module directly, but instead use the
`readLOFARData <NuRadioReco.modules.io.LOFAR.readLOFARData>` module, which converts the
TBB data to the NuRadio ``Event`` format, taking care to use the correct units etc.
Note that to minimize changes, **this** module adheres to the LOFAR internal units (seconds / Hz)
rather than using the NuRadio unit system.

Original description
--------------------
This module is strongly based on pyCRtools module tbb.py by Pim Schellart, Tobias Winchen, and others.
However, it has been completely re-written for use with LOFAR-LIM

Author: Brian Hare

Definitions:
LOFAR is split into a number of different stations. There are three main types:

#. Core Stations (CS)
#. Remote Stations (RS)
#. international stations


Each station contains 96 low band antennas (LBA) and 48 high band antennas (HBA). Each antenna is dual polarized.

Each station is referred to by its name (e.g. "CS001"), which is a string, or its ID (e.g. 1), which is an integer. In
general, these are different! The mapping, however, is unique and is given in `utilities.py`.

There are a few complications with reading the data.

#. The data from each station is often spread over multiple files
   There is a class below that can combine multiple files (even from different stations)
#. It is entirely possible that one file could contain multiple stations
   This feature is not used, so I assume that it isn't a problem (for now)
#. Each Station has unknown clock offsets. Technically the core stations are all on one clock, but there are some
   unknown cable delays. This is a difficult  problem, not handled here
#. Each Antenna doesn't necessarily start reading data at precisely the same time.
   The code below picks the latest start time so this problem can be ignored by the end user
#. The software that inserts metadata (namely antenna positions and calibrations) sometimes "forgets" to do its job
   The code below will automatically read the metadata from other files when necessary
#. LOFAR is constantly changing
   So..keeping code up-to-date and still backwards compatible will be an interesting challenge
#. LOFAR only has 96 RCUs (receiver control units) per station (at the moment).
   Each RCU is essentially one digitizer. Each antenna needs two RCS to record both polarizations. The result is only
   1/3 of the antennas can be read out each time.

   LOFAR keeps track of things with two ways. First, the data is all referred to by its RCUid. 0 is the 0th RCU, ect...
   However, which antenna corresponds to which RCU depends on the antennaSet. For LOFAR-LIM the antenna set will
   generally be "LBA_OUTER". This could change, and sometimes the antenna set is spelled wrong in the data files. (This
   should be handled here though)

   In the code below each RCU is referred to by ANTENNA_NAME or antennaID. These are the same thing (I think). They are
   however, a misnomer, as they actually refer to the RCU, not antenna. The specific antenna depends on the antenna
   set. For the same antenna set, however, the ANTENNA_NAME will always refer to the same antenna.

   Each ANTENNA_NAME is a string of 9 digits. First three is the station ID (not name!), next three is the group (no
   idea, don't ask), final 3 is the RCU id

   For LBA_INNER data set, even RCU ids refer to X-polarized dipoles and odd RCU ids refer to Y-polarized dipoles. This
   is flipped for LBA_OUTER antenna set. X-polarization is NE-SW, and Y-polarization is NW-SE. antenna_response.py,
   which handles the antenna function, assumes the data is LBA_OUTER.

"""

import datetime
import os
import logging

import h5py
import numpy as np

import NuRadioReco.modules.io.LOFAR._rawTBBio_metadata as md
import NuRadioReco.modules.io.LOFAR._rawTBBio_utilities as util


logger = logging.getLogger('NuRadioReco.LOFAR.rawTBBio')


# nyquist_zone = {'LBA_10_90' : 1, 'LBA_30_90' : 1, 'HBA_110_190' : 2, 'HBA_170_230' : 3, 'HBA_210_250' : 3}
conversion_dict = {
    "": 1.0,
    "kHz": 1000.0,
    "MHz": 10.0 ** 6,
    "GHz": 10.0 ** 9,
    "THz": 10.0 ** 12,
}


# Helper functions

# The following four functions read what I call "correction files" these are corrections made to improve the data
def read_antenna_pol_flips(fname):
    antennas_to_flip = []
    with open(fname) as fin:
        for line in fin:
            ant_name = line.split()[0]
            antennas_to_flip.append(ant_name)
    return antennas_to_flip


def read_bad_antennas(fname):
    bad_antenna_data = []

    def parse_line_v1(line):
        ant_name, pol = line.split()[0:2]
        bad_antenna_data.append((ant_name, int(pol)))

    def parse_line_v2(line):
        ant_name = line.split()[0]
        pol = 0
        if not util.antName_is_even(ant_name):
            ant_name = util.even_antName_to_odd(ant_name)
            pol = 1
        bad_antenna_data.append((ant_name, pol))

    version = 1
    with open(fname) as fin:
        is_line_0 = True
        for file_line in fin:
            if is_line_0 and file_line[:2] == "v2":
                version = 2
            else:
                if version == 1:
                    parse_line_v1(file_line)
                elif version == 2:
                    parse_line_v2(file_line)

            if is_line_0:
                is_line_0 = False

    return bad_antenna_data


def read_antenna_delays(fname):
    additional_ant_delays = {}

    def parse_line_v1(line):
        ant_name, pol_E_delay, pol_O_delay = line.split()[0:3]
        additional_ant_delays[ant_name] = [float(pol_E_delay), float(pol_O_delay)]

    def parse_line_v2(line):
        ant_name, delay = line.split()[0:2]
        pol = 0
        if not util.antName_is_even(ant_name):
            ant_name = util.even_antName_to_odd(ant_name)
            pol = 1

        if ant_name not in additional_ant_delays:
            additional_ant_delays[ant_name] = [0.0, 0.0]

        additional_ant_delays[ant_name][pol] = float(delay)

    parse_function = parse_line_v1
    with open(fname) as fin:
        is_line_0 = True
        for file_line in fin:
            if is_line_0 and file_line[0] == "v":
                if file_line[:2] == "v1":
                    pass
                elif file_line[:2] == "v2":
                    parse_function = parse_line_v2
            else:
                parse_function(file_line)

            if is_line_0:
                is_line_0 = False

    return additional_ant_delays


def read_station_delays(fname):
    station_delays = {}
    with open(fname) as fin:
        for line in fin:
            sname, delay = line.split()[0:2]
            station_delays[sname] = float(delay)
    return station_delays


def decode_if_needed(input_decode):
    if not isinstance(input_decode, str):
        return input_decode.decode()
    return input_decode


class TBBData_Dal1:
    """
    A class for reading one station from one file.

    However, since one station is often spread between different files,
    use filePaths_by_stationName combined with :class:`MultiFile_Dal1` below.
    """

    def __init__(
            self,
            filename,
            force_metadata_ant_pos=False,
            forcemetadata_delays=True,
            metadata_dir=None,
    ):
        self.filename = filename
        self.metadata_dir = metadata_dir
        self.force_metadata_ant_pos = force_metadata_ant_pos
        self.forcemetadata_delays = forcemetadata_delays

        # open file and set some basic info
        self.file = h5py.File(filename, "r")

        stationKeys = [s for s in self.file.keys() if s.startswith("Station")]
        # assume there is only one station in the file
        if len(stationKeys) != 1:
            logger.warning(f"File {self.filename} has more then one station")
        self.stationKey = stationKeys[0]

        self.antennaSet = decode_if_needed(self.file.attrs["ANTENNA_SET"][0])
        self.dipoleNames = list(self.file[self.stationKey].keys())
        self.StationID = self.file[self.stationKey][self.dipoleNames[0]].attrs[
            "STATION_ID"
        ][0]
        self.StationName = util.SId_to_Sname[self.StationID]
        # assume all antennas have the same sample frequency

        self.SampleFrequency = (
                self.file[self.stationKey][self.dipoleNames[0]].attrs[
                    "SAMPLE_FREQUENCY_VALUE"
                ][0]
                * conversion_dict[
                    decode_if_needed(self.file[self.stationKey][self.dipoleNames[0]].attrs[
                                         "SAMPLE_FREQUENCY_UNIT"
                                     ][0])
                ]
        )

        # PyCRTools comparison (testing purposes)
        # print(
        #     self.file.attrs["OBSERVATION_FREQUENCY_MIN"],
        #     self.file.attrs["OBSERVATION_FREQUENCY_CENTER"],
        #     self.file.attrs["OBSERVATION_FREQUENCY_MAX"],
        # )  # all return 0. ???

        # filter selection is typically "LBA_10_90"
        self.FilterSelection = decode_if_needed(self.file.attrs["FILTER_SELECTION"][0])

        # check that all antennas start in the same second, and record the same number of samples #
        self.Time = None
        self.DataLengths = np.zeros(len(self.dipoleNames), dtype=int)
        self.SampleNumbers = np.zeros(len(self.dipoleNames), dtype=int)
        for dipole_i, dipole in enumerate(self.dipoleNames):

            if self.Time is None:
                self.Time = self.file[self.stationKey][dipole].attrs["TIME"][0]
            else:
                if self.Time != self.file[self.stationKey][dipole].attrs["TIME"][0]:
                    raise IOError(
                        "antennas do not start at same time in " + self.filename
                    )

            self.DataLengths[dipole_i] = self.file[self.stationKey][dipole].attrs[
                "DATA_LENGTH"
            ][0]
            self.SampleNumbers[dipole_i] = self.file[self.stationKey][dipole].attrs[
                "SAMPLE_NUMBER"
            ][0]

        # get position and delay metadata...maybe
        self.have_metadata = (
                "DIPOLE_CALIBRATION_DELAY_VALUE"
                in self.file[self.stationKey][self.dipoleNames[0]].attrs
        )
        self.antenna_filter = md.make_antennaID_filter(self.dipoleNames)

        # load antenna locations from metadata and from file. IF they are too far apart, then give warning,
        # and use metadata
        self.ITRF_dipole_positions = md.getItrfAntennaPosition(self.StationName, self.antennaSet, self.metadata_dir)[
            self.antenna_filter
        ]  # load positions from metadata file
        if self.have_metadata and not self.force_metadata_ant_pos:

            use_TBB_positions = True
            TBB_ITRF_dipole_positions = np.empty(
                (len(self.dipoleNames), 3), dtype=np.double
            )
            for i, dipole in enumerate(self.dipoleNames):
                TBB_ITRF_dipole_positions[i] = self.file[self.stationKey][dipole].attrs[
                    "ANTENNA_POSITION_VALUE"
                ]

                dif = np.linalg.norm(
                    TBB_ITRF_dipole_positions[i] - self.ITRF_dipole_positions[i]
                )
                if dif > 1 and use_TBB_positions:
                    logger.status(
                        f"Station {self.StationName} has suspicious antenna locations. Using metadata instead",
                    )
                    use_TBB_positions = False

            if use_TBB_positions:
                self.ITRF_dipole_positions = TBB_ITRF_dipole_positions

        self.calibrationDelays = np.zeros(
            len(self.dipoleNames), dtype=np.double
        )  # defined as calibration values in file. Never from external metadata!
        if self.have_metadata:  # and not self.forcemetadata_delays:

            for i, dipole in enumerate(self.dipoleNames):
                self.calibrationDelays[i] = self.file[self.stationKey][dipole].attrs[
                    "DIPOLE_CALIBRATION_DELAY_VALUE"
                ]

        # get the offset, in number of samples, needed so that each antenna starts at the same time #
        self.nominal_sample_number = np.max(self.SampleNumbers)
        self.sample_offsets = self.nominal_sample_number - self.SampleNumbers
        self.nominal_DataLengths = self.DataLengths - self.sample_offsets

    def close_file(self):
        self.file.close()
        return

    # GETTERS #
    def needs_metadata(self):
        """return true if this file does not have metadata"""
        return not self.have_metadata

    def get_station_name(self):
        """returns the name of the station, as a string"""
        return self.StationName

    def get_station_ID(self):
        """
        returns the ID of the station, as an integer.

        This is not the same as StationName. Mapping is given in utilities
        """
        return self.StationID

    def get_antenna_names(self):
        """return name of antenna as a list of strings. This is really the RCU id, and the physical antenna depends
        on the antennaSet """
        return self.dipoleNames

    def get_antenna_set(self):
        """return the antenna set as a string. Typically "LBA_OUTER" """
        return self.antennaSet

    def get_sample_frequency(self):
        """gets samples per second. Typically 200 MHz."""
        return self.SampleFrequency

    def get_filter_selection(self):
        """return a string that represents the frequency filter used. Typically "LBA_10_90"""
        return self.FilterSelection

    def get_timestamp(self):
        """return the POSIX timestamp of the first data point"""
        return self.Time

    def get_full_data_lengths(self):
        """get the number of samples stored for each antenna. Note that due to the fact that the antennas do not
        start recording at the exact same instant (in general), this full data length is not all usable returns array
        of ints """
        return self.DataLengths

    def get_all_sample_numbers(self):
        """return numpy array that contains the sample numbers of each antenna. Divide this by the sample frequency
        to get time since the timestamp of the first data point. Note that since these are, in general, different,
        they do NOT refer to sample 0 of "get_data" in general """
        return self.SampleNumbers

    def get_nominal_sample_number(self):
        """return the sample number of the 0th data sample returned by get_data.
        Divide by sample_frequency to get time from timestamp of the 0th data sample"""
        return self.nominal_sample_number

    def get_nominal_data_lengths(self):
        """return the number of data samples that are usable for each antenna, accounting for different starting
        sample numbers. returns array of ints """
        return self.nominal_DataLengths

    def get_ITRF_antenna_positions(self, copy=False):
        """returns the ITRF positions of the antennas. Returns a 2D numpy array. If copy is False, then this just
        returns the internal array of values """
        if copy:
            return np.array(self.ITRF_dipole_positions)
        else:
            return self.ITRF_dipole_positions

    def get_LOFAR_centered_positions(self, out=None):
        """returns the positions (as a 2D numpy array) of the antennas with respect to CS002.
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated"""
        return md.convertITRFToLocal(self.ITRF_dipole_positions, self.metadata_dir, out=out)

    def get_timing_callibration_phases(self):
        """only a test function for the moment, do not use"""
        fpath = os.path.dirname(self.filename) + "/" + self.StationName
        phase_calibration = md.getStationPhaseCalibration(self.StationName, self.antennaSet,
                                                          metadata_dir=self.metadata_dir, file_location=fpath)
        phase_calibration = phase_calibration[self.antenna_filter]
        return phase_calibration

    def get_timing_callibration_delays(self, force_file_delays=False):
        """return the timing calibration of the antennas, as a 1D np array. If not included in the metadata, will look
        for a data file in the same directory as this file. Otherwise returns None"""

        if (self.have_metadata and not self.forcemetadata_delays) or force_file_delays:
            return self.calibrationDelays
        else:
            fpath = os.path.dirname(self.filename) + "/" + self.StationName
            phase_calibration = md.getStationPhaseCalibration(self.StationName, self.antennaSet,
                                                              metadata_dir=self.metadata_dir, file_location=fpath)
            phase_calibration = phase_calibration[self.antenna_filter]
            return md.convertPhase_to_Timing(
                phase_calibration, 1.0 / self.SampleFrequency
            )

    def get_data(self, start_index, num_points, antenna_index=None, antenna_ID=None):
        """
        return the raw data for a specific antenna, as an 1D int16 numpy array, of length num_points.

        First point returned is start_index past get_nominal_sample_number().
        Specify the antenna by giving the antenna_ID (which
        is a string, same as output from get_antenna_names(), or as an integer antenna_index. An antenna_index of 0
        is the first antenna in get_antenna_names().

        """

        if antenna_index is None:
            if antenna_ID is None:
                raise LookupError("need either antenna_ID or antenna_index")
            antenna_index = self.dipoleNames.index(antenna_ID)
        else:
            antenna_ID = self.dipoleNames[antenna_index]

        initial_point = self.sample_offsets[antenna_index] + start_index
        final_point = initial_point + num_points

        if final_point > len(self.file[self.stationKey][antenna_ID]):
            raise IndexError(
                "Data point", final_point,
                "is off end of file with length", len(self.file[self.stationKey][antenna_ID]),
            )

        return self.file[self.stationKey][antenna_ID][initial_point:final_point]


class MultiFile_Dal1:
    """
    A class for reading the data from one station from multiple files

    """

    def __init__(
            self,
            filename_list,
            metadata_dir,
            force_metadata_ant_pos=False,
            polarization_flips=None,
            bad_antennas=None,
            additional_ant_delays=None,
            station_delay=0.0,
            only_complete_pairs=True,
            pol_flips_are_bad=False,
    ):
        """
        Parameters
        ----------
        filename_list: list
            List of filenames for this station for this event
        force_metadata_ant_pos : bool, default=False
            If True, then load antenna positions from a metadata file and not the raw data file
        polarization_flips : list
            List of even antennas where it is known that even and odd antenna names are flipped in file. This is
            assumed to apply both to data and timing calibration
        bad_antennas : list
            Antennas that should not be used. Each item in the list is a tuple, first item of tuple is name of even
            antenna, second item is a 0 or 1 indicating if even or odd antenna is bad. assumed to be BEFORE antenna flips are accounted for
        additional_ant_delays : dict
            Each key is name of even antenna, each value is a tuple with additional even and odd antenna delays.
            This should rarely be needed. assumed to be found BEFORE antenna flips are accounted for
        station_delay : float
            A single number that represents the clock offset of this station, as a delay
        only_complete_pairs : bool
            If True, discards antenna if the other in pair is not present or is bad.
            If False, keeps all good antennas with a 'none' value if other antenna in pair is missing
        pol_flips_are_bad : bool
            If True, antennas that are in pol-flips are included in `bad_antennas`

        Notes
        -----
        This module always defaults to using antenna timing calibration from metadata.

        Also, polarization_flips, bad_antennas, additional_ant_delays, and station_delay can now be strings
        that are file names. If this is the case, they will be read automatically
        """

        self.metadata_dir = metadata_dir
        self.files = [
            TBBData_Dal1(fname, force_metadata_ant_pos, metadata_dir=self.metadata_dir)
            for fname in filename_list
        ]

        if isinstance(polarization_flips, str):
            polarization_flips = read_antenna_pol_flips(polarization_flips)
        if bad_antennas is None:
            bad_antennas = []
        elif isinstance(bad_antennas, str):
            bad_antennas = read_bad_antennas(bad_antennas)
        if isinstance(additional_ant_delays, str):
            additional_ant_delays = read_antenna_delays(additional_ant_delays)

        if polarization_flips is not None and pol_flips_are_bad:
            for even_ant in polarization_flips:
                bad_antennas.append((even_ant, 0))
                bad_antennas.append((even_ant, 1))
            polarization_flips = []

        # get some data that should be constant
        self.antennaSet = self.files[0].antennaSet
        self.StationID = self.files[0].StationID
        self.StationName = self.files[0].StationName
        self.SampleFrequency = self.files[0].SampleFrequency
        self.FilterSelection = self.files[0].FilterSelection
        self.Time = self.files[0].Time
        self.bad_antennas = bad_antennas
        self.odd_pol_additional_timing_delay = (
            0.0  # another timing delay to add to all odd-polarized antennas
        )

        if isinstance(station_delay, str):
            station_delay = read_station_delays(station_delay)[self.StationName]

        self.station_delay = station_delay

        # check consistency of data
        for TBB_file in self.files:
            if TBB_file.antennaSet != self.antennaSet:
                raise IOError(
                    "antenna set not the same between files for station: "
                    + self.StationName
                )
            if TBB_file.StationID != self.StationID:
                raise IOError(
                    "station ID not the same between files for station: "
                    + self.StationName
                )
            if TBB_file.StationName != self.StationName:
                raise IOError(
                    "station name not the same between files for station: "
                    + self.StationName
                )
            if TBB_file.FilterSelection != self.FilterSelection:
                raise IOError(
                    "filter selection not the same between files for station: "
                    + self.StationName
                )
            if TBB_file.Time != self.Time:
                raise IOError(
                    "antenna set not the same between files for station: "
                    + self.StationName
                )

        # check LBA outer antenna set
        if self.antennaSet != "LBA_OUTER":
            logger.warning(
                f"Antenna set on station {self.StationName} is not LBA_OUTER"
            )

        # find best files to get antennas from #
        # require each antenna shows up once, and even pol is followed by odd pol

        self.dipoleNames = []
        self.antenna_to_file = (
            []
        )  # each item is a tuple. First item is file object, second is antenna index in file

        unused_antenna_names = []
        unused_antenna_to_file = []
        bad_PolE_antennas = [ant for ant, pol in bad_antennas if pol == 0]
        bad_PolO_antennas = [
            ant for ant, pol in bad_antennas if pol == 1
        ]  # note that this is still the name of the even antenna, although it is the ODD antenna that is bad!!!
        for TBB_file in self.files:
            file_ant_names = TBB_file.get_antenna_names()

            for ant_i, ant_name in enumerate(file_ant_names):
                if ant_name in self.dipoleNames:
                    continue
                ant_ID = int(ant_name[-3:])

                if ant_ID % 2 == 0:  # check if antenna is even
                    if ant_name in bad_PolE_antennas:
                        continue

                    odd_ant_name = ant_name[:-3] + str(ant_ID + 1).zfill(3)
                    if odd_ant_name in unused_antenna_names:  # we have the odd antenna
                        self.dipoleNames.append(ant_name)
                        self.dipoleNames.append(odd_ant_name)

                        self.antenna_to_file.append((TBB_file, ant_i))
                        odd_unused_index = unused_antenna_names.index(odd_ant_name)
                        self.antenna_to_file.append(
                            unused_antenna_to_file[odd_unused_index]
                        )

                        unused_antenna_names.pop(odd_unused_index)
                        unused_antenna_to_file.pop(odd_unused_index)
                    else:  # we haven't found the odd antenna, so store info for now
                        unused_antenna_names.append(ant_name)
                        unused_antenna_to_file.append((TBB_file, ant_i))

                else:  # antenna is odd
                    even_ant_name = ant_name[:-3] + str(ant_ID - 1).zfill(3)
                    if (
                            even_ant_name in bad_PolO_antennas
                    ):  # note that have to check if EVEN antenna is in bad antenna names...
                        continue

                    if (
                            even_ant_name in unused_antenna_names
                    ):  # we have the odd antenna
                        self.dipoleNames.append(even_ant_name)
                        self.dipoleNames.append(ant_name)

                        even_unused_index = unused_antenna_names.index(even_ant_name)
                        self.antenna_to_file.append(
                            unused_antenna_to_file[even_unused_index]
                        )

                        unused_antenna_names.pop(even_unused_index)
                        unused_antenna_to_file.pop(even_unused_index)

                        self.antenna_to_file.append((TBB_file, ant_i))

                    else:  # we haven't found the odd antenna, so store info for now
                        unused_antenna_names.append(ant_name)
                        unused_antenna_to_file.append((TBB_file, ant_i))

        if not only_complete_pairs:
            for ant_name, to_file in zip(unused_antenna_names, unused_antenna_to_file):
                ant_ID = int(ant_name[-3:])
                if ant_ID % 2 == 0:  # check if antenna is even

                    self.dipoleNames.append(ant_name)
                    self.antenna_to_file.append(to_file)

                    self.dipoleNames.append(
                        ant_name[:-3] + str(ant_ID + 1).zfill(3)
                    )  # add the odd antenna
                    self.antenna_to_file.append(None)  # doesn't exist in a file

                else:

                    self.dipoleNames.append(
                        ant_name[:-3] + str(ant_ID - 1).zfill(3)
                    )  # add the even antenna
                    self.antenna_to_file.append(None)  # doesn't exist in a file

                    self.dipoleNames.append(ant_name)
                    self.antenna_to_file.append(to_file)

        self.index_adjusts = np.arange(
            len(self.antenna_to_file)
        )  # used to compensate for polarization flips
        # when given an antenna index to open data, use this index instead to open the correct data location

        # get sample numbers and offsets and lengths and other related stuff #
        self.SampleNumbers = []
        self.DataLengths = []
        for TBB_file, file_ant_i in self.antenna_to_file:
            self.SampleNumbers.append(TBB_file.SampleNumbers[file_ant_i])
            self.DataLengths.append(TBB_file.DataLengths[file_ant_i])

        self.SampleNumbers = np.array(self.SampleNumbers, dtype=int)
        self.DataLengths = np.array(self.DataLengths, dtype=int)

        self.nominal_sample_number = np.max(self.SampleNumbers)
        self.sample_offsets = self.nominal_sample_number - self.SampleNumbers
        self.nominal_DataLengths = self.DataLengths - self.sample_offsets

        self.even_ant_pol_flips = None
        if polarization_flips is not None:
            self.set_polarization_flips(polarization_flips)
        self.additional_ant_delays = additional_ant_delays

    def set_polarization_flips(self, even_antenna_names):
        """given a set of names(IDs) of even antennas, flip the data between the even and odd antennas"""
        self.even_ant_pol_flips = even_antenna_names
        for ant_name in even_antenna_names:
            if ant_name in self.dipoleNames:
                even_antenna_index = self.dipoleNames.index(ant_name)

                self.index_adjusts[even_antenna_index] += 1
                self.index_adjusts[even_antenna_index + 1] -= 1

    def set_odd_polarization_delay(self, new_delay):
        self.odd_pol_additional_timing_delay = new_delay

    def set_station_delay(self, station_delay):
        """set the station delay, should be a number"""
        self.station_delay = station_delay

    def find_and_set_polarization_delay(self, verbose=False, tolerance=1e-9):
        fpath = os.path.dirname(self.files[0].filename) + "/" + self.StationName
        phase_calibration = md.getStationPhaseCalibration(self.StationName, self.antennaSet,
                                                          metadata_dir=self.metadata_dir, file_location=fpath)
        all_antenna_calibrations = md.convertPhase_to_Timing(
            phase_calibration, 1.0 / self.SampleFrequency
        )

        even_delays = all_antenna_calibrations[::2]
        odd_delays = all_antenna_calibrations[1::2]
        odd_offset = odd_delays - even_delays
        median_odd_offset = np.median(odd_offset)
        logger.info("median offset is:", median_odd_offset)

        below_tolerance = np.abs(odd_offset - median_odd_offset) < tolerance
        logger.info(
            np.sum(below_tolerance),
            "antennas below tolerance.",
            len(below_tolerance) - np.sum(below_tolerance),
            "above.",
        )

        ave_best_offset = np.average(odd_offset[below_tolerance])
        logger.info("average of below-tolerance offset is:", ave_best_offset)

        self.set_odd_polarization_delay(-ave_best_offset)

        above_tolerance = np.zeros(len(all_antenna_calibrations), dtype=bool)
        above_tolerance[::2] = np.logical_not(below_tolerance)
        above_tolerance[1::2] = above_tolerance[::2]
        above_tolerance = above_tolerance[
            md.make_antennaID_filter(self.get_antenna_names())
        ]
        return [AN for AN, AT in zip(self.get_antenna_names(), above_tolerance) if AT]

    def close_file(self):
        """
        Properly close all the TBBData_Dal1 files.
        """
        for file in self.files:
            file.close_file()

    # GETTERS
    def needs_metadata(self):
        for TBB_file in self.files:
            if TBB_file.needs_metadata():
                return True
        return False

    def get_station_name(self):
        """returns the name of the station, as a string"""
        return self.StationName

    def get_station_ID(self):
        """returns the ID of the station, as an integer. This is not the same as StationName. Mapping is given in
        utilities """
        return self.StationID

    def get_antenna_names(self):
        """return name of antenna as a list of strings. This is really the RCU id, and the physical antenna depends
        on the antennaSet """
        return self.dipoleNames

    def has_antenna(self, antenna_name):
        """if only_complete_pairs is False, then we could have antenna names without the data. Return True if we
        actually have the antenna, False otherwise. Account for polarization flips. """
        if antenna_name in self.dipoleNames:
            index = self.index_adjusts[self.dipoleNames.index(antenna_name)]
            if self.antenna_to_file[index] is None:
                return False
            else:
                return True
        else:
            return False

    def get_antenna_set(self):
        """return the antenna set as a string. Typically "LBA_OUTER" """
        return self.antennaSet

    def get_sample_frequency(self):
        """gets samples per second. Typically 200 MHz."""
        return self.SampleFrequency

    def get_filter_selection(self):
        """return a string that represents the frequency filter used. Typically "LBA_10_90"""
        return self.FilterSelection

    def get_timestamp(self):
        """return the POSIX timestamp of the first data point"""
        return self.Time

    def get_timestamp_as_datetime(self):
        """return the POSIX timestamp of the first data point as a python datetime localized to UTC"""
        return datetime.datetime.fromtimestamp(
            self.get_timestamp(), tz=datetime.timezone.utc
        )

    def get_full_data_lengths(self):
        """get the number of samples stored for each antenna. Note that due to the fact that the antennas do not
        start recording at the exact same instant (in general), this full data length is not all usable returns array
        of ints """
        return self.DataLengths

    def get_all_sample_numbers(self):
        """return numpy array that contains the sample numbers of each antenna. Divide this by the sample frequency
        to get time since the timestamp of the first data point. Note that since these are, in general, different,
        they do NOT refer to sample 0 of "get_data" """
        return self.SampleNumbers

    def get_nominal_sample_number(self):
        """return the sample number of the 0th data sample returned by get_data.
        Divide by sample_frequency to get time from timestamp of the 0th data sample"""
        return self.nominal_sample_number

    def get_nominal_data_lengths(self):
        """return the number of data samples that are usable for each antenna, accounting for different starting
        sample numbers. returns array of ints """
        return self.nominal_DataLengths

    def get_ITRF_antenna_positions(self, out=None):
        """returns the ITRF positions of the antennas. Returns a 2D numpy array.
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated.
        Does not account for polarization flips, but shouldn't need too."""
        if out is None:
            out = np.empty((len(self.dipoleNames), 3))

        for ant_i, (TBB_file, station_ant_i) in enumerate(self.antenna_to_file):
            out[ant_i] = TBB_file.ITRF_dipole_positions[station_ant_i]

        return out

    def get_LOFAR_centered_positions(self, out=None):
        """returns the positions (as a 2D numpy array) of the antennas with respect to CS002.
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated.
        Does not account for polarization flips, but shouldn't need too."""
        if out is None:
            out = np.empty((len(self.dipoleNames), 3))

        md.convertITRFToLocal(self.get_ITRF_antenna_positions(), self.metadata_dir, out=out)

        return out

    def get_timing_callibration_phases(self):
        """only a test function for the moment, do not use"""

        out = [None for _ in range(len(self.dipoleNames))]

        for TBB_file in self.files:
            ret = TBB_file.get_timing_callibration_phases()
            if ret is None:
                return None

            for ant_i, (TBB_fileA, station_ant_i) in enumerate(self.antenna_to_file):
                if TBB_fileA is TBB_file:
                    out[ant_i] = ret[station_ant_i]

        return np.array(out)

    def get_timing_callibration_delays(self, out=None, force_file_delays=False):
        """
        return the timing calibration of the antennas, as a 1D np array.

        If not included in the metadata, will look for a data file in the same directory as this file.
        Otherwise returns None. If out is a numpy
        array, it is used to store the antenna delays, otherwise a new array is allocated. This takes polarization
        flips, and additional_ant_delays into account (assuming that both were found BEFORE the pol flip was found).
        Also can account for a timing difference between even and odd antennas, if it is set. """

        if out is None:
            out = np.zeros(len(self.dipoleNames))

        for TBB_file in self.files:
            ret = TBB_file.get_timing_callibration_delays(force_file_delays)
            if ret is None:
                return None

            for ant_i, adjust_i in enumerate(self.index_adjusts):
                TBB_fileA, station_ant_i = self.antenna_to_file[adjust_i]

                if TBB_fileA is TBB_file:
                    out[ant_i] = ret[station_ant_i]

                if self.additional_ant_delays is not None:
                    # additional_ant_delays stores only even antenna names for historical reasons. so we need to be
                    # clever here
                    antenna_polarization = 0 if (ant_i % 2 == 0) else 1
                    even_ant_name = self.dipoleNames[ant_i - antenna_polarization]
                    if even_ant_name in self.additional_ant_delays:
                        if even_ant_name in self.even_ant_pol_flips:
                            antenna_polarization = int(not antenna_polarization)
                        out[ant_i] += self.additional_ant_delays[even_ant_name][
                            antenna_polarization
                        ]

        out[1::2] += self.odd_pol_additional_timing_delay

        return out

    def get_total_delays(self, out=None):
        """Return the total delay for each antenna, accounting for all antenna delays, polarization delay,
        station clock offsets, and trigger time offsets (nominal sample number). This function should be preferred
        over 'get_timing_callibration_delays', but the offsets can have a large average. It is recommended to pick one
        antenna (on your reference station) and use it as a reference antenna so that it has zero timing delay. Note:
        this creates two definitions of T=0. I will call 'uncorrected time' is when the result of this function is
        used as-is, and a reference antenna is not chosen. (IE, the reference station can have a large total_delay
        offset), 'corrected time' will be otherwise. """

        delays = self.get_timing_callibration_delays(out)
        delays += self.station_delay - self.get_nominal_sample_number() * 5.0e-9

        return delays

    def get_time_from_second(self, out=None):
        """return the time (in units of seconds) since the second of each antenna (which should be get_timestamp).
        accounting for delays. This is literally just the opposite of get_total_delays """
        out = self.get_total_delays(out)
        out *= -1
        return out

    def get_geometric_delays(self, source_location, out=None, antenna_locations=None):
        """
        Calculate travel time from a XYZ location to each antenna.

        out can be an array of length equal to number
        of antennas. antenna_locations is the table of antenna locations, given by get_LOFAR_centered_positions(). If
        None, it is calculated. Note that antenna_locations CAN be modified in this function. If antenna_locations is
        less then all antennas, then the returned array will be correspondingly shorter. The output of this function
        plus??? get_total_delays plus emission time of the source is the time the source is seen on each antenna.
        """

        if antenna_locations is None:
            antenna_locations = self.get_LOFAR_centered_positions()

        if out is None:
            out = np.empty(len(antenna_locations), dtype=np.double)

        if len(out) != len(antenna_locations):
            logger.error("Arrays are not of same length in geometric_delays()")
            return None

        antenna_locations -= source_location
        antenna_locations *= antenna_locations
        np.sum(antenna_locations, axis=1, out=out)
        np.sqrt(out, out=out)
        out /= util.v_air
        return out

    def get_data(self, start_index, num_points, antenna_index=None, antenna_ID=None):
        """
        return the raw data for a specific antenna, as an 1D int16 numpy array, of length num_points.

        First point returned is start_index past get_nominal_sample_number(). Specify the antenna by giving the antenna_ID (which
        is a string, same as output from get_antenna_names()) or as an integer antenna_index. An antenna_index of 0
        is the first antenna in get_antenna_names().
        """

        if antenna_index is None:
            if antenna_ID is None:
                raise LookupError("need either antenna_ID or antenna_index")
            antenna_index = self.dipoleNames.index(antenna_ID)

        antenna_index = self.index_adjusts[
            antenna_index
        ]  # in case of polarization flips

        initial_point = self.sample_offsets[antenna_index] + start_index
        final_point = initial_point + num_points

        to_file = self.antenna_to_file[antenna_index]
        if to_file is None:
            raise LookupError("do not have data for this antenna")
        TBB_file, station_antenna_index = to_file
        antenna_ID = self.dipoleNames[antenna_index]

        if final_point > len(TBB_file.file[TBB_file.stationKey][antenna_ID]):
            raise IndexError(
                "Data point", final_point,
                "is off end of file with length", len(TBB_file.file[TBB_file.stationKey][antenna_ID]),
            )

        return TBB_file.file[TBB_file.stationKey][antenna_ID][initial_point:final_point]
