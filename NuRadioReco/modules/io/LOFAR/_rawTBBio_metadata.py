"""
Module to read in calibration metadata from LOFAR TBB files.

Adapted from https://github.com/Bhare8972/LOFAR-LIM/blob/master/LoLIM/IO/metadata.py;
original description included below.
Most users will not want to use this module directly, but instead use the
`readLOFARData <NuRadioReco.modules.io.LOFAR.readLOFARData>` module, which converts the
TBB data to the NuRadio ``Event`` format, taking care to use the correct units etc.
Note that to minimize changes, **this** module adheres to the LOFAR internal units (seconds / Hz)
rather than using the NuRadio unit system.

Original description
--------------------
This module reads in calibration metadata from file in the early phases of LOFAR. In the future this should be replaced
by reading the metadata from the files.

.. moduleauthor:: Sander ter Veen <s.terveen@astro.ru.nl>

Modified by Brian Hare for use with LOFAR for Lightning Imaging.
"""

import logging
import numpy as np

from NuRadioReco.modules.io.LOFAR._rawTBBio_utilities import SId_to_Sname


logger = logging.getLogger('NuRadioReco.LOFAR.rawTBBio_metadata')


def make_antennaID_filter(antenna_ids):
    """
    For a list of antennaIDs, return a filter to filter data by antenna.

    Examples
    --------
    >>> getStationPhaseCalibration("CS001","LBA_OUTER",)
    [ make_antennaID_filter(["002000001"]) ]

    Notes
    -----
    Only works for one station at a time.
    Assumes that the array you want to filter includes ALL antennas in the appropriate antenna set.
    """

    RCU_id = np.array([int(ID[-3:]) for ID in antenna_ids])
    return RCU_id


def mapAntennasetKeyword(antenna_set):
    """
    Ugly fix to map correct antenna names in input to wrong antenna names
    for metadata module.
    """

    # Strip whitespace
    antenna_set = antenna_set.strip()

    allowed = ["LBA_OUTER", "LBA_INNER",
               "LBA_SPARSE_EVEN", "LBA_SPARSE_ODD",
               "LBA_X", "LBA_Y",
               "HBA", "HBA_0", "HBA_1"]

    incorrect = {
        "LBA_INNER": "LBA_INNER",
        "LBA_OUTER": "LBA_OUTER",
        "LBA_SPARSE0": "LBA_SPARSE_EVEN",
        "LBA_SPARSE1": "LBA_SPARSE_ODD",
        "HBA_ZERO": "HBA_0",
        "HBA_ONE": "HBA_1",
        "HBA_DUAL": "HBA",
        "HBA_JOINED": "HBA",
        "HBA_ZERO_INNER": "HBA_0",  # Only true for core stations
        "HBA_ONE_INNER": "HBA_1",  # Only true for core stations
        "HBA_DUAL_INNER": "HBA",  # Only true for core stations
        "HBA_JOINED_INNER": "HBA",
    }  # Only true for core stations

    if antenna_set in incorrect:
        antenna_set = incorrect[antenna_set]
    elif antenna_set == "HBA_BOTH":
        # This keyword is also wrong but present in file headers
        print("Keyword " + antenna_set + " does not comply with ICD, mapping...")
        antenna_set = "HBA"

    assert antenna_set in allowed, f"Antenna set {antenna_set} is not allowed!"

    return antenna_set


def getItrfAntennaPosition(station, antenna_set, metadata_dir):
    """
    Returns the antenna positions of all the antennas in the station
    in ITRF coordinates for the specified antennaset.
    station can be the name or id of the station.

    Parameters
    ----------
    station : int or str
        Name or id of the station. e.g. "CS302" or 142
    antenna_set : {LBA_INNER, LBA_OUTER, LBA_X, LBA_Y, LBA_SPARSE0, LBA_SPARSE1, HBA_0, HBA_1, HBA}
        Antenna set used for this station
    metadata_dir: str
        Path to the directory containing the LOFAR static metadata.

    """
    # Check station id type
    if isinstance(station, int):
        # Convert a station id to a station name
        station = SId_to_Sname[station]

    antenna_set = mapAntennasetKeyword(antenna_set)

    if "LBA" in antenna_set:
        antennatype = "LBA"
    elif "HBA" in antenna_set:
        antennatype = "HBA"

    # Obtain filename of antenna positions
    filename = (
            metadata_dir
            + "/lofar/StaticMetaData/AntennaFields/"
            + station
            + "-AntennaField.conf"
    )

    # Open file
    f = open(filename, "r")

    if station[0:2] != "CS":
        if "HBA" in antenna_set:
            antenna_set = "HBA"

    # Find position of antennaset in file
    str_line = ""
    while antennatype != str_line.strip():
        str_line = f.readline()
        if len(str_line) == 0:
            # end of file reached, no data available
            assert False

    # Find the location of the station. Antenna locations are relative to this
    str_line = f.readline()
    str_split = str_line.split()
    stationX = float(str_split[2])
    stationY = float(str_split[3])
    stationZ = float(str_split[4])

    str_line = f.readline()

    # Get number of antennas and the number of directions
    nrantennas = int(str_line.split()[0])
    nrdir = int(str_line.split()[4])

    antenna_positions = np.empty((2 * nrantennas, nrdir), dtype=np.double)
    for i in range(nrantennas):
        line = f.readline().split()

        antenna_positions[2 * i, 0] = float(line[0]) + stationX
        antenna_positions[2 * i, 1] = float(line[1]) + stationY
        antenna_positions[2 * i, 2] = float(line[2]) + stationZ

        antenna_positions[2 * i + 1, 0] = float(line[3]) + stationX
        antenna_positions[2 * i + 1, 1] = float(line[4]) + stationY
        antenna_positions[2 * i + 1, 2] = float(line[5]) + stationZ

    if antennatype == "LBA":
        # There are three types of feed
        # H for HBA
        # h for lbh
        # l for lbl
        feed = {"CS": {}, "RS": {}, "DE": {}}
        feed["CS"]["LBA_SPARSE_EVEN"] = "24hhll"
        feed["CS"]["LBA_SPARSE_ODD"] = "24llhh"
        feed["CS"]["LBA_X"] = "48hl"
        feed["CS"]["LBA_Y"] = "48lh"
        feed["CS"]["LBA_INNER"] = "96h"
        feed["CS"]["LBA_OUTER"] = "96l"
        feed["RS"]["LBA_SPARSE_EVEN"] = "24hhll"
        feed["RS"]["LBA_SPARSE_ODD"] = "24llhh"
        feed["RS"]["LBA_X"] = "48hl"
        feed["RS"]["LBA_Y"] = "48lh"
        feed["RS"]["LBA_INNER"] = "96h"
        feed["RS"]["LBA_OUTER"] = "96l"
        feed["DE"]["LBA"] = "192h"
        if station[0:2] == "CS" or "RS":
            feedsel = feed[station[0:2]][antenna_set]
            nrset = int(feedsel.split("l")[0].split("h")[0].split("H")[0])
            feeds = ""
            feedsel = feedsel[len(str(nrset)) :]
            for i in range(nrset):
                feeds += feedsel

        indexselection = []
        for i in range(len(feeds)):
            if feeds[i] == "l":
                # The 'l' feeds are the last 96 numbers of the total list
                indexselection.append(i + 96)
            elif feeds[i] == "h":
                # The 'h' feeds are the first 96 numbers of the total list
                indexselection.append(i)
            else:
                # This selection is not yet supported
                assert False
        antenna_positions = antenna_positions[indexselection]

    return antenna_positions


def getStationPositions(station, antenna_set, coordinate_system, metadata_dir):
    """
    Returns the antenna positions of all the antennas in the station
    relative to the station center for the specified antenna set.
    station can be the name or id of the station.

    Parameters
    ----------

    station : int or str
        Name or id of the station. e.g. "CS302" or 142
    antenna_set : {LBA_INNER, LBA_OUTER, LBA_X, LBA_Y, LBA_SPARSE0, LBA_SPARSE1, HBA_0, HBA_1, HBA}
        Antenna set used for this station
    coordinate_system : {WGS84, ITRF}
        The coordinate system to use when returning antenna positions (see also Notes section).\
    metadata_dir : str
        Path to the directory containing the LOFAR static metadata.

    Notes
    -----
    For the coordinate system, when using "WGS84", the function returns the postions as a Numpy array containing
    [lat, lon, alt] of antenna positions. Else, when using the ITRF coordinate system, is return the positions as
    [X, Y, Z].

    """

    # Check if requested antennaset is known
    assert coordinate_system in ["WGS84", "ITRF"]

    # Check station id type
    if isinstance(station, int):
        # Convert a station id to a station name
        station = SId_to_Sname[station]

    antenna_set = mapAntennasetKeyword(antenna_set)

    # Obtain filename of antenna positions
    if "WGS84" in coordinate_system:
        filename = (
                metadata_dir
                + "/lofar/StaticMetaData/AntennaArrays/"
                + station
                + "-AntennaArrays.conf"
        )
    else:
        filename = (
                metadata_dir
                + "/lofar/StaticMetaData/AntennaFields/"
                + station
                + "-AntennaField.conf"
        )

    # Open file
    f = open(filename, "r")

    if "LBA" in antenna_set:
        antenna_set = "LBA"

    if station[0:2] != "CS":
        if "HBA" in antenna_set:
            antenna_set = "HBA"

    # Find position of antennaset in file
    str_line = ""
    while antenna_set != str_line.strip():
        str_line = f.readline()
        if len(str_line) == 0:
            # end of file reached, no data available
            print("Antenna set not found in calibration file", filename)
            return None

    # Skip name and station reference position
    str_line = f.readline().split()

    A = float(str_line[2])  # lon in WGS84, X in ITRF
    B = float(str_line[3])  # lat in WGS84, Y in ITRF
    C = float(str_line[4])  # alt in WGS84, Z in ITRF

    return np.array([A, B, C])


def convertITRFToLocal(itrfpos, metadata_dir, phase_center=None, ref_lat_lon=None, out=None):
    """

    Parameters
    ----------
    itrfpos : list or np.ndarray
        The ITRF positions as 1D numpy array, or list of positions as a 2D array
    metadata_dir: str
        Path to the directory containing the LOFAR static metadata.
    phase_center: list or np.ndarray, default=None
        The origin of the coordinate system, in ITRF. Default is the coordinates of station CS002.
    ref_lat_lon: list or np.ndarray, default=None
        The rotation of the coordinate system, i.e. the [lat, lon] (in degrees) on the Earth which defines "UP".
        If not specified, the coordinates of CS002 will be used.
    out: np.ndarray, default=None
        If given, the output will be stored in this array. Otherwise, a new array will be created and returned.
        Cannot be same array as itrfpos

    Notes
    -----
    Function returns a 2D numpy array (even if input is 1D).

    """
    if ref_lat_lon is None:
        ref_lat_lon = [52.91512249, 6.869837540]

    if phase_center is None:
        phase_center = getStationPositions("CS002", "LBA_OUTER", coordinate_system="ITRF", metadata_dir=metadata_dir)
        # ($LOFARSOFT/data/lofar/StaticMetaData/AntennaFields/CS002-AntennaField.conf)
    if out is itrfpos:
        logger.error(
            "The `out` array cannot be same as `itrfpos` in convertITRFToLocal."
        )
        raise ValueError

    lat = np.deg2rad(ref_lat_lon[0])
    lon = np.deg2rad(ref_lat_lon[1])
    arg0 = np.array(
        [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)]
    )
    arg1 = np.array(
        [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)]
    )
    arg2 = np.array([0.0, np.cos(lat), np.sin(lat)])

    if out is None:
        ret = np.empty(itrfpos.shape, dtype=np.double)
    else:
        ret = out

    ret[:] = np.outer(itrfpos[..., 0] - phase_center[0], arg0)
    ret += np.outer(itrfpos[..., 1] - phase_center[1], arg1)
    ret += np.outer(itrfpos[..., 2] - phase_center[2], arg2)

    return ret


def getStationPhaseCalibration(
    station, antenna_set, metadata_dir, file_location=None
):
    """
    Read phase calibration data for a station.

    Parameters
    ----------

    station : int or str
        Name or id of the station. e.g. "CS302" or 142
    antenna_set : {LBA_INNER, LBA_OUTER, LBA_X, LBA_Y, LBA_SPARSE0, LBA_SPARSE1, HBA_0, HBA_1, HBA}
        Antenna set used for this station
    metadata_dir : str
        Path to the directory containing the LOFAR static metadata.
    file_location : str, default=None
        The path to the LOFAR calibration tables. If None, it is assumed to be in the /lofar/StaticMetaData/CalTables
        directory relative to `xMetaData_directory`.

    Returns
    -------

    The weights for 512 subbands.

    Examples
    --------

    >>> getStationPhaseCalibration("CS002","LBA_OUTER",)
    array([[ 1.14260161 -6.07397622e-18j,  1.14260161 -6.05283530e-18j,
         1.14260161 -6.03169438e-18j, ...,  1.14260161 +4.68675289e-18j,
         1.14260161 +4.70789381e-18j,  1.14260161 +4.72903474e-18j],
       [ 0.95669876 +2.41800591e-18j,  0.95669876 +2.41278190e-18j,
         0.95669876 +2.40755789e-18j, ...,  0.95669876 -2.41017232e-19j,
         0.95669876 -2.46241246e-19j,  0.95669876 -2.51465260e-19j],
       [ 0.98463207 +6.80081617e-03j,  0.98463138 +6.89975906e-03j,
         0.98463069 +6.99870187e-03j, ...,  0.98299670 +5.71319125e-02j,
         0.98299096 +5.72306908e-02j,  0.98298520 +5.73294686e-02j],
       ...,
       [ 1.03201290 +7.39535744e-02j,  1.03144532 +8.14880844e-02j,
         1.03082273 +8.90182487e-02j, ..., -0.82551740 -6.23731331e-01j,
        -0.82094046 -6.29743206e-01j, -0.81631975 -6.35721497e-01j],
       [ 1.12370332 -1.15296909e-01j,  1.12428451 -1.09484545e-01j,
         1.12483564 -1.03669252e-01j, ..., -0.92476286 +6.48703460e-01j,
        -0.92810503 +6.43912711e-01j, -0.93142239 +6.39104744e-01j],
       [ 1.10043006 -6.18995646e-02j,  1.10075250 -5.58731668e-02j,
         1.10104193 -4.98450938e-02j, ..., -1.01051042 +4.40052904e-01j,
        -1.01290481 +4.34513198e-01j, -1.01526883 +4.28960464e-01j]])

    >>> getStationPhaseCalibration(122,"LBA_OUTER",)
    Calibration data not yet available. Returning 1
    array([[ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
       [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
       [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
       ...,
       [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
       [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
       [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j]])

    """

    # Return mode nr depending on observation mode
    antennasetToMode = {
        "LBA_OUTER": "LBA_OUTER-10_90",
        "LBA_INNER": "LBA_INNER-10_90",
        "HBA": "HBA-110_190",
        "HBA_0": "HBA-110_190",
        "HBA_1": "HBA-110_190",
    }

    antenna_set = mapAntennasetKeyword(antenna_set)

    if antenna_set not in antennasetToMode.keys():
        raise KeyError("Not a valid antennaset " + antenna_set)

    mode_name = antennasetToMode[antenna_set]
    if not isinstance(station, str):
        # Convert a station id to a station name
        station = SId_to_Sname[station]

    stationNr = station[2:]

    # filename
    if file_location is None:
        file_location = metadata_dir + "/lofar/StaticMetaData/CalTables"

    filename = file_location + "/CalTable-" + stationNr + "-" + mode_name + ".dat"
    with open(filename, "rb") as fin:
        # Test for header record above raw data - present in newer caltables (starting 2012)
        line = fin.readline().decode()
        if "HeaderStart" in line:
            while not "HeaderStop" in line:
                line = fin.readline().decode()
        else:  # no header present, seek to starting position
            fin.seek(0)

        data = np.fromfile(fin, dtype=np.double)

    data.resize(512, 96, 2)

    complexdata = np.empty(shape=(512, 96), dtype=complex)
    complexdata.real = data[:, :, 0]
    complexdata.imag = data[:, :, 1]

    return complexdata.transpose()


def convertPhase_to_Timing(phase_calibration, sample_time=5.0e-9):
    """
    Given the phase calibration of the 512 LOFAR subbands,
    such as the output of getStationPhaseCalibration,
    return the timing callibration of each antenna.
    Not sure how well this works with HBA antennas.
    Sample time should be seconds per sample. Default is 5 ns
    """
    phases = np.angle(phase_calibration)
    delays = (phases[:, 1] - phases[:, 0]) * (1024 / (2 * np.pi)) * sample_time
    return delays


# Functions for previously known clock offsets. Only used for compatibility with past data!
def getClockCorrectionFromParsetAddition(metadata_dir):
    parsetFilename = (
            metadata_dir + "/lofar/station_clock_offsets/StationCalibration.parset"
    )

    offsetDictX = {}
    offsetDictY = {}

    infile = open(parsetFilename, "r")
    for line in infile:
        s = line.split("=")
        value = s[1]
        params = s[0].split(".")
        thisStation = params[2][0:5]
        thisAntennaSet = params[3]
        thisFilter = params[4]
        thisValueType = params[5]
        thisPolarization = params[6][0]

        if (
            thisAntennaSet == "LBA_OUTER"
            and thisFilter == "LBA_30_90"
            and thisValueType == "delay"
        ):
            if thisPolarization == "X":
                offsetDictX[thisStation] = float(value)
            elif thisPolarization == "Y":
                offsetDictY[thisStation] = float(value)
            else:
                raise ValueError("Wrong!")
    infile.close()

    offsetDictCombined = {}

    for key in offsetDictX.keys():
        combined = 0.5 * (offsetDictX[key] + offsetDictY[key])
        offsetDictCombined[key] = combined

    return offsetDictCombined


def getClockCorrections(antenna_set="LBA", time=1383264000 - 1000, metadata_dir=None):
    """Get clock correction for superterp stations in seconds. Currently static values.

    *station* Station name or number for which to get the correction.
    *time* Optional. Linux time of observation. As clocks drift the value from the correct time should be given. Not yet implemented.
    """

    clockcorrection = {}
    if "LBA" in antenna_set:
        if time < (1383264000):
            # Values before 1 Nov 2013, eventID-time 120960000, Unix time: add 1262304000.
            clockcorrection["CS002"] = 8.32233e-06  # definition, global offset
            # Addition is the finetuning using Smilde from 1 or 2 random events, to about +/- 0.2 ns.
            # TODO: Need to check constancy over time.
            clockcorrection["CS003"] = 6.921444e-06 + 0.35e-9
            clockcorrection["CS004"] = 7.884847e-06 + 1.0e-9
            clockcorrection["CS005"] = 8.537828e-06 + 0.14e-9
            clockcorrection["CS006"] = 7.880705e-06 - 0.24e-9
            clockcorrection["CS007"] = 7.916458e-06 - 0.22e-9

            clockcorrection["CS001"] = 4.755947e-06
            clockcorrection["CS011"] = 7.55500e-06 - 0.3e-9
            clockcorrection["CS013"] = 9.47910e-06
            clockcorrection["CS017"] = 1.540812e-05 - 0.87e-9
            clockcorrection["CS021"] = 6.044335e-06 + 1.12e-9
            clockcorrection["CS024"] = 4.66335e-06 - 1.24e-9
            clockcorrection["CS026"] = 1.620482e-05 - 1.88e-9
            clockcorrection["CS028"] = 1.6967048e-05 + 1.28e-9
            clockcorrection["CS030"] = 9.7110576e-06 + 3.9e-9
            clockcorrection["CS031"] = 6.375533e-06 + 1.87e-9
            clockcorrection["CS032"] = 8.541675e-06 + 1.1e-9
            clockcorrection["CS101"] = 1.5155471e-05
            clockcorrection["CS103"] = 3.5503206e-05
            clockcorrection["CS201"] = 1.745439e-05
            clockcorrection["CS301"] = 7.685249e-06
            clockcorrection["CS302"] = 1.2317004e-05
            clockcorrection["CS401"] = 8.052200e-06
            clockcorrection["CS501"] = 1.65797e-05
        else:
            clockcorrection = getClockCorrectionFromParsetAddition(metadata_dir)
            clockcorrection["CS003"] = clockcorrection["CS003"] - 1.7e-9 + 2.0e-9
            clockcorrection["CS004"] = clockcorrection["CS004"] - 9.5e-9 + 4.2e-9
            clockcorrection["CS005"] = clockcorrection["CS005"] - 6.9e-9 + 0.4e-9
            clockcorrection["CS006"] = clockcorrection["CS006"] - 8.3e-9 + 3.8e-9
            clockcorrection["CS007"] = clockcorrection["CS007"] - 3.6e-9 + 3.4e-9
            clockcorrection["CS011"] = clockcorrection["CS011"] - 18.7e-9 + 0.6e-9

    # Old values were
    elif "HBA" in antenna_set:
        # Correct to 2013-03-26 values from parset L111421
        clockcorrection["CS001"] = 4.759754e-06
        clockcorrection["CS002"] = 8.318834e-06
        clockcorrection["CS003"] = 6.917926e-06
        clockcorrection["CS004"] = 7.889961e-06
        clockcorrection["CS005"] = 8.542093e-06
        clockcorrection["CS006"] = 7.882892e-06
        clockcorrection["CS007"] = 7.913020e-06
        clockcorrection["CS011"] = 7.55852e-06
        clockcorrection["CS013"] = 9.47910e-06
        clockcorrection["CS017"] = 1.541095e-05
        clockcorrection["CS021"] = 6.04963e-06
        clockcorrection["CS024"] = 4.65857e-06
        clockcorrection["CS026"] = 1.619948e-05
        clockcorrection["CS028"] = 1.6962571e-05
        clockcorrection["CS030"] = 9.7160576e-06
        clockcorrection["CS031"] = 6.370090e-06
        clockcorrection["CS032"] = 8.546255e-06
        clockcorrection["CS101"] = 1.5157971e-05
        clockcorrection["CS103"] = 3.5500922e-05
        clockcorrection["CS201"] = 1.744924e-05
        clockcorrection["CS301"] = 7.690431e-06
        clockcorrection["CS302"] = 1.2321604e-05
        clockcorrection["CS401"] = 8.057504e-06
        clockcorrection["CS501"] = 1.65842e-05

    else:
        print("ERROR: no clock offsets available for this antennaset: ", antenna_set)
        return 0

    return clockcorrection
