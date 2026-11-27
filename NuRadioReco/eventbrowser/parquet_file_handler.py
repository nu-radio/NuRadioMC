"""
File handler of the eventbrowser for the parquet files of `NuRadioReco.modules.io.eventWriter`.

Offers the interface the eventbrowser uses of `NuRadioRecoio` (`get_n_events`, `get_event_ids`, `get_event_i`,
`get_header`, `get_detector`). Events are read with random access by `eventReader.get_event_i`,
the header from the station parameter columns only (no traces).
"""

import logging
import os
import threading
from collections import defaultdict

import numpy as np

from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.io import eventReader as reader_module
from NuRadioReco.utilities import parquet_utilities as pu

logger = logging.getLogger("NuRadioReco.eventbrowser.parquet_file_handler")


def station_file(path):
    """
    The station file to open for `path`: `path` itself, or the station file ``<name>.parquet`` of an event file
    ``<name>_event.parquet``. Raises `ValueError` for an event file without its station file.
    """
    if path.endswith("_event.parquet") and pu.file_kind(path) == "event":
        candidate = path[: -len("_event.parquet")] + ".parquet"
        if not os.path.exists(candidate):
            raise ValueError(
                f"{path} is an event file, its station file {candidate} does not exist: "
                "the eventbrowser needs the station file."
            )

        return candidate

    return path


class ParquetFileHandler:
    """Random access to the events of parquet files, with the `NuRadioRecoio` interface used by the eventbrowser."""

    def __init__(self, filenames):
        """
        Parameters
        ----------
        filenames : str or list of str
            Station files (an event file is replaced by its station file, see `station_file`).
        """
        if isinstance(filenames, str):
            filenames = [filenames]

        # Event reader (random access), shared by the dash callbacks
        self.__filenames = [station_file(name) for name in filenames]
        self.__reader = reader_module.eventReader()
        self.__reader.begin(self.__filenames)
        self.__lock = threading.Lock()  # dash calls back from several threads

        # Station header (`get_header`), read on first use
        self.__header = None

    def get_filenames(self):
        return self.__filenames

    def get_n_events(self):
        return self.__reader.get_n_events()

    def get_event_ids(self):
        """(run, event id) per event, array of shape (n_events, 2)."""
        return self.__reader.get_event_ids()

    def get_event_i(self, event_number):
        """The `Event` with index `event_number`; None if out of range (as `NuRadioRecoio`)."""
        if event_number < 0 or event_number >= self.get_n_events():
            logger.error(f"event number {event_number} out of bounds, only {self.get_n_events()} present in file")
            return None

        with self.__lock:
            return self.__reader.get_event_i(event_number)

    def get_event(self, event_id):
        """The `Event` with (run, event id) `event_id`."""
        ids = self.get_event_ids()
        index = np.nonzero((ids[:, 0] == event_id[0]) & (ids[:, 1] == event_id[1]))[0]
        if len(index) == 0:
            logger.error(f"event {event_id} not present in file")
            return None

        return self.get_event_i(index[0])

    def get_detector(self):
        """Parquet files store no detector: always None."""
        return None

    def get_header(self):
        """
        Station parameters of all events, as `NuRadioRecoio.get_header`: ``{station_id: {parameter: array}}``
        (and ``stationParameters.station_time``: `astropy.time.Time` objects), read from the parameter columns.

        Unlike for ``.nur`` files, the arrays have one entry per event with this station (in event order),
        a missing value is None (NaN for float parameters). Only parameters set in at least one event are included.
        """
        with self.__lock:
            if self.__header is None:
                self.__header = self.__read_header()

        return self.__header

    def __read_header(self):
        number = {(int(run), int(event_id)): i for i, (run, event_id) in enumerate(self.get_event_ids())}
        rows = defaultdict(list)  # station id -> list of (event number, {key: value})

        for path in pu.find_files(self.__filenames)["station"]:
            file = reader_module._File(path, "station", read_traces=False)
            parameter_keys = [
                key for key in reader_module._parameter_keys(file.station_groups["station"], "") if key[0] in file.columns
            ]
            time_columns = [f"station_time_{suffix}" for suffix in ["jd1", "jd2", "scale"]]
            columns = ["run", "event_id", "station_id"] + [c for c in time_columns if c in file.columns]
            table = file.file.read(columns=columns + [column for column, _, _ in parameter_keys]).to_pydict()

            for i in range(len(table["run"])):
                key = (table["run"][i], table["event_id"][i])
                if key not in number:
                    continue

                values = {member: pu.from_arrow_value(table[column][i], member.dtype) for column, member, _ in parameter_keys}
                values[stnp.station_time] = reader_module._time(
                    {c: table[c][i] for c in time_columns if c in table}, "station_time"
                )
                rows[table["station_id"][i]].append((number[key], values))

        header = {}

        for station_id, station_rows in rows.items():
            station_rows.sort(key=lambda row: row[0])
            header[station_id] = {}

            for member in station_rows[0][1]:
                values = [row[1][member] for row in station_rows]
                if all(value is None for value in values):
                    continue

                if member == stnp.station_time:
                    for value in values:
                        if value is not None:
                            value.format = "isot"

                    array = np.empty(len(values), dtype=object)
                    array[:] = values
                elif member.dtype is float:
                    array = np.array([np.nan if value is None else value for value in values], dtype=float)
                elif member.dtype in (int, bool) and None not in values:
                    array = np.array(values)
                else:
                    array = np.empty(len(values), dtype=object)
                    array[:] = values
                header[station_id][member] = array

        return header
