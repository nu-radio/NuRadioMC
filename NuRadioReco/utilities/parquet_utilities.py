"""
Helpers to map typed parameter enums (see `NuRadioReco.framework.parameters`) to pyarrow (library to write parquet files).

The declared ``dtype`` of an enum member determines the column type. Parameterised types such as
``dict[int, dict[int, float]]`` or ``list[float]`` are mapped to native arrow ``map``/``list`` types.
Parameters whose type is not specific enough to define a schema (``Any``, bare ``dict``/``list``,
unknown classes) are dropped with a warning, nothing is pickled.

`ParquetReader` reads the files written by `NuRadioReco.modules.io.eventWriterParquet` with polars:

* The input is a root directory (searched recursively) or a glob pattern.
* Stations and runs are selected via the file names, so unselected files are never opened.
* The electric field and sim channel files can be read in addition.
* `ParquetReader.read` / `ParquetReader.scan` return the event table left-joined with the requested
  table (eager / lazy), `ParquetReader.tables` returns the tables separately.
* `ParquetReader.events` returns only the event table. `ParquetReader.related_tables` then returns the
  electric fields / sim channels of a selection of these events.
* Polars is only imported when a reader is created. It returns parquet maps as ``list[struct{key, value}]``.
"""
import glob
import logging
import os
import re
import typing

import numpy as np
import pyarrow as pa

logger = logging.getLogger("NuRadioReco.parquet_utilities")

_SCALAR_TYPES = {
    bool: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    str: pa.string(),
    complex: pa.struct([("real", pa.float64()), ("imag", pa.float64())]),
    np.ndarray: pa.list_(pa.float64()),
}


def to_arrow_type(dtype):
    """
    Map a python type annotation to an arrow type.

    Parameters
    ----------
    dtype : type or typing generic
        The declared type of a parameter.

    Returns
    -------
    pyarrow.DataType or None
        None if the type can not be mapped to a native arrow type.
    """
    if dtype in _SCALAR_TYPES:
        return _SCALAR_TYPES[dtype]

    origin = typing.get_origin(dtype)
    args = typing.get_args(dtype)
    if origin in (list, tuple) and len(args) >= 1:
        value_type = to_arrow_type(args[0])
        if value_type is not None:
            return pa.list_(value_type)
    elif origin is dict and len(args) == 2:
        key_type, value_type = to_arrow_type(args[0]), to_arrow_type(args[1])
        if key_type is not None and value_type is not None:
            return pa.map_(key_type, value_type)

    return None


def to_arrow_value(value, arrow_type):
    """ Convert a parameter value to a python object which `pyarrow` accepts for `arrow_type`. """
    if pa.types.is_struct(arrow_type):  # complex
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if pa.types.is_map(arrow_type):
        return [(_to_python(k), _to_python(v)) for k, v in value.items()]
    return _to_python(value)


def _to_python(value):
    """ Recursively convert numpy types to builtin python types. """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return [(_to_python(k), _to_python(v)) for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    return value


def column_schema(enum_classes, prefix="", exclude=()):
    """
    Build ``{column name: arrow type}`` for all members of the enum classes.

    Column names are ``<prefix><enum class name>.<member name>``. Members whose type can not be mapped
    (see `to_arrow_type`) are dropped with a warning.

    Parameters
    ----------
    enum_classes : list of TypedEnum
    prefix : str, default=""
    exclude : iterable of str, optional
        Column names (without `prefix`) to leave out silently, e.g. parameters that are stored explicitly.
    """
    columns = {}
    for enum_class in enum_classes:
        for member in enum_class:
            name = f"{enum_class.__name__}.{member.name}"
            if name in exclude:
                continue

            arrow_type = to_arrow_type(member.dtype)
            if arrow_type is None:
                logger.warning(f"Parameter {name} has the type {member.dtype}, which can not be stored in parquet. Dropping it.")
                continue

            columns[prefix + name] = arrow_type

    return columns


_KEYS = ["run", "event_id", "station_id"]

# Sort order of the tables (the last columns are only tie breakers)
_SORT_KEYS = {
    "event": ["station_id", "run", "event_id"],
    "efield": ["station_id", "run", "event_id", "source", "channel_ids", "ray_tracing_id", "shower_id"],
    "simchannel": ["station_id", "run", "event_id", "channel_id", "ray_tracing_id", "shower_id"],
}
# station<id>_run<run>[_<suffix>][_efield|_simchannel].parquet (groups: station, run, kind). The suffix is
# matched lazily so that a trailing ``_efield`` / ``_simchannel`` is not swallowed by it.
_FILE_PATTERN = re.compile(r"^station(\d+)_run(\d+)(?:_.*?)??(?:_(efield|simchannel))?\.parquet$")


class ParquetReader:
    """
    Reads the files written by `NuRadioReco.modules.io.eventWriterParquet` with polars.

    Files are selected by the station and run in their name, so unselected files are never opened.
    Note that polars returns parquet maps as ``list[struct{key, value}]``.

    Examples
    --------
    >>> reader = ParquetReader("out/", stations=[11], runs=range(100, 110), efields=True)
    >>> df = reader.read()  # events joined with their electric fields
    """

    def __init__(self, path, stations=None, runs=None, sim_channels=False, efields=False, sort=True):
        """
        Parameters
        ----------
        path : str
            Root directory (searched recursively for ``*.parquet``) or a glob pattern (``**`` allowed),
            e.g. ``"out/station11_run*.parquet"`` or ``"data/**/station*_run0001*.parquet"``.
        stations : int or list of int, optional
            Station ids to read. Default: all.
        runs : int or list of int, optional
            Run numbers to read. Default: all.
        sim_channels : bool, default=False
            Also read the sim channel files (``*_simchannel.parquet``) next to the selected event files.
            A warning is logged for each event file without one.
        efields : bool, default=False
            Also read the electric field files (``*_efield.parquet``), analogous to `sim_channels`.
        sort : bool, default=True
            Sort the returned tables, because the row order after reading many files or joining is not
            guaranteed by polars. Order: ``station_id, run, event_id``, for the electric fields and sim channels
            followed by their channel id(s) and ray tracing id (and tie breakers: ``source`` and ``shower_id``).
            This needs the whole table in memory: disable it for very large data sets if the order does not matter.

        Raises
        ------
        ImportError
            If polars is not installed.
        FileNotFoundError
            If no event file matches `path`, `stations` and `runs`.
        """
        try:
            import polars
        except ImportError as e:
            raise ImportError(
                "ParquetReader requires polars. Install it with `pip install polars` "
                "or `pip install -e .[parquet]`.") from e

        self.__pl = polars  # imported here (not at module level) to keep polars optional
        self.__sort = sort
        self.__kinds = ["event"] + (["efield"] if efields else []) + (["simchannel"] if sim_channels else [])
        self.__stations = self.__as_set(stations)
        self.__runs = self.__as_set(runs)

        # A directory is searched recursively, anything else is used as glob pattern
        pattern = os.path.join(path, "**", "*.parquet") if os.path.isdir(path) else path
        self.__files = {kind: [] for kind in self.__kinds}
        self.__event_ids = {}  # event file -> (station id, run)
        for file in sorted(glob.glob(pattern, recursive=True)):
            match = _FILE_PATTERN.match(os.path.basename(file))
            if match is None:
                logger.warning(
                    f"Skipping {file}: the file name does not match the pattern "
                    "station<id>_run<run>[_<suffix>][_efield|_simchannel].parquet")
                continue

            # Station and run are taken from the name: unselected files are never opened.
            # Only event files are collected here, the side files are found next to them.
            station, run, kind = int(match.group(1)), int(match.group(2)), match.group(3) or "event"
            if kind == "event" and self.__selected(station, self.__stations) and self.__selected(run, self.__runs):
                self.__files["event"].append(file)
                self.__event_ids[file] = (station, run)

        if not self.__files["event"]:
            raise FileNotFoundError(f"No matching parquet files found for {path}")

        for kind in self.__kinds[1:]:
            self.__files[kind] = self.__find_side_files(kind, self.__files["event"])

    @property
    def files(self):
        """ dict: kind (``event``, ``efield``, ``simchannel``) -> list of the selected files. """
        return {kind: list(files) for kind, files in self.__files.items()}

    def events(self):
        """
        Only the event table (lazy), without looking at the electric field / sim channel files.

        Use this to select events first (e.g. ``reader.events().filter(...)``) and get the corresponding
        electric fields / sim channels afterwards with `related_tables`.
        """
        return self.__scan_table("event", self.__files["event"], self.__sort)

    def tables(self):
        """
        The tables as separate lazy frames.

        Returns
        -------
        dict of polars.LazyFrame
            Keys are ``event`` and, if requested, ``efield`` and ``simchannel``.
            The tables are linked by the columns ``run``, ``event_id`` and ``station_id``.
        """
        return {
            kind: self.__scan_table(kind, files, self.__sort)
            for kind, files in self.__files.items() if files
        }

    def related_tables(self, events, efields=True, sim_channels=True):
        """
        The electric field and / or sim channel tables of a selection of events (lazy).

        Only the files of the stations and runs which occur in `events` are read, and only the rows of
        these events are kept. The reader does not need to be created with ``efields`` / ``sim_channels``.

        Parameters
        ----------
        events : polars.DataFrame or polars.LazyFrame
            A (filtered) selection of events: needs the columns ``run``, ``event_id`` and ``station_id``,
            e.g. ``reader.events().filter(...)``.
        efields : bool, default=True
            Return the electric field table.
        sim_channels : bool, default=True
            Return the sim channel table.

        Returns
        -------
        dict of polars.LazyFrame
            Keys are ``efield`` and / or ``simchannel`` (a table is missing if no file exists).
            Sorted as the tables of `tables` if the reader sorts.
        """
        keys = events.lazy().select(_KEYS).unique()

        # Only the files of the stations and runs which occur in the selection are needed
        selected = {tuple(row) for row in keys.select("station_id", "run").unique().collect().rows()}
        event_files = [file for file, ids in self.__event_ids.items() if ids in selected]

        tables = {}
        for kind in (["efield"] if efields else []) + (["simchannel"] if sim_channels else []):
            files = self.__find_side_files(kind, event_files)
            if files:
                # Semi join: keep the rows whose event is in the selection (adds no columns). Sort afterwards,
                # so that only the selected rows are sorted.
                table = self.__scan_table(kind, files, sort=False).join(keys, on=_KEYS, how="semi")
                tables[kind] = table.sort(_SORT_KEYS[kind]) if self.__sort else table

        return tables

    def scan(self):
        """
        The event table left-joined with the requested electric field / sim channel table (lazy).

        Events without electric fields (sim channels) are kept with null values. Columns that exist in both
        tables get the suffix ``_efield`` (``_simchannel``) in the joined part.
        Requesting both electric fields and sim channels is ambiguous (one row per combination per event),
        use `tables` instead.
        """
        tables = self.tables()
        if "efield" in tables and "simchannel" in tables:
            raise ValueError("Joining electric fields and sim channels gives all combinations per event. "
                             "Request only one of them or use `tables()`.")

        # Left join: events without electric fields / sim channels are kept (null values)
        frame = tables["event"]
        for kind in ["efield", "simchannel"]:
            if kind in tables:
                frame = frame.join(tables[kind], on=_KEYS, how="left", suffix=f"_{kind}")

                if self.__sort:  # a join does not guarantee the order: sort by the keys of both tables
                    # columns of the joined table which exist in both tables carry the suffix
                    event_columns = set(frame.collect_schema().names())
                    keys = _SORT_KEYS["event"] + [
                        f"{key}_{kind}" if f"{key}_{kind}" in event_columns else key
                        for key in _SORT_KEYS[kind][len(_SORT_KEYS["event"]):]]
                    frame = frame.sort(keys)

        return frame

    def read(self):
        """ Like `scan` but returns the collected `polars.DataFrame`. """
        return self.scan().collect()

    def __scan_table(self, kind, files, sort):
        """ Lazily concatenate the files of one table (sorted if `sort`). """
        pl = self.__pl
        # "diagonal_relaxed": tolerate files written with different parameter sets
        table = pl.concat([pl.scan_parquet(file) for file in files], how="diagonal_relaxed")
        return table.sort(_SORT_KEYS[kind]) if sort else table

    @staticmethod
    def __find_side_files(kind, event_files):
        """
        The ``efield`` / ``simchannel`` files next to the event files (independent of the glob pattern).
        A warning is logged for each event file without one.
        """
        files = []
        for event_file in event_files:
            side_file = f"{event_file[:-len('.parquet')]}_{kind}.parquet"
            if os.path.exists(side_file):
                files.append(side_file)
            else:
                logger.warning(f"No {kind} file found for {event_file} (expected {side_file}).")

        return files

    @staticmethod
    def __as_set(values):
        """ Convert an int or an iterable of ints to a set (None stays None, i.e. no selection). """
        if values is None:
            return None

        return {int(values)} if isinstance(values, (int, float)) else {int(value) for value in values}

    @staticmethod
    def __selected(value, selection):
        """ True if `value` is in the `selection` (None selects everything). """
        return selection is None or value in selection
