"""
Helpers to map typed parameter enums (see `NuRadioReco.framework.parameters`) to pyarrow (library to write parquet files).

The declared ``dtype`` of an enum member determines the column type. Parameterised types such as
``dict[int, dict[int, float]]`` or ``list[float]`` are mapped to native arrow ``map``/``list`` types.
Parameters whose type is not specific enough to define a schema (``Any``, bare ``dict``/``list``,
unknown classes) are dropped with a warning, nothing is pickled.

`ParquetReader` reads the files written by `NuRadioReco.modules.io.eventWriterParquet` with polars:

* The input is a root directory (searched recursively) or a glob pattern.
* Stations and runs are selected via the file names (which contain the range of runs of a file),
  so unselected files are never opened.
* The electric field and sim channel files can be read in addition.
* `ParquetReader.read` / `ParquetReader.scan` return the event table left-joined with the requested
  table (eager / lazy), `ParquetReader.tables` returns the tables separately.
* `ParquetReader.events` returns only the event table. `ParquetReader.related_tables` then returns the
  electric fields / sim channels of a selection of these events.
* `ParquetReader.columns` lists the columns (by default only those with data). Command line tool
  (``nuradio-parquet`` if installed, else ``python -m NuRadioReco.utilities.parquet_utilities``):
  ``nuradio-parquet columns <path> [--class channelParameters] [--match snr]`` and
  ``nuradio-parquet hist <column> <path> [--channels 0 1] [--key peak_amplitude] [-o hist.png]``
* Polars is optional (only needed by the reader). It returns parquet maps as ``list[struct{key, value}]``.
"""
import glob
import logging
import os
import re
import sys
import typing

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:  # optional: only the reader needs polars (the writer imports this module as well)
    import polars as pl
except ImportError:
    pl = None

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
    Build ``{column name: (enum member, arrow type)}`` for all members of the enum classes.

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

            columns[prefix + name] = (member, arrow_type)

    return columns


def nur_to_parquet(filename, output_dir):
    """ Convert a `.nur` file to parquet files (station / channel parameters incl. the RNO-G ones). """
    # Imported here: eventWriterParquet imports this module
    from NuRadioReco.framework import parameters
    from NuRadioReco.modules.io.eventReader import eventReader
    from NuRadioReco.modules.io.eventWriterParquet import eventWriterParquet

    reader = eventReader()
    reader.begin(filename, read_detector=False)

    writer = eventWriterParquet()
    writer.begin(
        output_dir,
        station_enums=[parameters.stationParameters, parameters.stationParametersRNOG],
        channel_enums=[parameters.channelParameters, parameters.channelParametersRNOG],
        comment=f"converted from {filename}",
    )
    for evt in reader.run():
        writer.run(evt, evt.get_station())

    print(f"Wrote {writer.end()} events to {output_dir}")


_KEYS = ["run", "event_id", "station_id"]

# Sort order (their values) of the tables (the last columns are only tie breakers)
_SORT_KEYS = {
    "event": ["station_id", "run", "event_id"],
    "efield": ["station_id", "run", "event_id", "source", "channel_ids", "ray_tracing_id", "shower_id"],
    "simchannel": ["station_id", "run", "event_id", "channel_id", "ray_tracing_id", "shower_id"],
}
# station<id>_run<run>[-<last run>][_<suffix>][_efield|_simchannel].parquet (groups: station, first run, last run, kind).
# The suffix is matched lazily so that a trailing ``_efield`` / ``_simchannel`` is not swallowed by it.
_FILE_PATTERN = re.compile(r"^station(\d+)_run(\d+)(?:-(\d+))?(?:_.*?)??(?:_(efield|simchannel))?\.parquet$")


class ParquetReader:
    """
    Reads the files written by `NuRadioReco.modules.io.eventWriterParquet` with polars.

    Files are selected by the station and run(s) in their name, so unselected files are never opened.
    If a file holds several runs, only the rows of the selected runs are read from it.
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
        path : str or list of str
            A root directory (searched recursively for ``*.parquet``), a file or a glob pattern (``**`` allowed),
            e.g. ``"out/station11_run*.parquet"`` or ``"data/**/station*_run0001*.parquet"``.
            A list combines several of these, e.g. ``["out/a", "out/b/station11_run0001*.parquet"]``.
            Only the event files are collected, the electric field and sim channel files are found next to them.
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
            If no event file matches `path`, `stations` and `runs`, or an entry of a list `path` matches nothing.
        """
        if pl is None:
            raise ImportError(
                "ParquetReader requires polars. Install it with `pip install polars` "
                "or `pip install -e .[parquet]`.")

        self.__sort = sort
        self.__kinds = ["event"] + (["efield"] if efields else []) + (["simchannel"] if sim_channels else [])
        self.__stations = self.__as_set(stations)
        self.__runs = self.__as_set(runs)

        self.__files = {kind: [] for kind in self.__kinds}
        self.__event_ids = {}  # event file -> (station id, first run, last run)
        for file in self.__find_files(path):
            match = _FILE_PATTERN.match(os.path.basename(file))
            if match is None:
                logger.warning(
                    f"Skipping {file}: the file name does not match the pattern "
                    "station<id>_run<run>[-<last run>][_<suffix>][_efield|_simchannel].parquet")
                continue

            # Station and run are taken from the name: unselected files are never opened.
            # Only event files are collected here, the side files are found next to them.
            station, first, kind = int(match.group(1)), int(match.group(2)), match.group(4) or "event"
            last = int(match.group(3) or first)
            if kind == "event" and self.__selected(station, self.__stations) and self.__has_run(first, last, self.__runs):
                self.__files["event"].append(file)
                self.__event_ids[file] = (station, first, last)

        if not self.__files["event"]:
            raise FileNotFoundError(f"No matching parquet files found for {path}")

        # Files with several runs can hold runs which are not selected
        self.__filter_runs = self.__runs is not None and any(
            first != last for _, first, last in self.__event_ids.values())

        for kind in self.__kinds[1:]:
            self.__files[kind] = self.__find_side_files(kind, self.__files["event"])

    @property
    def files(self):
        """ dict: kind (``event``, ``efield``, ``simchannel``) -> list of the selected files. """
        return {kind: list(files) for kind, files in self.__files.items()}

    def columns(self, kind="event", non_null=True, parameter_classes=None, pattern=None):
        """
        The columns of a table and their (polars) data types.

        Parameters
        ----------
        kind : {"event", "efield", "simchannel"}, default="event"
            The table.
        non_null : bool, default=True
            Only columns with at least one non-null value in the selected files (nested: a list of nulls is null).
            Uses the parquet statistics, i.e., does not read the data.
        parameter_classes : str or list of str, optional
            Only columns of these parameter classes (the prefix before the first ``.``), e.g. ``"channelParameters"``.
        pattern : str, optional
            Only columns whose name matches this regular expression (`re.search`, case insensitive).

        Returns
        -------
        dict
            Column name -> polars data type.
        """
        # The efield / simchannel files are only collected if requested in `__init__`: look them up otherwise
        files = self.__files[kind] if kind in self.__kinds else self.__find_side_files(kind, self.__files["event"])
        if not files:
            raise FileNotFoundError(f"No {kind} files found.")

        if isinstance(parameter_classes, str):
            parameter_classes = [parameter_classes]

        # Union of the columns of all files (only the file footers are read)
        schema = self.__scan_table(kind, files, sort=False).collect_schema()

        # Name filters first, so that the statistics are only checked for the remaining columns.
        # Columns without a "." (keys like `run`, `channel_ids`) belong to no parameter class.
        columns = {
            name: dtype for name, dtype in schema.items()
            if (parameter_classes is None or ("." in name and name.partition(".")[0] in parameter_classes))
            and (pattern is None or re.search(pattern, name, re.IGNORECASE))}

        if non_null:  # keeps the order of the schema
            filled = self.__non_null_columns(files, columns)
            columns = {name: dtype for name, dtype in columns.items() if name in filled}

        return columns

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
            Keys are ``efield`` and / or ``simchannel``. A table is missing (and an error is logged) if none of its
            files exists. Sorted as the tables of `tables` if the reader sorts.
        """
        # Collected once: a lazy frame would evaluate the selection again in each join below (few columns, small)
        keys = events.lazy().select(_KEYS).unique().collect()

        # Only the files of the stations and runs which occur in the selection are needed
        selected = keys.select("station_id", "run").unique().rows()
        event_files = [
            file for file, (station, first, last) in self.__event_ids.items()
            if any(s == station and first <= run <= last for s, run in selected)]

        if not event_files:
            logger.warning("The selection contains no events: no electric fields / sim channels to read.")
            return {}

        tables = {}
        for kind, requested in [("efield", efields), ("simchannel", sim_channels)]:
            if not requested:
                continue

            files = self.__find_side_files(kind, event_files)
            if not files:
                logger.error(f"No {kind} files found for the {len(event_files)} event file(s) of the selection.")
                continue

            # Semi join: keep the rows whose event is in the selection (adds no columns). Sort afterwards,
            # so that only the selected rows are sorted.
            table = self.__scan_table(kind, files, sort=False).join(keys.lazy(), on=_KEYS, how="semi")
            tables[kind] = table.sort(_SORT_KEYS[kind]) if self.__sort else table

        return tables

    def scan(self):
        """
        The event table left-joined with the requested electric field / sim channel table (lazy).

        Requesting both electric fields and sim channels is ambiguous (one row per combination per event),
        use `tables` instead.

        Returns
        -------
        polars.LazyFrame
            Without electric fields / sim channels: the event table, one row per event.
            Otherwise one row per (event, electric field) or (event, sim channel): the event columns are repeated
            for each electric field (sim channel) of the event. Events without electric fields (or sim channels) are kept 
            (single row), with null values in the joined columns. Joined columns which also exist in the event table 
            (e.g. ``channel_ids``) get the suffix ``_efield`` (``_simchannel``). Sorted by the sort keys of both tables 
            if the reader sorts.

        Raises
        ------
        ValueError
            If the reader was created with both ``efields`` and ``sim_channels`` and files of both exist.

        Examples
        --------
        >>> frame = ParquetReader("out/", efields=True).scan()
        >>> frame.filter(pl.col("event_id") == 42).collect()  # one row per electric field of event 42
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
        # "diagonal": tolerate files written with different parameter sets but not dtyoes of common columns
        table = pl.concat([pl.scan_parquet(file) for file in files], how="diagonal")
        if self.__filter_runs:
            table = table.filter(pl.col("run").is_in(sorted(self.__runs)))

        return table.sort(_SORT_KEYS[kind]) if sort else table

    @staticmethod
    def __non_null_columns(files, names):
        """
        The columns of `names` with at least one non-null value, from the parquet statistics (no data is read).

        Nested columns are stored as several leaf columns (e.g. ``<name>.list.element``): a column has data if
        one of its leaves has fewer nulls than values. Leaves without statistics count as having data.
        """
        # Column names contain dots as well: assign each leaf path to the longest matching name
        names = sorted(names, key=len, reverse=True)
        filled, no_statistics = set(), set()
        for file in files:
            metadata = pq.read_metadata(file)  # only the footer
            for i_group in range(metadata.num_row_groups):  # statistics are stored per row group and leaf
                group = metadata.row_group(i_group)
                for i_leaf in range(group.num_columns):
                    leaf = group.column(i_leaf)
                    statistics = leaf.statistics
                    if statistics is None or not statistics.has_null_count:
                        no_statistics.add(file)  # unknown: count as data
                    elif statistics.null_count == leaf.num_values:
                        continue  # only nulls in this row group

                    # A leaf path is e.g. "channelParameters.SNR.list.element.key_value.value"
                    path = leaf.path_in_schema
                    filled.add(next((name for name in names if path == name or path.startswith(name + ".")), None))

        if no_statistics:
            logger.warning(
                f"{len(no_statistics)} file(s) without null counts (written without statistics), "
                f"their columns are listed as having data, e.g. {sorted(no_statistics)[0]}")

        filled.discard(None)  # leaves of columns which are not in `names`
        return filled

    @staticmethod
    def __find_files(path):
        """
        The parquet files of `path`: a string or each entry of a list is a directory (searched recursively),
        a file or a glob pattern. Raises `FileNotFoundError` if an entry of a list matches nothing.
        """
        entries = [path] if isinstance(path, (str, os.PathLike)) else list(path)

        files = set()
        for entry in map(os.fspath, entries):
            pattern = os.path.join(entry, "**", "*.parquet") if os.path.isdir(entry) else entry
            matches = glob.glob(pattern, recursive=True)
            if not matches and len(entries) > 1:
                raise FileNotFoundError(f"{entry} matches no file.")

            files.update(matches)

        return sorted(files)

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
    def __has_run(first, last, selection):
        """ True if one of the runs `first` to `last` is in the `selection` (None selects everything). """
        return selection is None or any(first <= run <= last for run in selection)

    @staticmethod
    def __selected(value, selection):
        """ True if `value` is in the `selection` (None selects everything). """
        return selection is None or value in selection


def _print_columns(args):
    """ The ``columns`` command: summary (stderr), then the columns as list or table (stdout). """
    reader = ParquetReader(args.path, stations=args.stations, runs=args.runs, sort=False,
                           efields=args.table == "efield", sim_channels=args.table == "simchannel")
    result = reader.columns(
        kind=args.table, non_null=not args.all, parameter_classes=args.parameter_classes, pattern=args.match)
    table = reader.tables()[args.table]

    # Summary on stderr: stdout stays a plain list of columns (e.g. for grep).
    # The row count comes from the file footers, stations / runs read only these two columns.
    keys = table.select(pl.len(), pl.col("station_id").unique().sort().implode(), pl.col("run").n_unique()).collect()
    n_rows, stations, n_runs = keys.row(0)
    print(f"{len(reader.files[args.table])} files, {n_rows} rows, stations {stations}, {n_runs} runs, "
          f"{len(result)} of {len(table.collect_schema())} columns", file=sys.stderr)

    rows = [[name] + ([str(dtype)] if args.types else []) for name, dtype in result.items()]
    if args.stats:
        # Lists are flattened, NaN / inf are counted separately. Maps and strings have no statistics.
        # All statistics are computed in one query (one pass over the data), named "<column>|<statistic>".
        exprs = []
        for name, dtype in result.items():
            expr = pl.col(name)
            while isinstance(dtype, pl.List):  # e.g. channel parameters: one value per channel
                expr, dtype = expr.explode(empty_as_null=False), dtype.inner
            if dtype.is_numeric() or dtype == pl.Boolean:
                expr = expr.cast(pl.Float64)
                valid = pl.when(expr.is_finite()).then(expr)  # NaN / inf -> null (ignored by the aggregations)
                exprs += [
                    valid.count().alias(f"{name}|n"), (~expr.is_finite()).sum().alias(f"{name}|nan/inf"),
                    valid.min().alias(f"{name}|min"), valid.max().alias(f"{name}|max"),
                    valid.mean().alias(f"{name}|mean"), valid.std().alias(f"{name}|std")]

        statistics = table.select(exprs).collect().row(0, named=True) if exprs else {}
        # Columns without statistics (maps, strings) get "-"
        rows = [row + [_format(statistics.get(f"{row[0]}|{stat}")) for stat in _STATISTICS] for row in rows]

    if args.stats or args.types:
        header = ["column"] + (["type"] if args.types else []) + (_STATISTICS if args.stats else [])
        _print_table(header, rows, n_text_columns=2 if args.types else 1)
    else:  # plain list (e.g. for grep)
        print(*(row[0] for row in rows), sep="\n")


_STATISTICS = ["n", "nan/inf", "min", "max", "mean", "std"]


def _print_table(header, rows, n_text_columns=1):
    """ Print a table with a border and grid lines. The first `n_text_columns` are left aligned, the others right. """
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]

    def line(left, middle, right, fill="─"):  # horizontal line with the given corner / crossing characters
        return left + middle.join(fill * (width + 2) for width in widths) + right

    def cells(row):
        return "│" + "│".join(
            f" {value.ljust(width) if i < n_text_columns else value.rjust(width)} "
            for i, (value, width) in enumerate(zip(row, widths))) + "│"

    print(line("┌", "┬", "┐"))
    print(cells(header))
    print(line("╞", "╪", "╡", fill="═"))
    for i, row in enumerate(rows):
        if i:
            print(line("├", "┼", "┤"))
        print(cells(row))
    print(line("└", "┴", "┘"))


def _format(value):
    """ Integers (e.g. counts, ids) without decimals, other numbers with 4 significant digits, None as "-". """
    if value is None:
        return "-"
    return str(int(value)) if float(value).is_integer() and abs(value) < 1e9 else f"{value:.4g}"


def _histogram_values(reader, kind, column, channels=None, key=None):
    """
    The values of `column` for a histogram as `polars.DataFrame` with the column ``value``
    (and ``channel_id`` for channel parameters).

    Lists are flattened. Channel parameters are lists aligned with ``channel_ids``: `channels` selects channel ids.
    Maps (e.g. ``channelParameters.SNR``) need the `key` to histogram.
    """
    table = reader.tables()[kind]
    schema = table.collect_schema()

    # Full name or name without the parameter class, e.g. "coherent_snr"
    candidates = [name for name in schema if column in (name, name.partition(".")[2])]
    if len(candidates) != 1:
        raise ValueError(f"'{column}' matches {candidates or 'no column'}. See the `columns` command.")
    name = candidates[0]

    is_channel_parameter = kind == "event" and name.startswith("channelParameters")
    if is_channel_parameter:
        # Explode the channel ids together with the values: one row per (event, channel)
        frame = table.select(pl.col("channel_ids").alias("channel_id"), pl.col(name).alias("value"))
        frame = frame.explode("channel_id", "value", empty_as_null=False)
        if channels is not None:
            frame = frame.filter(pl.col("channel_id").is_in(channels))
    elif channels is not None:
        raise ValueError(f"--channels only applies to channel parameters, not to {name}.")
    else:
        frame = table.select(pl.col(name).alias("value"))

    # Flatten remaining lists, e.g. one value per target band of `spectral_energy_ratios` (all elements are kept)
    dtype = frame.collect_schema()["value"]
    while isinstance(dtype, pl.List):
        frame, dtype = frame.explode("value", empty_as_null=False), dtype.inner

    if isinstance(dtype, pl.Struct):  # map: list of {key, value}, exploded to one row per entry
        if key is None:
            keys = frame.select(pl.col("value").struct.field("key").unique().sort()).collect().to_series()
            raise ValueError(f"{name} is a map, choose a key with --key: {keys.drop_nulls().to_list()}")
        frame = frame.filter(pl.col("value").struct.field("key") == key)
        frame = frame.with_columns(pl.col("value").struct.field("value"))
    elif key is not None:
        raise ValueError(f"--key only applies to maps, {name} has the type {dtype}.")

    return name, frame.drop_nulls("value").collect()


def _format_channels(channels):
    """ [0, 1, 2, 3, 5] -> '0-3, 5' """
    channels = sorted(set(channels))
    groups = []
    for channel in channels:
        if groups and channel == groups[-1][-1] + 1:
            groups[-1][-1] = channel
        else:
            groups.append([channel, channel])

    return ", ".join(str(first) if first == last else f"{first}-{last}" for first, last in groups)


def _plot_histogram(args):
    """ The ``hist`` command: histogram of one column (all channels, selected channels or one per channel). """
    import matplotlib.pyplot as plt

    reader = ParquetReader(args.path, stations=args.stations, runs=args.runs, sort=False,
                           efields=args.table == "efield", sim_channels=args.table == "simchannel")
    name, frame = _histogram_values(reader, args.table, args.column, args.channels, args.key)

    values = frame["value"].cast(float).to_numpy()
    if args.log:  # log bins need positive values
        values = values[values > 0]
    if not len(values):
        raise ValueError(f"No values to plot for {name}.")

    # Same bins for all channels (--per-channel), so that the histograms are comparable
    value_range = args.range or (values.min(), values.max())
    bins = (np.geomspace if args.log else np.linspace)(*value_range, args.bins + 1)

    fig, ax = plt.subplots()
    if args.per_channel and "channel_id" in frame.columns:  # only channel parameters have a channel id
        for (channel_id,), group in frame.group_by("channel_id", maintain_order=True):
            ax.hist(group["value"].cast(float).to_numpy(), bins=bins, histtype="step", label=f"channel {channel_id}")
        ax.legend(fontsize="small", ncols=2)
    else:
        ax.hist(values, bins=bins, histtype="step", label=f"{len(values)} entries")
        ax.legend(title=f"channels {_format_channels(args.channels)}" if args.channels else None)

    ax.set_xlabel(name + (f"\n[{args.key}]" if args.key else ""))
    ax.set_ylabel("entries")
    if args.log:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")
    fig.tight_layout()

    if args.output is None:
        plt.show()
    else:
        fig.savefig(args.output)
        print(f"Saved {args.output}")


def main():
    """ Command line interface, see ``python -m NuRadioReco.utilities.parquet_utilities --help``. """
    import argparse

    parser = argparse.ArgumentParser(description="Inspect parquet files written by eventWriterParquet.")
    commands = parser.add_subparsers(dest="command", required=True)

    # Options of all commands. The paths are the last positional argument of each command.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--table", choices=list(_SORT_KEYS), default="event")
    common.add_argument("--stations", nargs="+", type=int, default=None)
    common.add_argument("--runs", nargs="+", type=int, default=None)

    columns = commands.add_parser(
        "columns", parents=[common], help="List the columns (by default only those with data).")
    columns.add_argument("path", nargs="+", help="Directories, files or glob patterns (see `ParquetReader`).")
    columns.add_argument("--class", dest="parameter_classes", nargs="+", metavar="CLASS",
                         help="Only these parameter classes, e.g. channelParameters stationParametersRNOG.")
    columns.add_argument("--match", default=None, metavar="REGEX",
                         help="Only columns matching this regular expression (case insensitive).")
    columns.add_argument("--all", action="store_true", help="Also list columns without any (non-null) value.")
    columns.add_argument("--types", action="store_true", help="Also print the data types.")
    columns.add_argument("--stats", action="store_true",
                         help="Also print the number of (finite) values, of NaN / inf, min, max, mean and std (reads the data, lists are flattened).")

    hist = commands.add_parser("hist", parents=[common], help="Plot the histogram of a column.")
    hist.add_argument("column", help="Column name, the parameter class can be omitted (e.g. coherent_snr).")
    hist.add_argument("path", nargs="+", help="Directories, files or glob patterns (see `ParquetReader`).")
    hist.add_argument("--channels", nargs="+", type=int, default=None,
                      help="Channel ids (channel parameters only). Default: all channels.")
    hist.add_argument("--per-channel", action="store_true", help="One histogram per channel (channel parameters).")
    hist.add_argument("--key", default=None, help="Key of a map column, e.g. peak_amplitude for SNR.")
    hist.add_argument("--bins", type=int, default=100)
    hist.add_argument("--range", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"))
    hist.add_argument("--log", action="store_true", help="Logarithmic bins and x-axis.")
    hist.add_argument("--logy", action="store_true", help="Logarithmic y-axis.")
    hist.add_argument("-o", "--output", default=None, help="Save the figure to this file instead of showing it.")

    args = parser.parse_args()

    try:
        if args.command == "columns":
            _print_columns(args)
        elif args.command == "hist":
            _plot_histogram(args)
    except (ValueError, FileNotFoundError) as e:  # user errors: no traceback
        parser.exit(1, f"error: {e}\n")


if __name__ == "__main__":
    main()
