"""
Read the parquet files written by `NuRadioReco.modules.io.eventWriter` as tables for analyses, with polars.

`NuRadioReco.modules.io.eventReader` rebuilds complete events instead; both share the file handling of
`NuRadioReco.utilities.parquet_utilities`.

* The input is a root directory (searched recursively), a file or a glob pattern; files are recognised by their
  metadata, an event file is found next to its station file.
* Stations and runs are selected by filtering the ``station_id`` and ``run`` columns; files (and row groups)
  whose statistics exclude the selection are not read.
* Events must be unique across the selected files: several processings of the same runs (e.g. different file
  names in one directory) raise an error, select one of them with a glob pattern.
* `AnalysisReader.read` / `AnalysisReader.scan` return the station rows (one per event and station) joined with the
  event files (eager / lazy), `AnalysisReader.tables` returns both tables separately. The traces are only read
  on request (``traces=True``).
* `AnalysisReader.columns` lists the columns (by default only those with data). Command line tool
  (``nuradio-analyse`` if installed, else ``python -m NuRadioReco.utilities.analysis_reader``):
  ``nuradio-analyse columns <path> [--class channelParameters] [--match snr]`` and
  ``nuradio-analyse hist <column> <path> [--channels 0 1] [--key peak_amplitude] [-o hist.png]``,
  ``nuradio-analyse convert <.nur files> [--output-dir DIR]`` (see `NuRadioReco.modules.io.nur_to_parquet`)
* Needs polars >= 2 (``pip install nuradiomc[analysis]``). It returns parquet maps as ``pl.Map`` (polars < 2
  returned them as ``list[struct{key, value}]``), e.g.
  ``pl.col("channelParameters.SNR").list.eval(pl.element().map.get("peak_amplitude"))``.
"""

import logging
import re
import sys

import numpy as np
import pyarrow.parquet as pq

try:  # optional dependency: an error is raised when the reader is used without it
    import polars as pl
except ImportError:
    pl = None

from NuRadioReco.utilities import parquet_utilities as pu
from NuRadioReco.utilities import units

logger = logging.getLogger("NuRadioReco.analysis_reader")

_MIN_POLARS_MAJOR = 2  # the reader relies on parquet maps being read as `pl.Map` (new in polars 2)


class AnalysisReader:
    """
    Reads the files written by `NuRadioReco.modules.io.eventWriter` with polars.

    Stations and runs are selected by filtering the rows; files whose row group statistics exclude the selection
    are not read (see `pu.find_files`).
    Each event must occur only once in the selected files (see ``check_unique``).
    Requires polars >= 2, which returns parquet maps as ``pl.Map``.

    Examples
    --------
    >>> reader = AnalysisReader("out/", stations=[11], runs=range(100, 110))
    >>> df = reader.read()  # one row per event and station, with the event level columns (``event.*``)
    """

    def __init__(self, path, stations=None, runs=None, join_events=True, traces=False, sort=True, check_unique=True):
        """
        Parameters
        ----------
        path : str or list of str
            A root directory (searched recursively for ``*.parquet``), a file or a glob pattern (``**`` allowed),
            e.g. ``"out/run1000.parquet"`` or ``"data/**/run10*.parquet"``.
            A list combines several of these, e.g. ``["out/a", "out/b/run1000*.parquet"]``.
            The event files are also found next to the selected station files (see `pu.find_files`).
        stations : int or list of int, optional
            Station ids to read. Default: all.
        runs : int or list of int, optional
            Run numbers to read. Default: all.
        join_events : bool, default=True
            `scan` / `read` left-join each station file with its event file (``<name>_event.parquet``)
            on (``run``, ``event_id``). Their columns get the prefix ``event.``, as in files written with
            ``event_file=False`` (which have no event files).
        traces : bool, default=False
            Also read the trace columns (``*.trace``). They are read eagerly with pyarrow
            (polars' parquet reader is much slower for nested lists of floats).
        sort : bool, default=True
            Sort the returned tables (by ``station_id, run, event_id``; event table: ``run, event_id``), because the
            row order after reading many files is not guaranteed by polars. Rows with the same keys (duplicates,
            read with ``check_unique=False``) keep their order in the file. This needs the whole table in memory:
            disable it for very large data sets if the order does not matter.
        check_unique : bool, default=True
            Check that each event (``run``, ``event_id``, ``station_id``) occurs only once in the selected station
            files (``run``, ``event_id`` in the event files), e.g. to catch overlapping files or several processing
            variants of the same runs. Done once, when the first table is requested (`tables`, `scan`, `read`),
            not by `columns`. Reads only the key columns.

        Raises
        ------
        ImportError
            If polars is not installed or older than version 2.
        FileNotFoundError
            If no station file matches `path`, `stations` and `runs`, or an entry of a list `path` matches nothing.
        """
        if pl is None or int(pl.__version__.split(".")[0]) < _MIN_POLARS_MAJOR:
            found = "not installed" if pl is None else f"found {pl.__version__}"
            raise ImportError(
                f"AnalysisReader requires polars >= {_MIN_POLARS_MAJOR} ({found}). Install it with "
                f"`pip install 'polars>={_MIN_POLARS_MAJOR}'` or `pip install -e .[analysis]`."
            )

        # Files and selected rows
        self.__files = pu.find_files(path, stations, runs)
        if not self.__files["station"]:
            raise FileNotFoundError(f"No matching parquet files found for {path}")

        self.__selection = {"station_id": pu._as_set(stations), "run": pu._as_set(runs)}

        # Read options
        self.__join_events = join_events
        self.__traces = traces
        self.__sort = sort
        self.__check_pending = {"station", "event"} if check_unique else set()  # see `__check_unique`

    @property
    def files(self):
        """dict: kind (``station``, ``event``) -> list of the selected files."""
        return {kind: list(files) for kind, files in self.__files.items()}

    def columns(self, kind="station", non_null=True, parameter_classes=None, pattern=None):
        """
        The columns of the station or event files and their (polars) data types.

        Parameters
        ----------
        kind : {"station", "event"}, default="station"
            The files.
        non_null : bool, default=True
            Only columns with at least one non-null value in the selected files (nested: a list of nulls is null).
            Uses the parquet statistics, i.e., does not read the data.
        parameter_classes : str or list of str, optional
            Only parameter columns of these classes, e.g. ``"channelParameters"`` (also matches the prefixed
            columns, e.g. ``sim_channel.channelParameters.SNR``).
        pattern : str, optional
            Only columns whose name matches this regular expression (`re.search`, case insensitive).

        Returns
        -------
        dict
            Column name -> polars data type. Trace columns only if the reader was created with ``traces=True``.
        """
        files = self.files[kind]
        if not files:
            raise FileNotFoundError(f"No {kind} files found.")

        if isinstance(parameter_classes, str):
            parameter_classes = [parameter_classes]

        # Union of the columns of all files (only the file footers are read).
        # Name filters first, so that the statistics are only checked for the remaining columns.
        schema = self.__scan_files(files, kind).collect_schema()
        columns = {
            name: dtype
            for name, dtype in schema.items()
            if (self.__traces or not pu.is_trace_column(name))
            and (parameter_classes is None or any(part in parameter_classes for part in name.split(".")[:-1]))
            and (pattern is None or re.search(pattern, name, re.IGNORECASE))
        }

        if non_null:  # keeps the order of the schema
            filled = self.__non_null_columns(files, columns)
            columns = {name: dtype for name, dtype in columns.items() if name in filled}

        return columns

    def tables(self):
        """
        The station and event tables as separate lazy frames (not joined).

        Returns
        -------
        dict of polars.LazyFrame
            ``station`` (one row per event and station) and, if there are event files, ``event`` (one row per event;
            with a station selection only the events with a selected station). They are linked by the columns
            ``run`` and ``event_id``.

        Raises
        ------
        ValueError
            If ``check_unique`` (see `__init__`) and an event occurs more than once.
        """
        tables = {}

        for kind, files in self.__files.items():
            if files:
                self.__check_unique(kind)
                tables[kind] = self.__read(files, kind)

        if "event" in tables and self.__selection["station_id"] is not None:
            keys = tables["station"].select(pu._KEYS["event"]).unique()
            tables["event"] = tables["event"].join(keys, on=pu._KEYS["event"], how="semi")

        return {kind: self.__sorted(table, kind) for kind, table in tables.items()}

    def scan(self):
        """
        The station table, left-joined with the event table if ``join_events`` (lazy).

        Each station file is joined with its event file (same base name, see `pu.event_file_of`).

        Returns
        -------
        polars.LazyFrame
            One row per event and station. The event level columns have the prefix ``event.`` (also in files
            written with ``event_file=False``); they are null for events missing in the event files.
        """
        tables = self.tables()  # checks that the events are unique
        if not self.__join_events or "event" not in tables:
            return tables["station"]

        single = [file for file in self.__files["station"] if pu._file_metadata(file).get("layout") == "single"]
        if single:
            logger.warning(
                f"Not joining the event files: {len(single)} station file(s) already contain the event "
                f"columns (written with event_file=False), e.g. {single[0]}"
            )
            return tables["station"]

        event_files = set(self.__files["event"])
        frames = []

        for station_file in self.__files["station"]:
            frame = self.__read([station_file], "station")
            event_file = pu.event_file_of(station_file)
            if event_file in event_files:
                events = self.__read([event_file], "event")
                events = events.rename(
                    {name: pu.EVENT_PREFIX + name for name in events.collect_schema().names() if name not in pu._KEYS["event"]}
                )
                frame = frame.join(events, on=pu._KEYS["event"], how="left", maintain_order="left")
            frames.append(frame)

        return self.__sorted(pl.concat(frames, how="diagonal"), "station")

    def read(self):
        """Like `scan` but returns the collected `polars.DataFrame`."""
        return self.scan().collect()

    def __sorted(self, table, kind):
        """`table` sorted by the keys of `kind` if requested (rows with equal keys keep their order)."""
        return table.sort(pu._SORT_KEYS[kind], maintain_order=True) if self.__sort else table

    def __read(self, files, kind):
        """The rows of the selected stations and runs of `files` of `kind` (lazy), with the traces if requested."""
        frames = []

        for file in files:
            names = pq.read_schema(file).names
            trace_columns = [name for name in names if pu.is_trace_column(name)]
            if not trace_columns:
                frames.append(pl.scan_parquet(file))
            elif not self.__traces:
                frames.append(pl.scan_parquet(file).drop(trace_columns))
            else:  # traces via pyarrow, the other columns with polars (same row order)
                others = pl.read_parquet(file, columns=[name for name in names if name not in trace_columns])
                frames.append(
                    pl.concat([others, pl.from_arrow(pq.read_table(file, columns=trace_columns))], how="horizontal").lazy()
                )

        # "diagonal": tolerate files written with different parameter sets (but not different types)
        return self.__filter(pl.concat(frames, how="diagonal"), kind)

    def __filter(self, table, kind):
        """The rows of `table` with the selected stations (station files only) and runs."""
        for column, values in self.__selection.items():
            if values is not None and column in pu._KEYS[kind]:
                table = table.filter(pl.col(column).is_in(sorted(values)))

        return table

    def __scan_files(self, files, kind, file_column=None):
        """All columns of `files` (lazy, with the file path as `file_column` if given), selected rows only."""
        table = pl.concat([pl.scan_parquet(file, include_file_paths=file_column) for file in files], how="diagonal")

        return self.__filter(table, kind)

    def __check_unique(self, kind):
        """
        Raise a `ValueError` if an event occurs more than once in the files of `kind` (reads only the key columns).
        Runs only once per kind (if ``check_unique``), eagerly when the lazy table is built.
        """
        if kind not in self.__check_pending:
            return

        keys = pu._KEYS[kind]
        duplicates = (
            self.__scan_files(self.__files[kind], kind, file_column="file")
            .group_by(keys)
            .agg(pl.len(), pl.col("file").unique().sort())
            .filter(pl.col("len") > 1)
            .sort(pu._SORT_KEYS[kind])
            .collect()
        )

        if len(duplicates):
            # Usually whole files overlap: list the (sets of) files which share events
            files = duplicates.group_by("file").len().sort("len", descending=True)
            overlaps = "\n".join(f"  {n} events in: {', '.join(names)}" for names, n in files.head(5).rows())
            raise ValueError(
                f"{len(duplicates)} events ({', '.join(keys)}) occur more than once in the {kind} files, "
                f"e.g. {duplicates.row(0)[:len(keys)]}:\n{overlaps}\n"
                "Select the files with a more specific path / glob pattern (or pass check_unique=False)."
            )

        self.__check_pending.discard(kind)  # only after passing: a failed check raises again on the next call

    @staticmethod
    def calpulsers(events, select=None, time_range=None):
        """
        Flag (and optionally select or remove) possible calpulser events.

        Calpulsers fire shortly after the PPS (full second): an event is flagged if its station time,
        modulo one second, lies within `time_range`. Other events which trigger in this window are
        flagged as well (false positives).

        Parameters
        ----------
        events : polars.LazyFrame or polars.DataFrame
            Station table with the column ``station_time_jd2``, e.g. ``reader.scan()``.
        select : bool, optional
            None: only add the boolean column ``is_calpulser``. True: keep only the calpulsers.
            False: remove them. Events without station time (``is_calpulser`` null) are removed by both filters.
        time_range : tuple of float, optional
            Time window after the full second (in NuRadio units, bounds included).
            Default: ``(1 * units.microsecond, 5 * units.microsecond)``.

        Returns
        -------
        polars.LazyFrame or polars.DataFrame
            `events` with the column ``is_calpulser`` (same type as `events`).

        Examples
        --------
        >>> events = AnalysisReader.calpulsers(reader.scan(), select=False)  # remove the calpulsers
        """
        if time_range is None:
            time_range = (1 * units.microsecond, 5 * units.microsecond)
        elif len(time_range) != 2 or not 0 <= time_range[0] < time_range[1] <= units.second:
            raise ValueError(
                f"time_range must be (start, stop) with 0 <= start < stop <= 1 s (in NuRadio units), got {time_range}."
            )

        # jd1 is a whole number of days (whole seconds): the time within the second follows from jd2 alone
        # (about 10 ps precision)
        time_in_second = (pl.col("station_time_jd2") * units.day) % units.second
        events = events.with_columns(is_calpulser=time_in_second.is_between(*time_range))

        if select is None:
            return events

        return events.filter(pl.col("is_calpulser") if select else ~pl.col("is_calpulser"))

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
                    # A leaf path is e.g. "channelParameters.SNR.list.element.key_value.value"
                    path = leaf.path_in_schema
                    name = next((name for name in names if path == name or path.startswith(name + ".")), None)
                    statistics = leaf.statistics
                    if statistics is None or not statistics.has_null_count:
                        if name is not None and not pu.is_trace_column(name):  # traces are written without statistics
                            no_statistics.add(file)  # unknown: count as data
                    elif statistics.null_count == leaf.num_values:
                        continue  # only nulls in this row group

                    filled.add(name)

        if no_statistics:
            logger.warning(
                f"{len(no_statistics)} file(s) without null counts (written without statistics), "
                f"their columns are listed as having data, e.g. {sorted(no_statistics)[0]}"
            )

        filled.discard(None)  # leaves of columns which are not in `names`

        return filled


def _print_columns(args):
    """The ``columns`` command: summary (stderr), then the columns as list or table (stdout), optionally the triggers."""
    if args.triggers and args.table != "station":
        raise ValueError("--triggers needs the station table (the trigger is a column of the station files).")

    reader = AnalysisReader(args.path, stations=args.stations, runs=args.runs, sort=False)
    result = reader.columns(
        kind=args.table, non_null=not args.all, parameter_classes=args.parameter_classes, pattern=args.match
    )
    table = reader.tables()[args.table]

    # Summary on stderr: stdout stays a plain list of columns (e.g. for grep).
    # The row count comes from the file footers, stations / runs read only these two columns.
    n_rows, n_runs = table.select(pl.len(), pl.col("run").n_unique()).collect().row(0)
    stations = ""
    if args.table == "station":
        stations = f"stations {table.select(pl.col('station_id').unique().sort()).collect().to_series().to_list()}, "
    print(
        f"{len(reader.files[args.table])} files, {n_rows} rows, {stations}{n_runs} runs, "
        f"{len(result)} of {len(table.collect_schema())} columns",
        file=sys.stderr,
    )

    rows = [[name] + ([str(dtype)] if args.types else []) for name, dtype in result.items()]
    if args.stats:
        # Lists are flattened, NaN / inf are counted separately. Other types (strings, maps, structs) only get
        # the number of (non-null) values; for maps, their entries are counted.
        # All statistics are computed in one query (one pass over the data), named "<column>|<statistic>".
        exprs = []

        for name, dtype in result.items():
            expr = pl.col(name)

            while isinstance(dtype, (pl.List, pl.Map)):  # e.g. channel parameters: one value per channel
                if isinstance(dtype, pl.Map):  # -> list of {key, value} structs: one per entry
                    expr, dtype = expr.map.entries(), pl.List(pl.Struct({"key": dtype.key, "value": dtype.value}))
                expr, dtype = expr.explode(empty_as_null=False), dtype.inner

            if dtype.is_numeric() or dtype == pl.Boolean:
                expr = expr.cast(pl.Float64)
                valid = pl.when(expr.is_finite()).then(expr)  # NaN / inf -> null (ignored by the aggregations)
                exprs += [
                    valid.count().alias(f"{name}|n"),
                    (~expr.is_finite()).sum().alias(f"{name}|nan/inf"),
                    valid.min().alias(f"{name}|min"),
                    valid.max().alias(f"{name}|max"),
                    valid.mean().alias(f"{name}|mean"),
                    valid.std().alias(f"{name}|std"),
                ]
            else:
                exprs.append(expr.count().alias(f"{name}|n"))

        statistics = table.select(exprs).collect().row(0, named=True) if exprs else {}
        # Missing statistics (e.g. min of a string column) are printed as "-"
        rows = [row + [_format(statistics.get(f"{row[0]}|{stat}")) for stat in _STATISTICS] for row in rows]

    if args.stats or args.types:
        header = ["column"] + (["type"] if args.types else []) + (_STATISTICS if args.stats else [])
        _print_table(header, rows, n_text_columns=2 if args.types else 1)
    else:  # plain list (e.g. for grep)
        print(*(row[0] for row in rows), sep="\n")

    if args.triggers:
        _print_triggers(table)


def _print_triggers(table):
    """Number and fraction of the events per trigger (the name of the first trigger of the station)."""
    if "trigger" not in table.collect_schema().names():
        raise ValueError("The files have no column 'trigger' (written by an older version of the writer).")

    counts = table.group_by("trigger").len().sort("len", descending=True).collect()
    n_events = counts["len"].sum()
    rows = [
        ["(none)" if name is None else name, str(count), f"{100 * count / n_events:.1f}%"] for name, count in counts.rows()
    ]

    print()
    _print_table(["trigger", "events", "fraction"], rows)


_STATISTICS = ["n", "nan/inf", "min", "max", "mean", "std"]


def _print_table(header, rows, n_text_columns=1):
    """Print a table with a border and grid lines. The first `n_text_columns` are left aligned, the others right."""
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]

    def line(left, middle, right, fill="─"):  # horizontal line with the given corner / crossing characters
        return left + middle.join(fill * (width + 2) for width in widths) + right

    def cells(row):
        return (
            "│"
            + "│".join(
                f" {value.ljust(width) if i < n_text_columns else value.rjust(width)} "
                for i, (value, width) in enumerate(zip(row, widths))
            )
            + "│"
        )

    print(line("┌", "┬", "┐"))
    print(cells(header))
    print(line("╞", "╪", "╡", fill="═"))

    for i, row in enumerate(rows):
        if i:
            print(line("├", "┼", "┤"))
        print(cells(row))

    print(line("└", "┴", "┘"))


def _format(value):
    """Integers (e.g. counts, ids) without decimals, other numbers with 4 significant digits, None as "-"."""
    if value is None:
        return "-"

    return str(int(value)) if float(value).is_integer() and abs(value) < 1e9 else f"{value:.4g}"


def _channel_id_column(name):
    """The channel id column the (list) column `name` is aligned with, None if it is no channel column."""
    if name.startswith(("channelParameters", "channel.")):
        return "channel_ids"

    if name.startswith("sim_channel."):
        return "sim_channel.channel_id"

    if name.startswith("trigger_channel."):
        return "trigger_channel.id"

    return None


def _histogram_values(reader, kind, column, channels=None, key=None):
    """
    The values of `column` for a histogram as `polars.DataFrame` with the column ``value``
    (and ``channel_id`` for channel parameters).

    Lists are flattened. Channel parameters are lists aligned with the channel ids: `channels` selects channel ids.
    Maps (e.g. ``channelParameters.SNR``) need the `key` to histogram.
    """
    table = reader.tables()[kind]
    schema = table.collect_schema()

    # Full name, else the name without the parameter class, e.g. "coherent_snr"
    candidates = [name for name in schema if column == name] or [name for name in schema if column == name.partition(".")[2]]
    if len(candidates) != 1:
        raise ValueError(f"'{column}' matches {candidates or 'no column'}. See the `columns` command.")

    name = candidates[0]

    id_column = _channel_id_column(name) if kind == "station" else None
    if id_column is not None:
        # Explode the channel ids together with the values: one row per (event, channel)
        frame = table.select(pl.col(id_column).alias("channel_id"), pl.col(name).alias("value"))
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

    if isinstance(dtype, pl.Map):  # the value of one key (maps without it give null)
        if key is None:
            keys = frame.select(pl.col("value").map.keys().explode(empty_as_null=False).unique().sort()).collect()
            raise ValueError(f"{name} is a map, choose a key with --key: {keys.to_series().drop_nulls().to_list()}")

        frame = frame.with_columns(pl.col("value").map.get(pl.lit(key).cast(dtype.key)))  # --key is a string
    elif key is not None:
        raise ValueError(f"--key only applies to maps, {name} has the type {dtype}.")

    return name, frame.drop_nulls("value").collect()


def _format_channels(channels):
    """[0, 1, 2, 3, 5] -> '0-3, 5'"""
    channels = sorted(set(channels))
    groups = []

    for channel in channels:
        if groups and channel == groups[-1][-1] + 1:
            groups[-1][-1] = channel
        else:
            groups.append([channel, channel])

    return ", ".join(str(first) if first == last else f"{first}-{last}" for first, last in groups)


def _plot_histogram(args):
    """The ``hist`` command: histogram of one column (all channels, selected channels or one per channel)."""
    import matplotlib.pyplot as plt

    reader = AnalysisReader(args.path, stations=args.stations, runs=args.runs, sort=False)
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


def main(argv=None):
    """Command line interface, see ``python -m NuRadioReco.utilities.analysis_reader --help``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect parquet files written by eventWriter, convert .nur files to parquet."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    # Options of all commands. The paths are the last positional argument of each command.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--table",
        choices=list(pu._SORT_KEYS),
        default="station",
        help="station files (one row per event and station) or event files (one row per event).",
    )

    common.add_argument("--stations", nargs="+", type=int, default=None)

    common.add_argument("--runs", nargs="+", type=int, default=None)

    columns = commands.add_parser("columns", parents=[common], help="List the columns (by default only those with data).")
    columns.add_argument("path", nargs="+", help="Directories, files or glob patterns (see `AnalysisReader`).")

    columns.add_argument(
        "--class",
        dest="parameter_classes",
        nargs="+",
        metavar="CLASS",
        help="Only these parameter classes, e.g. channelParameters stationParametersRNOG.",
    )

    columns.add_argument(
        "--match", default=None, metavar="REGEX", help="Only columns matching this regular expression (case insensitive)."
    )

    columns.add_argument("--all", action="store_true", help="Also list columns without any (non-null) value.")

    columns.add_argument("--types", action="store_true", help="Also print the data types.")

    columns.add_argument(
        "--stats",
        action="store_true",
        help="Also print the number of (finite) values, of NaN / inf, min, max, mean and std (reads the data, lists are "
        "flattened). Strings, maps (entries) and structs only get the number of values.",
    )

    columns.add_argument(
        "--triggers", action="store_true", help="Also print the number and fraction of events per (first) trigger."
    )

    hist = commands.add_parser("hist", parents=[common], help="Plot the histogram of a column.")
    hist.add_argument("column", help="Column name, the parameter class can be omitted (e.g. coherent_snr).")

    hist.add_argument("path", nargs="+", help="Directories, files or glob patterns (see `AnalysisReader`).")

    hist.add_argument(
        "--channels", nargs="+", type=int, default=None, help="Channel ids (channel parameters only). Default: all channels."
    )

    hist.add_argument("--per-channel", action="store_true", help="One histogram per channel (channel parameters).")

    hist.add_argument("--key", default=None, help="Key of a map column, e.g. peak_amplitude for SNR.")

    hist.add_argument("--bins", type=int, default=100)

    hist.add_argument("--range", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"))

    hist.add_argument("--log", action="store_true", help="Logarithmic bins and x-axis.")

    hist.add_argument("--logy", action="store_true", help="Logarithmic y-axis.")

    hist.add_argument("-o", "--output", default=None, help="Save the figure to this file instead of showing it.")

    convert = commands.add_parser(
        "convert", help="Convert .nur files to parquet (a detector stored in a .nur file is not converted)."
    )
    convert.add_argument("inputs", nargs="+", help=".nur files or glob patterns.")

    convert.add_argument(
        "--output", default=None, help="Write all inputs into <OUTPUT>.parquet. Default: one output per input, same name."
    )

    convert.add_argument("--output-dir", default=None, help="Output directory. Default: next to each input.")

    convert.add_argument("--no-traces", action="store_true", help="Do not store traces.")

    convert.add_argument(
        "--single-file", action="store_true", help="No separate event file: event columns are added to the station rows."
    )

    convert.add_argument(
        "--max-file-size",
        type=float,
        default=1024,
        help="Maximum size of an output file pair in MB, larger outputs are split (default: 1024).",
    )

    convert.add_argument("--compression", default="snappy", help="Parquet compression codec, e.g. zstd.")

    convert.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")

    args = parser.parse_args(argv)

    try:
        if args.command == "columns":
            _print_columns(args)
        elif args.command == "hist":
            _plot_histogram(args)
        elif args.command == "convert":
            from NuRadioReco.modules.io.nur_to_parquet import convert_nur  # imports the writer (no cycle at import)

            convert_nur(
                args.inputs,
                output=args.output,
                output_dir=args.output_dir,
                traces=not args.no_traces,
                event_file=not args.single_file,
                max_file_size=args.max_file_size,
                compression=args.compression,
                overwrite=args.overwrite,
            )
    except (ValueError, FileNotFoundError, FileExistsError) as e:  # user errors: no traceback
        parser.exit(1, f"error: {e}\n")


if __name__ == "__main__":
    main()
