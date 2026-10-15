"""
Writes the parameter storage content of stations to parquet files (never traces).

Purpose
-------
The `.nur` format stores complete events (including traces) as pickled objects: it is flexible, but
reading one parameter of many events means deserializing everything. This module writes the derived
(simulated / reconstructed) quantities into a table based format instead, for fast analysis later on:

* One row per event and one column per parameter, with a fixed schema defined by the typed parameter
  enums (`NuRadioReco.framework.parameters`). No pickle, so the files are readable without NuRadio.
* Parquet is columnar and compressed: a reader loads only the columns it needs, and per-row-group
  statistics (min / max) let it skip data that can not match a filter, e.g. on ``run``,
  ``event_id`` or a reconstructed energy.
* Standard tools read the files directly (polars, pandas, pyarrow, DuckDB, ...), also lazily and
  for data sets larger than memory. `NuRadioReco.utilities.parquet_utilities.ParquetReader`
  adds station / run selection and joins the tables.

Reading
-------
Use `NuRadioReco.utilities.parquet_utilities.ParquetReader` to read a directory or glob of these files
as polars frames, selecting stations and runs and optionally joining the electric field and sim channel
tables::

    reader = ParquetReader("out/", stations=[11], runs=[100, 101], efields=True)
    df = reader.read()

Files
-----
One file ``station<id>_run<run number>[_<suffix>].parquet`` is written per station and run, with one
row per event. With ``group_runs`` a block of consecutive runs is written to one file
``station<id>_run<first>-<last>[_<suffix>].parquet``.

Each station parameter is one column with the type of the parameter (scalar, list or map, e.g.
``nu_vertex`` is a ``list<double>``). Each channel parameter is a list column with one entry per
channel, aligned with the ``channel_ids`` column (null entry if a channel lacks the parameter), so a
list (map) valued channel parameter becomes a list of lists (maps). Column names are
``<enum class>.<parameter>``; the schema is derived from the types of the parameter enums
(see `NuRadioReco.utilities.parquet_utilities`). Parameters without a mappable type are dropped
with a warning. The column ``trigger`` holds the name of the first (earliest) trigger of the station
(null if no trigger fired).

Optionally (see `EventWriterParquet.begin`):

* ``store_sim``: the parameters of the sim station are added as ``sim_<enum class>.<parameter>`` columns,
  and the sim channels are written to a second file ``station<id>_run<run number>[_<suffix>]_simchannel.parquet``
  with one row per sim channel (``channel_id``, ``shower_id``, ``ray_tracing_id`` + channel parameters).
* ``store_efields``: the electric fields are written to a second file
  ``station<id>_run<run number>[_<suffix>]_efield.parquet`` with one row per electric field.

The columns ``run``, ``event_id`` and ``station_id`` link these files to the main file
(the key of an event), e.g. for a join.

Batching
--------
Events are buffered per file and written as one row group per ``batch_size`` events. Only the files
of the current run (block) of each station are kept open: they are flushed and closed once that station
gets an event of a different run (block).

Metadata
--------
The file metadata (read with ``pyarrow.parquet.read_schema(file).metadata``) contains the keys
``nuradio_version``, ``nuradio_commit`` and ``comment``. The commit hash does not capture
uncommitted changes.
"""

import logging
import os
from timeit import default_timer as timer

import pyarrow as pa
import pyarrow.parquet as pq

import NuRadioReco
from NuRadioReco.framework import parameters
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import parquet_utilities as pu
from NuRadioReco.utilities import version
from NuRadioReco.utilities.logging import LOGGING_STATUS

logger = logging.getLogger("NuRadioReco.eventWriterParquet")

# Experiment specific enums (e.g. stationParametersRNOG) have to be passed to `begin` explicitly
DEFAULT_STATION_ENUMS = [parameters.stationParameters]
DEFAULT_CHANNEL_ENUMS = [parameters.channelParameters]

# Written as dedicated columns (station_time_jd1/2), not as generic parameter column
_EXPLICIT_STATION_PARAMETERS = ["stationParameters.station_time"]

# Prefix of the columns holding simulated quantities, e.g. ``sim_stationParameters.nu_energy``
_SIM_PREFIX = "sim_"


# Columns which link the files (the key of an event)
_KEY_FIELDS = [pa.field("run", pa.int64()), pa.field("event_id", pa.int64()), pa.field("station_id", pa.int64())]


def _warn_once(warned, column, dtype, error):
    """ Warn about a value of `column` which does not match its declared type (once per column). """
    if column not in warned:
        logger.warning(f"Parameter {column} does not match its declared type {dtype}, writing null instead: {error}")
        warned.add(column)


class _BatchedWriter:
    """ Buffers rows (dicts: column name -> value) and writes them as one row group per `batch_size` rows. """

    def __init__(self, path, schema, batch_size, warned):
        """
        Parameters
        ----------
        path : str
            Output file (opened immediately, overwritten if it exists).
        schema : pyarrow.Schema
            Schema of the file, defines the columns of the rows passed to `add`.
        batch_size : int
            Number of rows after which the buffer is written as one row group.
        warned : set
            Columns with a type mismatch which were already warned about (shared between the files).
        """
        self.__schema = schema
        self.__batch_size = batch_size
        self.__warned = warned
        self.__writer = pq.ParquetWriter(path, schema)
        self.__buffer = self.__new_buffer()
        self.__n_rows = 0  # rows in the buffer

        self.write_time = 0.  #: seconds spent in `close` (last flush and closing the file)

    def add(self, row):
        """ Append a row (dict: column name -> value, all columns of the schema). """
        for name, value in row.items():
            self.__buffer[name].append(value)

        self.__n_rows += 1
        if self.__n_rows >= self.__batch_size:
            self.flush()

    def flush(self):
        """ Write the buffered rows as one row group. """
        if self.__n_rows == 0:
            return

        arrays = [self.__to_array(field) for field in self.__schema]
        self.__writer.write_table(pa.Table.from_arrays(arrays, schema=self.__schema))

        self.__buffer = self.__new_buffer()
        self.__n_rows = 0

    def close(self):
        """ Flush and close the file (the footer is only written on close). """
        start = timer()
        self.flush()
        self.__writer.close()
        self.write_time += timer() - start

    def __to_array(self, field):
        """
        The buffered values of `field` as arrow array. Only if the conversion fails (a value does not match
        the declared type), the values are checked one by one and the mismatching ones are written as null.
        """
        values = self.__buffer[field.name]
        try:
            return pa.array(values, type=field.type)
        except (pa.ArrowException, TypeError, ValueError) as e:
            _warn_once(self.__warned, field.name, field.type, e)
            return pa.array([self.__checked(value, field.type) for value in values], type=field.type)

    @staticmethod
    def __checked(value, dtype):
        """ `value` if it matches `dtype`, else None. """
        try:
            pa.scalar(value, type=dtype)
            return value
        except (pa.ArrowException, TypeError, ValueError):
            return None

    def __new_buffer(self):
        """ An empty buffer: column name -> list of values (one per row). """
        return {name: [] for name in self.__schema.names}


class EventWriterParquet:
    """
    Writes the parameters of stations (and optionally sim stations and electric fields) to parquet files.

    See the module documentation for the file layout. Call `begin`, then `run` for each event and
    station, and finally `end`.
    """

    def __init__(self):
        """ Only sets the empty state: the configuration (and the schemas) are set in `begin`. """
        self.__output_dir = None
        self.__suffix = None
        self.__overwrite = False
        self.__store_channels = True
        self.__store_sim = False
        self.__store_efields = False
        self.__batch_size = None
        self.__group_runs = None

        # station id -> {"event" / "efield" / "simchannel": _BatchedWriter or None} for the current run (block) of the station
        self.__files = {}
        # station id -> first run of the block of its open files
        self.__blocks = {}
        # (station id, first run of the block) -> number of files written so far (numbers the parts of revisited blocks)
        self.__parts = {}
        self.__n_events = 0

        # For the summary in `end`
        self.__n_files = 0
        self.__write_time = 0.

        # Parameters which could not be converted (warn only once per column)
        self.__warned = set()

        # Set in `begin`
        self.__schema = None
        self.__efield_schema = None
        self.__sim_channel_schema = None
        self.__station_columns = {}
        self.__channel_columns = {}
        self.__efield_columns = {}

    def begin(self, output_dir, suffix="", batch_size=10000, group_runs=None,
              station_enums=None, channel_enums=None, store_channels=True, store_sim=False, store_efields=False,
              overwrite=False, comment="", log_level=logging.NOTSET):
        """
        Parameters
        ----------
        output_dir : str
            Directory of the output files (created if needed).
        suffix : str, default=""
            Optional suffix of the file names: ``station<id>_run<run number>_<suffix>.parquet``.
        batch_size : int, default=10000
            Maximum number of events (electric fields for the efield file) buffered before they are
            written as one row group.
        group_runs : int, optional
            Write blocks of `group_runs` consecutive run numbers to the same file, e.g. 100: runs 100 to 199 go to
            ``station<id>_run000100-000199[_<suffix>].parquet``. Default: one file per run.
        station_enums : list of TypedEnum, optional
            Parameter enums stored as station columns. Default: ``stationParameters``.
            Add experiment specific ones, e.g. ``parameters.stationParametersRNOG``.
        channel_enums : list of TypedEnum, optional
            Parameter enums stored as channel columns. Default: ``channelParameters``.
            Add experiment specific ones, e.g. ``parameters.channelParametersRNOG``.
        store_channels : bool, default=True
            Store the channel parameters as list columns of the main file. If False only ``channel_ids`` is
            written (the sim channel file is not affected).
        store_sim : bool, default=False
            Also store the parameters of the sim station (same enums as the station, as extra columns; null
            if there is no sim station) and the sim channels (own file, same enums as the channels).
        store_efields : bool, default=False
            Write the electric fields of the station (and of the sim station if `store_sim`) to a second file.
        overwrite : bool, default=False
            If False, a `FileExistsError` is raised when a file to be written already exists
            (checked when the file is opened, i.e. at the first event of a station and run / block).
            If True, existing files are overwritten.
        comment : str, default=""
            Free text stored in the file metadata (key ``comment``).
        log_level : int, default=logging.NOTSET
            Use this to override the logging level for this module.
        """
        logger.setLevel(log_level)

        self.__output_dir = output_dir
        self.__suffix = suffix
        self.__overwrite = overwrite
        self.__store_channels = store_channels
        self.__store_sim = store_sim
        self.__store_efields = store_efields
        self.__batch_size = batch_size
        if group_runs is not None and group_runs < 1:
            raise ValueError(f"group_runs has to be a positive integer, got {group_runs}.")

        self.__group_runs = group_runs
        self.__files = {}
        self.__blocks = {}
        self.__parts = {}
        self.__n_events = 0
        self.__n_files = 0
        self.__write_time = 0.

        # `is None` (not `or`) so that an empty list can be passed to store no parameters of that kind
        self.__build_schemas(
            DEFAULT_STATION_ENUMS if station_enums is None else station_enums,
            DEFAULT_CHANNEL_ENUMS if channel_enums is None else channel_enums,
            comment)

    @register_run()
    def run(self, evt, station):
        """
        Add the parameters of `station` to the output.

        Parameters
        ----------
        evt : NuRadioReco.framework.event.Event
        station : NuRadioReco.framework.station.Station
        """
        run_number = evt.get_run_number()
        station_id = station.get_id()

        # A new run (block) of a station closes its files of the previous one (this bounds the number of open files)
        first_run, last_run = self.__run_block(run_number)
        if self.__blocks.get(station_id, first_run) != first_run:
            self.__close_files(station_id)
        if station_id not in self.__files:
            self.__open_files(station_id, first_run, last_run)
        files = self.__files[station_id]

        keys = {"run": run_number, "event_id": evt.get_id(), "station_id": station_id}
        sim_station = station.get_sim_station() if self.__store_sim else None

        files["event"].add(self.__get_event_row(keys, station, sim_station))

        if files["simchannel"] is not None and sim_station is not None:
            for sim_channel in sorted(sim_station.iter_channels(), key=self.__sim_channel_sort_key):
                files["simchannel"].add(self.__get_sim_channel_row(keys, sim_channel))

        if files["efield"] is not None:
            for source, efield_station in [("station", station), ("sim_station", sim_station)]:
                if efield_station is None:
                    continue

                for efield in efield_station.get_electric_fields():
                    files["efield"].add(self.__get_efield_row(keys, source, efield))

        self.__n_events += 1

    def __run_block(self, run_number):
        """ The first and last run of the file of `run_number` (see `group_runs`). """
        if self.__group_runs is None:
            return run_number, run_number

        first_run = run_number - run_number % self.__group_runs
        return first_run, first_run + self.__group_runs - 1

    def end(self):
        """
        Flush remaining events, close all files and log a summary. Returns the number of events written.
        A repeated call (e.g. by `end_all_modules`) finds no open file and logs nothing.
        """
        if self.__files:
            self.__close_files()
            logger.log(
                LOGGING_STATUS,
                f"EventWriterParquet: wrote {self.__n_events} events to {self.__n_files} files in {self.__output_dir} "
                f"(writing to disk: {self.__write_time:.2f}s).")

        return self.__n_events

    def __build_schemas(self, station_enums, channel_enums, comment):
        """ Derive the (fixed) arrow schemas of all output files from the types of the enum members. """
        self.__station_columns = pu.column_schema(station_enums, exclude=_EXPLICIT_STATION_PARAMETERS)
        self.__channel_columns = pu.column_schema(channel_enums)
        self.__efield_columns = pu.column_schema([parameters.electricFieldParameters])

        fields = _KEY_FIELDS + [
            pa.field("station_time_jd1", pa.float64()),  # astropy jd1 + jd2 (UTC): ns precision, as in .nur
            pa.field("station_time_jd2", pa.float64()),
            pa.field("trigger", pa.string()),  # name of the first (earliest) trigger, null if none fired
            pa.field("channel_ids", pa.list_(pa.int64())),
        ]
        fields += [pa.field(name, dtype) for name, (_, dtype) in self.__station_columns.items()]

        # Channel parameters are lists with one entry per channel
        if self.__store_channels:
            fields += [pa.field(name, pa.list_(dtype)) for name, (_, dtype) in self.__channel_columns.items()]

        if self.__store_sim:
            fields += [pa.field(_SIM_PREFIX + name, dtype) for name, (_, dtype) in self.__station_columns.items()]

        # Stored as key-value metadata in the footer of every file
        metadata = {
            "nuradio_version": str(NuRadioReco.__version__),
            "nuradio_commit": version.get_NuRadioMC_commit_hash(),
            "comment": comment,
        }
        self.__schema = pa.schema(fields, metadata=metadata)

        # Electric fields: one row per electric field, linked to the main file by (run, event_id, station_id)
        efield_fields = _KEY_FIELDS + [
            pa.field("source", pa.string()),  # "station" or "sim_station"
            pa.field("channel_ids", pa.list_(pa.int64())),
            pa.field("shower_id", pa.int64()),
            pa.field("ray_tracing_id", pa.int64()),
            pa.field("position", pa.list_(pa.float64())),
            pa.field("trace_start_time", pa.float64()),
        ]
        efield_fields += [pa.field(name, dtype) for name, (_, dtype) in self.__efield_columns.items()]
        self.__efield_schema = pa.schema(efield_fields, metadata=metadata)

        # Sim channels: one row per (channel, shower, ray tracing solution)
        sim_channel_fields = _KEY_FIELDS + [
            pa.field("channel_id", pa.int64()),
            pa.field("shower_id", pa.int64()),
            pa.field("ray_tracing_id", pa.int64()),
            pa.field("trace_start_time", pa.float64()),
        ]
        sim_channel_fields += [pa.field(name, dtype) for name, (_, dtype) in self.__channel_columns.items()]
        self.__sim_channel_schema = pa.schema(sim_channel_fields, metadata=metadata)

    def __get_event_row(self, keys, station, sim_station):
        """ The row of the main file: keys, station time, station / channel parameters (and sim parameters). """
        row = dict(keys)

        # Station time as two floats (jd1 + jd2) to keep ns precision, same as in the .nur format
        time = station.get_station_time()
        time = time.utc if time is not None else None  # .utc returns a copy
        row["station_time_jd1"] = None if time is None else float(time.jd1)
        row["station_time_jd2"] = None if time is None else float(time.jd2)

        trigger = station.get_first_trigger()
        row["trigger"] = None if trigger is None else str(trigger.get_name())

        # Station parameters: one scalar (or nested) value per column
        for name, (key, dtype) in self.__station_columns.items():
            row[name] = self.__get_value(station, key, name, dtype)

        # Channel parameters: one list per column, aligned with `channel_ids`
        channels = list(station.iter_channels(sorted=True))
        row["channel_ids"] = [channel.get_id() for channel in channels]
        if self.__store_channels:
            for name, (key, dtype) in self.__channel_columns.items():
                row[name] = [self.__get_value(channel, key, name, dtype) for channel in channels]

        # Sim station parameters (null without sim station)
        if self.__store_sim:
            for name, (key, dtype) in self.__station_columns.items():
                row[_SIM_PREFIX + name] = None if sim_station is None else self.__get_value(sim_station, key, name, dtype)

        return row

    @staticmethod
    def __sim_channel_sort_key(sim_channel):
        """ Deterministic order: channel id, shower id, ray tracing id (the latter two can be None). """
        return tuple(-1 if x is None else x for x in sim_channel.get_unique_identifier())

    def __get_sim_channel_row(self, keys, sim_channel):
        """ The row of the sim channel file for one sim channel. """
        row = dict(keys)
        row["channel_id"] = sim_channel.get_id()
        row["shower_id"] = sim_channel.get_shower_id()
        row["ray_tracing_id"] = sim_channel.get_ray_tracing_solution_id()
        row["trace_start_time"] = sim_channel.get_trace_start_time()

        for name, (key, dtype) in self.__channel_columns.items():
            row[name] = self.__get_value(sim_channel, key, name, dtype)

        return row

    def __get_efield_row(self, keys, source, efield):
        """ The row of the electric field file for one electric field. """
        row = dict(keys)
        row["source"] = source
        row["channel_ids"] = [int(channel_id) for channel_id in efield.get_channel_ids()]
        row["shower_id"] = efield.get_shower_id()
        row["ray_tracing_id"] = efield.get_ray_tracing_solution_id()
        row["position"] = [float(x) for x in efield.get_position()]
        row["trace_start_time"] = efield.get_trace_start_time()

        for name, (key, dtype) in self.__efield_columns.items():
            row[name] = self.__get_value(efield, key, name, dtype)

        return row

    def __open_files(self, station_id, first_run, last_run):
        """ Open the new file(s) for (station, run block) with empty buffers. """
        # Revisiting a run (block) after its files were closed must not overwrite the earlier file
        part = self.__parts.get((station_id, first_run), 0)
        self.__parts[(station_id, first_run)] = part + 1

        name = f"station{station_id:02d}_run{first_run:06d}"
        if last_run != first_run:
            name += f"-{last_run:06d}"
        if self.__suffix:
            name += f"_{self.__suffix}"
        if part:
            name += f"_part{part:02d}"

        files = {"event": self.__open_file(f"{name}.parquet", self.__schema), "efield": None, "simchannel": None}
        if self.__store_efields:
            files["efield"] = self.__open_file(f"{name}_efield.parquet", self.__efield_schema)
        if self.__store_sim:
            files["simchannel"] = self.__open_file(f"{name}_simchannel.parquet", self.__sim_channel_schema)

        self.__files[station_id] = files
        self.__blocks[station_id] = first_run
        self.__n_files += sum(writer is not None for writer in files.values())
        logger.debug(f"Writing station {station_id}, runs {first_run}-{last_run} to {os.path.join(self.__output_dir, name)}*.parquet")

    def __open_file(self, filename, schema):
        """ Open `filename` in the output directory (fails if it exists and `overwrite` is not set). """
        path = os.path.join(self.__output_dir, filename)
        if os.path.exists(path) and not self.__overwrite:
            raise FileExistsError(f"{path} already exists. Use `overwrite=True` in `begin` to overwrite it.")

        os.makedirs(self.__output_dir, exist_ok=True)
        return _BatchedWriter(path, schema, self.__batch_size, self.__warned)

    def __close_files(self, station_id=None):
        """ Flush and close the open files of `station_id` (default: of all stations). """
        for station in list(self.__files) if station_id is None else [station_id]:
            self.__blocks.pop(station)
            for writer in self.__files.pop(station).values():
                if writer is not None:
                    writer.close()
                    self.__write_time += writer.write_time

    def __get_value(self, obj, key, column, dtype):
        """
        The parameter `key` of `obj` converted for arrow, or None if not set / not convertible. Whether it matches
        `dtype` is only checked when the batch is written (see `_BatchedWriter`).
        """
        if not obj.has_parameter(key):
            return None

        try:
            return pu.to_arrow_value(obj.get_parameter(key, copy=False), dtype)
        except (TypeError, ValueError, AttributeError) as e:  # e.g. not a dict for a map type
            _warn_once(self.__warned, column, dtype, e)
            return None
