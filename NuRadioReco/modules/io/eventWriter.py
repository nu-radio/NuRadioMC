"""
Writes complete NuRadio events (parameters, traces, simulated quantities) to parquet files.

This is the default event format of NuRadioReco; the pickle based ``.nur`` writer was removed (``.nur`` files are
still read by `NuRadioReco.modules.io.eventReader`, convert them with `NuRadioReco.modules.io.nur_to_parquet.convert_nur`).

Parquet is a typed, columnar and compressed table format: an analysis reads only the columns it needs
(e.g. one reconstructed quantity of many events) without decoding the traces, and standard tools
(polars, pyarrow, pandas, DuckDB, ...) read the files without NuRadio. Nothing is pickled: every column has
the type declared in the parameter enums (`NuRadioReco.framework.parameters`).

Files
-----
All stations and runs go, in the order `eventWriter.run` receives them, to the station file
``<filename>.parquet`` (one row per event and station) and the event file ``<filename>_event.parquet``
(one row per event: event parameters, showers, particles, ...). With ``event_file=False`` (see
`eventWriter.begin`) there is no event file: its columns are added to the station rows (prefix ``event.``,
e.g. ``event.modules``, repeated for every station of an event). When the files reach ``max_file_size`` or
``events_per_file``, the output continues in ``<filename>_part02.parquet`` (+ ``<filename>_part02_event.parquet``),
``_part03``, ...
(numbered as the ``.nur`` files were: the first file pair is part 1).
A station file and its event file (same base name) hold the same events.
All files are first written to hidden temporary files in the target directory (``.<file name>.<random>.tmp``)
and moved into place by `eventWriter.close`: an output can replace the files it was read from, and a run that
stops with an exception leaves existing files untouched (see `eventWriter.end`). A replaced file keeps its
permissions, a new file gets those of `open` (``0o666 & ~umask``).
``run``, ``event_id`` (and ``station_id``) are the keys linking the files; readers select stations and runs
via the row group statistics of these columns.

Columns
-------
Objects of one kind (channels, sim channels, triggers, electric fields, showers, ...) are stored as list columns, one
entry per object, aligned with an id column of their kind (e.g. ``channel_ids``). Parameter columns are named
``<prefix><enum class>.<member>`` (e.g. ``channelParameters.SNR``, ``sim_channel.channelParameters.SNR``),
other attributes ``<kind>.<attribute>`` (e.g. ``channel.trace``), except those of the event itself (e.g. ``modules``,
``covariances``: the event file holds only events). Traces are stored in the time domain
(float64, as in ``.nur``); trace columns are not dictionary encoded and have no statistics.
A null parameter value means "not set"; the names of the parameters set to None are listed in
``<kind>.none_parameters`` (e.g. ``particle.none_parameters``: ``["particleParameters.parent_id"]``).
The triggers of a station are stored with name, type and `triggerParameters` (``trigger.name``, ``trigger.type``,
``trigger.triggerParameters.threshold``, ...); the column ``trigger`` holds the name of the earliest fired trigger.
Triggers of sim stations are not stored.

The keyword arguments of the registered modules are stored as JSON (see `io_utilities._to_json`).

Metadata
--------
The file metadata (``pyarrow.parquet.read_schema(file).metadata``) contains ``nuradio_version``,
``nuradio_commit``, ``comment``, ``layout`` (``dual`` / ``single``), ``kind`` (``station`` / ``event``),
``sim_config`` (JSON, from the first event of the file), ``parameter_enums`` (JSON: the names of the
``station_enums`` and ``channel_enums`` passed to `eventWriter.begin`) and the keys passed with ``metadata``.
"""

import glob
import json
import logging
import os
import stat
import tempfile
import warnings
from timeit import default_timer as timer

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import NuRadioReco
from NuRadioReco.framework import parameters
from NuRadioReco.framework.base_station import BaseStation
from NuRadioReco.modules.base import module
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import io_utilities
from NuRadioReco.utilities import parquet_utilities as pu
from NuRadioReco.utilities import version
from NuRadioReco.utilities.logging import LOGGING_STATUS

logger = logging.getLogger("NuRadioReco.eventWriter")

# Further experiment specific enums have to be passed to `begin` explicitly
DEFAULT_STATION_ENUMS = [parameters.stationParameters, parameters.stationParametersRNOG]
DEFAULT_CHANNEL_ENUMS = [parameters.channelParameters, parameters.channelParametersRNOG]
EVENT_ENUMS = [parameters.eventParameters, parameters.generatorAttributes]

# Not stored as parameter columns: station_time has dedicated columns, sim_config goes to the file metadata
_EXCLUDED_PARAMETERS = ["stationParameters.station_time", "eventParameters.sim_config"]

# Prefix of the event level columns in the station files with ``event_file=False``
EVENT_PREFIX = pu.EVENT_PREFIX

_TRACE_KINDS = ("Channels", "ElectricFields", "SimChannels", "SimElectricFields")  # keys of `run`'s mode

# "values": covariance of array valued parameters (e.g. set_parameter_error with an array), flattened
_COVARIANCE_TYPE = pa.list_(
    pa.struct([("a", pa.string()), ("b", pa.string()), ("value", pa.float64()), ("values", pa.list_(pa.float64()))])
)
_TRACE_TYPES = {1: pa.list_(pa.float64()), 2: pa.list_(pa.list_(pa.float64()))}  # per object: 1D / (3, n) trace


def _warn_once(warned, column, message):
    """Log `message` once per `column`."""
    if column not in warned:
        logger.warning(f"{column}: {message}")
        warned.add(column)


def _warn_deprecated(message, stacklevel=3):
    """
    Warn about a deprecated use (`FutureWarning`: shown by default, unlike `DeprecationWarning`).
    `stacklevel`: 3 points to the caller of the method calling this, 4 if that method is decorated by `register_run`.
    """
    warnings.warn(message, FutureWarning, stacklevel=stacklevel)


def _temporary_path(target):
    """
    The path to write `target` to: a hidden temporary file next to it, moved into place by `eventWriter.close`.
    (To use a temporary file only if `target` exists, return `target` here if it does not exist.)
    """
    fd, path = tempfile.mkstemp(dir=os.path.dirname(target) or ".", prefix=f".{os.path.basename(target)}.", suffix=".tmp")
    os.close(fd)

    return path


def _umask():
    """The umask of the process (`os.umask` can only be read by setting it: restored immediately)."""
    umask = os.umask(0o022)
    os.umask(umask)

    return umask


def _file_mode(target, umask):
    """
    The permissions for the file moved to `target` (temporary files are created with 0600): those of an existing
    `target`, else those of a new file created with `open` (``0o666 & ~umask``).
    """
    try:
        return stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        return 0o666 & ~umask


def _trace_mode(mode):
    """Trace kind -> bool from the `mode` argument of `eventWriter.run` (None / True: all, False: none, dict)."""
    if mode is None or mode is True:
        return dict.fromkeys(_TRACE_KINDS, True)
    if mode is False:
        return dict.fromkeys(_TRACE_KINDS, False)

    unknown = set(mode) - set(_TRACE_KINDS)
    if unknown:
        warnings.warn(
            f"eventWriter.run: unknown keys {sorted(unknown)} in mode are ignored, valid keys: " f"{list(_TRACE_KINDS)}",
            stacklevel=4,
        )

    return {kind: bool(mode.get(kind, False)) for kind in _TRACE_KINDS}


def _time_fields(name):
    """Columns of an astropy time: jd1 + jd2 (about 10 ps precision, as in .nur) and the time scale."""
    return [
        pa.field(f"{name}_jd1", pa.float64()),
        pa.field(f"{name}_jd2", pa.float64()),
        pa.field(f"{name}_scale", pa.string()),
    ]


def _time_values(name, time):
    """Values of the `_time_fields` of an `astropy.time.Time` (or None)."""
    if time is None:
        return {f"{name}_jd1": None, f"{name}_jd2": None, f"{name}_scale": None}

    return {f"{name}_jd1": float(time.jd1), f"{name}_jd2": float(time.jd2), f"{name}_scale": time.scale}


def _nested_array(items, arrow_type):
    """
    Arrow list array of `arrow_type` from (nested) sequences of numpy arrays, without python lists of floats.
    `items`: one entry per row, each None or a sequence (the innermost level: 1D arrays).
    """
    lengths = np.fromiter((0 if item is None else len(item) for item in items), dtype=np.int32, count=len(items))
    offsets = np.zeros(len(items) + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    mask = np.fromiter((item is None for item in items), dtype=bool, count=len(items))

    value_type = arrow_type.value_type
    if pa.types.is_list(value_type):
        values = _nested_array([child for item in items if item is not None for child in item], value_type)
    else:
        arrays = [np.asarray(item, dtype=pu.to_numpy_dtype(value_type)) for item in items if item is not None]
        values = pa.array(np.concatenate(arrays) if arrays else np.empty(0), type=value_type)

    return pa.ListArray.from_arrays(pa.array(offsets), values, type=arrow_type, mask=pa.array(mask) if mask.any() else None)


def _leaf_paths(name, dtype):
    """The parquet leaf column paths of an arrow column (nested types are stored as several leaf columns)."""
    if pa.types.is_map(dtype):
        return _leaf_paths(f"{name}.key_value.key", dtype.key_type) + _leaf_paths(f"{name}.key_value.value", dtype.item_type)
    if pa.types.is_list(dtype):
        return _leaf_paths(f"{name}.list.element", dtype.value_type)
    if pa.types.is_struct(dtype):
        return [path for field in dtype for path in _leaf_paths(f"{name}.{field.name}", field.type)]

    return [name]


def _record(obj, save_trace=None):
    """
    The record of a framework object for `_Group.values`: the object, its parameters (the stored dict, not copied)
    and, for objects with a trace (`save_trace` not None), the trace if `save_trace` (see `_trace`).
    """
    record = {"object": obj, "parameters": obj.get_parameters(copy=False)}
    if save_trace is not None:
        record["trace"] = _trace(obj) if save_trace else None

    return record


def _trace(obj):
    """Time trace, sampling rate and start time of a `BaseTrace` (dict), None if it has no trace."""
    trace = obj.get_trace()
    if not trace.shape:  # no trace: np.array(None)
        return None

    return {"time_trace": trace, "sampling_rate": obj.get_sampling_rate(), "trace_start_time": obj.get_trace_start_time()}


def _trace_value(record, key):
    trace = None if record is None else record.get("trace")
    return None if trace is None else trace[key]


def _none_parameters(record):
    """The names of the parameters set to None (a null column means "not set"), None if there are none."""
    names = [str(member) for member, value in record["parameters"].items() if value is None]
    return names or None


def _covariance(a, b, value):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        return {"a": str(a), "b": str(b), "value": float(value), "values": None}

    return {"a": str(a), "b": str(b), "value": None, "values": value.ravel().tolist()}


def _covariances(record):
    covariances = record["object"].get_parameter_covariances()
    return [_covariance(a, b, value) for (a, b), value in covariances.items()]


class _Group:
    """
    The columns of one kind of object, e.g. the channels of a station.

    Each object is passed as record (see `_record`); an entry None of an aligned group writes nulls.
    With ``aligned`` every column is a list with one entry per object, else one object per row.
    """

    def __init__(self, name, parameter_prefix, enums, attributes, trace_dims=0, aligned=True, attribute_prefix=None):
        """
        Parameters
        ----------
        name : str
            Name of the kind of object.
        parameter_prefix : str
            Prefix of the parameter columns: ``<parameter_prefix><enum class>.<member>``.
        enums : list of TypedEnum
            The parameter enums stored as columns.
        attributes : list of (str, pyarrow.DataType, callable)
            Attribute columns: name, type and a function returning the value from the record.
        trace_dims : int
            0: no trace, 1: 1D trace (channels), 2: (3, n) trace (electric fields).
        aligned : bool
            One list entry per object (True) or one object per row (False).
        attribute_prefix : str, optional
            Prefix of the attribute columns (see `column`). Default: ``<name>.``.
        """
        # Layout of the group
        self.name = name
        self.attribute_prefix = f"{name}." if attribute_prefix is None else attribute_prefix
        self.aligned = aligned
        self.trace_dims = trace_dims

        # Parameter columns
        self.parameter_columns = pu.column_schema(enums, prefix=parameter_prefix, exclude=_EXCLUDED_PARAMETERS)
        self.parameter_prefix = parameter_prefix
        self.__members = {member for member, _ in self.parameter_columns.values()}

        # Attribute columns (covariances, parameter bookkeeping, trace metadata)
        self.attributes = list(attributes) + [
            (self.column("covariances"), _COVARIANCE_TYPE, _covariances),
            (self.column("none_parameters"), pa.list_(pa.string()), _none_parameters),
            (
                self.column("parameter_types"),
                pa.list_(pa.string()),
                lambda record: [parameter_type.__name__ for parameter_type in record["object"].get_parameter_types()],
            ),
        ]
        if trace_dims:
            self.attributes += [
                (self.column("sampling_rate"), pa.float64(), lambda record: _trace_value(record, "sampling_rate")),
                (self.column("trace_start_time"), pa.float64(), lambda record: _trace_value(record, "trace_start_time")),
            ]

    def column(self, attribute):
        """The column of an attribute: ``<attribute_prefix><attribute>``, e.g. ``channel.trace``."""
        return self.attribute_prefix + attribute

    @property
    def trace_column(self):
        return self.column("trace") if self.trace_dims else None

    def fields(self):
        """The arrow fields of the group (trace column last)."""
        wrap = pa.list_ if self.aligned else (lambda dtype: dtype)
        fields = [pa.field(column, wrap(dtype)) for column, dtype, _ in self.attributes]
        fields += [pa.field(column, wrap(dtype)) for column, (_, dtype) in self.parameter_columns.items()]
        if self.trace_dims:
            fields.append(pa.field(self.trace_column, wrap(_TRACE_TYPES[self.trace_dims])))

        return fields

    def values(self, records, warned):
        """
        Column name -> value for `records` (list of records if aligned, else one record or None).
        Also returns the number of bytes of the traces (to limit the size of a row group).
        """
        if not self.aligned:
            records = None if records is None else [records]

        columns = {}

        for column, dtype, getter in self.attributes:
            columns[column] = (
                None
                if records is None
                else [None if record is None else self.__convert(getter(record), dtype, column, warned) for record in records]
            )

        for column, (member, dtype) in self.parameter_columns.items():
            columns[column] = (
                None
                if records is None
                else [
                    None if record is None else self.__convert(record["parameters"].get(member), dtype, column, warned)
                    for record in records
                ]
            )

        n_bytes = 0
        if self.trace_dims:
            traces = None if records is None else [_trace_value(record, "time_trace") for record in records]
            columns[self.trace_column] = traces
            n_bytes = 0 if traces is None else sum(trace.nbytes for trace in traces if trace is not None)

        if records is not None:
            self.__check_unknown(records, warned)

        if not self.aligned:
            columns = {column: None if value is None else value[0] for column, value in columns.items()}

        return columns, n_bytes

    @staticmethod
    def __convert(value, dtype, column, warned):
        if value is None:
            return None

        try:
            return pu.to_arrow_value(value, dtype)
        except pu.ElementTypeError as e:
            raise pu.ElementTypeError(f"{column}: {e}") from None
        except (TypeError, ValueError, AttributeError) as e:
            _warn_once(warned, column, f"can not convert {value!r} to {dtype}, writing null: {e}")
            return None

    def __check_unknown(self, records, warned):
        """
        Warn (once) about parameters without a column (not in the enums passed to `begin`): they are lost
        (unless set to None, see `_none_parameters`).
        """
        for record in records:
            if record is None:
                continue

            for member, value in record["parameters"].items():
                if value is not None and member not in self.__members and str(member) not in _EXCLUDED_PARAMETERS:
                    _warn_once(
                        warned,
                        f"{self.name}:{member}",
                        "parameter has no column (enum not passed to `begin` or untyped), it is not stored",
                    )


_INT64 = np.iinfo(np.int64)


def _id(value, name, required=False):
    """
    An id (run number, event/station/channel/object id) as python int, TypeError if it is no int or outside the
    int64 range (pyarrow would truncate it or write null). None only if not `required`.
    """
    if value is None and not required:
        return None
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or not _INT64.min <= int(value) <= _INT64.max
    ):
        raise TypeError(f"{name} must be an int (int64), got {value!r} ({type(value).__name__})")

    return int(value)


def _int_list(values, name):
    """A list of ids (None if `values` is None), each required (see `_id`)."""
    return None if values is None else [_id(value, name, required=True) for value in values]


def _group_id(channel):
    """The channel group id, None if it is the channel id (the default of `Channel.get_group_id`)."""
    group_id = _id(channel.get_group_id(), "channel group id")
    return None if group_id == channel.get_id() else group_id


def _station_groups(station_enums, channel_enums):
    """The groups of the station files (station, sim station, channels, sim channels, triggers, electric fields)."""
    efield_enums = [parameters.electricFieldParameters]
    channel_attributes = [("channel.group_id", pa.int64(), lambda r: _group_id(r["object"]))]
    efield_attributes = lambda name: [  # noqa: E731
        (
            f"{name}.channel_ids",
            pa.list_(pa.int64()),
            lambda r: _int_list(r["object"].get_channel_ids(), f"{name} channel id"),
        ),
        (f"{name}.shower_id", pa.int64(), lambda r: _id(r["object"].get_shower_id(), f"{name} shower id")),
        (
            f"{name}.ray_tracing_id",
            pa.int64(),
            lambda r: _id(r["object"].get_ray_tracing_solution_id(), f"{name} ray tracing id"),
        ),
        (f"{name}.position", pa.list_(pa.float64()), lambda r: r["object"].get_position()),
    ]
    station_attributes = lambda name: [  # noqa: E731
        (f"{name}.particle_type", pa.string(), lambda r: r["object"].get_particle_type()),
        (f"{name}.triggered", pa.bool_(), lambda r: r["object"].has_triggered()),
    ]

    return {
        "station": _Group(
            "station",
            "",
            station_enums,
            station_attributes("station")
            + [("station.reference_reconstruction", pa.string(), lambda r: r["object"].get_reference_reconstruction())],
            aligned=False,
        ),
        "sim_station": _Group(
            "sim_station",
            "sim_station.",
            station_enums,
            station_attributes("sim_station")
            + [
                ("sim_station.station_id", pa.int64(), lambda r: _id(r["object"].get_id(), "sim station id", True)),
                (
                    "sim_station.magnetic_field_vector",
                    pa.list_(pa.float64()),
                    lambda r: r["object"].get_magnetic_field_vector(),
                ),
                ("sim_station.simulation_weight", pa.float64(), lambda r: r["object"].get_simulation_weight()),
            ],
            aligned=False,
        ),
        "channel": _Group("channel", "", channel_enums, channel_attributes, trace_dims=1),
        "trigger_channel": _Group(
            "trigger_channel",
            "trigger_channel.",
            channel_enums,
            [
                ("trigger_channel.id", pa.int64(), lambda r: _id(r["object"].get_id(), "trigger channel id", True)),
                ("trigger_channel.group_id", pa.int64(), lambda r: _group_id(r["object"])),
            ],
            trace_dims=1,
        ),
        "sim_channel": _Group(
            "sim_channel",
            "sim_channel.",
            channel_enums,
            [
                ("sim_channel.channel_id", pa.int64(), lambda r: _id(r["object"].get_id(), "sim channel id", True)),
                ("sim_channel.shower_id", pa.int64(), lambda r: _id(r["object"].get_shower_id(), "sim channel shower id")),
                (
                    "sim_channel.ray_tracing_id",
                    pa.int64(),
                    lambda r: _id(r["object"].get_ray_tracing_solution_id(), "sim channel ray tracing id"),
                ),
                ("sim_channel.group_id", pa.int64(), lambda r: _group_id(r["object"])),
            ],
            trace_dims=1,
        ),
        "trigger": _Group(
            "trigger",
            "trigger.",
            [parameters.triggerParameters],
            [
                ("trigger.name", pa.string(), lambda r: r["object"].get_name()),
                ("trigger.type", pa.string(), lambda r: r["object"].get_type()),
            ],
        ),
        "efield": _Group("efield", "efield.", efield_enums, efield_attributes("efield"), trace_dims=2),
        "sim_efield": _Group("sim_efield", "sim_efield.", efield_enums, efield_attributes("sim_efield"), trace_dims=2),
    }


def _event_groups():
    """The groups of the event level information."""
    shower_enums = [parameters.showerParameters]
    shower_attributes = lambda name: [  # noqa: E731
        (f"{name}.id", pa.int64(), lambda r: _id(r["object"].get_id(), f"{name} id")),
        (f"{name}.station_ids", pa.list_(pa.int64()), lambda r: _int_list(r["object"].get_station_ids(), "station id")),
    ]

    return {
        # Without prefix: the event file holds only events, the station rows add ``event.`` (see `EVENT_PREFIX`)
        "event": _Group("event", "", EVENT_ENUMS, [], aligned=False, attribute_prefix=""),
        "shower": _Group("shower", "shower.", shower_enums, shower_attributes("shower")),
        "sim_shower": _Group("sim_shower", "sim_shower.", shower_enums, shower_attributes("sim_shower")),
        "hybrid_shower": _Group(
            "hybrid_shower",
            "hybrid_shower.",
            shower_enums,
            [
                ("hybrid_shower.id", pa.int64(), lambda r: _id(r["object"].get_id(), "hybrid shower id")),
                ("hybrid_shower.name", pa.string(), lambda r: r["object"].get_name()),
            ],
        ),
        "particle": _Group(
            "particle",
            "particle.",
            [parameters.particleParameters],
            [("particle.id", pa.int64(), lambda r: _id(r["object"].get_id(), "particle id"))],
        ),
        "sim_emitter": _Group(
            "sim_emitter",
            "sim_emitter.",
            [parameters.emitterParameters],
            [
                ("sim_emitter.id", pa.int64(), lambda r: _id(r["object"].get_id(), "sim emitter id")),
                (
                    "sim_emitter.station_ids",
                    pa.list_(pa.int64()),
                    lambda r: _int_list(r["object"].get_station_ids(), "station id"),
                ),
            ],
        ),
    }


class _BatchedWriter:
    """Buffers rows (dicts: column name -> value) and writes them as row groups."""

    def __init__(self, path, schema, trace_columns, row_group_size, max_row_group_bytes, compression, warned):
        """
        Parameters
        ----------
        path : str
            Output file (opened immediately, overwritten if it exists).
        schema : pyarrow.Schema
            Schema of the file, defines the columns of the rows passed to `add`.
        trace_columns : set of str
            Columns holding (nested lists of) numpy arrays: converted without python lists, not dictionary
            encoded and without statistics.
        row_group_size : int
            Maximum number of rows per row group.
        max_row_group_bytes : int
            A row group is also written once the buffered traces exceed this size.
        compression : str
            Parquet compression codec.
        warned : set
            Columns with a conversion problem which were already warned about (shared between the files).
        """
        # Settings
        self.__schema = schema
        self.__trace_columns = trace_columns
        self.__row_group_size = row_group_size
        self.__max_row_group_bytes = max_row_group_bytes
        self.__warned = warned

        # Output file. The options take the paths of the leaf columns (e.g. ``channel_ids.list.element``)
        other_columns = [
            path for field in schema if field.name not in trace_columns for path in _leaf_paths(field.name, field.type)
        ]
        self.__sink = pa.OSFile(path, "wb")  # its position is the number of bytes written (see `size`)
        self.__writer = pq.ParquetWriter(
            self.__sink, schema, compression=compression, use_dictionary=other_columns, write_statistics=other_columns
        )

        # Buffered rows of the next row group
        self.__buffer = self.__new_buffer()
        self.__n_rows = 0
        self.__n_bytes = 0

        self.write_time = 0.0  #: seconds spent writing row groups and closing the file

    def add(self, row, n_bytes=0):
        """Append a row (dict: column name -> value, all columns of the schema)."""
        for name, value in row.items():
            self.__buffer[name].append(value)

        self.__n_rows += 1
        self.__n_bytes += n_bytes
        if self.__n_rows >= self.__row_group_size or self.__n_bytes >= self.__max_row_group_bytes:
            self.flush()

    def flush(self):
        """Write the buffered rows as one row group."""
        if self.__n_rows == 0:
            return

        start = timer()
        arrays = [self.__to_array(field) for field in self.__schema]
        self.__writer.write_table(pa.Table.from_arrays(arrays, schema=self.__schema))
        self.write_time += timer() - start

        self.__buffer = self.__new_buffer()
        self.__n_rows = 0
        self.__n_bytes = 0

    def size(self):
        """Estimated file size: bytes written plus the (uncompressed) bytes of the buffered traces."""
        return self.__sink.tell() + self.__n_bytes

    def close(self):
        """Flush and close the file (the footer is only written on close)."""
        self.flush()
        start = timer()
        self.__writer.close()
        self.__sink.close()
        self.write_time += timer() - start

    def __to_array(self, field):
        """
        The buffered values of `field` as arrow array. Only if the conversion fails (a value does not match
        the declared type), the values are checked one by one and the mismatching ones are written as null.
        """
        values = self.__buffer[field.name]
        if field.name in self.__trace_columns:
            return _nested_array(values, field.type)

        try:
            return pa.array(values, type=field.type)
        except (pa.ArrowException, TypeError, ValueError, OverflowError) as e:
            _warn_once(self.__warned, field.name, f"value does not match the type {field.type}, writing null: {e}")
            return pa.array([self.__checked(value, field.type) for value in values], type=field.type)

    @staticmethod
    def __checked(value, dtype):
        """`value` if it matches `dtype`, else None."""
        try:
            pa.scalar(value, type=dtype)
            return value
        except (pa.ArrowException, TypeError, ValueError, OverflowError):
            return None

    def __new_buffer(self):
        return {name: [] for name in self.__schema.names}


class eventWriter:
    """
    Writes complete events to parquet files (see the module documentation for the layout).

    Call `begin`, then `run` for each event, and finally `close` (or `end`, which is also called at exit).
    """

    def __init__(self):
        """Only sets the empty state: the configuration (and the schemas) are set in `begin`."""
        # Output files (set in `begin`)
        self.__filename = None
        self.__max_file_size = None
        self.__events_per_file = None
        self.__check_for_duplicates = False
        self.__overwrite = False
        self.__event_file = True

        # Row groups, compression and file metadata (set in `begin`)
        self.__row_group_size = None
        self.__max_row_group_bytes = None
        self.__compression = None
        self.__metadata = {}
        self.__umask = 0o022  # of the process, read in `begin` (for the permissions of the files, see `close`)

        # Open files and the events written to them
        self.__writers = {}  # "station" / "event" -> _BatchedWriter of the open files
        self.__pending = []  # (temporary file, target) of all files written, moved into place by `close`
        self.__part = 1  # number of the open file pair (1: no _part suffix)
        self.__events_in_file = 0
        self.__sim_config = None  # sim_config JSON in the metadata of the open files
        self.__last_event = None  # (run, event id) of the last event (its event row is written once)
        self.__seen_events = set()  # (run, event id) of all events, only with `check_for_duplicates`
        self.__event_stations = set()  # station ids written for the last event, only with `check_for_duplicates`

        # Statistics and warnings
        self.__n_events = 0
        self.__n_files = 0
        self.__write_time = 0.0
        self.__warned = set()

        # Column groups and schemas (built in `begin`)
        self.__station_groups = {}
        self.__event_groups = {}
        self.__station_schema = None
        self.__event_schema = None
        self.__trace_columns = set()

    def begin(
        self,
        filename,
        max_file_size=1024,
        check_for_duplicates=False,
        events_per_file=None,
        log_level=logging.NOTSET,
        *,
        event_file=True,
        row_group_size=100,
        max_row_group_bytes=256 * 2**20,
        compression="snappy",
        station_enums=None,
        channel_enums=None,
        overwrite=False,
        comment="",
        metadata=None,
    ):
        """
        Parameters
        ----------
        filename : str
            Output file ``<filename>.parquet`` (the extension ``.parquet`` may be omitted), the event file is
            ``<filename>_event.parquet``. The directory is created if needed. A ``.nur`` extension (deprecated)
            is replaced by ``.parquet``.
        max_file_size : float, default=1024
            Maximum size of a station file plus its event file in MB: once reached, the next event goes to a new
            file pair ``<filename>_part<nn>[_event].parquet``. Estimated from the bytes written and the buffered
            (uncompressed) traces, so the files usually stay smaller.
        check_for_duplicates : bool, default=False
            Raise a `ValueError` if an event (run, event id) is written again after another event, or a station
            of an event twice.
        events_per_file : int, optional
            Maximum number of events per file pair. Default: no limit.
        log_level : int, default=logging.NOTSET
            Use this to override the logging level for this module.

        Other Parameters
        ----------------
        event_file : bool, default=True
            Write the event level information to a separate event file (one row per event). If False, it is added
            to every station row (columns ``event.*``): one file, but repeated for multi-station events
            (count events with ``unique(["run", "event_id"])``) and events without station are not written.
        row_group_size : int, default=100
            Maximum number of rows per row group. Readers decode row groups in parallel and read them one by one.
        max_row_group_bytes : int, default=256 MiB
            A row group is also written once its traces exceed this size (a row group must stay below 2 GiB).
        compression : str, default="snappy"
            Parquet compression codec, e.g. "zstd" for smaller (and slower to write) files.
        station_enums : list of TypedEnum, optional
            Parameter enums stored as station (and sim station) columns. Default: ``stationParameters``,
            ``stationParametersRNOG``.
        channel_enums : list of TypedEnum, optional
            Parameter enums stored as channel (and sim / trigger channel) columns. Default: ``channelParameters``,
            ``channelParametersRNOG``.
        overwrite : bool, default=False
            If False, a `FileExistsError` is raised when a file to be written already exists.
        comment : str, default=""
            Free text stored in the file metadata (key ``comment``).
        metadata : dict of str, optional
            Additional key-value pairs (strings) stored in the file metadata.
        """
        logger.setLevel(log_level)
        self.close()  # a previous output of this instance
        if events_per_file is not None and events_per_file < 1:
            raise ValueError(f"events_per_file has to be a positive integer, got {events_per_file}.")

        filename = os.fspath(filename)
        if filename.endswith(".nur"):
            filename = filename[: -len(".nur")]
            _warn_deprecated(
                f"eventWriter writes parquet: {filename}.parquet instead of {filename}.nur "
                "(pass the name without the .nur extension)."
            )
        self.__filename = filename[: -len(".parquet")] if filename.endswith(".parquet") else filename

        leftovers = glob.glob(
            os.path.join(
                glob.escape(os.path.dirname(self.__filename) or "."), f".{glob.escape(os.path.basename(self.__filename))}*.tmp"
            )
        )
        if leftovers:
            logger.warning(
                f"Temporary files of an earlier run which did not finish (crash?): {sorted(leftovers)}. "
                "They are not used, remove them."
            )

        self.__max_file_size = max_file_size * 1024 * 1024  # in bytes
        self.__events_per_file = events_per_file
        self.__check_for_duplicates = check_for_duplicates
        self.__event_file = event_file

        self.__row_group_size = row_group_size
        self.__max_row_group_bytes = max_row_group_bytes
        self.__compression = compression
        self.__overwrite = overwrite
        self.__umask = _umask()

        # Reset the state of a previous output (see `__init__`)
        self.__writers = {}
        self.__pending = []
        self.__part = 1
        self.__events_in_file = 0
        self.__sim_config = None
        self.__last_event = None
        self.__seen_events = set()
        self.__event_stations = set()

        self.__n_events = 0
        self.__n_files = 0
        self.__write_time = 0.0

        # `is None` (not `or`) so that an empty list can be passed to store no parameters of that kind
        station_enums = DEFAULT_STATION_ENUMS if station_enums is None else station_enums
        channel_enums = DEFAULT_CHANNEL_ENUMS if channel_enums is None else channel_enums
        self.__station_groups = _station_groups(station_enums, channel_enums)
        self.__event_groups = _event_groups()

        # The reader rebuilds the column groups from these
        parameter_enums = json.dumps(
            {"station": [enum.__name__ for enum in station_enums], "channel": [enum.__name__ for enum in channel_enums]}
        )
        self.__build_schemas(comment, metadata or {}, parameter_enums)

    @register_run()
    def run(self, evt, station=None, mode=None, det=None):
        """
        Add an event (all stations, or only `station`) to the output.

        Parameters
        ----------
        evt : NuRadioReco.framework.event.Event
        station : NuRadioReco.framework.station.Station, optional
            Write only this station (the event level information is written once per event). Write all stations
            of an event in consecutive calls: a new file pair is only started between events.
        mode : dict or bool, optional
            Which traces to store. Default (None or True): all; False: none. A dict switches them per kind
            (missing keys: False): ``Channels`` (channels and their trigger channels), ``ElectricFields``
            (electric fields of the station), ``SimChannels``, ``SimElectricFields`` (of the sim station).
        det : detector, optional
            Deprecated and ignored: the detector description is not stored (the ``.nur`` writer stored it).
            A detector passed as second positional argument (old call ``run(evt, det)``) is ignored as well.
        """
        if station is not None and not isinstance(station, BaseStation):  # old signature run(evt, det, mode)
            station, det = None, station
        if det is not None and "det" not in self.__warned:
            self.__warned.add("det")
            _warn_deprecated(
                "eventWriter.run: the detector argument is ignored, the detector description is not "
                "stored in the file (load the detector separately when reading the events).",
                stacklevel=4,
            )

        mode = _trace_mode(mode)
        event_key = (evt.get_run_number(), evt.get_id())
        sim_config = evt.get_parameters(copy=False).get(parameters.eventParameters.sim_config)
        sim_config = None if sim_config is None else io_utilities._to_json(sim_config)
        stations = list(evt.get_stations()) if station is None else [station]

        new_event = event_key != self.__last_event
        if self.__check_for_duplicates:
            self.__check_duplicates(event_key, new_event, stations)

        # Build all rows first: an invalid value raises before anything of this event is buffered or recorded
        event_row, event_bytes = None, 0
        if (self.__event_file and new_event) or (not self.__event_file and stations):
            event_row, event_bytes = self.__event_row(evt)

        station_rows = []

        for station in stations:
            row, n_bytes = self.__station_row(evt, station, mode)
            if not self.__event_file:
                row.update(
                    {EVENT_PREFIX + column: value for column, value in event_row.items() if column not in ("run", "event_id")}
                )
                n_bytes += event_bytes
            station_rows.append((row, n_bytes))

        if self.__check_for_duplicates:
            if new_event:
                self.__seen_events.add(event_key)
                self.__event_stations = set()
            self.__event_stations.update(station.get_id() for station in stations)

        if new_event and (self.__event_file or stations):
            if self.__writers and self.__files_full():
                self.__close_files()
                self.__part += 1
            self.__events_in_file += 1

        if self.__event_file and new_event:
            self.__writer("event", sim_config).add(event_row, event_bytes)
        self.__last_event = event_key

        if not stations and not self.__event_file:
            _warn_once(self.__warned, "event", "events without station are not written with event_file=False")

        for row, n_bytes in station_rows:
            self.__writer("station", sim_config).add(row, n_bytes)

        if new_event:
            self.__n_events += 1

    def get_n_events(self):
        """Number of events written so far (several `run` calls for the stations of one event count once)."""
        return self.__n_events

    def close(self):
        """
        Flush remaining events, close all files, move them into place and log a summary. Safe to call several
        times and before `begin`. Call it to read the files before the program ends (else `end` does it at exit).

        The files get the permissions of the files they replace, new files those of `open` (``0o666 & ~umask``).
        Files of an earlier output with the same name which were not written now (e.g. ``<name>_part03.parquet``
        if the new output has two parts) are removed with ``overwrite=True``, else they are warned about.
        """
        if not self.__pending:
            return

        self.__close_files()

        for path, target in self.__pending:
            if path != target:
                os.chmod(path, _file_mode(target, self.__umask))
                os.replace(path, target)

        self.__remove_stale_files({target for _, target in self.__pending})
        self.__pending = []
        logger.log(
            LOGGING_STATUS,
            f"eventWriter: wrote {self.__n_events} events to {self.__n_files} files {self.__filename}*.parquet "
            f"(writing to disk: {self.__write_time:.2f}s).",
        )

    def end(self):
        """
        Close the files (see `close`) and return the number of events written. Called at exit (`end_all_modules`).

        If the program stopped with an exception, the output is incomplete: it is moved into place only if none
        of its files exists, else the existing files are kept and the closed temporary files are left for inspection.
        """
        if self.__pending and module.exiting_after_exception():
            self.__close_files()
            if any(os.path.exists(target) for path, target in self.__pending if path != target):
                logger.error(
                    "The program stopped with an exception: the existing files are kept, the incomplete "
                    f"output is in {[path for path, _ in self.__pending]}."
                )
                self.__pending = []
                return self.__n_events

            logger.error(f"The program stopped with an exception: {self.__filename}*.parquet is incomplete.")

        self.close()

        return self.__n_events

    def __remove_stale_files(self, written):
        """
        Remove (``overwrite=True``) or warn about the files of an earlier output with the same name which are not in
        `written`: parts up to the last one written (e.g. an event file) and all parts after it.
        """
        stale = []
        part = 1

        while True:
            paths = [path for path in (self.__path("station", part), self.__path("event", part)) if os.path.exists(path)]
            if part > self.__part and not paths:
                break

            stale += [path for path in paths if path not in written]
            part += 1

        if not stale:
            return

        if self.__overwrite:
            for path in stale:
                os.remove(path)

            logger.info(f"Removed the files of an earlier output: {stale}")
        else:
            logger.warning(
                f"Files of an earlier output exist: {stale}. Readers selecting {self.__filename}* will "
                "read them as well. Remove them or use overwrite=True."
            )

    def __check_duplicates(self, event_key, new_event, stations):
        """Raise a `ValueError` for an event written again after another event, or a station written twice."""
        if new_event and event_key in self.__seen_events:
            raise ValueError(f"Event (run, event_id) = {event_key} was already written (check_for_duplicates).")

        written = set() if new_event else self.__event_stations

        for station in stations:
            if station.get_id() in written:
                raise ValueError(
                    f"Station {station.get_id()} of event (run, event_id) = {event_key} was already "
                    "written (check_for_duplicates)."
                )

            written = written | {station.get_id()}

    def __build_schemas(self, comment, extra_metadata, parameter_enums):
        """Derive the (fixed) arrow schemas of the station and event files."""
        event_fields = (
            [pa.field("run", pa.int64()), pa.field("event_id", pa.int64())]
            + _time_fields("event_time")
            + [pa.field("modules", pa.string())]
        )

        for group in self.__event_groups.values():
            event_fields += group.fields()

        station_fields = [pa.field("run", pa.int64()), pa.field("event_id", pa.int64()), pa.field("station_id", pa.int64())]
        station_fields += (
            _time_fields("station_time")
            + _time_fields("sim_station.station_time")
            + [
                pa.field("trigger", pa.string()),  # name of the first (earliest) trigger, null if none fired
                pa.field("station.modules", pa.string()),
                pa.field("channel_ids", pa.list_(pa.int64())),
            ]
        )

        for group in self.__station_groups.values():
            station_fields += group.fields()

        groups = list(self.__station_groups.values())
        if not self.__event_file:
            station_fields += [field.with_name(EVENT_PREFIX + field.name) for field in event_fields[2:]]
            groups += list(self.__event_groups.values())

        self.__trace_columns = {group.trace_column for group in groups if group.trace_dims}
        self.__trace_columns |= {EVENT_PREFIX + column for column in self.__trace_columns} if not self.__event_file else set()

        metadata = {
            "nuradio_version": str(NuRadioReco.__version__),
            "nuradio_commit": version.get_NuRadioMC_commit_hash(),
            "comment": comment,
            "layout": "dual" if self.__event_file else "single",
            "parameter_enums": parameter_enums,
        }
        reserved = (set(metadata) | {"kind", "sim_config"}) & set(extra_metadata)
        if reserved:
            raise ValueError(f"The metadata keys {reserved} are reserved.")

        metadata.update(extra_metadata)
        self.__metadata = metadata

        self.__station_schema = pa.schema(station_fields)
        self.__event_schema = pa.schema(event_fields) if self.__event_file else None

    def __event_row(self, evt):
        """The event level columns (without the station ones) and the bytes of their traces."""
        row = {"run": _id(evt.get_run_number(), "run number", True), "event_id": _id(evt.get_id(), "event id", True)}
        row.update(_time_values("event_time", evt.get_event_time() if evt.has_event_time() else None))
        # Module instances are not stored (as in .nur)
        row["modules"] = io_utilities._to_json([[name, None, kwargs] for name, _, kwargs in evt.get_module_registry()])

        hybrid_showers = list(evt.get_hybrid_information().get_hybrid_showers())
        if any(shower.get_hybrid_detector() is not None for shower in hybrid_showers):
            _warn_once(self.__warned, "hybrid_shower.detector", "the hybrid detector of a hybrid shower is not stored")

        records = {
            "event": _record(evt),
            "shower": [_record(shower) for shower in evt.get_showers()],
            "sim_shower": [_record(shower) for shower in evt.get_sim_showers()],
            "hybrid_shower": [_record(shower) for shower in hybrid_showers],
            "particle": [_record(particle) for particle in evt.get_particles()],
            "sim_emitter": [_record(emitter) for emitter in evt.get_sim_emitters()],
        }
        n_bytes = 0

        for name, group in self.__event_groups.items():
            values, group_bytes = group.values(records[name], self.__warned)
            row.update(values)
            n_bytes += group_bytes

        return row, n_bytes

    def __station_row(self, evt, station, mode):
        """The row of a station file and the bytes of its traces."""
        station_id = station.get_id()
        row = {
            "run": _id(evt.get_run_number(), "run number", True),
            "event_id": _id(evt.get_id(), "event id", True),
            "station_id": _id(station_id, "station id", True),
        }
        row.update(_time_values("station_time", station.get_station_time()))

        fired = [
            trigger
            for trigger in station.get_triggers().values()
            if trigger.has_triggered() and trigger.get_trigger_time() is not None
        ]
        row["trigger"] = min(fired, key=lambda trigger: trigger.get_trigger_time()).get_name() if fired else None
        row["station.modules"] = io_utilities._to_json(
            [[index, name, None, kwargs] for index, name, _, kwargs in evt.get_module_registry(station_id)]
        )

        channels = list(station.iter_channels(sorted=True))
        row["channel_ids"] = [_id(channel.get_id(), "channel id", True) for channel in channels]
        trigger_channels = [
            _record(channel.get_trigger_channel(), mode["Channels"]) if channel.has_extra_trigger_channel() else None
            for channel in channels
        ]

        sim_station = station.get_sim_station()
        if sim_station is not None and sim_station.get_triggers():
            _warn_once(self.__warned, "sim_station.triggers", "triggers of the sim station are not stored")
        row.update(_time_values("sim_station.station_time", None if sim_station is None else sim_station.get_station_time()))

        records = {
            "station": _record(station),
            "sim_station": None if sim_station is None else _record(sim_station),
            "channel": [_record(channel, mode["Channels"]) for channel in channels],
            "trigger_channel": None if all(record is None for record in trigger_channels) else trigger_channels,
            "sim_channel": (
                []
                if sim_station is None
                else sorted(
                    (_record(channel, mode["SimChannels"]) for channel in sim_station.iter_channels()),
                    key=_sim_channel_sort_key,
                )
            ),
            "trigger": [_record(trigger) for trigger in station.get_triggers().values()],
            "efield": [_record(efield, mode["ElectricFields"]) for efield in station.get_electric_fields()],
            "sim_efield": (
                []
                if sim_station is None
                else [_record(efield, mode["SimElectricFields"]) for efield in sim_station.get_electric_fields()]
            ),
        }

        n_bytes = 0

        for name, group in self.__station_groups.items():
            values, group_bytes = group.values(records[name], self.__warned)
            row.update(values)
            n_bytes += group_bytes

        return row, n_bytes

    def __files_full(self):
        """True if the open file pair reached `max_file_size` or `events_per_file`."""
        if self.__events_per_file is not None and self.__events_in_file >= self.__events_per_file:
            return True

        return sum(writer.size() for writer in self.__writers.values()) >= self.__max_file_size

    def __path(self, kind, part):
        """The path of the station / event file of file pair `part` (counted from 1, no suffix for 1)."""
        name = self.__filename + (f"_part{part:02d}" if part > 1 else "")
        return f"{name}_event.parquet" if kind == "event" else f"{name}.parquet"

    def __writer(self, kind, sim_config):
        """
        The writer of the open station / event file (opened if needed).
        `sim_config`: JSON of the sim config of the event (stored in the metadata of a new file).
        """
        if kind not in self.__writers:
            if not self.__writers:
                self.__sim_config = sim_config
            self.__open_file(kind)
        elif sim_config != self.__sim_config:
            _warn_once(self.__warned, "sim_config", "differs between events of one file, only the first is stored")

        return self.__writers[kind]

    def __open_file(self, kind):
        """Open the station / event file of the current part (fails if it exists and `overwrite` is not set)."""
        path = self.__path(kind, self.__part)
        if os.path.exists(path) and not self.__overwrite:
            raise FileExistsError(f"{path} already exists. Use `overwrite=True` in `begin` to overwrite it.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        schema = self.__event_schema if kind == "event" else self.__station_schema
        metadata = dict(self.__metadata, kind=kind)
        if self.__sim_config is not None:
            metadata["sim_config"] = self.__sim_config

        temporary = _temporary_path(path)
        self.__pending.append((temporary, path))
        self.__writers[kind] = _BatchedWriter(
            temporary,
            schema.with_metadata(metadata),
            self.__trace_columns,
            self.__row_group_size,
            self.__max_row_group_bytes,
            self.__compression,
            self.__warned,
        )
        self.__n_files += 1
        logger.debug(f"Writing to {temporary} (-> {path})")

    def __close_files(self):
        """Flush and close the open files."""
        for writer in self.__writers.values():
            writer.close()
            self.__write_time += writer.write_time

        self.__writers = {}
        self.__events_in_file = 0


def _sim_channel_sort_key(record):
    """Deterministic order: channel id, shower id, ray tracing id (the latter two can be None)."""
    return tuple(-1 if x is None else x for x in record["object"].get_unique_identifier())
