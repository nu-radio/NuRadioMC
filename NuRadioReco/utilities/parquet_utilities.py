"""
Helpers to map typed parameter enums (see `NuRadioReco.framework.parameters`) to pyarrow (library to write parquet files).

The declared ``dtype`` of an enum member determines the column type. Parameterised types such as
``dict[int, dict[int, float]]`` or ``list[float]`` are mapped to native arrow ``map``/``list`` types,
a ``TypedDict`` (dict with fixed keys of different types) to an arrow ``struct`` (missing keys are null).
`from_arrow_value` converts the values read back (``to_pylist``) to the declared type.
Parameters whose type is not specific enough to define a schema (``Any``, bare ``dict``/``list``,
unknown classes) are dropped with a warning, nothing is pickled.

`find_files` finds the files written by `NuRadioReco.modules.io.eventWriter` and selects them by station and run.
It is shared by `NuRadioReco.modules.io.eventReader` (complete events) and
`NuRadioReco.utilities.analysis_reader.AnalysisReader` (tables for analyses, needs polars).
"""

import functools
import glob
import logging
import os
import re
import typing

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from NuRadioReco.framework.parameters import OneOf, _kind_of_value


logger = logging.getLogger("NuRadioReco.parquet_utilities")

_COMPLEX = pa.struct([("real", pa.float64()), ("imag", pa.float64())])

_SCALAR_TYPES = {
    bool: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    str: pa.string(),
    complex: _COMPLEX,
    np.ndarray: pa.list_(pa.float64()),
}


# Field metadata marking the struct of a `OneOf` type: one field per alternative, only the matching one is set
_ONE_OF = {"one_of": "1"}


def _is_one_of(arrow_type):
    """True for the struct type of a `OneOf` (see `to_arrow_type`)."""
    return pa.types.is_struct(arrow_type) and len(arrow_type) > 0 and arrow_type.field(0).metadata == {b"one_of": b"1"}


def _arrow_kind(arrow_type):
    """Container kind of an arrow type, as `NuRadioReco.framework.parameters._kind` of a declared type."""
    if pa.types.is_map(arrow_type):
        return "dict of dict" if pa.types.is_map(arrow_type.item_type) else "dict"

    if pa.types.is_list(arrow_type):
        return "sequence"

    return "scalar"


def _is_typeddict(dtype):
    """True for a `typing.TypedDict` class (`typing.is_typeddict` needs python >= 3.10)."""
    return isinstance(dtype, type) and issubclass(dtype, dict) and hasattr(dtype, "__total__")


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
    if isinstance(dtype, OneOf):
        fields = [(name, to_arrow_type(alternative)) for name, alternative in dtype.alternatives.items()]
        if all(field_type is not None for _, field_type in fields):
            return pa.struct([pa.field(name, field_type, metadata=_ONE_OF) for name, field_type in fields])

        return None

    if dtype in _SCALAR_TYPES:
        return _SCALAR_TYPES[dtype]

    if _is_typeddict(dtype):
        fields = [(name, to_arrow_type(field_type)) for name, field_type in typing.get_type_hints(dtype).items()]
        if fields and all(field_type is not None for _, field_type in fields):
            return pa.struct(fields)

        return None

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


class ElementTypeError(TypeError):
    """
    An element of a list, dict or struct with integer type which is no int: a float with a fractional part,
    NaN or inf, or a number outside the range of the type. Raised instead of writing the whole value as null
    (the writer does not catch it).
    """


def to_arrow_value(value, arrow_type, _element=False):
    """
    Convert a parameter value to a python object which ``pyarrow`` accepts for `arrow_type`.

    Raises a ValueError if a dict has keys which are not fields of a struct `arrow_type` (they would be lost)
    or if a value for an integer type is a float with a fractional part (pyarrow would truncate it silently),
    NaN or inf, or outside the range of the type; an `ElementTypeError` if such a value is an element of a
    list, dict or struct.
    """
    if value is None:
        return None

    if pa.types.is_integer(arrow_type):
        if not _element:
            return _to_int(value, arrow_type)

        try:
            return _to_int(value, arrow_type)
        except ValueError as e:
            raise ElementTypeError(f"element {e}") from None
    if arrow_type == _COMPLEX:
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}

    if _is_one_of(arrow_type):
        kind = _kind_of_value(value)
        field = next((field for field in arrow_type if _arrow_kind(field.type) == kind), None)
        if field is None:
            raise ValueError(f"{value!r} matches none of the alternatives {arrow_type}")

        return {field.name: to_arrow_value(value, field.type, _element)}

    if pa.types.is_struct(arrow_type):
        names = {field.name for field in arrow_type}
        unknown = set(value) - names
        if unknown:
            raise ValueError(f"keys {sorted(unknown)} are not fields of {arrow_type}")

        return {field.name: to_arrow_value(value.get(field.name), field.type, True) for field in arrow_type}

    if pa.types.is_map(arrow_type):
        return [
            (to_arrow_value(k, arrow_type.key_type, True), to_arrow_value(v, arrow_type.item_type, True))
            for k, v in value.items()
        ]

    if pa.types.is_list(arrow_type) and (
        pa.types.is_nested(arrow_type.value_type) or pa.types.is_integer(arrow_type.value_type)
    ):
        return [to_arrow_value(v, arrow_type.value_type, True) for v in value]

    return _to_python(value)


def to_numpy_dtype(arrow_type):
    """
    The numpy dtype of a numeric or boolean arrow type, else ``object``.

    Unlike ``pyarrow.DataType.to_pandas_dtype``, it does not need pandas.
    """
    if pa.types.is_integer(arrow_type):
        return np.dtype(f"{'int' if pa.types.is_signed_integer(arrow_type) else 'uint'}{arrow_type.bit_width}")

    if pa.types.is_floating(arrow_type):
        return np.dtype(f"float{arrow_type.bit_width}")

    if pa.types.is_boolean(arrow_type):
        return np.dtype(bool)

    return np.dtype(object)


def _to_int(value, arrow_type):
    """``value`` as python int (a float only if it is a whole number), ValueError if outside ``arrow_type``."""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"{float(value)} is not an int")

        if not float(value).is_integer():
            raise ValueError(f"{float(value)} is not a whole number")

        value = int(value)
    value = _to_python(value)
    if isinstance(value, int):
        info = np.iinfo(to_numpy_dtype(arrow_type))
        if not info.min <= value <= info.max:
            raise ValueError(f"{value!r} is outside the range of {arrow_type}")

    return value


def _to_python(value):
    """Recursively convert numpy types to builtin python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return [(_to_python(k), _to_python(v)) for k, v in value.items()]

    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]

    return value


def from_arrow_value(value, dtype):
    """
    Convert a value read from parquet (python object as returned by ``pyarrow`` ``to_pylist``) to `dtype`.

    Reverse of `to_arrow_value`: maps (lists of key-value tuples or dicts) become dicts, ``np.ndarray`` becomes
    a float array, ``complex`` a complex number, ``tuple[...]`` a tuple. Null struct fields are left out of
    the dict (the key was missing when writing). Numpy scalars come back as python scalars.

    Parameters
    ----------
    value : object
        The value of one cell.
    dtype : type or typing generic
        The declared type of the parameter (``member.dtype``).
    """
    return None if value is None else _converter(dtype)(value)


def _identity(value):
    return value


@functools.lru_cache(maxsize=None)
def _converter(dtype):
    """The function converting a (non-null) value read from parquet to `dtype` (see `from_arrow_value`)."""
    if dtype is np.ndarray:
        return lambda value: np.asarray(value, dtype=float)

    if dtype is complex:
        return lambda value: complex(value["real"], value["imag"])

    if isinstance(dtype, OneOf):
        converters = {name: _converter(alternative) for name, alternative in dtype.alternatives.items()}

        def convert_one_of(value):
            name = next((name for name, item in value.items() if item is not None), None)
            return None if name is None else converters[name](value[name])

        return convert_one_of

    if _is_typeddict(dtype):
        converters = {key: _converter(hint) for key, hint in typing.get_type_hints(dtype).items()}
        return lambda value: {k: converters[k](v) for k, v in value.items() if v is not None}

    origin = typing.get_origin(dtype)
    args = typing.get_args(dtype)
    if origin in (list, tuple) and len(args) >= 1:
        item = _optional(_converter(args[0]))
        if origin is list:
            return lambda value: [item(v) for v in value]

        return lambda value: tuple(item(v) for v in value)

    if origin is dict and len(args) == 2:
        key, item = _optional(_converter(args[0])), _optional(_converter(args[1]))

        def convert(value):
            return {key(k): item(v) for k, v in (value.items() if isinstance(value, dict) else value)}

        return convert

    return _identity


def _optional(converter):
    """`converter` which passes None through (elements of lists and maps can be null)."""
    if converter is _identity:
        return converter

    return lambda value: None if value is None else converter(value)


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
                logger.warning(
                    f"Parameter {name} has the type {member.dtype}, which can not be stored in parquet. Dropping it."
                )
                continue

            columns[prefix + name] = (member, arrow_type)

    return columns


_KEYS = {"station": ["run", "event_id", "station_id"], "event": ["run", "event_id"]}
_SORT_KEYS = {"station": ["station_id", "run", "event_id"], "event": ["run", "event_id"]}

EVENT_PREFIX = "event."  #: prefix of the event level columns in the station rows (``event_file=False`` or joined)


def is_trace_column(name):
    """True for the trace columns (``channel.trace``, ``sim_efield.trace``, ...)."""
    return name.endswith(".trace")


def event_file_of(station_file):
    """The event file paired with a station file (same base name): ``<name>.parquet`` -> ``<name>_event.parquet``."""
    station_file = os.fspath(station_file)

    return station_file[: -len(".parquet")] + "_event.parquet"


def file_kind(file):
    """``station`` / ``event`` (metadata key ``kind``) of a file of eventWriter, None for other files."""
    return _file_metadata(file).get("kind")


def _as_set(values):
    """Convert an int or an iterable of ints to a set (None stays None, i.e. no selection)."""
    if values is None:
        return None

    return {int(values)} if isinstance(values, (int, float)) else {int(value) for value in values}


def _glob(path):
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


def _may_contain(file, selection):
    """
    False if the row group statistics (min / max) show that no row group of `file` contains the selected values
    of all columns in `selection` (dict column -> set of values, None: no selection). Missing columns or
    statistics count as "may contain".
    """
    selection = {column: values for column, values in selection.items() if values is not None}
    if not selection:
        return True

    metadata = pq.read_metadata(file)
    leaves = {metadata.schema.column(i).path: i for i in range(len(metadata.schema))}

    for i_group in range(metadata.num_row_groups):
        group = metadata.row_group(i_group)

        for column, values in selection.items():
            if column not in leaves:
                continue

            statistics = group.column(leaves[column]).statistics
            if (
                statistics is not None
                and statistics.has_min_max
                and not any(statistics.min <= value <= statistics.max for value in values)
            ):
                break  # no selected value in this row group
        else:
            return True

    return False


def _file_order(file):
    """Sort key: numbers compare numerically (``x.parquet``, ``x_part02.parquet``, ..., ``x_part100.parquet``)."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", file)]


def find_files(path, stations=None, runs=None):
    """
    The station files of `path` which may contain the selected stations and runs, and their event files.

    Files are recognised by their metadata (key ``kind``), other parquet files are skipped with a warning.
    The event file of a station file (``<name>_event.parquet`` next to ``<name>.parquet``) is also found if
    `path` matches only the station file. Files are left out if the row group statistics of their ``station_id``
    and ``run`` columns exclude the selection (only the footers are read); the selected files may still contain
    other stations and runs, i.e., the rows have to be filtered as well.

    Parameters
    ----------
    path : str or list of str
        Directory (searched recursively), file or glob pattern, or a list of these.
    stations, runs : int or list of int, optional
        Station ids / run numbers to select. Default: all.

    Returns
    -------
    dict
        ``{"station": [files], "event": [files]}``, sorted by file name (numbers compared numerically).
    """
    stations, runs = _as_set(stations), _as_set(runs)
    found = {"station": set(), "event": set()}

    for file in _glob(path):
        kind = file_kind(file)
        if kind not in found:
            logger.warning(f"Skipping {file}: not written by eventWriter (no 'kind' in the metadata).")
            continue

        found[kind].add(file)

    station_files = {file for file in found["station"] if _may_contain(file, {"station_id": stations, "run": runs})}
    event_files = {event_file_of(file) for file in found["station"]}
    event_files = {file for file in event_files if os.path.exists(file)} | found["event"]
    if stations is not None:  # only the event files of the selected station files
        event_files &= {event_file_of(file) for file in station_files}
    event_files = {file for file in event_files if _may_contain(file, {"run": runs})}

    return {"station": sorted(station_files, key=_file_order), "event": sorted(event_files, key=_file_order)}


def _file_metadata(file):
    """The key-value metadata of a parquet file (str -> str)."""
    metadata = pq.read_metadata(file).metadata or {}

    return {key.decode(): value.decode() for key, value in metadata.items()}
