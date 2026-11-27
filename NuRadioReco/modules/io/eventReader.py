"""
Reads the parquet files of `NuRadioReco.modules.io.eventWriter` back into complete
`NuRadioReco.framework.event.Event` objects. ``.nur`` files (deprecated) are passed on to
`NuRadioReco.modules.io.eventReaderNur` with a warning; convert them with ``nuradio-analyse convert``.

The events are returned in file order (as for ``.nur`` files): the order of the event rows
(files sorted by name, numbers numerically: ``<name>.parquet``, ``<name>_part02.parquet``, ...,
``<name>_part10.parquet``), that is the order in which `eventWriter` received them. This way, consecutive events
come from consecutive row groups, each decoded once. They are read in batches of row groups to bound the memory.
To sort by (run, event id), use `NuRadioReco.utilities.analysis_reader.AnalysisReader`.
`get_event_i` gives random access to single events (e.g. for the eventbrowser).
For each event, the event row (event file, or the ``event.*`` columns with ``event_file=False``) is combined
with the rows of all selected stations of this event.

The objects are built with the constructors and public setters of the framework classes (nothing is pickled),
so the events are restored as from a ``.nur`` file.
Not restored: triggers of sim stations, the hybrid detector of hybrid showers.
A null parameter value means "not set", parameters set to None are restored as None.
For a column analysis without building events, use `NuRadioReco.utilities.analysis_reader.AnalysisReader`.
"""

import collections
import functools
import json
import logging
import os
import warnings
from timeit import default_timer as timer

import astropy.time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from NuRadioReco.framework import parameters
from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.electric_field import ElectricField
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.hybrid_shower import HybridShower
from NuRadioReco.framework.particle import Particle
from NuRadioReco.framework.radio_shower import RadioShower
from NuRadioReco.framework.sim_channel import SimChannel
from NuRadioReco.framework.sim_emitter import SimEmitter
from NuRadioReco.framework.sim_station import SimStation
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.trigger import create_trigger
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.io import eventWriter as writer
from NuRadioReco.modules.io.eventReaderNur import eventReaderNur
from NuRadioReco.utilities import analysis_reader
from NuRadioReco.utilities import io_utilities
from NuRadioReco.utilities import parquet_utilities as pu

logger = logging.getLogger("NuRadioReco.eventReader")


def _to_arrays(array):
    """
    Nested arrow list array (e.g. a trace column) -> python list with one entry per element: None, a list
    (outer levels) or a numpy array (innermost level, a read-only view of the arrow buffer).
    """
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    valid = np.asarray(array.is_valid())
    offsets = np.asarray(array.offsets)
    if pa.types.is_list(array.type.value_type):
        children = _to_arrays(array.values)
    else:
        children = array.values.to_numpy(zero_copy_only=False)

    return [children[offsets[i] : offsets[i + 1]] if valid[i] else None for i in range(len(array))]


def _time(row, name):
    """The `astropy.time.Time` of the columns ``<name>_jd1/jd2/scale`` (None if null)."""
    jd1 = row.get(f"{name}_jd1")
    if jd1 is None:
        return None

    return astropy.time.Time(jd1, row[f"{name}_jd2"], format="jd", scale=row[f"{name}_scale"])


@functools.lru_cache(maxsize=None)
def _member(name):
    """The parameter enum member of a name ``<enum class>.<member>``, e.g. ``stationParameters.zenith``."""
    enum_name, _, member = name.partition(".")
    return getattr(parameters, enum_name)[member]


@functools.lru_cache(maxsize=None)
def _parameter_keys(group, prefix):
    """(column, enum member, dtype or None if the value needs no conversion) of the parameters of `group`."""
    return [
        (prefix + column, member, None if member.dtype in (bool, int, float, str) else member.dtype)
        for column, (member, _) in group.parameter_columns.items()
    ]


class _Objects:
    """
    Access to the objects of one `eventWriter._Group` in a row: ``get(column, i)`` and restoring
    the parameters and traces of object `i` (``restore``).
    """

    def __init__(self, group, row, prefix=""):
        """`row`: dict column -> value of one file row; `prefix`: prefix of the columns (``event.`` or "")."""
        self.group = group
        self.row = row
        self.prefix = prefix
        types = self.get(group.column("parameter_types"))
        self.n = 0 if types is None else len(types) if group.aligned else 1
        # Only the parameter columns with values in this row
        self.parameters = [
            (row[column], member, dtype)
            for column, member, dtype in _parameter_keys(group, prefix)
            if row.get(column) is not None
        ]

    def get(self, column, i=None):
        value = self.row.get(self.prefix + column)
        if i is None or not self.group.aligned:
            return value

        return None if value is None else value[i]

    def restore(self, obj, i=None):
        """
        Set the parameter types, parameters (incl. those set to None), covariances and the trace (if the group has one) of
        object `i` (aligned group) or of the object of the row on `obj`. Returns `obj`.
        """
        column = self.group.column
        present = obj.get_parameter_types()

        for type_name in self.get(column("parameter_types"), i) or []:
            parameter_type = getattr(parameters, type_name)
            if parameter_type not in present:
                obj.add_parameter_type(parameter_type)

        aligned = self.group.aligned

        for value, member, dtype in self.parameters:
            if aligned:
                value = value[i]
            if value is not None:
                obj.set_parameter(member, value if dtype is None else pu.from_arrow_value(value, dtype))

        for member_name in self.get(column("none_parameters"), i) or []:
            obj.set_parameter(_member(member_name), None)

        for covariance in self.get(column("covariances"), i) or []:
            value = covariance["value"]
            if value is None and covariance.get("values") is not None:  # array valued parameter
                value = np.array(covariance["values"])
            obj.set_parameter_covariance(_member(covariance["a"]), _member(covariance["b"]), value)

        trace = self.get(column("trace"), i) if self.group.trace_dims else None
        if trace is not None:
            if isinstance(trace, list):  # electric field: list of the components
                trace = np.array(trace)
            obj.set_trace(
                trace, self.get(column("sampling_rate"), i), trace_start_time=self.get(column("trace_start_time"), i)
            )

        return obj


def _channels(objects, id_column, cls, *ids):
    """
    The channels of `objects` (`Channel` or `SimChannel` per `cls`). `ids`: further id columns (sim channels:
    shower and ray tracing id) passed to `cls` after the channel id. Missing entries (null id) are None.
    """
    name = objects.group.name
    channels = []

    for i in range(objects.n):
        channel_id = objects.get(id_column, i)
        if channel_id is None:
            channels.append(None)
            continue

        channel = cls(
            channel_id, *[objects.get(column, i) for column in ids], channel_group_id=objects.get(f"{name}.group_id", i)
        )
        channels.append(objects.restore(channel, i))

    return channels


def _efields(objects):
    name = objects.group.name
    efields = []

    for i in range(objects.n):
        position = objects.get(f"{name}.position", i)
        efield = ElectricField(
            objects.get(f"{name}.channel_ids", i),
            shower_id=objects.get(f"{name}.shower_id", i),
            ray_tracing_id=objects.get(f"{name}.ray_tracing_id", i),
        )
        efield.set_position(None if position is None else np.array(position))
        efields.append(objects.restore(efield, i))

    return efields


def _restore_base_station(station, objects, time, efields, triggers=None):
    """Parameters, station time, particle type, triggers, trigger flag and electric fields of a (sim) station."""
    name = objects.group.name
    objects.restore(station)
    if time is not None:
        station.set_station_time(time)

    for i in range(0 if triggers is None else triggers.n):
        trigger = create_trigger(triggers.get("trigger.name", i), triggers.get("trigger.type", i))
        station.set_trigger(triggers.restore(trigger, i))

    particle_type = objects.get(f"{name}.particle_type")
    if particle_type == "nu":
        station.set_is_neutrino()
    elif particle_type == "cr":
        station.set_is_cosmic_ray()
    elif particle_type:
        logger.warning(f"Unknown particle type {particle_type!r} of station {station.get_id()}, not set.")

    triggered = objects.get(f"{name}.triggered")
    if triggered is not None:
        station.set_triggered_flag(triggered)

    for efield in _efields(efields):
        station.add_electric_field(efield)


def _station(row, groups):
    """The `Station` of a station row (`groups`: the groups of its file)."""

    def objects(name):
        return _Objects(groups[name], row)

    station = Station(row["station_id"])
    _restore_base_station(station, objects("station"), _time(row, "station_time"), objects("efield"), objects("trigger"))
    reference_reconstruction = row["station.reference_reconstruction"]
    if reference_reconstruction is not None:
        station.set_reference_reconstruction(reference_reconstruction)

    channels = objects("channel")
    trigger_channels = _channels(objects("trigger_channel"), "trigger_channel.id", Channel)

    for i, channel_id in enumerate(row["channel_ids"] or []):
        channel = channels.restore(Channel(channel_id, channel_group_id=channels.get("channel.group_id", i)), i)
        if i < len(trigger_channels) and trigger_channels[i] is not None:
            channel.set_trigger_channel(trigger_channels[i])
        station.add_channel(channel)

    sim_objects = objects("sim_station")
    if sim_objects.n:
        sim_station = SimStation(sim_objects.get("sim_station.station_id"))
        _restore_base_station(sim_station, sim_objects, _time(row, "sim_station.station_time"), objects("sim_efield"))
        magnetic_field = sim_objects.get("sim_station.magnetic_field_vector")
        sim_station.set_magnetic_field_vector(None if magnetic_field is None else np.array(magnetic_field))
        sim_station.set_simulation_weight(sim_objects.get("sim_station.simulation_weight"))

        for channel in _channels(
            objects("sim_channel"), "sim_channel.channel_id", SimChannel, "sim_channel.shower_id", "sim_channel.ray_tracing_id"
        ):
            sim_station.add_channel(channel)

        station.set_sim_station(sim_station)

    return station


def _event(row, prefix, groups, station_rows, sim_config):
    """
    The `Event` from the event level columns of `row` (columns with `prefix`) and the station rows
    (list of (row, groups of its file)).
    """

    def objects(name):
        return _Objects(groups[name], row, prefix)

    evt = Event(row["run"], row["event_id"])
    objects("event").restore(evt)
    if sim_config is not None:
        evt.set_parameter(parameters.eventParameters.sim_config, io_utilities._from_json(sim_config))
    event_time = _time(row, prefix + "event_time")
    if event_time is not None:
        evt.set_event_time(event_time)

    modules = row.get(prefix + "modules")
    if modules is not None:
        evt._set_module_registry(io_utilities._from_json(modules))

    for station_row, file_groups in station_rows:
        station = _station(station_row, file_groups)
        evt.set_station(station)
        modules = io_utilities._from_json(station_row["station.modules"])
        if modules:
            evt._set_module_registry(modules, station.get_id())

    for name, add in [("shower", evt.add_shower), ("sim_shower", evt.add_sim_shower)]:
        showers = objects(name)

        for i in range(showers.n):
            add(
                showers.restore(
                    RadioShower(showers.get(f"{name}.id", i), station_ids=showers.get(f"{name}.station_ids", i)), i
                )
            )

    emitters = objects("sim_emitter")

    for i in range(emitters.n):
        evt.add_sim_emitter(
            emitters.restore(
                SimEmitter(emitters.get("sim_emitter.id", i), station_ids=emitters.get("sim_emitter.station_ids", i)), i
            )
        )

    particles = objects("particle")

    for i in range(particles.n):
        evt.add_particle(particles.restore(Particle(particles.get("particle.id", i)), i))

    # The id of a hybrid shower is always 0 (`HybridShower` sets no other)
    hybrid = objects("hybrid_shower")

    for i in range(hybrid.n):
        evt.get_hybrid_information().add_hybrid_shower(hybrid.restore(HybridShower(hybrid.get("hybrid_shower.name", i)), i))

    return evt


class _File:
    """One parquet file: its groups (from the metadata), row groups and decoded row groups."""

    def __init__(self, path, kind, read_traces):
        self.path = path
        self.kind = kind
        self.file = pq.ParquetFile(path)
        metadata = {key.decode(): value.decode() for key, value in (self.file.schema_arrow.metadata or {}).items()}
        self.layout = metadata.get("layout", "dual")
        self.sim_config = metadata.get("sim_config")

        enums = json.loads(metadata.get("parameter_enums", "{}"))
        station_enums = self.__enums(enums.get("station"), writer.DEFAULT_STATION_ENUMS)
        channel_enums = self.__enums(enums.get("channel"), writer.DEFAULT_CHANNEL_ENUMS)
        self.station_groups = writer._station_groups(station_enums, channel_enums)
        self.event_groups = writer._event_groups()

        names = self.file.schema_arrow.names
        self.columns = [name for name in names if read_traces or not pu.is_trace_column(name)]
        self.trace_columns = {name for name in self.columns if pu.is_trace_column(name)}
        n_rows = [self.file.metadata.row_group(i).num_rows for i in range(self.file.num_row_groups)]
        self.row_group_starts = np.cumsum([0] + n_rows)

    def __enums(self, names, default):
        if names is None:
            return default

        enums = []

        for name in names:
            enum = getattr(parameters, name, None)
            if enum is None:
                logger.warning(f"{self.path}: unknown parameter enum {name}, its columns are not read.")
            else:
                enums.append(enum)

        return enums

    def keys(self):
        """The key columns of all rows (numpy arrays)."""
        keys = ["run", "event_id"] + (["station_id"] if self.kind == "station" else [])
        table = self.file.read(columns=keys)

        return {key: table[key].to_numpy() for key in keys}

    def read_row_group(self, index):
        """The rows of a row group as list of dicts (column -> value). Traces as numpy arrays."""
        return self.rows(self.read_table(index))

    def read_table(self, index):
        """A row group as arrow table (the selected columns)."""
        return self.file.read_row_group(index, columns=self.columns)

    def rows(self, table):
        """The rows of a table of `read_table` (or a slice of it) as list of dicts, see `read_row_group`."""
        columns = {
            name: _to_arrays(table[name]) if name in self.trace_columns else table[name].to_pylist() for name in self.columns
        }
        return [{name: values[i] for name, values in columns.items()} for i in range(table.num_rows)]


def _is_nur(path):
    return os.fspath(path).endswith((".nur", ".ari"))


class eventReader:
    """
    Reads the parquet files of `eventWriter` and returns complete `Event` objects (see the module documentation).
    ``.nur`` files are read with `eventReaderNur` (deprecated).

    Call `begin`, iterate over `run`, and finally `close` (or `end`, which is also called at exit).
    """

    def __init__(self):
        # Index of the selected files and events (built in `begin`)
        self.__files = []
        self.__events = None  # per event (file order): event row (file, row) or None
        self.__station_rows = None  # per event: list of (file, row)
        self.__event_ids = None  # per event: (run, event id)

        # Reading
        self.__tables = collections.OrderedDict()  # (file, row group) -> arrow table, for `get_event_i`
        self.__batch_size = None
        self.__read_time = 0.0
        self.__selection = None  # arguments of `AnalysisReader` for `get_header`
        self.__nur = None  # `eventReaderNur` reading .nur files

    def begin(
        self,
        filename,
        read_detector=False,
        log_level=logging.NOTSET,
        *,
        stations=None,
        runs=None,
        read_traces=True,
        batch_size=100,
    ):
        """
        Parameters
        ----------
        filename : str or list of str
            Directories (searched recursively), files or glob patterns (see `parquet_utilities.find_files`).
            Event files are also found next to the selected station files.
            ``.nur`` files (deprecated) are read with `eventReaderNur` (with ``read_detector`` and ``log_level``);
            they can not be mixed with parquet files.
        read_detector : bool, default=False
            Only for ``.nur`` files: parse the detector description stored in the files (see `get_detector`).
            Parquet files store no detector.
        log_level : int, default=logging.NOTSET
            Use this to override the logging level for this module.

        Other Parameters
        ----------------
        stations : int or list of int, optional
            Read only these stations; events without one of them are skipped. Default: all.
            Stations and runs are selected by their columns, files whose row group statistics exclude the
            selection are not read (see `parquet_utilities.find_files`).
        runs : int or list of int, optional
            Read only these runs. Default: all.
        read_traces : bool, default=True
            Read the traces (channels, sim channels, electric fields). Without them, reading is much faster.
        batch_size : int, default=100
            Number of events read at once (the row groups holding them are decoded together).

        Raises
        ------
        FileNotFoundError
            If no file matches.
        ValueError
            If an event (station) occurs more than once in the selected files, or ``.nur`` and parquet files
            are mixed.
        """
        logger.setLevel(log_level)
        self.close()
        self.__nur = None
        entries = [filename] if isinstance(filename, (str, os.PathLike)) else list(filename)
        is_nur = [_is_nur(entry) for entry in entries]
        if any(is_nur):
            if not all(is_nur):
                raise ValueError(
                    f"eventReader: .nur and parquet files can not be read together: {entries}. "
                    "Convert the .nur files with `nuradio-analyse convert`."
                )
            if stations is not None or runs is not None or not read_traces:
                raise ValueError("eventReader: stations, runs and read_traces are only supported for parquet files.")

            warnings.warn(
                "eventReader: the .nur format is deprecated, convert the files with "
                "`nuradio-analyse convert` (they are read with eventReaderNur for now).",
                FutureWarning,
                stacklevel=2,
            )
            self.__nur = eventReaderNur()
            self.__nur.begin(filename, read_detector=read_detector, log_level=log_level)
            return

        if read_detector:
            warnings.warn(
                "eventReader: read_detector is ignored, parquet files store no detector " "(get_detector returns None).",
                FutureWarning,
                stacklevel=2,
            )

        found = pu.find_files(filename, stations, runs)
        if not found["station"] and not found["event"]:
            raise FileNotFoundError(f"No matching parquet files found for {filename}")

        self.__selection = {"path": filename, "stations": stations, "runs": runs}
        self.__batch_size = batch_size
        self.__read_time = 0.0
        self.__files = [_File(path, kind, read_traces) for kind in ["event", "station"] for path in found[kind]]
        self.__tables.clear()
        if stations is not None and not found["station"]:
            logger.warning("No station file of the selected stations found.")

        self.__index(pu._as_set(runs), pu._as_set(stations))
        logger.info(f"Found {len(self.__events)} events in {len(self.__files)} files.")

    def __index(self, runs, station_ids):
        """
        Assign the event and station rows (of the selected runs and stations) to events, numbered in file order:
        by the first event row, else the first station row (the event files come first in `self.__files`).
        """
        rows = {"event": [], "station": []}

        for i_file, file in enumerate(self.__files):
            keys = file.keys()
            selected = np.ones(len(keys["run"]), dtype=bool) if runs is None else np.isin(keys["run"], list(runs))
            if station_ids is not None and file.kind == "station":
                selected &= np.isin(keys["station_id"], list(station_ids))
            row_index = np.nonzero(selected)[0]
            columns = [np.full(len(row_index), i_file), row_index, keys["run"][selected], keys["event_id"][selected]]
            if file.kind == "station":
                columns.append(keys["station_id"][selected])
            rows[file.kind].append(np.array(columns, dtype=np.int64).reshape(len(columns), -1))

        events = np.concatenate(rows["event"] or [np.empty((4, 0), dtype=np.int64)], axis=1)
        stations = np.concatenate(rows["station"] or [np.empty((5, 0), dtype=np.int64)], axis=1)

        # Event of each row: index of its (run, event id) in the unique keys
        unique_keys, inverse = np.unique(np.concatenate([events[2:4], stations[2:4]], axis=1), axis=1, return_inverse=True)
        inverse = inverse.reshape(-1)
        selected = np.ones(unique_keys.shape[1], dtype=bool)
        if station_ids is not None:  # only events with a selected station
            selected[:] = False
            selected[inverse[events.shape[1] :]] = True

        # Number the selected events by their first row (rows are concatenated in file order)
        first_row = np.full(unique_keys.shape[1], len(inverse))
        np.minimum.at(first_row, inverse, np.arange(len(inverse)))
        order = np.argsort(np.where(selected, first_row, len(inverse) + 1), kind="stable")[: selected.sum()]
        number_of_key = np.full(unique_keys.shape[1], -1)
        number_of_key[order] = np.arange(len(order))
        numbers = number_of_key[inverse]
        unique_keys = unique_keys[:, order]
        event_numbers, station_numbers = numbers[: events.shape[1]], numbers[events.shape[1] :]

        self.__event_ids = unique_keys.T
        self.__events = [None] * unique_keys.shape[1]

        for i_file, row, number in zip(events[0], events[1], event_numbers):
            if number < 0:
                continue  # an event without the selected stations
            if self.__events[number] is not None:
                raise ValueError(
                    f"Event (run, event_id) = {tuple(unique_keys[:, number])} occurs more than once in the event files "
                    f"{self.__files[self.__events[number][0]].path} and {self.__files[i_file].path}. "
                    "Select the files with a more specific path / glob pattern."
                )

            self.__events[number] = (int(i_file), int(row))

        self.__station_rows = [[] for _ in range(unique_keys.shape[1])]

        for index in np.lexsort((stations[4], station_numbers)):  # stations of an event sorted by id
            i_file, row, station_id, number = (
                stations[0, index],
                stations[1, index],
                stations[4, index],
                station_numbers[index],
            )
            station_rows = self.__station_rows[number]
            if station_rows and station_rows[-1][2] == station_id:
                raise ValueError(
                    f"Station {station_id} of event (run, event_id) = {tuple(unique_keys[:, number])} occurs more than "
                    f"once: {self.__files[station_rows[-1][0]].path} and {self.__files[i_file].path}. "
                    "Select the files with a more specific path / glob pattern."
                )

            station_rows.append((int(i_file), int(row), int(station_id)))

    def get_n_events(self):
        """The number of events (after the selection of stations and runs)."""
        if self.__nur is not None:
            return self.__nur.get_n_events()

        return len(self.__events)

    @register_run()
    def run(self):
        """Generator of the `Event` objects, in file order (see the module documentation)."""
        if self.__nur is not None:
            return self.__nur.run()

        return self.__iter_events()

    def __iter_events(self):
        cache = {}  # (file, row group) -> rows

        for start in range(0, len(self.__events), self.__batch_size):
            batch = range(start, min(start + self.__batch_size, len(self.__events)))
            t0 = timer()
            needed = set()

            for number in batch:
                rows = self.__station_rows[number] + ([self.__events[number]] if self.__events[number] else [])
                needed.update((i_file, self.__row_group(i_file, row)) for i_file, row, *_ in rows)

            for key in set(cache) - needed:  # bounded memory: only the row groups of this batch
                del cache[key]

            for i_file, row_group in sorted(needed - set(cache)):
                cache[(i_file, row_group)] = self.__files[i_file].read_row_group(row_group)

            self.__read_time += timer() - t0

            for number in batch:
                yield self.__event(number, lambda i_file, row: self.__row(cache, i_file, row))

    def get_event_ids(self):
        """The (run, event id) of the events, array of shape (n_events, 2), in the order of `run`."""
        return self.__event_ids

    def get_event_i(self, number, cache_size=4):
        """
        Random access: the `Event` with index `number` (order of `run`).

        Decodes only the rows of this event; the arrow tables of the last `cache_size` row groups are kept,
        so neighbouring events are read without decoding their row groups again.
        """

        def row(i_file, row):
            row_group = self.__row_group(i_file, row)
            key = (i_file, row_group)
            if key in self.__tables:
                self.__tables.move_to_end(key)
            else:
                t0 = timer()
                self.__tables[key] = self.__files[i_file].read_table(row_group)
                self.__read_time += timer() - t0

                while len(self.__tables) > cache_size:
                    self.__tables.popitem(last=False)
            offset = row - self.__files[i_file].row_group_starts[row_group]

            return self.__files[i_file].rows(self.__tables[key].slice(offset, 1))[0]

        return self.__event(number, row)

    def __row_group(self, i_file, row):
        return int(np.searchsorted(self.__files[i_file].row_group_starts, row, side="right")) - 1

    def __row(self, cache, i_file, row):
        row_group = self.__row_group(i_file, row)
        return cache[(i_file, row_group)][row - self.__files[i_file].row_group_starts[row_group]]

    def __event(self, number, get_row):
        """The event `number`; `get_row(file index, row)` returns a row as dict."""
        station_rows = [(get_row(i_file, row), self.__files[i_file]) for i_file, row, _ in self.__station_rows[number]]

        if self.__events[number] is not None:
            i_file, row = self.__events[number]
            event_file = self.__files[i_file]
            event_row, prefix = get_row(i_file, row), ""
        else:  # single file layout: the event columns are part of the station rows
            event_row, event_file = station_rows[0]
            prefix = pu.EVENT_PREFIX
            if event_file.layout != "single":
                logger.warning(
                    f"Event (run, event_id) = ({event_row['run']}, {event_row['event_id']}) has no row "
                    "in the event files: only its stations are restored."
                )

        return _event(
            event_row,
            prefix,
            event_file.event_groups,
            [(row, file.station_groups) for row, file in station_rows],
            event_file.sim_config,
        )

    def get_header(self):
        """
        Overview of all events without reading the traces.

        Parquet: the station rows joined with the event columns as `polars.DataFrame` (one row per event and
        station, see `analysis_reader.AnalysisReader`; needs polars). ``.nur``: the header dict of `eventReaderNur`.
        """
        if self.__nur is not None:
            return self.__nur.get_header()

        return analysis_reader.AnalysisReader(
            self.__selection["path"], stations=self.__selection["stations"], runs=self.__selection["runs"]
        ).read()

    def get_detector(self):
        """The detector stored in ``.nur`` files (see `eventReaderNur.get_detector`). Parquet files: None."""
        if self.__nur is not None:
            return self.__nur.get_detector()

        warnings.warn(
            "eventReader.get_detector: parquet files store no detector, returning None.", FutureWarning, stacklevel=2
        )

        return None

    def close(self):
        """Close all input files. Safe to call several times and before `begin`."""
        if self.__nur is not None:
            self.__nur.close()
        if self.__files:
            logger.info(f"eventReader: {self.__read_time:.2f}s reading row groups.")

        for file in self.__files:
            file.file.close()

        self.__files = []

    def end(self):
        self.close()
