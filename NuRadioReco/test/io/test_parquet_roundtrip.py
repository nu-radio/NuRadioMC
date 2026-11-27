"""
Round trip .nur -> parquet -> Event: read each input with `eventReaderNur`, write it with `eventWriter`
(dual and single file layout), read it back with `eventReader` and compare event by event, object by object.

Compared: parameters (exact), covariances, parameter types, ids, times (jd1, jd2, scale), traces (bit-identical:
sampling rate, start time, samples), channels incl. trigger channels, sim channels, electric fields, showers,
sim showers, particles, sim emitters, hybrid showers, the triggers of stations (name, class, parameters) and the
module registry (via its JSON round trip).
Not compared: the order of channels (the writer sorts channels and sim channels by id).

Accepted differences (`from_arrow_value`): numpy scalars come back as python scalars, ``np.ndarray`` parameters
as float arrays. Everything else is a difference, unless listed in `KNOWN_DIFFERENCES`.

`test_roundtrip_synthetic` covers the objects the input files lack (trigger channels, station electric fields,
covariances, showers, emitters, hybrid showers, phased / per-channel triggers, several stations, an event without
stations).
Usage: ``python test_parquet_roundtrip.py [more.nur or .parquet ...]``
"""

import collections
import pathlib
import pickle
import re
import sys
import tempfile
from timeit import default_timer as timer

import astropy.time
import numpy as np

from NuRadioReco.framework import parameters
from NuRadioReco.framework.base_trace import BaseTrace
from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.electric_field import ElectricField
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.hybrid_shower import HybridShower
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import emitterParameters as ep
from NuRadioReco.framework.parameters import eventParameters as evp
from NuRadioReco.framework.parameters import generatorAttributes as genattrs
from NuRadioReco.framework.parameters import particleParameters as pap
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.particle import Particle
from NuRadioReco.framework.radio_shower import RadioShower
from NuRadioReco.framework.sim_channel import SimChannel
from NuRadioReco.framework.sim_emitter import SimEmitter
from NuRadioReco.framework.sim_station import SimStation
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.trigger import HighLowTrigger, SimplePhasedTrigger
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.io.eventReaderNur import eventReaderNur
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.utilities import units
from NuRadioReco.utilities.io_utilities import _from_json, _to_json

_ROOT = pathlib.Path(__file__).parents[3]
# Further .nur files can be passed as command line arguments
INPUTS = [
    _ROOT / "NuRadioMC/test/SingleEvents/1e18_output_reference.nur",
    # Measured RNO-G data. Not tracked in git (74 MB): skipped if missing, e.g. in the CI
    _ROOT / "NuRadioReco/examples/RNOG/st12_run1927.nur",
]

# Known differences: (input file name or "*", regular expression searched in the difference, cause).
# Only these are tolerated. Except for the start time, the stored values do not match their declared type:
# they are written (and read back) as the declared type.
KNOWN_DIFFERENCES = [
    (
        "*",
        r"/trace_start_time: type int != float \(0 / 0.0\)",
        "BaseTrace's default start time is the int 0, parquet stores float64",
    ),
    (
        "1e18_output_reference.nur",
        r"/particleParameters.weight: missing \(original array\(\[1\.\]\)\)",
        "old reference file; fixed in simulation.calculate_particle_weight since 6b3e3b90a: a 1-element array "
        "although typed float (written as null)",
    ),
    (
        "1e18_output_reference.nur",
        r"/generatorAttributes.flavors: type ndarray != list",
        "old reference file; fixed in simulation.build_NuRadioEvents_from_hdf5 since 4429c6fdb: "
        "an integer array although typed list[int]",
    ),
    (
        "1e18_output_reference.nur",
        r"/electricFieldParameters.reflection_coefficient_(theta|phi): type float != complex",
        "old reference file; fixed in the SignalProp ray tracers since 30240bdfe: a float although typed complex",
    ),
]


def _scalar(value):
    """numpy scalars -> python scalars (the documented difference), other values unchanged."""
    return value.item() if isinstance(value, np.generic) else value


def _difference(a, b, float_array=False):
    """
    None if `a` and `b` are equal including their types (lists, tuples, dicts, arrays; NaN equal to NaN),
    else a description of the first difference. `float_array`: a declared ``np.ndarray`` parameter (restored as
    float array).
    """
    a, b = _scalar(a), _scalar(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return f"type {type(a).__name__} != {type(b).__name__}"

        if a.shape != b.shape:
            return f"shape {a.shape} != {b.shape}"

        if float_array and a.dtype.kind in "biuf":
            a = a.astype(float)
        if a.dtype != b.dtype:
            return f"dtype {a.dtype} != {b.dtype}"

        return None if a.tobytes() == b.tobytes() or np.array_equal(a, b, equal_nan=True) else "values differ"

    if type(a) is not type(b):
        return f"type {type(a).__name__} != {type(b).__name__} ({_short(a, 40)} / {_short(b, 40)})"

    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"keys {sorted(map(str, a.keys() ^ b.keys()))} not in both"

        children = ((f"[{key!r}]", _difference(a[key], b[key])) for key in a)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"length {len(a)} != {len(b)}"

        children = ((f"[{i}]", _difference(x, y)) for i, (x, y) in enumerate(zip(a, b)))
    elif isinstance(a, float) and np.isnan(a):
        return None if np.isnan(b) else f"{a!r} != {b!r}"
    else:
        return None if a == b else f"{_short(a)} != {_short(b)}"

    return next((f"{key}: {difference}" for key, difference in children if difference is not None), None)


class _Comparison:
    """Collects the differences (path: message) of one event."""

    def __init__(self):
        self.differences = []

    def check(self, path, a, b, float_array=False):
        difference = _difference(a, b, float_array)
        if difference is not None:
            self.differences.append(f"{path}: {difference}")

    def storage(self, path, a, b):
        """Parameters (exact), covariances and parameter types of two parameter storages."""
        for key in sorted(set(a._parameters) | set(b._parameters), key=str):
            if key not in b._parameters:
                self.differences.append(f"{path}/{key}: missing (original {_short(a._parameters[key])})")
            elif key not in a._parameters:
                self.differences.append(f"{path}/{key}: not in the original ({_short(b._parameters[key])})")
            else:
                self.check(f"{path}/{key}", a._parameters[key], b._parameters[key], key.dtype is np.ndarray)

        self.check(
            f"{path}/covariances",
            {(str(k[0]), str(k[1])): v for k, v in a._parameter_covariances.items()},
            {(str(k[0]), str(k[1])): v for k, v in b._parameter_covariances.items()},
        )
        self.check(
            f"{path}/parameter_types", [t.__name__ for t in a._parameter_types], [t.__name__ for t in b._parameter_types]
        )

    def trace(self, path, a, b):
        """Sampling rate, start time and samples, bit-identical (as `BaseTrace.serialize` stores them)."""
        a, b = BaseTrace.serialize(a), BaseTrace.serialize(b)
        if a is None or b is None:
            self.check(f"{path}/trace", a, b)
            return

        a, b = pickle.loads(a), pickle.loads(b)

        for key in ["sampling_rate", "trace_start_time"]:
            self.check(f"{path}/{key}", a[key], b[key])

        self.check(f"{path}/trace", a["time_trace"], b["time_trace"])

    def time(self, path, a, b):
        if a is None or b is None:
            self.check(path, a, b)
        else:
            self.check(path, (a.jd1, a.jd2, a.scale), (b.jd1, b.jd2, b.scale))

    def objects(self, path, a, b, key, compare):
        """Compare two lists of objects matched by `key` (a function of the object)."""
        a, b = {key(x): x for x in a}, {key(x): x for x in b}
        self.check(f"{path} ids", sorted(a, key=str), sorted(b, key=str))

        for object_key in a.keys() & b.keys():
            compare(f"{path} {object_key}", a[object_key], b[object_key])

    def channel(self, path, a, b):
        self.storage(path, a, b)
        self.check(f"{path}/group_id", a.get_group_id(), b.get_group_id())
        self.trace(path, a, b)
        self.check(f"{path}/has trigger channel", a.has_extra_trigger_channel(), b.has_extra_trigger_channel())
        if a.has_extra_trigger_channel() and b.has_extra_trigger_channel():
            self.check(f"{path}/trigger channel id", a.get_trigger_channel().get_id(), b.get_trigger_channel().get_id())
            self.channel(f"{path}/trigger channel", a.get_trigger_channel(), b.get_trigger_channel())

    def efields(self, path, a, b):
        self.check(f"{path} count", len(a), len(b))

        for i, (x, y) in enumerate(zip(a, b)):  # the writer keeps their order
            p = f"{path} {i}"
            self.storage(p, x, y)
            self.check(f"{p}/channel_ids", [int(c) for c in x.get_channel_ids()], y.get_channel_ids())
            self.check(f"{p}/position", np.asarray(x.get_position(), dtype=float), y.get_position())
            self.check(f"{p}/shower_id", x.get_shower_id(), y.get_shower_id())
            self.check(f"{p}/ray_tracing_id", x.get_ray_tracing_solution_id(), y.get_ray_tracing_solution_id())
            self.trace(p, x, y)

    def base_station(self, path, a, b):
        self.storage(path, a, b)
        self.check(f"{path}/id", a.get_id(), b.get_id())
        self.time(f"{path}/station_time", a._station_time, b._station_time)
        self.check(f"{path}/particle_type", a._particle_type, b._particle_type)
        self.check(f"{path}/triggered", a._triggered, b._triggered)
        self.efields(f"{path}/efield", a.get_electric_fields(), b.get_electric_fields())

    def trigger(self, path, a, b):
        self.check(f"{path}/class", type(a).__name__, type(b).__name__)
        self.check(f"{path}/type", a.get_type(), b.get_type())
        self.storage(path, a, b)

    def station(self, path, a, b):
        self.base_station(path, a, b)
        self.objects(
            f"{path}/trigger", a.get_triggers().values(), b.get_triggers().values(), lambda t: t.get_name(), self.trigger
        )
        self.check(f"{path}/reference_reconstruction", a.get_reference_reconstruction(), b.get_reference_reconstruction())
        self.objects(f"{path}/channel", a.iter_channels(), b.iter_channels(), lambda c: c.get_id(), self.channel)

        self.check(f"{path}/has sim station", a.has_sim_station(), b.has_sim_station())
        if a.has_sim_station() and b.has_sim_station():
            sa, sb = a.get_sim_station(), b.get_sim_station()
            self.base_station(f"{path}/sim", sa, sb)
            magnetic_field = sa.get_magnetic_field_vector()
            self.check(
                f"{path}/sim/magnetic_field_vector",
                None if magnetic_field is None else np.asarray(magnetic_field, dtype=float),
                sb.get_magnetic_field_vector(),
            )
            self.check(f"{path}/sim/simulation_weight", sa.get_simulation_weight(), sb.get_simulation_weight())
            self.objects(
                f"{path}/sim_channel",
                sa.iter_channels(),
                sb.iter_channels(),
                lambda c: c.get_unique_identifier(),
                self.channel,
            )

    def showers(self, path, a, b, compare=None):
        a, b = list(a), list(b)
        self.check(f"{path} count", len(a), len(b))

        for x, y in zip(a, b):
            p = f"{path} {x.get_id()}"
            self.storage(p, x, y)
            self.check(f"{p}/id", x.get_id(), y.get_id())
            if compare is not None:
                compare(p, x, y)

    def event(self, a, b):
        self.check("run", a.get_run_number(), b.get_run_number())
        self.check("event_id", a.get_id(), b.get_id())
        self.time("event_time", a._Event__event_time, b._Event__event_time)
        self.storage("event", a, b)
        self.objects("station", a.get_stations(), b.get_stations(), lambda s: s.get_id(), self.station)

        def station_ids(path, x, y):
            self.check(f"{path}/station_ids", _int_list(x.get_station_ids()), y.get_station_ids())

        self.showers("shower", a.get_showers(), b.get_showers(), station_ids)
        self.showers("sim_shower", a.get_sim_showers(), b.get_sim_showers(), station_ids)
        self.showers("particle", a.get_particles(), b.get_particles())
        self.showers("sim_emitter", a.get_sim_emitters(), b.get_sim_emitters(), station_ids)
        self.showers(
            "hybrid_shower",
            a.get_hybrid_information().get_hybrid_showers(),
            b.get_hybrid_information().get_hybrid_showers(),
            lambda path, x, y: self.check(f"{path}/name", x.get_name(), y.get_name()),
        )

        # Module registry as `Event.serialize` stores it (without the module instances), after the JSON round trip
        # (values JSON can not express are stored as their repr, e.g. the TypeError `register_run` stores for
        # arguments it can not pickle)
        modules_event = [[m[0], None, m[2]] for m in a._Event__modules_event]
        modules_station = {
            key: [[m[0], m[1], None, m[3]] for m in value] for key, value in a._Event__modules_station.items() if value
        }
        self.check("modules event", _from_json(_to_json(modules_event)), b._Event__modules_event)
        self.check("modules station", _from_json(_to_json(modules_station)), dict(b._Event__modules_station))


def _int_list(values):
    return None if values is None else [int(value) for value in values]


def _short(value, length=120):
    text = repr(value)

    return text if len(text) <= length else text[:length] + "..."


def _read_nur(path):
    """The events of a .nur file (or of a parquet file, e.g. written by an example: parquet -> parquet)."""
    reader = eventReaderNur() if path.suffix == ".nur" else eventReader()
    reader.begin(str(path))
    events = list(reader.run())
    reader.end()

    return events


def roundtrip(originals, output_dir, event_file):
    """
    Write the events to parquet (`event_file`: dual / single layout), read them back and compare.

    Returns the differences (list of str, prefixed with run and event id) and the timings.
    """
    times = {}
    writer = eventWriter()
    writer.begin(
        str(output_dir / "events"),
        event_file=event_file,
        station_enums=[parameters.stationParameters, parameters.stationParametersRNOG],
        channel_enums=[parameters.channelParameters, parameters.channelParametersRNOG],
    )
    t0 = timer()

    for evt in originals:
        writer.run(evt)

    writer.end()
    times["write parquet"] = timer() - t0

    reader = eventReader()
    reader.begin(str(output_dir))
    t0 = timer()
    restored = list(reader.run())
    times["read parquet"] = timer() - t0
    reader.end()

    differences = []  # the reader returns the events in the order they were written
    if len(originals) != len(restored):
        differences.append(f"number of events: {len(originals)} != {len(restored)}")

    for a, b in zip(originals, restored):
        comparison = _Comparison()
        comparison.event(a, b)
        differences += [f"run {a.get_run_number()} event {a.get_id()}: {d}" for d in comparison.differences]

    return differences, times


def _known_cause(file_name, difference):
    """The cause of a known difference (see `KNOWN_DIFFERENCES`), else None."""
    return next(
        (cause for name, pattern, cause in KNOWN_DIFFERENCES if name in ("*", file_name) and re.search(pattern, difference)),
        None,
    )


def _check(name, differences):
    """Print the known differences (one line per cause) and fail on the others."""
    causes = collections.Counter()
    unexpected = []

    for difference in differences:
        cause = _known_cause(name, difference)
        if cause is None:
            unexpected.append(difference)
        else:
            causes[cause] += 1

    for cause, n in causes.items():
        print(f"  known ({n}x): {cause}")

    assert not unexpected, f"{len(unexpected)} unexpected differences:\n" + "\n".join(unexpected[:50])


def test_roundtrip(tmp_path):
    for nur_file in INPUTS:
        if not nur_file.exists():
            print(f"{nur_file.name}: skipped (not found; a tracked RNO-G data .nur file is needed for the CI)")
            continue

        for event_file in [True, False]:
            layout = "dual" if event_file else "single"
            output_dir = tmp_path / f"{nur_file.stem}_{layout}"

            t0 = timer()
            events = _read_nur(nur_file)
            read_time = timer() - t0
            differences, times = roundtrip(events, output_dir, event_file)
            times = {"read .nur": read_time, **times}

            print(
                f"{nur_file.name} ({layout}): {len(differences)} differences; "
                + ", ".join(f"{key} {value:.2f}s" for key, value in times.items())
            )
            _check(nur_file.name, differences)


class _Module:
    """Registers module keyword arguments which JSON can not express directly."""

    @register_run()
    def run(self, evt, station, det, factor=(1, 2), array=np.arange(3), mapping={1: "a"}, generator=None):
        pass


def _trace_channel(channel_id, rng, group_id=None):
    channel = Channel(channel_id, channel_group_id=group_id)
    channel.set_trace(rng.normal(size=16), 2.4 * units.GHz, trace_start_time=rng.normal() * units.ns)
    channel[chp.noise_rms] = rng.normal()

    return channel


def _synthetic_events():
    """
    Events with the objects the input files do not have: trigger channels, station electric fields, covariances,
    showers, particles, emitters, hybrid showers, event time, several stations, an event without stations.
    Returned as read from a .nur file (`Event.serialize` / `deserialize`).
    """
    rng = np.random.default_rng(3)
    evt = Event(3, 42)
    evt.set_event_time(astropy.time.Time("2024-05-01T10:00:00.123456789", format="isot", scale="utc"))
    evt.set_parameter(evp.hash_NuRadioReco, "abc")
    evt.set_parameter(genattrs.n_events, 10)

    for station_id in [11, 12]:
        station = Station(station_id)
        station.set_station_time(astropy.time.Time(60432.5 + station_id, format="mjd", scale="tai"))
        station.set_is_neutrino()
        station.set_triggered(station_id == 11)

        if station_id == 11:
            phased = SimplePhasedTrigger(
                "phased",
                2.5,
                channels=[0, 1],
                primary_angles=np.array([0.1, 0.2]),
                trigger_delays={0: {0: 1.0, 1: 2.0}},
                window_size=24,
                step_size=8,
                maximum_amps=np.array([3.0, 4.0]),
            )
            phased.set_triggered(True)
            phased.set_trigger_time(12.0)
            phased.set_trigger_times({0: np.array([12.0, 20.0]), 1: np.array([15.0])})
            phased.set_triggered_channels([0, 1])
            station.set_trigger(phased)

            # per-channel thresholds and pre trigger times (dict alternative of the `OneOf` types)
            high_low = HighLowTrigger(
                "high_low", {0: 0.1, 1: 0.2}, -0.1, 5.0, 10.0, channels=[0, 1], pre_trigger_times={0: 50.0, 1: 60.0}
            )
            station.set_trigger(high_low)

        station[stnp.zenith] = 1.0
        station[stnp.azimuth] = 2.0
        station.set_parameter_error(stnp.zenith, 0.1)
        station.set_parameter_covariance(stnp.zenith, stnp.azimuth, 0.01)
        station[stnp.viewing_angles] = {0: {1: 0.5}}
        station[stnp.cr_energy] = None  # parameters set to None are listed in <kind>.none_parameters

        for channel_id in [3, 0, 1]:  # unsorted: the writer sorts by id
            channel = _trace_channel(channel_id, rng, group_id=channel_id // 2)
            if channel_id == 3:
                channel[chp.SNR] = None
            if channel_id == 0:
                channel[chp.SNR] = 4.5  # scalar alternative of the `OneOf` type (LOFAR)
            if channel_id == 1:
                channel[chp.SNR] = {"peak_amplitude": 3.0, "integrated_power": 7.0}
                channel.set_trigger_channel(_trace_channel(channel_id, rng))
            station.add_channel(channel)

        station.add_channel(Channel(5))  # no trace

        efield = ElectricField([0, 1], position=np.array([1.0, 2.0, -3.0]), shower_id=2, ray_tracing_id=1)
        efield.set_trace(rng.normal(size=(3, 16)), 2.4 * units.GHz, trace_start_time=5.0)
        efield[efp.zenith] = 0.3
        efield.set_parameter_error(efp.zenith, 0.03)
        efield[efp.signal_energy_fluence] = np.array([0.0, 10.0, 20.0])  # per polarization
        efield.set_parameter_error(efp.signal_energy_fluence, np.array([0.0, 1.0, 2.0]))  # array valued error
        station.add_electric_field(efield)
        efield_no_trace = ElectricField([3])
        efield_no_trace[efp.signal_energy_fluence] = {"80-300": 5.0}  # per passband
        station.add_electric_field(efield_no_trace)  # no trace

        if station_id == 11:
            sim_station = SimStation(station_id)
            sim_station.set_station_time(station.get_station_time())
            sim_station.set_magnetic_field_vector(np.array([0.0, 0.2, -0.5]))
            sim_station.set_simulation_weight(0.7)
            sim_station[stnp.nu_energy] = 1e18

            for channel_id, shower_id, ray_id in [(1, 0, 1), (0, 0, 0), (0, 0, 1)]:
                sim_channel = SimChannel(channel_id, shower_id, ray_id)
                sim_channel.set_trace(rng.normal(size=16), 2.4 * units.GHz)
                sim_station.add_channel(sim_channel)

            sim_efield = ElectricField([0], shower_id=0, ray_tracing_id=1)
            sim_efield.set_trace(rng.normal(size=(3, 16)), 2.4 * units.GHz)
            sim_efield[efp.ray_path_type] = "direct"
            sim_station.add_electric_field(sim_efield)
            station.set_sim_station(sim_station)

        evt.set_station(station)
        _Module().run(evt, station, None, generator=(x for x in []))  # not picklable: stored as TypeError

    for shower_id, add in [(0, evt.add_shower), (1, evt.add_sim_shower)]:
        shower = RadioShower(shower_id, station_ids=[11, 12])
        shower[shp.energy] = 1e18 + shower_id
        shower.set_parameter_error(shp.energy, 1e17)
        add(shower)

    particle = Particle(5)
    particle[pap.flavor] = 14
    particle[pap.vertex] = np.array([1.0, 2.0, 3.0])
    particle[pap.parent_id] = None  # restored from particle.none_parameters
    evt.add_particle(particle)

    emitter = SimEmitter(2, station_ids=[11])
    emitter[ep.amplitude] = 1.5
    evt.add_sim_emitter(emitter)

    hybrid = HybridShower("scintillator")
    hybrid[shp.zenith] = 0.2
    evt.get_hybrid_information().add_hybrid_shower(hybrid)

    events = [evt, Event(3, 43)]  # the second one without stations: not written with event_file=False
    restored = []

    for evt in events:
        restored.append(Event(0, 0))
        restored[-1].deserialize(
            evt.serialize({"Channels": True, "ElectricFields": True, "SimChannels": True, "SimElectricFields": True})
        )

    return restored


def test_roundtrip_synthetic(tmp_path):
    for event_file in [True, False]:
        events = _synthetic_events()
        if not event_file:
            events = [evt for evt in events if evt.get_station_ids()]

        differences, _ = roundtrip(events, tmp_path / str(event_file), event_file)
        print(f"synthetic events ({'dual' if event_file else 'single'}): {len(differences)} differences")
        _check("synthetic", differences)


if __name__ == "__main__":
    INPUTS += [pathlib.Path(path) for path in sys.argv[1:]]

    # Run without pytest (as in the CI): every test gets its own temporary directory
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            with tempfile.TemporaryDirectory() as directory:
                test(pathlib.Path(directory))

            print(f"{name}: passed")
