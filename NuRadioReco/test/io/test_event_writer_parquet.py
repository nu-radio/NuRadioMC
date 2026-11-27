import collections
import logging
import os
import pathlib
import stat
import sys
import tempfile
import warnings

import astropy.time
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

import NuRadioReco.framework.channel
import NuRadioReco.framework.electric_field
import NuRadioReco.framework.event
import NuRadioReco.framework.radio_shower
import NuRadioReco.framework.sim_channel
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.station
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import eventParameters as evp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.base import module
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.utilities import units
from NuRadioReco.utilities.parquet_utilities import _file_order, find_files


def _make_event(run, event_id, channel_ids=(0, 1, 2), station_id=11):
    evt = NuRadioReco.framework.event.Event(run, event_id)
    station = NuRadioReco.framework.station.Station(station_id)
    station.set_station_time(astropy.time.Time("2024-01-01T12:00:00.123456789", format="isot"))
    station[stnp.nu_energy] = 1e18 * event_id
    station[stnp.nu_vertex] = np.array([1.0, 2.0, 3.0])
    station[stnp.triggered] = True
    station[stnp.viewing_angles] = {np.int64(0): {1: np.float64(0.5)}}
    station[stnp.flagged_channels] = collections.defaultdict(list, {2: ["a", "b"]})

    for channel_id in channel_ids:
        channel = NuRadioReco.framework.channel.Channel(channel_id)
        if channel_id != 1:  # leave one channel unset
            channel[chp.SNR] = {"peak_amplitude": 3.0}
            channel[chp.noise_rms] = 0.1 * channel_id
        station.add_channel(channel)

    evt.set_station(station)

    return evt, station


def test_write_read(tmp_path):
    writer = eventWriter()
    writer.begin(str(tmp_path / "out.parquet"), row_group_size=2, comment="my comment")  # extension optional

    for run, event_id in [(1, 1), (1, 2), (1, 3), (2, 1), (1, 4)]:  # runs are mixed in one file
        evt, station = _make_event(run, event_id)
        writer.run(evt, station)

    assert writer.end() == 5

    assert _names(tmp_path) == ["out.parquet", "out_event.parquet"]
    assert pq.ParquetFile(tmp_path / "out.parquet").num_row_groups == 3  # 5 events, row_group_size=2

    df = pl.read_parquet(tmp_path / "out.parquet")
    assert df["run"].to_list() == [1, 1, 1, 2, 1]
    assert df["event_id"].to_list() == [1, 2, 3, 1, 4]
    assert df["stationParameters.nu_energy"].to_list()[:3] == [1e18, 2e18, 3e18]
    assert df["stationParameters.nu_vertex"][0].to_list() == [1.0, 2.0, 3.0]
    assert df["stationParameters.zenith"].null_count() == 5
    assert df["channel_ids"][0].to_list() == [0, 1, 2]
    assert df["channelParameters.noise_rms"][0].to_list() == [0.0, None, 0.2]

    time = astropy.time.Time(df["station_time_jd1"][0], df["station_time_jd2"][0], format="jd", scale="utc")
    assert abs((time - astropy.time.Time("2024-01-01T12:00:00.123456789", format="isot")).to_value("s")) < 1e-9

    metadata = pq.read_schema(tmp_path / "out.parquet").metadata
    assert metadata[b"comment"] == b"my comment"
    assert metadata[b"layout"] == b"dual" and metadata[b"kind"] == b"station"
    assert {b"nuradio_version", b"nuradio_commit"} <= set(metadata)

    assert df.schema["stationParameters.viewing_angles"] == pl.Map(pl.Int64, pl.Map(pl.Int64, pl.Float64))
    assert df["channelParameters.SNR"][0].map.get("peak_amplitude").to_list() == [3.0, None, 3.0]
    table = pq.read_table(tmp_path / "out.parquet")  # pyarrow: maps as lists of (key, value)
    assert table["stationParameters.viewing_angles"][0].as_py() == [(0, [(1, 0.5)])]
    assert table["stationParameters.flagged_channels"][0].as_py() == [(2, ["a", "b"])]

    events = pl.read_parquet(tmp_path / "out_event.parquet")
    assert events["event_id"].to_list() == [1, 2, 3, 1, 4]
    assert pq.read_schema(tmp_path / "out_event.parquet").metadata[b"kind"] == b"event"


def test_multiple_stations(tmp_path):
    writer = eventWriter()
    writer.begin(str(tmp_path / "out"))

    for event_id in [1, 2]:
        evt = NuRadioReco.framework.event.Event(5, event_id)

        for station_id in [1, 12]:
            evt.set_station(_make_event(5, event_id, station_id=station_id)[1])

        writer.run(evt)  # all stations

    assert writer.end() == 2
    assert _names(tmp_path) == ["out.parquet", "out_event.parquet"]
    df = pl.read_parquet(tmp_path / "out.parquet")
    assert df.select("event_id", "station_id").rows() == [(1, 1), (1, 12), (2, 1), (2, 12)]
    assert pl.read_parquet(tmp_path / "out_event.parquet")["event_id"].to_list() == [1, 2]


def test_overwrite(tmp_path):
    def write(**kwargs):
        writer = eventWriter()
        writer.begin(str(tmp_path / "out"), **kwargs)
        writer.run(*_make_event(1, 1))
        writer.end()

    write()

    try:
        write()
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing file was overwritten")

    write(overwrite=True)


def test_file_permissions(tmp_path):
    """New files get the permissions of `open` (umask), replaced files keep theirs (not the 0600 of mkstemp)."""

    def write():
        writer = eventWriter()
        writer.begin(str(tmp_path / "out"), overwrite=True)
        writer.run(*_make_event(1, 1))
        writer.end()

    def modes():
        return {path.name: stat.S_IMODE(path.stat().st_mode) for path in tmp_path.iterdir()}

    umask = os.umask(0o027)
    try:
        write()
        assert modes() == {"out.parquet": 0o640, "out_event.parquet": 0o640}

        os.chmod(tmp_path / "out.parquet", 0o604)
        write()
        assert modes() == {"out.parquet": 0o604, "out_event.parquet": 0o640}
    finally:
        os.umask(umask)


def test_split_files(tmp_path):
    writer = eventWriter()
    writer.begin(str(tmp_path / "out"), events_per_file=2)

    for event_id in range(5):
        evt = NuRadioReco.framework.event.Event(1, event_id)

        for station_id in [11, 12]:  # the stations of an event stay in one file
            writer.run(evt, _make_event(1, event_id, station_id=station_id)[1])

    writer.end()

    # numbered as in eventWriter: the second file pair is part 2
    assert _names(tmp_path) == [
        "out.parquet",
        "out_event.parquet",
        "out_part02.parquet",
        "out_part02_event.parquet",
        "out_part03.parquet",
        "out_part03_event.parquet",
    ]
    assert pl.read_parquet(tmp_path / "out_part02.parquet").select("event_id", "station_id").rows() == [
        (2, 11),
        (2, 12),
        (3, 11),
        (3, 12),
    ]
    assert pl.read_parquet(tmp_path / "out_part03_event.parquet")["event_id"].to_list() == [4]

    # max_file_size: estimated from the bytes written and the buffered traces
    writer.begin(str(tmp_path / "size" / "out"), max_file_size=0)

    for event_id in range(3):
        writer.run(*_make_event(1, event_id))

    writer.end()

    assert _names(tmp_path / "size") == [
        "out.parquet",
        "out_event.parquet",
        "out_part02.parquet",
        "out_part02_event.parquet",
        "out_part03.parquet",
        "out_part03_event.parquet",
    ]
    event_reader = eventReader()
    event_reader.begin(str(tmp_path / "size"))
    assert [evt.get_id() for evt in event_reader.run()] == [0, 1, 2]

    # files are read in numerical order (out.parquet first, part10 after part09, part100 after part99)
    writer.begin(str(tmp_path / "many" / "out"), events_per_file=1)

    for event_id in range(11):
        writer.run(*_make_event(1, event_id))

    writer.end()

    files = find_files(str(tmp_path / "many"))["station"]
    assert [pathlib.Path(file).name for file in files] == ["out.parquet"] + [
        f"out_part{part:02d}.parquet" for part in range(2, 12)
    ]
    event_reader = eventReader()
    event_reader.begin(str(tmp_path / "many"))
    assert [evt.get_id() for evt in event_reader.run()] == list(range(11))

    names = ["out_part100.parquet", "out_part99.parquet", "out_part02.parquet", "out.parquet"]
    assert sorted(names, key=_file_order) == names[::-1]


def _add_sim_and_efields(station):
    sim_station = NuRadioReco.framework.sim_station.SimStation(station.get_id())
    sim_station[stnp.nu_energy] = 5e18

    for channel_id, shower_id, ray_id in [(1, 0, 1), (0, 0, 0), (0, 0, 1)]:  # unsorted on purpose
        sim_channel = NuRadioReco.framework.sim_channel.SimChannel(channel_id, shower_id, ray_id)
        sim_channel[chp.noise_rms] = 0.5 * channel_id
        sim_channel.set_trace(np.full(4, channel_id + 0.5), 2 * units.GHz)
        sim_station.add_channel(sim_channel)

    sim_efield = NuRadioReco.framework.electric_field.ElectricField(
        [0, 1], position=[1.0, 2.0, 3.0], shower_id=0, ray_tracing_id=1
    )
    sim_efield[efp.ray_path_type] = "direct"
    sim_efield.set_trace(np.arange(12.0).reshape(3, 4), 2 * units.GHz)
    sim_station.add_electric_field(sim_efield)
    station.set_sim_station(sim_station)
    efield = NuRadioReco.framework.electric_field.ElectricField([2])
    efield[efp.zenith] = 0.3
    station.add_electric_field(efield)


def test_sim_and_efields(tmp_path):
    evt, station = _make_event(3, 9)
    _add_sim_and_efields(station)

    writer = eventWriter()
    writer.begin(str(tmp_path / "out"))
    writer.run(evt, station)
    writer.end()

    df = pl.read_parquet(tmp_path / "out.parquet")
    assert df["sim_station.stationParameters.nu_energy"].to_list() == [5e18]

    # sim channels: list columns sorted by (channel id, shower id, ray tracing id)
    assert df["sim_channel.channel_id"][0].to_list() == [0, 0, 1]
    assert df["sim_channel.ray_tracing_id"][0].to_list() == [0, 1, 1]
    assert df["sim_channel.channelParameters.noise_rms"][0].to_list() == [0.0, 0.0, 0.5]
    assert df["sim_channel.trace"][0].to_list() == [[0.5] * 4, [0.5] * 4, [1.5] * 4]

    assert df["efield.channel_ids"][0].to_list() == [[2]]
    assert df["efield.electricFieldParameters.zenith"][0].to_list() == [0.3]
    assert df["efield.trace"][0].to_list() == [None]  # no trace set
    assert df["sim_efield.position"][0].to_list() == [[1.0, 2.0, 3.0]]
    assert df["sim_efield.electricFieldParameters.ray_path_type"][0].to_list() == ["direct"]
    assert df["sim_efield.trace"][0][0].to_list() == np.arange(12.0).reshape(3, 4).tolist()


def test_mode(tmp_path):
    evt, station = _make_event(3, 9)
    _add_sim_and_efields(station)

    writer = eventWriter()
    writer.begin(str(tmp_path / "out"))
    writer.run(evt, station, mode={"Channels": True, "ElectricFields": True, "SimElectricFields": True})
    writer.end()

    df = pl.read_parquet(tmp_path / "out.parquet")
    assert df["sim_channel.trace"][0].to_list() == [None] * 3
    assert df["sim_channel.sampling_rate"][0].to_list() == [None] * 3
    assert df["sim_efield.trace"][0][0] is not None


def test_single_file_layout(tmp_path):
    evt, station = _make_event(3, 9)
    evt.set_parameter(evp.hash_NuRadioReco, "abc")

    writer = eventWriter()
    writer.begin(str(tmp_path / "out"), event_file=False)
    writer.run(evt)
    writer.end()

    assert _names(tmp_path) == ["out.parquet"]
    df = pl.read_parquet(tmp_path / "out.parquet")
    assert df["event.eventParameters.hash_NuRadioReco"].to_list() == ["abc"]
    assert pq.read_schema(tmp_path / "out.parquet").metadata[b"layout"] == b"single"


def test_no_channel_parameters(tmp_path):
    writer = eventWriter()
    writer.begin(str(tmp_path / "out"), channel_enums=[])
    writer.run(*_make_event(1, 1))
    writer.end()

    columns = pl.read_parquet(tmp_path / "out.parquet").columns
    assert "channel_ids" in columns
    assert not any(column.startswith("channelParameters.") for column in columns)


def test_event_reader_selection(tmp_path):
    # Station 11 in runs 1-3, station 12 only in run 1 (file a), station 11 in run 5 (file b)
    for name, keys in [("a", [(11, 1), (12, 1), (11, 2), (11, 3)]), ("b", [(11, 5)])]:
        writer = eventWriter()
        writer.begin(str(tmp_path / name), row_group_size=1)

        for event_id, (station_id, run) in enumerate(keys):
            writer.run(*_make_event(run, event_id, station_id=station_id))

        writer.end()

    # eventReader: same selection, events in file order
    event_reader = eventReader()
    event_reader.begin(str(tmp_path), stations=[11], runs=[1, 3, 5])
    assert [(evt.get_run_number(), evt.get_id(), evt.get_station_ids()) for evt in event_reader.run()] == [
        (1, 0, [11]),
        (3, 3, [11]),
        (5, 0, [11]),
    ]


def test_event_reader_file_order(tmp_path):
    writer = eventWriter()
    writer.begin(str(tmp_path / "out"), events_per_file=2)
    keys = [(2, 3), (1, 5), (2, 1), (1, 1), (3, 0)]  # unsorted on purpose

    for run, event_id in keys:
        writer.run(*_make_event(run, event_id))

    writer.end()

    event_reader = eventReader()
    event_reader.begin(str(tmp_path))
    assert event_reader.get_n_events() == 5
    assert [(evt.get_run_number(), evt.get_id()) for evt in event_reader.run()] == keys


def _write_runs(writer, events_per_run):
    """Write `events_per_run` (run -> number of events) in ascending order."""
    for run, n_events in events_per_run.items():
        for event_id in range(n_events):
            writer.run(*_make_event(run, event_id))

    return writer.end()


def _names(path):
    return sorted(p.name for p in path.glob("*.parquet"))


def test_type_mismatch(tmp_path):
    writer = eventWriter()
    writer.begin(str(tmp_path / "out"), row_group_size=4)

    for event_id in range(4):
        evt, station = _make_event(1, event_id)
        if event_id == 1:
            station[stnp.nu_energy] = "not a float"  # only this value is written as null
            station[stnp.viewing_angles] = [1, 2]  # not a dict: fails already in the conversion
        # int column: whole-number floats are stored, others are null (pyarrow would truncate them),
        # as are values outside the int64 range (pyarrow raises OverflowError)
        station[stnp.nu_flavor] = [14, 14.0, np.float64(14.7), 1e30][event_id]
        writer.run(evt, station)

    writer.end()

    df = pl.read_parquet(tmp_path / "out.parquet")
    assert df["stationParameters.nu_energy"].to_list()[:3] == [0.0, None, 2e18]
    assert df["stationParameters.viewing_angles"].null_count() == 1
    assert df["stationParameters.nu_flavor"].to_list() == [14, 14, None, None]


@pytest.mark.parametrize(
    "make_bad",
    [
        lambda evt, station: evt.set_id(1.0),
        lambda evt, station: evt.set_run_number(np.float64(3.7)),
        lambda evt, station: station.add_channel(NuRadioReco.framework.channel.Channel(2.5)),
        lambda evt, station: station.add_electric_field(NuRadioReco.framework.electric_field.ElectricField([0.5])),
        lambda evt, station: evt.add_shower(NuRadioReco.framework.radio_shower.RadioShower("1")),
        lambda evt, station: evt.set_id(np.uint64(2**63 + 5)),  # outside int64: pyarrow would write null
        lambda evt, station: evt.set_id(None),
    ],
    ids=["event_id", "run", "channel_id", "efield_channel_ids", "shower_id", "uint64_event_id", "none_event_id"],
)
def test_non_int_id(tmp_path, make_bad):
    """Ids must be ints (no null, no truncation): the writer raises."""
    writer = eventWriter()
    writer.begin(str(tmp_path / "out"))
    evt, station = _make_event(1, 0)
    make_bad(evt, station)

    with pytest.raises(TypeError, match="must be an int"):
        writer.run(evt, station)


def test_failed_event_leaves_no_partial_rows(tmp_path):
    """An event that raises while its rows are built is not written at all and can be written again."""
    writer = eventWriter()
    writer.begin(str(tmp_path / "out"), check_for_duplicates=True)
    writer.run(*_make_event(1, 0))

    evt, _ = _make_event(1, 1)
    good, bad = (NuRadioReco.framework.station.Station(station_id) for station_id in (11, 12))
    bad.add_channel(NuRadioReco.framework.channel.Channel(2.5))
    evt.set_station(good)
    evt.set_station(bad)

    with pytest.raises(TypeError, match="channel id must be an int"):
        writer.run(evt)

    bad.remove_channel(2.5)
    writer.run(evt)  # not rejected as duplicate
    writer.end()

    stations = pl.read_parquet(tmp_path / "out.parquet")
    assert stations.select("event_id", "station_id").rows() == [(0, 11), (1, 11), (1, 12)]
    assert pl.read_parquet(tmp_path / "out_event.parquet")["event_id"].to_list() == [0, 1]
    assert writer.get_n_events() == 2


@pytest.mark.parametrize(
    "parameter, value, message",
    [
        (stnp.viewing_angles, {0.5: {1: 0.5}}, "0.5 is not a whole number"),  # dict key (channel id)
        (stnp.dirty_fft_channels, np.array([3.0, np.nan]), "nan is not an int"),
        (stnp.dirty_fft_channels, [1, 1e30], "outside the range"),
        (stnp.cr_xcorrelations, {"number_of_templates": 2.5}, "2.5 is not a whole number"),  # struct field
    ],
    ids=["dict_key", "nan", "overflow", "struct_field"],
)
def test_invalid_int_element(tmp_path, parameter, value, message):
    """An invalid element of an int list/dict/struct raises (null would also lose the other elements)."""
    from NuRadioReco.utilities.parquet_utilities import ElementTypeError

    writer = eventWriter()
    writer.begin(str(tmp_path / "out"))
    evt, station = _make_event(1, 0)
    station[parameter] = value

    with pytest.raises(ElementTypeError, match=f"{parameter.name}: element .*{message}"):
        writer.run(evt, station)


def test_trigger(tmp_path):
    import NuRadioReco.framework.trigger

    writer = eventWriter()
    writer.begin(str(tmp_path / "out"))

    for event_id in range(2):
        evt, station = _make_event(1, event_id)
        if event_id == 1:
            # Not the first set trigger, but the earliest one is stored
            for name, time in [("late", 20.0), ("early", 10.0)]:
                trigger = NuRadioReco.framework.trigger.Trigger(name, trigger_type=f"{name}_type")
                trigger.set_triggered(True)
                trigger.set_trigger_time(time)
                station.set_trigger(trigger)
        writer.run(evt, station)

    writer.end()

    df = pl.read_parquet(tmp_path / "out.parquet")
    assert df["trigger"].to_list() == [None, "early"]


def test_end_summary(tmp_path):
    messages = []

    class Handler(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    logger = logging.getLogger("NuRadioReco.eventWriter")
    handler = Handler()
    logger.addHandler(handler)
    try:
        writer = eventWriter()
        writer.begin(str(tmp_path / "out"), row_group_size=2)

        for event_id in range(3):
            evt, station = _make_event(1, event_id)
            station.add_electric_field(NuRadioReco.framework.electric_field.ElectricField([0]))
            station.add_electric_field(NuRadioReco.framework.electric_field.ElectricField([1]))
            writer.run(evt, station)

        assert writer.end() == 3
        assert writer.end() == 3  # a second call (e.g. by end_all_modules) does nothing
    finally:
        logger.removeHandler(handler)

    summaries = [message for message in messages if message.startswith("eventWriter: wrote")]
    assert len(summaries) == 1
    assert "wrote 3 events to 2 files" in summaries[0] and "writing to disk" in summaries[0]


class _Detector:
    """Stands in for a detector object (anything that is not a station)."""


def test_old_call_forms(tmp_path):
    """Calls written for the removed .nur writer: .nur name, detector argument, mode dict (missing keys: False)."""
    evt, station = _make_event(1, 1)
    _add_sim_and_efields(station)
    writer = eventWriter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        writer.begin(str(tmp_path / "out.nur"), 1024, True)  # max_file_size, check_for_duplicates
        writer.run(evt, _Detector(), {"Channels": True})  # old positional run(evt, det, mode)
        writer.run(*_make_event(1, 2), det=_Detector())  # warned once per writer

    assert [w.category for w in caught] == [FutureWarning] * 2, [str(w.message) for w in caught]
    assert ".parquet instead of" in str(caught[0].message)
    assert all(w.filename == __file__ for w in caught), [w.filename for w in caught]  # points to the caller

    try:
        writer.run(*_make_event(1, 1))
        raise AssertionError("duplicate event not detected")
    except ValueError as e:
        assert "already written" in str(e)

    assert writer.get_n_events() == 2
    writer.close()
    writer.close()  # safe to call twice
    assert writer.end() == 2
    assert _names(tmp_path) == ["out.parquet", "out_event.parquet"]

    df = pl.read_parquet(tmp_path / "out.parquet")
    assert df["sim_channel.trace"][0].to_list() == [None] * 3  # not in mode
    assert df["sim_efield.trace"][0].to_list() == [None]


def test_check_for_duplicates_stations(tmp_path):
    """Stations of one event written in consecutive calls are no duplicates, the same station twice is."""
    writer = eventWriter()
    writer.begin(str(tmp_path / "out"), check_for_duplicates=True)
    evt, station = _make_event(1, 1)
    second = NuRadioReco.framework.station.Station(12)
    evt.set_station(second)
    writer.run(evt, station)
    writer.run(evt, second)

    try:
        writer.run(evt, second)
        raise AssertionError("duplicate station not detected")
    except ValueError as e:
        assert "Station 12" in str(e)

    assert writer.end() == 1


def _write_three_events(path, **kwargs):
    writer = eventWriter()
    writer.begin(str(path), **kwargs)

    for event_id in range(3):
        evt, station = _make_event(1, event_id)
        _add_sim_and_efields(station)
        writer.run(evt)

    writer.close()


def _read_events(path):
    reader = eventReader()
    reader.begin(str(path))

    return list(reader.run())


def test_write_in_place(tmp_path):
    """Read events, add a parameter and write them to the same file name: files are replaced on close."""
    from test_parquet_roundtrip import _Comparison

    _write_three_events(tmp_path / "out", events_per_file=2)  # 2 parts: out_part02 becomes stale below
    originals = _read_events(tmp_path)
    assert _names(tmp_path) == ["out.parquet", "out_event.parquet", "out_part02.parquet", "out_part02_event.parquet"]

    reader = eventReader()
    reader.begin(str(tmp_path))
    writer = eventWriter()
    writer.begin(str(tmp_path / "out"), overwrite=True)

    for evt in reader.run():
        evt.get_station(11)[stnp.zenith] = 0.5
        writer.run(evt)

    writer.close()

    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.parquet", "out_event.parquet"]  # no tmp, no stale
    restored = _read_events(tmp_path)
    assert len(restored) == len(originals)

    for a, b in zip(originals, restored):
        assert b.get_station(11)[stnp.zenith] == 0.5
        b.get_station(11).remove_parameter(stnp.zenith)
        comparison = _Comparison()
        comparison.event(a, b)
        # the module registry has the run of the writer of the second output in addition
        assert all(d.startswith("modules") for d in comparison.differences), comparison.differences


def test_exception_keeps_existing_files(tmp_path):
    """A run that stops with an exception (end at exit) leaves existing files untouched."""
    _write_three_events(tmp_path / "out")
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    writer = eventWriter()
    writer.begin(str(tmp_path / "out"), overwrite=True)
    writer.run(*_make_event(2, 1))

    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.parquet")} == original  # nothing replaced yet

    # as `end_all_modules` at exit after an uncaught exception
    last_exc = getattr(sys, "last_exc", None)
    module._exiting, sys.last_exc = True, RuntimeError("crash")
    try:
        assert writer.end() == 1
    finally:
        module._exiting, sys.last_exc = False, last_exc

    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.parquet")} == original
    leftovers = list(tmp_path.glob(".out*.tmp"))
    assert len(leftovers) == 2  # station and event file, closed (readable)
    assert pq.read_table(min(leftovers, key=lambda p: "event" in p.name)).num_rows == 1

    messages = []

    class Handler(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    logger = logging.getLogger("NuRadioReco.eventWriter")
    handler = Handler()
    logger.addHandler(handler)
    try:
        eventWriter().begin(str(tmp_path / "out"), overwrite=True)
    finally:
        logger.removeHandler(handler)

    assert any("Temporary files of an earlier run" in message for message in messages), messages


def test_stale_files_without_overwrite(tmp_path):
    """Parts of an earlier output which are not overwritten are only warned about without overwrite."""
    _write_three_events(tmp_path / "out", events_per_file=1)

    for name in ["out.parquet", "out_event.parquet"]:
        (tmp_path / name).unlink()

    _write_three_events(tmp_path / "out", event_file=False)
    assert _names(tmp_path) == [
        "out.parquet",
        "out_part02.parquet",
        "out_part02_event.parquet",
        "out_part03.parquet",
        "out_part03_event.parquet",
    ]


if __name__ == "__main__":
    # Run without pytest (as in the CI): every test gets its own temporary directory
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            with tempfile.TemporaryDirectory() as directory:
                test(pathlib.Path(directory))

            print(f"{name}: passed")
