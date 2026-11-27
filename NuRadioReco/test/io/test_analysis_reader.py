"""
Tests of `NuRadioReco.utilities.analysis_reader` (`AnalysisReader`, command line tool ``nuradio-analyse``).
Usage: ``python test_analysis_reader.py`` (or with pytest).
"""

import pathlib
import tempfile

import astropy.time
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import NuRadioReco.framework.event
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import eventParameters as evp
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.modules.io.nur_to_parquet import convert_nur
from NuRadioReco.utilities import analysis_reader, units
from NuRadioReco.utilities.analysis_reader import AnalysisReader
from test_event_writer_parquet import _add_sim_and_efields, _make_event, _write_runs
from test_nur_to_parquet import NUR_FILE, _read_nur


def test_columns_of_sim_station(tmp_path):
    evt, station = _make_event(3, 9)
    _add_sim_and_efields(station)

    writer = eventWriter()
    writer.begin(str(tmp_path / "out"))
    writer.run(evt, station)
    writer.end()

    assert "sim_station.stationParameters.nu_energy" in AnalysisReader(str(tmp_path)).columns(
        parameter_classes="stationParameters"
    )


def test_split_files_read_in_order(tmp_path):
    writer = eventWriter()
    writer.begin(str(tmp_path / "out"), events_per_file=1)

    for event_id in range(11):
        writer.run(*_make_event(1, event_id))

    writer.end()

    # files are read in numerical order (out.parquet first, part10 after part09)
    assert AnalysisReader(str(tmp_path), sort=False).read()["event_id"].to_list() == list(range(11))


def test_event_reader_header(tmp_path):
    convert_nur(str(NUR_FILE), output_dir=str(tmp_path))

    reader = eventReader()
    reader.begin(str(tmp_path / "1e18_output_reference.parquet"))
    assert reader.get_header().height == len(_read_nur())  # polars frame of the station rows
    reader.end()


def test_reader(tmp_path):
    writer = eventWriter()
    writer.begin(str(tmp_path / "sub" / "out"))

    for run in [1, 2]:
        evt = NuRadioReco.framework.event.Event(run, 1)
        evt.set_parameter(evp.hash_NuRadioReco, f"run{run}")

        for station_id in [11, 12]:
            station = _make_event(run, 1, station_id=station_id)[1]
            station.get_channel(0).set_trace(np.arange(4.0), 2 * units.GHz)
            evt.set_station(station)

        writer.run(evt)

    writer.end()

    reader = AnalysisReader(str(tmp_path))  # root directory, searched recursively
    assert len(reader.files["station"]) == 1 and len(reader.files["event"]) == 1
    df = reader.read()
    assert df.height == 4
    assert df["event.eventParameters.hash_NuRadioReco"].to_list() == ["run1", "run2"] * 2  # sorted by station first
    assert "channel.trace" not in df.columns  # traces only on request

    # The pattern matches only the station file: its event file is found next to it
    df = AnalysisReader(str(tmp_path / "sub" / "out.parquet"), stations=11, runs=[2], traces=True).read()
    assert df.height == 1 and df["event.eventParameters.hash_NuRadioReco"].to_list() == ["run2"]
    assert df["channel.trace"][0].to_list() == [[0.0, 1.0, 2.0, 3.0], None, None]

    tables = AnalysisReader(str(tmp_path), join_events=False).tables()
    assert sorted(tables) == ["event", "station"]
    assert tables["event"].collect().height == 2
    assert "event.eventParameters.hash_NuRadioReco" not in AnalysisReader(str(tmp_path), join_events=False).read().columns

    assert "channel.trace" in AnalysisReader(str(tmp_path), traces=True).columns()
    assert "channel.trace" not in AnalysisReader(str(tmp_path)).columns()
    assert "eventParameters.hash_NuRadioReco" in AnalysisReader(str(tmp_path)).columns("event")


def test_reader_sorted(tmp_path):
    writer = eventWriter()
    writer.begin(str(tmp_path / "out"))

    for station_id, run, event_id in [(12, 2, 3), (11, 2, 2), (12, 1, 5), (11, 2, 1), (11, 1, 9)]:  # unsorted on purpose
        writer.run(*_make_event(run, event_id, station_id=station_id))

    writer.end()

    keys = ["station_id", "run", "event_id"]
    tables = AnalysisReader(str(tmp_path)).tables()
    assert tables["station"].select(keys).collect().rows() == sorted(tables["station"].select(keys).collect().rows())
    assert tables["station"].collect().height == 5
    assert tables["event"].select("run", "event_id").collect().rows() == [(1, 5), (1, 9), (2, 1), (2, 2), (2, 3)]

    joined = AnalysisReader(str(tmp_path)).read()
    assert joined.select(keys).rows() == sorted(joined.select(keys).rows())

    unsorted = AnalysisReader(str(tmp_path), sort=False).read()  # file order: no sorting
    assert unsorted.height == 5


def test_reader_list_of_paths(tmp_path):
    for sub in ["a", "b"]:
        writer = eventWriter()
        writer.begin(str(tmp_path / sub / f"out_{sub}"))
        _write_runs(writer, {1 if sub == "a" else 2: 2})

    assert AnalysisReader([str(tmp_path / "a")]).read().height == 2
    assert AnalysisReader([str(tmp_path / "a"), str(tmp_path / "b")]).read().height == 4
    assert AnalysisReader([str(tmp_path / "b" / "out_b.parquet")]).read().height == 2
    assert AnalysisReader(tmp_path / "b").read().height == 2  # a single path-like

    # Entries can be patterns too; an entry without match is an error
    assert AnalysisReader([str(tmp_path / "*" / "out_a*.parquet"), str(tmp_path / "b")]).read().height == 4
    try:
        AnalysisReader([str(tmp_path / "a"), str(tmp_path / "c")])
        raise AssertionError("no FileNotFoundError")
    except FileNotFoundError:
        pass


def test_reader_duplicate_events(tmp_path):
    # Two processings of the same runs in one directory
    for name in ["a", "b"]:
        writer = eventWriter()
        writer.begin(str(tmp_path / name))
        _write_runs(writer, {1: 2, 2: 1})

    reader = AnalysisReader(str(tmp_path))
    assert reader.columns()  # does not read the data: no check

    for method in [reader.scan, reader.tables, reader.read]:
        try:
            method()
            raise AssertionError("no ValueError for duplicated events")
        except ValueError as e:
            assert "3 events" in str(e) and "a.parquet" in str(e) and "b.parquet" in str(e)

    assert AnalysisReader(str(tmp_path / "a*.parquet")).read().height == 3
    assert AnalysisReader(str(tmp_path), runs=2, check_unique=False).read().height == 2


def test_reader_duplicate_events_order(tmp_path):
    # Each event twice in the same file, in random event order: the sorted table keeps the file order of
    # the copies (an unstable sort mixes them up for large tables)
    events = np.random.default_rng(1).permutation(100_000).repeat(2)
    table = pl.DataFrame({"station_id": 11, "run": 1, "event_id": events, "copy": np.tile([0, 1], len(events) // 2)})
    pq.write_table(
        table.to_arrow().replace_schema_metadata({"kind": "station"}), tmp_path / "out.parquet", row_group_size=5000
    )

    table = AnalysisReader(str(tmp_path), check_unique=False).scan().collect()

    assert table["event_id"].is_sorted()
    assert table["copy"].to_list() == [0, 1] * 100_000


def test_calpulsers(tmp_path):
    writer = eventWriter()
    writer.begin(str(tmp_path / "out"))

    # 2 us after the full second (calpulser), 0.5 s and 10 us after it
    for event_id, fraction in enumerate(["000002", "500000", "000010"]):
        evt, station = _make_event(1, event_id)
        station.set_station_time(astropy.time.Time(f"2024-01-01T12:00:01.{fraction}", format="isot"))
        writer.run(evt, station)

    writer.end()

    events = AnalysisReader(str(tmp_path)).scan()
    flagged = AnalysisReader.calpulsers(events).collect()
    assert flagged["is_calpulser"].to_list() == [True, False, False]
    assert AnalysisReader.calpulsers(events, select=True).collect()["event_id"].to_list() == [0]
    assert AnalysisReader.calpulsers(events, select=False).collect()["event_id"].to_list() == [1, 2]

    wide = AnalysisReader.calpulsers(events.collect(), time_range=(0, 20 * units.microsecond))  # DataFrame works too
    assert wide["is_calpulser"].to_list() == [True, False, True]

    for time_range in [(1,), (5 * units.microsecond, 1 * units.microsecond), (0, 2 * units.second)]:
        try:
            AnalysisReader.calpulsers(events, time_range=time_range)
            raise AssertionError(f"no ValueError for time_range={time_range}")
        except ValueError:
            pass


def test_reader_selection(tmp_path):
    # Station 11 in runs 1-3, station 12 only in run 1 (file a), station 11 in run 5 (file b)
    for name, keys in [("a", [(11, 1), (12, 1), (11, 2), (11, 3)]), ("b", [(11, 5)])]:
        writer = eventWriter()
        writer.begin(str(tmp_path / name), row_group_size=1)

        for event_id, (station_id, run) in enumerate(keys):
            writer.run(*_make_event(run, event_id, station_id=station_id))

        writer.end()

    # Files whose row group statistics exclude the selection are not read
    assert [pathlib.Path(file).name for file in AnalysisReader(str(tmp_path), runs=5).files["station"]] == ["b.parquet"]
    reader = AnalysisReader(str(tmp_path), stations=12)
    assert [pathlib.Path(file).name for file in reader.files["station"]] == ["a.parquet"]
    assert [pathlib.Path(file).name for file in reader.files["event"]] == ["a_event.parquet"]

    # Rows are filtered within the files, the event table keeps only the events of the selected stations
    tables = {kind: table.collect() for kind, table in reader.tables().items()}
    assert tables["station"].select("station_id", "run").rows() == [(12, 1)]
    assert tables["event"].select("run", "event_id").rows() == [(1, 1)]

    reader = AnalysisReader(str(tmp_path), runs=[2, 5])
    assert reader.read().select("run", "event_id").rows() == [(2, 2), (5, 0)]
    assert reader.read()["event.event_time_jd1"].len() == 2


def test_single_file_layout(tmp_path):
    """The reader returns the same columns for both layouts."""
    evt, station = _make_event(3, 9)
    evt.set_parameter(evp.hash_NuRadioReco, "abc")

    writer = eventWriter()
    writer.begin(str(tmp_path / "out"), event_file=False)
    writer.run(evt)
    writer.end()

    single = AnalysisReader(str(tmp_path)).read()
    writer.begin(str(tmp_path / "dual" / "out"))
    writer.run(evt)
    writer.end()
    dual = AnalysisReader(str(tmp_path / "dual")).read()

    assert single.columns == dual.columns
    assert dual["event.eventParameters.hash_NuRadioReco"].to_list() == ["abc"]

    # The event attributes have no prefix in the event file, `event.` (once) in the station rows
    assert {"modules", "covariances"} <= set(pq.read_schema(tmp_path / "dual" / "out_event.parquet").names)
    assert {"event.modules", "event.covariances"} <= set(single.columns)
    assert not [name for name in single.columns if name.startswith("event.event.")]
    assert single["event.modules"].str.contains("eventWriter").all()


def test_histogram_one_of(tmp_path):
    # SNR is a `OneOf` parameter: a single value (e.g. LOFAR) or a map per definition
    evt, station = _make_event(1, 1)  # SNR {"peak_amplitude": 3.0} on channels 0 and 2
    station.get_channel(1)[chp.SNR] = 4.5

    writer = eventWriter()
    writer.begin(str(tmp_path / "out"))
    writer.run(evt, station)
    writer.end()

    reader = AnalysisReader(str(tmp_path))
    _, frame = analysis_reader._histogram_values(reader, "station", "SNR", key="peak_amplitude")
    assert sorted(zip(frame["channel_id"], frame["value"])) == [(0, 3.0), (2, 3.0)]

    _, frame = analysis_reader._histogram_values(reader, "station", "SNR")
    assert list(zip(frame["channel_id"], frame["value"])) == [(1, 4.5)]


def test_cli(tmp_path):
    # All inputs into one output, single file layout, no traces
    analysis_reader.main(
        [
            "convert",
            str(NUR_FILE),
            "--output-dir",
            str(tmp_path),
            "--output",
            "combined",
            "--no-traces",
            "--single-file",
            "--compression",
            "zstd",
        ]
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["combined.parquet"]

    df = pl.read_parquet(tmp_path / "combined.parquet")
    assert df.height == len(_read_nur())
    assert df["channel.trace"].list.eval(pl.element().is_null().all()).list.all().all()  # traces not stored


if __name__ == "__main__":
    # Run without pytest (as in the CI): every test gets its own temporary directory
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            with tempfile.TemporaryDirectory() as directory:
                test(pathlib.Path(directory))

            print(f"{name}: passed")
