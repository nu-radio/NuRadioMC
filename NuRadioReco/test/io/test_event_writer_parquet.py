import collections
import pathlib
import tempfile

import astropy.time
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import NuRadioReco.framework.channel
import NuRadioReco.framework.electric_field
import NuRadioReco.framework.event
import NuRadioReco.framework.sim_channel
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.station
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.io.eventWriterParquet import eventWriterParquet
from NuRadioReco.utilities.parquet_utilities import ParquetReader


def _make_event(run, event_id, channel_ids=(0, 1, 2), station_id=11):
    evt = NuRadioReco.framework.event.Event(run, event_id)
    station = NuRadioReco.framework.station.Station(station_id)
    station.set_station_time(astropy.time.Time("2024-01-01T12:00:00.123456789", format="isot"))
    station[stnp.nu_energy] = 1e18 * event_id
    station[stnp.nu_vertex] = np.array([1., 2., 3.])
    station[stnp.triggered] = True
    station[stnp.viewing_angles] = {np.int64(0): {1: np.float64(0.5)}}
    station[stnp.flagged_channels] = collections.defaultdict(list, {2: ["a", "b"]})
    for channel_id in channel_ids:
        channel = NuRadioReco.framework.channel.Channel(channel_id)
        if channel_id != 1:  # leave one channel unset
            channel[chp.SNR] = {"peak_amplitude": 3.}
            channel[chp.noise_rms] = 0.1 * channel_id
        station.add_channel(channel)
    evt.set_station(station)
    return evt, station


def test_write_read(tmp_path):
    writer = eventWriterParquet()
    writer.begin(str(tmp_path), suffix="out", batch_size=2, comment="my comment")
    for run, event_id in [(1, 1), (1, 2), (1, 3), (2, 1), (1, 4)]:  # run 1 is revisited
        evt, station = _make_event(run, event_id)
        writer.run(evt, station)
    assert writer.end() == 5

    files = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert files == ["station11_run000001_out.parquet", "station11_run000001_out_part01.parquet", "station11_run000002_out.parquet"]
    assert pq.ParquetFile(tmp_path / files[0]).num_row_groups == 2  # 3 events, batch_size=2

    df = pl.read_parquet(tmp_path / "station11_run000001_out.parquet")
    assert df["event_id"].to_list() == [1, 2, 3]
    assert df["stationParameters.nu_energy"].to_list() == [1e18, 2e18, 3e18]
    assert df["stationParameters.nu_vertex"][0].to_list() == [1., 2., 3.]
    assert df["stationParameters.zenith"].null_count() == 3
    assert df["channel_ids"][0].to_list() == [0, 1, 2]
    assert df["channelParameters.noise_rms"][0].to_list() == [0., None, 0.2]
    time = astropy.time.Time(df["station_time_jd1"][0], df["station_time_jd2"][0], format="jd", scale="utc")
    assert abs((time - astropy.time.Time("2024-01-01T12:00:00.123456789", format="isot")).to_value("s")) < 1e-9
    metadata = pq.read_schema(tmp_path / "station11_run000001_out.parquet").metadata
    assert metadata[b"comment"] == b"my comment"
    assert {b"nuradio_version", b"nuradio_commit"} <= set(metadata)
    table = pq.read_table(tmp_path / "station11_run000001_out.parquet")  # polars reads maps as list[struct]
    assert table["stationParameters.viewing_angles"][0].as_py() == [(0, [(1, 0.5)])]
    assert table["stationParameters.flagged_channels"][0].as_py() == [(2, ["a", "b"])]


def test_multiple_stations(tmp_path):
    writer = eventWriterParquet()
    writer.begin(str(tmp_path))
    for event_id in [1, 2]:
        for station_id in [1, 12]:
            writer.run(*_make_event(5, event_id, station_id=station_id))
    assert writer.end() == 4
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == ["station01_run000005.parquet", "station12_run000005.parquet"]
    assert pl.read_parquet(tmp_path / "station12_run000005.parquet")["event_id"].to_list() == [1, 2]


def test_overwrite(tmp_path):
    def write(**kwargs):
        writer = eventWriterParquet()
        writer.begin(str(tmp_path), **kwargs)
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


def test_sim_and_efields(tmp_path):
    evt, station = _make_event(3, 9)
    sim_station = NuRadioReco.framework.sim_station.SimStation(station.get_id())
    sim_station[stnp.nu_energy] = 5e18
    for channel_id, shower_id, ray_id in [(1, 0, 1), (0, 0, 0), (0, 0, 1)]:  # unsorted on purpose
        sim_channel = NuRadioReco.framework.sim_channel.SimChannel(channel_id, shower_id, ray_id)
        sim_channel[chp.noise_rms] = 0.5 * channel_id
        sim_station.add_channel(sim_channel)
    sim_efield = NuRadioReco.framework.electric_field.ElectricField([0, 1], position=[1., 2., 3.], shower_id=0, ray_tracing_id=1)
    sim_efield[efp.ray_path_type] = "direct"
    sim_station.add_electric_field(sim_efield)
    station.set_sim_station(sim_station)
    efield = NuRadioReco.framework.electric_field.ElectricField([2])
    efield[efp.zenith] = 0.3
    station.add_electric_field(efield)

    writer = eventWriterParquet()
    writer.begin(str(tmp_path), store_sim=True, store_efields=True)
    writer.run(evt, station)
    writer.end()

    base = "station11_run000003"
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == [
        f"{base}.parquet", f"{base}_efield.parquet", f"{base}_simchannel.parquet"]

    main = pl.read_parquet(tmp_path / f"{base}.parquet")
    assert main["sim_stationParameters.nu_energy"].to_list() == [5e18]

    sim_channels = pl.read_parquet(tmp_path / f"{base}_simchannel.parquet")
    assert sim_channels["channel_id"].to_list() == [0, 0, 1]
    assert sim_channels["ray_tracing_id"].to_list() == [0, 1, 1]
    assert sim_channels["channelParameters.noise_rms"].to_list() == [0., 0., 0.5]
    assert sim_channels["event_id"].to_list() == [9, 9, 9]

    efields = pl.read_parquet(tmp_path / f"{base}_efield.parquet")
    assert efields["source"].to_list() == ["station", "sim_station"]
    assert efields["channel_ids"].to_list() == [[2], [0, 1]]
    assert efields["position"][1].to_list() == [1., 2., 3.]
    assert efields["electricFieldParameters.zenith"].to_list() == [0.3, None]
    assert efields["electricFieldParameters.ray_path_type"].to_list() == [None, "direct"]


def test_no_channel_parameters(tmp_path):
    writer = eventWriterParquet()
    writer.begin(str(tmp_path), store_channels=False)
    writer.run(*_make_event(1, 1))
    writer.end()

    columns = pl.read_parquet(tmp_path / "station11_run000001.parquet").columns
    assert "channel_ids" in columns
    assert not any(column.startswith("channelParameters.") for column in columns)


def test_reader(tmp_path):
    writer = eventWriterParquet()
    writer.begin(str(tmp_path / "sub"), store_sim=True, store_efields=True)
    for run in [1, 2]:
        for station_id in [11, 12]:
            evt, station = _make_event(run, 1, station_id=station_id)
            evt.set_station(station)
            sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
            sim_station.add_channel(NuRadioReco.framework.sim_channel.SimChannel(0, 0, 0))
            station.set_sim_station(sim_station)
            station.add_electric_field(NuRadioReco.framework.electric_field.ElectricField([0]))
            station.add_electric_field(NuRadioReco.framework.electric_field.ElectricField([1]))
            writer.run(evt, station)
    writer.end()

    reader = ParquetReader(str(tmp_path))  # root directory, searched recursively
    assert sorted(reader.files) == ["event"]
    assert len(reader.files["event"]) == 4
    assert reader.read().height == 4

    df = ParquetReader(str(tmp_path / "sub" / "*.parquet"), stations=11, runs=[2], efields=True).read()
    assert df.height == 2  # one event with two electric fields
    assert df["station_id"].unique().to_list() == [11] and df["run"].unique().to_list() == [2]
    assert "channel_ids_efield" in df.columns

    tables = ParquetReader(str(tmp_path), sim_channels=True, efields=True).tables()
    assert sorted(tables) == ["efield", "event", "simchannel"]
    assert tables["efield"].collect().height == 8

    try:
        ParquetReader(str(tmp_path), sim_channels=True, efields=True).scan()
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous join was allowed")


def test_interleaved_stations_and_runs(tmp_path):
    writer = eventWriterParquet()
    writer.begin(str(tmp_path))
    # Station 11 and 12 are in different runs and alternate: no file must be closed early
    for event_id, (run_11, run_12) in enumerate([(5, 3), (5, 3), (5, 3), (6, 3)]):
        writer.run(*_make_event(run_11, event_id, station_id=11))
        writer.run(*_make_event(run_12, event_id, station_id=12))
    assert writer.end() == 8

    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == [
        "station11_run000005.parquet", "station11_run000006.parquet", "station12_run000003.parquet"]
    assert pl.read_parquet(tmp_path / "station12_run000003.parquet").height == 4
    assert pl.read_parquet(tmp_path / "station11_run000005.parquet").height == 3


def test_reader_missing_side_files(tmp_path):
    writer = eventWriterParquet()
    writer.begin(str(tmp_path), store_efields=True)  # no sim channel files
    writer.run(*_make_event(1, 1))
    writer.end()

    # The pattern only matches the event file, the efield file is still found next to it
    reader = ParquetReader(str(tmp_path / "station11_run*[0-9].parquet"), efields=True)
    assert len(reader.files["efield"]) == 1

    reader = ParquetReader(str(tmp_path), efields=True, sim_channels=True)
    assert reader.files["efield"] != [] and reader.files["simchannel"] == []


def test_reader_sorted(tmp_path):
    writer = eventWriterParquet()
    writer.begin(str(tmp_path), store_sim=True, store_efields=True)
    for station_id, run, event_id in [(12, 2, 3), (11, 2, 2), (12, 1, 5), (11, 2, 1), (11, 1, 9)]:  # unsorted on purpose
        evt, station = _make_event(run, event_id, station_id=station_id)
        sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
        for channel_id, ray_id in [(2, 1), (0, 1), (2, 0)]:
            sim_station.add_channel(NuRadioReco.framework.sim_channel.SimChannel(channel_id, 0, ray_id))
        station.set_sim_station(sim_station)
        for channel_ids, ray_id in [([3], 1), ([1], 1), ([1], 0)]:
            station.add_electric_field(
                NuRadioReco.framework.electric_field.ElectricField(channel_ids, shower_id=0, ray_tracing_id=ray_id))
        writer.run(evt, station)
    writer.end()

    keys = ["station_id", "run", "event_id"]
    tables = ParquetReader(str(tmp_path), sim_channels=True, efields=True).tables()
    assert tables["event"].select(keys).collect().rows() == sorted(tables["event"].select(keys).collect().rows())
    assert tables["event"].collect().height == 5

    sim_channels = tables["simchannel"].collect()
    assert sim_channels.rows() == sim_channels.sort(keys + ["channel_id", "ray_tracing_id", "shower_id"]).rows()
    assert sim_channels.filter(sim_channels["event_id"] == 9)["channel_id"].to_list() == [0, 2, 2]
    assert sim_channels.filter(sim_channels["event_id"] == 9)["ray_tracing_id"].to_list() == [1, 0, 1]

    efields = tables["efield"].collect().filter(pl.col("event_id") == 9)
    assert [(c.to_list(), r) for c, r in zip(efields["channel_ids"], efields["ray_tracing_id"])] == [([1], 0), ([1], 1), ([3], 1)]

    joined = ParquetReader(str(tmp_path), efields=True).read()
    assert joined.select(keys).rows() == sorted(joined.select(keys).rows())
    assert joined.filter(pl.col("event_id") == 9)["ray_tracing_id"].to_list() == [0, 1, 1]

    unsorted = ParquetReader(str(tmp_path), sort=False).read()  # file order: no sorting
    assert unsorted.height == 5


def test_reader_events_and_related_tables(tmp_path):
    writer = eventWriterParquet()
    writer.begin(str(tmp_path), store_sim=True, store_efields=True)
    for station_id, run, event_id in [(11, 1, 1), (11, 1, 2), (11, 2, 3), (12, 1, 4)]:
        evt, station = _make_event(run, event_id, station_id=station_id)
        sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
        for channel_id in [1, 0]:
            sim_station.add_channel(NuRadioReco.framework.sim_channel.SimChannel(channel_id, 0, 0))
        station.set_sim_station(sim_station)
        station.add_electric_field(NuRadioReco.framework.electric_field.ElectricField([event_id], shower_id=0, ray_tracing_id=0))
        writer.run(evt, station)
    writer.end()

    reader = ParquetReader(str(tmp_path))  # events only: the side files are not even looked for
    assert reader.files["event"] and sorted(reader.files) == ["event"]
    events = reader.events()
    assert events.collect().height == 4

    selection = events.filter((pl.col("station_id") == 11) & (pl.col("event_id") <= 2))
    related = reader.related_tables(selection)
    assert sorted(related) == ["efield", "simchannel"]
    assert related["efield"].collect()["event_id"].to_list() == [1, 2]
    assert related["simchannel"].collect()["event_id"].to_list() == [1, 1, 2, 2]
    assert related["simchannel"].collect()["channel_id"].to_list() == [0, 1, 0, 1]

    # a (collected) DataFrame works as well, a single table can be requested
    related = reader.related_tables(events.collect().filter(pl.col("event_id") == 4), sim_channels=False)
    assert sorted(related) == ["efield"] and related["efield"].collect()["station_id"].to_list() == [12]


if __name__ == "__main__":
    # Run without pytest (as in the CI): every test gets its own temporary directory
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            with tempfile.TemporaryDirectory() as directory:
                test(pathlib.Path(directory))

            print(f"{name}: passed")
