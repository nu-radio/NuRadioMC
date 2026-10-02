"""
Example: write and read the table based parquet format.

The parquet files store the derived quantities of the events (the content of the parameter storages),
never the traces. They are meant for the analysis after the processing: reading, filtering and
joining is fast and works with standard tools (here: polars).

The script has two parts:

1. Write: a small set of simulated events is written with `eventWriterParquet`
   (skipped if a directory with existing files is passed with ``--parquetdir``).
   In a real processing chain, the writer is used like the `eventWriter`: call `begin` once, `run` for
   every event and station (e.g. at the end of the processing loop), and `end` at the very end.
2. Read: `ParquetReader` finds the files, selects stations and runs, and returns polars frames.

Requires pyarrow and polars: ``pip install -e .[parquet]``.

Usage::

    python parquet_io_example.py                       # write a small example data set, then read it
    python parquet_io_example.py --parquetdir out/     # read the files in out/ (e.g. written by your processing)
"""
import argparse
import tempfile

import numpy as np
import polars as pl

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
from NuRadioReco.utilities import units
from NuRadioReco.utilities.parquet_utilities import ParquetReader


def make_event(rng, run, event_id, station_id, n_channels=4):
    """ A fake simulated event with station / channel parameters, sim channels and electric fields. """
    evt = NuRadioReco.framework.event.Event(run, event_id)
    station = NuRadioReco.framework.station.Station(station_id)

    # Station parameters: stored in one scalar (or nested) column each
    station[stnp.nu_energy] = 10 ** rng.uniform(17, 19) * units.eV
    station[stnp.zenith] = rng.uniform(0, 180) * units.deg
    station[stnp.azimuth] = rng.uniform(0, 360) * units.deg
    station[stnp.nu_vertex] = rng.uniform(-1000, 1000, 3) * units.m
    station[stnp.triggered] = bool(rng.random() > 0.5)

    # Channel parameters: stored as list columns, one entry per channel
    for channel_id in range(n_channels):
        channel = NuRadioReco.framework.channel.Channel(channel_id)
        channel[chp.noise_rms] = 10 * units.mV
        channel[chp.SNR] = {"peak_amplitude": rng.uniform(1, 20), "peak_2_peak_amplitude": rng.uniform(1, 30)}
        station.add_channel(channel)

    # Sim station and sim channels (one per channel and ray tracing solution): own table
    sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
    sim_station[stnp.nu_energy] = station[stnp.nu_energy]
    for channel_id in range(n_channels):
        for ray_id in [0, 1]:
            sim_channel = NuRadioReco.framework.sim_channel.SimChannel(channel_id, shower_id=0, ray_tracing_id=ray_id)
            sim_channel[chp.noise_rms] = 10 * units.mV
            sim_station.add_channel(sim_channel)

    # Electric fields (here: of the sim station): own table with one row per field
    for channel_id in range(n_channels):
        for ray_id, ray_path_type in [(0, "direct"), (1, "refracted")]:
            efield = NuRadioReco.framework.electric_field.ElectricField(
                [channel_id], shower_id=0, ray_tracing_id=ray_id)
            efield[efp.ray_path_type] = ray_path_type
            efield[efp.zenith] = rng.uniform(0, 180) * units.deg
            sim_station.add_electric_field(efield)

    station.set_sim_station(sim_station)
    evt.set_station(station)
    return evt, station


def write_example_data(output_dir):
    """ Write 2 stations x 2 runs x 25 events with `eventWriterParquet`. """
    rng = np.random.default_rng(42)

    writer = eventWriterParquet()
    writer.begin(
        output_dir,  # directory: the file names are station<id>_run<run>[_<suffix>].parquet
        suffix="example",  # optional
        store_sim=True,  # sim station columns + sim channel table (*_simchannel.parquet)
        store_efields=True,  # electric field table (*_efield.parquet)
        overwrite=True,  # default: raise an error if a file exists
        comment="parquet io example",  # free text, stored in the file metadata
    )
    for run in [1, 2]:
        for event_id in range(25):
            for station_id in [11, 12]:
                writer.run(*make_event(rng, run, event_id, station_id))

    print(f"Wrote {writer.end()} events to {output_dir}\n")


def read_example(path):
    # The path is a directory (searched recursively) or a glob pattern, e.g. "out/station11_run*.parquet".
    # Stations and runs are selected via the file names: other files are never opened.
    # Only the event files are read here (``efields`` / ``sim_channels`` are False by default).
    reader = ParquetReader(path, stations=[11, 12], runs=[1, 2])
    print("Event files:", *reader.files["event"], sep="\n  ")

    # 1. Event table. It is *lazy*: nothing is read until `collect()`, and then only the needed columns
    #    and row groups. The columns are named <parameter enum class>.<parameter>.
    events = reader.events()
    print(f"\nThe event table has {len(events.collect_schema().names())} columns, e.g.:")
    print(events.select("station_id", "run", "event_id", "stationParameters.nu_energy").head(3).collect())

    # 2. Filter and select columns: this is the typical analysis step.
    selection = (
        events.filter(pl.col("stationParameters.triggered") & (pl.col("stationParameters.nu_energy") > 1e18 * units.eV))
        .select("station_id", "run", "event_id", "stationParameters.nu_energy", "stationParameters.zenith")
    )
    df = selection.collect()
    print(f"\n{df.height} triggered events with E > 1e18 eV:")
    print(df.head(3))

    # 3. Channel parameters are list columns aligned with the column `channel_ids`.
    #    Maps (e.g. the SNR, a dict in the parameter storage) are read by polars as list of {key, value}.
    channels = events.select("event_id", "channel_ids", "channelParameters.noise_rms", "channelParameters.SNR")
    print("\nChannel parameters (one entry per channel):")
    print(channels.head(1).collect())

    # `explode` creates one row per list entry; `unnest` turns the {key, value} structs into columns.
    per_channel = (
        events.select("run", "event_id", "channel_ids", "channelParameters.SNR")
        .explode("channel_ids", "channelParameters.SNR", empty_as_null=True)  # one row per channel
        .explode("channelParameters.SNR", empty_as_null=True)  # one row per map entry
        .unnest("channelParameters.SNR")
        .filter(pl.col("key") == "peak_amplitude")
        .rename({"channel_ids": "channel_id", "value": "SNR_peak_amplitude"})
        .drop("key")
    )
    print("\nOne row per channel and event:")
    print(per_channel.head(3).collect())

    # 4. The electric fields and sim channels of a *selection* of events (own tables, one row per object).
    #    Only the files of the stations / runs in the selection are read, and only the rows of these events.
    related = reader.related_tables(selection)
    efields = related["efield"].collect()
    print(f"\nElectric fields of the selection: {efields.height} rows, e.g.:")
    print(efields.select("event_id", "source", "channel_ids", "ray_tracing_id", "electricFieldParameters.ray_path_type").head(3))
    print(f"Sim channels of the selection: {related['simchannel'].collect().height} rows")

    # 5. Alternatively: read everything joined to the event table (left join on station_id, run, event_id).
    #    One row per electric field; events without one are kept. Request only one of efields / sim_channels
    #    for the join, or use `tables()` to get all tables separately.
    joined = ParquetReader(path, efields=True).read()
    print(f"\nEvents joined with their electric fields: {joined.height} rows")

    # The tables are sorted (station_id, run, event_id, ...) unless ``sort=False`` is passed.
    # That needs the whole table in memory: disable it for very large data sets.

    # 6. Metadata (version, git commit, comment) of a file
    import pyarrow.parquet as pq
    print("\nMetadata:", {k.decode(): v.decode() for k, v in pq.read_schema(reader.files["event"][0]).metadata.items()
                          if k.startswith(b"nuradio") or k == b"comment"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write and read the parquet format.")
    parser.add_argument("--parquetdir", type=str, default=None,
                        help="Directory (or glob pattern) with parquet files to read. Default: write an example data set.")
    args = parser.parse_args()

    path = args.parquetdir
    if path is None:
        path = tempfile.mkdtemp(prefix="parquet_example_")
        write_example_data(path)

    read_example(path)
