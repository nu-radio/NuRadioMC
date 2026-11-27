"""
Example: write and read the table based parquet format.

The parquet files store complete events (parameters, traces, simulated quantities). Each parameter is a column,
so an analysis reads only the columns it needs without decoding the traces: reading and filtering is fast and
works with standard tools (here: polars).

The script has two parts:

1. Write: a small set of simulated events is written with `eventWriter`
   (skipped if a directory with existing files is passed with ``--parquetdir``).
   In a real processing chain, call `begin` once, `run` for every event or station (e.g. at the end of the
   processing loop), and `close` (or `end`) at the very end.
2. Read: `AnalysisReader` finds the files, selects stations and runs, and returns polars frames.
   (To loop over complete `Event` objects instead, use `NuRadioReco.modules.io.eventReader`.)

Requires pyarrow and polars >= 2: ``pip install -e .[analysis]``.

Usage::

    python parquet_io_example.py                       # write a small example data set, then read it
    python parquet_io_example.py --parquetdir out/     # read the files in out/ (e.g. written by your processing)
"""

import argparse
import os
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
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.utilities import units
from NuRadioReco.utilities.analysis_reader import AnalysisReader


def make_event(rng, run, event_id, station_id, n_channels=4):
    """A fake simulated event with station / channel parameters, sim channels and electric fields."""
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

    # Traces are stored as well (read only on request)
    for channel in station.iter_channels():
        channel.set_trace(rng.normal(0, 10 * units.mV, 256), 3.2 * units.GHz)

    # Sim station and sim channels (one per channel and ray tracing solution): list columns ``sim_channel.*``
    sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
    sim_station[stnp.nu_energy] = station[stnp.nu_energy]

    for channel_id in range(n_channels):
        for ray_id in [0, 1]:
            sim_channel = NuRadioReco.framework.sim_channel.SimChannel(channel_id, shower_id=0, ray_tracing_id=ray_id)
            sim_channel[chp.noise_rms] = 10 * units.mV
            sim_station.add_channel(sim_channel)

    # Electric fields (here: of the sim station): list columns ``sim_efield.*``, one entry per field
    for channel_id in range(n_channels):
        for ray_id, ray_path_type in [(0, "direct"), (1, "refracted")]:
            efield = NuRadioReco.framework.electric_field.ElectricField([channel_id], shower_id=0, ray_tracing_id=ray_id)
            efield[efp.ray_path_type] = ray_path_type
            efield[efp.zenith] = rng.uniform(0, 180) * units.deg
            sim_station.add_electric_field(efield)

    station.set_sim_station(sim_station)
    evt.set_station(station)

    return evt, station


def write_example_data(output_dir):
    """Write 2 stations x 2 runs x 25 events with `eventWriter`."""
    rng = np.random.default_rng(42)

    writer = eventWriter()
    writer.begin(
        os.path.join(output_dir, "example"),  # example.parquet (stations) + example_event.parquet (event level)
        max_file_size=1024,  # MB, then continue in example_part01[_event].parquet, ... (also: events_per_file)
        overwrite=True,  # default: raise an error if a file exists
        comment="parquet io example",  # free text, stored in the file metadata
    )

    for run in [1, 2]:
        for event_id in range(25):
            for station_id in [11, 12]:
                writer.run(*make_event(rng, run, event_id, station_id))  # (event, station): writes this station

    print(f"Wrote {writer.end()} events to {output_dir}\n")


def read_example(path):
    # The path is a directory (searched recursively), a file or a glob pattern, e.g. "out/example*.parquet".
    # Stations and runs are selected by filtering the rows; files whose row group statistics exclude them are not read.
    # The event files (event level information) are found next to the station files.
    reader = AnalysisReader(path, stations=[11, 12], runs=[1, 2])
    print("Station files:", *reader.files["station"], sep="\n  ")
    print("Event files:", *reader.files["event"], sep="\n  ")

    # 1. One row per event and station, joined with the event files (columns ``event.*``). It is *lazy*: nothing
    #    is read until `collect()`, and then only the needed columns and row groups.
    #    The parameter columns are named <parameter enum class>.<parameter>.
    events = reader.scan()
    print(f"\nThe table has {len(events.collect_schema().names())} columns, e.g.:")
    print(events.select("station_id", "run", "event_id", "stationParameters.nu_energy").head(3).collect())

    # 2. Filter and select columns: this is the typical analysis step.
    selection = events.filter(
        pl.col("stationParameters.triggered") & (pl.col("stationParameters.nu_energy") > 1e18 * units.eV)
    ).select("station_id", "run", "event_id", "stationParameters.nu_energy", "stationParameters.zenith")
    df = selection.collect()
    print(f"\n{df.height} triggered events with E > 1e18 eV:")
    print(df.head(3))

    # 3. Channel parameters are list columns aligned with the column `channel_ids`.
    #    Maps (e.g. the SNR, a dict in the parameter storage) are read by polars as `pl.Map`.
    channels = events.select("event_id", "channel_ids", "channelParameters.noise_rms", "channelParameters.SNR")
    print("\nChannel parameters (one entry per channel):")
    print(channels.head(1).collect())

    # `explode` creates one row per list entry. SNR is a struct with the fields `value` (a single SNR) and
    # `definitions` (a map per SNR definition); `map.get` returns the value of one key.
    per_channel = (
        events.select("run", "event_id", "channel_ids", "channelParameters.SNR")
        .explode("channel_ids", "channelParameters.SNR", empty_as_null=True)  # one row per channel
        .select(
            "run",
            "event_id",
            pl.col("channel_ids").alias("channel_id"),
            pl.col("channelParameters.SNR")
            .struct.field("definitions")
            .map.get("peak_amplitude")
            .alias("SNR_peak_amplitude"),
        )
    )
    print("\nOne row per channel and event:")
    print(per_channel.head(3).collect())

    # 4. Sim channels and electric fields are list columns as well, aligned with their ids
    #    (``sim_channel.channel_id``, ``sim_channel.ray_tracing_id``, ``sim_efield.channel_ids``, ...).
    efields = events.select(
        "event_id", "sim_efield.channel_ids", "sim_efield.ray_tracing_id", "sim_efield.electricFieldParameters.ray_path_type"
    ).explode(
        pl.exclude("event_id"), empty_as_null=True
    )  # one row per electric field
    print("\nElectric fields of the sim station, one row per field:")
    print(efields.head(3).collect())

    # 5. Traces are only read on request (``traces=True``, columns ``*.trace``): one list of samples per channel
    traces = AnalysisReader(path, stations=11, runs=1, traces=True).read()
    trace = traces["channel.trace"][0][0].to_numpy()
    print(
        f"\nTrace of the first channel of the first event: {len(trace)} samples, "
        f"sampling rate {traces['channel.sampling_rate'][0][0] / units.GHz} GHz"
    )

    # The tables are sorted (station_id, run, event_id) unless ``sort=False`` is passed.
    # That needs the whole table in memory: disable it for very large data sets.

    # 6. Metadata (version, git commit, comment, layout) of a file
    import pyarrow.parquet as pq

    print(
        "\nMetadata:",
        {
            k.decode(): v.decode()
            for k, v in pq.read_schema(reader.files["station"][0]).metadata.items()
            if k.startswith(b"nuradio") or k in (b"comment", b"layout")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write and read the parquet format.")
    parser.add_argument(
        "--parquetdir",
        type=str,
        default=None,
        help="Directory (or glob pattern) with parquet files to read. Default: write an example data set.",
    )
    args = parser.parse_args()

    path = args.parquetdir
    if path is None:
        path = tempfile.mkdtemp(prefix="parquet_example_")
        write_example_data(path)

    read_example(path)
