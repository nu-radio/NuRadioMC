"""
Eventbrowser file handler for parquet files (`ParquetFileHandler`): write a ``.nur`` file to parquet, open it via the
`DataProvider` and check the number of events, the ids, random access (out of order; events equal to those of the
sequential `eventReader`), the header and opening an event file.

Usage: ``python test_eventbrowser_parquet.py [file.nur]``
"""

import pathlib
import sys
import tempfile

import numpy as np

from NuRadioReco.eventbrowser.dataprovider import DataProvider
from NuRadioReco.eventbrowser.parquet_file_handler import ParquetFileHandler
from NuRadioReco.framework import parameters
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.io.eventWriter import eventWriter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_parquet_roundtrip import _Comparison  # noqa: E402

INPUT = pathlib.Path(__file__).parents[3] / "NuRadioMC/test/SingleEvents/1e18_output_reference.nur"


def _write(nur_file, output, event_file):
    """Writes `nur_file` to parquet (small row groups: random access across them). Returns the (run, id) list."""
    reader = eventReader()
    reader.begin(str(nur_file))
    writer = eventWriter()
    writer.begin(
        str(output),
        event_file=event_file,
        row_group_size=3,
        station_enums=[parameters.stationParameters, parameters.stationParametersRNOG],
        channel_enums=[parameters.channelParameters, parameters.channelParametersRNOG],
    )
    ids = []

    for evt in reader.run():
        writer.run(evt)
        ids.append((evt.get_run_number(), evt.get_id()))

    writer.end()
    reader.end()

    return ids


def _check(station_file, ids):
    handler = DataProvider().get_file_handler(f"test {station_file}", str(station_file))

    assert isinstance(handler, ParquetFileHandler)
    assert handler.get_n_events() == len(ids)
    assert [tuple(i) for i in handler.get_event_ids()] == ids
    assert handler.get_detector() is None
    assert handler.get_event_i(len(ids)) is None

    reader = eventReader()
    reader.begin(str(station_file))
    sequential = list(reader.run())
    reader.end()

    order = np.random.default_rng(1).permutation(len(ids))

    for i in list(order) + [0, 0, len(ids) - 1]:
        comparison = _Comparison()
        comparison.event(sequential[i], handler.get_event_i(int(i)))
        assert not comparison.differences, comparison.differences[:5]

    assert handler.get_event(ids[-1]).get_id() == ids[-1][1]

    # Header: one entry per event of the station, in event order
    header = handler.get_header()

    for station_id, values in header.items():
        stations = [evt.get_station(station_id) for evt in sequential if station_id in evt.get_station_ids()]
        assert len(values[stnp.station_time]) == len(stations)

        for member, array in values.items():
            for station, value in zip(stations, array):
                if member == stnp.station_time:
                    assert value == station.get_station_time()
                elif station.has_parameter(member):
                    expected = station.get_parameter(member)
                    if isinstance(expected, float):
                        assert value == expected or (np.isnan(value) and np.isnan(expected))
                    elif member.dtype in (int, bool, str):
                        assert value == expected

    return handler, header


def test_eventbrowser_parquet(tmp_path, nur_file=INPUT):
    ids = _write(nur_file, tmp_path / "dual", event_file=True)
    handler, header = _check(tmp_path / "dual.parquet", ids)

    assert header

    # An event file is opened via its station file
    event_handler = ParquetFileHandler(str(tmp_path / "dual_event.parquet"))
    assert event_handler.get_filenames() == [str(tmp_path / "dual.parquet")]

    ids = _write(nur_file, tmp_path / "single", event_file=False)
    _check(tmp_path / "single.parquet", ids)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_eventbrowser_parquet(pathlib.Path(directory), *[pathlib.Path(path) for path in sys.argv[1:2]])
    print("test_eventbrowser_parquet: passed")
