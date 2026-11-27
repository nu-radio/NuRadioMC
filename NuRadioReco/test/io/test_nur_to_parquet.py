"""
Test of the .nur -> parquet converter `nur_to_parquet.convert_nur` and of `eventReader` for both formats.
Usage: ``python test_nur_to_parquet.py`` (or with pytest).
"""

import json
import pathlib
import tempfile
import warnings

import numpy as np
import pyarrow.parquet as pq

from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.io.eventReaderNur import eventReaderNur
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.io.nur_to_parquet import convert_nur

NUR_FILE = pathlib.Path(__file__).parents[3] / "NuRadioMC/test/SingleEvents/1e18_output_reference.nur"


def _read_nur():
    reader = eventReaderNur()
    reader.begin(str(NUR_FILE))
    events = list(reader.run())
    reader.end()

    return events


def _read_parquet(path):
    reader = eventReader()
    reader.begin(str(path))
    events = list(reader.run())
    reader.end()

    return events


def test_convert(tmp_path):
    written = convert_nur(str(NUR_FILE), output_dir=str(tmp_path))
    assert sorted(pathlib.Path(f).name for f in written) == [
        "1e18_output_reference.parquet",
        "1e18_output_reference_event.parquet",
    ]

    metadata = pq.read_schema(tmp_path / "1e18_output_reference.parquet").metadata
    assert json.loads(metadata[b"source_files"]) == [NUR_FILE.name]
    assert NUR_FILE.name in json.loads(metadata[b"nur_file_version"])

    originals, converted = _read_nur(), _read_parquet(tmp_path)
    assert len(converted) == len(originals) > 0
    assert [(e.get_run_number(), e.get_id()) for e in converted] == [(e.get_run_number(), e.get_id()) for e in originals]

    a, b = originals[0].get_station(), converted[0].get_station()
    assert a.get_id() == b.get_id()
    assert a[stnp.channels_max_amplitude] == b[stnp.channels_max_amplitude]
    channel_a, channel_b = a.get_channel(0), b.get_channel(0)
    assert channel_a[chp.SNR] == channel_b[chp.SNR]
    np.testing.assert_array_equal(channel_a.get_trace(), channel_b.get_trace())

    try:  # no overwrite by default
        convert_nur(str(NUR_FILE), output_dir=str(tmp_path))
        raise AssertionError("FileExistsError expected")
    except FileExistsError:
        pass


def test_event_reader_formats(tmp_path):
    """eventReader: parquet, .nur (routed to eventReaderNur with a warning), mixed inputs, old keywords."""
    convert_nur(str(NUR_FILE), output_dir=str(tmp_path))
    n_events = len(_read_nur())

    reader = eventReader()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader.begin(filename=str(NUR_FILE), read_detector=True)  # old keyword names

    assert [w.category for w in caught] == [FutureWarning] and "convert" in str(caught[0].message)
    assert caught[0].filename == __file__
    assert reader.get_n_events() == n_events == len(list(reader.run()))
    assert reader.get_header()  # the header dict of the .nur reader
    reader.close()

    parquet_file = str(tmp_path / "1e18_output_reference.parquet")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader.begin(parquet_file, read_detector=True)
        assert reader.get_detector() is None

    assert [w.category for w in caught] == [FutureWarning] * 2  # read_detector, get_detector
    assert reader.get_n_events() == n_events == len(list(reader.run()))
    reader.close()
    reader.close()  # safe to call twice
    reader.end()

    try:
        reader.begin([str(NUR_FILE), parquet_file])
        raise AssertionError("ValueError expected for mixed inputs")
    except ValueError as e:
        assert ".nur and parquet" in str(e)


if __name__ == "__main__":
    # Run without pytest (as in the CI): every test gets its own temporary directory
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            with tempfile.TemporaryDirectory() as directory:
                test(pathlib.Path(directory))

            print(f"{name}: passed")
