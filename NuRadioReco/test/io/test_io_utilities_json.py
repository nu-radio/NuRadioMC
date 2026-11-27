""" Round trip of the tagged JSON helpers `io_utilities._to_json` / `_from_json`. """

import json
import logging

import astropy.time
import numpy as np

from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.utilities import io_utilities
from NuRadioReco.utilities import units


def _assert_equal(a, b):
    """Deep equality, also checking types (e.g. tuple vs list, ndarray dtype / shape)."""
    assert type(a) is type(b), (a, b)
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, (a, b)
        if a.dtype.kind == "O":
            for x, y in zip(a.ravel(), b.ravel()):
                _assert_equal(x, y)
        else:
            np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), (a, b)

        for key in a:
            _assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), (a, b)

        for x, y in zip(a, b):
            _assert_equal(x, y)
    elif isinstance(a, astropy.time.Time):
        assert a.jd1 == b.jd1 and a.jd2 == b.jd2 and a.scale == b.scale
    elif isinstance(a, float) and np.isnan(a):
        assert np.isnan(b)
    else:
        assert a == b, (a, b)


def _roundtrip(obj):
    string = io_utilities._to_json(obj)
    json.loads(string)  # plain JSON (apart from nan / inf)

    return io_utilities._from_json(string)


def test_native_types():
    obj = {
        "s": "text",
        "i": 3,
        "f": 0.1 + 0.2,
        "b": True,
        "n": None,
        "l": [1, 2.5, "x"],
        "nested": {"a": [{"b": 1}]},
        "unit": 50 * units.MHz,
        "nan": float("nan"),
        "inf": float("inf"),
    }

    _assert_equal(_roundtrip(obj), obj)


def test_tagged_types():
    obj = {
        "tuple": (1, (2, 3)),
        "int_keys": {1: "a", 2: "b"},
        "tuple_keys": {(1, 2): [3]},
        "tag_key": {"__type__": "not a tag"},
        "complex": 1 + 2j,
        "bytes": b"\x00\xff",
        "array": np.arange(6, dtype=np.float32).reshape(2, 3),
        "int_array": np.array([1, 2], dtype=np.int16),
        "complex_array": np.array([1j, 2]),
        "empty_array": np.zeros((0, 3)),
        "str_array": np.array(["a", "bc"]),
        "object_array": np.array([[1, 2], "x"], dtype=object),
        "time": astropy.time.Time("2024-01-01T12:00:00.123456789", format="isot"),
        "enums": [stnp.nu_energy, chp.SNR],
    }

    _assert_equal(_roundtrip(obj), obj)


def test_numpy_scalars():
    restored = _roundtrip([np.float64(1.5), np.float32(0.1), np.int64(3), np.bool_(True), np.str_("a")])

    _assert_equal(restored, [1.5, float(np.float32(0.1)), 3, True, "a"])


def test_repr_fallback():
    class Unknown:
        def __repr__(self):
            return "Unknown()"

    messages = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    io_utilities.logger.addHandler(handler)

    try:
        restored = _roundtrip({"x": Unknown()})
    finally:
        io_utilities.logger.removeHandler(handler)

    assert restored == {"x": "Unknown()"}
    assert len(messages) == 1 and "repr" in messages[0]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"{name}: passed")
