#!/usr/bin/env python3
"""
Tests for the removal of the detector arguments ``antenna_by_depth`` and ``assume_inf`` (v3.2):
``ant_type`` is used as the antenna model as is, ``antenna_by_depth=True`` raises, any other explicit
value of either argument only logs a warning, and loading a model name which lacks the
'_InfFirn'/'_InfAir' suffix raises an error which suggests the full name.

None of the tests needs an antenna model file.
"""
import datetime
import os
import unittest
import unittest.mock

import NuRadioReco.utilities.dataservers
from NuRadioReco.detector import detector, detector_base, generic_detector
from NuRadioReco.detector.antennapattern import AntennaPatternProvider
from NuRadioReco.utilities import units

# absolute paths: only `detector.Detector` resolves paths relative to NuRadioReco/detector
DETECTOR_DIR = os.path.dirname(os.path.abspath(detector.__file__))
ARIANNA_JSON = os.path.join(DETECTOR_DIR, "ARIANNA", "arianna_detector_db.json")
GENERIC_JSON = os.path.join(DETECTOR_DIR, "RNO_G", "RNO_single_station.json")

_case = unittest.TestCase()


def assert_warns_removed(names, func):
    """ Run `func` and check that it logs one "was removed" warning for each argument in `names` (none if empty) """
    # patch the logger instead of `assertNoLogs` (python >= 3.10) to also check for no warning on python 3.8
    with unittest.mock.patch.object(detector_base.logger, "warning") as warning:
        result = func()

    messages = [call.args[0] for call in warning.call_args_list]
    removed = [msg for msg in messages if "was removed in v3.2" in msg]
    assert len(removed) == len(names), messages
    for name in names:
        assert any(f"`{name}`" in msg for msg in removed), messages

    return result


def assert_raises_antenna_by_depth(func):
    with _case.assertRaisesRegex(ValueError, "`antenna_by_depth` was removed"):
        func()


def test_check_removed_arguments():
    assert_warns_removed([], detector_base.check_removed_arguments)  # nothing is logged for the default (None)

    assert_raises_antenna_by_depth(lambda: detector_base.check_removed_arguments(antenna_by_depth=True))
    assert_raises_antenna_by_depth(
        lambda: detector_base.check_removed_arguments(assume_inf=True, antenna_by_depth=True))

    assert_warns_removed(
        ["antenna_by_depth"], lambda: detector_base.check_removed_arguments(antenna_by_depth=False))
    assert_warns_removed(["assume_inf"], lambda: detector_base.check_removed_arguments(assume_inf=True))
    assert_warns_removed(["assume_inf"], lambda: detector_base.check_removed_arguments(assume_inf=False))
    assert_warns_removed(
        ["assume_inf", "antenna_by_depth"],
        lambda: detector_base.check_removed_arguments(assume_inf=True, antenna_by_depth=False))


def test_detector_arguments():
    """ The removed arguments are checked for all ways to create a json detector """
    for create in [
            lambda **kw: detector.Detector(json_filename=ARIANNA_JSON, create_new=True, **kw),
            lambda **kw: detector_base.DetectorBase(json_filename=ARIANNA_JSON, create_new=True, **kw),
            lambda **kw: generic_detector.GenericDetector(
                json_filename=GENERIC_JSON, create_new=True, **kw)]:

        assert_raises_antenna_by_depth(lambda: create(antenna_by_depth=True))
        assert_warns_removed(["antenna_by_depth"], lambda: create(antenna_by_depth=False))
        assert_warns_removed(["assume_inf"], lambda: create(assume_inf=True))

    # positional arguments: Detector(json_filename, source, dictionary, assume_inf, antenna_by_depth)
    assert_raises_antenna_by_depth(lambda: detector.Detector(ARIANNA_JSON, "json", None, None, True))
    assert_warns_removed(
        ["assume_inf", "antenna_by_depth"], lambda: detector.Detector(ARIANNA_JSON, "json", None, True, False))


def test_antenna_model_is_ant_type():
    """ The antenna model is the ant_type, independent of the depth and the signal direction """
    det = detector.Detector(json_filename=ARIANNA_JSON, create_new=True)
    det.update(datetime.datetime(2018, 1, 1))

    n_channels = 0
    for station_id in det.get_station_ids():
        try:
            channel_ids = det.get_channel_ids(station_id)
        except LookupError:  # station not commissioned at this time
            continue

        for channel_id in channel_ids:
            ant_type = det.get_antenna_type(station_id, channel_id)
            for zenith in [None, 45 * units.deg, 135 * units.deg]:
                assert det.get_antenna_model(station_id, channel_id, zenith) == ant_type
            n_channels += 1

    assert n_channels
    assert det.get_antenna_model(1, 0) == "createLPDA_100MHz_InfFirn"


def test_model_without_suffix_raises():
    """ A model name without the depth suffix raises (with a suggestion) before any download is attempted """
    with unittest.mock.patch.object(
            NuRadioReco.utilities.dataservers, "download_from_dataserver",
            side_effect=AssertionError("download attempted")):
        with _case.assertRaisesRegex(ValueError, "createLPDA_100MHz_InfFirn or createLPDA_100MHz_InfAir"):
            AntennaPatternProvider().load_antenna_pattern("createLPDA_100MHz")


if __name__ == "__main__":
    for test in [test_check_removed_arguments, test_detector_arguments,
                 test_antenna_model_is_ant_type, test_model_without_suffix_raises]:
        print("{} ...".format(test.__name__))
        test()

    print("\nTests for the removal of antenna_by_depth passed")
