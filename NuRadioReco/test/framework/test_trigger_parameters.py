"""
Tests for the trigger classes stored as `triggerParameters`:

* every trigger class constructs and survives a serialize / deserialize cycle exactly,
* .nur files written before the triggerParameters (pickled ``__dict__``) are read into the parameters.

``legacy_triggers.nur`` (one trigger of every type, legacy format) was written with the trigger classes
before the triggerParameters were introduced.
"""
import os

import numpy as np

from NuRadioReco.framework import trigger as trg
from NuRadioReco.framework.parameters import triggerParameters as trp
from NuRadioReco.framework.station import Station
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.utilities import units

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(TEST_DIR, "..", "..", ".."))

LEGACY_FILES = [
    os.path.join(TEST_DIR, "legacy_triggers.nur"),
    os.path.join(REPO_DIR, "NuRadioMC/test/SingleEvents/1e18_output_reference.nur"),
    os.path.join(REPO_DIR, "NuRadioMC/examples/01_Veff_simulation_B/1e19_output_reference.nur"),
    os.path.join(REPO_DIR, "NuRadioReco/examples/neutrinoDirectionReconstructor/data/triggeredNeutrinoEvents.nur"),
]

BASE_PARAMETERS = {trp.channels, trp.triggered, trp.trigger_time, trp.trigger_times, trp.triggered_channels,
                   trp.primary, trp.pre_trigger_times, trp.sample_block_size, trp.gaussian_jitter}

# class -> parameters it sets (besides the base ones)
CLASS_PARAMETERS = {
    trg.Trigger: [],
    trg.SimpleThresholdTrigger: [trp.threshold, trp.number_of_coincidences, trp.coinc_window],
    trg.HighLowTrigger: [trp.threshold_high, trp.threshold_low,
                         trp.high_low_window, trp.coinc_window, trp.number_of_coincidences],
    trg.IntegratedPowerTrigger: [trp.threshold, trp.coinc_window, trp.number_of_coincidences,
                                 trp.power_mean, trp.power_std, trp.integration_window],
    trg.EnvelopeTrigger: [trp.passband, trp.filter_order, trp.threshold, trp.number_of_coincidences, trp.coinc_window],
    trg.RNOGSurfaceTrigger: [trp.threshold, trp.number_of_coincidences, trp.coinc_window, trp.temperature, trp.bias_voltage],
    trg.SimplePhasedTrigger: [trp.phasing_angles, trp.secondary_channels, trp.secondary_angles, trp.threshold,
                              trp.trigger_delays, trp.sec_trigger_delays, trp.window_size, trp.step_size, trp.maximum_amps],
    trg.DigitalEnvelopePhasedTrigger: [trp.phasing_angles, trp.threshold, trp.trigger_delays, trp.maximum_amps],
    trg.EnvelopePhasedTrigger: [trp.phasing_angles, trp.threshold_factor, trp.power_mean, trp.power_std,
                                trp.trigger_delays, trp.output_passband],
}


def _example_triggers():
    """ One trigger of every class, with the values the trigger modules set. """
    triggers = []

    t = trg.Trigger("LT")
    t.set_triggered(True)
    t.set_trigger_time(10.)
    triggers.append(t)

    t = trg.SimpleThresholdTrigger("simple", {0: 1.5, 1: 2.5}, channels=[0, 1], number_of_coincidences=2,
                                   channel_coincidence_window=30 * units.ns, pre_trigger_times={0: 50., 1: 60.})
    t.set_triggered(True)
    t.set_trigger_time(np.float64(100.))
    t.set_triggered_channels([0, 1])
    t.set_primary(True)
    t.set_jitter_params(sample_block_size=4, gaussian_jitter=1.5)
    triggers.append(t)

    t = trg.HighLowTrigger("high_low", 0.1, -0.1, 5., 30., channels=[0, 1, 2], number_of_coincidences=2)
    t.set_trigger_times(np.array([120., 130.]))
    triggers.append(t)

    t = trg.IntegratedPowerTrigger("int_power", 0.3, 20., channels=[0, 1], power_mean=0.1, power_std=0.05)
    triggers.append(t)

    triggers.append(trg.EnvelopeTrigger("envelope", [80 * units.MHz, 200 * units.MHz], 4, 0.05, channels=[0, 1]))
    triggers.append(trg.RNOGSurfaceTrigger("surface", 0.02))

    t = trg.SimplePhasedTrigger("phased", 30., channels=[0, 1, 2], primary_angles=np.deg2rad([-10., 0., 10.]),
                                trigger_delays={0: {0: 0., 1: 1.}}, window_size=24, step_size=8,
                                maximum_amps=np.array([31., 2., 40.]))
    t.set_triggered(np.bool_(True))
    t.set_trigger_time(200.)
    t.set_trigger_times({0: np.array([200., 210.]), 2: np.array([205.])})
    triggers.append(t)

    t = trg.DigitalEnvelopePhasedTrigger("digital_env", 12., trigger_channels=[0, 1],
                                         phasing_angles=np.array([0.1, 0.2]), trigger_delays={},
                                         maximum_amps=np.array([3., 13.]))
    triggers.append(t)

    t = trg.EnvelopePhasedTrigger("analog_env", 2.5, 1.2, 0.3, triggered_channels=[0, 1, 2],
                                  phasing_angles=np.array([0.1, 0.2]), trigger_delays={0: 1., 1: 2.},
                                  output_passband=(None, 200 * units.MHz))
    triggers.append(t)

    return triggers


def _assert_identical(a, b):
    """ Same class, name, type and parameters (values and their types). """
    assert type(a) is type(b), (type(a), type(b))
    assert a.get_name() == b.get_name() and a.get_type() == b.get_type()
    pa, pb = a.get_parameters(), b.get_parameters()
    assert pa.keys() == pb.keys(), (a.get_name(), set(pa) ^ set(pb))
    for key in pa:
        np.testing.assert_equal(pb[key], pa[key], err_msg=f"{a.get_name()} {key}")
        # numpy scalars are stored as their python equivalent (`io_utilities._NurPickler`)
        expected_type = type(pa[key].item()) if isinstance(pa[key], np.generic) else type(pa[key])
        assert type(pb[key]) is expected_type, (a.get_name(), key, type(pa[key]), type(pb[key]))


def _assert_complete(trigger):
    """ All base parameters and all parameters of the trigger's class are set. """
    keys = set(trigger.get_parameters())
    name = (trigger.get_name(), trigger.get_type())
    expected = BASE_PARAMETERS | set(CLASS_PARAMETERS[type(trigger)])
    assert keys == expected, (name, keys ^ expected)


def _read_triggers(filename):
    """ All triggers (of stations and sim stations) in a .nur file. """
    triggers = []
    reader = eventReader()
    reader.begin(filename)
    for evt in reader.run():
        for station in evt.get_stations():
            triggers += list(station.get_triggers().values())
            if station.has_sim_station():
                triggers += list(station.get_sim_station().get_triggers().values())
    reader.end()
    return triggers


def test_construct_and_roundtrip():
    triggers = _example_triggers()
    assert {type(t) for t in triggers} == set(CLASS_PARAMETERS)
    for trigger in triggers:
        _assert_complete(trigger)
        trigger_read = trg.deserialize([trigger.serialize()])[trigger.get_name()]
        _assert_identical(trigger, trigger_read)

    # via a station (the path used for .nur files)
    station = Station(11)
    for trigger in triggers:
        station.set_trigger(trigger)
    station_read = Station(0)
    station_read.deserialize(station.serialize(mode={}))
    for trigger in triggers:
        _assert_identical(trigger, station_read.get_trigger(trigger.get_name()))


def test_registry():
    for trigger in _example_triggers():
        assert trg._TRIGGER_CLASSES[trigger.get_type()] is type(trigger)
    assert trg._TRIGGER_CLASSES["envelope_phased"] is trg.EnvelopePhasedTrigger

    # unknown type: read as base class
    trigger = trg.Trigger("custom", trigger_type="my_custom_trigger")
    trigger_read = trg.deserialize([trigger.serialize()])["custom"]
    assert type(trigger_read) is trg.Trigger and trigger_read.get_type() == "my_custom_trigger"


def test_getters():
    t = _example_triggers()
    settings = t[1].get_trigger_settings()
    for key in ["triggered", "trigger_time", "triggered_channels"]:  # used by trigger_tests/compare_to_reference.py
        assert key in settings
    assert t[1].get_pre_trigger_times() == {0: 50., 1: 60.}
    assert t[1].get_pre_trigger_time_channel(1) == 60.
    assert t[2].get_pre_trigger_time_channel(1) == 55 * units.ns
    assert t[1][trp.threshold] == {0: 1.5, 1: 2.5}

    # get_triggered_channels returns the stored list
    t[1].get_triggered_channels().append(5)
    assert t[1].get_triggered_channels() == [0, 1, 5]

    # phased: trigger times per beam
    phased = t[6]
    np.testing.assert_equal(phased.get_trigger_times(), {0: [200., 210.], 2: [205.]})

    assert "phased" in str(phased) and phased.show(print_stdout=False).startswith("Trigger(phased)")


def test_legacy_fixture():
    triggers = {t.get_name(): t for t in _read_triggers(LEGACY_FILES[0])}
    for name, trigger in triggers.items():
        if name != "int_power_bug":
            _assert_complete(trigger)

    t = triggers["simple"]
    assert type(t) is trg.SimpleThresholdTrigger
    assert t[trp.threshold] == {0: 1.5, 1: 2.5} and t[trp.pre_trigger_times] == {0: 50., 1: 60.}
    assert t.is_primary() and t.get_sample_block_size() == 4 and t.get_gaussian_jitter() == 1.5
    assert t[trp.number_of_coincidences] == 2 and t[trp.coinc_window] == 30 * units.ns

    t = triggers["high_low_dict"]
    assert t[trp.threshold_high] == {0: 0.1, 1: 0.2} and t[trp.threshold_low] == {0: -0.1, 1: -0.2}
    np.testing.assert_equal(triggers["high_low"].get_trigger_times(), [120., 130.])

    t = triggers["phased"]
    assert t.get_triggered_channels() == [] and t[trp.channels] == [0, 1, 2, 3]
    np.testing.assert_equal(t.get_trigger_times(), {0: [200., 210.], 2: [205.]})
    np.testing.assert_allclose(t[trp.phasing_angles], np.deg2rad([-10., 0., 10.]))
    assert t[trp.trigger_delays] == {0: {0: 0., 1: 1.}, 2: {0: 0., 1: -1.}}
    assert t[trp.window_size] == 24 and t[trp.step_size] == 8
    assert triggers["phased_old"][trp.step_size] == 8  # stored as `_step_side`

    t = triggers["digital_env"]
    assert t[trp.channels] == [0, 1] and t[trp.trigger_delays] == {1: {0: 0.5, 1: 0.}}

    t = triggers["analog_env"]
    assert type(t) is trg.EnvelopePhasedTrigger
    assert t[trp.trigger_delays] == {0: 1., 1: 2.} and t.get_triggered_channels() == [0, 1, 2]
    assert tuple(t[trp.output_passband]) == (None, 200 * units.MHz) and t[trp.threshold_factor] == 2.5

    t = triggers["int_power_bug"]  # `_coinc_window` held the channel list, an unknown attribute: both dropped
    assert not t.has_parameter(trp.coinc_window)
    assert set(t.get_trigger_settings()) == {"name", "trigger_type"} | {k.name for k in t.get_parameters()}

    assert triggers["envelope"][trp.filter_order] == 4
    assert triggers["surface"][trp.bias_voltage] == 2 * units.volt
    assert triggers["LT"].has_triggered() and triggers["LT"].get_trigger_time() == 10.


def test_legacy_files():
    for filename in LEGACY_FILES:
        triggers = _read_triggers(filename)
        assert len(triggers), filename
        for trigger in triggers:
            if trigger.get_name() != "int_power_bug":
                _assert_complete(trigger)
            # new format reads back exactly
            _assert_identical(trigger, trg.deserialize([trigger.serialize()])[trigger.get_name()])


if __name__ == "__main__":
    test_construct_and_roundtrip()
    test_registry()
    test_getters()
    test_legacy_fixture()
    test_legacy_files()
    print("All trigger parameter tests passed.")
