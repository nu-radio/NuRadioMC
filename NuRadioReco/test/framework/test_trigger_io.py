"""
Tests that triggers survive a write / read cycle through a ``.nur`` file and that
`Trigger.get_trigger_times` handles triggers without a trigger time.
"""
import os
import tempfile

import numpy as np

from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework import trigger as trg
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.utilities import units


def _write_and_read(triggers):
    """ Write one event with the given triggers to a .nur file and return the station read back. """
    evt = Event(1, 1)
    station = Station(11)
    for trigger in triggers:
        station.set_trigger(trigger)
    evt.set_station(station)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "trigger_io.nur")
        writer = eventWriter()
        writer.begin(filename)
        writer.run(evt)
        writer.end()

        reader = eventReader()
        reader.begin(filename)
        events = list(reader.run())
        reader.end()

    assert len(events) == 1
    return events[0].get_station(11)


def test_envelope_phased_trigger_nur_roundtrip():
    trigger = trg.EnvelopePhasedTrigger(
        "envelope_phased", 2.5, 1.2, 0.3, triggered_channels=[0, 1, 2, 3],
        phasing_angles=np.deg2rad([-10, 0, 10]), trigger_delays={0: 1 * units.ns},
        output_passband=(None, 200 * units.MHz))
    trigger.set_triggered(True)
    trigger.set_trigger_time(123 * units.ns)

    station = _write_and_read([trigger])
    trigger_read = station.get_trigger("envelope_phased")

    assert isinstance(trigger_read, trg.EnvelopePhasedTrigger)
    assert trigger_read.get_type() == trigger.get_type()
    assert trigger_read.has_triggered()
    assert trigger_read.get_trigger_time() == 123 * units.ns
    settings, settings_read = trigger.get_trigger_settings(), trigger_read.get_trigger_settings()
    assert settings.keys() == settings_read.keys()
    for key in ["threshold_factor", "power_mean", "power_std", "triggered_channels",
                "trigger_delays", "output_passband"]:
        assert settings_read[key] == settings[key], key
    np.testing.assert_allclose(settings_read["phasing_angles"], settings["phasing_angles"])


def test_simple_phased_trigger_step_size_nur_roundtrip():
    trigger = trg.SimplePhasedTrigger("simple_phased", 1.5, channels=[0, 1], window_size=16, step_size=8)

    station = _write_and_read([trigger])
    settings_read = station.get_trigger("simple_phased").get_trigger_settings()

    assert settings_read["window_size"] == 16
    assert settings_read["step_size"] == 8

    # files written before the fix store the step size as `_step_side`
    trigger.__dict__["_step_side"] = trigger.__dict__.pop("_step_size")
    settings_read = trg.deserialize([trigger.serialize()])["simple_phased"].get_trigger_settings()
    assert settings_read["step_size"] == 8
    assert "step_side" not in settings_read


def test_get_trigger_times_without_trigger_time():
    trigger = trg.Trigger("not_triggered")
    assert trigger.get_trigger_time() is None
    assert trigger.get_trigger_times() is None

    trigger.set_trigger_time(np.nan)
    assert trigger.get_trigger_times() is None

    trigger.set_trigger_time(5 * units.ns)
    np.testing.assert_equal(trigger.get_trigger_times(), 5 * units.ns)

    station = _write_and_read([trg.Trigger("not_triggered")])
    assert station.get_trigger("not_triggered").get_trigger_times() is None


if __name__ == "__main__":
    test_envelope_phased_trigger_nur_roundtrip()
    test_simple_phased_trigger_step_size_nur_roundtrip()
    test_get_trigger_times_without_trigger_time()
    print("All trigger I/O tests passed.")
