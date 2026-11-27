"""
Tests of `BaseStation` (particle type, trigger flag) and `Event` (event time, module registry) accessors.
"""

import astropy.time

from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station


def test_get_particle_type():
    station = Station(11)
    assert station.get_particle_type() == ''

    station.set_is_neutrino()
    assert station.get_particle_type() == 'nu'

    station.set_is_cosmic_ray()
    assert station.get_particle_type() == 'cr'


def test_set_triggered_flag():
    station = Station(11)
    assert not station.has_triggered()

    station.set_triggered_flag(True)
    assert station.has_triggered()
    assert station.get_triggers() == {}

    station.set_triggered_flag(False)
    assert not station.has_triggered()


def test_has_event_time():
    evt = Event(1, 2)
    assert not evt.has_event_time()

    evt.set_event_time(astropy.time.Time("2024-01-01T00:00:00", format="isot"))
    assert evt.has_event_time()


def test_module_registry_round_trip():
    evt = Event(1, 2)
    evt._register_module_event(None, "eventModule", {"a": 1})
    evt._register_module_station(11, None, "stationModule", {"b": 2})

    event_modules = evt.get_module_registry()
    station_modules = evt.get_module_registry(station_id=11)

    assert event_modules == [["eventModule", None, {"a": 1}]]
    assert station_modules == [[1, "stationModule", None, {"b": 2}]]
    assert evt.get_module_registry(station_id=12) == []

    event_modules.append(["other", None, {}])  # a copy
    assert len(evt.get_module_registry()) == 1

    new_evt = Event(1, 2)
    new_evt._set_module_registry(evt.get_module_registry())
    new_evt._set_module_registry(station_modules, station_id=11)

    assert new_evt.get_module_registry() == evt.get_module_registry()
    assert new_evt.get_module_registry(station_id=11) == station_modules
    assert list(new_evt.iter_modules(11)) == list(evt.iter_modules(11))


if __name__ == "__main__":
    test_get_particle_type()
    test_set_triggered_flag()
    test_has_event_time()
    test_module_registry_round_trip()
    print("test_station_event.py: all tests passed")
