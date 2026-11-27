"""
Test of ``NuRadioMC.simulation.output_writer_hdf5.outputWriterHDF5``: the trigger datasets of the HDF5 output when the
set or the order of the trigger names changes between event groups.

The columns of ``multiple_triggers`` and ``trigger_times`` (top level and station groups) and of
``multiple_triggers_per_event`` and ``trigger_times_per_event`` (station groups) have to follow the ``trigger_names``
attribute: a trigger that first appears in a later event group adds a column (False / NaN for the earlier rows).
"""
import os
import tempfile

import h5py
import numpy as np

from NuRadioMC.simulation.output_writer_hdf5 import outputWriterHDF5
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.sim_station import SimStation
from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.particle import Particle
from NuRadioReco.framework.radio_shower import RadioShower
from NuRadioReco.framework.trigger import Trigger
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import particleParameters as pap
from NuRadioReco.framework.parameters import showerParameters as shp

nan = np.nan


class _Detector:
    """ The detector methods ``write_output_file`` needs: stations with one channel at the origin. """

    def get_number_of_channels(self, station_id):
        return 1

    def get_channel_ids(self, station_id):
        return [0]

    def get_relative_position(self, station_id, channel_id):
        return np.zeros(3)

    def get_absolute_position(self, station_id):
        return np.zeros(3)


def _event(group_id, station_id, triggers):
    """
    Event 0 of the event group ``group_id`` with one sim shower (id ``group_id``) and the station ``station_id``.
    ``triggers`` is a list of (name, has_triggered, trigger_time), set on the station in this order.
    """
    evt = Event(group_id, 0)
    particle = Particle(0)
    for key, value in [(pap.energy, 1e18), (pap.inelasticity, 0.5), (pap.weight, 1.), (pap.shower_id, group_id),
                       (pap.vertex, np.array([0., 0., -100.])), (pap.vertex_time, 0.), (pap.azimuth, 0.),
                       (pap.zenith, 1.), (pap.flavor, 14), (pap.n_interaction, 1), (pap.interaction_type, "cc")]:
        particle[key] = value
    evt.add_particle(particle)

    shower = RadioShower(np.int64(group_id))
    for key, value in [(shp.parent_id, 0), (shp.vertex, np.array([0., 0., -100.])), (shp.vertex_time, 0.),
                       (shp.azimuth, 0.), (shp.zenith, 1.), (shp.energy, 1e17), (shp.type, "had"),
                       (shp.flavor, 14), (shp.n_interaction, 1), (shp.interaction_type, "cc")]:
        shower[key] = value
    evt.add_sim_shower(shower)

    station = Station(station_id)
    station.set_sim_station(SimStation(station_id))
    channel = Channel(0)
    channel[chp.maximum_amplitude] = 1.
    channel[chp.maximum_amplitude_envelope] = 1.
    station.add_channel(channel)
    for name, fired, time in triggers:
        trigger = Trigger(name)
        trigger.set_triggered(fired)
        if fired:
            trigger.set_trigger_time(time)
        station.set_trigger(trigger)
    evt.set_station(station)
    return evt


def _write(groups, station_ids=(101,)):
    """
    Write one event group per entry of ``groups`` and return the trigger names and datasets of the output file.
    Each entry maps a station id to the trigger list of its (only) event; stations not in the entry did not trigger.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "out.hdf5")
        writer = outputWriterHDF5(filename, {'speedup': {'amp_per_ray_solution': False}}, _Detector(),
                                  list(station_ids), 2)
        for group_id, group in enumerate(groups):
            writer.add_event_group({
                station_id: ({0: _event(group_id, station_id, group[station_id])} if station_id in group else {})
                for station_id in station_ids})
        assert writer.write_output_file()

        with h5py.File(filename, "r") as f:
            out = {'trigger_names': [str(name) for name in f.attrs['trigger_names']]}
            for key in ['triggered', 'multiple_triggers', 'trigger_times']:
                out[key] = f[key][()]
            for station_id in station_ids:
                sg = f[f"station_{station_id}"]
                if 'multiple_triggers' not in sg:
                    continue
                for key in ['multiple_triggers', 'trigger_times', 'multiple_triggers_per_event',
                            'trigger_times_per_event']:
                    out[f"{station_id}/{key}"] = sg[key][()]
    return out


def _check(groups, trigger_names, multiple_triggers, trigger_times):
    """ Single station 101 with one event and one shower per group: all trigger datasets hold the same rows. """
    out = _write([{101: g} for g in groups])
    assert out['trigger_names'] == trigger_names, out['trigger_names']
    for prefix in ['', '101/']:
        np.testing.assert_array_equal(out[prefix + 'multiple_triggers'], multiple_triggers)
        np.testing.assert_array_equal(out[prefix + 'trigger_times'], trigger_times)
    np.testing.assert_array_equal(out['101/multiple_triggers_per_event'], multiple_triggers)
    np.testing.assert_array_equal(out['101/trigger_times_per_event'], trigger_times)


def test_same_triggers():
    _check([[("A", True, 5.), ("B", False, 0)], [("A", True, 7.), ("B", False, 0)]],
           ["A", "B"], [[True, False], [True, False]], [[5., nan], [7., nan]])


def test_new_trigger_fires():
    _check([[("A", True, 5.)], [("A", True, 7.), ("B", True, 9.)]],
           ["A", "B"], [[True, False], [True, True]], [[5., nan], [7., 9.]])


def test_new_trigger_fires_alone():
    _check([[("A", True, 5.)], [("A", False, 0), ("B", True, 9.)]],
           ["A", "B"], [[True, False], [False, True]], [[5., nan], [nan, 9.]])


def test_new_trigger_with_two_known():
    _check([[("A", True, 5.), ("B", False, 0)], [("A", True, 7.), ("B", False, 0), ("C", True, 9.)]],
           ["A", "B", "C"], [[True, False, False], [True, False, True]], [[5., nan, nan], [7., nan, 9.]])


def test_trigger_missing_in_later_group():
    _check([[("A", True, 5.), ("B", True, 6.)], [("A", True, 7.)]],
           ["A", "B"], [[True, True], [True, False]], [[5., 6.], [7., nan]])


def test_trigger_order_changes():
    _check([[("A", True, 5.), ("B", False, 0)], [("B", True, 8.), ("A", False, 0)]],
           ["A", "B"], [[True, False], [False, True]], [[5., nan], [nan, 8.]])


def test_stations_with_different_triggers():
    # station 101 runs trigger A and triggers in group 0, station 102 runs A and B and triggers in group 1
    out = _write([{101: [("A", True, 5.)]}, {102: [("A", False, 0), ("B", True, 9.)]}], station_ids=(101, 102))
    assert out['trigger_names'] == ["A", "B"], out['trigger_names']
    np.testing.assert_array_equal(out['triggered'], [True, True])
    np.testing.assert_array_equal(out['multiple_triggers'], [[True, False], [False, True]])
    np.testing.assert_array_equal(out['trigger_times'], [[5., nan], [nan, 9.]])
    for station_id, row, times in [(101, [True, False], [5., nan]), (102, [False, True], [nan, 9.])]:
        for key in ['multiple_triggers', 'multiple_triggers_per_event']:
            np.testing.assert_array_equal(out[f"{station_id}/{key}"], [row])
        for key in ['trigger_times', 'trigger_times_per_event']:
            np.testing.assert_array_equal(out[f"{station_id}/{key}"], [times])


if __name__ == "__main__":
    test_same_triggers()
    test_new_trigger_fires()
    test_new_trigger_fires_alone()
    test_new_trigger_with_two_known()
    test_trigger_missing_in_later_group()
    test_trigger_order_changes()
    test_stations_with_different_triggers()
    print("OK")
