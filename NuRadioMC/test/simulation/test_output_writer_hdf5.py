"""
Test of ``NuRadioMC.simulation.output_writer_hdf5.outputWriterHDF5``: the station datasets of the HDF5 output for an
event group that the simulation split into several events.
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

STATION_ID = 101
EVENT_GROUP_ID = 3


class _Detector:
    """ The detector methods ``write_output_file`` needs: a station with one channel at the origin. """

    def get_number_of_channels(self, station_id):
        return 1

    def get_channel_ids(self, station_id):
        return [0]

    def get_relative_position(self, station_id, channel_id):
        return np.zeros(3)

    def get_absolute_position(self, station_id):
        return np.zeros(3)


def _event(event_id, shower_ids, int_type):
    """
    Triggered event ``event_id`` of the event group ``EVENT_GROUP_ID`` with one sim shower per shower id, of type
    ``int_type``.
    """
    evt = Event(EVENT_GROUP_ID, event_id)
    particle = Particle(0)
    for key, value in [(pap.energy, 1e18), (pap.inelasticity, 0.5), (pap.weight, 1.), (pap.shower_id, min(shower_ids)),
                       (pap.vertex, np.array([0., 0., -100.])), (pap.vertex_time, 0.), (pap.azimuth, 0.),
                       (pap.zenith, 1.), (pap.flavor, 14), (pap.n_interaction, 1), (pap.interaction_type, "cc")]:
        particle[key] = value
    evt.add_particle(particle)

    for shower_id in shower_ids:
        shower = RadioShower(int_type(shower_id))
        for key, value in [(shp.parent_id, 0), (shp.vertex, np.array([0., 0., -100.])), (shp.vertex_time, 0.),
                           (shp.azimuth, 0.), (shp.zenith, 1.), (shp.energy, 1e17), (shp.type, "had"),
                           (shp.flavor, 14), (shp.n_interaction, 1), (shp.interaction_type, "cc")]:
            shower[key] = value
        evt.add_sim_shower(shower)

    station = Station(STATION_ID)
    station.set_sim_station(SimStation(STATION_ID))
    channel = Channel(0)
    channel[chp.maximum_amplitude] = 1.
    channel[chp.maximum_amplitude_envelope] = 1.
    station.add_channel(channel)
    trigger = Trigger("trigger")
    trigger.set_triggered(True)
    trigger.set_trigger_time(0.)
    station.set_trigger(trigger)
    evt.set_station(station)
    return evt


def _check_event_id_per_shower(int_type):
    # event 1 holds the lower shower ids, so sorting the showers by id also tests that the event ids follow their showers;
    # shower 5 contributes to both events and is stored once, with the first event
    event_buffer = {STATION_ID: {0: _event(0, [5], int_type), 1: _event(1, [2, 4, 5], int_type)}}
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "out.hdf5")
        writer = outputWriterHDF5(filename, {'speedup': {'amp_per_ray_solution': False}}, _Detector(), [STATION_ID], 2)
        writer.add_event_group(event_buffer)
        assert writer.write_output_file()
        with h5py.File(filename, "r") as f:
            sg = f[f"station_{STATION_ID}"]
            np.testing.assert_array_equal(sg["shower_id"], [2, 4, 5])
            np.testing.assert_array_equal(sg["event_group_id_per_shower"], [EVENT_GROUP_ID] * 3)
            np.testing.assert_array_equal(sg["event_id_per_shower"], [1, 1, 0])
            np.testing.assert_array_equal(sg["event_ids"], [0, 1])


def test_event_id_per_shower():
    # numpy integers, as in the simulation (shower ids read from the input file)
    _check_event_id_per_shower(np.int64)


def test_event_id_per_shower_python_int():
    _check_event_id_per_shower(int)


if __name__ == "__main__":
    test_event_id_per_shower()
    test_event_id_per_shower_python_int()
    print("OK")
