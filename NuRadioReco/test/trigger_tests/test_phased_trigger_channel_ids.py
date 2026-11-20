"""
Tests that the trigger time of the phased array trigger does not depend on the channel ids:
the same station with channels 0-3 and with the channels relabeled to 10-13 must trigger at the same time.
"""
import datetime
import json
import os
import tempfile

import numpy as np

from NuRadioReco.detector import detector
from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.modules.phasedarray.beamformedPowerIntegrationTrigger import BeamformedPowerIntegrationTrigger
from NuRadioReco.utilities import units

DETECTOR_FILE = os.path.join(os.path.dirname(__file__), 'trigger_test_detector.json')
STATION_ID = 1
CHANNEL_IDS = [0, 1, 2, 3]
ID_OFFSET = 10


def _make_detector(json_filename):
    det = detector.Detector(json_filename=json_filename, antenna_by_depth=False)
    det.update(datetime.datetime(2018, 10, 1))
    return det


def _make_event(channel_ids):
    """ Station with identical (seeded) traces: a pulse on top of noise in each channel. """
    rng = np.random.default_rng(42)
    sampling_rate = 1 * units.GHz
    times = np.arange(256) / sampling_rate
    evt = Event(1, 1)
    station = Station(STATION_ID)
    for i, channel_id in enumerate(channel_ids):
        pulse = 0.2 * units.V * np.exp(-0.5 * ((times - (100 + 3 * i) * units.ns) / (2 * units.ns)) ** 2)
        channel = Channel(channel_id)
        channel.set_trace(pulse + rng.normal(0, 10 * units.mV, len(times)), sampling_rate)
        channel.set_trace_start_time(50 * units.ns)
        station.add_channel(channel)
    evt.set_station(station)
    return evt, station


def _run_trigger(det, channel_ids):
    evt, station = _make_event(channel_ids)
    trigger = BeamformedPowerIntegrationTrigger()  # fresh instance: phasing delays are buffered per station id
    trigger.run(evt, station, det, threshold=3e4 * units.mV ** 2, triggered_channels=channel_ids,
                apply_digitization=False, trigger_name='phased')
    return station.get_trigger('phased')


def test_phased_trigger_time_independent_of_channel_ids():
    with open(DETECTOR_FILE) as f:
        description = json.load(f)
    for channel in description['channels'].values():
        channel['channel_id'] = int(channel['channel_id']) + ID_OFFSET

    with tempfile.TemporaryDirectory() as tmpdir:
        relabeled_file = os.path.join(tmpdir, 'trigger_test_detector_relabeled.json')
        with open(relabeled_file, 'w') as f:
            json.dump(description, f)

        trigger = _run_trigger(_make_detector(DETECTOR_FILE), CHANNEL_IDS)
        trigger_relabeled = _run_trigger(
            _make_detector(relabeled_file), [channel_id + ID_OFFSET for channel_id in CHANNEL_IDS])

    assert trigger.has_triggered() and trigger_relabeled.has_triggered()
    np.testing.assert_allclose(trigger_relabeled.get_trigger_time(), trigger.get_trigger_time())

    trigger_times, trigger_times_relabeled = trigger.get_trigger_times(), trigger_relabeled.get_trigger_times()
    assert trigger_times.keys() == trigger_times_relabeled.keys()
    for beam in trigger_times:
        np.testing.assert_allclose(trigger_times_relabeled[beam], trigger_times[beam])


if __name__ == "__main__":
    test_phased_trigger_time_independent_of_channel_ids()
    print("Phased trigger channel-id test passed.")
