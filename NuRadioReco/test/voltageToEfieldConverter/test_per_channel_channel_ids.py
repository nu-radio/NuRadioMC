"""
Test that ``voltageToEfieldConverterPerChannel`` assigns each electric field the id of the channel it was
unfolded from, also when the channel ids do not start at 0.
"""
import numpy as np
from astropy.time import Time

from NuRadioReco.detector import detector
from NuRadioReco.framework import channel, event, station
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.voltageToEfieldConverterPerChannel import voltageToEfieldConverterPerChannel
from NuRadioReco.utilities import units

CHANNEL_IDS = [1, 0]  # channel 0 uses analytic_LPDA too; added second so that ids and insertion order differ


def test_channel_ids():
    det = detector.Detector(json_filename="NuRadioReco/examples/example_data/dummy_detector.json")
    det.update(Time("2020-01-01"))

    evt = event.Event(0, 0)
    stn = station.Station(101)
    stn[stnp.zenith], stn[stnp.azimuth] = 60 * units.deg, 0
    stn.set_is_cosmic_ray()
    rng = np.random.default_rng(0)
    for channel_id in CHANNEL_IDS:
        ch = channel.Channel(channel_id)
        ch.set_trace(rng.normal(size=256), 1 * units.GHz)
        stn.add_channel(ch)
    evt.set_station(stn)

    voltageToEfieldConverterPerChannel().run(evt, stn, det)

    efield_channel_ids = [efield.get_channel_ids() for efield in stn.get_electric_fields()]
    assert efield_channel_ids == [[channel_id] for channel_id in CHANNEL_IDS], \
        f"electric fields have channel ids {efield_channel_ids}, expected {[[c] for c in CHANNEL_IDS]}"


if __name__ == "__main__":
    test_channel_ids()
    print("OK")
