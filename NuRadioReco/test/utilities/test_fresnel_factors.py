"""
Test ``geometryUtilities.fresnel_factors_and_signal_zenith`` for an antenna above the surface and a signal
coming from below, i.e. refraction from the ice into the air.
"""
import numpy as np
from astropy.time import Time

from NuRadioReco.detector import detector
from NuRadioReco.framework import station
from NuRadioReco.utilities import geometryUtilities, ice, units

STATION_ID, CHANNEL_ID = 101, 0  # channel 0 of the dummy detector is 1 m above the surface


def test_signal_from_below():
    det = detector.Detector(json_filename="NuRadioReco/examples/example_data/dummy_detector.json")
    det.update(Time("2020-01-01"))
    assert det.get_relative_position(STATION_ID, CHANNEL_ID)[2] > 0
    stn = station.Station(STATION_ID)
    stn.set_is_neutrino()
    n_ice = ice.get_refractive_index(-0.01, det.get_site(STATION_ID))

    # transmitted into the air: Snell's law n_ice * sin(zenith) = 1 * sin(zenith_antenna)
    zenith = 150 * units.deg
    zenith_antenna, t_theta, t_phi = geometryUtilities.fresnel_factors_and_signal_zenith(det, stn, CHANNEL_ID, zenith)
    assert zenith_antenna > 90 * units.deg, f"signal should still come from below, got {zenith_antenna / units.deg} deg"
    np.testing.assert_allclose(np.sin(zenith_antenna), n_ice * np.sin(zenith), rtol=1e-12)

    # total internal reflection: no signal reaches the antenna
    zenith = 120 * units.deg
    assert n_ice * np.sin(zenith) > 1
    assert geometryUtilities.fresnel_factors_and_signal_zenith(det, stn, CHANNEL_ID, zenith) == (None, None, None)


if __name__ == "__main__":
    test_signal_from_below()
    print("OK")
