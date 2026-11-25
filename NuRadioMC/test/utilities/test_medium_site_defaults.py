"""
Tests of the site defaults and refractive-index helpers in NuRadioMC.utilities.medium
(get_station_ice_model, get_refractive_index, get_n_air).
"""
import numpy as np

from NuRadioMC.utilities import medium, medium_base
from NuRadioReco.utilities import units


class SiteDetector:
    """Minimal detector: only get_site (as e.g. detector.Detector / GenericDetector)."""

    def __init__(self, site):
        self.site = site

    def get_site(self, station_id):
        return self.site


class IceModelDetector(SiteDetector):
    """Detector that can carry an ice model per station (as rnog_detector.Detector, PR #1224)."""

    def __init__(self, site, ice_model=None):
        super().__init__(site)
        self.ice_model = ice_model

    def get_ice_model(self, station_id):
        if self.ice_model is None:
            raise KeyError("no ice model attached")
        return self.ice_model


def n_model(ice_model, z):
    return float(ice_model.get_index_of_refraction(np.array([0, 0, z])))


def test_site_defaults():
    for site, name in [("summit", "greenland_3exp_timing_2025"), ("Greenland", "greenland_3exp_timing_2025"),
                       ("southpole", "southpole_2015"), ("SP", "southpole_2015"), ("mooresbay", "mooresbay_simple")]:
        ice_model = medium.get_station_ice_model(SiteDetector(site), 1)
        assert type(ice_model).__name__ == name, (site, type(ice_model).__name__)

    for site in ["lofar", "LOFAR", "auger", "ska"]:
        assert medium.get_station_ice_model(SiteDetector(site), 1) is None, site

    for site in ["simulation", "", "unknown_site"]:
        try:
            medium.get_station_ice_model(SiteDetector(site), 1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"site '{site}' did not raise")


def test_resolution_order():
    own_model = medium.uniform_ice()
    # explicit argument (object or name) beats detector and site
    assert medium.get_station_ice_model(IceModelDetector("summit", medium.southpole_2015()), 1, own_model) is own_model
    assert type(medium.get_station_ice_model(SiteDetector("summit"), 1, "greenland_simple")).__name__ == "greenland_simple"
    # explicit name works for sites without a default
    assert type(medium.get_station_ice_model(SiteDetector("simulation"), 1, "southpole_2015")).__name__ == "southpole_2015"
    # detector model beats the site default
    assert medium.get_station_ice_model(IceModelDetector("summit", own_model), 1) is own_model
    # no model attached to the detector (KeyError or None): site default
    assert type(medium.get_station_ice_model(IceModelDetector("summit"), 1)).__name__ == "greenland_3exp_timing_2025"
    none_detector = IceModelDetector("summit")
    none_detector.get_ice_model = lambda station_id: None
    assert type(medium.get_station_ice_model(none_detector, 1)).__name__ == "greenland_3exp_timing_2025"


def test_n_air():
    assert medium.get_n_air("summit") == medium.N_AIR_SUMMIT
    assert medium.get_n_air("Greenland") == medium.N_AIR_SUMMIT
    for site in ["southpole", "mooresbay", "lofar", "auger", "ska", "anything"]:
        assert medium.get_n_air(site) == medium.N_AIR_DEFAULT, site


def test_refractive_index():
    # in ice: the model's n(z)
    for name in ["greenland_3exp_timing_2025", "southpole_2015", "mooresbay_simple"]:
        ice_model = medium.get_ice_model(name)
        for z in np.array([-0.01, -2, -100]) * units.m:
            assert medium.get_refractive_index(z, ice_model, "summit") == n_model(ice_model, z), (name, z)

    # above the surface, model without air layer (returns n = 1 itself): site air value
    southpole = medium.southpole_2015()
    assert n_model(southpole, 1 * units.m) == 1
    assert medium.get_refractive_index(1 * units.m, southpole, "southpole") == medium.N_AIR_DEFAULT
    assert medium.get_refractive_index(1 * units.m, medium.greenland_simple(), "summit") == medium.N_AIR_SUMMIT

    # above the surface, model with air layer: the air layer
    greenland = medium.greenland_3exp_timing_2025()
    for z in np.array([0.01, 100, 9000]) * units.m:
        assert medium.get_refractive_index(z, greenland, "summit") == n_model(greenland, z), z
    np.testing.assert_allclose(medium.get_refractive_index(0.01 * units.m, greenland, "summit"), medium.N_AIR_SUMMIT, rtol=1e-8)
    assert medium.get_refractive_index(9 * units.km, greenland, "summit") < medium.N_AIR_SUMMIT

    # air layer of IceModelExpLayers.from_IceModel (exponential_air_layer): the air layer
    converted = medium_base.IceModelExpLayers.from_IceModel(medium.southpole_2015(), n_air_surface=1.0003)
    assert medium.get_refractive_index(1 * units.m, converted, "southpole") == n_model(converted, 1 * units.m)
    np.testing.assert_allclose(medium.get_refractive_index(0.01 * units.m, converted, "southpole"), 1.0003, rtol=1e-8)

    # air layer without the "region" key of exponential_air_layer (n ~ 1 placeholder): site air value
    layers = [dict(layer) for layer in converted.layers]
    for layer in layers:
        if layer.get("region") == "air":
            del layer["region"]
    placeholder = medium_base.IceModelExpLayers(layers)
    assert medium.get_refractive_index(1 * units.m, placeholder, "southpole") == medium.N_AIR_DEFAULT

    # no ice (air-only site): air value everywhere
    for z in np.array([10, -0.01, -2]) * units.m:
        assert medium.get_refractive_index(z, None, "lofar") == medium.N_AIR_DEFAULT, z


def test_station_refractive_index():
    greenland = medium.greenland_3exp_timing_2025()
    for z in np.array([10, -0.01, -100]) * units.m:
        assert medium.get_station_refractive_index(SiteDetector("summit"), 1, z) == \
            medium.get_refractive_index(z, greenland, "summit"), z

    # site without default: air value above the surface, error below
    assert medium.get_station_refractive_index(SiteDetector("simulation"), 1, 1 * units.m) == medium.N_AIR_DEFAULT
    try:
        medium.get_station_refractive_index(SiteDetector("simulation"), 1, -1 * units.m)
    except ValueError:
        pass
    else:
        raise AssertionError("in-ice value for site 'simulation' did not raise")
    # ... unless a model is given
    southpole = medium.southpole_2015()
    assert medium.get_station_refractive_index(SiteDetector("simulation"), 1, -1 * units.m, southpole) == \
        n_model(southpole, -1 * units.m)


if __name__ == "__main__":
    test_site_defaults()
    test_resolution_order()
    test_n_air()
    test_refractive_index()
    test_station_refractive_index()
    print("OK")
