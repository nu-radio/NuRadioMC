"""
Tests that `ParameterStorage.get_parameter` (and the `[]` operator, which uses it
under the hood) returns a copy of mutable parameter values, so that modifying the
returned value does not silently corrupt the object's internal state.
"""
import numpy as np

from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.electric_field import ElectricField
from NuRadioReco.framework.parameters import electricFieldParameters as efp


def _make_storage():
    # `ParameterStorage` is not meant to be instantiated directly; use a real
    # subclass (`Station`) instead, which is set up with valid parameter types.
    return Station(0)


def test_list_parameter_is_copied():
    station = _make_storage()
    station.set_parameter(stnp.dirty_fft_channels, [1, 2, 3])

    retrieved = station.get_parameter(stnp.dirty_fft_channels)
    retrieved.append(4)

    assert station.get_parameter(stnp.dirty_fft_channels) == [1, 2, 3]
    assert retrieved == [1, 2, 3, 4]


def test_dict_parameter_is_copied():
    station = _make_storage()
    station.set_parameter(stnp.cr_xcorrelations, {"a": 1})

    retrieved = station.get_parameter(stnp.cr_xcorrelations)
    retrieved["b"] = 2

    assert station.get_parameter(stnp.cr_xcorrelations) == {"a": 1}
    assert retrieved == {"a": 1, "b": 2}


def test_nested_mutable_parameter_is_deep_copied():
    station = _make_storage()
    station.set_parameter(stnp.viewing_angles, {0: {0: 1.23}})

    retrieved = station.get_parameter(stnp.viewing_angles)
    retrieved[0][0] = 99.0

    # the nested dict must not be shared, otherwise this in-place edit would
    # leak into the stored parameter
    assert station.get_parameter(stnp.viewing_angles) == {0: {0: 1.23}}


def test_numpy_array_parameter_is_copied():
    station = _make_storage()
    array = np.array([1.0, 2.0, 3.0])
    station.set_parameter(stnp.nu_vertex, array)

    retrieved = station.get_parameter(stnp.nu_vertex)
    retrieved[0] = 99.0

    np.testing.assert_array_equal(station.get_parameter(stnp.nu_vertex), [1.0, 2.0, 3.0])
    assert retrieved[0] == 99.0


def test_getitem_operator_also_copies():
    station = _make_storage()
    station.set_parameter(stnp.dirty_fft_channels, [1, 2, 3])

    station[stnp.dirty_fft_channels].append(4)

    assert station[stnp.dirty_fft_channels] == [1, 2, 3]


def test_scalar_parameter_returned_directly_without_copy_overhead():
    station = _make_storage()
    station.set_parameter(stnp.nu_energy, 1e18)

    # scalars are immutable, so identity is preserved (and no needless copy is made)
    assert station.get_parameter(stnp.nu_energy) is station.get_parameter(stnp.nu_energy)


def test_copy_false_returns_the_stored_object_itself():
    station = _make_storage()
    stored = [1, 2, 3]
    station.set_parameter(stnp.dirty_fft_channels, stored)

    retrieved = station.get_parameter(stnp.dirty_fft_channels, copy=False)

    assert retrieved is stored


def test_set_parameter_after_get_modify_round_trip():
    # regression test for the pattern used throughout the codebase to safely
    # mutate a dict/list-valued parameter: get it, modify the copy, then set it back
    efield = ElectricField([0])
    efield.set_parameter(efp.max_amp_antenna, {})

    max_amp_antenna = efield.get_parameter(efp.max_amp_antenna)
    max_amp_antenna[0] = 1.5
    efield.set_parameter(efp.max_amp_antenna, max_amp_antenna)

    assert efield.get_parameter(efp.max_amp_antenna) == {0: 1.5}


if __name__ == "__main__":
    test_list_parameter_is_copied()
    test_dict_parameter_is_copied()
    test_nested_mutable_parameter_is_deep_copied()
    test_numpy_array_parameter_is_copied()
    test_getitem_operator_also_copies()
    test_scalar_parameter_returned_directly_without_copy_overhead()
    test_copy_false_returns_the_stored_object_itself()
    test_set_parameter_after_get_modify_round_trip()
    print("test_parameter_storage.py: all tests passed")
