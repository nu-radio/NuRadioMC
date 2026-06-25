import numpy as np

from NuRadioReco.modules.io.LOFAR.readLOFARData import parse_block_number_file
from NuRadioReco.modules.LOFAR.stationTraceCropper_LOFAR import stationTraceCropper
from NuRadioReco.modules.LOFAR.utilities import iftDataHelpers
from NuRadioReco.modules.LOFAR.utilities import jaxHelpers


def test_parse_block_number_file_preserves_raw_columns(tmp_path):
    block_file = tmp_path / "blocks.txt"
    block_file.write_text(
        "# event_sec event_ns tbb_sec tbb_ref offset quality\n"
        "1305550330 842917919 1305550331 112650240 0.294005870819 1660331.13774\n"
        "bad row\n"
    )

    rows = parse_block_number_file(block_file)
    event_id = 1305550330 - 1262304000

    assert sorted(rows) == [event_id]
    assert rows[event_id]["event_timestamp"] == 1305550330
    assert rows[event_id]["event_nanoseconds"] == 842917919
    assert rows[event_id]["tbb_timestamp"] == 1305550331
    assert rows[event_id]["tbb_reference"] == 112650240
    assert rows[event_id]["raw_columns"][-1] == 1660331.13774


def test_simple_energy_fluence_uses_integrated_field_power():
    trace = np.array([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]])
    times = np.array([0.0, 0.5, 1.0])

    fluence = iftDataHelpers.get_electric_field_energy_fluence(trace, times)
    expected = np.sum(trace ** 2, axis=1) * 0.5 * iftDataHelpers.conversion_factor_integrated_signal

    assert np.allclose(fluence, expected)


def test_power_peak_finder_finds_synthetic_pulse():
    times = np.arange(100, dtype=float)
    traces = np.zeros((2, 100))
    traces[:, 42] = 10.0

    peak_time, snr = iftDataHelpers.power_peak_finder(traces, times)

    assert abs(peak_time - 42.0) <= 5.0
    assert snr > 0


def test_trace_cropper_global_peak_uses_clean_high_snr_channels():
    cropper = stationTraceCropper()
    cropper.begin(target_length=16, snr_threshold=2.0, fluence_outlier_sigma=1.0)

    peak = cropper._global_peak_index([
        {"peak_index": 5, "fluence": 10.0, "snr": 3.0, "trace_length": 32},
        {"peak_index": 7, "fluence": 11.0, "snr": 4.0, "trace_length": 32},
        {"peak_index": 31, "fluence": 1000.0, "snr": 20.0, "trace_length": 32},
        {"peak_index": 1, "fluence": 1.0, "snr": 0.5, "trace_length": 32},
    ])

    assert peak == 6


def test_jax_helper_package_data_lookup():
    path = jaxHelpers.get_ift_data_path("geo_rcut_b_splines.pickle")

    assert path.endswith("NuRadioReco/utilities/data/LOFAR/ift/geo_rcut_b_splines.pickle")
    assert jaxHelpers.get_ift_data_path("not-present.npz", required=False) is None
