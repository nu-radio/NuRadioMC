from NuRadioReco.detector import detector
from NuRadioReco.framework import electric_field
import logging
import datetime
import numpy as np
import os


def test_detector(always_query_entire_description=True, cleanup=True):
    test_filepath = os.path.join(
        os.path.dirname(__file__), 'test-rnog-detector.json.xz')

    det = detector.Detector(
        source="rnog_mongo", log_level=logging.DEBUG,
        always_query_entire_description=always_query_entire_description,
        database_connection='RNOG_public', select_stations=24)
    det.update(datetime.datetime(2023, 8, 2, 0, 0))
    response = det.get_signal_chain_response(24, 0)

    det.export(test_filepath)

    # test detector loaded from json
    det_from_file = detector.Detector(
        source="rnog_mongo", detector_file=test_filepath,
        select_stations=24, create_new=True)
    det_from_file.update(datetime.datetime(2023, 8, 2, 0, 0))

    response_from_file = det_from_file.get_signal_chain_response(24, 0)

    efields = []
    for i in range(2):
        ef = electric_field.ElectricField(channel_ids=[0])
        ef.set_frequency_spectrum(np.ones(1025, dtype=complex), sampling_rate=2.4)
        efields.append(ef)

    ef_from_db = efields[0] * response
    ef_from_file = efields[1] * response_from_file

    assert np.allclose(
        ef_from_db.get_frequency_spectrum(), ef_from_file.get_frequency_spectrum(),
        atol=1e-8, rtol=1e-6)

    if cleanup and os.path.isfile(test_filepath):
        os.remove(test_filepath)


if __name__ == "__main__":
    test_detector()
    test_detector(always_query_entire_description=False)
