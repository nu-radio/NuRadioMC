from NuRadioReco.detector import detector
import logging
import datetime
import os


def test_detector():

    det = detector.Detector(source="rnog_mongo", log_level=logging.DEBUG, always_query_entire_description=True,
                            database_connection='RNOG_public', select_stations=24)
    det.update(datetime.datetime(2023, 8, 2, 0, 0))
    det.export("station24.json.xz")

    det2 = detector.Detector(source="rnog_mongo", detector_file="station24.json.xz", select_stations=24)
    det2.update(datetime.datetime(2023, 8, 2, 0, 0))

    response = det.get_signal_chain_response(24, 0)


    from NuRadioReco.framework import electric_field
    with open(f"{os.path.dirname(__file__)}/test_ef.pkl", "rb") as f:
        ef = electric_field.ElectricField(channel_ids=[0])
        ef.deserialize(f.read())

    ef *= response

if __name__ == "__main__":
    test_detector()