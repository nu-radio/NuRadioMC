from NuRadioReco.detector.db_mongo_write import Database
from NuRadioReco.detector.webinterface import config

from rnog_data.runtable import RunTable
runtab = RunTable()

def get_station_ids_from_db():
    return runtab.get_quantity_names('runtable', 'station')


def get_firmware_from_db():
    return runtab.get_quantity_names('runtable', 'firmware_version')


def load_runs(station_list, start_time, end_time, flag_list):
    results = runtab.get_runs(station_list, start_time, end_time, flag_list)
    results = runtab.add_quality_flags(results)
    return results
