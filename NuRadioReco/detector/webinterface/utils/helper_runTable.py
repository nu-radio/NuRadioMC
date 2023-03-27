from NuRadioReco.detector.db_mongo_write import Database
from NuRadioReco.detector.webinterface import config

db = Database(database_connection=config.DATABASE_TARGET)


def get_station_ids_from_db():
    return db.get_quantity_names('runtable', 'station')


def get_firmware_from_db():
    return db.get_quantity_names('runtable', 'firmware_version')


def load_runs(station_list, start_time, end_time, flag_list, trigger_list, min_duration, firmware_list):
    results = db.get_runs(station_list, start_time, end_time, flag_list, trigger_list, min_duration, firmware_list)
    return results
