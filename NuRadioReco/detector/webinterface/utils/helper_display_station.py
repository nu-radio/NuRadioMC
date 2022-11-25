import streamlit as st
from plotly import subplots
import plotly.graph_objs as go
import pandas as pd
from NuRadioReco.detector.webinterface.utils.units_helper import str_to_unit
from NuRadioReco.utilities import units
import numpy as np
from NuRadioReco.detector.detector_mongo import Detector
from NuRadioReco.detector.webinterface import config
from datetime import datetime
from datetime import time

det = Detector(config.DATABASE_TARGET)
# det = Detector(database_connection='test')


# def load_station_infos(station_id, coll_name):
#     det.update(datetime.now(), coll_name)
#     station_info = det.get_station_information(coll_name, station_id)
#
#     return station_info


def build_station_selection(cont, collection_name):
    # selection of the collection from which the station will be displayed
    # selection of the station which will be displayed
    station_names = det.get_object_names(collection_name)
    station_ids = det.get_station_ids(collection_name)
    station_list = []
    for sta_id, name in zip(station_ids, station_names):
        station_list.append(f'station {sta_id} ({name})')
    selected_station = cont.selectbox('Select a station:', station_list)

    selected_station_name = selected_station[selected_station.find('(')+1:-1]
    selected_station_id = int(selected_station[len('station '):selected_station.find('(')-1])

    return selected_station_name, selected_station_id


def load_station_position_info(station_id, primary_time, measurement_name):
    if measurement_name == 'not specified':
        measurement_name = None
    return det.get_collection_information('station_position', station_id, primary_time, measurement_name)


def load_channel_position_info(station_id, primary_time, measurement_name):
    if measurement_name == 'not specified':
        measurement_name = None
    return det.get_collection_information('channel_position', station_id, primary_time, measurement_name)


def load_signal_chain_information(station_id, primary_time, config_name):
    if config_name == 'not specified':
        return det.get_collection_information('signal_chain', station_id, primary_time, measurement_name=None)
    elif config_name == 'built-in':
        pass
    else:
        return det.get_collection_information('signal_chain', station_id, primary_time, config_name)


def load_general_info(station_id):
    collection_name = 'station_rnog'
    det.update(datetime.now(), collection_name)
    return det.get_general_station_information(collection_name, station_id)


def get_all_station_measurement_names():
    return det.get_quantity_names('station_position', 'measurements.measurement_name')


def get_all_channel_measurements_names():
    return det.get_quantity_names('channel_position', 'measurements.measurement_name')


def get_all_signal_chain_config_names():
    return det.get_quantity_names('signal_chain', 'measurements.measurement_name')


def build_channel_selection(cont, channel_dic):
    # selection of the channel which will be displayed
    error_cont = cont.container()
    if channel_dic == {}:
        error_cont.warning('No channels in the database.')
        channel_list = ['']
    else:
        channel_list = list(channel_dic.keys())
    selected_channel = cont.selectbox('Select a channel:', channel_list)

    return selected_channel
