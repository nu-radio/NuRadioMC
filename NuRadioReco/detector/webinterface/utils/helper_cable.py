import streamlit as st
from plotly import subplots
import plotly.graph_objs as go
import pandas as pd
from NuRadioReco.detector.webinterface.utils.units_helper import str_to_unit
from NuRadioReco.utilities import units
import numpy as np
from NuRadioReco.detector.detector_mongo import Detector
from datetime import datetime

det = Detector(database_connection='env_pw_user')


def select_cable(page_name, main_container, warning_container_top):
    col1_cable, col2_cable, col3_cable = main_container.columns([1,1,1])
    cable_types = []
    cable_stations = []
    cable_channels = []
    if page_name == 'surface_cable':
        cable_types=['Choose an option', '11 meter signal']
        cable_stations = ['Choose an option', 'Station 1 (11 Nanoq)', 'Station 2 (12 Terianniaq)', 'Station 3 (13 Ukaleq)', 'Station 4 (14 Tuttu)', 'Station 5 (15 Umimmak)', 'Station 6 (21 Amaroq)', 'Station 7 (22 Avinngaq)', 'Station 8 (23 Ukaliatsiaq)', 'Station 9 (24 Qappik)','Station 10 (25 Aataaq)']
        cable_channels = ['Choose an option', 'Channel 1 (0)', 'Channel 2 (1)', 'Channel 3 (2)', 'Channel 4 (3)', 'Channel 5 (4)', 'Channel 6 (5)', 'Channel 7 (6)', 'Channel 8 (7)', 'Channel 9 (8)']
    elif page_name == 'downhole_cable':
        cable_types = ['Choose an option', 'Orange (1m)', 'Blue (2m)', 'Green (3m)', 'White (4m)', 'Brown (5m)', 'Red/Grey (6m)']
        cable_stations = ['Choose an option', 'Station 11 (Nanoq)', 'Station 12 (Terianniaq)', 'Station 13 (Ukaleq)', 'Station 14 (Tuttu)', 'Station 15 (Umimmak)', 'Station 21 (Amaroq)',
                          'Station 22 (Avinngaq)', 'Station 23 (Ukaliatsiaq)', 'Station 24 (Qappik)', 'Station 25 (Aataaq)']
        cable_channels = ['Choose an option', 'A (Power String)', 'B (Helper String)', 'C (Helper String)']

    cable_type = col1_cable.selectbox('Select existing cable :', cable_types)
    cable_station = col2_cable.selectbox('', cable_stations)
    cable_channel = col3_cable.selectbox('', cable_channels)

    cable_name = ""
    if page_name == 'surfCABLE':
        cable_name = cable_station[cable_station.find('(')+1:cable_station.find('(')+3] + cable_channel[cable_channel.find('(')+1:cable_channel.rfind(')')] + cable_type[:cable_type.find(' meter')]
    elif page_name == 'CABLE':
        cable_name = cable_station[len('stations'): len('stations') + 2] + cable_channel[:1] + cable_type[cable_type.find('(')+1:cable_type.find(')')-1]

    if cable_name in det.get_object_names(page_name):
        if page_name == 'surfCABLE':
            warning_container_top.warning(f'You are about to override the {page_name} unit \'{cable_name[-2:]} meter, station {cable_name[:2]}, channel {cable_name[2:-2]}\'!')
        elif page_name == 'CABLE':
            warning_container_top.warning(f'You are about to override the {page_name} unit \'{cable_name[-1:]} meter, station {cable_name[:2]}, string {cable_name[2:-1]}\'!')


    return cable_type, cable_station, cable_channel, cable_name


def validate_global_cable(container_bottom, cable_type, cable_sta, cable_cha, channel_working, Sdata_validated_magnitude, Sdata_validated_phase, uploaded_data_magnitude, uploaded_data_phase):
    disable_insert_button = True
    name_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if cable_type == 'Choose an option' or cable_sta == 'Choose an option' or cable_cha == 'Choose an option':
        container_bottom.error('Not all cable options are selected')
        name_validation = False
    else:
        name_validation = True

    if name_validation:
        if not Sdata_validated_magnitude and uploaded_data_magnitude is not None:
            container_bottom.error('There is a problem with the magnitude input data')
            disable_insert_button = True

        if not Sdata_validated_phase and uploaded_data_phase is not None:
            container_bottom.error('There is a problem with the phase input data')
            disable_insert_button = True

        if Sdata_validated_magnitude and Sdata_validated_phase:
            disable_insert_button = False
            container_bottom.success('All inputs validated')

        if not channel_working:
            container_bottom.warning('The channel is set to not working')
            disable_insert_button = False
            container_bottom.success('All inputs validated')
    else:
        disable_insert_button = True

    return disable_insert_button


def insert_cable_to_db(page_name, s_name, cable_name, data_m, data_p, input_units, working, primary, protocol):
    if not working:
        if primary and cable_name in det.get_object_names(page_name):
            det.update_primary(page_name, cable_name)
        det.set_not_working(page_name, cable_name, primary)
    else:
        if primary and cable_name in det.get_object_names(page_name):
            det.update_primary(page_name, cable_name)
        det.cable_add_Sparameter(page_name, cable_name, [s_name], data_m, data_p, input_units, primary, protocol)
