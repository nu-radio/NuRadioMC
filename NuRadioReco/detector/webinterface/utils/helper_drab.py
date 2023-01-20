import streamlit as st
from plotly import subplots
import plotly.graph_objs as go
import pandas as pd
from NuRadioReco.detector.webinterface.utils.units_helper import str_to_unit
from NuRadioReco.utilities import units
import numpy as np
# from NuRadioReco.detector.detector_mongo import det
from NuRadioReco.detector.detector_mongo import Detector
from NuRadioReco.detector.webinterface import config
# from NuRadioReco.detector.detector_mongo import Detector
from datetime import datetime

# det = Detector(config.DATABASE_TARGET)
# det = Detector(database_connection='env_pw_user')
# det = Detector(database_connection='test')
det = Detector(database_connection=config.DATABASE_TARGET)


def select_drab(page_name, main_container, warning_container):
    col1_I, col2_I, col3_I, col4_I, col5_I, col6_I = main_container.columns([1.2,1,1,0.8,1,1])

    selected_drab_name = ''
    drab_names = det.get_object_names(page_name)
    drab_names.insert(0, f'new {page_name}')
    iglu_names = det.get_object_names('iglu_board')

    drab_dropdown = col1_I.selectbox('Select existing board or enter unique name of new board:', drab_names)
    if drab_dropdown == f'new {page_name}':
        disable_new_input = False

        selected_drab_infos = []
    else:
        disable_new_input = True
        selected_drab_name = drab_dropdown
        warning_container.warning(f'You are about to override the {page_name} unit {drab_dropdown}!')

        # load all the information for this board
        selected_drab_infos = det.load_board_information(page_name, selected_drab_name, ['photodiode_serial', 'channel_id', 'IGLU_id', 'measurement_temp'])

    new_board_name = col2_I.text_input('', placeholder=f'new unique board name', disabled=disable_new_input)
    if drab_dropdown == f'new {page_name}':
        selected_drab_name = new_board_name

    # always editable (maybe want to change the photodiode serial number)
    if selected_drab_infos == []:
        photodiode_number = col3_I.text_input('', placeholder='photodiode serial', disabled=False)
    else:
        photodiode_number = col3_I.text_input('', value=selected_drab_infos[0], disabled=False)
    col3_I.markdown(photodiode_number)

    channel_numbers = ['Choose a channel-id', '0', '1', '2', '3']
    # if an exiting drab is selected, change the default option to the saved IGLU
    if selected_drab_infos != []:
        cha_index = channel_numbers.index(str(selected_drab_infos[1]))
        channel_numbers.pop(cha_index)
        channel_numbers.insert(0, str(selected_drab_infos[1]))
    selected_channel_id = col4_I.selectbox('', channel_numbers)

    # if an exiting drab is selected, change the default option to the saved IGLU
    if selected_drab_infos != []:
        if selected_drab_infos[2] in iglu_names:
            iglu_index = iglu_names.index(selected_drab_infos[2])
            iglu_names.pop(iglu_index)
        iglu_names.insert(0, selected_drab_infos[2])
    else:
        # select golden IGLU as the default option
        if 'Golden_IGLU' in iglu_names:
            golden_iglu_index = iglu_names.index('Golden_IGLU')
            iglu_names.pop(golden_iglu_index)
        iglu_names.insert(0, f'Golden_IGLU')
    selected_IGLU = col5_I.selectbox('', iglu_names)

    temp_list = ['room temp (20°C)', '-50°C', '-40°C', '-30°C', '-20°C', '-10°C', '0°C', '10°C', '30°C', '40°C']
    # if an exiting DRAB is selected, change the default option to the saved temperature
    if selected_drab_infos != []:
        if selected_drab_infos[3] == 20:
            saved_temp = 'room temp (20°C)'
        else:
            saved_temp = str(selected_drab_infos[3]) + '°C'
        temp_index = temp_list.index(saved_temp)
        temp_list.pop(temp_index)
        temp_list.insert(0, saved_temp)
    selected_Temp = col6_I.selectbox('', temp_list)
    if 'room temp' in selected_Temp:
        selected_Temp = int(selected_Temp[len('room temp ('):-3])
    else:
        selected_Temp = int(selected_Temp[:-2])

    return selected_drab_name, drab_dropdown, photodiode_number, selected_channel_id,selected_IGLU, selected_Temp


def validate_global_drab(page_name, container_bottom, drab_name, new_drab_name, photodiode_number, channel_id, channel_working, Sdata_validated, uploaded_data):
    disable_insert_button = True
    name_validation = False
    input_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if drab_name == '':
        container_bottom.error('DRAB name is not set')
    elif drab_name == f'new {page_name}' and (new_drab_name is None or new_drab_name == ''):
        container_bottom.error(f'DRAB name dropdown is set to \'new {page_name}\', but no new DRAB name was entered.')
    else:
        name_validation = True

    if (photodiode_number == '' and channel_working) or ('Choose' in channel_id and channel_working):
        container_bottom.error('Not all input options are entered.')
    else:
        input_validation = True

    if name_validation and input_validation:
        if not Sdata_validated and uploaded_data is not None:
            container_bottom.error('There is a problem with the input data')
            disable_insert_button = True
        elif Sdata_validated:
            disable_insert_button = False
            container_bottom.success('Input fields are validated')

        if not channel_working:
            container_bottom.warning('The channel is set to not working')
            disable_insert_button = False
            container_bottom.success('Input fields are validated')
    else:
        disable_insert_button = True

    return disable_insert_button


def insert_drab_to_db(page_name, s_names, drab_name, data, input_units, working, primary, protocol, iglu_id, photodiode_id, channel_id, temp, measurement_time, time_delay):
    if not working:
        det.set_not_working(page_name, drab_name, primary, channel_id=int(channel_id))
    else:
        det.drab_add_Sparameters(page_name, s_names, drab_name, iglu_id, photodiode_id, int(channel_id), temp, data, measurement_time, primary, time_delay, protocol, input_units)
