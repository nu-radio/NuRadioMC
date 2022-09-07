import streamlit as st
from plotly import subplots
import plotly.graph_objs as go
import pandas as pd
from NuRadioReco.detector.webinterface.utils.units_helper import str_to_unit
from NuRadioReco.utilities import units
import numpy as np
from NuRadioReco.detector.detector_mongo import det
# from NuRadioReco.detector.detector_mongo import Detector
from datetime import datetime

# det = Detector(database_connection='env_pw_user')
# det = Detector(database_connection='test')

def select_iglu(page_name, main_container, warning_container):
    col1_I, col2_I, col3_I, col4_I, col5_I = main_container.columns([1,1,1,1,1])

    selected_iglu_name = ''
    iglu_names = det.get_object_names(page_name)
    iglu_names.insert(0, f'new {page_name}')
    drab_names = det.get_object_names('drab_board')

    iglu_dropdown = col1_I.selectbox('Select existing board or enter unique name of new board:', iglu_names)
    if iglu_dropdown == f'new {page_name}':
        disable_new_input = False

        selected_iglu_infos = []
    else:
        disable_new_input = True
        selected_iglu_name = iglu_dropdown
        warning_container.warning(f'You are about to override the {page_name} unit {iglu_dropdown}!')

        # load all the information for this board
        selected_iglu_infos = det.load_board_information(page_name, selected_iglu_name, ['laser_id', 'DRAB_id', 'measurement_temp'])

    new_board_name = col2_I.text_input('', placeholder=f'new unique board name', disabled=disable_new_input)
    if iglu_dropdown == f'new {page_name}':
        selected_iglu_name = new_board_name

    # always editable (maybe want to change the laser serial number)
    if selected_iglu_infos == []:
        laser_serial_name = col3_I.text_input('', placeholder='laser serial', disabled=False)
    else:
        laser_serial_name = col3_I.text_input('', value=selected_iglu_infos[0], disabled=False)
    col3_I.markdown(laser_serial_name)

    # if an exiting IGLU is selected, change the default option to the saved DRAB
    if selected_iglu_infos != []:
        drab_index = drab_names.index(selected_iglu_infos[1])
        drab_names.pop(drab_index)
        drab_names.insert(0, selected_iglu_infos[1])
    else:
        # select golden DRAB as the default option
        golden_drab_index = drab_names.index('Golden_DRAB')
        drab_names.pop(golden_drab_index)
        drab_names.insert(0, f'Golden_DRAB')
    selected_DRAB = col4_I.selectbox('', drab_names)

    temp_list = ['room temp (20°C)', '-50°C', '-40°C', '-30°C', '-20°C', '-10°C', '0°C', '10°C', '30°C', '40°C']
    # if an exiting IGLU is selected, change the default option to the saved temperature
    if selected_iglu_infos != []:
        if selected_iglu_infos[2] == 20:
            saved_temp = 'room temp (20°C)'
        else:
            saved_temp = str(selected_iglu_infos[2]) + '°C'
        temp_index = temp_list.index(saved_temp)
        temp_list.pop(temp_index)
        temp_list.insert(0, saved_temp)
    selected_Temp = col5_I.selectbox('', temp_list)
    if 'room temp' in selected_Temp:
        selected_Temp = int(selected_Temp[len('room temp ('):-3])
    else:
        selected_Temp = int(selected_Temp[:-2])

    return selected_iglu_name, iglu_dropdown, laser_serial_name, selected_DRAB, selected_Temp


def validate_global_iglu(page_name, container_bottom, iglu_name, new_iglu_name, laser_serial_name, channel_working, Sdata_validated, uploaded_data):
    disable_insert_button = True
    name_validation = False
    laser_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if iglu_name == '':
        container_bottom.error('IGLU name is not set')
    elif iglu_name == f'new {page_name}' and (new_iglu_name is None or new_iglu_name == ''):
        container_bottom.error(f'IGLU name dropdown is set to \'new {page_name}\', but no new IGLU name was entered.')
    else:
        name_validation = True

    if laser_serial_name == '' and channel_working:
        container_bottom.error('Laser serial number is not entered.')
    else:
        laser_validation = True

    if name_validation and laser_validation:
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


def insert_iglu_to_db(page_name, s_names, iglu_name, data, input_units, working, primary, protocol, drab_id, laser_id, temp, measurement_time, time_delay):
    if not working:
        if primary and iglu_name in det.get_object_names(page_name):
            # temperature is not used for update_primary (if the board doesn't work, it will not work for every temperature)
            det.update_primary(page_name, iglu_name)
        det.set_not_working(page_name, iglu_name, primary)
    else:
        if primary and iglu_name in det.get_object_names(page_name):
            det.update_primary(page_name, iglu_name, temp)
        det.iglu_add_Sparameters(page_name, s_names, iglu_name, drab_id, laser_id, temp, data, measurement_time, primary, time_delay, protocol, input_units)
