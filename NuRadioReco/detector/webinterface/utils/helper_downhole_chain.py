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


def select_downhole(page_name, main_container, warning_container):
    col1_I, col2_I, col3_I, col4_I, col5_I, col6_I, col7_I = main_container.columns([1.35,1.1,0.9,0.9,1,1,0.75])

    selected_downhole_name = ''
    downhole_names = det.get_object_names(page_name)
    downhole_names.insert(0, f'new fiber')

    downhole_dropdown = col1_I.selectbox('Select existing fiber or enter unique name of new fiber:', downhole_names)
    if downhole_dropdown == f'new fiber':
        disable_new_input = False

        selected_downhole_infos = []
    else:
        disable_new_input = True
        selected_downhole_name = downhole_dropdown
        warning_container.warning(f'You are about to override the fiber unit {downhole_dropdown}!')

        # load all the information for this board
        selected_downhole_infos = det.load_board_information(page_name, selected_downhole_name, ['breakout', 'breakout_channel', 'IGLU_id', 'DRAB_id', 'measurement_temp'])

    new_board_name = col2_I.text_input('', placeholder=f'new unique fiber name', disabled=disable_new_input)
    if downhole_dropdown == f'new fiber':
        selected_downhole_name = new_board_name

    breakout_ids = ['breakout-id', '1', '2', '3']
    # if an exiting fiber is selected, change the default option to the saved number
    if selected_downhole_infos != []:
        breakout_index = breakout_ids.index(str(selected_downhole_infos[0]))
        breakout_ids.pop(breakout_index)
        breakout_ids.insert(0, str(selected_downhole_infos[0]))
    selected_breakout_id = col3_I.selectbox('', breakout_ids)

    breakout_cha_ids = ['breakout channel-id', 'p1', 'p2', 'p3', 's1', 's2', 's3']
    # if an exiting fiber is selected, change the default option to the saved number
    if selected_downhole_infos != []:
        breakout_cha_index = breakout_cha_ids.index(str(selected_downhole_infos[1]))
        breakout_cha_ids.pop(breakout_cha_index)
        breakout_cha_ids.insert(0, str(selected_downhole_infos[1]))
    selected_breakout_cha_id = col4_I.selectbox('', breakout_cha_ids)

    # if an exiting fiber is selected, change the default option to the saved IGLU
    iglu_names = det.get_object_names('iglu_board')
    if selected_downhole_infos != []:
        iglu_index = iglu_names.index(selected_downhole_infos[2])
        iglu_names.pop(iglu_index)
        iglu_names.insert(0, selected_downhole_infos[2])
    else:
        # select golden IGLU as the default option
        golden_iglu_index = iglu_names.index('Golden_IGLU')
        iglu_names.pop(golden_iglu_index)
        iglu_names.insert(0, f'Golden_IGLU')
    selected_IGLU = col5_I.selectbox('', iglu_names)

    # if an exiting fiber is selected, change the default option to the saved DRAB
    drab_names = det.get_object_names('drab_board')
    if selected_downhole_infos != []:
        drab_index = drab_names.index(selected_downhole_infos[3])
        drab_names.pop(drab_index)
        drab_names.insert(0, selected_downhole_infos[3])
    else:
        # select golden DRAB as the default option
        golden_drab_index = drab_names.index('Golden_DRAB')
        drab_names.pop(golden_drab_index)
        drab_names.insert(0, f'Golden_DRAB')
    selected_DRAB = col6_I.selectbox('', drab_names)

    temp_list = ['room temp (20°C)', '-50°C', '-40°C', '-30°C', '-20°C', '-10°C', '0°C', '10°C', '30°C', '40°C']
    # if an exiting fiber is selected, change the default option to the saved temperature
    if selected_downhole_infos != []:
        if selected_downhole_infos[4] == 20:
            saved_temp = 'room temp (20°C)'
        else:
            saved_temp = str(selected_downhole_infos[4]) + '°C'
        temp_index = temp_list.index(saved_temp)
        temp_list.pop(temp_index)
        temp_list.insert(0, saved_temp)
    selected_Temp = col7_I.selectbox('', temp_list)
    if 'room temp' in selected_Temp:
        selected_Temp = int(selected_Temp[len('room temp ('):-3])
    else:
        selected_Temp = int(selected_Temp[:-2])

    return selected_downhole_name, downhole_dropdown, selected_breakout_id, selected_breakout_cha_id, selected_IGLU, selected_DRAB, selected_Temp


def validate_global_downhole(page_name, container_bottom, surface_name, new_surface_name, breakout_id, breakout_cha_id, channel_working, Sdata_validated, uploaded_data):
    disable_insert_button = True
    name_validation = False
    input_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if surface_name == '':
        container_bottom.error(f'fiber name is not set')
    elif surface_name == f'new fiber' and (new_surface_name is None or new_surface_name == ''):
        container_bottom.error(f'fiber name dropdown is set to \'new fiber\', but no new fiber name was entered.')
    else:
        name_validation = True

    if ('breakout' in breakout_id and channel_working) or ('breakout' in breakout_cha_id and channel_working):
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


def insert_downhole_to_db(page_name, s_names, downhole_name, data, input_units, working, primary, protocol, breakout_id, breakout_cha_id, iglu_id, drab_id, temp, measurement_time, time_delay):
    if not working:
        if primary and downhole_name in det.get_object_names(page_name):
            det.update_primary(page_name, downhole_name)
        det.set_not_working(page_name, downhole_name, primary)
    else:
        if primary and downhole_name in det.get_object_names(page_name):
            det.update_primary(page_name, downhole_name, temp)
        det.downhole_add_Sparameters(page_name, s_names, downhole_name, int(breakout_id), breakout_cha_id, iglu_id, drab_id, temp, data, measurement_time, primary, time_delay, protocol, input_units)
