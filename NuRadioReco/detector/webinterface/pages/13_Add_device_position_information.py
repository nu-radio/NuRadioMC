import copy
import time
import streamlit as st
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go
import numpy as np
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page
from NuRadioReco.detector.webinterface.utils.helper_station import insert_channel_position_to_db, load_measurement_names, load_general_station_infos, load_current_primary_device_information, insert_device_position_to_db
from NuRadioReco.utilities import units
from datetime import datetime
from datetime import time

page_name = 'station'
collection_name = 'station_rnog'


def insert_device_pos_info(warning_cont, info_cont, selected_station_id, selected_device_id, measurement_name, measurement_time, position, orientation, rotation, primary):
    if st.session_state.insert_device:
        device_info = load_current_primary_device_information(selected_station_id, selected_device_id)
        device_pos_info = device_info['devices'][selected_device_id]['device_position']
        if device_pos_info != {}:
            warning_cont.warning('YOU ARE ABOUT TO CHANGE AN EXISTING DEVICE!')
            warning_cont.markdown('Do you really want to change an existing device?')
            col_b1, col_b2, phold = warning_cont.columns([0.2, 0.2, 1.6])
            yes_button = col_b1.button('YES')
            no_button = col_b2.button('NO')
            if yes_button:
                insert_device_position_to_db(selected_station_id, selected_device_id, measurement_name, measurement_time, position, orientation, rotation, primary)
                st.session_state.insert_device = False
                st.session_state.device_success = True
                st.experimental_rerun()

            if no_button:
                st.session_state.insert_device = False
                st.session_state.device_success = False
                st.experimental_rerun()
        else:
            # information will be inserted into the database, without requiring any action
            insert_device_position_to_db(selected_station_id, selected_device_id, measurement_name, measurement_time, position, orientation, rotation, primary)
            st.session_state.insert_device = False
            st.session_state.device_success = True

    if st.session_state.device_success:
        info_cont.success('Device successfully added to the database!')
        st.session_state.device_success = False

        # if there is no corresponding station in the database -> the insert button is disabled (see validate_channel_inputs())


def validate_inputs(container_bottom, station_info, measurement_name, device_id):
    device_id_correct = False
    station_in_db = False
    measurement_name_correct = False

    disable_insert_button = True

    # validate that a the device id is correct
    if device_id >= 0:
        device_id_correct = True
    else:
        container_bottom.error('No device ID is chosen.')

    # validate that a valid measurement name is given
    if measurement_name != '' and measurement_name != 'new measurement':
        measurement_name_correct = True
    else:
        container_bottom.error('Measurement name is not valid.')

    if station_info != {}:
        station_in_db = True
    else:
        container_bottom.error('The selected station is not in the database.')

    if device_id_correct and station_in_db and measurement_name_correct:
        disable_insert_button = False

    return disable_insert_button


def build_main_page(main_cont):
    main_cont.title('Add device position and orientation information')
    main_cont.markdown(page_name)

    if 'insert_device' not in st.session_state:
        st.session_state.insert_device = False

    # select a unique name for the measurement (survey_01, tape_measurement, ...)
    col1_name, col2_name = main_cont.columns([1, 1])
    measurement_list = load_measurement_names('device_position')
    measurement_list.insert(0, 'new measurement')
    selected_name = col1_name.selectbox('Select or enter a unique name for the measurement:', measurement_list)
    disabled_name_input = True
    if selected_name == 'new measurement':
        disabled_name_input = False
    name_input = col2_name.text_input('Select or enter a unique name for the measurement:', disabled=disabled_name_input, label_visibility='hidden')
    measurement_name = selected_name
    if measurement_name == 'new measurement':
        measurement_name = name_input

    # enter the information for the station
    cont_warning_top = main_cont.container()
    station_list = ['Station 11 (Nanoq)', 'Station 12 (Terianniaq)', 'Station 13 (Ukaleq)', 'Station 14 (Tuttu)', 'Station 15 (Umimmak)', 'Station 21 (Amaroq)', 'Station 22 (Avinngaq)',
                    'Station 23 (Ukaliatsiaq)', 'Station 24 (Qappik)', 'Station 25 (Aataaq)']
    selected_station = main_cont.selectbox('Select a station', station_list)
    # get the name and id out of the string
    selected_station_name = selected_station[selected_station.find('(') + 1:-1]
    selected_station_id = int(selected_station[len('Station '):len('Station ') + 2])

    cont_warning_top = main_cont.container()

    main_cont.subheader('Input device information')
    cont_device = main_cont.container()

    # load the general station information
    station_info = load_general_station_infos(selected_station_id, 'station_rnog')

    # load all devices which are already in the database
    if station_info != {}:
        device_db = []
        for key in station_info[selected_station_id]['devices'].keys():
            device_db.append(station_info[selected_station_id]['devices'][key]['id'])
    else:
        device_db = []
        cont_device.error('There are no general information about devices saved in this database.')

    device_db.insert(0, 'Select a device id')

    device_id = cont_device.selectbox('Select a device id:', device_db)
    if device_id == 'Select a device id':
        selected_device_id = -10
    else:
        selected_device_id = int(device_id)

    # primary measurement?
    primary = main_cont.checkbox('Is this the primary measurement?', value=True)

    # measurement time
    measurement_time = main_cont.date_input('Enter the time of the measurement:', value=datetime.utcnow(), help='The date when the measurement was conducted.')
    measurement_time = datetime.combine(measurement_time, time(0, 0, 0))

    position = None
    rot = None
    ori = None
    if selected_device_id >= 0:
        # insert the position of the antenna; if the channel already exist enter the position of the existing channel
        device_info = load_current_primary_device_information(selected_station_id, selected_device_id)
        device_pos_info = device_info['devices'][selected_device_id]['device_position']

        if device_info['devices'][selected_device_id]['device_name'] in ['Solar panel 1', 'Solar panel 1', 'wind-turbine', 'daq box']:
            if device_pos_info == {}:
                db_dev_pos1 = [0, 0, 0]
                db_dev_pos2 = [0, 0, 0]
            else:
                db_dev_pos1 = device_pos_info['position'][0]
                db_dev_pos2 = device_pos_info['position'][1]

            main_cont.markdown('First measurement point:')
            col1_pos, col2_pos, col3_pos = main_cont.columns([1, 1, 1])
            x_pos_dev1 = col1_pos.text_input('x position', key='x_dev1', value=db_dev_pos1[0])
            y_pos_dev1 = col2_pos.text_input('y position', key='y_dev1', value=db_dev_pos1[1])
            z_pos_dev1 = col3_pos.text_input('z position', key='z_dev1', value=db_dev_pos1[2])
            position1 = [x_pos_dev1, y_pos_dev1, z_pos_dev1]

            main_cont.markdown('Second measurement point:')
            col11_pos, col21_pos, col31_pos = main_cont.columns([1, 1, 1])
            x_pos_dev2 = col11_pos.text_input('x position', key='x_dev2', value=db_dev_pos2[0])
            y_pos_dev2 = col21_pos.text_input('y position', key='y_dev2', value=db_dev_pos2[1])
            z_pos_dev2 = col31_pos.text_input('z position', key='z_dev2', value=db_dev_pos2[2])
            position2 = [x_pos_dev2, y_pos_dev2, z_pos_dev2]
            position = [position1, position2]
        else:
            if device_pos_info == {}:
                db_dev_pos = [0, 0, 0]
                db_rot = {'theta': 0, 'phi': 0}
                db_ori = {'theta': 0, 'phi': 0}
            else:
                db_dev_pos = device_pos_info['position']
                db_rot = device_pos_info['rotation']
                db_ori = device_pos_info['orientation']

            col1_pos, col2_pos, col3_pos = main_cont.columns([1, 1, 1])
            x_pos_ant = col1_pos.text_input('x position', key='x_antenna', value=db_dev_pos[0])
            y_pos_ant = col2_pos.text_input('y position', key='y_antenna', value=db_dev_pos[1])
            z_pos_ant = col3_pos.text_input('z position', key='z_antenna', value=db_dev_pos[2])
            position = [x_pos_ant, y_pos_ant, z_pos_ant]

            # input the orientation, rotation of the antenna; if the channel already exist, insert the values from the database
            col1a, col2a, col3a, col4a = main_cont.columns([1, 1, 1, 1])
            ant_ori_theta = col1a.text_input('orientation (theta):', value=db_ori['theta'])
            ant_ori_phi = col2a.text_input('orientation (phi):', value=db_ori['phi'])
            ant_rot_theta = col3a.text_input('rotation (theta):', value=db_rot['theta'])
            ant_rot_phi = col4a.text_input('rotation (phi):', value=db_rot['phi'])
            ori = {'theta': ant_ori_theta, 'phi': ant_ori_phi}
            rot = {'theta': ant_rot_theta, 'phi': ant_rot_phi}

    # container for warnings/infos at the botton
    cont_warning_bottom = main_cont.container()

    disable_insert_button = validate_inputs(cont_warning_bottom, station_info, measurement_name, selected_device_id)

    insert_device = main_cont.button('INSERT DEVICE TO DB', disabled=disable_insert_button)

    cont_device_warning = main_cont.container()

    if insert_device:
        st.session_state.insert_device = True
    insert_device_pos_info(cont_device_warning, cont_warning_bottom, selected_station_id, selected_device_id, measurement_name, measurement_time, position, ori, rot, primary)

# main page setup
page_configuration()

if 'station_success' not in st.session_state:
    st.session_state['station_success'] = False

if 'device_success' not in st.session_state:
    st.session_state['device_success'] = False

# initialize the session key (will be used to display different pages depending on the button clicked)
if 'key' not in st.session_state:
    st.session_state['key'] = '0'

main_container = st.container()  # container for the main part of the page (with all the input filed)
success_container = st.container()  # container to display the page when the data was submitted

if st.session_state.key == '0':
    build_main_page(main_container)  # after clicking the submit button, the session key is set to '1' and the page is rerun

if st.session_state.key == '1':
    build_success_page(success_container, page_name)  # after clicking the 'add another measurement' button, the session key is set to '0' and the page is rerun
