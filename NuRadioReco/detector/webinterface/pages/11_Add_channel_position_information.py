import copy
import time
import streamlit as st
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go
import numpy as np
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page
from NuRadioReco.detector.webinterface.utils.helper_station import insert_channel_position_to_db
from NuRadioReco.utilities import units
from datetime import datetime
from datetime import time

page_name = 'station'
collection_name = 'station_rnog'


def insert_channel_info(warning_cont, info_cont, selected_station_id, selected_channel, measurement_name, measurement_time, position, orientation, rotation, primary):
    if st.session_state.insert_channel:
        # TODO:
        # station_info = load_station_infos(selected_station_id, collection_name)
        station_info = {}
        if station_info != {}:
            channel_info = station_info[selected_station_id]['channels']
            if selected_channel in channel_info.keys():
                warning_cont.warning('YOU ARE ABOUT TO CHANGE AN EXISTING CHANNEL!')
                warning_cont.markdown('Do you really want to change an existing channel?')
                col_b1, col_b2, phold = warning_cont.columns([0.2, 0.2, 1.6])
                yes_button = col_b1.button('YES')
                no_button = col_b2.button('NO')
                if yes_button:
                    insert_channel_position_to_db(selected_station_id, selected_channel, measurement_name, measurement_time, position, orientation, rotation, primary)
                    st.session_state.insert_channel = False
                    st.session_state.channel_success = True
                    st.experimental_rerun()

                if no_button:
                    st.session_state.insert_channel = False
                    st.session_state.channel_success = False
                    st.experimental_rerun()
        else:
            # information will be inserted into the database, without requiring any action
            insert_channel_position_to_db(selected_station_id, selected_channel, measurement_name, measurement_time, position, orientation, rotation, primary)
            st.session_state.insert_channel = False
            st.session_state.channel_success = True

    if st.session_state.channel_success:
        info_cont.success('Channel successfully added to the database!')
        st.session_state.channel_success = False

        # if there is no corresponding station in the database -> the insert button is disabled (see validate_channel_inputs())


def validate_inputs(container_bottom, station_name, measurement_name, channel):
    channel_correct = False
    station_in_db = False
    measurement_name_correct = False

    disable_insert_button = True

    # validate that a valid channel is given
    possible_channel_ids = np.arange(0,24,1)
    if channel != '':
        if channel in possible_channel_ids:
            channel_correct = True
        else:
            container_bottom.error('The channel number must be between 0 and 23.')

    # validate that a valid measurement name is given
    if measurement_name != '' and measurement_name != 'new measurement':
        measurement_name_correct = True
    else:
        container_bottom.error('Measurement name is not valid.')

    # TODO:
    # # check if there is an entry for the station in the db
    # if station_name in det.get_object_names(collection):
    #     station_in_db = True
    # else:
    #     container_bottom.error('There is no corresponding entry for the station in the database.')
    station_in_db = True
    if channel_correct and station_in_db and measurement_name_correct:
        disable_insert_button = False

    return disable_insert_button

def build_main_page(main_cont):
    main_cont.title('Add channel position and orientation information')
    main_cont.markdown(page_name)

    if 'insert_channel' not in st.session_state:
        st.session_state.insert_channel = False

    # select a unique name for the measurement (survey_01, tape_measurement, ...)
    col1_name, col2_name = main_cont.columns([1, 1])
    # TODO load the list of names
    selected_name = col1_name.selectbox('Select or enter a unique name for the measurement:', ['new measurement'])
    disabled_name_input = True
    if selected_name == 'new measurement':
        disabled_name_input = False
    name_input = col2_name.text_input('Select or enter a unique name for the measurement:', disabled=disabled_name_input, label_visibility='hidden')
    measurement_name = selected_name
    if measurement_name == 'new measurement':
        measurement_name = name_input

    # enter the information for the single stations
    cont_warning_top = main_cont.container()
    station_list = ['Station 11 (Nanoq)', 'Station 12 (Terianniaq)', 'Station 13 (Ukaleq)', 'Station 14 (Tuttu)', 'Station 15 (Umimmak)', 'Station 21 (Amaroq)', 'Station 22 (Avinngaq)',
                    'Station 23 (Ukaliatsiaq)', 'Station 24 (Qappik)', 'Station 25 (Aataaq)']
    selected_station = main_cont.selectbox('Select a station', station_list)
    # get the name and id out of the string
    selected_station_name = selected_station[selected_station.find('(') + 1:-1]
    selected_station_id = int(selected_station[len('Station '):len('Station ') + 2])


    # page to enter the station information
    cont_warning_top = main_cont.container()

    main_cont.subheader('Input channel information')
    cont_channel = main_cont.container()

    # TODO load the station information
    # station_info = load_station_infos(station_id, coll_name)

    # TODO
    # # load all channels which are already in the database
    # if station_info != {}:
    #     channels_db = list(station_info[station_id]['channels'].keys())
    # else:
    #     channels_db = []
    # cont.info(f'Channels included in the database: {len(channels_db)}/24')

    # if not all channels are in the db, add the possibility to add another channel number
    channels_db  = [] # TODO: load this from the database
    channel_help = channels_db
    if len(channels_db) < 24:
        channel_help.insert(0, 'new channel number')
        disable_new_entry = False
    else:
        disable_new_entry = True
    col1_cha, col2_cha = main_cont.columns([1,1])
    channel = col1_cha.selectbox('Select a channel or enter a new channel number:', channel_help, help='The channel number must be an integer between 0 and 23.')
    new_cha = col2_cha.text_input('', placeholder='channel number', disabled=disable_new_entry)
    if channel == 'new channel number':
        if new_cha == '':
            selected_channel = -10
        else:
            selected_channel = int(new_cha)
    else:
        selected_channel = int(channel)

    # # if the channel already exist in the database, the channel info will be loaded
    # if channel == 'new channel number':
    #     channel_info = {}
    # else:
    #     channel_info = station_info[station_id]['channels'][selected_channel]
    #
    # # tranform the channel number from a string into an int
    # if selected_channel != '':
    #     selected_channel = int(selected_channel)

    # primary measurement?
    primary = main_cont.checkbox('Is this the primary measurement?', value=True)

    # measurement time
    measurement_time = main_cont.date_input('Enter the time of the measurement:', value=datetime.utcnow(), help='The date when the measurement was conducted.')
    measurement_time = datetime.combine(measurement_time, time(0, 0, 0))

    # insert the position of the antenna; if the channel already exist enter the position of the existing channel
    col1_pos, col2_pos, col3_pos = main_cont.columns([1, 1, 1])
    # TODO load the already existing information
    db_ant_position = [0, 0, 0]

    x_pos_ant = col1_pos.text_input('x position', key='x_antenna', value=db_ant_position[0])
    y_pos_ant = col2_pos.text_input('y position', key='y_antenna', value=db_ant_position[1])
    z_pos_ant = col3_pos.text_input('z position', key='z_antenna', value=db_ant_position[2])
    position = [x_pos_ant, y_pos_ant, z_pos_ant]

    # input the orientation, rotation of the antenna; if the channel already exist, insert the values from the database
    col1a, col2a, col3a, col4a = main_cont.columns([1, 1, 1, 1])
    # TODO load the already existing information
    db_ori_rot = [0, 0, 0, 0]
    ant_ori_theta = col1a.text_input('orientation (theta):', value=db_ori_rot[0])
    ant_ori_phi = col2a.text_input('orientation (phi):', value=db_ori_rot[1])
    ant_rot_theta = col3a.text_input('rotation (theta):', value=db_ori_rot[2])
    ant_rot_phi = col4a.text_input('rotation (phi):', value=db_ori_rot[3])
    ori = {'theta': ant_ori_theta, 'phi': ant_ori_phi}
    rot = {'theta': ant_rot_theta, 'phi': ant_rot_phi}

    # container for warnings/infos at the botton
    cont_warning_bottom = main_cont.container()

    disable_insert_button = validate_inputs(cont_warning_bottom, selected_station_name, measurement_name, selected_channel)

    insert_channel = main_cont.button('INSERT CHANNEL TO DB', disabled=disable_insert_button)

    cont_channel_warning = main_cont.container()

    if insert_channel:
        st.session_state.insert_channel = True
    print(st.session_state.insert_channel)
    insert_channel_info(cont_channel_warning, cont_warning_bottom, selected_station_id, selected_channel, measurement_name, measurement_time, position, ori, rot, primary)

# main page setup
page_configuration()

if 'station_success' not in st.session_state:
    st.session_state['station_success'] = False

if 'channel_success' not in st.session_state:
    st.session_state['channel_success'] = False

# initialize the session key (will be used to display different pages depending on the button clicked)
if 'key' not in st.session_state:
    st.session_state['key'] = '0'

main_container = st.container()  # container for the main part of the page (with all the input filed)
success_container = st.container()  # container to display the page when the data was submitted

if st.session_state.key == '0':
    build_main_page(main_container)  # after clicking the submit button, the session key is set to '1' and the page is rerun

if st.session_state.key == '1':
    build_success_page(success_container, page_name)  # after clicking the 'add another measurement' button, the session key is set to '0' and the page is rerun
