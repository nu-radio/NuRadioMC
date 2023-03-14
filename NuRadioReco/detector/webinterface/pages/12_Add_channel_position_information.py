import copy
import time
import streamlit as st
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go
import numpy as np
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page, read_position_data
from NuRadioReco.detector.webinterface.utils.helper_station import insert_channel_position_to_db, load_measurement_names, load_general_station_infos, load_measurement_station_information, load_station_ids
from NuRadioReco.utilities import units
from datetime import datetime
from datetime import time

page_name = 'station'
collection_name = 'station_rnog'


def validate_inputs(container_bottom, station_id, selected_measurement_name, channel, channels_db, orientation, rotation):
    channel_correct = False
    station_in_db = False
    channel_in_db = False
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
    if selected_measurement_name != '' and selected_measurement_name != 'new measurement':
        measurement_name_correct = True
    else:
        container_bottom.error('Measurement name is not valid.')

    # check if there is an entry for the station in the db
    if station_id in load_station_ids():
        station_in_db = True
    else:
        container_bottom.error('There is no corresponding entry for the station in the database. Please insert the general station information first!')

    # check if the channel already exits in the db for this measurement
    if channel in channels_db:
        container_bottom.error('There is already a channel position information saved in the database for this measurement name and this channel id.')
    else:
        channel_in_db = True

    # check that rotation and orientation are realistic values (0,360°)
    check_rot_ori = False
    check_value_rot = 0
    check_value_ori = 0
    if not 0 <= rotation['theta'] <= 360 or not 0 <= rotation['phi'] <= 360:
        check_value_rot = 1
    if not 0 <= orientation['theta'] <= 360 or not 0 <= orientation['phi'] <= 360:
        check_value_ori = 1

    if check_value_rot == 1:
        container_bottom.error('The rotation values are invalid. Please give values between 0° and 360°.')
    elif check_value_ori == 1:
        container_bottom.error('The orientation values are invalid. Please give values between 0° and 360°.')
    else:
        check_rot_ori = True

    if channel_correct and station_in_db and measurement_name_correct and channel_in_db and check_rot_ori:
        disable_insert_button = False

    return disable_insert_button


def validate_csv_inputs(container_bottom, selected_measurement_name, input_data, channel_id_in_db):
    disable_input_button = True

    # check if a file is given
    check_file = False
    if input_data != {}:
        check_file = True
    else:
        container_bottom.error('No position data file selected.')

    # check if a measurement name is given
    check_measurement_name = False
    # validate that a valid measurement name is given
    if selected_measurement_name != '' and selected_measurement_name != 'new measurement':
        check_measurement_name = True
    else:
        container_bottom.error('Measurement name is not valid.')

    # check if the channel ids are allowed
    check_csv_channel_ids = False
    check_value_channel = 0
    for key in input_data.keys():
        if key > 23 or key < 0:
            check_value_channel = 1
    if check_value_channel == 0:
        check_csv_channel_ids = True
    else:
        container_bottom.error('The channel numbers in the input file are out of bounds. They must be between 0 and 23.')

    # check that rotation and orientation are realistic values (0,360°)
    check_csv_rot_ori = False
    check_value_rot = 0
    check_value_ori = 0
    for key in input_data.keys():
        if not 0 <= input_data[key]['rotation']['theta'] <= 360 or not 0 <= input_data[key]['rotation']['phi'] <= 360:
            check_value_rot = 1
        if not 0 <= input_data[key]['orientation']['theta'] <= 360 or not 0 <= input_data[key]['orientation']['phi'] <= 360:
            check_value_ori = 1

    if check_value_rot == 1:
        container_bottom.error('The rotation values are invalid. Please give values between 0° and 360°.')
    elif check_value_ori == 1:
        container_bottom.error('The orientation values are invalid. Please give values between 0° and 360°.')
    else:
        check_csv_rot_ori = True

    # check if the all inserted channels are not already in the database
    check_channel_in_db = False
    check_val = 0
    for key in input_data.keys():
        if key in channel_id_in_db:
            check_val = 1
            break
    if check_val == 0:
        check_channel_in_db = True
    else:
        container_bottom.error('Some of the channels are already in the database. The file cannot be uploaded.')

    if check_csv_channel_ids and check_channel_in_db and check_measurement_name and check_csv_rot_ori and check_file:
        disable_input_button = False

    return disable_input_button


def build_main_page(input_cont, selected_input_option):
    cont = input_cont.container()
    if selected_input_option == 'single channel input':
        cont.empty()
        channel = cont.selectbox('Select a channel:', channel_help)
        if channel is not None:
            selected_channel = int(channel)
        else:
            selected_channel = -10

        # if the channel already exist in the database, the channel info will be loaded
        current_channel_info_db = {}
        if selected_channel != -10:
            for entry in measurement_info:
                if int(entry['measurements']['channel_id']) == selected_channel:
                    current_channel_info_db = entry['measurements']

        # primary measurement?
        primary = cont.checkbox('Is this the primary measurement?', value=True)

        # if the channel already exist enter the existing channel information, otherwise fall back to the default
        default_position = [0, 0, 0]
        default_rot = {'theta': 0, 'phi': 0}
        default_ori = {'theta': 0, 'phi': 0}
        default_measurement_time = datetime.utcnow()
        # if current_channel_info_db != {}:
        #     default_measurement_time = datetime.date(current_channel_info_db['measurement_time'])
        #     default_position = current_channel_info_db['position']
        #     default_rot = current_channel_info_db['orientation']
        #     default_ori = current_channel_info_db['rotation']

        # measurement time
        measurement_time = cont.date_input('Enter the time of the measurement:', value=default_measurement_time, help='The date when the measurement was conducted.')
        measurement_time = datetime.combine(measurement_time, time(0, 0, 0))

        # insert the position of the antenna;
        col1_pos, col2_pos, col3_pos = cont.columns([1, 1, 1])
        x_pos_ant = col1_pos.text_input('x position', key='x_antenna', value=default_position[0])
        y_pos_ant = col2_pos.text_input('y position', key='y_antenna', value=default_position[1])
        z_pos_ant = col3_pos.text_input('z position', key='z_antenna', value=default_position[2])
        position = [float(x_pos_ant), float(y_pos_ant), float(z_pos_ant)]

        # input the orientation, rotation of the antenna; if the channel already exist, insert the values from the database
        col1a, col2a, col3a, col4a = cont.columns([1, 1, 1, 1])

        ant_ori_theta = col1a.text_input('orientation (theta):', value=default_ori['theta'])
        ant_ori_phi = col2a.text_input('orientation (phi):', value=default_ori['phi'])
        ant_rot_theta = col3a.text_input('rotation (theta):', value=default_rot['theta'])
        ant_rot_phi = col4a.text_input('rotation (phi):', value=default_rot['phi'])
        ori = {'theta': float(ant_ori_theta), 'phi': float(ant_ori_phi)}
        rot = {'theta': float(ant_rot_theta), 'phi': float(ant_rot_phi)}

        # container for warnings/infos at the botton
        cont_warning_bottom = cont.container()

        disable_insert_button = validate_inputs(cont_warning_bottom, selected_station_id, measurement_name, selected_channel, measurement_channel_ids_db, ori, rot)

        insert_channel = cont.button('INSERT CHANNEL TO DB', disabled=disable_insert_button, key='1')

        cont_channel_warning = cont.container()

        if insert_channel:
            insert_channel_position_to_db(selected_station_id, selected_channel, measurement_name, measurement_time, position, ori, rot, primary)
            cont.empty()
            st.session_state.key = '1'
            st.experimental_rerun()

    elif selected_input_option == 'file input':
        cont.empty()
        csv_file = cont.file_uploader('Please upload a position CSV file.', type='csv')

        position_data = read_position_data(cont, csv_file)

        # primary measurement?
        primary = cont.checkbox('Is this the primary measurement?', value=True)

        # measurement time
        default_measurement_time = datetime.utcnow()
        measurement_time = cont.date_input('Enter the time of the measurement:', value=default_measurement_time, help='The date when the measurement was conducted.')
        measurement_time = datetime.combine(measurement_time, time(0, 0, 0))

        cont_warning_bottom_csv = cont.container()

        disable_insert_csv_button = validate_csv_inputs(cont_warning_bottom_csv, measurement_name, position_data, measurement_channel_ids_db)

        # necessary to replace the button by the spinner, when the data is uploaded
        button_cont = cont.empty()

        insert_channel_csv = button_cont.button('INSERT CHANNEL TO DB', disabled=disable_insert_csv_button, key='2')

        if insert_channel_csv:
            button_cont.empty()
            with st.spinner('Data is being uploaded ...'):
                for key in position_data.keys():
                    insert_channel_position_to_db(selected_station_id, key, measurement_name, measurement_time, position_data[key]['position'], position_data[key]['orientation'], position_data[key]['rotation'], primary)
            cont.empty()
            st.session_state.key = '1'
            st.experimental_rerun()


# main page setup
page_configuration()

# initialize the session key (will be used to display different pages depending on the button clicked)
if 'key' not in st.session_state:
    st.session_state['key'] = '0'

main_container = st.container()  # container for the main part of the page (with all the input filed)
success_container = st.container()  # container to display the page when the data was submitted

main_container.title('Add channel position and orientation information')
main_container.markdown(page_name)

# select a unique name for the measurement (survey_01, tape_measurement, ...)
col1_name, col2_name = main_container.columns([1, 1])
measurement_list = load_measurement_names('channel_position')
measurement_list.insert(0, 'new measurement')
selected_name = col1_name.selectbox('Select or enter a unique name for the measurement:', measurement_list)
disabled_name_input = True
if selected_name == 'new measurement':
    disabled_name_input = False
name_input = col2_name.text_input('Select or enter a unique name for the measurement:', disabled=disabled_name_input, label_visibility='hidden')
measurement_name = selected_name
if measurement_name == 'new measurement':
    measurement_name = name_input

# enter the information for the single stations
cont_warning_top = main_container.container()
station_list = ['Station 11 (Nanoq)', 'Station 12 (Terianniaq)', 'Station 13 (Ukaleq)', 'Station 14 (Tuttu)', 'Station 15 (Umimmak)', 'Station 21 (Amaroq)', 'Station 22 (Avinngaq)',
                'Station 23 (Ukaliatsiaq)', 'Station 24 (Qappik)', 'Station 25 (Aataaq)']
selected_station = main_container.selectbox('Select a station', station_list)
# get the name and id out of the string
selected_station_name = selected_station[selected_station.find('(') + 1:-1]
selected_station_id = int(selected_station[len('Station '):len('Station ') + 2])

# page to enter the station information
cont_warning_top = main_container.container()

main_container.subheader('Input channel information')
# tab1, tab2 = main_container.tabs(['single channel input', 'file input'])

# load the selected measurement information  (to display how many channels are inserted for this specific measurement)
measurement_info = load_measurement_station_information(selected_station_id, measurement_name)

# extract the channel ids from the measurement info (also used from checking if a channel already exists)
measurement_channel_ids_db = []
for entry in measurement_info:
    measurement_channel_ids_db.append(entry['measurements']['channel_id'])

main_container.info(f'Channels included in the database: {len(measurement_channel_ids_db)}/24')

channel_help = []
for cha in range(24):
    if cha not in measurement_channel_ids_db:
        channel_help.append(cha)

input_option = main_container.radio('Input form of channel position:', ['single channel input', 'file input'], horizontal=True, index=0)

if st.session_state.key == '0':
    build_main_page(main_container, input_option)  # after clicking the submit button, the session key is set to '1' and the page is rerun

if st.session_state.key == '1':
    build_success_page(success_container, 'channel position')  # after clicking the 'add another measurement' button, the session key is set to '0' and the page is rerun
